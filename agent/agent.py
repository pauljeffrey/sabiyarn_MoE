"""The agent loop: render -> generate -> run the tool -> continue the SAME turn.

    from agent import Agent, HFBackend, tool

    @tool
    def get_weather(city: str) -> dict:
        "Current weather for a city."
        return {"city": city, "temp": 31}

    bot = Agent(HFBackend(model, tok), tools=[get_weather])
    print(bot.chat("What's the weather in Lagos?"))

Implements the serving contract in docs/TOOL_CALL_FORMAT.md section 4:

    1. prompt = render(messages, tools=..., add_generation_prompt=True); after a
       tool result the prompt ends in </tool_response> and the model continues
       its turn with no new <|assistant|>;
    2. generate with stop [</s>, </tool_call>]; ending at </tool_call> means
       parse the JSON call, run it, append the assistant message (scaffold +
       the call) and the tool result, re-render and continue; ending at </s>
       means the turn is done;
    3. one call per step -- </tool_call> is a stop, so a rollout halts after the
       first call and nothing is trained to follow the last one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from . import protocol as P
from .events import (
    ConfirmNeeded, Delta, Event, Failed, Final, Plan, RoundStarted, Think, ToolCalled, ToolReturned,
)
from .llm import LLM
from .memory import Memory, SummarizingMemory, WindowMemory, approx_tokens
from .structured import StructuredOutputError, extract_json, schema_instruction, validate_payload
from .tools import Tool, Toolbox

__all__ = ["Agent"]

ConfirmFn = Callable[[P.ParsedCall], bool]


class _Visible:
    """Decide what of the raw stream the user should see.

    Everything is suppressed until <response> appears; after that, text streams
    through until </response> (or EOS). A tail that could be the start of a tag
    is held back, so a partial "</resp" never reaches a UI. A step that produces
    only a tool call shows nothing, which is correct: the answer comes after the
    tool result, in the continuation of the same turn.
    """

    def __init__(self, hold: int = len(P.RESPONSE_CLOSE)) -> None:
        self.hold = hold
        self.raw = ""
        self.open = False
        self.emitted = 0

    def push(self, chunk: str) -> str:
        self.raw += chunk
        if not self.open:
            i = self.raw.find(P.RESPONSE_OPEN)
            if i == -1:
                return ""
            self.open = True
            self.emitted = i + len(P.RESPONSE_OPEN)
        body = self._body()
        if len(body) > self.hold:
            out = body[: -self.hold]
            self.emitted += len(out)
            return out
        return ""

    def _body(self) -> str:
        body = self.raw[self.emitted :]
        for tag in (P.RESPONSE_CLOSE, P.EOS, P.TOOL_CALL_OPEN, P.USER):
            cut = body.find(tag)
            if cut != -1:
                body = body[:cut]
        return body

    def flush(self) -> str:
        if not self.open:
            return ""
        body = self._body()
        self.emitted += len(body)
        return body


@dataclass
class Agent:
    """Conversation plus tools plus memory.

    llm       anything implementing .stream()/.complete() (see llm.LLM)
    tools     Toolbox, or any iterable of @tool functions; rendered into the
              system turn as a <tools> block by the template, via `tools=`
    memory    defaults to SummarizingMemory sized for a 4096-token context
    identity  the system message text; the <tools> block is appended to it
    confirm   called for tools marked requires_confirmation; return True to run.
              Without it, run() yields ConfirmNeeded and declines by default.
    """

    llm: LLM
    tools: Toolbox | Iterable[Any] = field(default_factory=Toolbox)
    memory: Memory | None = None
    identity: str = P.DEFAULT_IDENTITY
    system_extra: str = ""
    max_rounds: int = 6
    confirm: ConfirmFn | None = None
    max_new_tokens: int = 512
    stop: Sequence[str] = P.STOP_STRINGS
    repair_bad_arguments: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.tools, Toolbox):
            self.tools = Toolbox(self.tools)
        if self.memory is None:
            self.memory = SummarizingMemory(
                max_tokens=3072,
                count_tokens=getattr(self.llm, "count_tokens", approx_tokens),
                summarize=self._summarize,
            )
        self._install_system()

    # -- setup --------------------------------------------------------------
    def _install_system(self, extra: str = "") -> None:
        """Set the system turn. Tool schemas are NOT written in here: they are
        passed to the renderer as `tools=` and become the <tools> block, which
        is what the template and the training data expect."""
        parts = [self.identity, self.system_extra, extra]
        content = "\n".join(p for p in parts if p)
        self.memory.add(P.system_message(content))  # type: ignore[union-attr]

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas()  # type: ignore[union-attr]

    def _render(self, messages: Sequence[dict[str, Any]]) -> str:
        return P.render(messages, tools=self._tool_schemas(), add_generation_prompt=True)

    def _summarize(self, prompt: str) -> str:
        """Used by SummarizingMemory. Greedy, and with no tools attached: a
        digest should be reproducible, and the catalogue is irrelevant to it."""
        return self.llm.complete(
            P.render([P.user_message(prompt)], add_generation_prompt=True),
            stop=[P.EOS],
            max_new_tokens=256,
            do_sample=False,
        ).replace(P.EOS, "").strip()

    def add_tool(self, t: Tool | Callable[..., Any]) -> Tool:
        return self.tools.add(t)  # type: ignore[union-attr]

    def reset(self) -> None:
        memory = self.memory
        if isinstance(memory, WindowMemory):
            memory.clear()
        self._install_system()

    # -- public API ---------------------------------------------------------
    def chat(self, message: str, **kw: Any) -> str:
        text = ""
        for ev in self.run(message, **kw):
            if isinstance(ev, (Final, Failed)):
                text = ev.text
        return text

    def structured(self, message: str, schema: Any, *, retries: int = 1, **kw: Any) -> Any:
        """Ask for schema-conforming JSON and return the validated object."""
        instruction = schema_instruction(schema)
        attempt, last_error = 0, None
        while attempt <= retries:
            prompt = message if attempt == 0 else (
                f"{message}\n\nYour previous reply was rejected: {last_error}\nReturn only valid JSON."
            )
            text = self.chat(prompt, system_extra=instruction, **kw)
            try:
                return validate_payload(extract_json(text), schema)
            except StructuredOutputError as exc:
                last_error, attempt = str(exc), attempt + 1
        raise StructuredOutputError(f"schema not satisfied after {retries + 1} attempt(s): {last_error}")

    def run(
        self,
        message: str | None = None,
        *,
        stream: bool = True,
        system_extra: str = "",
        max_new_tokens: int | None = None,
        **gen_kw: Any,
    ) -> Iterator[Event]:
        """Drive one user turn, yielding events until Final or Failed.

        Always a generator. With stream=False the answer arrives in one Delta;
        the event sequence is otherwise identical, so callers do not branch.
        """
        if system_extra:
            self._install_system(system_extra)
        if message is not None:
            self.memory.add(P.user_message(message))  # type: ignore[union-attr]

        # Text the model has produced across this turn, including the parts
        # written before each tool call -- a turn can span several generate
        # calls, and the answer is parsed from the whole thing.
        turn_text = ""

        for round_index in range(self.max_rounds):
            messages = self.memory.messages()  # type: ignore[union-attr]
            prompt = self._render(messages)
            yield RoundStarted(round=round_index, prompt_tokens=self._count(prompt))

            try:
                raw, deltas = self._generate(prompt, stream, max_new_tokens, gen_kw)
            except Exception as exc:
                yield Failed(reason=f"{type(exc).__name__}: {exc}", text=P.parse_assistant(turn_text).text)
                return
            for chunk in deltas:
                yield Delta(chunk)

            turn_text += raw
            step = P.parse_assistant(raw)
            if step.think:
                yield Think(step.think)
            if step.task_plan:
                yield Plan(step.task_plan, step.invalid_verbs)

            if not step.wants_tool:
                if step.truncated_tool_call:
                    yield Failed(
                        reason="tool call was cut off by the token budget; raise max_new_tokens",
                        text=step.text,
                        turn=step,
                    )
                    return
                # End of the turn: one assistant message, content only, which is
                # what makes the template close it with a single </s>.
                self.memory.add({"role": "assistant", "content": raw})  # type: ignore[union-attr]
                whole = P.parse_assistant(turn_text)
                yield Final(text=whole.text, turn=whole, rounds=round_index + 1)
                return

            # A tool was requested. Record scaffold + structured call (NOT the
            # raw text: an assistant message without tool_calls would close the
            # turn with </s>, which the model never saw mid-exchange).
            self.memory.add(P.record_tool_call_turn(raw, step.tool_calls))  # type: ignore[union-attr]

            # One call per step: </tool_call> is a stop, so there is normally
            # exactly one. Extra calls are executed in order if a backend
            # without stop support produced several.
            for call in step.tool_calls:
                yield ToolCalled(call.name, call.arguments or {}, call.raw_arguments)
                result = yield from self._execute(call)
                yield ToolReturned(result.name, result.value, result.error, result.declined)
                self.memory.add(P.tool_message(result.payload(), call.name))  # type: ignore[union-attr]

        whole = P.parse_assistant(turn_text)
        yield Failed(
            reason=f"stopped after {self.max_rounds} rounds without a final answer",
            text=whole.text,
            turn=whole,
        )

    def resume(self, **kw: Any) -> Iterator[Event]:
        """Continue with no new user message -- after tool results were added
        out of band, or after a ConfirmNeeded was answered."""
        return self.run(None, **kw)

    # -- internals ----------------------------------------------------------
    def _count(self, text: str) -> int | None:
        counter = getattr(self.llm, "count_tokens", None)
        try:
            return counter(text) if counter else None
        except Exception:
            return None

    def _generate(
        self, prompt: str, stream: bool, max_new_tokens: int | None, gen_kw: dict[str, Any]
    ) -> tuple[str, list[str]]:
        """Produce this step's raw text plus the user-visible chunks to emit."""
        budget = max_new_tokens or self.max_new_tokens
        if not stream:
            raw = self.llm.complete(prompt, stop=self.stop, max_new_tokens=budget, **gen_kw)
            visible = P.parse_assistant(raw).text if P.RESPONSE_OPEN in raw else ""
            return raw, [visible] if visible else []

        view = _Visible()
        chunks: list[str] = []
        for piece in self.llm.stream(prompt, stop=self.stop, max_new_tokens=budget, **gen_kw):
            out = view.push(piece)
            if out:
                chunks.append(out)
        tail = view.flush()
        if tail:
            chunks.append(tail)
        return view.raw, chunks

    def _execute(self, call: P.ParsedCall):
        """Validate, confirm if required, then run. Yields ConfirmNeeded when no
        confirm callback is configured, so a UI can answer in-band."""
        from .tools import ToolResult

        if call.arguments is None or not call.name:
            detail = call.error or "the tool call did not parse"
            if self.repair_bad_arguments:
                detail += '. Re-issue it as {"name": "...", "arguments": {...}}.'
            return ToolResult(name=call.name or "unknown", error=detail)

        if call.name not in self.tools:  # type: ignore[operator]
            known = ", ".join(self.tools.names) or "none"  # type: ignore[union-attr]
            return ToolResult(name=call.name, error=f"unknown tool {call.name!r}. Available: {known}")

        spec = self.tools[call.name]  # type: ignore[index]
        if spec.requires_confirmation:
            if self.confirm is not None:
                try:
                    approved = bool(self.confirm(call))
                except Exception as exc:
                    return ToolResult(name=call.name, error=f"confirmation failed: {exc}")
            else:
                ask = ConfirmNeeded(call=call)
                yield ask
                approved = bool(ask.approved)
            if not approved:
                return ToolResult(name=call.name, declined=True, error="the user declined this call")

        return self.tools.call(call.name, call.arguments)  # type: ignore[union-attr]
