"""The agent loop: render -> generate -> run tools -> feed results back -> answer.

    from agent import Agent, Toolbox, tool

    @tool
    def get_weather(city: str) -> dict:
        "Current weather for a city."
        return {"city": city, "temp": 31}

    bot = Agent(llm, tools=[get_weather])
    print(bot.chat("What's the weather in Lagos?"))        # non-streaming
    for ev in bot.run("And in Kano?"):                     # streaming + events
        ...
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

    Everything is suppressed until `<response>` appears; after that, text
    streams through. A tail that could be the start of a tag is held back, so a
    partial "<resp" never reaches a UI. If the turn never emits <response> (a
    pure tool call, or a malformed turn), nothing is shown -- the agent surfaces
    the parsed result instead.
    """

    def __init__(self, hold: int = len(P.RESPONSE_OPEN)) -> None:
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
        body = self.raw[self.emitted :]
        for tag in (P.EOS, P.TOOL_CALL_OPEN, P.USER):
            cut = body.find(tag)
            if cut != -1:
                body = body[:cut]
        if len(body) > self.hold:
            out = body[: -self.hold]
            self.emitted += len(out)
            return out
        return ""

    def flush(self) -> str:
        if not self.open:
            return ""
        body = self.raw[self.emitted :]
        for tag in (P.EOS, P.TOOL_CALL_OPEN, P.USER):
            cut = body.find(tag)
            if cut != -1:
                body = body[:cut]
        self.emitted += len(body)
        return body


@dataclass
class Agent:
    """Conversation plus tools plus memory.

    llm       anything implementing .stream()/.complete() (see llm.LLM)
    tools     Toolbox, or any iterable of @tool functions
    memory    defaults to SummarizingMemory sized for a 4096-token context
    identity  first sentence of the system prompt; the tool catalogue follows it
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
    # Retry once when a tool call's arguments will not parse: the model is told
    # what was wrong, which recovers far more often than failing the turn.
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
    def _system_text(self) -> str:
        text = P.format_tools(self.tools.schemas(), self.identity)
        return f"{text}\n{self.system_extra}" if self.system_extra else text

    def _install_system(self, extra: str = "") -> None:
        base = self._system_text()
        content = f"{base}\n{extra}" if extra else base
        self.memory.add(P.system_message(content))  # type: ignore[union-attr]

    def _summarize(self, prompt: str) -> str:
        """Used by SummarizingMemory. Greedy and short: a digest should be
        reproducible, and sampling here would make the same history summarise
        differently on every turn."""
        return self.llm.complete(
            P.render([P.user_message(prompt)], add_generation_prompt=True),
            stop=[P.EOS],
            max_new_tokens=256,
            do_sample=False,
        ).replace(P.EOS, "").strip()

    def add_tool(self, t: Tool | Callable[..., Any]) -> Tool:
        """Register a tool and refresh the system prompt so the model sees it."""
        added = self.tools.add(t)  # type: ignore[union-attr]
        self._install_system()
        return added

    def reset(self) -> None:
        memory = self.memory
        if isinstance(memory, WindowMemory):
            memory.clear()
        self._install_system()

    # -- public API ---------------------------------------------------------
    def chat(self, message: str, **kw: Any) -> str:
        """Run to completion and return the answer text."""
        text = ""
        for ev in self.run(message, **kw):
            if isinstance(ev, (Final, Failed)):
                text = ev.text
        return text

    def structured(self, message: str, schema: Any, *, retries: int = 1, **kw: Any) -> Any:
        """Ask for schema-conforming JSON and return the validated object.

        On a validation failure the schema error is fed back once (by default),
        which is cheaper and more reliable than asking a small model to get
        nested JSON right first time.
        """
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

        Always a generator. With stream=False the model's text arrives in one
        Delta instead of many; the event sequence is otherwise identical, so
        callers do not branch on mode.
        """
        if system_extra:
            self._install_system(system_extra)
        if message is not None:
            self.memory.add(P.user_message(message))  # type: ignore[union-attr]

        for round_index in range(self.max_rounds):
            messages = self.memory.messages()  # type: ignore[union-attr]
            prompt = P.render(messages, add_generation_prompt=True)
            yield RoundStarted(round=round_index, prompt_tokens=self._count(prompt))

            try:
                raw, deltas = self._generate(prompt, stream, max_new_tokens, gen_kw)
            except Exception as exc:
                yield Failed(reason=f"{type(exc).__name__}: {exc}")
                return
            for chunk in deltas:
                yield Delta(chunk)

            turn = P.parse_assistant(raw)
            if turn.think:
                yield Think(turn.think)
            if turn.task_plan:
                yield Plan(turn.task_plan, turn.invalid_verbs)

            if not turn.wants_tool:
                if turn.truncated_tool_call:
                    # A call cut off mid-JSON: nothing to execute and nothing to
                    # show. Say so rather than presenting half a tag as an answer.
                    yield Failed(
                        reason="tool call was cut off by the token budget; raise max_new_tokens",
                        text=turn.text,
                        turn=turn,
                    )
                    return
                self.memory.add({"role": "assistant", "content": raw})  # type: ignore[union-attr]
                yield Final(text=turn.text, turn=turn, rounds=round_index + 1)
                return

            # The model asked for tools. Record the turn verbatim (content-only)
            # so the next prompt keeps the <think>/<task_plan> prefix.
            self.memory.add(P.record_tool_call_turn(raw))  # type: ignore[union-attr]

            for call in turn.tool_calls:
                yield ToolCalled(call.name, call.arguments or {}, call.raw_arguments)
                result = yield from self._execute(call)
                yield ToolReturned(result.name, result.value, result.error, result.declined)
                self.memory.add(P.tool_message(call.name, result.payload()))  # type: ignore[union-attr]

        yield Failed(reason=f"stopped after {self.max_rounds} rounds without a final answer")

    def resume(self, **kw: Any) -> Iterator[Event]:
        """Continue after tool results were added out of band (or after a
        ConfirmNeeded was answered): same loop, no new user message."""
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
        """Produce the raw turn text plus the user-visible chunks to emit."""
        budget = max_new_tokens or self.max_new_tokens
        if not stream:
            raw = self.llm.complete(prompt, stop=self.stop, max_new_tokens=budget, **gen_kw)
            visible = P.parse_assistant(raw).text
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

        if call.arguments is None:
            detail = call.error or "arguments did not parse"
            if self.repair_bad_arguments:
                detail += ". Re-issue the call with valid JSON arguments."
            return ToolResult(name=call.name, error=detail)

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
