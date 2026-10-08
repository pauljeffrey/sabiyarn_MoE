"""The SabiYarn wire format (chat template v2): tags, rendering and parsing.

Pure stdlib so the package ports anywhere. The renderer is a hand-written
equivalent of sabiyarn/chat_template.jinja (jinja2 is not a root dependency, so
tokenizer.apply_chat_template is unavailable there) and is pinned against it
byte-for-byte by tests/test_agent_render_parity.py. See docs/TOOL_CALL_FORMAT.md
for the contract this implements; if this file and the template disagree, the
template wins.

One assistant TURN runs from <|assistant|> to </s>, with tool use INSIDE it:

    <|assistant|>SCAFFOLD<tool_call>{"name": "f", "arguments": {...}}</tool_call>
    <tool_response>RESULT</tool_response>SCAFFOLD<response>ANSWER</response></s>

Four consequences the whole framework is built around:

1. </tool_call> is where the model stops. The serving loop stops there, runs the
   tool, appends <tool_response>RESULT</tool_response>, and lets the model
   CONTINUE the same turn -- no new <|assistant|>, no </s> in between.
2. One call per step. </tool_call> being a stop means a rollout halts after the
   first call, and nothing is trained to follow the last one.
3. The tool result carries no function name; results answer the calls above
   them, in order. It is also masked from the loss: the model never predicts it.
4. Function definitions are passed as `tools=` and rendered once in the system
   turn as <tools>\\n{json}\\n{json}\\n</tools>, not as free text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# --- tokens -----------------------------------------------------------------
# Single tokens in BeardedMonster/SabiYarn-32k, except <tools>/</tools> (52117/
# 52118) which are pending -- see scripts/add_tool_tokens.py. Until those are
# pushed they tokenise as several byte-BPE pieces, which costs tokens but
# renders and parses identically.
BOS = "<s>"
EOS = "</s>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<tool_response>", "</tool_response>"
TOOLS_OPEN, TOOLS_CLOSE = "<tools>", "</tools>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TASK_PLAN_OPEN, TASK_PLAN_CLOSE = "<task_plan>", "</task_plan>"
RESPONSE_OPEN, RESPONSE_CLOSE = "<response>", "</response>"
CONTEXT_OPEN, CONTEXT_CLOSE = "<context>", "</context>"
INPUT_LANG, TARGET_LANG = "<|input_lang|>", "<|target_lang|>"

VALID_VERBS = frozenset(
    {
        "<|chat|>", "<|generate|>", "<|edit|>", "<|data_extract|>", "<|math|>",
        "<|JSON|>", "<|code|>", "<|plan|>", "<|analyze|>", "<|recommend|>",
        "<|explain|>", "<|debug|>", "<|RAG|>", "<summarize>", "<translate>",
        "<classify>", "<NER>", "<identify>", "<qa>", "<diacritize>", "<clean>",
    }
)

# Stop at the close of a call so the tool can run, and at EOS for the turn.
# </s> is handled by the backend through eos_token_id as well; the role markers
# are defensive -- a model that tries to open the next turn itself should not be
# allowed to write it.
STOP_STRINGS: tuple[str, ...] = (TOOL_CALL_CLOSE, USER, SYSTEM)

# --- parsing ----------------------------------------------------------------
R_THINK = re.compile(r"<think>(.*?)</think>", re.S)
R_TASK_PLAN = re.compile(r"<task_plan>(.*?)</task_plan>", re.S)
R_RESPONSE = re.compile(r"<response>(.*?)(?:</response>|$)", re.S)
R_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# The pre-v2 shape, kept because checkpoints trained on it still emit it (and
# rl/verifiers.py reads both).
R_TOOL_CALL_LEGACY = re.compile(r"<tool_call>([^\n{<][^\n<]*)\n(.*?)</tool_call>", re.S)
R_TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
R_INPUT_LANG = re.compile(r"<\|input_lang\|><([a-z]{2,4})>")
R_TARGET_LANG = re.compile(r"<\|target_lang\|><([a-z]{2,4})>")
R_VERB = re.compile(r"<\|?[a-zA-Z_]+\|?>")


def _tojson(value: Any) -> str:
    """The template's `tojson` filter, matching transformers exactly:
    json.dumps(value, ensure_ascii=False) -- default separators, key order
    preserved. Both matter: sorted keys or \\uXXXX escapes would make training
    text and inference text differ on precisely the Yoruba queries."""
    return json.dumps(value, ensure_ascii=False)


def tool_response_content(payload: Any) -> str:
    """Serialize a tool's return value for a role="tool" message. Strings pass
    through untouched; the corpus keeps tool payloads verbatim."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_tool_call(name: str, arguments: Any) -> str:
    """One <tool_call> in the v2 (Hermes/Qwen) shape.

    Mirrors the template's string building exactly, including the spaces after
    the colons: `{"name": "f", "arguments": {"a": 1}}`. A string `arguments` is
    embedded as-is (it is already JSON), which is what the template does.
    """
    args = arguments if isinstance(arguments, str) else _tojson(arguments if arguments is not None else {})
    return f'{TOOL_CALL_OPEN}{{"name": {_tojson(name)}, "arguments": {args}}}{TOOL_CALL_CLOSE}'


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any] | None  # None when the JSON would not parse
    raw_arguments: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ParsedTurn:
    """Everything of interest in generated text.

    Note `response` is read from after the LAST </tool_response>, because the
    model writes a fresh scaffold there and the answer belongs to that second
    scaffold, not to anything before the tool result.
    """

    raw: str
    think: str | None = None
    task_plan: tuple[str, ...] = ()
    tool_calls: tuple[ParsedCall, ...] = ()
    response: str | None = None
    input_lang: str | None = None
    target_lang: str | None = None
    truncated_tool_call: bool = False
    invalid_verbs: tuple[str, ...] = ()

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

    @property
    def text(self) -> str:
        """What a user should see: the <response> body (closing tag optional --
        pre-v2 data has none), else the raw text with scaffolding stripped.

        When </response> is missing the body runs to the end of the match, so it
        is trimmed at EOS and at any role marker the model wrote past it;
        otherwise `<response>answer</s>` would surface a trailing </s>.
        """
        if self.response is not None:
            body = self.response
            for marker in (RESPONSE_CLOSE, EOS, USER, SYSTEM, ASSISTANT, TOOL_CALL_OPEN):
                cut = body.find(marker)
                if cut != -1:
                    body = body[:cut]
            return body.strip()
        return strip_tags(self.raw).strip()

    @property
    def scaffold(self) -> str:
        """The text before the first tool call: <|input_lang|>, <think>,
        <task_plan>. This is what goes in the assistant message's `content`
        when the call is recorded, so the template can re-render the turn."""
        cut = self.raw.find(TOOL_CALL_OPEN)
        return self.raw if cut == -1 else self.raw[:cut]


def _parse_one_call(blob: str) -> ParsedCall:
    """Parse a v2 call body: {"name": ..., "arguments": {...}}."""
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as exc:
        return ParsedCall("", None, blob, f"invalid JSON in tool call: {exc.msg}")
    if not isinstance(obj, dict):
        return ParsedCall("", None, blob, "a tool call must be a JSON object")
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return ParsedCall("", None, blob, 'tool call is missing a string "name"')
    args = obj.get("arguments", {})
    if isinstance(args, str):  # arguments may arrive as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            return ParsedCall(name, None, blob, f'invalid JSON in "arguments": {exc.msg}')
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ParsedCall(name, None, blob, '"arguments" must be a JSON object')
    return ParsedCall(name, args, _tojson(args))


def parse_assistant(text: str) -> ParsedTurn:
    """Parse generated text. Never raises: a malformed turn is data, and the
    agent decides whether to retry, so problems come back as fields."""
    think_m = R_THINK.search(text)
    in_m, tgt_m = R_INPUT_LANG.search(text), R_TARGET_LANG.search(text)

    # The answer belongs to the scaffold written after the tool result.
    tail = text
    last_result = text.rfind(TOOL_RESPONSE_CLOSE)
    if last_result != -1:
        tail = text[last_result + len(TOOL_RESPONSE_CLOSE) :]
    resp_m = R_RESPONSE.search(tail) or R_RESPONSE.search(text)

    plan_m = R_TASK_PLAN.search(tail) or R_TASK_PLAN.search(text)
    verbs: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()
    if plan_m:
        verbs = tuple(R_VERB.findall(plan_m.group(1)))
        invalid = tuple(v for v in verbs if v not in VALID_VERBS)

    calls = [_parse_one_call(blob) for blob in R_TOOL_CALL.findall(text)]
    if not calls:  # pre-v2 `name\n{args}` form
        for name, raw_args in R_TOOL_CALL_LEGACY.findall(text):
            try:
                parsed = json.loads(raw_args)
                calls.append(
                    ParsedCall(name.strip(), parsed, raw_args)
                    if isinstance(parsed, dict)
                    else ParsedCall(name.strip(), None, raw_args, "arguments must be a JSON object")
                )
            except json.JSONDecodeError as exc:
                calls.append(ParsedCall(name.strip(), None, raw_args, f"invalid JSON arguments: {exc.msg}"))

    truncated = not calls and TOOL_CALL_OPEN in text and TOOL_CALL_CLOSE not in text

    return ParsedTurn(
        raw=text,
        think=think_m.group(1).strip() if think_m else None,
        task_plan=verbs,
        tool_calls=tuple(calls),
        response=resp_m.group(1) if resp_m else None,
        input_lang=in_m.group(1) if in_m else None,
        target_lang=tgt_m.group(1) if tgt_m else None,
        truncated_tool_call=truncated,
        invalid_verbs=invalid,
    )


def strip_tags(text: str) -> str:
    """Remove scaffolding so text is presentable."""
    out = R_THINK.sub("", text)
    out = R_TASK_PLAN.sub("", out)
    out = R_TOOL_CALL.sub("", out)
    out = R_TOOL_CALL_LEGACY.sub("", out)
    out = R_TOOL_RESPONSE.sub("", out)
    for tag in (RESPONSE_OPEN, RESPONSE_CLOSE, EOS, BOS, ASSISTANT, USER, SYSTEM,
                CONTEXT_OPEN, CONTEXT_CLOSE):
        out = out.replace(tag, "")
    out = re.sub(r"<\|[a-zA-Z_]+\|><[a-z]{2,4}>", "", out)  # <|input_lang|><yor>
    return out


# --- rendering --------------------------------------------------------------
def format_tools(schemas: Iterable[dict[str, Any]]) -> str:
    """The <tools> block: one compact-ish JSON object per line.

    `json.dumps(tool, ensure_ascii=False)` per line, matching the template's
    `tool | tojson` -- note that means default separators (a space after each
    colon and comma), not json.dumps' compact mode.
    """
    catalogue = list(schemas)
    if not catalogue:
        return ""
    body = "\n".join(_tojson(t) for t in catalogue)
    return f"{TOOLS_OPEN}\n{body}\n{TOOLS_CLOSE}"


def render(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Iterable[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    bos: str = BOS,
    eos: str = EOS,
) -> str:
    """Render messages (and the tool catalogue) to the exact training text.

    Equivalent to sabiyarn/chat_template.jinja v2. The subtle parts, all pinned
    by the parity test:

    * tools are appended to the system turn, separated by a blank line when the
      system message has content; with no system message at all they get a
      system turn of their own, emitted before the first message;
    * an assistant message that FOLLOWS a tool result gets no <|assistant|>
      prefix -- it continues the same turn;
    * an assistant message carrying tool_calls gets NO eos (the turn is not
      over); one without them is the end of the turn and gets eos;
    * a tool result gets neither a name nor an eos;
    * add_generation_prompt appends <|assistant|> only when the last message is
      not a tool result, for the same continuation reason.

    One deliberate divergence: the template silently skips an unrecognised role,
    turning a typo into a prompt with a missing turn. This raises instead.
    """
    tools_text = format_tools(tools or ())
    out: list[str] = []
    for i, message in enumerate(messages):
        role = message.get("role")
        if i == 0:
            out.append(bos)
            if tools_text and role != "system":
                out += [SYSTEM, tools_text, eos]
        if role == "system":
            content = message.get("content", "")
            out += [SYSTEM, content]
            if tools_text:
                out.append(("\n\n" if content else "") + tools_text)
            out.append(eos)
        elif role == "user":
            out += [USER, message.get("content", ""), eos]
        elif role == "tool":
            out += [TOOL_RESPONSE_OPEN, message.get("content", ""), TOOL_RESPONSE_CLOSE]
        elif role == "assistant":
            previous = messages[i - 1] if i > 0 else None
            if previous is None or previous.get("role") != "tool":
                out.append(ASSISTANT)
            if message.get("content"):
                out.append(message["content"])
            calls = message.get("tool_calls") or ()
            if calls:
                for call in calls:
                    fn = call.get("function", call)
                    out.append(render_tool_call(fn["name"], fn.get("arguments")))
            else:
                out.append(eos)
        else:
            raise ValueError(f"unknown message role {role!r} (expected system/user/assistant/tool)")
    if add_generation_prompt and (not messages or messages[-1].get("role") != "tool"):
        out.append(ASSISTANT)
    return "".join(out)


# --- message builders -------------------------------------------------------
DEFAULT_IDENTITY = (
    "You are a helpful assistant. When you need facts you do not have, "
    "call one of the functions listed below."
)


def record_tool_call_turn(raw_text: str, calls: Sequence[ParsedCall]) -> dict[str, Any]:
    """The assistant message to append after the model asked for a tool.

    Split, not verbatim: `content` keeps the scaffold the model wrote before the
    call (<|input_lang|>, <think>, <task_plan>) and `tool_calls` carries the
    structured call. That is what the template expects -- an assistant message
    WITHOUT tool_calls is the end of a turn and gets an eos, so recording the
    raw text as content-only would wedge a </s> between the call and its result
    and break the single-turn shape the model is trained on.
    """
    cut = raw_text.find(TOOL_CALL_OPEN)
    return {
        "role": "assistant",
        "content": raw_text if cut == -1 else raw_text[:cut],
        "tool_calls": [
            {"function": {"name": c.name, "arguments": c.arguments or {}}}
            for c in calls
            if c.ok
        ],
    }


def tool_message(payload: Any, name: str | None = None) -> dict[str, Any]:
    """A role="tool" message. `name` is kept for logging and debugging only --
    the template deliberately does not render it, because results answer the
    calls above them in order."""
    msg: dict[str, Any] = {"role": "tool", "content": tool_response_content(payload)}
    if name:
        msg["name"] = name
    return msg


def user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}
