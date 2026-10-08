"""The SabiYarn wire format: the one place tags, rendering and parsing live.

Everything here is pure stdlib so the package ports to any codebase. The
renderer is a hand-written equivalent of sabiyarn/chat_template.jinja (jinja2
is NOT a dependency of the root environment, and tokenizer.apply_chat_template
needs it), byte-identical to the strings asserted in tests/test_chat_template.py.

Two format facts drive the whole framework:

1. At INFERENCE the policy emits a tool call INLINE inside one assistant turn:
       <|input_lang|><yor><think>...</think><task_plan>...</task_plan>
       <tool_call>name\n{"arg": "value"}</tool_call>
   So the agent loop parses text, not a structured `tool_calls` field.

2. The chat template DROPS message['content'] whenever 'tool_calls' is present.
   A tool-calling turn must therefore go back into the history as content-only
   (the raw generated text, tags and all) or the <think>/<task_plan> prefix
   vanishes from the next prompt and the model sees a format it never saw in
   training. record_tool_call_turn() is the helper that gets this right.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# --- tokens -----------------------------------------------------------------
# Every one of these is a single token in BeardedMonster/SabiYarn-32k. Do not
# invent new ones: <tool_result> (not in the vocab) cost ~5 byte-BPE tokens per
# tag and is an actively guarded regression.
BOS = "<s>"
EOS = "</s>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TASK_PLAN_OPEN, TASK_PLAN_CLOSE = "<task_plan>", "</task_plan>"
RESPONSE_OPEN = "<response>"  # deliberately has no closing tag: it runs to EOS
INPUT_LANG, TARGET_LANG = "<|input_lang|>", "<|target_lang|>"

# Ordered task-plan verbs. An out-of-set verb is a hard error in the data
# pipeline, so the agent treats one as a format violation worth surfacing.
VALID_VERBS = frozenset(
    {
        "<|chat|>", "<|generate|>", "<|edit|>", "<|data_extract|>", "<|math|>",
        "<|JSON|>", "<|code|>", "<|plan|>", "<|analyze|>", "<|recommend|>",
        "<|explain|>", "<|debug|>", "<|RAG|>", "<summarize>", "<translate>",
        "<classify>", "<NER>", "<identify>", "<qa>", "<diacritize>", "<clean>",
    }
)

# Generation must stop at the close of a tool call so the tool can actually run,
# and at a role marker in case the model tries to continue the conversation itself.
STOP_STRINGS: tuple[str, ...] = (TOOL_CALL_CLOSE, USER, SYSTEM)

# --- parsing ----------------------------------------------------------------
R_THINK = re.compile(r"<think>(.*?)</think>", re.S)
R_TASK_PLAN = re.compile(r"<task_plan>(.*?)</task_plan>", re.S)
R_RESPONSE = re.compile(r"<response>(.*)", re.S)
R_TOOL_CALL = re.compile(r"<tool_call>([^\n<]+)\n(.*?)</tool_call>", re.S)
R_INPUT_LANG = re.compile(r"<\|input_lang\|><([a-z]{2,4})>")
R_TARGET_LANG = re.compile(r"<\|target_lang\|><([a-z]{2,4})>")
R_VERB = re.compile(r"<\|?[a-zA-Z_]+\|?>")
# A tool call the model opened but never closed, because generation hit the token
# budget mid-call. Worth telling apart from "no tool call" when deciding to retry.
R_TOOL_CALL_OPEN_ONLY = re.compile(r"<tool_call>([^\n<]*)\n?(.*)$", re.S)


def _tojson(value: Any) -> str:
    """The template's `tojson` filter: json.dumps(value, ensure_ascii=False).

    Note the default separators, so arguments render as {"city": "Lagos"} with a
    space after the colon -- that is what the model was trained on.
    """
    return json.dumps(value, ensure_ascii=False)


def tool_response_content(payload: Any) -> str:
    """Serialize a tool's return value for a role="tool" message.

    Compact separators, matching data-gen's tool_result_content(). Strings pass
    through untouched: the corpus keeps tool payloads verbatim rather than
    re-encoding them as JSON strings.
    """
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ParsedCall:
    """One <tool_call> found in generated text."""

    name: str
    arguments: dict[str, Any] | None  # None when the JSON would not parse
    raw_arguments: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ParsedTurn:
    """Everything of interest in one assistant turn's raw text."""

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
        """What a user should see: the <response> body, else the raw text with
        the scaffolding tags stripped (a turn that is only a tool call has no
        <response>, and a malformed turn may have none either).

        <response> has no closing tag -- it runs to EOS -- so the body is cut at
        EOS and at any role marker the model may have hallucinated past it.
        """
        if self.response is not None:
            body = self.response
            for marker in (EOS, USER, SYSTEM, ASSISTANT, TOOL_CALL_OPEN):
                cut = body.find(marker)
                if cut != -1:
                    body = body[:cut]
            return body.strip()
        return strip_tags(self.raw).strip()


def parse_assistant(text: str) -> ParsedTurn:
    """Parse one assistant turn. Never raises: a malformed turn is data, and the
    agent decides whether to retry, so every problem is reported as a field."""
    think_m = R_THINK.search(text)
    plan_m = R_TASK_PLAN.search(text)
    resp_m = R_RESPONSE.search(text)
    in_m, tgt_m = R_INPUT_LANG.search(text), R_TARGET_LANG.search(text)

    verbs: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()
    if plan_m:
        verbs = tuple(R_VERB.findall(plan_m.group(1)))
        invalid = tuple(v for v in verbs if v not in VALID_VERBS)

    calls = []
    for name, raw_args in R_TOOL_CALL.findall(text):
        name = name.strip()
        try:
            parsed = json.loads(raw_args)
            if not isinstance(parsed, dict):
                calls.append(ParsedCall(name, None, raw_args, "arguments must be a JSON object"))
                continue
            calls.append(ParsedCall(name, parsed, raw_args))
        except json.JSONDecodeError as exc:
            calls.append(ParsedCall(name, None, raw_args, f"invalid JSON arguments: {exc.msg}"))

    truncated = False
    if not calls and TOOL_CALL_OPEN in text and TOOL_CALL_CLOSE not in text:
        truncated = True

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
    """Remove scaffolding so text is presentable: drop <think> and <task_plan>
    blocks with their contents, drop tool calls, then drop any remaining tags."""
    out = R_THINK.sub("", text)
    out = R_TASK_PLAN.sub("", out)
    out = R_TOOL_CALL.sub("", out)
    for tag in (RESPONSE_OPEN, EOS, BOS, ASSISTANT, USER, SYSTEM):
        out = out.replace(tag, "")
    out = re.sub(r"<\|[a-zA-Z_]+\|><[a-z]{2,4}>", "", out)  # <|input_lang|><yor>
    return out


# --- rendering --------------------------------------------------------------
def render(
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    bos: str = BOS,
    eos: str = EOS,
) -> str:
    """Render messages to the exact training text.

    Equivalent to sabiyarn/chat_template.jinja. Verified against the ground
    truth in tests/test_chat_template.py:

        <s><|system|>S</s><|user|>hi</s><|assistant|>yo</s>
        <tool_call>get_weather\\n{"city": "Lagos"}</tool_call></s>
        <tool_response>get_weather 31C</tool_response></s>

    One deliberate divergence: the template silently skips an unrecognised role,
    which turns a typo into a prompt that quietly omits a turn. This raises.
    """
    out: list[str] = []
    for i, message in enumerate(messages):
        if i == 0:
            out.append(bos)
        role = message.get("role")
        if role == "system":
            out += [SYSTEM, message.get("content", ""), eos]
        elif role == "user":
            out += [USER, message.get("content", ""), eos]
        elif role == "tool":
            name = message.get("name")
            out += [
                TOOL_RESPONSE_OPEN,
                f"{name} " if name else "",
                message.get("content", ""),
                TOOL_RESPONSE_CLOSE,
                eos,
            ]
        elif role == "assistant":
            out.append(ASSISTANT)
            # Key-presence, not truthiness: the template branches on
            # `'tool_calls' not in message`, so an empty list emits nothing.
            if "tool_calls" not in message:
                out += [message.get("content", ""), eos]
            else:
                for call in message["tool_calls"]:
                    fn = call["function"]
                    out += [
                        TOOL_CALL_OPEN,
                        fn["name"],
                        "\n",
                        _tojson(fn.get("arguments") or {}),
                        TOOL_CALL_CLOSE,
                        eos,
                    ]
        else:
            raise ValueError(f"unknown message role {role!r} (expected system/user/assistant/tool)")
    if add_generation_prompt:
        out.append(ASSISTANT)
    return "".join(out)


# --- message builders -------------------------------------------------------
DEFAULT_IDENTITY = "You are an AI assistant."


def format_tools(schemas: Iterable[dict[str, Any]], identity: str = DEFAULT_IDENTITY) -> str:
    """Build the system message that carries the tool catalogue.

    Compact JSON on one line, matching the corpus form built by data-gen's
    _rebuild_system_message(); indent=2 cost roughly 40% more tokens for no
    measured benefit. Tool schemas reach the model as plain text in the system
    turn -- there is no <tools> slot in the template.
    """
    catalogue = list(schemas)
    if not catalogue:
        return identity
    return f"{identity} You have these tools:\n" + json.dumps(
        catalogue, ensure_ascii=False, separators=(",", ":")
    )


def record_tool_call_turn(raw_text: str) -> dict[str, Any]:
    """The assistant message to append after the model asked for a tool.

    Content-only on purpose. Passing `tool_calls` here would make the template
    drop the <think>/<task_plan> prefix, so the next prompt would show the model
    a shape it never saw in training; keeping the raw text reproduces the inline
    form exactly. (data-gen/postprocess_gen.py re-appends the call to content
    for the same reason: 0 of 108 candidates contained <tool_call> without it.)
    """
    return {"role": "assistant", "content": raw_text}


def tool_message(name: str, payload: Any) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": tool_response_content(payload)}


def user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


@dataclass
class Turn:
    """A rendered prompt plus the messages it came from, for debugging."""

    prompt: str
    messages: list[dict[str, Any]] = field(default_factory=list)
