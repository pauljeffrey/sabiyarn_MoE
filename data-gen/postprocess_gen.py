"""Model response -> validated training record.

Every record carries BOTH representations, deliberately:
  * `messages` / `prompt_messages` -- role/content dicts. The source of truth.
  * `text` / `prompt_text`         -- exactly what the SabiYarn chat template renders from them.
Keeping both means a later template change is a cheap re-render (`python postprocess_gen.py --rerender`)
instead of regenerating the corpus, and training can consume the flat text without a Jinja dependency.

`instruction` / `input` / `context` / `response` are flattened from the LAST exchange for tooling that
expects the Alpaca shape. They are lossy for multi-turn samples; `messages` is what to train on.

Anything that fails validation is dropped and counted, never silently repaired -- a malformed special-token
sequence that reaches training teaches the model to emit malformed special-token sequences.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from providers.base import Response
from rendering.render import render_messages
from schemas.seed import TAGS, Seed

ROLES = ("system", "user", "assistant", "tool")
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_RESP = re.compile(r"<response>(.*)", re.S)

STATS: dict[str, int] = {}


def _drop(reason: str) -> None:
    STATS[reason] = STATS.get(reason, 0) + 1


def _json(text: str) -> Optional[dict]:
    """Parse the model's JSON, tolerating a ```json fence but nothing more adventurous."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _clean_messages(raw: Any) -> Optional[list[dict]]:
    """Normalise to template-shaped dicts, or None if the structure is unusable."""
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict] = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            return None
        role = m["role"]
        msg: dict[str, Any] = {"role": role, "content": (m.get("content") or "")}
        if role == "tool":
            if not msg["content"]:
                return None
            msg["name"] = m.get("name") or ""
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = (c or {}).get("function") or {}
                name, args = fn.get("name"), fn.get("arguments")
                if not name:
                    return None
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return None
                calls.append({"function": {"name": name, "arguments": args or {}}})
            msg["tool_calls"] = calls
            msg.pop("content", None) if not msg["content"] else None
        out.append(msg)
    return out


def _validate_conversation(msgs: list[dict], seed: Seed, *, ends_with: str) -> bool:
    conv = seed.conversation
    if not (conv["min_messages"] - 1 <= len(msgs) <= conv["max_messages"]):
        _drop(f"message_count:{len(msgs)}")
        return False
    if msgs[-1]["role"] != ends_with:
        _drop(f"ends_with:{msgs[-1]['role']}")
        return False
    if any(m["role"] == "system" for m in msgs[1:]):
        _drop("system_not_first")
        return False
    # a tool result must follow an assistant tool_call
    for i, m in enumerate(msgs):
        if m["role"] == "tool" and not (i and msgs[i - 1].get("tool_calls")):
            _drop("orphan_tool_result")
            return False
    tool_names = {t.name for t in seed.tools}
    for m in msgs:
        for c in m.get("tool_calls", []):
            if c["function"]["name"] not in tool_names:
                _drop("unknown_tool")
                return False
    return True


def _assistant_parts(content: str) -> dict[str, str]:
    think = _THINK.search(content)
    resp = _RESP.search(content)
    return {"think": (think.group(1).strip() if think else ""),
            "response": (resp.group(1).strip() if resp else "")}


def _flatten(msgs: list[dict]) -> dict[str, str]:
    """instruction/input/context/response from the last exchange, for Alpaca-shaped tooling."""
    last_user = next((m for m in reversed(msgs) if m["role"] == "user"), None)
    last_asst = next((m for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), None)
    ctx = "\n\n".join(m.get("content", "") for m in msgs if m["role"] == "tool")
    return {"instruction": (last_user or {}).get("content", ""), "input": "",
            "context": ctx, "response": _assistant_parts((last_asst or {}).get("content", ""))["response"]}


def _tags(raw: Any, fallback: list[str]) -> list[str]:
    got = [t for t in (raw or []) if t in TAGS]
    return sorted(set(got) | set(fallback)) or list(fallback)


def to_record(seed: Seed, resp: Response) -> Optional[dict]:
    md = resp.metadata or {}
    data = _json(resp.text)
    if data is None:
        _drop("json_invalid")
        return None
    base = {"id": resp.custom_id, "lang": md.get("lang", ""),
            "model": resp.model, "domain": md.get("domain", ""), "subtopic": md.get("subtopic", "")}

    if seed.kind == "pretrain":
        text = (data.get("text") or "").strip()
        if len(text.split()) < 60:
            _drop("too_short")
            return None
        if data.get("language_self_check") is False:
            _drop("language_self_check_false")
            return None
        return {**base, "genre": md.get("genre", ""), "title": (data.get("title") or "").strip(), "text": text}

    if seed.kind == "sft":
        msgs = _clean_messages(data.get("messages"))
        if msgs is None:
            _drop("messages_malformed")
            return None
        if not _validate_conversation(msgs, seed, ends_with="assistant"):
            return None
        if not _assistant_parts(msgs[-1].get("content", ""))["response"]:
            _drop("final_turn_has_no_response_token")
            return None
        return {**base, "tags": _tags(data.get("tags"), md.get("tags", [])),
                "tasks": data.get("tasks") or md.get("tasks", []),
                "tools": md.get("tools", []), "messages": msgs,
                "text": render_messages(msgs), **_flatten(msgs)}

    # rl
    msgs = _clean_messages(data.get("prompt_messages"))
    if msgs is None:
        _drop("prompt_messages_malformed")
        return None
    if not _validate_conversation(msgs, seed, ends_with="user"):
        return None
    responses = data.get("responses") or []
    want = int(seed.conversation["responses_per_prompt"])
    responses = [r for r in responses if isinstance(r, dict) and (r.get("content") or "").strip()]
    if len(responses) < 2:
        _drop("too_few_responses")
        return None
    quals = [(r.get("quality") or "").lower() for r in responses]
    if "best" not in quals or "worst" not in quals:
        _drop("no_best_or_worst")
        return None
    texts = [r["content"].strip() for r in responses]
    if len(set(texts)) < len(texts):
        _drop("duplicate_responses")
        return None
    order = sorted(range(len(responses)), key=lambda i: {"best": 0, "partial": 1, "worst": 2}.get(quals[i], 1))
    rec = {**base, "tags": _tags(data.get("tags"), md.get("tags", [])),
           "tasks": data.get("tasks") or md.get("tasks", []), "tools": md.get("tools", []),
           "prompt_messages": msgs, "prompt_text": render_messages(msgs, add_generation_prompt=True),
           "ranking": [quals[i] for i in order],
           "rationale": [responses[i].get("why", "") for i in order],
           **_flatten(msgs)}
    for n in range(want):
        rec[f"response_{n + 1}"] = texts[order[n]] if n < len(order) else None
    return rec


def summary() -> str:
    if not STATS:
        return "no drops"
    return "drops: " + ", ".join(f"{k}={v}" for k, v in sorted(STATS.items(), key=lambda kv: -kv[1]))
