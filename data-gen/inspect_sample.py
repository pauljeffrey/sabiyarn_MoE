"""Print one generated record, readably, so a long run is watchable instead of a wall of counters.

Called once per chunk by both drivers with a RANDOMLY chosen record from that chunk. Random rather than the
first: the first record of a chunk is always the same (lang, task) because work is sorted for prefix-cache
locality, so printing it would show the same shape for hours and hide everything else.

Kept separate from the drivers because it is the one place that decides how much of a 20,000-token document to
show -- the whole point is to be able to read it in a terminal.
"""

from __future__ import annotations

import json
from typing import Any

# Enough to judge whether a turn is real, short enough that a 20k-token document does not fill the screen.
_HEAD = 700
_DOC_HEAD = 400
_RULE = "-" * 100


def _clip(text: str, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [+{len(text) - limit:,} chars]"


def format_record(rec: dict[str, Any], *, kind: str = "sft") -> str:
    """One record as something a person can actually read."""
    out = [_RULE]
    head = [f"id={rec.get('id')}", f"lang={rec.get('lang')}"]
    for key in ("io_direction", "confidence", "doc_words", "doc_lang", "doc_form"):
        if rec.get(key) is not None:
            head.append(f"{key}={rec[key]}")
    if rec.get("tasks"):
        head.append(f"tasks={','.join(rec['tasks'][:3])}")
    if rec.get("tools"):
        head.append(f"tools={','.join(rec['tools'][:4])}")
    out.append("  ".join(head))
    out.append(_RULE)

    if kind == "pretrain":
        out.append(f"TITLE: {_clip(rec.get('title', ''), 160)}")
        words = len((rec.get("text") or "").split())
        out.append(f"TEXT ({words:,} words): {_clip(rec.get('text', ''), _HEAD * 2)}")
        out.append(_RULE)
        return "\n".join(out)

    msgs = rec.get("messages") or rec.get("prompt_messages") or []
    for m in msgs:
        role = m.get("role", "?")
        if role == "tool":
            out.append(f"[tool:{m.get('name', '')}] {_clip(m.get('content', ''), _DOC_HEAD)}")
        elif m.get("tool_calls"):
            fn = m["tool_calls"][0]["function"]
            plan = _clip(m.get("content", ""), 300)
            out.append(f"[assistant->call] {plan}")
            out.append(f"    {fn['name']}({json.dumps(fn.get('arguments', {}), ensure_ascii=False)[:220]})")
        else:
            limit = _DOC_HEAD if role == "user" and len(m.get("content", "")) > 4000 else _HEAD
            out.append(f"[{role}] {_clip(m.get('content', ''), limit)}")
    for n in (1, 2):
        if rec.get(f"response_{n}"):
            rank = (rec.get("ranking") or [None, None])[n - 1]
            out.append(f"[candidate {n} ({rank})] {_clip(rec[f'response_{n}'], _HEAD)}")
    out.append(_RULE)
    return "\n".join(out)


def print_one(records: list[dict], *, kind: str, rng) -> None:
    """Print a random record from `records`, or nothing if the chunk kept none."""
    if not records:
        return
    print(format_record(rng.choice(records), kind=kind), flush=True)
