#!/usr/bin/env python3
"""Pick structurally PERFECT conversations out of already-generated data and save them as few-shot exemplars.

    python scripts/build_fewshot.py                    # scan data/out/**, write seeds/fewshot.json

Why mined rather than hand-written: the exemplars have to demonstrate the exact marker format, think
placement and tool-call/result plumbing. Writing them by hand means inventing text in languages I cannot
check, and a subtly wrong exemplar is worse than none -- the model copies it. These are real samples that
passed every structural check, so what is being shown is known-good.

They are chosen for STRUCTURE, not language: a correct Yoruba conversation teaches an Efik generation what
the scaffolding looks like, and the prompt says so explicitly. Structure is what fails in low-resource
languages (70 invented pipe tokens vs 21 correct markers), not orthography.

Exemplars live in the SYSTEM prompt, so they are part of the shared prefix -- constant per (kind, language)
and free under prefix caching.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from postprocess_gen import _invented_pipe_tokens, _degenerate_user_message  # noqa: E402

OUT = HERE / "seeds" / "fewshot.json"
MARKER = re.compile(r"<\|input_lang\|><([a-z]{2,4})>")
THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _perfect(rec: dict) -> bool:
    """Every structural property an exemplar must demonstrate, with nothing to un-teach."""
    msgs = rec.get("messages") or []
    if not (6 <= len(msgs) <= 16) or msgs[-1]["role"] != "assistant":
        return False
    if _invented_pipe_tokens(msgs) or _degenerate_user_message(msgs):
        return False
    if rec.get("io_markers_ok") is not True:
        return False
    asst = [m for m in msgs if m["role"] == "assistant"]
    if not asst or not any("<response>" in (m.get("content") or "") for m in asst):
        return False
    # every assistant turn carries the marker pair, and every think block is English-looking
    for m in asst:
        c = m.get("content") or ""
        if not MARKER.search(c):
            return False
        for t in THINK.findall(c):
            if sum(ord(ch) < 128 for ch in t) / max(len(t), 1) < 0.95:
                return False
    return True


def _uses_tools(rec: dict) -> bool:
    msgs = rec["messages"]
    has_call = any(m.get("tool_calls") for m in msgs)
    has_result = any(m["role"] == "tool" for m in msgs)
    # and a think block on BOTH sides of the tool round trip, which is the behaviour hardest to elicit
    think_after = any(m["role"] == "assistant" and THINK.search(m.get("content") or "")
                      for i, m in enumerate(msgs) if i and msgs[i - 1]["role"] == "tool")
    return has_call and has_result and think_after


def main() -> int:
    recs = []
    root = HERE / "data" / "out"
    for shard in root.glob("sft*/*/shard-*.jsonl"):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    continue
    good = [r for r in recs if _perfect(r)]
    print(f"scanned {len(recs)} records, {len(good)} structurally perfect")
    if not good:
        raise SystemExit("no perfect samples found; generate some high-resource data first")

    with_tools = [r for r in good if _uses_tools(r)]
    without = [r for r in good if not _uses_tools(r)]
    # shortest of each, so the exemplars stay cheap in the shared prefix
    picked = []
    if with_tools:
        picked.append(min(with_tools, key=lambda r: len(r["text"])))
    if without:
        picked.append(min(without, key=lambda r: len(r["text"])))
    if not picked:
        raise SystemExit("no usable exemplars")

    payload = {
        "note": ("Structurally perfect conversations mined from generated data. They demonstrate the MARKER "
                 "FORMAT, THINK PLACEMENT and TOOL PLUMBING only -- their language is whatever they were "
                 "generated in, and a generation in another language must copy the structure, not the words."),
        "exemplars": [{"id": r["id"], "lang": r["lang"], "io_direction": r.get("io_direction"),
                       "uses_tools": _uses_tools(r), "messages": r["messages"]} for r in picked],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in picked:
        print(f"  {r['id']}  lang={r['lang']} io={r.get('io_direction')} "
              f"tools={_uses_tools(r)} msgs={len(r['messages'])} chars={len(r['text'])}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
