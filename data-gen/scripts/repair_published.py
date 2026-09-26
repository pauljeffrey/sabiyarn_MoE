#!/usr/bin/env python3
"""Repair the already-published data: parse the broken marker strings back into fields, re-assemble correctly.

    python scripts/repair_published.py --report                 # audit only, writes nothing
    python scripts/repair_published.py --out data/out/sft__repaired
    python scripts/repair_published.py --out ... --push         # and push the repaired shards

The published records were generated when the model was asked for the finished marker string, and it got it
wrong at scale. Measured over 108 records:

    114  tool-call turn with no <task_plan>
     88  assistant turn missing <|input_lang|>
     80  records with <think> on EVERY assistant turn (token waste)
     67  turn missing <|target_lang|>
    ~150 pseudo-markers: <|pcm>, <|eng|>, <|ful>, <|translate|>
     78  junk tokens after <response>: <tag>, <sentiment>, <lang>pcm<target_lang>igbo

What can be repaired and what cannot:

  REPAIRABLE -- the information is present, only the scaffolding is wrong. A `<|pcm>` where `<|input_lang|>`
  belonged still tells us the language is pcm. A missing `<task_plan>` on a tool-calling turn can be inferred
  from the tool. A `<tag>`/`<sentiment>` label after `<response>` is a real label in the wrong place.

  NOT REPAIRABLE -- the information is absent or contradicted. `<response><lang>pcm<target_lang>yoruba` on a
  turn whose content is Yoruba means the generator ASSERTED the wrong target language; there is no way to know
  what it meant, so the record is dropped. Same for a turn with no recoverable response text.

Nothing is guessed. Every repair is a re-expression of information already in the record, and anything else is
dropped and counted. The output goes through the same assemble.py as fresh generation, so repaired records are
byte-identical in structure to new ones.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from assemble import LABEL_TOKENS, VALID_LANGS, VALID_VERBS, AssemblyError, build_messages  # noqa: E402
from rendering.render import render_messages  # noqa: E402

# The marker-string parser lives in disassemble.py -- the few-shot renderer (prompts._fewshot_block) and the
# post-processor need exactly the same recoveries, and three copies of this arithmetic would drift apart.
from disassemble import (SEQUENCE_TASKS, STATS, _lang_code, _langs_from_content,  # noqa: E402,F401
                         _plan_for_tool, _turn_from_assistant, _verbs)


def repair(rec: dict) -> Optional[dict]:
    msgs = rec.get("messages")
    if isinstance(msgs, str):
        STATS["drop_messages_was_a_string"] += 1
        return None
    if not isinstance(msgs, list) or not msgs:
        STATS["drop_no_messages"] += 1
        return None
    lang = rec.get("lang") or "eng"
    sequence = bool(SEQUENCE_TASKS & set(rec.get("tasks") or [])) or "ner" in (rec.get("tags") or []) \
        or "pos-tagging" in (rec.get("tags") or [])
    turns: list[dict] = []
    for m in msgs:
        if not isinstance(m, dict):
            STATS["drop_non_dict_message"] += 1
            return None
        role = m.get("role")
        if role == "assistant":
            t = _turn_from_assistant(m, lang, sequence)
            if t is None:
                return None
            turns.append(t)
        elif role in ("user", "system", "tool"):
            content = (m.get("content") or "").strip()
            if not content:
                STATS["drop_empty_non_assistant_turn"] += 1
                return None
            turns.append({"role": role, "content": content,
                          **({"name": m.get("name") or ""} if role == "tool" else {})})
        else:
            STATS["drop_unknown_role"] += 1
            return None
    try:
        rebuilt = build_messages(turns)
    except AssemblyError as exc:
        STATS[f"drop_assembly:{str(exc)[:44]}"] += 1
        return None
    out = dict(rec)
    out["messages"] = rebuilt
    out["text"] = render_messages(rebuilt)
    out["repaired"] = True
    STATS["repaired_ok"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/tmp/hfdata", help="downloaded dataset snapshot")
    ap.add_argument("--out", default=None, help="write repaired shards here (mirrors <kind>/<lang>/)")
    ap.add_argument("--report", action="store_true", help="audit only")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default="BeardedMonster/data-gen")
    a = ap.parse_args()

    shards = sorted(Path(a.src).rglob("*.jsonl"))
    if not shards:
        raise SystemExit(f"no shards under {a.src}. Download first:\n"
                         f"    huggingface-cli download {a.repo_id} --repo-type dataset --local-dir {a.src}")
    written: list[Path] = []
    total = 0
    for shard in shards:
        kind = shard.parts[-3] if len(shard.parts) >= 3 else "sft"
        if kind not in ("sft", "rl", "pretrain"):
            continue
        keep = []
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except ValueError:
                STATS["drop_unparseable_line"] += 1
                continue
            if kind == "pretrain":          # plain prose, no markers to repair
                STATS["pretrain_passthrough"] += 1
                keep.append(rec)
                continue
            fixed = repair(rec)
            if fixed:
                keep.append(fixed)
        if keep and a.out and not a.report:
            dest = Path(a.out) / shard.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            p = dest / shard.name.replace("shard-", "repaired-")
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
            written.append(p)

    ok = STATS["repaired_ok"]
    print(f"\n{total} records read, {ok} repaired ({ok/max(total,1):.0%})\n")
    print("repairs applied (information re-expressed, nothing guessed):")
    for k, v in sorted(STATS.items()):
        if k.startswith("fixed_"):
            print(f"  {v:>5}  {k[6:]}")
    print("\ndropped (information absent or self-contradictory):")
    for k, v in sorted(STATS.items(), key=lambda kv: -kv[1]):
        if k.startswith("drop_"):
            print(f"  {v:>5}  {k[5:]}")
    if written:
        print(f"\nwrote {len(written)} repaired shard(s) under {a.out}")
        if a.push:
            from hub import push_shards
            push_shards("sft", written, repo_id=a.repo_id, name_prefix="repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
