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

STATS: Counter = Counter()

# <|input_lang|><yor>  /  <|yor|>  /  <|yor>  -- all three seen in the wild
_LANG_AFTER_MARKER = re.compile(r"<\|(?:input_lang|target_lang)\|>\s*<([a-z]{2,4})>")
_PSEUDO_LANG = re.compile(r"<\|([a-z]{2,4})\|?>")
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_PLAN = re.compile(r"<task_plan>(.*?)</task_plan>", re.S)
_RESPONSE = re.compile(r"<response>(.*)", re.S)
_VERB = re.compile(r"<\|?[A-Za-z_][A-Za-z0-9_|]*\|?>")
_LABEL_AT_START = re.compile(r"^\s*(<[a-z_A-Z]{2,12}>)")
# "pcm->igbo", "pcm -> igbo", "<lang>pcm<target_lang>igbo"
_LANGPAIR_JUNK = re.compile(
    r"^\s*(?:<lang>)?\s*([a-z]{2,4})\s*(?:->|<target_lang>)\s*([a-zA-Z]{2,10})\s*")


def _lang_code(name: str) -> Optional[str]:
    """'igbo' -> 'ibo', 'yoruba' -> 'yor'. Only names we can map unambiguously."""
    n = name.strip().lower()
    if n in VALID_LANGS:
        return n
    return {"igbo": "ibo", "yoruba": "yor", "hausa": "hau", "english": "eng", "pidgin": "pcm",
            "efik": "efi", "urhobo": "urh", "twi": "twi", "akan": "aka", "fon": "fon", "ewe": "ewe",
            "fulah": "ful", "fulfulde": "fuv"}.get(n)


def _langs_from_content(c: str, fallback: str) -> tuple[str, Optional[str]]:
    """(input_lang, target_lang) recovered from however the generator wrote them."""
    proper = _LANG_AFTER_MARKER.findall(c)
    pseudo = [g for g in _PSEUDO_LANG.findall(c) if g in VALID_LANGS]
    found = [g for g in (proper + pseudo) if g in VALID_LANGS]
    if not found:
        return fallback, fallback
    src = found[0]
    tgt = found[1] if len(found) > 1 else None
    return src, tgt


def _verbs(c: str) -> list[str]:
    m = _PLAN.search(c)
    if not m:
        return []
    return [v for v in _VERB.findall(m.group(1)) if v in VALID_VERBS]


def _plan_for_tool(name: str) -> list[str]:
    """A tool-calling turn that never carried a plan: infer one from the tool it called."""
    by_tool = {
        "search_documents": ["<|RAG|>"], "search_internet": ["<|explain|>"],
        "lookup_health_guidance": ["<|recommend|>"], "find_health_facility": ["<|recommend|>"],
        "calculate": ["<|math|>"], "run_statistics": ["<|math|>"], "convert_units": ["<|math|>"],
        "get_exchange_rate": ["<|math|>"], "get_market_prices": ["<|analyze|>"],
        "search_db": ["<|data_extract|>"], "db_get_record": ["<|data_extract|>"],
        "db_insert_record": ["<|plan|>"], "send_message": ["<|plan|>"], "set_reminder": ["<|plan|>"],
        "translate_text": ["<translate>"], "get_weather_forecast": ["<|explain|>"],
        "get_transport_route": ["<|plan|>"], "lookup_crop_guidance": ["<|recommend|>"],
        "lookup_legal_info": ["<|explain|>"], "get_current_datetime": ["<|chat|>"],
    }
    return by_tool.get(name, ["<|plan|>"])


SEQUENCE_TASKS = {"ner", "pos_tagging", "token_labelling"}


def _turn_from_assistant(m: dict, lang: str, sequence: bool = False) -> Optional[dict]:
    c = m.get("content") or ""
    calls = m.get("tool_calls") or []
    src, tgt = _langs_from_content(c, lang)
    think = None
    tm = _THINK.search(c)
    if tm and tm.group(1).strip():
        think = tm.group(1).strip()
    verbs = _verbs(c)

    if calls:
        fn = (calls[0] or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            STATS["drop_tool_call_without_name"] += 1
            return None
        if not verbs:
            verbs = _plan_for_tool(name)
            STATS["fixed_inferred_plan_for_tool_turn"] += 1
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        return {"role": "assistant", "input_lang": src, "think": think, "task_plan": verbs,
                "tool_call": {"name": name, "arguments": args or {}}}

    rm = _RESPONSE.search(c)
    if not rm:
        STATS["drop_no_response_text"] += 1
        return None
    body = rm.group(1)

    # A language-pair prefix ASSERTS a target language. If it contradicts what the markers said, the record
    # is unrecoverable: we cannot know which the generator meant.
    jm = _LANGPAIR_JUNK.match(body)
    label = None
    if jm:
        asserted = _lang_code(jm.group(2))
        if asserted and tgt and asserted != tgt:
            STATS["drop_contradictory_target_language"] += 1
            return None
        if asserted:
            tgt = asserted
            STATS["fixed_target_from_langpair_prefix"] += 1
        body = body[jm.end():]
    # A label token sitting after <response> is a real label in the wrong place.
    lm = _LABEL_AT_START.match(body)
    if lm:
        tok = lm.group(1)
        key = next((k for k, v in LABEL_TOKENS.items() if v == tok), None)
        if key:
            label = key
            body = body[lm.end():]
            STATS["fixed_label_token_moved"] += 1
        else:
            body = body[lm.end():]
            STATS["fixed_stray_tag_removed"] += 1
    body = body.strip()
    if not body:
        STATS["drop_empty_response_after_cleanup"] += 1
        return None
    if not verbs:
        verbs = ["<|chat|>"]
        STATS["fixed_default_plan_for_answer_turn"] += 1
    return {"role": "assistant", "input_lang": src, "target_lang": tgt or src, "think": think,
            "task_plan": verbs, "response": body, "label_token": label,
            "sequence_labels": sequence}


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
