"""Run converters on a few hundred rows and print what comes out.

    nice -n 10 python -m sft_new.samples NAME [NAME ...] [--limit 300] [--show 2] [--out DIR]

For each converter: how many records came out, why rows were dropped, and a few finished examples (user turn, assistant turn). The
rendered `text` of every record is the string that would be tokenized. Writes DIR/NAME.jsonl when --out is given.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def run(name: str, limit: int, show: int, out: Path | None, seed: int = 0):
    from sft_new import converters as C
    from sft_new.compose import Late, Skip, finish
    fn, desc = C.CONVERTERS[name]
    rng = random.Random(seed)
    stats, rows, late = collections.Counter(), [], []
    for item in fn(rng, limit):
        if isinstance(item, Skip):
            stats["drop:" + str(item)[:48]] += 1
            continue
        if isinstance(item, Late):
            stats["late:" + item.lang] += 1
            if len(late) < show:
                late.append(item)
            continue
        try:
            rows.append(finish(item, rng))
        except Skip as e:
            stats["drop:" + str(e)[:48]] += 1
        except Exception as e:                                   # a converter bug must show up, not vanish
            stats["ERROR:" + type(e).__name__ + ":" + str(e)[:60]] += 1
        if len(rows) >= limit * 3:
            break
    print(f"\n=== {name}: {desc}\n    kept {len(rows)}   " + "   ".join(f"{k}={v}" for k, v in stats.most_common(8)))
    by = collections.Counter((r["task"], r["source_lang"], r["targ_lang"]) for r in rows)
    print("    task/src>tgt:", ", ".join(f"{t}:{s}>{g}x{n}" for (t, s, g), n in by.most_common(8)))
    picks = rng.sample(rows, min(show, len(rows)))
    for r in picks:
        msgs = json.loads(r["messages"])
        for m in msgs:
            body = m.get("content") or json.dumps(m.get("tool_calls"), ensure_ascii=False)
            print(f"    [{m['role']}] {body[:420]!r}")
        print("    ---")
    for it in late:
        print(f"    [late pretrain {it.lang}] {it.text[:240]!r}")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--match", help="only source files whose path contains this (naija/naija2 families)")
    a = ap.parse_args()
    from sft_new import converters as C
    C.FILE_MATCH = a.match
    if a.list or not a.names:
        for k, (_, d) in C.CONVERTERS.items():
            print(f"{k:24s} {d}")
        return
    for n in a.names:
        run(n, a.limit, a.show, a.out)


if __name__ == "__main__":
    main()
