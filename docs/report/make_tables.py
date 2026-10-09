"""Turn a finished sft-all build directory into markdown tables for the technical report.

    python docs/report/make_tables.py BUILD_DIR            # writes docs/report/generated/{sft_all_stats,sft_all_audit}.md

Reads BUILD_DIR/stats/*.json (one per finished step), BUILD_DIR/audit.json and BUILD_DIR/drop_list.json (from `python -m sft_new.audit_translation`).
Works on a partial build too (it reports only the steps that finished).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    build = Path(sys.argv[1])
    steps = [json.loads(f.read_text()) for f in sorted((build / "stats").glob("*.json"))]
    tasks, src, tgt, pairs, drops, ninput = (collections.Counter() for _ in range(6))
    for j in steps:
        for k, v in j["counts"].items():
            if k.startswith("kept:"):
                tasks[k[5:]] += v
            elif k.startswith("lang:") and ">" in k:
                s, t = k[5:].split(">", 1)
                pairs[k[5:]] += v
                src[s] += v
                tgt[t] += v
            elif k.startswith("drop:"):
                drops[k[5:]] += v
            elif k.startswith("ninput:"):
                ninput[k[7:]] += v
    total = sum(j["kept"] for j in steps)
    late = sum(j.get("late_rows", 0) for j in steps)
    parts = [f"## A.7 Result of the `sft-all` build ({len(steps)} steps finished)\n",
             f"**Rows kept: {total:,}**; rows routed to late pretraining folders: {late:,}; total build time of the finished steps: {sum(j['seconds'] for j in steps) / 3600:.1f} h.\n",
             "### By converter\n", md([(j["converter"], f"{j['kept']:,}", f"{j.get('late_rows', 0):,}", j["seconds"]) for j in sorted(steps, key=lambda j: -j["kept"])], ["converter", "kept", "late rows", "seconds"]),
             "\n### By task\n", md([(k, f"{v:,}", f"{100 * v / max(1, total):.1f}%") for k, v in tasks.most_common()], ["task", "rows", "share"]),
             "\n### By language (as source / as target)\n",
             md([(k, f"{src[k]:,}", f"{tgt[k]:,}") for k in sorted(set(src) | set(tgt), key=lambda k: -(src[k] + tgt[k]))], ["language", "as source", "as target"]),
             "\n### Languages per input (`<|input_lang|>` size)\n", md([(k, f"{v:,}") for k, v in sorted(ninput.items())], ["languages in input", "rows"]),
             "\n### Top 40 language pairs\n", md([(k, f"{v:,}") for k, v in pairs.most_common(40)], ["src>tgt", "rows"]),
             "\n### Most frequent drop reasons\n", md([(k, f"{v:,}") for k, v in drops.most_common(25)], ["reason", "rows"])]
    gen = HERE / "generated"
    gen.mkdir(exist_ok=True)
    (gen / "sft_all_stats.md").write_text("\n".join(parts) + "\n")

    ap = build / "audit.json"
    if ap.exists():
        audit = json.loads(ap.read_text())
        low = next((k for m in audit.values() for k in m if k.startswith("share<")), "share<0.46")
        rows = sorted(audit.items(), key=lambda kv: kv[1]["mean"])
        drop = json.loads((build / "drop_list.json").read_text()) if (build / "drop_list.json").exists() else []
        a = ["## A.8 AfriCOMET-QE audit of translation rows\n",
             f"{len(audit)} (source, pair) groups scored; {len(drop)} on the drop list (>= 20 scored, share below the 0.46 failure line >= 0.8, mean < 0.38).\n",
             md([(k, m["n"], f"{m['mean']:.3f}", f"{m.get('median', float('nan')):.3f}", f"{m.get('p10', float('nan')):.3f}", f"{m[low]:.2f}", f"{m.get('population', '')}") for k, m in rows],
                ["source | pair", "scored", "mean", "median", "P10", low, "rows in group"]),
             "\n### Drop list\n", md([(d["group"], d["n"], f"{d['mean']:.3f}", f"{d[low]:.2f}") for d in drop], ["group", "scored", "mean", low]) if drop else "_empty_"]
        (gen / "sft_all_audit.md").write_text("\n".join(a) + "\n")
    print("wrote", [p.name for p in gen.glob("*.md")])


if __name__ == "__main__":
    main()
