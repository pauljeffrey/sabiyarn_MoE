"""Turn the per-converter sample files written by `sft_new.samples --out DIR` into one readable SAMPLES.md.

    python -m sft_new.make_samples_md DIR OUT.md [--per 2]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cut(s: str, n: int) -> str:
    s = s.replace("\r", "")
    return s if len(s) <= n else s[:n].rstrip() + " [...]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    from sft_new import converters as C
    rng = random.Random(7)
    out = ["# Samples of the SFT conversions", "",
           "Two converted rows per converter, taken from the first few hundred rows of each source. `user` is the prompt as the model sees it, `assistant` is the",
           "canonical turn. Long fields are cut. Nothing here was edited by hand.", ""]
    for name, (_, desc) in C.CONVERTERS.items():
        files = [a.dir / f"{name}.jsonl"] + sorted(a.dir.glob(f"{name}__*.jsonl"))          # naija/naija2 have one file per source family
        out += [f"## {name}", "", desc, ""]
        rows = [json.loads(l) for f in files if f.exists() for l in f.read_text(encoding="utf-8").splitlines()]
        if not rows:
            out += ["_no row survived the filters in the sample (see the report for why)_", ""]
            continue
        for r in rng.sample(rows, min(a.per if len(files) == 1 else a.per * 5, len(rows))):
            meta = json.loads(r["metadata"])
            out += [f"**task** `{r['task']}` · **source_lang** `{r['source_lang']}` · **targ_lang** `{r['targ_lang']}` · **dataset** `{r['source_dataset']}`"
                    + (f" · **instruction language** `{meta['instruction_language']}`" if meta.get("instruction_language") else ""), ""]
            for m in json.loads(r["messages"]):
                body = m.get("content") or json.dumps(m.get("tool_calls"), ensure_ascii=False)
                out += [f"- **{m['role']}**: `{cut(body, 520).replace(chr(10), ' ⏎ ').replace(chr(96), chr(39))}`"]
            out += [""]
    a.out.write_text("\n".join(out), encoding="utf-8")
    print("wrote", a.out, len(out), "lines")


if __name__ == "__main__":
    main()
