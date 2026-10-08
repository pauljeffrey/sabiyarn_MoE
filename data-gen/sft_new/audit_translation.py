"""AfriCOMET-QE audit of the translation rows of a finished (or running) sft-all build.

    HF_HOME unset (the QE model lives in the normal cache)
    python -m sft_new.audit_translation BUILD_DIR [--per 30] [--comet-python ../.venv-comet/bin/python]

The build keeps, for every source dataset and language pair, a uniform sample of up to 40 SHORT translation rows (<= 500 characters both sides) in
BUILD_DIR/audit_samples/. This script scores up to `--per` of them per (source dataset, pair) with masakhane/africomet-qe-stl-1.1 and writes
BUILD_DIR/audit.json (mean, median, P10, share below the 0.46 failure line, pairs per second) and BUILD_DIR/drop_list.json: the (source dataset, pair)
groups that are clearly unfaithful (at least 20 scored, share below 0.46 of 0.8 or more, mean below 0.38). The drop list is applied when the data are
re-downloaded for tokenization; nothing is deleted from the Hub here.

Why a sample and not every row: QE runs at roughly 1 pair per second on this CPU (the throughput is printed), so millions of rows are out of reach,
but a 30-pair sample per group already separates a 0.63 control from a 0.33 unfaithful source. Short rows only, because QE scores fall with length
(the same known-good Igbo set scores 0.63 on sentences and 0.46 on paragraphs).

Pairs QE cannot judge are not scored: both languages must be in AfriCOMET's encoder list, and the four pairs the repo's own validity run found
unreliable (tsn>sot, sot>tsn, fon>ewe, ewe>twi) are skipped.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from sft_new.quality_check import LOW, scorer, summarize  # noqa: E402

UNRELIABLE = {("tsn", "sot"), ("sot", "tsn"), ("fon", "ewe"), ("ewe", "twi")}
MIN_SCORED, MAX_SHARE_LOW, MAX_MEAN = 20, 0.8, 0.38


def main():
    from rl.bin_data import AFRICOMET_CODE
    ap = argparse.ArgumentParser()
    ap.add_argument("build", type=Path)
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--comet-python", default=str(HERE.parent / ".venv-comet" / "bin" / "python"))
    a = ap.parse_args()
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted((a.build / "audit_samples").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["source_lang"] in AFRICOMET_CODE and r["targ_lang"] in AFRICOMET_CODE and (r["source_lang"], r["targ_lang"]) not in UNRELIABLE:
                groups[f"{r['source_dataset']}|{r['source_lang']}>{r['targ_lang']}"].append(r)
    prev = json.loads((a.build / "audit.json").read_text()) if (a.build / "audit.json").exists() else {}
    score, proc = scorer(a.comet_python)
    res, drop, t0, n_scored = dict(prev), [], time.time(), 0
    try:
        for key, rows in sorted(groups.items()):
            if key in prev or len(rows) < MIN_SCORED:
                continue
            rows = rows[:a.per]
            sc = score([(r["input"], r["response"]) for r in rows])
            n_scored += len(sc)
            res[key] = {**summarize(sc), "population": rows[0]["population"]}
            (a.build / "audit.json").write_text(json.dumps(res, indent=1))
            print(key[:90], res[key], f"{n_scored / (time.time() - t0):.2f} pairs/s", flush=True)
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    for key, m in res.items():
        if m["n"] >= MIN_SCORED and m[f"share<{LOW}"] >= MAX_SHARE_LOW and m["mean"] < MAX_MEAN:
            drop.append({"group": key, **m})
    (a.build / "drop_list.json").write_text(json.dumps(drop, indent=1))
    print(f"{len(res)} groups scored, {len(drop)} on the drop list")


if __name__ == "__main__":
    main()
