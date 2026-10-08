"""Cap the translation rows of every language in `BeardedMonster/sft-new` at MAX rows (default 1,000,000), keeping the best.

Ranking (highest first), applied across all `translation-*` folders of the language:
  1. quality tier:  FineTranslations rows (edu-score gated, <= 100k per language)  >  multi-open rows by their 1-10 judge score  >  OPUS-100 rows
                    (no quality signal; software strings and subtitles, ranked as a score-7 row)
  2. inside a tier: pairs whose length ratio is closest to the language's median ratio (a wildly different length is the commonest sign of a
                    misaligned or truncated translation), then longer sentences (up to 25 words) before very short ones
  3. a stable hash as the final tie-break, so the choice is reproducible and not biased towards any file order.
Only languages above the cap are touched. Rows are removed from the parquet files in place and `text` / `messages` stay consistent.

    python -m sft_new.cap_translate [--max 1000000] [--langs afr] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "BeardedMonster/sft-new"


def tier_score(source: str, quality) -> float:
    if source == "HuggingFaceFW/finetranslations":
        return 100.0 + float(quality or 0)
    if source == "African-Languages-Lab/multi-open":
        return float(quality or 0)
    return 7.0                                                    # OPUS-100 and anything without a quality signal


def rank_keys(rows: list[dict], lang_median_logratio: float) -> list[tuple]:
    out = []
    for r in rows:
        a, b = (r["input"] or ""), (r["response"] or "")
        la, lb = max(1, len(a)), max(1, len(b))
        dev = abs(np.log(lb / la) - lang_median_logratio)
        words = min(25, len((a if r["source_lang"] == "eng" else b).split()))
        h = int.from_bytes(hashlib.blake2b((a + "\x1f" + b).encode(), digest_size=4).digest(), "little")
        out.append((tier_score(r["source_dataset"], r["quality"]), -round(dev, 1), words, h))
    return out


def select_top(keys: list[tuple], max_rows: int) -> np.ndarray:
    """-> boolean keep mask over `keys` (highest keys first)."""
    order = sorted(range(len(keys)), key=lambda i: keys[i], reverse=True)[:max_rows]
    keep = np.zeros(len(keys), dtype=bool)
    keep[order] = True
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1_000_000)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from africomet_translation.build import hf_token
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, HfFileSystem, hf_hub_download

    token = hf_token()
    api, fs = HfApi(token=token), HfFileSystem(token=token)
    files = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
                   if f.path.endswith(".parquet") and f.path.split("/")[1].startswith("translation-"))
    by_lang: dict[str, list[str]] = defaultdict(list)
    for p in files:
        by_lang[p.split("/")[1].rsplit("-", 1)[1]].append(p)
    cols = ["input", "response", "source_lang", "targ_lang", "source_dataset", "quality"]
    for lang, ps in sorted(by_lang.items()):
        if a.langs and lang not in a.langs:
            continue
        rows, owner = [], []
        for p in ps:
            with fs.open(f"datasets/{REPO}/{p}", "rb") as fh:
                t = pq.read_table(fh, columns=cols).to_pylist()
            rows += t
            owner += [(p, i) for i in range(len(t))]
        if len(rows) <= a.max:
            print(lang, len(rows), "rows: under the cap, untouched")
            continue
        ratios = [np.log(max(1, len(r["response"] or "")) / max(1, len(r["input"] or ""))) for r in rows[:200_000]]
        keys = rank_keys(rows, float(np.median(ratios)))
        keep = select_top(keys, a.max)
        print(lang, f"{len(rows):,} rows -> keeping {int(keep.sum()):,}")
        if a.dry_run:
            continue
        masks: dict[str, list[int]] = defaultdict(list)
        for (p, i), k in zip(owner, keep):
            if k:
                masks[p].append(i)
        sizes = defaultdict(int)
        for q, _ in owner:
            sizes[q] += 1
        ops = []
        work = Path(tempfile.mkdtemp())
        for p in ps:
            idx = masks.get(p, [])
            if len(idx) == sizes[p]:
                continue                                             # nothing removed from this file
            if not idx:
                ops.append(CommitOperationDelete(p))
                continue
            local = hf_hub_download(REPO, p, repo_type="dataset", token=token)
            tb = pq.read_table(local).take(pa.array(idx))
            tmp = work / f"{Path(p).parent.name}_{Path(p).name}"
            pq.write_table(tb, tmp, compression="zstd")
            ops.append(CommitOperationAdd(p, str(tmp)))
            if len(ops) >= 8:
                api.create_commit(REPO, ops, repo_type="dataset", commit_message=f"{lang}: cap translation rows at {a.max:,} (best kept)")
                ops = []
        if ops:
            api.create_commit(REPO, ops, repo_type="dataset", commit_message=f"{lang}: cap translation rows at {a.max:,} (best kept)")
        import shutil
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
