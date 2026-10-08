"""Build per-language stopword lists (function words that mark a language) from our own pretrain corpus.

    nice -n 15 python -m sft_new.build_stopwords            # -> sft_new/stopwords.json (committed, ~50 KB)

For each language: count words over ~15k documents of BeardedMonster/pretrain-new (read through the Hub filesystem; only the
first row groups of the first parquet file are touched, nothing is stored), then keep the words that are both frequent in that
language and specific to it (frequency per million words at least SPECIFIC x the mean of the other languages). Specific words
rather than raw top-N, because most frequent words ("a", "na", "ni", "ya") are shared by several of the 22 languages.
English has no shard in pretrain-new, so its list is written by hand.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LANGS = ["hau", "ibo", "yor", "pcm", "efi", "ibb", "urh", "twi", "ewe", "fon", "kin", "sna", "zul", "swh", "orm", "wol", "xho",
         "tsn", "nya", "afr", "som", "sot"]
ENGLISH = ("the of and to in is that it for was on are as with his they at be this from have or by one had not but what all were "
           "when we there can an your which their said if do will each about how up out them then she many some so these would "
           "other into has more her two like him see time could no make than first been its who now people my made over did down "
           "only way find use may water long little very after words called just where most know").split()
TOKEN = re.compile(r"[^\W\d_]+", re.U)
DOCS, SPECIFIC, KEEP, MIN_COUNT = 15_000, 6.0, 90, 25


def words(text: str):
    return [w.lower() for w in TOKEN.findall(text) if 1 <= len(w) <= 14]


def sample_counts(lang: str, fs, files: list[str], n_docs: int) -> tuple[collections.Counter, int]:
    import pyarrow.parquet as pq
    c, total, seen = collections.Counter(), 0, 0
    for f in files:
        with fs.open(f"datasets/BeardedMonster/pretrain-new/{f}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                    ws = words(t[:6000])
                    c.update(ws)
                    total += len(ws)
                    seen += 1
                    if seen >= n_docs:
                        return c, total
    return c, total


def main():
    from africomet_translation.build import hf_token
    from huggingface_hub import HfApi, HfFileSystem
    tok = hf_token()
    fs, api = HfFileSystem(token=tok), HfApi(token=tok)
    allf = sorted(s.rfilename for s in api.dataset_info("BeardedMonster/pretrain-new").siblings if s.rfilename.endswith(".parquet"))
    counts, totals = {}, {}
    for lang in LANGS:
        files = [f for f in allf if f.startswith(f"data/{lang}/")][:2]
        if not files:
            print("no shard for", lang, file=sys.stderr)
            continue
        counts[lang], totals[lang] = sample_counts(lang, fs, files, DOCS)
        print(lang, totals[lang], "words", file=sys.stderr, flush=True)
    out = {"eng": ENGLISH}
    for lang, c in counts.items():
        others = [l for l in counts if l != lang]
        scored = []
        for w, n in c.items():
            if n < MIN_COUNT:
                continue
            f = n / totals[lang] * 1e6
            other = sum(counts[o].get(w, 0) / totals[o] * 1e6 for o in others) / len(others) + 1.0
            if f / other >= SPECIFIC:
                scored.append((n, w))
        out[lang] = [w for _, w in sorted(scored, reverse=True)[:KEEP]]
        print(lang, len(out[lang]), out[lang][:12], file=sys.stderr, flush=True)
    (HERE / "stopwords.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
