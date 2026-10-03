#!/usr/bin/env python3
"""Harvest a pool of real technical documents from Wikipedia, for the RAG and summarisation tasks.

    python scripts/build_document_pool.py --want 3000
    python scripts/build_document_pool.py --want 500 --domains health_medicine,agriculture_crops
    python scripts/build_document_pool.py --report

The pool is a jsonl file at data/document_pool.jsonl, appended to and deduplicated by page id, so this can
be run repeatedly and interrupted safely.

Length is filtered on the search result's `wordcount` before anything is downloaded, so the request count
is roughly (documents wanted) rather than (pages inspected). Selection is biased towards specific,
checkable, numeric content -- see doc_sources.technicality for why that is the criterion that makes the
tool-using behaviour learnable rather than arbitrary.
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from budgets import budget_for  # noqa: E402
from doc_sources import MIN_TECHNICALITY, TOPIC_SEEDS, DocumentPool, harvest  # noqa: E402

DEFAULT_POOL = ROOT / "data" / "document_pool.jsonl"


def report(pool: DocumentPool) -> None:
    if not len(pool):
        print(f"{pool.path} is empty")
        return
    words = [int(d.get("words") or 0) for d in pool.docs]
    tech = [float(d.get("technicality") or 0) for d in pool.docs]
    by_dom = collections.Counter(d.get("domain") for d in pool.docs)
    print(f"{len(pool):,} documents in {pool.path}")
    print(f"  words      : median {statistics.median(words):,.0f}  "
          f"min {min(words):,}  max {max(words):,}")
    print(f"  technicality: median {statistics.median(tech):.2f}  min {min(tech):.2f}  max {max(tech):.2f}")
    print(f"  licence    : {pool.docs[0].get('licence')} -- attribution required if the corpus is published")
    print("  domains    :")
    for dom, n in by_dom.most_common():
        print(f"     {dom:<28} {n:>5}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--want", type=int, default=0, help="stop after this many NEW documents")
    ap.add_argument("--domains", default="", help="comma list from doc_sources.TOPIC_SEEDS")
    ap.add_argument("--per-seed", type=int, default=40, help="search results to consider per query")
    ap.add_argument("--min-technicality", type=float, default=MIN_TECHNICALITY)
    ap.add_argument("--kind", default="rl", choices=["rl", "sft"],
                    help="whose document band to target")
    ap.add_argument("--report", action="store_true", help="describe the existing pool and exit")
    a = ap.parse_args()

    pool = DocumentPool(Path(a.pool))
    if a.report:
        report(pool)
        return 0

    # The band comes from budgets.py, so the pool matches what the samples will ask for rather than a
    # number chosen here.
    b = budget_for(a.kind)
    lo_tok, hi_tok = b.rag_context_tokens
    # English averages ~1.28 tokens per word against this tokenizer (assemble._TOKENS_PER_WORD).
    min_words, max_words = int(lo_tok / 1.28), int(hi_tok / 1.28)
    domains = [s.strip() for s in a.domains.split(",") if s.strip()] or None
    if domains:
        unknown = [d for d in domains if d not in TOPIC_SEEDS]
        if unknown:
            print(f"unknown domain(s): {unknown}\nknown: {sorted(TOPIC_SEEDS)}", file=sys.stderr)
            return 1

    print(f"harvesting into {pool.path}  (have {len(pool):,})")
    print(f"  band: {min_words:,}-{max_words:,} words  ({lo_tok:,}-{hi_tok:,} tokens, {a.kind})")
    print(f"  technicality >= {a.min_technicality}")
    added = pool.extend(harvest(min_words=min_words, max_words=max_words,
                                min_technicality=a.min_technicality, per_seed=a.per_seed,
                                domains=domains, want=a.want))
    print(f"\nadded {added:,}; pool now {len(pool):,}")
    report(pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
