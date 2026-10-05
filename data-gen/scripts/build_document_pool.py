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
from doc_sources import (  # noqa: E402
    MIN_TECHNICALITY, NATIVE_RAG_MAX_WORDS, NATIVE_RAG_MIN_WORDS, NATIVE_WIKIS, NEWS_CONFIGS,
    TOPIC_SEEDS, DocumentPool, harvest, harvest_native, harvest_news,
)

DEFAULT_POOL = ROOT / "data" / "document_pool.jsonl"


def report(pool: DocumentPool) -> None:
    if not len(pool):
        print(f"{pool.path} is empty")
        return
    words = [int(d.get("words") or 0) for d in pool.docs]
    tech = [float(d.get("technicality") or 0) for d in pool.docs]
    by_dom = collections.Counter(d.get("domain") for d in pool.docs)
    by_lang = collections.Counter(d.get("doc_lang") or "eng" for d in pool.docs)
    print(f"{len(pool):,} documents in {pool.path}")
    print(f"  words      : median {statistics.median(words):,.0f}  "
          f"min {min(words):,}  max {max(words):,}")
    print(f"  technicality: median {statistics.median(tech):.2f}  min {min(tech):.2f}  max {max(tech):.2f}")
    lic = collections.Counter(d.get("licence") or "unknown" for d in pool.docs)
    print("  licences   : (all apply if the corpus is published)")
    for name, cnt in lic.most_common():
        print(f"     {cnt:>5}  {name}")
    print(f"  languages  : {dict(by_lang.most_common())}")
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
    ap.add_argument("--lang", default="eng",
                    help="document language. Only the languages in doc_sources.NATIVE_WIKIS can be "
                         "harvested; everything else falls through to generation, which is the only option "
                         "when the Wikipedia has nothing long enough to take.")
    ap.add_argument("--source", default="auto", choices=["auto", "wiki", "news"],
                    help="where native documents come from. 'auto' uses the Wikipedia where one is viable "
                         "and MasakhaNEWS otherwise; the wikis are thin for most of these languages.")
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

    if a.lang != "eng":
        src = a.source
        if src == "auto":
            src = "wiki" if a.lang in NATIVE_WIKIS else "news"
        if src == "wiki" and a.lang not in NATIVE_WIKIS:
            print(f"{a.lang} has no harvestable Wikipedia (measured: too few long articles). "
                  f"Harvestable: {sorted(NATIVE_WIKIS)}", file=sys.stderr)
            return 1
        if src == "news" and a.lang not in NEWS_CONFIGS:
            print(f"{a.lang} is not in MasakhaNEWS. Available: {sorted(NEWS_CONFIGS)}", file=sys.stderr)
            return 1
    print(f"harvesting into {pool.path}  (have {len(pool):,}; {pool.count(a.lang):,} in {a.lang})")
    if a.lang == "eng":
        print(f"  band: {min_words:,}-{max_words:,} words  ({lo_tok:,}-{hi_tok:,} tokens, {a.kind})")
    else:
        print(f"  band: {NATIVE_RAG_MIN_WORDS:,}-{NATIVE_RAG_MAX_WORDS:,} words  (native; a tool result is "
              f"a few passages, so shorter documents are usable)  source={src}")
    print(f"  technicality >= {a.min_technicality}")
    if a.lang == "eng":
        stream = harvest(min_words=min_words, max_words=max_words,
                         min_technicality=a.min_technicality, per_seed=a.per_seed,
                         domains=domains, want=a.want)
    else:
        # Native documents may be SHORTER: a tool result is a few passages now, so a 400-word article is a
        # legitimate source and insisting on 1,600+ would reject the whole available corpus.
        min_words, max_words = NATIVE_RAG_MIN_WORDS, NATIVE_RAG_MAX_WORDS
        if src == "wiki":
            # Selected on length and coherence rather than topic: see doc_sources.harvest_native.
            stream = harvest_native(a.lang, min_words=min_words, max_words=max_words, want=a.want)
        else:
            stream = harvest_news(a.lang, min_words=min_words, max_words=max_words, want=a.want)
    added = pool.extend(stream)
    print(f"\nadded {added:,}; pool now {len(pool):,}")
    report(pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
