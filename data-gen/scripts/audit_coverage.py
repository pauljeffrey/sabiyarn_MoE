#!/usr/bin/env python3
"""Verify that the preset volumes (configs/*.yaml) request EVERY domain, (domain, sub-topic) pair, genre,
SFT task type and DPO flaw type, in every language -- without writing a batch file or calling any API.

Exit status is non-zero if any language misses any value. Used by tests/test_full_preset_coverage.py and
runnable by hand:

    python scripts/audit_coverage.py                    # all three presets
    python scripts/audit_coverage.py --kinds sft --per-language 400
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.corpus_config import load_config
from generators.base import BatchRequestSpec  # noqa: F401  (import check)
from pipeline.build_corpus import GENERATOR_MODULES
from sampling.taxonomy import DOMAINS, GENRES, all_pairs


def vocabularies(kind: str) -> dict[str, set[str]]:
    vocab = {"domain": set(DOMAINS), "pair": {f"{d}::{s}" for d, s in all_pairs()}}
    if kind == "pretrain":
        vocab["genre"] = set(GENRES)
    else:
        from generators.sft_tasks import TASKS

        vocab["task"] = set(TASKS)
        if kind == "dpo":
            from generators.dpo import REJECTION_KEYS

            vocab["rejection_type"] = set(REJECTION_KEYS)
    return vocab


def audit(kind: str, per_language: int | None = None, languages: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """language -> attribute -> {"expected": n, "used": n, "missing": [...], "min": n, "max": n}."""
    cfg = load_config(kind)
    if per_language is not None:
        cfg = cfg.with_overrides(per_language=per_language, languages=languages)
    gen = importlib.import_module(GENERATOR_MODULES[kind])
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for spec in gen.iter_requests(cfg, languages):
        a = spec.context["attributes"]
        c = counts[spec.language]
        c["domain"][a["domain"]] += 1
        c["pair"][f"{a['domain']}::{a['subtopic']}"] += 1
        for name in ("genre", "task", "rejection_type"):
            if name in a:
                c[name][a[name]] += 1
    vocab = vocabularies(kind)
    out: dict[str, dict[str, dict]] = {}
    for lang, per_attr in counts.items():
        out[lang] = {}
        for name, expected in vocab.items():
            cnt = per_attr[name]
            used = set(cnt) & expected
            out[lang][name] = {
                "expected": len(expected), "used": len(used), "missing": sorted(expected - set(cnt)),
                "unknown": sorted(set(cnt) - expected),
                "min": min((cnt[v] for v in used), default=0), "max": max(cnt.values(), default=0),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", default="pretrain,sft,dpo")
    ap.add_argument("--per-language", type=int, default=None, help="override preset volume (default: the preset)")
    args = ap.parse_args()
    bad = 0
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        result = audit(kind, args.per_language)
        print(f"\n[{kind}] requested coverage per language (used/expected, min..max requests per value)")
        for lang, attrs in sorted(result.items()):
            cells = []
            for name, r in attrs.items():
                cells.append(f"{name} {r['used']}/{r['expected']} ({r['min']}..{r['max']})")
                if r["missing"] or r["unknown"]:
                    bad += 1
                    cells.append(f"  !! {name}: missing={r['missing'][:6]} unknown={r['unknown'][:6]}")
            print(f"  {lang}: " + "; ".join(cells))
    print("\nOK: every value is covered in every language" if not bad else f"\nFAILED: {bad} coverage gap(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
