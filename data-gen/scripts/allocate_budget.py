#!/usr/bin/env python3
"""Spend a fixed budget across languages, proportionate to the seed volumes, under a per-language cap.

The cap is the interesting part. Volumes are population-scaled, so without one Pidgin alone eats a third of
the budget and the languages that most need data get a few hundred samples. With one, the capped languages
release budget that is redistributed over the uncapped -- and because that can push a second language into
the cap, the redistribution iterates to a fixed point rather than being applied once.

    python scripts/allocate_budget.py --kind rl --budget 50 --cap 12000
    python scripts/allocate_budget.py --kind sft --budget 50 --langs fon,ewe,urh,efi,twi,aka

Cost is measured in tokens, not samples: a Fon sample costs 2.6 tokens per word against Pidgin's 1.24, so
a flat per-sample price would overstate what the cheap languages cost and understate the dear ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "seeds"))

from assemble import _TOKENS_PER_WORD  # noqa: E402
from schemas.seed import Seed  # noqa: E402

# Doubleword batch rates, dollars per million tokens, from providers/doubleword.py.
PRICE_IN, PRICE_OUT = 0.06, 0.18

# ---------------------------------------------------------------------------- the cost model
# Costed in TOKENS AS THE PROVIDER BILLS THEM, not in words. Two reasons, both learned the hard way here:
#
#  1. assemble._TOKENS_PER_WORD is SabiYarn's tokenizer, which is tuned for these languages. The provider
#     bills on gemma's, which is not: the same Fon sentence costs roughly twice as many gemma tokens. Pricing
#     a gemma API call with SabiYarn's ratio understates the bill, and understates it worst exactly for the
#     low-resource languages the budget is meant to protect.
#  2. The prompt is far larger than the payload suggests. Measured at 14.5k characters -- about 4.7k gemma
#     tokens -- because every request carries the seed instruction, the task description, the tool schemas
#     and the field contract. A words-of-content estimate missed this by a factor of four.
#
# PROMPT_TOKENS is measured (scripts/allocate_budget.py --measure-prompt re-derives it from the live seed).
# Prompt scaffolding is English and essentially identical across languages, so it does not scale by language.
# MEASURED on a live Doubleword run (80 RL requests, pcm/eng/twi/fon, gemma-4-31B-it): 703,498 prompt tokens
# and 95,269 completion tokens over 78 requests. The estimate this replaced was 4,700, and it was wrong
# because it counted only the visible scaffolding and not the tool catalogue, the few-shot exemplar or the
# spliced source document.
PROMPT_TOKENS = 9_000
# Document-backed tasks additionally carry the source IN the prompt, at the band longdocs.py targets.
DOCUMENT_TOKENS = 4_000  # already inside the measured PROMPT_TOKENS above; kept for the sft/pretrain paths
DOCUMENT_TASKS = {"rag_document_qa", "long_document_summarization", "retrieval_insufficient"}

# Completion is the whole generated payload: for RL BOTH candidates, chosen and rejected. In gemma tokens,
# per language -- this is where the language actually shows up in the bill.
COMPLETION_TOKENS_ENG = {"rl": 1_220, "sft": 820, "pretrain": 600}  # rl measured, per request
# Gemma tokens per SabiYarn token, i.e. how much worse the provider's tokenizer is on each language.
# Approximated from the ratio of gemma's byte-level fallback on unseen orthographies; calibrated by --probe.
_GEMMA_PENALTY = {
    "eng": 1.0, "pcm": 1.15, "yor": 1.9, "hau": 1.7, "ibo": 1.9, "fra": 1.1,
    "twi": 2.1, "aka": 2.1, "efi": 2.2, "urh": 2.2, "ewe": 2.4, "fon": 2.5,
    "ful": 2.2, "fuv": 2.2, "som": 2.0, "swh": 1.6, "orm": 2.3, "kin": 2.3,
    "sna": 2.2, "zul": 2.3,
}

# Yield: the share of requests that survive the four repetition gates and the assembler. Measured, and much
# worse for the low-resource languages -- so a Fon sample is dear twice over, in tokens AND in retries.
# MEASURED, not assumed. On the same run, kept-by-language over ~20 requests each: pcm 13, eng 10, twi 6,
# fon 5 -- so roughly 55% high, 30% medium, 25% low. The earlier 82/62/44 came from the PRETRAIN path, where
# one document is generated and checked; an RL sample is a prefix plus a ranked pair and every part of it can
# fail independently, so its yield is structurally lower. Using pretrain's numbers understated RL cost ~2x.
_YIELD = {"high": 0.55, "medium": 0.30, "low": 0.25}
_TIER = {
    "pcm": "high", "eng": "high", "yor": "high", "hau": "high", "ibo": "high",
    "twi": "medium", "aka": "medium", "swh": "medium", "fra": "medium",
    "fon": "low", "ewe": "low", "urh": "low", "efi": "low", "ful": "low", "fuv": "low",
    "orm": "low", "som": "low", "kin": "low", "sna": "low", "zul": "low",
}


def tokens_for(kind: str, tasks) -> dict[str, float]:
    """Prompt and (English-baseline) completion tokens for an average sample, weighted by the task mix."""
    total = sum(t.share for t in tasks) or 1.0
    doc_share = sum(t.share for t in tasks if t.name in DOCUMENT_TASKS) / total
    # For rl the measured figure already includes the spliced documents, so adding them again double-counts.
    extra = 0.0 if kind == "rl" else DOCUMENT_TOKENS * doc_share
    return {"prompt": PROMPT_TOKENS + extra,
            "completion": COMPLETION_TOKENS_ENG[kind], "doc_share": doc_share}


def cost_per_sample(lang: str, kind: str, tk: dict[str, float]) -> float:
    """Dollars for one KEPT sample, including the retries it took to get it.

    Retries are charged in full -- a dropped sample costs exactly what a kept one does -- so at 44% yield the
    low tier pays 2.3x over the sticker price. That division is most of why Fon costs four times Pidgin."""
    pen = _GEMMA_PENALTY.get(lang, 2.2)
    # The prompt is English scaffolding, so it does not take the penalty; the completion is in-language.
    per_request = (tk["prompt"] * PRICE_IN + tk["completion"] * pen * PRICE_OUT) / 1e6
    return per_request / _YIELD[_TIER.get(lang, "low")]


def allocate(weights: dict[str, int], budget: float, cap: int, kind: str,
             tk: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation proportional to `weights`, capped, with the freed budget redistributed
    until no language is both uncapped and over the cap."""
    costs = {l: cost_per_sample(l, kind, tk) for l in weights}
    capped: dict[str, int] = {}
    live = dict(weights)
    spare = budget

    while live:
        total_w = sum(live.values())
        # Spend `spare` across `live` in proportion to weight, converting dollars to samples per language.
        want = {}
        for l, w in live.items():
            dollars = spare * (w / total_w)
            want[l] = int(dollars / costs[l])
        over = {l: n for l, n in want.items() if n > cap}
        if not over:
            return {**capped, **want}
        # Cap the worst offenders, bank what they cost, and redistribute the rest.
        for l in over:
            capped[l] = cap
            spare -= cap * costs[l]
            del live[l]
    return capped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="rl", choices=["rl", "sft", "pretrain"])
    ap.add_argument("--budget", type=float, default=50.0, help="dollars")
    ap.add_argument("--cap", type=int, default=12_000, help="max samples for any one language")
    ap.add_argument("--langs", default="", help="comma-separated; default is every language in the seed")
    ap.add_argument("--json", default="", help="write the allocation here")
    a = ap.parse_args()

    seed = Seed.load(str(ROOT / "seeds" / f"{a.kind}.json"))
    volumes = {l.code: l.samples for l in seed.languages}
    if a.langs:
        want = [s.strip() for s in a.langs.split(",") if s.strip()]
        missing = [l for l in want if l not in volumes]
        if missing:
            print(f"not in the {a.kind} seed: {missing}", file=sys.stderr)
            return 1
        volumes = {l: volumes[l] for l in want}

    tk = tokens_for(a.kind, seed.tasks)
    alloc = allocate(volumes, a.budget, a.cap, a.kind, tk)
    spent = sum(n * cost_per_sample(l, a.kind, tk) for l, n in alloc.items())

    print(f"{a.kind}  budget ${a.budget:.2f}  cap {a.cap:,}  languages {len(alloc)}")
    print(f"average request: {tk['prompt']:,.0f} prompt tokens + {tk['completion']:,.0f} completion tokens "
          f"(English baseline); {tk['doc_share']:.0%} of the mix carries a source document")
    print(f"{'lang':<6}{'tier':<8}{'samples':>9}{'$/1k':>8}{'cost':>9}{'share':>8}")
    for l, n in sorted(alloc.items(), key=lambda kv: -kv[1]):
        c = n * cost_per_sample(l, a.kind, tk)
        flag = "  (capped)" if n >= a.cap else ""
        print(f"{l:<6}{_TIER.get(l, 'low'):<8}{n:>9,}{cost_per_sample(l, a.kind, tk)*1000:>8.2f}"
              f"{c:>9.2f}{c/spent:>7.0%}{flag}")
    print(f"{'total':<14}{sum(alloc.values()):>9,}{'':>8}{spent:>9.2f}")

    if a.json:
        Path(a.json).write_text(json.dumps({"kind": a.kind, "budget": a.budget, "cap": a.cap,
                                            "alloc": alloc, "cost": round(spent, 2)}, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
