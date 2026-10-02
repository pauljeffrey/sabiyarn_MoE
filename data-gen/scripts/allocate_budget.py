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
PROMPT_TOKENS = 8_425
# Document-backed tasks additionally carry the source IN the prompt, at the band longdocs.py targets.
DOCUMENT_TOKENS = 4_000  # already inside the measured PROMPT_TOKENS above; kept for the sft/pretrain paths
DOCUMENT_TASKS = {"rag_document_qa", "long_document_summarization", "retrieval_insufficient"}

# Completion is the whole generated payload: for RL BOTH candidates, chosen and rejected. In gemma tokens,
# per language -- this is where the language actually shows up in the bill.
COMPLETION_TOKENS_ENG = {"rl": 1_012, "sft": 820, "pretrain": 600}  # rl measured, per request
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
# MEASURED, not assumed, and re-measured after the generation contract was fixed: 54 kept of 78 requests
# (69%) across pcm/eng/twi/fon, against 44% before and 7% on the first run. The jump came from telling the
# generator to emit candidate FIELDS rather than a pre-assembled marker string, and from asking for exactly
# two labelled candidates instead of "N alternatives" with a three-way label scheme.
#
# These are scaled from that 69% mix rather than read off per language, because one 80-request run does not
# split four ways with any precision. They stay deliberately below the scaled figures.
#
# The earlier 82/62/44 came from the PRETRAIN path, where one document is generated and checked; an RL sample
# is a prefix plus a ranked pair and every part can fail independently.
_YIELD = {"high": 0.80, "medium": 0.47, "low": 0.38}
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
             tk: dict[str, float], floor: int = 0) -> dict[str, int]:
    """Proportional allocation under a per-language CAP and FLOOR.

    The floor is what makes the result usable rather than merely proportional. Volumes are population-scaled,
    so strict proportionality gives Fon a few hundred preference pairs -- too few to train on, which means the
    money spent on them buys nothing at all. A floor is paid FIRST, off the top, and the remainder is shared
    out proportionally; because paying it can push another language below the floor, and releasing capped
    budget can push one above the cap, both run to a fixed point instead of being applied once.

    Returns {} if the floor cannot be paid for every language -- better to say so than to return an
    allocation that silently funds some languages below a usable size.
    """
    costs = {l: cost_per_sample(l, kind, tk) for l in weights}
    fixed: dict[str, int] = {}
    live = dict(weights)
    spare = budget

    if floor:
        need = sum(floor * costs[l] for l in live)
        if need > budget:
            return {}

    for _ in range(len(weights) + 2):  # bounded: each pass fixes at least one language
        if not live:
            break
        total_w = sum(live.values())
        want = {l: int(spare * (w / total_w) / costs[l]) for l, w in live.items()}
        over = [l for l, n in want.items() if n > cap]
        under = [l for l, n in want.items() if floor and n < floor]
        if not over and not under:
            return {**fixed, **want}
        # Pin the languages that hit a bound, bank what they cost, and reshare the rest.
        for l in over:
            fixed[l] = cap
            spare -= cap * costs[l]
            del live[l]
        for l in under:
            fixed[l] = floor
            spare -= floor * costs[l]
            del live[l]
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="rl", choices=["rl", "sft", "pretrain"])
    ap.add_argument("--budget", type=float, default=50.0, help="dollars")
    ap.add_argument("--cap", type=int, default=12_000, help="max samples for any one language")
    ap.add_argument("--floor", type=int, default=0,
                    help="min samples for any one language, paid before the proportional share. Below a few "
                         "hundred preference pairs a language is not trainable, so the spend buys nothing.")
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
    alloc = allocate(volumes, a.budget, a.cap, a.kind, tk, a.floor)
    if not alloc:
        need = sum(a.floor * cost_per_sample(l, a.kind, tk) for l in volumes)
        print(f"a floor of {a.floor:,} for {len(volumes)} languages needs ${need:.2f}, "
              f"and the budget is ${a.budget:.2f}", file=sys.stderr)
        return 1
    spent = sum(n * cost_per_sample(l, a.kind, tk) for l, n in alloc.items())

    print(f"{a.kind}  budget ${a.budget:.2f}  cap {a.cap:,}"
          f"{f'  floor {a.floor:,}' if a.floor else ''}  languages {len(alloc)}")
    print(f"average request: {tk['prompt']:,.0f} prompt tokens + {tk['completion']:,.0f} completion tokens "
          f"(English baseline); {tk['doc_share']:.0%} of the mix carries a source document")
    print(f"{'lang':<6}{'tier':<8}{'samples':>9}{'$/1k':>8}{'cost':>9}{'share':>8}")
    for l, n in sorted(alloc.items(), key=lambda kv: -kv[1]):
        c = n * cost_per_sample(l, a.kind, tk)
        flag = "  (capped)" if n >= a.cap else ("  (floor)" if a.floor and n <= a.floor else "")
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
