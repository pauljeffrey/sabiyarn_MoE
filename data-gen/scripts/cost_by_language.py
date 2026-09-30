#!/usr/bin/env python3
"""Cost per language per phase, so a budget can be cut where it buys least. No API calls.

Everything here is MEASURED in this repo's runs rather than assumed:
  * tokens per word, per language, with the target tokenizer (assemble._TOKENS_PER_WORD)
  * yield per language, from pretraining runs under the four repetition gates
  * tokens per sample per phase, from the OpenRouter runs
The one estimate is how many WORDS the generator writes, which is ~350 regardless of the ask
(prompts._MAX_PRETRAIN_WORDS records the measurement).

    python scripts/cost_by_language.py
    python scripts/cost_by_language.py --price 0.08 0.45     # in/out USD per 1M for another model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "seeds"))
# Seed.load() takes a KIND and resolves it relative to the working directory, so "rl" run from the repo root
# finds the rl/ package directory instead of data-gen/seeds/rl.json. Load by explicit path instead, so this
# script works from anywhere.
_SEED_PATH = {k: str(HERE / "seeds" / f"{k}.json") for k in ("pretrain", "sft", "rl")}

from assemble import _TOKENS_PER_WORD                       # noqa: E402
from schemas.seed import Seed                               # noqa: E402

# MEASURED yield, per language, under all four repetition gates. Where a language has not been measured
# directly it takes its tier's figure, and that is flagged in the output.
YIELD = {
    "pcm": 0.96, "yor": 0.96, "hau": 0.96, "ibo": 0.96, "eng": 0.96,   # 23/24 on the easy set
    "swh": 0.76, "fra": 0.76,                                          # 73/96 across the added languages
    "twi": 0.55, "aka": 0.55,                                          # weaker than the easy set in the live run
    "orm": 0.30,                                                       # 18/60, measured directly
    "efi": 0.44, "urh": 0.44, "fon": 0.44, "ewe": 0.44, "ful": 0.44, "fuv": 0.44,   # 16/36 on the low set
    "zul": 0.44, "som": 0.44, "kin": 0.44, "sna": 0.44,                # tier default, NOT measured
}
MEASURED_DIRECTLY = {"pcm", "yor", "hau", "ibo", "swh", "fra", "orm", "efi", "urh", "fon", "ewe", "ful", "fuv"}

# Tokens the generator spends per SAMPLE, by phase. Input is the shared brief amortised over a pack of 6.
PHASE = {"pretrain": dict(inp=668, words=350),
         "sft":      dict(inp=3733, words=0, out=2539),
         "rl":       dict(inp=3088, words=0, out=3430)}
LOW_TIER_WORD_SCALE = 0.55


def out_tokens(kind: str, lang: str, tier: str) -> float:
    p = PHASE[kind]
    if p["words"] == 0:
        return p["out"]
    words = p["words"] * (LOW_TIER_WORD_SCALE if tier == "low" else 1.0)
    return words * _TOKENS_PER_WORD.get(lang, 2.5) + 120        # + title and JSON scaffolding


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--price", nargs=2, type=float, default=[0.08, 0.45], metavar=("IN", "OUT"),
                    help="USD per 1M tokens; default is google/gemma-3-27b-it")
    a = ap.parse_args()
    pin, pout = a.price
    S = {k: Seed.load(_SEED_PATH[k]) for k in ("pretrain", "sft", "rl")}
    tiers = {l.code: l.tier for l in S["sft"].languages}
    codes = sorted({l.code for s in S.values() for l in s.languages})

    print(f"\ncost at ${pin}/${pout} per 1M tokens, ALL FOUR repetition gates applied\n")
    print(f"{'lang':5s} {'tier':7s} {'t/word':>7s} {'yield':>6s} "
          f"{'pretrain':>10s} {'sft':>9s} {'rl':>9s} {'TOTAL':>10s}  {'$/kept':>8s}")
    rows, tot = [], {"pretrain": 0.0, "sft": 0.0, "rl": 0.0}
    for c in codes:
        tier = tiers.get(c, "medium")
        y = YIELD.get(c, 0.5)
        cost = {}
        for k, s in S.items():
            n = next((l.samples for l in s.languages if l.code == c), 0)
            if not n:
                cost[k] = 0.0
                continue
            per = PHASE[k]["inp"] / 1e6 * pin + out_tokens(k, c, tier) / 1e6 * pout
            cost[k] = n * per / y                      # requests wasted on rejects are paid for too
            tot[k] += cost[k]
        total = sum(cost.values())
        n_all = sum(next((l.samples for l in s.languages if l.code == c), 0) for s in S.values())
        rows.append((total, c, tier, y, cost, n_all))
    for total, c, tier, y, cost, n_all in sorted(rows, reverse=True):
        star = "" if c in MEASURED_DIRECTLY else " *"
        print(f"{c:5s} {tier:7s} {_TOKENS_PER_WORD.get(c, 2.5):>7.2f} {y:>5.0%}{star:2s}"
              f"${cost['pretrain']:>9,.0f} ${cost['sft']:>8,.0f} ${cost['rl']:>8,.0f} ${total:>9,.0f}"
              f"  ${total / max(n_all, 1):>7.4f}")
    print(f"{'TOTAL':5s} {'':7s} {'':7s} {'':7s}"
          f"${tot['pretrain']:>9,.0f} ${tot['sft']:>8,.0f} ${tot['rl']:>8,.0f} "
          f"${sum(tot.values()):>9,.0f}")
    print("\n* yield is the tier default, not measured for that language.")
    print("Yields used:", ", ".join(f"{k} {v:.0%}" for k, v in sorted(YIELD.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
