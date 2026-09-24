#!/usr/bin/env python3
"""Cost estimate for the seed-driven pipeline. Makes NO API calls.

    python estimate_gen.py                     # all kinds, Together standard + batch
    python estimate_gen.py --kind sft --sample 200

Input tokens are MEASURED, not guessed: the real meta-prompt is built for a sample of plan rows and its
characters counted, then divided by CHARS_PER_TOKEN. The prompts are English, so ~4 chars/token is close;
the seed brief and the tool JSON dominate and they are identical every time, which is why this is tight.

Output tokens are the real unknown, so the estimate sweeps a RANGE rather than pretending to one number. A
pretrain document of 200-320 words is ~600-1400 tokens depending on the language (diacritic-heavy languages
fragment badly); a 6-10 message conversation with think blocks, tool calls and results is ~700-2500.

Prices below are Together AI's published rates. Batch is 50% off with a 24h window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate import plan_rows           # noqa: E402
from prompts import build_request        # noqa: E402
from schemas.seed import Seed            # noqa: E402

# Together AI, USD per 1M tokens. Verify at https://www.together.ai/pricing before a large run.
STANDARD = {"input": 0.15, "output": 0.60}
BATCH = {"input": 0.075, "output": 0.300}

CHARS_PER_TOKEN = 4.0          # English meta-prompts
OUTPUT_SWEEP = {
    "pretrain": [500, 800, 1200, 1600],
    "sft": [800, 1400, 2000, 2600],
    "rl": [1200, 1800, 2400, 3000],      # prefix + 3 candidate replies
    "judge": [150, 250, 350],            # a verdict is short
}


def measure_input_tokens(seed: Seed, sample: int = 120) -> float:
    """Mean input tokens of the real meta-prompt, over a spread of languages and tasks."""
    rows = list(plan_rows(seed))
    if not rows:
        return 0.0
    step = max(1, len(rows) // sample)
    picked = rows[::step][:sample]
    total = 0
    for row in picked:
        req = build_request(seed, row)
        chars = sum(len(m["content"]) for m in req.messages)
        # attached tool definitions are billed as input too
        if req.tools:
            chars += sum(len(str(t)) for t in req.tools)
        total += chars
    return total / len(picked) / CHARS_PER_TOKEN


def cost(n_requests: int, in_tok: float, out_tok: float, rates: dict) -> float:
    return (n_requests * in_tok / 1e6 * rates["input"]
            + n_requests * out_tok / 1e6 * rates["output"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="all", choices=["all", "pretrain", "sft", "rl"])
    ap.add_argument("--sample", type=int, default=120, help="plan rows to measure prompt size over")
    ap.add_argument("--judge-share", type=float, default=1.0,
                    help="fraction of RL samples that get a judge pass (default: all)")
    a = ap.parse_args()
    kinds = ["pretrain", "sft", "rl"] if a.kind == "all" else [a.kind]

    print("Together AI: standard $0.15/$0.60 per 1M in/out  |  batch $0.075/$0.300 (50% off, 24h window)")
    print(f"Input tokens measured from the real prompts (~{CHARS_PER_TOKEN:.0f} chars/token, English).\n")

    grand = {"standard": 0.0, "batch": 0.0}
    for kind in kinds:
        seed = Seed.load(kind)
        in_tok = measure_input_tokens(seed, a.sample)
        target, requests = seed.total_samples(), seed.total_requests()
        print(f"=== {kind.upper()}  target {target:,} samples -> {requests:,} requests "
              f"(yield budget {target/requests:.0%})")
        print(f"    measured input: {in_tok:,.0f} tokens/request "
              f"= {requests * in_tok / 1e6:,.1f}M input tokens total")
        print(f"    {'out tok/sample':>15}{'output M':>11}{'STANDARD':>12}{'BATCH':>12}")
        mid = None
        for out_tok in OUTPUT_SWEEP[kind]:
            s = cost(requests, in_tok, out_tok, STANDARD)
            b = cost(requests, in_tok, out_tok, BATCH)
            star = ""
            if out_tok == OUTPUT_SWEEP[kind][len(OUTPUT_SWEEP[kind]) // 2]:
                mid = (s, b)
                star = "  <- mid"
            print(f"    {out_tok:>15,}{requests*out_tok/1e6:>11,.0f}{'$' + format(s, ',.0f'):>12}"
                  f"{'$' + format(b, ',.0f'):>12}{star}")
        grand["standard"] += mid[0]
        grand["batch"] += mid[1]
        print()

    if "rl" in kinds and a.judge_share > 0:
        rl = Seed.load("rl")
        n = int(rl.total_samples() * a.judge_share)
        # A judge prompt carries the conversation prefix plus every candidate, so it is input-heavy.
        jin = 2200.0
        print(f"=== JUDGE PASS over {n:,} RL samples (input-heavy: ~{jin:,.0f} tokens/request)")
        print(f"    {'out tok/sample':>15}{'STANDARD':>12}{'BATCH':>12}")
        jmid = None
        for out_tok in OUTPUT_SWEEP["judge"]:
            s, b = cost(n, jin, out_tok, STANDARD), cost(n, jin, out_tok, BATCH)
            star = ""
            if out_tok == OUTPUT_SWEEP["judge"][1]:
                jmid, star = (s, b), "  <- mid"
            print(f"    {out_tok:>15,}{'$' + format(s, ',.0f'):>12}{'$' + format(b, ',.0f'):>12}{star}")
        grand["standard"] += jmid[0]
        grand["batch"] += jmid[1]
        print()

    print(f"GRAND TOTAL at the mid output estimate:  standard ${grand['standard']:,.0f}   "
          f"batch ${grand['batch']:,.0f}")
    print("\nNotes:")
    print("  * Failed/dropped generations are still billed -- that is already priced in, because the request")
    print("    count above is the yield-adjusted one, not the target.")
    print("  * Use --batch on Together for pretrain: it is the biggest block and the least urgent.")
    print("  * Run 200 samples per low-resource language first and read them. If Fon/Efik quality is poor,")
    print("    those volumes are wasted spend no matter how cheap the tokens are.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
