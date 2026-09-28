#!/usr/bin/env python3
"""What a GPU rental buys, per box and per phase. No GPU required.

    python scripts/rent_plan.py                      # every box, both contexts, $30
    python scripts/rent_plan.py --budget 30 --kind sft

WHAT IS MEASURED AND WHAT IS ESTIMATED
--------------------------------------
MEASURED, from the 4,736-sample OpenRouter run with this exact pipeline and gemma-4-31b-it:
  * tokens per kept sample (input and output), per phase;
  * yield -- kept samples per request;
  * the document stages' share of the work.
Those are the numbers that decide how much work a sample IS, and they do not depend on the GPU.

ESTIMATED, from published hardware specs:
  * decode throughput, from memory bandwidth and the fact that a decode step reads every weight once and
    emits `batch` tokens. This is the standard roofline for batched autoregressive decode and it is usually
    within ~25% for a dense model, but it is an estimate. `vllm_gen.py` prints the real figure within the
    first chunk, and that figure should replace this table.

The KV-cache geometry is NOT estimated: it comes from gemma-4-31b's own config (50 sliding-window layers of
1024 + 10 full-attention layers, K and V shared), which is why long-context concurrency here is far better
than a naive all-global calculation suggests.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from budgets import budget_for                                    # noqa: E402
from vllm_gen import _weights_gb, max_sequences, preset_for  # noqa: E402

MODEL = "google/gemma-4-31b-it"


@dataclass
class Box:
    """A rentable box. `vram_gb` is what the GPU has, which is NOT always the number in the listing title."""

    name: str
    vram_gb: float
    usd_per_hour: float
    bandwidth_gbs: float          # published memory bandwidth
    capability: float             # CUDA compute capability: fp8 needs >= 8.9
    caveat: str = ""


# Published specifications. Prices are the owner's quoted vast.ai rates (2026-09-28); both move, so re-check.
BOXES = [
    Box("RTX 5090 32GB", 32, 0.513, 1792, 12.0),
    Box("48GB Ada (L40S/A6000)", 48, 0.740, 864, 8.9),
    Box("A100 40GB", 40, 0.433, 1555, 8.0),
    Box("GB10 119GB", 119, 0.450, 273, 12.0,
        "unified LPDDR5X: huge but ~7x slower than A100 HBM, so it is memory-capacity cheap and "
        "bandwidth poor"),
    Box("CMP 170HX", 8, 0.605, 1493, 8.0,
        "the '64GB' in the listing is HOST RAM. The card has 8 GB of VRAM, and it is a mining SKU with "
        "crippled FP16 throughput and a x4 PCIe link. gemma-4-31b needs ~17 GB even at 4-bit, so this box "
        "cannot run it at all"),
    Box("A800 80GB", 80, 0.885, 2039, 8.0, "Ampere: no fp8 kernels"),
    Box("A100 80GB", 80, 1.115, 2039, 8.0, "Ampere: no fp8 kernels"),
]

# MEASURED per KEPT sample on the OpenRouter run (tranche 1: 1,057 requests -> 8,480,610 in + 5,769,238 out,
# 2,272 kept). Input is dominated by the shared brief, which prefix caching makes nearly free on the second and
# later rows of a group -- so `prefill_new` is the part actually computed per sample.
MEASURED = {
    #          output   prompt   new prompt tokens after prefix reuse   yield
    "pretrain": dict(out=1_100, prompt=1_106, prefill_new=254, yield_rate=0.90),
    "sft":      dict(out=2_539, prompt=3_428, prefill_new=1_817, yield_rate=0.63),
    "rl":       dict(out=2_800, prompt=3_088, prefill_new=1_451, yield_rate=0.60),
}
# Extra output tokens per sample spent on stage 0 + stage 1 documents, amortised over ALL samples of the phase.
# long_document_summarization is 2.8% of SFT exchanges and the two RAG tasks another 13.6%; at the 32k band a
# summarisation document averages ~9k output tokens and a RAG context ~5k.
DOC_OVERHEAD = {"pretrain": 0, "sft": int(0.028 * 9_000 + 0.136 * 5_000), "rl": int(0.14 * 5_000)}


# vLLM does not reach peak bandwidth: kernel launch, attention over the KV cache and sampling all cost. 0.75
# is a conservative, commonly observed fraction for a dense model under continuous batching.
_BW_EFFICIENCY = 0.75
# bitsandbytes is NOT a fast 4-bit path in vLLM. It dequantizes to bf16 inside the matmul with general-purpose
# kernels, so most of the bandwidth saving is spent again on dequantization -- it is what makes a small card
# work at all, not what makes it quick. A PRE-QUANTIZED AWQ or GPTQ checkpoint has fused kernels and does keep
# the saving; if one exists for this model, point --model at it and set --quantization auto.
_QUANT_KERNEL_EFFICIENCY = {None: 1.0, "fp8": 0.95, "bitsandbytes": 0.45, "awq": 0.9, "gptq": 0.9}


def decode_tokens_per_s(box: Box, quant: str | None, concurrency: int) -> float:
    """Roofline for batched decode: one step reads every weight once and emits `concurrency` tokens.

    Bandwidth is the binding constraint for a dense model at these batch sizes, which is why quantizing helps
    throughput and not only fit -- subject to the kernel penalty above.
    """
    weight_bytes = _weights_gb(MODEL, quant) * 2**30
    if weight_bytes <= 0:
        return 0.0
    step_s = weight_bytes / (box.bandwidth_gbs * 1e9)
    return concurrency / step_s * _BW_EFFICIENCY * _QUANT_KERNEL_EFFICIENCY.get(quant, 0.8)


def plan(box: Box, kind: str, context: int, budget_usd: float, mode: str = "auto") -> dict:
    """`mode` mirrors --quantization: `auto` takes the highest precision that fits (quality first),
    `throughput` the highest that also leaves room to batch."""
    gpus = [{"index": 0, "name": box.name, "total_gb": box.vram_gb, "capability": box.capability}]
    options = [q for q in ([None] + (["fp8"] if box.capability >= 8.9 else []) + ["bitsandbytes"])
               if _weights_gb(MODEL, q) * 1.15 <= box.vram_gb]
    if not options:
        return {"box": box, "fits": False}
    quant = options[0]
    if mode == "throughput":
        best, best_tps = quant, -1.0
        for q in options:
            s = max(1, max_sequences(MODEL, gpus, 1, q, 0.0, context))
            tps = decode_tokens_per_s(box, q, s)
            if tps > best_tps:
                best, best_tps = q, tps
        quant = best
    seqs = max(1, max_sequences(MODEL, gpus, 1, quant, 0.0, context))
    m = MEASURED[kind]
    out_per_sample = m["out"] + DOC_OVERHEAD[kind]
    tps = decode_tokens_per_s(box, quant, seqs)
    if tps <= 0:
        return {"box": box, "fits": False}
    # Prefill is compute-bound and small next to decode here once the prefix is cached; charge it at the same
    # roofline to stay conservative rather than ignore it.
    work_tokens = out_per_sample + m["prefill_new"] * 0.25
    samples_per_hour = tps * 3600 / work_tokens
    hours = budget_usd / box.usd_per_hour
    return {"box": box, "fits": True, "quant": quant or "bf16", "seqs": seqs, "tok_s": tps,
            "samples": int(samples_per_hour * hours), "usd_per_1k": box.usd_per_hour / max(samples_per_hour, 1e-9) * 1000,
            "hours": hours}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--kind", default="all", choices=["all", "pretrain", "sft", "rl"])
    ap.add_argument("--contexts", default="16384,32768")
    ap.add_argument("--modes", default="auto,throughput",
                    help="which --quantization policies to compare")
    a = ap.parse_args()
    kinds = ["pretrain", "sft", "rl"] if a.kind == "all" else [a.kind]
    contexts = [int(c) for c in a.contexts.split(",")]

    print(f"\nmodel {MODEL}: {preset_for(MODEL)['weight_gb_bf16']:.0f} GiB bf16 / "
          f"{_weights_gb(MODEL, 'fp8'):.0f} GiB fp8 / {_weights_gb(MODEL, 'bitsandbytes'):.0f} GiB 4-bit")
    for box in BOXES:
        if box.caveat:
            print(f"  NOTE {box.name}: {box.caveat}")

    for kind in kinds:
        for context in contexts:
            b = budget_for(kind, context)
            print(f"\n=== {kind.upper()} at context {context:,}  (budget ${a.budget:,.0f}) "
                  f"-- response<={b.max_response_tokens:,}, documents {b.doc_token_range[0]:,}-"
                  f"{b.doc_token_range[1]:,}")
            print(f"  {'box':24s} {'quant':13s} {'seqs':>5s} {'out tok/s':>10s} {'hours':>6s} "
                  f"{'kept':>9s} {'$/1k kept':>10s}   mode")
            allrows = []
            for mode in a.modes.split(","):
                rows = [plan(box, kind, context, a.budget, mode) for box in BOXES]
                allrows += [(mode, r) for r in rows if r["fits"]]
                for r in rows:
                    if not r["fits"]:
                        if mode == a.modes.split(",")[0]:
                            print(f"  {r['box'].name:24s} {'DOES NOT FIT':>13s}")
                        continue
                    print(f"  {r['box'].name:24s} {r['quant']:13s} {r['seqs']:>5d} {r['tok_s']:>10,.0f} "
                          f"{r['hours']:>6.1f} {r['samples']:>9,d} {r['usd_per_1k']:>10.2f}"
                          f"   {mode}")
            if allrows:
                mode, best = max(allrows, key=lambda mr: mr[1]["samples"])
                print(f"  -> most samples per ${a.budget:,.0f}: {best['box'].name} "
                      f"({best['quant']}, {mode}) {best['samples']:,} kept at "
                      f"${best['usd_per_1k']:.2f}/1k")
    print("\nThroughput is a published-spec roofline, not a measurement. vllm_gen.py prints the real "
          "out-tok/s and $/kept sample\nwithin its first chunk; trust that over this table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
