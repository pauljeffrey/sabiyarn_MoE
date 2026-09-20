"""Back-of-envelope micro-batch size + wall-clock estimate for a token budget.

    python scripts/estimate_training_plan.py [--tokens 50e9] [--seq 4096]

Everything is an ESTIMATE built from the model config and the code in
sabiyarn/model/modeling.py + training/new_train.py; there is no GPU here to
measure. Edit the ASSUMPTIONS block, or (better) calibrate against a real
probe: the trainer's `step` log line prints tokens_per_sec and mfu, and
time = tokens / tokens_per_sec.
"""

from __future__ import annotations

import argparse
import math

# --------------------------------------------------------------------------
# Model (from config.json) -----------------------------------------------
D, L, H, VOCAB, MOE_DIM = 768, 12, 12, 52050, 3072
EXPERTS_PER_LAYER = [2, 3] + [4] * 10
WPE_POSITIONS = 32768  # config block_size; the wpe table is only ~25M params

attn_params = L * (D * 3 * D + D * D)
expert_params = sum(EXPERTS_PER_LAYER) * 2 * D * MOE_DIM
wte_params = VOCAB * D  # tied with lm_head
wpe_params = WPE_POSITIONS * D
N_PARAMS = attn_params + expert_params + wte_params + wpe_params

# --------------------------------------------------------------------------
# ASSUMPTIONS ---------------------------------------------------------------
# Per-token activation bytes kept for backward, bf16 autocast, no activation
# checkpointing (none is used in training/new_train.py). LayerNorm outputs are
# fp32 under autocast, hence the 4-byte terms.
ATTN_LN_BYTES_PER_TOKEN_PER_LAYER = 23_000  # ln_1, j, qkv, attn out, ln_2 (+ bf16 casts)
MOE_BYTES_PER_TOKEN_PER_EXPERT = 12_288  # pre-GELU + post-GELU hidden, E x 3072 x 2 B x 2
LOSS_TAIL_BYTES_PER_TOKEN = 520_000  # bf16 logits + fp32 logits + fp32 log-softmax (V=52050); 0 with CCE
# The doc-causal mask is a (B,1,T,T) bool expanded to (B,H,T,T); SDPA turns a
# bool mask into a bf16 additive bias, materialised per layer and kept for
# backward: 2 B * H * T^2 per sample per layer. 0 if attention had no dense mask.
DENSE_MASK_BIAS = True
USABLE_FRACTION = 0.92  # CUDA context + allocator fragmentation headroom

# name -> (usable GiB, bf16 dense peak TFLOPS, assumed MFU with this codebase)
# MFU is a guess: dense-expert MoE, mem-efficient SDPA with a dense mask, no
# torch.compile on multi-GPU (see compile_skipped), small d_model. GDDR cards are
# lower because the elementwise ops are bandwidth-bound.
GPUS = {
    "45GB  L40S (A40 ~2.4x slower)": (44.5, 362.0, 0.18),
    "48GB  RTX 6000 Ada (A6000 ~2.3x slower)": (47.5, 364.0, 0.18),
    "80GB  A100": (79.6, 312.0, 0.25),
    "80GB  H100 SXM": (79.6, 989.0, 0.25),
    "96GB  RTX PRO 6000 Blackwell": (95.6, 500.0, 0.18),
}
MFU_SPARSE_PENALTY = 0.85  # sorting/gather/index_add and smaller matmuls: assume 15% lower utilisation
FSDP_THROUGHPUT_FACTOR = 0.85  # per-microbatch all-gather/reduce-scatter; DDP syncs once per optimizer step
# --------------------------------------------------------------------------


TOP_K = 2  # num_experts_per_tok
SPARSE_GATHER_BYTES_PER_TOKEN_PER_LAYER = 3_000  # x[rows] copies + index buffers kept for backward


def active_experts(sparse: bool) -> list[int]:
    """Experts that actually run per token in each layer."""
    return [min(TOP_K, e) if sparse else e for e in EXPERTS_PER_LAYER]


def flops_per_token(seq: int, sparse: bool = False) -> float:
    fwd = (
        L * 8 * D * D  # qkv + out proj
        + L * 4 * D * seq  # attention scores
        + sum(4 * D * MOE_DIM * e for e in active_experts(sparse))  # dense: every expert; sparse: top-k
        + 2 * D * VOCAB  # lm_head
    )
    return 3.0 * fwd


def per_sample_gib(seq: int, cce: bool, dense_mask: bool, sparse: bool = False) -> float:
    avg_experts = sum(active_experts(sparse)) / L
    layer_bytes = ATTN_LN_BYTES_PER_TOKEN_PER_LAYER + avg_experts * MOE_BYTES_PER_TOKEN_PER_EXPERT
    if sparse:
        layer_bytes += SPARSE_GATHER_BYTES_PER_TOKEN_PER_LAYER
    per_token = L * layer_bytes + (0 if cce else LOSS_TAIL_BYTES_PER_TOKEN)
    total = per_token * seq
    if dense_mask:
        total += L * 2 * H * seq * seq
    return total / 2**30


def static_gib(strategy: str, world: int) -> float:
    if strategy == "DDP":
        # fp32 master weights + fp32 grads + Adam m,v (16 B/param) + bf16 autocast copy.
        # (The current DDP path keeps params in bf16: 8 B/param -- smaller but see the notes.)
        return N_PARAMS * (16 + 2) / 2**30 + 0.6
    # FSDP: managed params sharded at 16 B/param; wte/lm_head are ignored modules
    # (replicated, bf16 param+grad+Adam = 8 B/param); ~0.6 GiB for gathered units.
    managed = N_PARAMS - wte_params
    return (managed * 16 / world + wte_params * 8) / 2**30 + 0.6


def max_micro_batch(gib: float, strategy: str, world: int, seq: int, cce=False, dense_mask=True, sparse=False) -> int:
    free = gib * USABLE_FRACTION - static_gib(strategy, world)
    return max(0, math.floor(free / per_sample_gib(seq, cce, dense_mask, sparse)))


def hours(tokens: float, world: int, tflops: float, mfu: float, strategy: str, seq: int, sparse: bool = False) -> float:
    # MFU is measured against the FLOPs actually executed, so a sparse run keeps the same MFU but
    # needs fewer FLOPs per token -> more tokens/s at equal utilisation (sparse gather/scatter
    # overhead is what MFU_SPARSE_PENALTY accounts for).
    tok_s = world * tflops * 1e12 * mfu * (MFU_SPARSE_PENALTY if sparse else 1.0) / flops_per_token(seq, sparse)
    if strategy == "FSDP":
        tok_s *= FSDP_THROUGHPUT_FACTOR
    return tokens / tok_s / 3600


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=50e9)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--dense-moe", action="store_true", help="model the original dense expert dispatch")
    ap.add_argument("--flex", action="store_true", help="model FlexAttention (no dense mask bias)")
    ap.add_argument("--cce", action="store_true", help="model cut cross entropy (no full logits)")
    args = ap.parse_args()
    sparse, dense_mask, cce = not args.dense_moe, not args.flex, args.cce

    print(f"params ~{N_PARAMS / 1e6:.0f}M ({(N_PARAMS - wpe_params) / 1e6:.0f}M without the {WPE_POSITIONS}-pos wpe table)")
    print(f"flops/token @ {args.seq}: dense MoE {flops_per_token(args.seq, False) / 1e9:.3f} GFLOPs | "
          f"sparse MoE {flops_per_token(args.seq, True) / 1e9:.3f} GFLOPs  (repo's mfu.py dense figure: 2.137 at 4096)")
    print(f"activations/sample @ {args.seq} (GiB): dense-moe+mask {per_sample_gib(args.seq, False, True, False):.1f} | "
          f"sparse+mask {per_sample_gib(args.seq, False, True, True):.1f} | "
          f"sparse+flex {per_sample_gib(args.seq, False, False, True):.1f} | "
          f"sparse+flex+cce {per_sample_gib(args.seq, True, False, True):.1f}")
    print(f"scenario for the table below: {'sparse' if sparse else 'dense'} MoE, "
          f"{'dense mask' if dense_mask else 'flex'}, {'CCE' if cce else 'full logits'}")
    print(f"static/GPU: DDP {static_gib('DDP', 2):.1f} GiB, FSDP x2 {static_gib('FSDP', 2):.1f}, FSDP x4 {static_gib('FSDP', 4):.1f}\n")

    hdr = f"{'GPU':42s}{'n':>3s} | {'B (DDP)':>8s}{'B (FSDP)':>9s} | {'kTok/s DDP':>11s}{'hrs DDP':>9s}{'GPU-hrs':>9s} | {'hrs FSDP':>9s}"
    print(f"micro-batch = sequences of {args.seq} tokens per GPU;  hours are for {args.tokens / 1e9:.0f}B tokens")
    print(hdr)
    print("-" * len(hdr))
    for name, (gib, tflops, mfu) in GPUS.items():
        for world in (2, 4):
            b_ddp = max_micro_batch(gib, "DDP", world, args.seq, cce, dense_mask, sparse)
            b_fsdp = max_micro_batch(gib, "FSDP", world, args.seq, cce, dense_mask, sparse)
            h_ddp = hours(args.tokens, world, tflops, mfu, "DDP", args.seq, sparse)
            h_fsdp = hours(args.tokens, world, tflops, mfu, "FSDP", args.seq, sparse)
            ktok = args.tokens / h_ddp / 3600 / 1e3
            print(f"{name:42s}{world:>3d} | {b_ddp:>8d}{b_fsdp:>9d} | {ktok:>11.0f}{h_ddp:>9.0f}{h_ddp * world:>9.0f} | {h_fsdp:>9.0f}")

    print("\nmicro-batch per GPU (DDP) by code configuration, and hours for the same token budget on 4 GPUs:")
    for name, (gib, tflops, mfu) in GPUS.items():
        cells = []
        for label, sp, dm, cc in (("dense+mask", False, True, False), ("sparse+mask", True, True, False),
                                  ("sparse+flex", True, False, False), ("sparse+flex+cce", True, False, True)):
            b = max_micro_batch(gib, "DDP", 4, args.seq, cc, dm, sp)
            h = hours(args.tokens, 4, tflops, mfu, "DDP", args.seq, sp)
            cells.append(f"{label}: B={b} {h:.0f}h")
        print(f"  {name:42s} " + " | ".join(cells))

if __name__ == "__main__":
    main()
