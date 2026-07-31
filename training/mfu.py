"""Model FLOPs Utilization (MFU) estimation.

Analytic (PaLM/nanoGPT-style) FLOPs-per-token estimate for GPTJXMoEConfig,
generalized to per-layer MoE. Unlike a typical sparse MoE, this model's
MoE.forward (sabiyarn/model/modeling.py) runs the fc_bank/proj_bank einsum
over EVERY configured expert for every token, then gathers the top-k result
afterward -- there is no token dispatch that skips non-selected experts'
matmuls. So the real hardware FLOPs per MoE layer are proportional to the
full expert count for that layer, not num_experts_per_tok, and that's what
this module counts: it reports actual hardware utilization (how busy the
GPUs really are), not algorithmic/"ideal-sparse-MoE" efficiency. See
HOW_TO_RUN.md / the MoE optimization note for the distinction and its
implications for training cost.
"""

from __future__ import annotations

import torch

# Dense (non-sparsity-accelerated) BF16/FP16 tensor-core peak TFLOPS per GPU,
# used as the MFU denominator. Best-effort figures collected from vendor specs
# -- treat as approximate, particularly for newer/less-common SKUs (Blackwell
# parts, PCIe vs SXM variants of the same chip can differ substantially).
# Match is by substring against torch.cuda.get_device_name(), longest-key-first
# so e.g. "H100 SXM" doesn't get shadowed by a plain "H100" entry.
_GPU_PEAK_TFLOPS: dict[str, float] = {
    "H200": 989.0,
    "H100 PCIE": 756.0,
    "H100 NVL": 990.0,
    "H100": 989.0,  # SXM, the common cloud form factor
    "A100": 312.0,  # 40GB and 80GB share the same compute
    "L40S": 362.0,
    "L4": 121.0,
    "A10G": 125.0,
    "A10": 125.0,
    "RTX 4090": 165.0,
    "RTX 3090": 71.0,
    "RTX A6000": 155.0,
    "RTX A5000": 111.0,
    "RTX A4000": 76.0,
    "T4": 65.0,
    "B200": 2250.0,  # approximate -- verify against current Blackwell specs
    "RTX PRO 6000": 1500.0,  # approximate -- verify against current specs
}


def peak_flops_for_current_device() -> float | None:
    """Peak BF16/FP16 tensor-core FLOPS/sec for the active CUDA device, or
    None if there's no CUDA device or the name doesn't match any known GPU
    (in which case MFU can't be computed, only raw achieved TFLOPS)."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(torch.cuda.current_device()).upper()
    for key in sorted(_GPU_PEAK_TFLOPS, key=len, reverse=True):
        if key.upper() in name:
            return _GPU_PEAK_TFLOPS[key] * 1e12
    return None


def model_flops_per_token(model_config, block_size: int) -> float:
    """Forward+backward FLOPs per token, analytic PaLM/nanoGPT-style estimate:

        total = 3 * (attn_proj + moe/ffn + lm_head + attn_score)   [3x = fwd + ~2x bwd]

    - attn_proj: qkv (D->3D) + out (D->D) projections, forward flops = 8*D^2/layer.
    - moe/ffn: for MoE layers, ALL configured experts for that layer (see module
      docstring) -- 4*D*moe_dim*num_experts_for_layer/layer; for a dense (non-MoE)
      layer, the usual 4x-expansion MLP -- 16*D^2/layer.
    - lm_head: final D->vocab_size projection, 2*D*V (embedding lookup itself is
      excluded -- it's a lookup, not a matmul).
    - attn_score: QK^T and softmax@V, scales with sequence length rather than
      parameter count -- 4*D*T/layer.
    """
    D = model_config.n_embd
    L = model_config.n_layer
    V = model_config.vocab_size
    T = block_size
    use_moe = bool(getattr(model_config, "use_moe", False))

    attn_proj_flops = 8 * D * D
    attn_score_flops = 4 * D * T

    if use_moe:
        moe_dim = getattr(model_config, "moe_dim", 4 * D)
        ffn_flops_per_layer = [
            4 * D * moe_dim * model_config.expert_count_for_layer(i) for i in range(L)
        ]
    else:
        ffn_flops_per_layer = [16 * D * D for _ in range(L)]

    lm_head_flops = 2 * D * V

    forward_flops_per_token = (
        L * attn_proj_flops + L * attn_score_flops + sum(ffn_flops_per_layer) + lm_head_flops
    )
    return 3.0 * forward_flops_per_token


def compute_mfu(
    flops_per_token: float,
    tokens_per_window: int,
    dt_seconds: float,
    world_size: int,
    peak_flops_per_gpu: float | None,
) -> tuple[float, float | None]:
    """Returns (achieved_tflops_per_gpu, mfu_fraction_or_None).

    achieved_tflops_per_gpu is always computable; mfu_fraction is None when
    peak_flops_per_gpu is unknown (unrecognized GPU -- see peak_flops_for_current_device).
    """
    if dt_seconds <= 0:
        return 0.0, None
    total_flops = flops_per_token * tokens_per_window
    achieved_flops_per_sec = total_flops / dt_seconds
    achieved_tflops_per_gpu = (achieved_flops_per_sec / max(1, world_size)) / 1e12
    if not peak_flops_per_gpu:
        return achieved_tflops_per_gpu, None
    mfu = achieved_flops_per_sec / (peak_flops_per_gpu * max(1, world_size))
    return achieved_tflops_per_gpu, mfu
