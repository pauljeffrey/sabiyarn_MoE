"""Muon (MomentUm Orthogonalized by Newton-Schulz) for the hidden weight
matrices, with AdamW for everything else, as ONE optimizer object.

References
  - Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/ : Nesterov
    momentum, 5 quintic Newton-Schulz steps with (3.4445, -4.7750, 2.0315),
    Muon only for hidden matrices; AdamW for embeddings / output head / scalars.
    Q, K and V are orthogonalized separately rather than as one fused matrix.
  - Liu et al. 2025 (Moonlight), arXiv 2502.16982 : decoupled weight decay, and
    scaling each update by `0.2 * sqrt(max(m, n))` so its RMS matches AdamW's
    (~0.2). That is what lets Muon reuse the AdamW learning rate and weight
    decay unchanged instead of needing its own sweep.

Not compatible with FSDP: Newton-Schulz needs the full matrix, and FSDP hands
each rank a flat shard. Use DDP (`ddp.enabled: true`) or one GPU.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_NO_MUON_NAME_PARTS = ("wte", "wpe", "lm_head", "gate")  # embeddings, tied head, router


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor of the last two dims of `g`
    (batched over any leading dims). Singular values land roughly in [0.7, 1.2]."""
    a, b, c = _NS_COEFFS
    x = g.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transposed else x


def split_muon_params(model: torch.nn.Module, n_embd: int):
    """Returns (muon_params, adam_params_2d, adam_params_1d), trainable only.

    Muon: attention projections and expert / MLP weight matrices (2-D, or the
    3-D (experts, in, out) banks). AdamW: token/position embeddings, the tied
    lm_head, MoE router gates, and every 1-D parameter (norms, biases).
    """
    muon, adam_2d, adam_1d = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            adam_1d.append(p)
        elif any(part in name for part in _NO_MUON_NAME_PARTS) or p.ndim > 3:
            adam_2d.append(p)
        else:
            muon.append(p)
    return muon, adam_2d, adam_1d


class Muon(torch.optim.Optimizer):
    """`param_groups` entries carry `use_muon`. Muon groups read momentum /
    ns_steps / rms; the others are AdamW (betas, eps). All share `lr` and
    `weight_decay` semantics (decoupled), so the trainer's per-step
    `pg["lr"] = ...` schedule works unchanged."""

    def __init__(self, param_groups: Iterable[dict], *, n_embd: int, lr: float, weight_decay: float,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, rms: float = 0.2,
                 betas=(0.9, 0.95), eps: float = 1e-8):
        groups = []
        for g in param_groups:
            g = dict(g)
            g.setdefault("lr", lr)
            g.setdefault("weight_decay", weight_decay)
            g.setdefault("use_muon", False)
            g.update(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, rms=rms, betas=betas, eps=eps)
            groups.append(g)
        super().__init__(groups, dict(lr=lr))
        self.n_embd = n_embd

    def _matrices(self, t: torch.Tensor) -> torch.Tensor:
        """View a parameter/gradient as a stack of matrices to orthogonalize."""
        if t.ndim == 3:  # (experts, in, out) banks: one matrix per expert
            return t
        if t.ndim == 2 and t.size(0) == 3 * self.n_embd and t.size(1) == self.n_embd:
            return t.view(3, self.n_embd, self.n_embd)  # fused qkv -> q, k, v separately
        return t.unsqueeze(0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    buf = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    buf.mul_(group["momentum"]).add_(p.grad)
                    update = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    ortho = newton_schulz(self._matrices(update), group["ns_steps"])
                    m, n = ortho.size(-2), ortho.size(-1)
                    ortho = ortho * (group["rms"] * math.sqrt(max(m, n)))
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(ortho.reshape(p.shape).to(p.dtype), alpha=-lr)
                else:
                    step = state["step"] = state.get("step", 0) + 1
                    m1 = state.setdefault("exp_avg", torch.zeros_like(p))
                    m2 = state.setdefault("exp_avg_sq", torch.zeros_like(p))
                    b1, b2 = group["betas"]
                    m1.mul_(b1).add_(p.grad, alpha=1 - b1)
                    m2.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    denom = (m2 / (1 - b2**step)).sqrt_().add_(group["eps"])
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(m1, denom, value=-lr / (1 - b1**step))
        return loss
