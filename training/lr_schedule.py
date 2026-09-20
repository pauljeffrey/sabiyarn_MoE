"""Learning-rate schedules as pure functions of the iteration number.

Kept torch-free so it is trivially unit-testable; Trainer._lr just delegates
here. All schedules share the same warmup: linear from 0 to `learning_rate`
over `warmup_iters`.

  cosine : cosine from peak to min_lr, reaching min_lr at lr_decay_iters.
  linear : straight line from peak to min_lr at lr_decay_iters. With min_lr=0
           this is the "linear decay to zero" (D2Z) schedule of Bergsma et al.
           2025 (arXiv 2502.15938).
  wsd    : warmup-stable-decay (Hagele et al. 2024, arXiv 2405.18392): hold the
           peak until the last `wsd_decay_frac` of max_iters, then cool down to
           min_lr by max_iters. Any stable-phase checkpoint can be cooled down
           on its own, so one run can be "finished" at several token budgets.
           `wsd_decay_shape` "sqrt" is the (1 - sqrt) cooldown they found best.
"""

from __future__ import annotations

import math
from typing import Any


def lr_at(it: int, cfg: Any) -> float:
    peak, floor = cfg.learning_rate, cfg.min_lr
    warmup = cfg.warmup_iters
    if it < warmup:
        return peak * it / max(1, warmup)

    scheduler = getattr(cfg, "scheduler", "cosine")
    if scheduler == "wsd":
        end = cfg.max_iters
        start = end - max(1, int(round(end * cfg.wsd_decay_frac)))
        if it <= start:
            return peak
        if it >= end:
            return floor
        t = (it - start) / (end - start)
        shape = getattr(cfg, "wsd_decay_shape", "sqrt")
        frac = {"sqrt": 1.0 - math.sqrt(t), "linear": 1.0 - t}.get(shape)
        if frac is None:
            frac = 0.5 * (1.0 + math.cos(math.pi * t))
        return floor + frac * (peak - floor)

    if it > cfg.lr_decay_iters:
        return floor
    t = (it - warmup) / max(1, cfg.lr_decay_iters - warmup)
    if scheduler == "linear":
        return floor + (1.0 - t) * (peak - floor)
    return floor + 0.5 * (1.0 + math.cos(math.pi * t)) * (peak - floor)
