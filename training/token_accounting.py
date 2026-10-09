"""Tokens-seen bookkeeping for the trainer, kept pure (no torch) so it is unit-testable anywhere.

Three numbers are tracked:

  * ``tokens_seen``          -- tokens consumed by optimizer steps in THIS run (resets on a fresh start)
  * ``tokens_seen_offset``   -- the total carried over from earlier runs (a fresh start folds
                                ``tokens_seen`` into it), so ``lifetime = offset + tokens_seen`` never resets
  * ``tokens_seen_by_bin``   -- ``tokens_seen`` split by the training bin that supplied the blocks

Every micro-batch adds ``batch x block_size x world_size`` tokens to BOTH ``tokens_seen`` and its bin
(``record_micro_batch``), so ``sum(by_bin) == tokens_seen`` for any run whose counts were exact from
step 0 (or restored from a checkpoint that saved them). A checkpoint that predates this bookkeeping is
restored with an ESTIMATE (``iter_num x tokens_per_step``) and an empty by-bin split; ``estimated=True``
flags it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


def record_micro_batch(by_bin: MutableMapping[str, int], bin_name: str, tokens: int) -> int:
    """Add one micro-batch (all ranks together) to the per-bin counts; returns ``tokens`` so the
    caller can add the very same number to its running ``tokens_seen``."""
    by_bin[bin_name] = by_bin.get(bin_name, 0) + int(tokens)
    return int(tokens)


@dataclass
class TokenState:
    seen: int = 0
    offset: int = 0
    by_bin: dict = field(default_factory=dict)
    estimated: bool = False


def restore_token_state(
    meta: Mapping[str, Any], *, use_best: bool, iter_num: int, fallback_tokens_per_step: int,
) -> TokenState:
    """Token state to resume with, from a trainer_state.json dict.

    ``use_best``: resuming from ckpt_best, whose weights are older than the "latest" counters in the
    file. The best checkpoint's own counts are saved as ``best_tokens_seen*``; without them the counts
    are estimated from ``iter_num`` (already rewound to the best iteration by the caller).
    A missing key falls back to the old estimate ``iter_num x tokens_per_step`` (``tokens_per_step``
    recorded in the file when present: wrong if the batch size changed since, hence ``estimated``).
    """
    offset = int(meta.get("tokens_seen_offset", 0) or 0)
    key = "best_tokens_seen" if use_best else "tokens_seen"
    if meta.get(key) is not None:
        by_bin_key = "best_tokens_seen_by_bin" if use_best else "tokens_seen_by_bin"
        by_bin = {str(k): int(v) for k, v in (meta.get(by_bin_key) or {}).items()}
        return TokenState(seen=int(meta[key]), offset=offset, by_bin=by_bin, estimated=False)
    per_step = int(meta.get("tokens_per_step") or fallback_tokens_per_step)
    return TokenState(seen=int(iter_num) * per_step, offset=offset, by_bin={}, estimated=True)


def token_state_for_save(seen: int, offset: int, by_bin: Mapping[str, int]) -> dict:
    """The keys written to trainer_state.json at EVERY checkpoint save."""
    return {
        "tokens_seen": int(seen),
        "tokens_seen_offset": int(offset),
        "tokens_seen_lifetime": int(offset) + int(seen),
        "tokens_seen_by_bin": {k: int(v) for k, v in by_bin.items()},
    }


def token_metrics(seen: int, offset: int, by_bin: Mapping[str, int]) -> dict:
    """Numeric metrics for the console / wandb / mlflow at every log interval:
    ``tokens_seen``, ``tokens_seen_lifetime`` and ``tokens_seen/<bin name>`` (raw token counts)."""
    out: dict = {"tokens_seen": int(seen), "tokens_seen_lifetime": int(offset) + int(seen)}
    for name, n in by_bin.items():
        out[f"tokens_seen/{name}"] = int(n)
    return out
