"""Label masking for pretrain and SFT modes."""

from __future__ import annotations

import torch

from training.constant_tokens import MASK
from training.utils import process_labels_optimized


def process_pretrain_labels(tokens: torch.Tensor, mask: int = MASK) -> torch.Tensor:
    """Mask prompt/action token spans (existing pretrain behaviour)."""
    return process_labels_optimized(tokens.clone(), mask=mask)


def process_sft_labels(
    tokens: torch.Tensor,
    user_token: int,
    assistant_token: int,
    system_token: int | None = None,
    mask: int = MASK,
) -> torch.Tensor:
    """
    Mask the prompt for chat SFT.

    The masked region starts at <|system|> if present, else at <|user|> if
    present, else falls back to sequence start; it runs through
    <|assistant|> (inclusive) when the assistant marker is present,
    otherwise to the end of the sequence. Loss is computed only on the
    assistant response tokens that follow.
    """
    result = tokens.clone()
    asst_hits = (tokens == assistant_token).nonzero(as_tuple=False).flatten()

    start = 0
    sys_hits = (
        (tokens == system_token).nonzero(as_tuple=False).flatten()
        if system_token is not None
        else None
    )
    if sys_hits is not None and sys_hits.numel() > 0:
        start = sys_hits[0].item()
    else:
        user_hits = (tokens == user_token).nonzero(as_tuple=False).flatten()
        if user_hits.numel() > 0:
            start = user_hits[0].item()

    if asst_hits.numel() > 0:
        result[start : asst_hits[0].item() + 1] = mask
    elif start > 0:
        result[start:] = mask
    return result


def apply_label_mask(
    tokens: torch.Tensor,
    mode: str,
    *,
    user_token: int,
    assistant_token: int,
    system_token: int | None = None,
    mask: int = MASK,
) -> torch.Tensor:
    if mode.lower() == "sft":
        return process_sft_labels(
            tokens, user_token, assistant_token, system_token=system_token, mask=mask
        )
    return process_pretrain_labels(tokens, mask=mask)
