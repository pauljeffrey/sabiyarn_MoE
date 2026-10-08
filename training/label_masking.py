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


def process_sft_labels_turns(
    tokens: torch.Tensor,
    assistant_token: int,
    eos_token: int,
    mask: int = MASK,
    mask_spans: tuple = (),
) -> torch.Tensor:
    """Loss on the assistant turns only, in EVERY turn of a (possibly packed) sequence.

    A turn is the run of tokens after <|assistant|> up to and including the </s> that closes it (a tool-call turn is one such turn: scaffold,
    <tool_call>..</tool_call>, </s>). Everything else is masked: system and user turns, the <|assistant|> marker itself, <s>, and above all the
    <tool_response> turns -- text the environment wrote, which the model must not learn to produce. An assistant turn with no closing </s>
    (a truncated sequence) runs to the end. Single-turn sequences give the same labels as `process_sft_labels`.

    Trained: the <think> reasoning, the task plan, tool calls and the final response (everything an assistant turn contains). Never trained:
    anything that is input to the model. `mask_spans` lists (open, close) token pairs -- <tool_response>..</tool_response> and
    <context>..</context> -- whose whole span, markers included, is masked wherever it occurs, also inside an assistant turn.
    """
    result = torch.full_like(tokens, mask)
    is_asst = (tokens == assistant_token).tolist()
    is_eos = (tokens == eos_token).tolist()
    inside = False
    for i, (a, e) in enumerate(zip(is_asst, is_eos)):
        if inside:
            result[i] = tokens[i]
            if e:
                inside = False
        elif a:
            inside = True
    for op, cl in mask_spans:                                   # input-to-the-model spans, even if they fall inside an assistant turn
        is_op, is_cl = (tokens == op).tolist(), (tokens == cl).tolist()
        start = None
        for i, (o, c) in enumerate(zip(is_op, is_cl)):
            if o and start is None:
                start = i
            if start is not None:
                result[i] = mask
                if c:
                    start = None
    return result


def apply_label_mask(
    tokens: torch.Tensor,
    mode: str,
    *,
    user_token: int,
    assistant_token: int,
    system_token: int | None = None,
    mask: int = MASK,
    eos_token: int | None = None,
    mask_spans: tuple = (),
) -> torch.Tensor:
    if mode.lower() == "sft":
        if eos_token is not None:
            return process_sft_labels_turns(tokens, assistant_token, eos_token, mask=mask, mask_spans=mask_spans)
        return process_sft_labels(
            tokens, user_token, assistant_token, system_token=system_token, mask=mask
        )
    return process_pretrain_labels(tokens, mask=mask)
