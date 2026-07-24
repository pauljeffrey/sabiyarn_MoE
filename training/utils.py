import torch
from transformers import AutoTokenizer
import numpy as np
from training.constant_tokens import *


def create_token_conditions():
    """
    Pre-compute token condition tensors for better performance.
    This avoids repeated tensor comparisons.
    """
    # Group tokens by category for efficient lookup
    token_groups = {
        'prompting_tokens': torch.tensor(prompting_tokens),
        'action_tokens': torch.tensor(action_tokens)
    }

    return token_groups


def find_tag_indices(tokens, token_groups):
    """
    Vectorized approach to find all tag indices at once.

    Args:
        tokens (torch.Tensor): Input token tensor
        token_groups (dict): Pre-computed token groups

    Returns:
        torch.Tensor: Indices where any tag tokens appear
    """
    # Combine all tokens into a single tensor
    all_tag_tokens = torch.cat(list(token_groups.values()))

    # Create a mask for all tag positions
    # Use broadcasting to compare tokens against all tag tokens at once
    tag_mask = (tokens.unsqueeze(1) == all_tag_tokens.unsqueeze(0)).any(dim=1)

    # Get indices
    indices = torch.where(tag_mask)[0]

    return indices


def _process_segment_labels(tokens, token_groups, mask):
    """
    The original process_labels_optimized pairing logic, applied to ONE
    document's tokens already sliced at that document's boundaries. Tag
    pairing (and the single-tag / trailing-odd-tag special cases) only ever
    sees tags belonging to this one document -- see process_labels_optimized
    for why that scoping matters.

    Args:
        tokens (torch.Tensor): Token tensor for a single document segment
        token_groups (dict): Pre-computed token groups (create_token_conditions())
        mask (int): Mask value

    Returns:
        torch.Tensor: Processed token tensor for this segment
    """
    if len(tokens) == 0:
        return tokens

    indices = find_tag_indices(tokens, token_groups)
    if len(indices) == 0:
        return tokens

    indices_list = indices.tolist()
    result = tokens.clone()
    num_indices = len(indices_list)

    if num_indices == 1:
        idx = indices_list[0]
        if tokens[idx].item() in action_tokens:
            result[:idx + 1] = mask
        else:
            result[idx:] = mask
        return result

    # Handle starting action token
    start_offset = 0
    if tokens[indices_list[0]].item() in action_tokens:
        result[:indices_list[0] + 1] = mask
        start_offset = 1

    # Handle ending token
    end_offset = 0
    remaining_indices = indices_list[start_offset:]
    if len(remaining_indices) % 2 == 1:  # Odd number remaining
        result[remaining_indices[-1]:] = mask
        end_offset = 1

    # Process pairs efficiently
    final_indices = remaining_indices[:len(remaining_indices) - end_offset]

    # Vectorized pair processing
    if len(final_indices) >= 2:
        # Convert to pairs using tensor operations
        pairs = torch.tensor(final_indices).view(-1, 2)

        # Apply masking for each pair
        for start_idx, end_idx in pairs:
            result[start_idx:end_idx + 1] = mask

    return result


def process_labels_optimized(tokens, mask=-100):
    """
    Mask prompt/action token spans, scoped independently PER DOCUMENT
    (split at end_of_text_token) so tag pairing never spans a document
    boundary.

    A packed pretraining window contains many independent documents
    (monolingual text with no tags, plus tagged task documents --
    translation, sentiment, NER, ...). The underlying pairing logic
    (_process_segment_labels) treats consecutive tag occurrences as
    open/close delimiters and masks everything between them -- without
    per-document scoping, a tag in one document could pair with a tag in a
    later, unrelated document, silently masking every token in between
    (including whole other, untagged documents) for no linguistic reason.
    Confirmed on real data (data/inspect_label_masking_modal.py): ~14% of
    previously-masked tokens belonged to spans that crossed a document
    boundary this way.

    Args:
        tokens (torch.Tensor): Input token tensor
        mask (int): Mask value

    Returns:
        torch.Tensor: Processed token tensor
    """
    if len(tokens) == 0:
        return tokens

    token_groups = create_token_conditions()
    eos_positions = (tokens == end_of_text_token).nonzero(as_tuple=False).flatten().tolist()

    if not eos_positions:
        # No document boundaries in this window -- single document, same as
        # applying the pairing logic directly to the whole thing.
        return _process_segment_labels(tokens, token_groups, mask)

    result = tokens.clone()
    seg_start = 0
    for eos_pos in eos_positions:
        seg_end = eos_pos + 1  # the EOS token itself closes out its own document
        result[seg_start:seg_end] = _process_segment_labels(tokens[seg_start:seg_end], token_groups, mask)
        seg_start = seg_end

    if seg_start < len(tokens):
        # Trailing partial document after the last EOS (packing cut it off
        # mid-document at the window boundary).
        result[seg_start:] = _process_segment_labels(tokens[seg_start:], token_groups, mask)

    return result
