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


def process_labels_optimized(tokens, mask=-100):
    """
    Highly optimized version of process_labels with vectorized operations.

    Args:
        tokens (torch.Tensor): Input token tensor
        mask (int): Mask value

    Returns:
        torch.Tensor: Processed token tensor
    """
    if len(tokens) == 0:
        return tokens

    # Pre-compute token groups
    token_groups = create_token_conditions()

    # Find all tag indices vectorized
    indices = find_tag_indices(tokens, token_groups)

    if len(indices) == 0:
        return tokens

    # Convert to list for easier manipulation (only once)
    indices_list = indices.tolist()

    # Clone tensor to avoid in-place modifications
    result = tokens.clone()

    # Optimized logic for different cases
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
