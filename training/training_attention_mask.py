import torch


def build_document_causal_mask(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Boolean attention mask combining standard causal masking with
    document-boundary blocking, for sequences packing multiple documents
    separated by eos_token_id.

    Each token attends causally within its own document only (from the
    token after the previous eos_token_id, through itself); it cannot
    attend to tokens from an earlier document packed into the same
    sequence. The eos token itself belongs to the document it ends.

    Args:
        input_ids (torch.Tensor): Input token ids, shape (batch, seq_len).
        eos_token_id (int): Document boundary / end-of-text token id.

    Returns:
        torch.Tensor: Boolean mask, shape (batch, 1, seq_len, seq_len),
        True = attend, matching GPTJXMoEForCausalLM.forward's
        attention_mask convention.
    """
    batch, seq_len = input_ids.shape
    device = input_ids.device

    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    is_eos = input_ids == eos_token_id
    doc_id = torch.cumsum(is_eos.long(), dim=1) - is_eos.long()  # eos stays in the doc it ends
    same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)

    return (causal.unsqueeze(0) & same_doc).unsqueeze(1)



if __name__ == "__main__":
    # Example usage
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [11, 12, 13, 14, 15, 16, 17, 5, 19, 20],
                              [5, 22, 23, 5, 25, 26, 27, 28, 29, 30]])
    eos_token_id = 5
    mask = build_document_causal_mask(input_ids, eos_token_id)
    print(mask)
