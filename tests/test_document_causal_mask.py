import torch

from training.training_attention_mask import build_document_causal_mask

EOS = 99


def test_single_document_is_plain_causal_mask():
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = build_document_causal_mask(ids, EOS)
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool)).view(1, 1, 4, 4)
    assert torch.equal(mask, expected)


def test_blocks_attention_across_document_boundary():
    # docs: [a, b, EOS] | [c, d, EOS] | [e]
    ids = torch.tensor([[1, 2, EOS, 3, 4, EOS, 5]])
    mask = build_document_causal_mask(ids, EOS)[0, 0]

    # position 3 ('c', doc 1) must not attend to positions 0-2 (doc 0)
    assert not mask[3, 0] and not mask[3, 1] and not mask[3, 2]
    assert mask[3, 3]  # attends to itself

    # position 4 ('d', doc 1) attends within doc 1 (positions 3,4) only
    assert mask[4, 3] and mask[4, 4]
    assert not mask[4, 0] and not mask[4, 1] and not mask[4, 2]

    # position 6 ('e', doc 2, starts a new doc after the second EOS)
    # attends only to itself
    assert mask[6, 6]
    assert not mask[6, 0] and not mask[6, 5]


def test_eos_token_attends_within_its_own_preceding_document():
    ids = torch.tensor([[1, 2, EOS, 3]])
    mask = build_document_causal_mask(ids, EOS)[0, 0]
    # EOS at position 2 belongs to doc 0 (with positions 0,1), causally
    assert mask[2, 0] and mask[2, 1] and mask[2, 2]


def test_causal_ordering_still_holds_within_a_document():
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = build_document_causal_mask(ids, EOS)[0, 0]
    for i in range(4):
        for j in range(4):
            assert mask[i, j].item() == (j <= i)


def test_batched_independent_documents():
    ids = torch.tensor([
        [1, EOS, 2, 3],
        [1, 2, 3, 4],
    ])
    mask = build_document_causal_mask(ids, EOS)
    # row 0: position 2 must not see position 0 (across EOS)
    assert not mask[0, 0, 2, 0]
    # row 1: no EOS at all, plain causal
    expected_row1 = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(mask[1, 0], expected_row1)


def test_output_shape_and_dtype():
    ids = torch.randint(0, 50, (3, 8))
    mask = build_document_causal_mask(ids, EOS)
    assert mask.shape == (3, 1, 8, 8)
    assert mask.dtype == torch.bool
