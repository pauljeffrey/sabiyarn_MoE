"""
Unit tests for training/utils.py's process_labels_optimized, in particular
its per-document scoping (see training/utils.py's docstring and
data/inspect_label_masking_modal.py for the real-data investigation that
found this bug).

training.constant_tokens loads the real tokenizer from Hugging Face at
import time, which needs network access this test environment may not
have. We stub it out with plain integer ids before importing
training.utils, which only needs the names to exist -- not the real
tokenizer. Same pattern as tests/test_label_masking.py.
"""
import sys
import types

import torch
import pytest

MASK = -100
EOS = 0
ANSWER = 100  # action token
QA = 200      # prompting token


@pytest.fixture(autouse=True)
def stub_constant_tokens(monkeypatch):
    stub = types.ModuleType("training.constant_tokens")
    stub.MASK = MASK
    stub.system_token = 1
    stub.user_token = 2
    stub.assistant_token = 3
    stub.end_of_text_token = EOS
    stub.prompting_tokens = [QA]
    stub.action_tokens = [ANSWER]
    monkeypatch.setitem(sys.modules, "training.constant_tokens", stub)
    monkeypatch.delitem(sys.modules, "training.utils", raising=False)
    yield


def test_single_document_single_action_tag_masks_start_through_tag():
    from training.utils import process_labels_optimized

    tokens = torch.tensor([ANSWER, 5, 6])
    result = process_labels_optimized(tokens.clone(), mask=MASK)
    assert result.tolist() == [MASK, 5, 6]


def test_single_document_single_prompting_tag_masks_tag_through_end():
    from training.utils import process_labels_optimized

    tokens = torch.tensor([5, QA, 6])
    result = process_labels_optimized(tokens.clone(), mask=MASK)
    assert result.tolist() == [5, MASK, MASK]


def test_single_document_pair_masks_between_tags_inclusive():
    from training.utils import process_labels_optimized

    tokens = torch.tensor([QA, 1, 2, QA, 3])
    result = process_labels_optimized(tokens.clone(), mask=MASK)
    assert result.tolist() == [MASK, MASK, MASK, MASK, 3]


def test_no_tags_returns_tokens_unchanged():
    from training.utils import process_labels_optimized

    tokens = torch.tensor([5, 6, 7, EOS, 8, 9])
    result = process_labels_optimized(tokens.clone(), mask=MASK)
    assert result.tolist() == tokens.tolist()


def test_untagged_document_between_two_tagged_documents_stays_unmasked():
    """The real bug: a tag in one document pairing with a tag in another,
    unrelated document, sweeping everything (including a completely
    untagged document) in between into the masked span. doc_A has a
    trailing QA tag, doc_M is pure untagged monolingual content, doc_B has
    a leading QA tag -- under the old flat (non-document-scoped) pairing,
    doc_A's and doc_B's QA tags would pair with each other and mask all of
    doc_M along with them.
    """
    from training.utils import process_labels_optimized

    doc_a = [3, QA, EOS]
    doc_m = [7, 8, 9, EOS]
    doc_b = [QA, 4, EOS]
    tokens = torch.tensor(doc_a + doc_m + doc_b)

    result = process_labels_optimized(tokens.clone(), mask=MASK)

    # doc_M (indices 3-6) must be completely untouched.
    assert result[3:7].tolist() == doc_m
    # doc_A: single trailing QA tag masks from the tag to doc_A's own end.
    assert result[0:3].tolist() == [3, MASK, MASK]
    # doc_B: single leading QA tag masks from the tag to doc_B's own end.
    assert result[7:10].tolist() == [MASK, MASK, MASK]


def test_two_tags_in_same_document_still_pair_within_that_document():
    """Sanity check that per-document scoping doesn't accidentally prevent
    real within-document pairing -- only cross-document pairing."""
    from training.utils import process_labels_optimized

    doc = [QA, 1, 2, QA, 3, EOS]
    tokens = torch.tensor(doc)
    result = process_labels_optimized(tokens.clone(), mask=MASK)
    assert result.tolist() == [MASK, MASK, MASK, MASK, 3, EOS]


def test_trailing_partial_document_after_last_eos_is_still_processed():
    """A window can end mid-document (packing cut it off) -- the trailing
    partial segment after the last EOS must still get its own masking
    pass, scoped independently from the documents before it."""
    from training.utils import process_labels_optimized

    doc_a = [ANSWER, 1, EOS]
    trailing_partial = [QA, 2, 3]  # no closing EOS -- window just ends here
    tokens = torch.tensor(doc_a + trailing_partial)

    result = process_labels_optimized(tokens.clone(), mask=MASK)
    assert result[0:3].tolist() == [MASK, 1, EOS]
    assert result[3:6].tolist() == [MASK, MASK, MASK]


def test_multiple_untagged_documents_are_all_left_alone():
    from training.utils import process_labels_optimized

    tokens = torch.tensor([1, 2, EOS, 3, 4, 5, EOS, 6, EOS])
    result = process_labels_optimized(tokens.clone(), mask=MASK)
    assert result.tolist() == tokens.tolist()
