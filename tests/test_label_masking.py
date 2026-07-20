"""
Unit tests for training/label_masking.py using synthetic token ids.

training.constant_tokens loads the real tokenizer from Hugging Face at
import time, which needs network access this test environment may not
have. We stub it out with plain integer ids before importing
training.label_masking / training.utils, which only need the names to
exist -- not the real tokenizer.
"""
import sys
import types

import torch
import pytest

MASK = -100
SYSTEM = 1
USER = 2
ASSISTANT = 3
QA = 10
ANSWER = 11


@pytest.fixture(autouse=True)
def stub_constant_tokens(monkeypatch):
    stub = types.ModuleType("training.constant_tokens")
    stub.MASK = MASK
    stub.system_token = SYSTEM
    stub.user_token = USER
    stub.assistant_token = ASSISTANT
    stub.end_of_text_token = 0
    stub.prompting_tokens = [QA]
    stub.action_tokens = [ANSWER, ASSISTANT]
    monkeypatch.setitem(sys.modules, "training.constant_tokens", stub)
    for mod in ("training.utils", "training.label_masking"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    yield


def test_sft_masks_from_system_through_assistant():
    from training.label_masking import process_sft_labels

    tokens = torch.tensor([SYSTEM, 5, 6, USER, 7, 8, ASSISTANT, 9, 10])
    result = process_sft_labels(tokens, USER, ASSISTANT, system_token=SYSTEM, mask=MASK)
    assert result.tolist() == [MASK] * 7 + [9, 10]


def test_sft_masks_from_user_when_no_system():
    from training.label_masking import process_sft_labels

    tokens = torch.tensor([100, USER, 7, 8, ASSISTANT, 9, 10])
    result = process_sft_labels(tokens, USER, ASSISTANT, system_token=SYSTEM, mask=MASK)
    assert result.tolist() == [100, MASK, MASK, MASK, MASK, 9, 10]


def test_sft_system_takes_priority_over_user():
    from training.label_masking import process_sft_labels

    tokens = torch.tensor([USER, SYSTEM, 7, ASSISTANT, 9])
    result = process_sft_labels(tokens, USER, ASSISTANT, system_token=SYSTEM, mask=MASK)
    # start = SYSTEM's position (index 1), not USER's (index 0)
    assert result.tolist() == [USER, MASK, MASK, MASK, 9]


def test_sft_no_assistant_masks_from_start_marker_onward():
    from training.label_masking import process_sft_labels

    tokens = torch.tensor([100, USER, 7, 8])
    result = process_sft_labels(tokens, USER, ASSISTANT, system_token=SYSTEM, mask=MASK)
    assert result.tolist() == [100, MASK, MASK, MASK]


def test_sft_no_markers_at_all_falls_back_to_no_masking():
    from training.label_masking import process_sft_labels

    tokens = torch.tensor([100, 101, 102])
    result = process_sft_labels(tokens, USER, ASSISTANT, system_token=SYSTEM, mask=MASK)
    assert result.tolist() == [100, 101, 102]


def test_sft_assistant_only_still_masks_from_zero_safety_net():
    from training.label_masking import process_sft_labels

    tokens = torch.tensor([100, 101, ASSISTANT, 9])
    result = process_sft_labels(tokens, USER, ASSISTANT, system_token=SYSTEM, mask=MASK)
    assert result.tolist() == [MASK, MASK, MASK, 9]


def test_sft_without_system_token_arg_behaves_like_user_only():
    from training.label_masking import process_sft_labels

    tokens = torch.tensor([SYSTEM, USER, 7, ASSISTANT, 9])
    result = process_sft_labels(tokens, USER, ASSISTANT, mask=MASK)
    # system_token=None -> falls straight to <|user|> (index 1), ignoring SYSTEM
    assert result.tolist() == [SYSTEM, MASK, MASK, MASK, 9]


def test_apply_label_mask_sft_mode():
    from training.label_masking import apply_label_mask

    tokens = torch.tensor([SYSTEM, 5, ASSISTANT, 9])
    result = apply_label_mask(
        tokens, "sft", user_token=USER, assistant_token=ASSISTANT,
        system_token=SYSTEM, mask=MASK,
    )
    assert result.tolist() == [MASK, MASK, MASK, 9]


def test_apply_label_mask_pretrain_mode_delegates():
    from training.label_masking import apply_label_mask

    tokens = torch.tensor([1, QA, 2, 3])
    result = apply_label_mask(
        tokens, "pretrain", user_token=USER, assistant_token=ASSISTANT, mask=MASK,
    )
    # pretrain mode ignores user/assistant/system entirely, delegates to
    # process_labels_optimized via the stubbed prompting/action token lists
    assert result[0].item() == 1
