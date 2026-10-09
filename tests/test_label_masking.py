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


def test_sft_turns_train_on_every_assistant_turn_and_nothing_else():
    from training.label_masking import apply_label_mask, process_sft_labels, process_sft_labels_turns

    EOS, TOOL = 0, 40
    seq = [USER, 5, 6, EOS, ASSISTANT, 7, 8, EOS, TOOL, 41, 42, EOS, ASSISTANT, 9, EOS, USER, 12, EOS, ASSISTANT, 13, 14, EOS]
    out = process_sft_labels_turns(torch.tensor(seq), ASSISTANT, EOS, mask=MASK).tolist()
    keep = {5, 6, 7, 13, 14, 15} | {19, 20, 21}                       # tool-call turn (7,8,EOS), final answer (13), second exchange (19..)
    expect = [t if i in {5, 6, 7, 13, 14, 19, 20, 21} else MASK for i, t in enumerate(seq)]
    assert out == expect
    assert out[9] == MASK and out[10] == MASK and out[16] == MASK       # the tool response and the later user turn are never trained on
    single = [SYSTEM, 5, USER, 6, EOS, ASSISTANT, 9, 10, EOS]
    assert process_sft_labels_turns(torch.tensor(single), ASSISTANT, EOS).tolist() == [MASK] * 6 + [9, 10, EOS]
    assert apply_label_mask(torch.tensor(single), "sft", user_token=USER, assistant_token=ASSISTANT, system_token=SYSTEM, eos_token=EOS).tolist() == [MASK] * 6 + [9, 10, EOS]
    # without an eos id the legacy behaviour (everything after the first assistant marker) is unchanged
    assert apply_label_mask(torch.tensor(single), "sft", user_token=USER, assistant_token=ASSISTANT, system_token=SYSTEM).tolist() == [MASK] * 6 + [9, 10, EOS]
    unfinished = [USER, 5, ASSISTANT, 9, 10]
    assert process_sft_labels_turns(torch.tensor(unfinished), ASSISTANT, EOS).tolist() == [MASK, MASK, MASK, 9, 10]


def test_sft_masks_tool_output_and_rag_context_wherever_they_appear():
    from training.label_masking import apply_label_mask, process_sft_labels_turns

    EOS, TR_O, TR_C, CT_O, CT_C = 0, 60, 61, 62, 63
    spans = ((TR_O, TR_C), (CT_O, CT_C))
    # user asks; assistant calls a tool (trained); tool answers inside <tool_response> (masked); assistant answers (trained) and quotes a context span (masked)
    seq = [USER, 5, EOS, ASSISTANT, 20, 21, EOS, TR_O, 30, 31, TR_C, EOS, ASSISTANT, 22, CT_O, 40, 41, CT_C, 23, EOS]
    out = process_sft_labels_turns(torch.tensor(seq), ASSISTANT, EOS, mask_spans=spans).tolist()
    assert out == [MASK] * 4 + [20, 21, EOS] + [MASK] * 6 + [22] + [MASK] * 4 + [23, EOS]
    # even a tool_response that (wrongly) sits inside an assistant turn is masked, markers included
    inside = [ASSISTANT, 7, TR_O, 8, TR_C, 9, EOS]
    assert process_sft_labels_turns(torch.tensor(inside), ASSISTANT, EOS, mask_spans=spans).tolist() == [MASK, 7, MASK, MASK, MASK, 9, EOS]
    assert apply_label_mask(torch.tensor(inside), "sft", user_token=USER, assistant_token=ASSISTANT, eos_token=EOS, mask_spans=spans).tolist() == [MASK, 7, MASK, MASK, MASK, 9, EOS]
    # an unclosed span masks to the end (it cannot be told apart from a truncated tool message)
    assert process_sft_labels_turns(torch.tensor([ASSISTANT, 7, CT_O, 8, 9]), ASSISTANT, EOS, mask_spans=spans).tolist() == [MASK, 7, MASK, MASK, MASK]
    # the real shape: ONE assistant turn = scaffold + call (trained, incl. </tool_call>) + <tool_response> (masked) + final answer (trained) + one </s>
    TC_C = 64
    turn = [USER, 5, EOS, ASSISTANT, 20, 21, TC_C, TR_O, 30, 31, TR_C, 22, 23, EOS]
    assert process_sft_labels_turns(torch.tensor(turn), ASSISTANT, EOS, mask_spans=spans).tolist() == [MASK] * 4 + [20, 21, TC_C] + [MASK] * 4 + [22, 23, EOS]
    # reasoning is trained: <think> tokens are plain assistant tokens
    think = [USER, 5, EOS, ASSISTANT, 70, 71, 72, 73, EOS]
    assert process_sft_labels_turns(torch.tensor(think), ASSISTANT, EOS, mask_spans=spans).tolist() == [MASK] * 4 + [70, 71, 72, 73, EOS]
