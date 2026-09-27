"""Sequence-length budgets for SFT and post-training.

Config-only on purpose: these assertions are the guard against silently truncating away the part of a sample
that carries the training signal, and they must be runnable without a GPU stack -- tests/test_rl.py imports
torch, so they do not live there.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_shipped_length_caps_fit_the_real_corpus():
    """The shipped defaults were 512 / 1024, which silently discarded 75.7% of prompts and 70.2% of whole
    samples. Measured with BeardedMonster/SabiYarn-32k over 4,736 generated records, taking the conversation
    prefix as the prompt exactly as encode_answer does:

        prompt        median 1,260  p90 2,034  p99 6,762  max 13,867
        final answer  median    68  p90   121  p99   293  max    716

    A tool-calling conversation carries the tool catalogue in its system message plus several tool results, so
    it cannot fit 512 tokens -- the old cap excluded the corpus by construction, and the only symptom was a
    `dpo_examples_dropped_too_long` line in the log. These assertions fail if anyone tightens the caps back
    below what the data actually needs.
    """
    from rl.config import RLConfig
    cfg = RLConfig()
    assert cfg.max_prompt_len >= 13_867          # the longest prompt observed
    assert cfg.max_seq_len >= 14_274             # the longest prompt + answer observed
    assert cfg.max_new_tokens >= 716             # the longest answer observed


def test_length_caps_are_internally_consistent():
    from rl.config import RLConfig
    cfg = RLConfig()
    assert cfg.max_prompt_len < cfg.max_seq_len
    # policy_gradient enforces this at startup; asserting it here means the shipped defaults cannot violate it
    assert cfg.max_prompt_len + cfg.max_new_tokens <= cfg.max_seq_len


def test_dpo_micro_batch_stays_within_the_swept_memory_budget():
    """DPO forwards chosen AND rejected, so a pair is two sequences. At the old batch of 4 pairs and the new
    16,384 max_seq_len that is 131k tokens per micro-step, against the ~49k the pretraining config was swept to
    fit on an A100-80GB."""
    from rl.config import RLConfig
    cfg = RLConfig()
    assert 2 * cfg.batch_size * cfg.max_seq_len <= 49_152 * 1.1
    assert cfg.batch_size * cfg.grad_accum_steps >= 16      # effective batch not reduced


def test_sft_phase_overrides_block_size_without_touching_pretraining(monkeypatch):
    """A long_document_summarization sample carries 4,000-16,000 tokens of document plus its summary, and at
    block_size 4096 the part truncated away is the summary -- the only part of that sample that carries a
    training signal. Pretraining, whose documents are 300-500 words, must be unaffected."""
    from training.load_config import load_train_config

    monkeypatch.setenv("TRAIN_MODE", "pretrain")
    pre = load_train_config("training/train_config.yaml")
    monkeypatch.setenv("TRAIN_MODE", "sft")
    sft = load_train_config("training/train_config.yaml")

    assert pre.block_size == 4096 and sft.block_size == 16_384
    # Tokens per optimiser step must MATCH, or the swept LR schedule and max_iters no longer apply.
    assert (pre.train_batch_size * pre.block_size * pre.gradient_accumulation_steps
            == sft.train_batch_size * sft.block_size * sft.gradient_accumulation_steps)
    # sdpa_mask would build a (B,1,T,T) mask: ~800 MiB per layer per sample at this length.
    assert sft.attention_impl == "flex"
    assert sft.gradient_checkpointing is True


def test_env_override_still_beats_the_sft_phase_value(monkeypatch):
    from training.load_config import load_train_config

    monkeypatch.setenv("TRAIN_MODE", "sft")
    monkeypatch.setenv("TRAIN_BATCH_SIZE", "6")
    assert load_train_config("training/train_config.yaml").train_batch_size == 6
