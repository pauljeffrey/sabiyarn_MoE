"""Tokens-seen bookkeeping (training/token_accounting.py): torch-free."""
import json

import pytest

from training.token_accounting import record_micro_batch, restore_token_state, token_metrics, token_state_for_save

BS, SL, WORLD, ACCUM = 12, 4096, 4, 4
MICRO = BS * SL * WORLD
STEP = MICRO * ACCUM  # 786,432


def run_steps(seen, by_bin, bins_per_step):
    """What Trainer.train does: one record_micro_batch per micro-batch."""
    for bins in bins_per_step:
        assert len(bins) == ACCUM
        for b in bins:
            seen += record_micro_batch(by_bin, b, MICRO)
    return seen


def test_every_optimizer_step_adds_the_real_tokens_split_by_the_bin_that_supplied_them():
    by_bin = {}
    seen = run_steps(0, by_bin, [["eng", "afr", "afr", "other"], ["afr", "afr", "afr", "other"]])
    assert seen == 2 * STEP == 1_572_864
    assert by_bin == {"eng": MICRO, "afr": 5 * MICRO, "other": 2 * MICRO}
    assert sum(by_bin.values()) == seen


def test_restore_then_continue_matches_an_uninterrupted_run():
    steps = [["eng", "afr", "afr", "other"]] * 6
    full_by_bin = {}
    full = run_steps(0, full_by_bin, steps)

    by_bin = {}
    part = run_steps(0, by_bin, steps[:2])
    meta = json.loads(json.dumps({**token_state_for_save(part, 0, by_bin), "tokens_per_step": STEP}))  # via the json file
    st = restore_token_state(meta, use_best=False, iter_num=2, fallback_tokens_per_step=STEP)
    assert not st.estimated
    resumed = run_steps(st.seen, st.by_bin, steps[2:])
    assert resumed == full and st.by_bin == full_by_bin


def test_state_for_save_has_all_four_keys_including_lifetime():
    d = token_state_for_save(5_000, 20_000, {"eng_training": 3_000, "other_african_pretrain": 2_000})
    assert d == {
        "tokens_seen": 5_000, "tokens_seen_offset": 20_000, "tokens_seen_lifetime": 25_000,
        "tokens_seen_by_bin": {"eng_training": 3_000, "other_african_pretrain": 2_000},
    }
    json.dumps(d)  # serialisable


def test_a_checkpoint_without_the_counters_falls_back_to_the_old_estimate():
    st = restore_token_state({"iter_num": 100}, use_best=False, iter_num=100, fallback_tokens_per_step=STEP)
    assert st.estimated and st.seen == 100 * STEP and st.by_bin == {} and st.offset == 0
    # the file's own tokens_per_step wins over the current run's (it is what that run actually used)
    st = restore_token_state({"tokens_per_step": 1000, "tokens_seen_offset": 7}, use_best=False, iter_num=10,
                             fallback_tokens_per_step=STEP)
    assert st.seen == 10_000 and st.offset == 7 and st.estimated


def test_resume_from_best_uses_the_best_checkpoints_own_counts():
    meta = {"tokens_seen": 9_000, "tokens_seen_by_bin": {"a": 9_000}, "tokens_seen_offset": 100,
            "best_tokens_seen": 4_000, "best_tokens_seen_by_bin": {"a": 1_000, "b": 3_000}}
    best = restore_token_state(meta, use_best=True, iter_num=40, fallback_tokens_per_step=10)
    assert (best.seen, best.by_bin, best.offset, best.estimated) == (4_000, {"a": 1_000, "b": 3_000}, 100, False)
    latest = restore_token_state(meta, use_best=False, iter_num=90, fallback_tokens_per_step=10)
    assert latest.seen == 9_000 and latest.by_bin == {"a": 9_000}
    # an older file that never recorded best_* estimates from the rewound iter_num
    old = restore_token_state({"tokens_seen": 9_000, "tokens_per_step": 10}, use_best=True, iter_num=40,
                              fallback_tokens_per_step=99)
    assert old.estimated and old.seen == 400


def test_logged_metrics_have_the_documented_names_and_numeric_values():
    m = token_metrics(3_000, 1_000, {"eng_training": 1_000, "training_cleaned": 1_200, "other_african_pretrain": 800})
    assert m == {
        "tokens_seen": 3_000, "tokens_seen_lifetime": 4_000, "tokens_seen/eng_training": 1_000,
        "tokens_seen/training_cleaned": 1_200, "tokens_seen/other_african_pretrain": 800,
    }
    assert all(isinstance(v, int) for v in m.values())
    assert token_metrics(5, 0, {})["tokens_seen_lifetime"] == 5


def test_fresh_start_folds_the_run_total_into_the_offset():
    meta = token_state_for_save(5_000, 20_000, {"a": 5_000})
    st = restore_token_state(meta, use_best=False, iter_num=1, fallback_tokens_per_step=1)
    offset, seen = st.offset + st.seen, 0  # what Trainer._prepare_for_training does on fresh_start
    assert token_metrics(seen, offset, {})["tokens_seen_lifetime"] == 25_000
    assert token_state_for_save(seen, offset, {})["tokens_seen_lifetime"] == 25_000
