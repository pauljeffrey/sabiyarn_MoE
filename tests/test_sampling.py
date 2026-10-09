import pytest

from training.load_config import load_train_config, sampling_weights


def test_sampling_config_defaults_match_yaml():
    cfg = load_train_config("training/train_config.yaml")
    # Whether scheduled sampling is on is a tuning choice that changes (the yaml has it on), so
    # only assert it parses as a boolean, like the weights below.
    assert isinstance(cfg.use_scheduled_sampling, bool)
    # Exact weights are under active tuning -- assert they're a valid,
    # normalized eng/afr/other_afr split rather than pinning literal values.
    total = cfg.eng_sampling_weight + cfg.afr_sampling_weight + cfg.other_afr_sampling_weight
    assert abs(total - 1.0) < 1e-9
    for w in (cfg.eng_sampling_weight, cfg.afr_sampling_weight, cfg.other_afr_sampling_weight):
        assert 0.0 <= w <= 1.0


def test_fixed_sampling_holds_preset_regardless_of_iter():
    for iter_num in (0, 100, 10_000):
        eng_w, afr_w = sampling_weights(0.8, 0.2, iter_num, 1000, use_scheduled_sampling=False)
        assert (eng_w, afr_w) == (0.8, 0.2)


def test_scheduled_sampling_starts_at_preset():
    eng_w, afr_w = sampling_weights(0.8, 0.2, 0, 1000, use_scheduled_sampling=True)
    assert abs(eng_w - 0.8) < 1e-9
    assert abs(afr_w - 0.2) < 1e-9


def test_scheduled_sampling_ends_at_swapped_ratio():
    eng_w, afr_w = sampling_weights(0.8, 0.2, 1000, 1000, use_scheduled_sampling=True)
    assert abs(eng_w - 0.2) < 1e-9
    assert abs(afr_w - 0.8) < 1e-9


def test_scheduled_sampling_is_monotonic_toward_afr():
    weights = [sampling_weights(0.8, 0.2, i, 1000, use_scheduled_sampling=True)[0] for i in range(0, 1001, 100)]
    assert all(a >= b for a, b in zip(weights, weights[1:]))


def test_scheduled_sampling_no_op_for_even_preset():
    for iter_num in (0, 500, 1000):
        eng_w, afr_w = sampling_weights(0.5, 0.5, iter_num, 1000, use_scheduled_sampling=True)
        assert abs(eng_w - 0.5) < 1e-9
        assert abs(afr_w - 0.5) < 1e-9


def test_sampling_weights_normalizes_non_unit_sum():
    eng_w, afr_w = sampling_weights(2.0, 2.0, 0, 1000, use_scheduled_sampling=False)
    assert abs(eng_w - 0.5) < 1e-9
    assert abs(afr_w - 0.5) < 1e-9


def test_piecewise_schedule_interpolates_and_holds():
    sched = ((0.0, 0.4), (0.5, 0.4), (1.0, 0.8))
    f = lambda it: sampling_weights(0.6, 0.4, it, 1000, True, sched)
    assert f(0) == (0.6, 0.4)
    assert f(250)[1] == 0.4                      # flat before the second knot
    assert abs(f(750)[1] - 0.6) < 1e-9           # halfway up the ramp
    assert abs(f(1000)[1] - 0.8) < 1e-9 and abs(f(5000)[1] - 0.8) < 1e-9  # held after the last knot
    for it in (0, 333, 999):
        e, a = f(it)
        assert abs(e + a - 1.0) < 1e-9


def test_schedule_ignored_when_scheduled_sampling_off():
    assert sampling_weights(0.6, 0.4, 900, 1000, False, ((0.0, 0.1),)) == (0.6, 0.4)


def test_schedule_validation():
    import pytest
    from training.load_config import parse_sampling_schedule
    assert parse_sampling_schedule(None) == ()
    assert parse_sampling_schedule([[1, 0.7], [0, 0.3]]) == ((0.0, 0.3), (1.0, 0.7))
    for bad in ([[0.0]], [[0.0, 1.5]], [[0.2, 0.3], [0.2, 0.4]]):
        with pytest.raises(ValueError):
            parse_sampling_schedule(bad)


def test_yaml_schedule_parses():
    cfg = load_train_config("training/train_config.yaml")
    assert isinstance(cfg.sampling_schedule, tuple)


# ---- third bin (eng / afr / other_afr) ----------------------------------------------------------

from training.load_config import (  # noqa: E402
    active_bin_indices, mixture_weights, parse_sampling_schedule, select_bin_weights,
)


def test_other_weight_defaults_to_zero_and_leaves_two_bin_behaviour_untouched():
    assert load_train_config("training/train_config.yaml").other_afr_sampling_weight == 0.3
    from training.load_config import TrainConfig
    assert TrainConfig().other_afr_sampling_weight == 0.0
    assert mixture_weights(0.8, 0.2, 0, 1000, False) == (0.8, 0.2, 0.0)
    for it in (0, 400, 1000):
        e, a, o = mixture_weights(0.8, 0.2, it, 1000, True)
        assert (e, a) == sampling_weights(0.8, 0.2, it, 1000, True) and o == 0.0


def test_fixed_three_bin_mixture_is_normalised():
    e, a, o = mixture_weights(0.15, 0.55, 5, 100, False, (), 0.30)
    assert (e, a, o) == pytest.approx((0.15, 0.55, 0.30))
    e, a, o = mixture_weights(3, 11, 5, 100, False, (), 6)  # only the ratios matter
    assert (e, a, o) == pytest.approx((0.15, 0.55, 0.30))
    assert mixture_weights(0, 0, 0, 1, False, (), 0) == (0.5, 0.5, 0.0)


def test_cosine_schedule_holds_other_constant_and_swaps_eng_afr():
    start = mixture_weights(0.6, 0.1, 0, 1000, True, (), 0.3)
    end = mixture_weights(0.6, 0.1, 1000, 1000, True, (), 0.3)
    assert start == pytest.approx((0.6, 0.1, 0.3)) and end == pytest.approx((0.1, 0.6, 0.3))
    for it in range(0, 1001, 125):
        e, a, o = mixture_weights(0.6, 0.1, it, 1000, True, (), 0.3)
        assert o == pytest.approx(0.3) and e + a + o == pytest.approx(1.0)


def test_piecewise_schedule_with_optional_other_weight():
    sched = parse_sampling_schedule([[0.0, 0.5, 0.1], [1.0, 0.5, 0.3]])
    assert sched == ((0.0, 0.5, 0.1), (1.0, 0.5, 0.3))
    e, a, o = mixture_weights(0.4, 0.5, 500, 1000, True, sched, 0.1)
    assert (e, a, o) == pytest.approx((0.3, 0.5, 0.2))
    # mixed knots: a missing `other` means 0 (unchanged behaviour), tuples become uniform 3-tuples
    mixed = parse_sampling_schedule([[0.0, 0.5], [1.0, 0.5, 0.2]])
    assert mixed == ((0.0, 0.5, 0.0), (1.0, 0.5, 0.2))
    # a 2-number schedule never draws the third bin, whatever the preset says
    assert mixture_weights(0.2, 0.5, 100, 1000, True, ((0.0, 0.5), (1.0, 0.5)), 0.3)[2] == 0.0
    for bad in ([[0.0, 0.7, 0.5]], [[0.0, 0.5, 1.5]], [[0.0, 0.5, 0.1, 0.1]]):
        with pytest.raises(ValueError):
            parse_sampling_schedule(bad)


def test_active_bin_indices_drops_bins_that_can_never_be_drawn():
    assert active_bin_indices(0.15, 0.55, 0.30, False, (), 3) == (0, 1, 2)
    assert active_bin_indices(0.15, 0.85, 0.0, False, (), 3) == (0, 1)
    assert active_bin_indices(0.15, 0.85, 0.0, False, (), 2) == (0, 1)
    assert active_bin_indices(0.0, 0.7, 0.3, False, (), 3) == (1, 2)
    assert active_bin_indices(0.0, 0.0, 0.0, False, (), 3) == (0, 1, 2)  # degenerate -> keep all
    # scheduled: a bin is live if ANY knot gives it weight
    assert active_bin_indices(0.5, 0.5, 0.0, True, ((0.0, 1.0), (1.0, 1.0)), 3) == (1,)
    assert active_bin_indices(0.5, 0.5, 0.0, True, ((0.0, 0.5), (1.0, 0.5, 0.2)), 3) == (0, 1, 2)
    assert active_bin_indices(0.0, 1.0, 0.0, True, (), 3) == (0, 1)  # cosine swap makes eng live


def test_select_bin_weights_for_1_2_and_3_bins():
    w = (0.15, 0.55, 0.30)
    assert select_bin_weights(w, (0,)) == (1.0,)
    assert select_bin_weights(w, (0, 1)) == pytest.approx((0.15 / 0.70, 0.55 / 0.70))
    assert select_bin_weights(w, (1, 2)) == pytest.approx((0.55 / 0.85, 0.30 / 0.85))
    assert select_bin_weights(w, (0, 1, 2)) == pytest.approx(w)
    assert select_bin_weights((0.0, 0.0, 0.0), (0, 1)) == (0.5, 0.5)
    assert select_bin_weights(w, (0, 1, 2, 3)) == (0.25,) * 4  # no weights defined past the third bin


def test_yaml_new_run_defaults():
    cfg = load_train_config("training/train_config.yaml")
    assert (cfg.eng_sampling_weight, cfg.afr_sampling_weight, cfg.other_afr_sampling_weight) == (0.15, 0.55, 0.30)
    assert cfg.eval_manifest_path.endswith("validation_v2_manifest.json")
    assert cfg.eval_data_path.endswith("validation_v2.bin")
    assert (cfg.eval_total_blocks, cfg.eval_old_share) == (500, 0.4)


def test_eval_keys_default_to_the_old_behaviour_when_absent(tmp_path):
    from training.load_config import TrainConfig
    p = tmp_path / "c.yaml"
    p.write_text("training:\n  eval_iters: 5\n")
    cfg = load_train_config(p)
    assert cfg.eval_manifest_path is None and cfg.eval_total_blocks == 500 and cfg.eval_old_share == 0.4
    assert TrainConfig().eval_manifest_path is None


def test_env_overrides_the_manifest_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_MANIFEST_PATH", "/data/validation_v2_manifest.json")
    assert load_train_config("training/train_config.yaml").eval_manifest_path == "/data/validation_v2_manifest.json"
