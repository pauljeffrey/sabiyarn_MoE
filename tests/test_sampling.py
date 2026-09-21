from training.load_config import load_train_config, sampling_weights


def test_sampling_config_defaults_match_yaml():
    cfg = load_train_config("training/train_config.yaml")
    # Whether scheduled sampling is on is a tuning choice that changes (the yaml has it on), so
    # only assert it parses as a boolean, like the weights below.
    assert isinstance(cfg.use_scheduled_sampling, bool)
    # Exact weights are under active tuning -- assert they're a valid,
    # normalized eng/afr split rather than pinning literal values.
    assert abs((cfg.eng_sampling_weight + cfg.afr_sampling_weight) - 1.0) < 1e-9
    assert 0.0 <= cfg.eng_sampling_weight <= 1.0
    assert 0.0 <= cfg.afr_sampling_weight <= 1.0


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
