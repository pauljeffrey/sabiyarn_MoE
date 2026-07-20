from training.load_config import load_train_config, sampling_weights


def test_sampling_config_defaults_match_yaml():
    cfg = load_train_config("training/train_config.yaml")
    assert cfg.use_scheduled_sampling is False
    assert cfg.eng_sampling_weight == 0.5
    assert cfg.afr_sampling_weight == 0.5


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
