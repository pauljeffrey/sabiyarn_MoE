import math
from types import SimpleNamespace

import pytest

from training.lr_schedule import lr_at


def cfg(**kw):
    base = dict(learning_rate=1e-3, min_lr=0.0, warmup_iters=100, lr_decay_iters=1000, max_iters=1000,
                scheduler="wsd", wsd_decay_frac=0.2, wsd_decay_shape="sqrt")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("scheduler", ["wsd", "linear", "cosine"])
def test_warmup_is_linear_from_zero_for_every_scheduler(scheduler):
    c = cfg(scheduler=scheduler)
    assert lr_at(0, c) == 0.0
    assert lr_at(50, c) == pytest.approx(0.5e-3)
    assert lr_at(100, c) == pytest.approx(1e-3)


def test_wsd_holds_peak_then_cools_to_min_lr_with_sqrt_shape():
    c = cfg()
    assert lr_at(500, c) == 1e-3 and lr_at(800, c) == 1e-3  # stable until 80%
    mid = lr_at(900, c)  # half-way through the cooldown: 1 - sqrt(0.5)
    assert mid == pytest.approx(1e-3 * (1 - math.sqrt(0.5)))
    assert lr_at(1000, c) == 0.0 and lr_at(5000, c) == 0.0


def test_wsd_linear_shape_and_nonzero_floor():
    c = cfg(wsd_decay_shape="linear", min_lr=1e-4)
    assert lr_at(900, c) == pytest.approx(1e-4 + 0.5 * (1e-3 - 1e-4))
    assert lr_at(1000, c) == pytest.approx(1e-4)


def test_linear_decays_straight_to_zero():
    c = cfg(scheduler="linear")
    assert lr_at(550, c) == pytest.approx(0.5e-3)
    assert lr_at(1000, c) == pytest.approx(0.0)


def test_cosine_matches_the_previous_behaviour():
    c = cfg(scheduler="cosine", min_lr=1e-4)
    assert lr_at(550, c) == pytest.approx(1e-4 + 0.5 * (1 + math.cos(math.pi * 0.5)) * (1e-3 - 1e-4))
    assert lr_at(2000, c) == 1e-4
