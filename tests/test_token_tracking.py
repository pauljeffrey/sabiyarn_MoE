"""Token accounting helpers on Trainer, exercised without building a real trainer/model."""
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# training.constant_tokens downloads the tokenizer at import time; stub it like the other tests do.
_stub = types.ModuleType("training.constant_tokens")
for _name, _val in dict(MASK=-100, assistant_token=1, end_of_text_token=2, system_token=3, user_token=4).items():
    setattr(_stub, _name, _val)
sys.modules["training.constant_tokens"] = _stub

from training.load_config import TrainConfig  # noqa: E402
from training.new_train import Trainer  # noqa: E402


def make_trainer(**kw):
    t = object.__new__(Trainer)
    t.cfg = TrainConfig(train_batch_size=6, block_size=4096, gradient_accumulation_steps=8, **kw)
    t.world_size = 4
    t.tokens_seen = 0
    t.tokens_seen_offset = 0
    t.tokens_seen_by_bin = {}
    t.master = True
    return t


def test_tokens_per_step_uses_the_per_rank_accumulation_times_world_size():
    t = make_trainer()
    assert t._tokens_per_step() == 6 * 4096 * 8 * 4 == 786_432


def test_max_tokens_derives_max_iters_from_real_tokens_per_step():
    t = make_trainer(max_tokens=50e9)
    t._resolve_max_iters()
    assert t.cfg.max_iters == -(-int(50e9) // 786_432)  # ceil


def test_log_fields_and_metrics_report_run_lifetime_and_per_bin_tokens():
    t = make_trainer()
    t.tokens_seen = 3_000_000_000
    t.tokens_seen_offset = 1_000_000_000
    t.tokens_seen_by_bin = {"eng_training": 2_000_000_000, "training_cleaned": 1_000_000_000}
    fields = t._token_log_fields()
    assert fields["tokens_seen"] == 3_000_000_000 and fields["tokens_seen_b"] == 3.0
    assert fields["lifetime_tokens_seen_b"] == 4.0
    m = t._token_metrics()
    assert m["train/tokens_seen"] == 3e9 and m["train/lifetime_tokens_seen"] == 4e9
    assert m["train/tokens_seen_eng_training"] == 2e9 and m["train/tokens_seen_training_cleaned"] == 1e9


def test_lifetime_fields_are_omitted_when_there_is_no_earlier_run():
    t = make_trainer()
    t.tokens_seen = 5
    assert "lifetime_tokens_seen_b" not in t._token_log_fields()
    assert "train/lifetime_tokens_seen" not in t._token_metrics()
