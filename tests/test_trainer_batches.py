"""Trainer.get_batch on tiny real memmaps: exactly-once epochs, fixed evals, resumable state."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_stub = types.ModuleType("training.constant_tokens")  # the real one downloads a tokenizer at import
for _n, _v in dict(MASK=-100, assistant_token=1, end_of_text_token=2, system_token=3, user_token=4).items():
    setattr(_stub, _n, _v)
sys.modules["training.constant_tokens"] = _stub

from training.load_config import TrainConfig  # noqa: E402
from training.new_train import Trainer  # noqa: E402

SL, BS = 8, 2


def write_bin(path, n_tokens):
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    arr[:] = np.arange(n_tokens, dtype=np.uint16)  # token value == position
    arr.flush()
    return str(path)


def make_trainer(tmp_path, n_eng=8 * 40 + 1, n_afr=8 * 20 + 1, n_eval=8 * 12 + 1):
    t = object.__new__(Trainer)
    t.cfg = TrainConfig(train_batch_size=BS, block_size=SL, use_loss_mask=False, seed=3,
                        eng_sampling_weight=0.5, afr_sampling_weight=0.5, use_scheduled_sampling=False)
    t.train_bins = [write_bin(tmp_path / "eng.bin", n_eng), write_bin(tmp_path / "afr.bin", n_afr)]
    t.eval_bin = write_bin(tmp_path / "val.bin", n_eval)
    t.accelerator = types.SimpleNamespace(process_index=0)
    t.world_size, t.master, t.device, t.iter_num = 1, False, "cpu", 0
    t._build_samplers()
    return t


def test_training_batches_cover_each_block_once_per_epoch_per_bin(tmp_path):
    t = make_trainer(tmp_path)  # eng: 40 blocks, afr: 20 blocks; 50/50 mix, 2 blocks per micro-step
    seen = {"eng": [], "afr": []}
    for _ in range(30):  # 15 micro-steps from each bin = 30 blocks each -> eng 0.75 epoch, afr 1.5 epochs
        x, y = t.get_batch("train", track=True)
        assert x.shape == (BS, SL) and torch.all(y == x + 1)
        seen[t._last_train_bin].extend((x[:, 0] // SL).tolist())  # block index from the first token
    eng, afr = seen["eng"], seen["afr"]
    assert len(eng) == len(set(eng)) == 30  # no block repeated inside the first epoch of eng
    assert sorted(afr[:20]) == list(range(20))  # afr: the complete first epoch, each block exactly once
    assert len(set(afr[20:])) == 10  # ...then a reshuffled second epoch begins
    assert t.sampler.epochs_done()["afr"] == pytest.approx(1.5)


def test_eval_batches_are_fixed_and_leave_the_training_stream_untouched(tmp_path):
    t = make_trainer(tmp_path)
    before = t.sampler.state_dict()

    def run_eval():
        out = []
        for split in ("train", "val"):
            (t._eval_train_sampler if split == "train" else t._eval_val_sampler).reset()
            out.append([t.get_batch(split)[0].tolist() for _ in range(4)])
        return out

    assert run_eval() == run_eval()  # same windows every eval
    assert t.sampler.state_dict() == before  # training position never moved by evaluating


def test_sampler_state_survives_a_restart(tmp_path):
    t1 = make_trainer(tmp_path)
    for _ in range(7):
        t1.get_batch("train", track=True)
    state = t1.sampler.state_dict()
    expected = [t1.get_batch("train", track=True)[0].tolist() for _ in range(5)]

    t2 = make_trainer(tmp_path)
    assert t2.sampler.load_state_dict(state) == []
    assert [t2.get_batch("train", track=True)[0].tolist() for _ in range(5)] == expected


def test_checkpoints_saved_from_the_local_model_class_ship_their_code_and_auto_map(tmp_path):
    """Regression guard: without register_local_model_code(), save_pretrained() writes only weights and
    a config without auto_map -- pushing that to the Hub breaks trust_remote_code loading."""
    import json

    from sabiyarn.model.configuration import GPTJXMoEConfig
    from sabiyarn.model.modeling import GPTJXMoEForCausalLM
    from training.new_train import register_local_model_code

    register_local_model_code()
    cfg = GPTJXMoEConfig(block_size=64, vocab_size=100, n_layer=2, n_heads=2, n_embd=32, use_moe=True,
                         num_experts=4, num_experts_per_tok=2, moe_dim=64, expert_per_layer={"0": 2, "1": 3})
    GPTJXMoEForCausalLM(cfg).save_pretrained(tmp_path)
    files = {p.name for p in tmp_path.iterdir()}
    assert {"modeling.py", "configuration.py", "config.json", "model.safetensors"} <= files
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["auto_map"]["AutoModelForCausalLM"] == "modeling.GPTJXMoEForCausalLM"
    assert saved["moe_sparse_dispatch"] is True
    assert "_sparse_experts" in (tmp_path / "modeling.py").read_text()


def test_scheduled_sampling_shifts_bin_proportions_through_the_trainer(tmp_path):
    """use_scheduled_sampling end to end: the afr share of drawn batches follows the schedule as iter_num
    advances, with no block repeated inside an epoch and the eval probe unaffected."""
    t = make_trainer(tmp_path, n_eng=8 * 4000 + 1, n_afr=8 * 4000 + 1)
    t.cfg.use_scheduled_sampling = True
    t.cfg.max_iters = 1000
    t.cfg.sampling_schedule = ((0.0, 0.2), (1.0, 0.8))

    def afr_share(it, n=400):
        t.iter_num = it
        return sum(t.get_batch("train", track=True) is not None and t._last_train_bin == "afr" for _ in range(n)) / n

    early, late = afr_share(0), afr_share(1000)
    assert abs(early - 0.2) < 0.02 and abs(late - 0.8) < 0.02  # credit sampler tracks the weights tightly
    # eval probe keeps the fixed start mixture regardless of iter_num
    assert t._eval_bin_weights() == (0.5, 0.5)
    t.iter_num = 1000
    assert t._eval_bin_weights() == (0.5, 0.5)
