"""Trainer-level checks for the 3-bin mixture, zero-weight bins, token accounting and the stratified
eval -- on tiny real memmaps, with the model forward faked, so no GPU is needed."""
import json
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

import training.new_train as nt  # noqa: E402
from training.load_config import TrainConfig  # noqa: E402
from training.new_train import Trainer  # noqa: E402
from training.token_accounting import record_micro_batch  # noqa: E402

SL, BS = 8, 2


def write_bin(path, n_tokens, fill=None):
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    arr[:] = np.arange(n_tokens, dtype=np.uint16) if fill is None else fill
    arr.flush()
    return str(path)


def make_trainer(tmp_path, monkeypatch, *, eng=0.15, afr=0.55, other=0.30, other_exists=True, mode="pretrain",
                 paths=3, **cfg_kw):
    names = ["eng_training", "training_cleaned", "other_african_pretrain"][:paths]
    files = []
    for i, n in enumerate(names):
        p = tmp_path / f"{n}.bin"
        if i < 2 or other_exists:
            write_bin(p, SL * 3000 + 1)
        files.append(str(p))
    val = write_bin(tmp_path / "validation.bin", SL * 100 + 1)
    t = object.__new__(Trainer)
    t.cfg = TrainConfig(
        mode=mode, train_batch_size=BS, block_size=SL, use_loss_mask=False, seed=3, eng_sampling_weight=eng,
        afr_sampling_weight=afr, other_afr_sampling_weight=other, train_data_paths=files, eval_data_path=val,
        tokenizer_name="x", **cfg_kw,
    )
    t.accelerator = types.SimpleNamespace(process_index=0)
    t.eval_bin = val
    t.world_size, t.master, t.device, t.iter_num = 1, False, "cpu", 0
    t._bin_index, t._eval_plan, t._eval_sanity_limit = None, None, None
    monkeypatch.setattr(nt, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: None))
    t._finalize_batch = lambda x, y: (x, y)  # keep the numpy windows; no device transfer needed here
    opened = []
    real = t._read_memmap
    t._read_memmap = lambda p: (opened.append(p), real(p))[1]
    t._opened = opened
    return t


def draw(t, n):
    counts, by_bin, seen = {}, {}, 0
    for _ in range(n):
        x, _y = t.get_batch("train", track=True)
        counts[t._last_train_bin] = counts.get(t._last_train_bin, 0) + 1
        seen += record_micro_batch(by_bin, t._last_train_bin, x.shape[0] * x.shape[1] * t.world_size)
    return counts, by_bin, seen


def test_three_bin_mixture_follows_the_weights_and_names_the_new_bin(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch)
    t._setup_data()
    assert [Path(p).stem for p in t.train_bins] == ["eng_training", "training_cleaned", "other_african_pretrain"]
    assert t.sampler.names[2] == "other_african_pretrain"
    assert t._bin_weights() == pytest.approx((0.15, 0.55, 0.30))
    counts, by_bin, seen = draw(t, 1000)
    assert counts["eng_training"] / 1000 == pytest.approx(0.15, abs=0.01)
    assert counts["training_cleaned"] / 1000 == pytest.approx(0.55, abs=0.01)
    assert counts["other_african_pretrain"] / 1000 == pytest.approx(0.30, abs=0.01)
    assert seen == 1000 * BS * SL and sum(by_bin.values()) == seen  # tokens_seen == sum of the per-bin counts
    assert by_bin["other_african_pretrain"] == counts["other_african_pretrain"] * BS * SL


def test_zero_weight_third_bin_is_never_opened_never_drawn_and_may_be_missing(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch, eng=0.15, afr=0.85, other=0.0, other_exists=False)
    t._setup_data()  # must not raise although other_african_pretrain.bin does not exist
    assert len(t.train_bins) == 2 and not any("other_african" in p for p in t._opened)
    assert t._bin_weights() == pytest.approx((0.15, 0.85)) and t._eval_bin_weights() == pytest.approx((0.15, 0.85))
    counts, _, _ = draw(t, 400)
    assert set(counts) <= {"eng_training", "training_cleaned"}
    assert t._sampling_weights() == pytest.approx((0.15, 0.85, 0.0))  # the step log still shows three numbers


def test_a_zero_weight_middle_bin_is_dropped_and_the_rest_renormalised(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch, eng=0.4, afr=0.0, other=0.6)
    t._setup_data()
    assert [Path(p).stem for p in t.train_bins] == ["eng_training", "other_african_pretrain"]
    assert t._bin_weights() == pytest.approx((0.4, 0.6))
    assert t._sampling_weights() == pytest.approx((0.4, 0.0, 0.6))


def test_scheduled_three_bin_mixture_moves_with_iter_num(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch, use_scheduled_sampling=True, max_iters=1000,
                     sampling_schedule=((0.0, 0.5, 0.1), (1.0, 0.5, 0.3)))
    t._setup_data()
    t.iter_num = 0
    assert t._bin_weights() == pytest.approx((0.4, 0.5, 0.1))
    t.iter_num = 1000
    assert t._bin_weights() == pytest.approx((0.2, 0.5, 0.3))
    assert t._eval_bin_weights() == pytest.approx((0.15, 0.55, 0.30))  # the probe stays on the preset


def test_stale_two_path_env_with_a_positive_third_weight_fails_loudly_in_pretrain_only(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch, paths=2)  # other weight 0.30 but only two paths configured
    with pytest.raises(ValueError, match="re-run the prefetch"):
        t._setup_data()
    t = make_trainer(tmp_path, monkeypatch, paths=2, mode="sft")
    t._setup_data()  # SFT has no third bin: the shared yaml's weight is ignored, not fatal
    assert len(t.train_bins) == 2 and t._bin_weights() == pytest.approx((0.15 / 0.70, 0.55 / 0.70))


def test_single_path_ignores_the_weights(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch, paths=1)
    t._setup_data()
    assert len(t.train_bins) == 1 and t._bin_weights() == (1.0,) and t._eval_bin_weights() == (1.0,)


# ---------------------------------------------------------------- stratified eval

def make_eval_trainer(tmp_path, monkeypatch, world=1, rank=0, total_blocks=40):
    """Validation bin = 'old' (100 blocks, value 1) + zul (30 blocks, value 2) + yor (4 blocks, value 3);
    the faked loss of a micro-batch is the value of its tokens, so every segment's loss is known."""
    pieces = [("old", SL * 100 + 1, 1), ("zul", SL * 30 + 1, 2), ("yor", SL * 4 + 1, 3)]
    data = np.concatenate([np.full(n, v, dtype=np.uint16) for _, n, v in pieces])
    val = write_bin(tmp_path / "validation_v2.bin", len(data), fill=data)
    segs, cur = [], 0
    for name, n, _ in pieces:
        segs.append({"name": name, "start": cur, "end": cur + n})
        cur += n
    man = tmp_path / "validation_v2_manifest.json"
    man.write_text(json.dumps({"dtype": "uint16", "total_tokens": cur, "segments": segs}))
    t = make_trainer(tmp_path, monkeypatch)
    t.cfg.eval_manifest_path, t.cfg.eval_total_blocks, t.cfg.eval_old_share = str(man), total_blocks, 0.5
    t.cfg.eval_data_path = val
    t.eval_bin = val
    t.world_size = world
    t.accelerator = types.SimpleNamespace(process_index=rank, reduce=lambda x, reduction: x)
    return t


def test_manifest_builds_a_fixed_plan_and_the_sanity_batch_stays_inside_the_old_segment(tmp_path, monkeypatch):
    t = make_eval_trainer(tmp_path, monkeypatch)
    t.train_bins = [t.cfg.train_data_paths[0]]
    t._build_samplers()
    plan = t._eval_plan
    assert plan.n_blocks == 40 and plan.counts["old"] == 20 and plan.counts["yor"] == 4  # tiny yor capped
    assert plan.counts["zul"] == 16
    assert t._eval_sanity_limit == SL * 100 + 1
    _, ix = t._sanity_indices_and_data()
    assert ix.max() < SL * 100 + 1 - SL - 1
    # and the same indices the old, un-segmented validation.bin of that length would have given
    rng = np.random.default_rng(1234567)
    assert ix.tolist() == rng.integers(0, SL * 100 + 1 - SL - 1, size=BS).tolist()
    t._build_samplers()
    assert t._eval_plan.starts == plan.starts  # deterministic across rebuilds


def test_no_manifest_keeps_the_original_eval(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch)
    t.train_bins = [t.cfg.train_data_paths[0]]
    t._build_samplers()
    assert t._eval_plan is None and t._eval_sanity_limit is None


def test_a_manifest_that_does_not_match_the_bin_fails_at_startup(tmp_path, monkeypatch):
    t = make_eval_trainer(tmp_path, monkeypatch)
    Path(t.cfg.eval_manifest_path).write_text(json.dumps(
        {"dtype": "uint16", "total_tokens": 10, "segments": [{"name": "old", "start": 0, "end": 10}]}))
    t.train_bins = [t.cfg.train_data_paths[0]]
    with pytest.raises(ValueError, match="do not belong together"):
        t._build_samplers()


@pytest.mark.parametrize("world", [1, 2])
def test_stratified_eval_reports_per_segment_and_overall_losses_across_ranks(tmp_path, monkeypatch, world):
    ranks = []
    for r in range(world):
        t = make_eval_trainer(tmp_path, monkeypatch, world=world, rank=r)
        t.train_bins = [t.cfg.train_data_paths[0]]
        t._build_samplers()
        # the fake forward sets _last_ce_loss = loss - 0.5 so CE and loss differ
        def fwd(x, y, t=t):
            loss = torch.tensor(float(x[0, 0]))
            t._last_ce_loss = loss - 0.5
            return loss
        t._forward_loss = fwd
        t._target_stats = lambda y: (torch.tensor(float(y.shape[0] * y.shape[1])), torch.tensor(2.0 * y.shape[0] * y.shape[1]))
        ranks.append(t)
    sums = []
    for t in ranks[1:]:  # run the other ranks first, capturing the totals they would all-reduce
        t.accelerator.reduce = lambda x, reduction: (sums.append(x), x)[1]
        t._estimate_val_stratified(t._eval_plan)
    ranks[0].accelerator.reduce = lambda x, reduction: sum([x] + sums)
    out = ranks[0]._estimate_val_stratified(ranks[0]._eval_plan)
    counts = ranks[0]._eval_plan.counts
    assert out["val_old"] == pytest.approx(1.0)
    assert out["val_by_lang"] == {"zul": pytest.approx(2.0), "yor": pytest.approx(3.0)}
    assert out["val_lang_macro"] == pytest.approx(2.5)  # unweighted over languages
    n = sum(counts.values())
    assert out["val"] == pytest.approx((1.0 * counts["old"] + 2.0 * counts["zul"] + 3.0 * counts["yor"]) / n)
    assert out["val_ce"] == pytest.approx(out["val"] - 0.5)
    assert set(out["val_bpb_by_lang"]) == {"zul", "yor"}


def test_val_metrics_use_the_documented_names():
    m = Trainer._val_metrics({"val": 2.0, "val_old": 1.9, "val_lang_macro": 3.0,
                              "val_by_lang": {"zul": 3.5, "yor": 2.5}, "val_bpb_by_lang": {"zul": 1.1}})
    assert m == {"val_loss": 2.0, "val_loss_old": 1.9, "val_loss_lang_macro": 3.0,
                 "val_loss/zul": 3.5, "val_loss/yor": 2.5, "val_bpb/zul": 1.1}
    assert Trainer._val_metrics({"val": 2.0}) == {"val_loss": 2.0}  # original eval: only val_loss
