"""resize_for_sft + the position regressor, on a stand-in model (only the attributes resize.py touches)."""
import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from sabiyarn.model.position_regressor import PCA, PositionRegressor, fill_untrained_positions  # noqa: E402
from sabiyarn.model.resize import build_piece_map, resize_for_sft, resize_position_embeddings, round_up, sft_vocab_size  # noqa: E402


def _model(vocab=100, block=64, d=16, tied=True):
    wte = nn.Embedding(vocab, d)
    head = nn.Linear(d, vocab, bias=False)
    if tied:
        head.weight = wte.weight
    cfg = types.SimpleNamespace(vocab_size=vocab, block_size=block, max_position_embeddings=block, tie_word_embeddings=tied)
    tr = nn.ModuleDict(dict(wte=wte, wpe=nn.Embedding(block, d), h=nn.ModuleList()))
    return types.SimpleNamespace(transformer=tr, lm_head=head, config=cfg)


def test_vocab_is_multiple_of_64():
    assert sft_vocab_size() == 52160 and round_up(52117) == 52160 and round_up(52160) == 52160


def test_resize_keeps_old_rows_and_ties_head():
    m = _model()
    old_w, old_p = m.transformer.wte.weight.data.clone(), m.transformer.wpe.weight.data.clone()
    r = resize_for_sft(m, vocab_size=128, block_size=256)
    assert r["tokens_added"] == 28 and r["positions_added"] == 192
    assert m.transformer.wte.weight.shape == (128, 16) and m.lm_head.weight.shape == (128, 16)
    assert m.lm_head.weight is m.transformer.wte.weight
    assert torch.equal(m.transformer.wte.weight[:100], old_w) and torch.equal(m.transformer.wpe.weight[:64], old_p)
    assert m.config.vocab_size == 128 and m.config.block_size == 256 and m.config.max_position_embeddings == 256
    assert resize_for_sft(m, vocab_size=128, block_size=256)["tokens_added"] == 0       # idempotent


def test_untied_head_is_grown_too():
    m = _model(tied=False)
    resize_for_sft(m, vocab_size=128, block_size=64)
    assert m.lm_head.weight.shape == (128, 16) and m.lm_head.weight is not m.transformer.wte.weight


def test_bad_vocab_rejected():
    with pytest.raises(ValueError):
        resize_for_sft(_model(), vocab_size=100)


def test_pca_roundtrip_low_rank():
    torch.manual_seed(0)
    x = torch.randn(300, 4) @ torch.randn(4, 20)
    p = PCA(explained=0.9999).fit(x)
    assert p.k <= 5
    assert (p.inverse(p.transform(x)) - x).abs().max() < 1e-3


def test_regressor_extrapolates_smooth_table_better_than_baselines():
    torch.manual_seed(0)
    n, d = 600, 24
    t = torch.arange(n).float().unsqueeze(1) / n
    basis = torch.randn(3, d)
    table = torch.cat([t, torch.sin(2 * 3.14159 * t), torch.cos(2 * 3.14159 * 2 * t)], 1) @ basis
    rep = PositionRegressor(epochs=250, hidden=64, layers=2).fit(table)
    assert rep["pca_components"] <= 6
    assert rep["beats_baselines"], rep


def test_fill_leaves_table_alone_without_signal():
    torch.manual_seed(0)
    m = _model(vocab=10, block=400, d=16)
    m.transformer.wpe.weight.data.normal_(0, 0.02)           # pure noise: nothing to extrapolate
    before = m.transformer.wpe.weight.data.clone()
    rep = fill_untrained_positions(m, n_trained=200, epochs=40, hidden=32, layers=2)
    assert (not rep["beats_baselines"]) == (rep["written"] == 0)
    if rep["written"] == 0:
        assert torch.equal(m.transformer.wpe.weight.data, before)


def test_estimate_trained_rows_finds_where_training_stopped():
    torch.manual_seed(0)
    d, n, trained = 64, 1000, 300
    w = torch.randn(n, d) * 0.02                                  # initialisation
    w[:trained] = torch.randn(trained, d) * 0.09 + 0.03           # rows that moved
    from sabiyarn.model.position_regressor import estimate_trained_rows
    assert abs(estimate_trained_rows(w) - trained) <= 8
    assert estimate_trained_rows(torch.randn(n, d) * 0.09) == n    # everything trained: no init-like tail


def test_new_tokens_start_as_the_mean_of_their_pieces():
    m = _model(vocab=100)
    old_w = m.transformer.wte.weight.data.clone()

    class New:                                       # ids 100/101 are new: '"ab"' and '<x>'; 102 has no usable pieces
        names = {100: '"ab"', 101: "<x>", 102: "???"}
        convert_ids_to_tokens = lambda self, i: self.names[i]

    class Old:                                       # the checkpoint-era tokenizer splits them into ids that exist
        def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
            assert split_special_tokens
            return {"input_ids": {'"ab"': [3, 4, 5], "<x>": [6, 7], "???": [500]}[text]}     # 500 is not a checkpoint row

    pm = build_piece_map(Old(), New(), 100, 102)
    assert pm == {100: [3, 4, 5], 101: [6, 7]}
    resize_for_sft(m, vocab_size=128, block_size=64, piece_map=pm)
    w = m.transformer.wte.weight.data
    assert torch.allclose(w[100], old_w[[3, 4, 5]].mean(0), atol=1e-6) and torch.allclose(w[101], old_w[[6, 7]].mean(0), atol=1e-6)
    assert (w[102] - old_w.mean(0)).abs().max() > 0                                  # noise was added
