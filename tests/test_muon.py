import math

import pytest

torch = pytest.importorskip("torch")

from training.muon import Muon, newton_schulz, split_muon_params

D = 32


def singular_values(x):
    return torch.linalg.svdvals(x.float())


def test_newton_schulz_orthogonalizes_wide_tall_and_batched():
    torch.manual_seed(0)
    for shape in [(D, D), (D, 4 * D), (4 * D, D), (3, D, D), (4, D, 3 * D)]:
        out = newton_schulz(torch.randn(*shape))
        assert out.shape == shape
        sv = singular_values(out)
        # 5 quintic steps push the bulk of the spectrum to ~1 (0.7-1.2) and cap the top; the
        # smallest singular values of a random square matrix are only lifted, not fixed.
        assert sv.max() < 1.3, (shape, sv.max().item())
        assert 0.6 < sv.median() < 1.2, (shape, sv.median().item())


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = torch.nn.Linear(D, 3 * D, bias=False)
        self.c_proj = torch.nn.Linear(D, D, bias=False)
        self.fc_bank = torch.nn.Parameter(torch.randn(4, D, 2 * D) * 0.02)
        self.gate = torch.nn.Linear(D, 4, bias=False)
        self.ln = torch.nn.LayerNorm(D)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(50, D)
        self.wpe = torch.nn.Embedding(16, D)
        self.h = TinyBlock()
        self.lm_head = torch.nn.Linear(D, 50, bias=False)


def test_param_split_sends_only_hidden_matrices_to_muon():
    model = TinyModel()
    muon, adam_2d, adam_1d = split_muon_params(model, D)
    names = {id(p): n for n, p in model.named_parameters()}
    assert {names[id(p)] for p in muon} == {"h.c_attn.weight", "h.c_proj.weight", "h.fc_bank"}
    assert {names[id(p)] for p in adam_2d} == {"wte.weight", "wpe.weight", "h.gate.weight", "lm_head.weight"}
    assert {names[id(p)] for p in adam_1d} == {"h.ln.weight", "h.ln.bias"}


def make_optimizer(model, **kw):
    muon, adam_2d, adam_1d = split_muon_params(model, D)
    return Muon(
        [{"params": muon, "use_muon": True}, {"params": adam_2d}, {"params": adam_1d, "weight_decay": 0.0}],
        n_embd=D, lr=kw.pop("lr", 1e-2), weight_decay=kw.pop("weight_decay", 0.0), **kw,
    )


def test_update_rms_is_matched_to_adamw_and_updates_are_finite():
    torch.manual_seed(0)
    model = TinyModel()
    opt = make_optimizer(model, lr=1e-2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    for name in ("h.c_proj.weight", "h.fc_bank", "h.c_attn.weight"):
        delta = dict(model.named_parameters())[name].detach() - before[name]
        rms = delta.pow(2).mean().sqrt().item() / 1e-2  # update RMS in units of lr
        assert 0.1 < rms < 0.3, (name, rms)  # ~0.2 by construction, ortho factor has RMS <= 1/sqrt(max(m,n))*sqrt(min)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_fused_qkv_is_orthogonalized_as_three_separate_matrices():
    opt = make_optimizer(TinyModel())
    g = torch.randn(3 * D, D)
    stacked = opt._matrices(g)
    assert stacked.shape == (3, D, D)
    assert torch.equal(stacked[1], g[D : 2 * D])


def test_weight_decay_is_decoupled_and_skips_1d_params():
    torch.manual_seed(0)
    model = TinyModel()
    opt = make_optimizer(model, lr=0.1, weight_decay=0.5)
    ln_before = model.h.ln.weight.detach().clone()
    proj_before = model.h.c_proj.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.equal(model.h.ln.weight, ln_before)  # wd 0 group, zero grad -> untouched
    assert torch.allclose(model.h.c_proj.weight, proj_before * (1 - 0.1 * 0.5))  # only the decay term


def test_optimizer_reduces_a_toy_loss_and_state_round_trips():
    torch.manual_seed(0)
    model = TinyModel()
    opt = make_optimizer(model, lr=2e-2)
    x = torch.randint(0, 50, (8, 16))

    def loss_fn():
        h = model.wte(x) + model.wpe(torch.arange(16))
        h = model.h.c_proj(h) + torch.einsum("btd,edh->bteh", h, model.h.fc_bank).mean(2)[..., :D]
        return torch.nn.functional.cross_entropy(model.lm_head(h).view(-1, 50), x.view(-1))

    first = loss_fn().item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert loss_fn().item() < first * 0.8

    state = opt.state_dict()
    opt2 = make_optimizer(model, lr=2e-2)
    opt2.load_state_dict(state)  # same structure -> loads
