"""Guess the position embeddings that training has not reached yet.

The model's wpe table has far more rows than any pretraining sequence ever used (pretraining runs at 4096 tokens,
the table has 32768 rows and SFT will use 65536): rows past the longest trained position still hold their random
initialisation. This module treats the trained rows as a multivariate time series indexed by position and
extrapolates it, so SFT at long context starts from rows that look like a continuation of the learned ones.

    1. PCA the trained rows (n_trained x d) down to k components -- the table is far lower-rank than d.
    2. A small neural net maps position -> the k PCA coordinates (one output per dim; position enters as a scalar
       plus multi-scale sin/cos features, so slow trends and periodic structure are both reachable).
    3. Predict the coordinates for the untrained positions and invert the PCA back to d dims.

Extrapolation is the hard part of any regression, so `fit` always holds out the tail of the trained range, predicts
it from the head, and reports the error next to two trivial baselines (the mean row; the last trained row). The
caller decides whether the regressor beat them -- `fill_untrained_positions` refuses to write the guesses unless it did.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class PCA:
    """Thin SVD PCA that can be inverted exactly (up to the dropped components)."""

    def __init__(self, n_components: Optional[int] = None, explained: float = 0.995, max_components: int = 128):
        self.n_components, self.explained, self.max_components = n_components, explained, max_components
        self.mean: Optional[torch.Tensor] = None
        self.components: Optional[torch.Tensor] = None      # (k, d)
        self.ratio: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "PCA":
        x = x.float()
        self.mean = x.mean(0, keepdim=True)
        _, s, vh = torch.linalg.svd(x - self.mean, full_matrices=False)
        var = s ** 2
        self.ratio = var / var.sum().clamp_min(1e-12)
        k = self.n_components
        if k is None:
            k = int((torch.cumsum(self.ratio, 0) < self.explained).sum().item()) + 1
        k = max(1, min(k, self.max_components, vh.shape[0]))
        self.components = vh[:k].contiguous()
        return self

    @property
    def k(self) -> int:
        return self.components.shape[0]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) @ self.components.T

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.components + self.mean


def estimate_trained_rows(wpe: torch.Tensor, init_std: float = 0.02, tol: float = 0.12, run: int = 64) -> int:
    """How many leading rows of a position table were actually trained? Untouched rows still hold the N(0, init_std) initialisation, whose
    L2 norm is init_std * sqrt(d) (within a few percent); trained rows drift away from it. Returns the length of the prefix that ends where the
    final run of >= `run` init-like rows starts (the table length if there is no such run)."""
    w = wpe.detach().float().cpu()
    expect = init_std * math.sqrt(w.shape[1])
    tol = max(tol, 4.0 / math.sqrt(2 * w.shape[1]))               # the norm of a random vector has relative spread 1/sqrt(2d)
    like = ((w.norm(dim=1) - expect).abs() / expect) < tol
    n = w.shape[0]
    i = n
    while i > 0 and like[i - 1]:
        i -= 1
    return i if n - i >= run else n


def position_features(pos: torch.Tensor, scale: float, n_freqs: int = 16) -> torch.Tensor:
    """(n,) positions -> (n, 1 + 2*n_freqs): the scaled position and sin/cos at geometrically spaced frequencies.

    Frequencies are in cycles per `scale` positions, from 0.5 (a slow half wave over the trained range) up to
    n_freqs/2 octaves above that, so both trends and fine periodic structure can be fitted."""
    x = pos.float().unsqueeze(1) / scale
    freqs = 0.5 * 2.0 ** (torch.arange(n_freqs, dtype=torch.float32) / 2.0)
    ang = 2 * math.pi * x * freqs.unsqueeze(0)
    return torch.cat([x, torch.sin(ang), torch.cos(ang)], dim=1)


class _Net(nn.Module):
    def __init__(self, d_in: int, k: int, hidden: int, layers: int):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, k))
        self.net = nn.Sequential(*mods)

    def forward(self, f):
        return self.net(f)


class PositionRegressor:
    def __init__(self, n_components: Optional[int] = None, explained: float = 0.995, hidden: int = 256, layers: int = 3,
                 n_freqs: int = 16, epochs: int = 600, lr: float = 2e-3, weight_decay: float = 1e-4, seed: int = 0):
        self.pca = PCA(n_components=n_components, explained=explained)
        self.hidden, self.layers, self.n_freqs = hidden, layers, n_freqs
        self.epochs, self.lr, self.weight_decay, self.seed = epochs, lr, weight_decay, seed
        self.scale = 1.0
        self.net: Optional[_Net] = None
        self.z_mean: Optional[torch.Tensor] = None
        self.z_std: Optional[torch.Tensor] = None

    def _fit_net(self, pos: torch.Tensor, z: torch.Tensor) -> None:
        torch.manual_seed(self.seed)
        self.z_mean, self.z_std = z.mean(0, keepdim=True), z.std(0, keepdim=True).clamp_min(1e-6)
        target = (z - self.z_mean) / self.z_std
        feats = position_features(pos, self.scale, self.n_freqs)
        net = _Net(feats.shape[1], z.shape[1], self.hidden, self.layers)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        # Per-dim loss weight = that component's share of the variance, so the net spends its capacity where the
        # table actually varies (the standardised targets would otherwise count a 0.1% component like the first one).
        w = (self.pca.ratio[: z.shape[1]] / self.pca.ratio[: z.shape[1]].sum()).unsqueeze(0) * z.shape[1]
        batch = min(1024, feats.shape[0])
        for _ in range(self.epochs):
            perm = torch.randperm(feats.shape[0])
            for i in range(0, feats.shape[0], batch):
                idx = perm[i:i + batch]
                loss = ((net(feats[idx]) - target[idx]) ** 2 * w).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        self.net = net.eval()

    def fit(self, trained: torch.Tensor, holdout_frac: float = 0.15) -> dict:
        """Fit on `trained` (n_trained, d); report held-out extrapolation error, then refit on everything."""
        trained = trained.detach().float().cpu()
        n, d = trained.shape
        cut = int(n * (1 - holdout_frac))
        head, tail = trained[:cut], trained[cut:]
        self.scale = float(cut)
        self.pca.fit(head)
        self._fit_net(torch.arange(cut), self.pca.transform(head))
        pred = self._predict_rows(torch.arange(cut, n))
        mse = ((pred - tail) ** 2).mean().item()
        base_mean = ((head.mean(0, keepdim=True) - tail) ** 2).mean().item()
        base_last = ((head[-1:] - tail) ** 2).mean().item()
        var_tail = tail.var(0).mean().item()
        report = {"n_trained": n, "n_holdout": n - cut, "pca_components": self.pca.k,
                  "pca_explained": float(self.pca.ratio[: self.pca.k].sum()),
                  "holdout_mse": mse, "baseline_mean_mse": base_mean, "baseline_last_mse": base_last,
                  "tail_variance": var_tail, "beats_baselines": bool(mse < min(base_mean, base_last))}
        self.scale = float(n)                     # final model sees the whole trained range
        self.pca.fit(trained)
        self._fit_net(torch.arange(n), self.pca.transform(trained))
        return report

    @torch.no_grad()
    def _predict_rows(self, pos: torch.Tensor) -> torch.Tensor:
        z = self.net(position_features(pos, self.scale, self.n_freqs)) * self.z_std + self.z_mean
        return self.pca.inverse(z)

    @torch.no_grad()
    def predict(self, positions: torch.Tensor) -> torch.Tensor:
        return self._predict_rows(positions.cpu())


def fill_untrained_positions(model, n_trained: int, end: Optional[int] = None, require_gain: bool = True,
                             **regressor_kwargs) -> dict:
    """Overwrite wpe rows [n_trained, end) with the regressor's extrapolation. `end` defaults to the table's length.

    Returns the fit report plus `written`. With require_gain (default) the rows are only written when the held-out
    test beat the baselines; otherwise the table is left alone."""
    wpe = model.transformer.wpe.weight
    end = wpe.shape[0] if end is None else min(end, wpe.shape[0])
    reg = PositionRegressor(**regressor_kwargs)
    report = reg.fit(wpe.data[:n_trained])
    if require_gain and not report["beats_baselines"]:
        report["written"] = 0
        return report
    rows = reg.predict(torch.arange(n_trained, end)).to(device=wpe.device, dtype=wpe.dtype)
    with torch.no_grad():
        wpe.data[n_trained:end] = rows
    report["written"] = int(end - n_trained)
    return report
