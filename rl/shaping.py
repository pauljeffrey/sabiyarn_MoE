"""Reward shaping for the QE content score: teach "avoid failures" AND "good versus great".

The content score c (AfriCOMET-QE, roughly 0-1) is mapped through a piecewise-linear curve with FOUR regions, defined by three anchors
per language pair:

      c < t_fail            failure zone: -2 at c = 0 rising to 0 at t_fail
      t_fail .. t_mid       acceptable -> good:   0 .. +1
      t_mid  .. t_great     good -> great:       +1 .. +2
      c > t_great           beyond great: tiny slope, +2 .. +2.25 at c = 1 (still ranks, but a score above what human translations get
                            earns almost nothing: that region is where a learned metric is gamed, not where translations get better)
t_fail is never allowed below FAIL_FLOOR (0.46) -- a human-reference P10 that low would mean the metric is unreliable for that pair.

The anchors are the whole point, and a single fixed number (the old 0.55) cannot be right for every language pair, because
AfriCOMET-QE scores differ by pair. They are measured, not guessed (rlhf/scripts/calibrate_shaping.py):
      t_fail   = P10 of QE scores of HUMAN reference translations on the benchmark   (even good human output rarely scores below it)
      t_mid    = P50 of the same
      t_great  = P90 of the same
so "great" means "as good as a human translation scores", "failure" means "worse than nearly all human translations". Pairs without a
calibration use `DEFAULT_ANCHORS`.

`adaptive`: t_fail stays anchored to the calibration (an absolute floor of acceptability), while t_mid and t_great follow rolling
quantiles (P50 / P90) of the policy's OWN recent scores for that pair, so the bar rises as the model improves and the curve keeps its
resolution where the policy actually is. Before `warmup` scores have been seen for a pair the calibrated anchors are used.

All maps are monotone increasing, so the ranking of completions inside a group never changes; only the spacing does.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Optional

import numpy as np

# AfriCOMET-QE below ~0.46 is gibberish with omissions (measured by hand by the project owner), so 0.46 is the failure line: the reward is
# negative below it and positive above it. The upper anchors are placeholders until calibrate_shaping.py measures them per pair.
FAIL_FLOOR = 0.46
DEFAULT_ANCHORS = (FAIL_FLOOR, 0.60, 0.75)  # (t_fail, t_mid, t_great)
Y = (0.0, 1.0, 2.0, 2.25)                   # reward at t_fail, t_mid, t_great, and at c = 1   (the -2 .. +2 scale)


def make_curve(anchors: tuple[float, float, float], neg_scale: float = 2.0):
    t_fail, t_mid, t_great = anchors
    t_fail = max(t_fail, FAIL_FLOOR)
    eps = 1e-3                                   # keep the knots strictly increasing whatever the calibration says
    t_mid = max(t_mid, t_fail + eps)
    t_great = max(t_great, t_mid + eps)
    xs = [0.0, t_fail, t_mid, t_great, max(1.0, t_great + eps)]
    ys = [-neg_scale, *Y]
    return xs, ys


def shape(c: float, anchors: tuple[float, float, float] = DEFAULT_ANCHORS, neg_scale: float = 2.0) -> float:
    xs, ys = make_curve(anchors, neg_scale)
    if c <= xs[0]:
        return ys[0]
    if c >= xs[-1]:
        return ys[-1]
    return float(np.interp(c, xs, ys))


class ContentShaper:
    """mode: calibrated | adaptive. `calibration` maps "src>tgt" -> [t_fail, t_mid, t_great] (see calibrate_shaping.py)."""

    def __init__(self, mode: str = "calibrated", calibration: Optional[dict] = None, neg_scale: float = 2.0, window: int = 2000,
                 warmup: int = 200, ema: float = 0.9, p_mid: float = 0.5, p_great: float = 0.9,
                 default_anchors: tuple[float, float, float] = DEFAULT_ANCHORS):
        self.default = tuple(default_anchors)
        self.mode, self.cal, self.neg, self.warmup, self.ema = mode, calibration or {}, neg_scale, warmup, ema
        self.p_mid, self.p_great = p_mid, p_great
        self._win: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._live: dict[str, tuple[float, float]] = {}

    @classmethod
    def from_file(cls, path: Optional[str], **kw) -> "ContentShaper":
        """Calibration file written by rlhf/scripts/calibrate_shaping.py: "src>tgt" -> {anchors, usable, reliability, ...}."""
        import os
        import sys
        cal, rel, unusable = {}, {}, set()
        if path and not os.path.exists(path):
            print(f"[shaping] calibration file {path!r} not found: using the default anchors for every pair "
                  f"(run rlhf/scripts/calibrate_shaping.py first)", file=sys.stderr)
            path = None
        if path:
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if isinstance(v, dict):
                        if not v.get("usable", True):
                            unusable.add(k)
                            continue
                        cal[k] = tuple(v["anchors"])
                        rel[k] = float(v.get("reliability", 1.0))
                    else:
                        cal[k] = tuple(v)
        sh = cls(calibration=cal, **kw)
        sh.rel, sh.unusable = rel, unusable
        return sh

    rel: dict = {}
    unusable: set = set()

    def is_usable(self, key: str) -> bool:
        return key not in self.unusable

    def reliability(self, key: str) -> float:
        """How much to trust QE on this pair, in [0, 1] (from its measured AUC). Unknown pairs get 1.0."""
        return 0.0 if key in self.unusable else float(self.rel.get(key, 1.0))

    def anchors(self, key: str) -> tuple[float, float, float]:
        base = tuple(self.cal.get(key) or self.cal.get("*") or self.default)
        if self.mode == "adaptive" and key in self._live:
            mid, great = self._live[key]
            return (max(base[0], FAIL_FLOOR), max(mid, base[0] + 1e-3), max(great, mid + 1e-3))
        return (max(base[0], FAIL_FLOOR), base[1], base[2])

    def observe(self, key: str, scores: list[float]) -> None:
        """adaptive only: feed the raw QE scores of this batch (gate-open rows)."""
        if self.mode != "adaptive" or not scores:
            return
        w = self._win[key]
        w.extend(scores)
        if len(w) >= self.warmup:
            mid, great = (float(x) for x in np.quantile(np.fromiter(w, float), [self.p_mid, self.p_great]))
            old = self._live.get(key)
            self._live[key] = (mid, great) if old is None else (self.ema * old[0] + (1 - self.ema) * mid,
                                                                  self.ema * old[1] + (1 - self.ema) * great)

    def __call__(self, c: float, key: str) -> float:
        return shape(c, self.anchors(key), self.neg)

    def state(self) -> dict:
        return {k: {"t_mid": v[0], "t_great": v[1], "n": len(self._win[k])} for k, v in self._live.items()}


# ------------------------------------------------------------------------------------------- calibration helpers
def compute_anchors(scores: list[float], q=(0.10, 0.50, 0.90)) -> tuple[float, float, float]:
    a = np.quantile(np.asarray(scores, dtype=float), q)
    return (float(a[0]), float(a[1]), float(a[2]))


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score(pos) > score(neg)): how well the metric separates real translations from mismatched ones (0.5 = chance)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos), np.asarray(neg)
    order = np.argsort(np.concatenate([p, n]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def reliability_from_auc(a: float, low: float = 0.65, high: float = 0.90) -> float:
    """Linear trust in the metric for a pair: AUC <= 0.65 -> 0, AUC >= 0.90 -> 1."""
    return float(min(1.0, max(0.0, (a - low) / (high - low))))


def pair_anchors(human: list[float], mismatched: list[float], floor: float = FAIL_FLOOR) -> tuple[float, float, float]:
    """t_fail = max(floor, P90 of QE on mismatched-but-fluent translations): a fluent wrong sentence must never land above the failure line;
    t_mid = P50 and t_great = P90 of QE on human references (kept a margin above the previous anchor)."""
    m90 = float(np.quantile(np.asarray(mismatched, dtype=float), 0.90)) if len(mismatched) else floor
    t_fail = max(floor, m90)
    h = np.quantile(np.asarray(human, dtype=float), [0.5, 0.9])
    t_mid = max(float(h[0]), t_fail + 0.05)
    t_great = max(float(h[1]), t_mid + 0.05)
    return (t_fail, t_mid, t_great)
