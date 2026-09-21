"""Deterministic coverage sampler: the diversity engine for corpus generation.

WHY not just `rng.choice`? Independent random draws are *unbalanced*: with
636 sub-topics and 4,000 documents, pure sampling leaves some sub-topics with
0 draws and others with 15, and the same holds for every secondary
attribute (genre, register, ...). Balance is cheap to enforce when you own
the sampling loop, so we do:

  Primary key (domain, sub-topic)
      Cycle through a *shuffled list of all pairs* before repeating any, and
      reshuffle for each new cycle. After N draws every pair has been used
      floor(N/P) or ceil(N/P) times (exactly, for uniform weights).

  Secondary attributes (genre / task, register, audience, length bucket,
  perspective, locale, difficulty, ...)
      "Least-used-first": among the values that are *compatible* with what
      has already been chosen, pick the one minimising
      (count + 1) / target_weight, breaking ties with the seeded RNG. This is
      the classic highest-averages apportionment rule, so marginal counts
      track the target weights to within about one draw whenever the
      compatibility constraints are feasible. When constraints make a value
      under-supplied it accumulates a deficit and wins the next time it is
      legal -- the sampler self-corrects instead of drifting.

Determinism: one sampler instance per (kind, language); its seed is derived
from (global seed, kind, language) with the SAME scheme for every language,
so all languages get identical coverage structure. Draw i is a pure function
of the config, so a failed request can be regenerated identically.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

CompatFn = Callable[[dict, str, str], bool]
"""compat(chosen_so_far, attribute_name, candidate_value) -> allowed?"""


@dataclass(frozen=True)
class AttributeSpec:
    """A secondary attribute: its name and target weights per value.

    Order matters: attributes are drawn in list order and the compat function
    may look at anything drawn earlier.
    """

    name: str
    weights: Mapping[str, float]


def derive_seed(seed: int, kind: str, language: str) -> int:
    """Stable 64-bit seed from (global seed, kind, language).

    Not Python's `hash()` (salted per process) and not shared across
    languages, so each (kind, language) gets its own reproducible stream
    built by the identical procedure.
    """
    digest = hashlib.sha256(f"{seed}|{kind}|{language}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def merge_weights(defaults: Mapping[str, float], overrides: Optional[Mapping[str, float]], *, name: str) -> dict[str, float]:
    """Apply yaml weight overrides on top of taxonomy defaults.

    Unknown keys raise (a typo in a yaml file should fail loudly, not
    silently produce an unbalanced dataset). A weight of 0 disables a value.
    """
    merged = dict(defaults)
    for key, value in (overrides or {}).items():
        if key not in merged:
            raise KeyError(f"Unknown value {key!r} for attribute {name!r}. Known values: {sorted(merged)}")
        if value < 0:
            raise ValueError(f"Negative weight for {name}.{key}")
        merged[key] = float(value)
    if not any(w > 0 for w in merged.values()):
        raise ValueError(f"All weights for attribute {name!r} are zero")
    return merged


class CoverageSampler:
    """Balanced, seeded sampler over (domain, sub-topic) x secondary attributes."""

    def __init__(
        self,
        *,
        kind: str,
        language: str,
        seed: int,
        pairs: list[tuple[str, str]],
        attributes: list[AttributeSpec],
        compat: Optional[CompatFn] = None,
        pair_weights: Optional[Mapping[tuple[str, str], float]] = None,
    ) -> None:
        if not pairs:
            raise ValueError("CoverageSampler needs at least one (domain, subtopic) pair")
        self.kind = kind
        self.language = language
        self.pairs = list(pairs)
        self.attributes = list(attributes)
        self._compat = compat
        self._pair_weights = dict(pair_weights or {})
        self._rng = random.Random(derive_seed(seed, kind, language))
        self._queue: list[tuple[str, str]] = []
        self.cycles_started = 0
        self.relaxed_count = 0  # times a compat constraint had to be dropped (should stay 0)
        self.pair_counts: Counter = Counter()
        self.counts: dict[str, Counter] = {a.name: Counter() for a in self.attributes}
        self.draws = 0

    # -- primary key ---------------------------------------------------

    def _refill(self) -> None:
        bag: list[tuple[str, str]] = []
        for pair in self.pairs:
            w = self._pair_weights.get(pair, 1.0)
            k = int(w)
            if self._rng.random() < (w - k):
                k += 1
            bag.extend([pair] * k)
        if not bag:
            raise ValueError("All (domain, subtopic) weights are zero")
        self._rng.shuffle(bag)
        self._queue = bag
        self.cycles_started += 1

    def _next_pair(self) -> tuple[str, str]:
        if not self._queue:
            self._refill()
        return self._queue.pop()

    # -- secondary attributes -----------------------------------------

    def _pick(self, spec: AttributeSpec, chosen: dict) -> str:
        counts = self.counts[spec.name]
        positive = [(v, w) for v, w in spec.weights.items() if w > 0]
        legal = [(v, w) for v, w in positive if self._compat is None or self._compat(chosen, spec.name, v)]
        if not legal:
            self.relaxed_count += 1
            legal = positive
        best_key = None
        best_value = None
        for v, w in legal:
            score = round((counts[v] + 1) / w, 9)
            key = (score, self._rng.random())
            if best_key is None or key < best_key:
                best_key, best_value = key, v
        assert best_value is not None
        return best_value

    # -- public API ----------------------------------------------------

    def draw(self) -> dict[str, str]:
        """One attribute tuple: {'domain', 'subtopic', <secondary attrs...>}."""
        domain, subtopic = self._next_pair()
        chosen: dict[str, str] = {"domain": domain, "subtopic": subtopic}
        for spec in self.attributes:
            value = self._pick(spec, chosen)
            chosen[spec.name] = value
            self.counts[spec.name][value] += 1
        self.pair_counts[(domain, subtopic)] += 1
        self.draws += 1
        return chosen

    def draw_many(self, n: int) -> list[dict[str, str]]:
        return [self.draw() for _ in range(n)]

    def __iter__(self):
        while True:
            yield self.draw()


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def normalized_entropy(counts: Mapping[str, int], vocabulary: Optional[Iterable[str]] = None) -> float:
    """Shannon entropy / log(K) in [0, 1]; 1 means perfectly uniform.

    K is the vocabulary size (values never drawn still count, which is what
    exposes under-coverage) or, if omitted, the number of observed values.
    """
    keys = list(vocabulary) if vocabulary is not None else list(counts)
    k = len(keys)
    total = sum(counts.get(key, 0) for key in keys)
    if k <= 1 or total == 0:
        return 1.0 if k <= 1 else 0.0
    h = 0.0
    for key in keys:
        c = counts.get(key, 0)
        if c:
            p = c / total
            h -= p * math.log(p)
    return h / math.log(k)


def _summarise(counts: Mapping[str, int], vocabulary: Optional[Iterable[str]] = None, targets: Optional[Mapping[str, float]] = None) -> dict:
    keys = list(vocabulary) if vocabulary is not None else list(counts)
    full = {k: counts.get(k, 0) for k in keys}
    values = list(full.values())
    total = sum(values)
    mn, mx = (min(values), max(values)) if values else (0, 0)
    out = {
        "distinct": sum(1 for v in values if v),
        "vocabulary": len(keys),
        "total": total,
        "min": mn,
        "max": mx,
        "max_min_ratio": (mx / mn) if mn else None,  # None: some value never used
        "entropy_norm": round(normalized_entropy(full, keys), 4),
    }
    if targets:
        tw = sum(w for w in targets.values() if w > 0) or 1.0
        dev = 0.0
        for k in keys:
            expected = total * (targets.get(k, 0.0) / tw)
            dev = max(dev, abs(full[k] - expected))
        out["max_abs_dev_from_target"] = round(dev, 2)
    return out


def coverage_report(
    draws: list[Mapping[str, str]],
    *,
    vocabularies: Optional[Mapping[str, Iterable[str]]] = None,
    targets: Optional[Mapping[str, Mapping[str, float]]] = None,
    include_counts: bool = True,
) -> dict:
    """Coverage of a list of attribute tuples.

    Reports, per attribute (plus the derived 'pair' key): counts per value,
    normalised entropy, max/min ratio, and (if targets given) the largest
    absolute deviation from the target share. `vocabularies` gives the
    complete value set per attribute so unused values are visible.
    """
    vocabularies = dict(vocabularies or {})
    targets = dict(targets or {})
    names: list[str] = []
    for d in draws:
        for k in d:
            if k not in names:
                names.append(k)
    report: dict = {"n": len(draws), "attributes": {}}

    pair_counts = Counter(f"{d['domain']}::{d['subtopic']}" for d in draws if "domain" in d and "subtopic" in d)
    if pair_counts:
        vocab = vocabularies.get("pair")
        report["attributes"]["pair"] = _summarise(pair_counts, vocab)

    for name in names:
        if name == "subtopic":
            continue  # covered via 'pair' (sub-topic strings are only unique within a domain)
        counts = Counter(d[name] for d in draws if name in d)
        entry = _summarise(counts, vocabularies.get(name), targets.get(name))
        if include_counts:
            entry["counts"] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        report["attributes"][name] = entry
    return report


def pair_weights_from_groups(pairs: list[tuple[str, str]], domain_group: Mapping[str, str], group_weights: Optional[Mapping[str, float]]) -> dict[tuple[str, str], float]:
    """Expand per-domain-group weights (yaml) into per-pair weights.

    Weight 1.0 (the default) keeps exact uniform cycling; 2.0 puts each pair
    of that group in every cycle twice; 0.5 includes it in every second cycle
    on average; 0 excludes the group.
    """
    if not group_weights:
        return {}
    known = set(domain_group.values())
    unknown = set(group_weights) - known
    if unknown:
        raise KeyError(f"Unknown domain group(s) {sorted(unknown)}. Known groups: {sorted(known)}")
    return {p: float(group_weights.get(domain_group[p[0]], 1.0)) for p in pairs}
