"""Fixed, stratified evaluation plan over a segmented validation bin. Pure Python + numpy, no torch.

The validation bin (datasets/validation_v2.bin) is the old validation set followed by a few million
tokens of multilingual text. A manifest JSON next to it says where each part lives:

    {"dtype": "uint16", "total_tokens": N,
     "segments": [{"name": "old", "start": 0, "end": 1800000000},
                  {"name": "zul", "start": 1800000000, "end": 1801000000}, ...]}

``start``/``end`` are token offsets into the bin; segments are contiguous, ascending and cover the whole
file; every name other than ``"old"`` is a language code.

Sampling design (build_eval_plan):

  * The set of eval blocks is FIXED: a pure function of (manifest, block_size, eval_total_blocks,
    eval_old_share, seed). Every eval step scores the same windows, so losses are comparable across steps
    (and across runs with the same inputs).
  * A block is a window of ``block_size`` inputs plus the one extra target token, and it NEVER crosses a
    segment boundary (``(len - 1) // block_size`` blocks fit in a segment, the same rule as
    data_sampler.n_blocks_for).
  * ``"old"`` gets ``round(total x old_share)`` blocks, the remaining blocks are shared EQUALLY by the
    language segments, each capped by how many full blocks its segment holds; any shortfall is
    re-distributed to the languages that still have room (and, if every language is full, handed back to
    "old"), so the plan has exactly ``total`` blocks whenever the file can hold that many.
  * Inside a segment the k blocks sit at equally spaced block indices (systematic sampling: index
    floor((j + u) * n / k), u in [0, 1) drawn once per segment from the seed). That spreads the sample over
    the WHOLE segment instead of clumping, and the stride is >= 1 block so blocks never overlap.
    The random phase u (rather than a fixed 0.5) is only there to avoid locking onto any periodicity in the
    data's order; it is deterministic given the seed.
  * Distributed: ``EvalPlan.rank_steps`` partitions every segment's blocks round-robin over ranks and cuts
    each rank's share into micro-batches of ``batch_size`` blocks, all from ONE segment (so a micro-batch's
    mean loss belongs to exactly one segment). Every rank gets the same number of steps (an empty share is
    a zero-weight dummy step), which keeps FSDP's collective forwards in lockstep.
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np

OLD = "old"
SUPPORTED_DTYPE = "uint16"


# --------------------------------------------------------------------------------------- manifest

@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    end: int

    @property
    def tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Manifest:
    dtype: str
    total_tokens: int
    segments: tuple


def parse_manifest(obj: Any, *, file_tokens: Optional[int] = None) -> Manifest:
    """Validate a decoded manifest. ``file_tokens`` (the bin's real length in tokens) must equal
    ``total_tokens`` when given: a truncated or mismatched download fails here, not mid-run."""
    if not isinstance(obj, Mapping):
        raise ValueError("eval manifest must be a JSON object")
    dtype = str(obj.get("dtype", SUPPORTED_DTYPE))
    if dtype != SUPPORTED_DTYPE:
        raise ValueError(f"eval manifest dtype must be {SUPPORTED_DTYPE!r} (the trainer reads uint16), got {dtype!r}")
    raw_segments = obj.get("segments")
    if not raw_segments:
        raise ValueError("eval manifest has no segments")
    segments, names, cursor = [], set(), 0
    for raw in raw_segments:
        try:
            seg = Segment(str(raw["name"]), int(raw["start"]), int(raw["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bad eval manifest segment {raw!r}: need name/start/end") from exc
        if seg.end <= seg.start:
            raise ValueError(f"eval manifest segment {seg.name!r} is empty or inverted: {seg.start}..{seg.end}")
        if seg.start != cursor:
            raise ValueError(
                f"eval manifest segments must be contiguous from 0: {seg.name!r} starts at {seg.start}, expected {cursor}"
            )
        if seg.name in names:
            raise ValueError(f"eval manifest has duplicate segment name {seg.name!r}")
        names.add(seg.name)
        segments.append(seg)
        cursor = seg.end
    total = int(obj.get("total_tokens", cursor))
    if total != cursor:
        raise ValueError(f"eval manifest total_tokens={total} but its segments end at {cursor}")
    if file_tokens is not None and int(file_tokens) != total:
        raise ValueError(
            f"eval manifest describes {total} tokens but the validation bin holds {int(file_tokens)}: "
            "the manifest and the bin do not belong together (stale or truncated download?)"
        )
    return Manifest(dtype=dtype, total_tokens=total, segments=tuple(segments))


def load_manifest(path: str, *, file_tokens: Optional[int] = None) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        return parse_manifest(json.load(f), file_tokens=file_tokens)


def resolve_manifest_path(configured: str, eval_bin: str) -> str:
    """The manifest file to read: ``configured`` as given, else the same file name next to the eval
    bin (the S3 sync flattens both into one local directory)."""
    candidates = [configured, os.path.join(os.path.dirname(eval_bin), os.path.basename(configured))]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"eval manifest not found; looked for {candidates}")


# --------------------------------------------------------------------------------------- allocation

def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def blocks_in(n_tokens: int, block_size: int) -> int:
    """Full blocks (plus their +1 target token) that fit in ``n_tokens``; matches data_sampler.n_blocks_for."""
    return max(0, (n_tokens - 1) // block_size)


def allocate_blocks(
    total: int, old_share: float, capacity: Mapping[str, int], *, old_name: str = OLD,
) -> dict:
    """Blocks per segment. ``old`` gets round(total x old_share) (capped by capacity), the rest is
    shared equally by the other segments with shortfalls re-distributed; the result sums to ``total``
    whenever the capacities allow (otherwise to the total capacity)."""
    if total < 0:
        raise ValueError("total blocks must be >= 0")
    if not 0.0 <= old_share <= 1.0:
        raise ValueError("old_share must be in [0, 1]")
    alloc = {name: 0 for name in capacity}
    langs = [n for n in capacity if n != old_name]
    if old_name in capacity:
        alloc[old_name] = min(int(total * old_share + 0.5), capacity[old_name], total)
    remaining = total - sum(alloc.values())
    active = [n for n in langs if capacity[n] > 0]
    while remaining > 0 and active:
        q, r = divmod(remaining, len(active))
        still_active = []
        for i, name in enumerate(active):
            want = q + (1 if i < r else 0)
            give = min(want, capacity[name] - alloc[name])
            alloc[name] += give
            remaining -= give
            if alloc[name] < capacity[name]:
                still_active.append(name)
        active = still_active
    if remaining > 0 and old_name in capacity:  # every language is full: give the rest back to "old"
        extra = min(remaining, capacity[old_name] - alloc[old_name])
        alloc[old_name] += extra
    return alloc


def systematic_indices(n_available: int, k: int, phase: float) -> list:
    """``k`` equally spaced indices in [0, n_available): floor((j + phase) * n / k). Strictly increasing
    (the spacing is > 1 whenever k < n_available); all indices when k >= n_available."""
    if k <= 0 or n_available <= 0:
        return []
    if k >= n_available:
        return list(range(n_available))
    step = n_available / k
    return [int((j + phase) * step) for j in range(k)]


# --------------------------------------------------------------------------------------- the plan

class EvalStep(NamedTuple):
    segment: str
    starts: tuple   # token offsets of the blocks of this micro-batch (1..batch_size of them)
    real: bool      # False = a dummy step that only keeps collective forwards aligned; weight it 0


@dataclass(frozen=True)
class EvalPlan:
    block_size: int
    segments: tuple          # segment names, "old" first, then manifest order
    starts: Mapping          # segment name -> tuple of block start offsets (ascending)
    requested_total: int
    capacity: Mapping        # segment name -> full blocks the segment holds

    @property
    def counts(self) -> dict:
        return {n: len(self.starts[n]) for n in self.segments}

    @property
    def n_blocks(self) -> int:
        return sum(len(v) for v in self.starts.values())

    @property
    def languages(self) -> tuple:
        return tuple(n for n in self.segments if n != OLD)

    def rank_steps(self, rank: int, world: int, batch_size: int) -> list:
        """The micro-batches rank ``rank`` of ``world`` evaluates, in a fixed order. Every rank gets the
        same number of steps; an exhausted share becomes a dummy step (``real=False``) on the segment's
        first block."""
        if not 0 <= rank < world:
            raise ValueError("rank must be in [0, world)")
        steps = []
        for name in self.segments:
            all_starts = self.starts[name]
            if not all_starts:
                continue
            share = all_starts[rank::world]
            n_steps = _ceil_div(_ceil_div(len(all_starts), world), batch_size)
            for k in range(n_steps):
                chunk = share[k * batch_size : (k + 1) * batch_size]
                if chunk:
                    steps.append(EvalStep(name, tuple(chunk), True))
                else:
                    steps.append(EvalStep(name, (all_starts[0],), False))
        return steps

    def describe(self) -> dict:
        return {
            n: {"blocks": len(self.starts[n]), "capacity": self.capacity[n],
                "tokens": len(self.starts[n]) * self.block_size}
            for n in self.segments
        }


def build_eval_plan(
    manifest: Manifest, *, block_size: int, total_blocks: int, old_share: float, seed: int,
) -> EvalPlan:
    if block_size <= 0 or total_blocks <= 0:
        raise ValueError("block_size and total_blocks must be positive")
    ordered = sorted(manifest.segments, key=lambda s: (s.name != OLD,))  # stable: "old" first
    capacity = {s.name: blocks_in(s.tokens, block_size) for s in ordered}
    alloc = allocate_blocks(total_blocks, old_share, capacity)
    starts = {}
    for seg in ordered:
        phase = float(np.random.default_rng([seed, zlib.crc32(seg.name.encode())]).random())
        idx = systematic_indices(capacity[seg.name], alloc[seg.name], phase)
        starts[seg.name] = tuple(seg.start + i * block_size for i in idx)
    return EvalPlan(
        block_size=block_size, segments=tuple(s.name for s in ordered), starts=starts,
        requested_total=total_blocks, capacity=capacity,
    )


# --------------------------------------------------------------------------------------- reading + summary

def read_windows(data: np.ndarray, starts: Sequence[int], block_size: int) -> tuple:
    """(x, y) int64 arrays of shape (len(starts), block_size) for windows at absolute token offsets."""
    x = np.stack([data[s : s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + block_size + 1] for s in starts]).astype(np.int64)
    return x, y


def summarize_segments(totals: Mapping[str, Sequence[float]], *, old_name: str = OLD) -> dict:
    """Combine per-segment sums into the eval metrics.

    ``totals[name] = (loss x tokens, ce x tokens, tokens, bytes)`` summed over the segment's blocks
    (loss includes the MoE aux term, ce does not). Returns:

      val             token-weighted mean loss over ALL blocks (what best-checkpoint logic keeps using)
      val_ce          the same for pure cross-entropy
      val_old         mean loss of the "old" segment only (continuity with earlier runs)
      val_lang_macro  UNWEIGHTED mean of the per-language mean losses
      val_by_lang     {lang: mean loss}
      by_segment      {name: {"loss", "ce", "tokens", "bytes"}} for every non-empty segment
      tokens, bytes   totals over all blocks
    """
    by_segment, loss_sum, ce_sum, tok_sum, byte_sum = {}, 0.0, 0.0, 0.0, 0.0
    for name, (l, c, t, b) in totals.items():
        loss_sum += l; ce_sum += c; tok_sum += t; byte_sum += b
        if t > 0:
            by_segment[name] = {"loss": l / t, "ce": c / t, "tokens": t, "bytes": b}
    out: dict = {"by_segment": by_segment, "tokens": tok_sum, "bytes": byte_sum}
    if tok_sum > 0:
        out["val"] = loss_sum / tok_sum
        out["val_ce"] = ce_sum / tok_sum
    if old_name in by_segment:
        out["val_old"] = by_segment[old_name]["loss"]
    langs = {n: s["loss"] for n, s in by_segment.items() if n != old_name}
    out["val_by_lang"] = langs
    if langs:
        out["val_lang_macro"] = sum(langs.values()) / len(langs)
    return out
