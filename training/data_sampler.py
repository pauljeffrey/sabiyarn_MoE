"""Epoch-based, no-repeat batch sampling over memory-mapped token bins.

The old sampler drew `bs` random start offsets with replacement for every micro-batch, so windows
overlapped and the same tokens could be sampled several times long before others were seen once.
It also re-seeded on every restart, so a resumed run replayed exactly the batches it had already
trained on. This module replaces that with:

  * Each bin is cut into NON-OVERLAPPING windows ("blocks") of `block_size` tokens. Block i feeds
    inputs data[i*bs : (i+1)*bs] and targets data[i*bs+1 : (i+1)*bs+1], so every token position is
    an input at most once and a prediction target at most once per epoch.
  * A block order is a fresh random permutation each epoch (seeded by (seed, bin, epoch), so it is
    reproducible). Every block is used exactly once per epoch, per bin, independently.
  * Distributed: all ranks share ONE global stream per bin. A step takes `batch * world` blocks and
    rank r takes slice r -- disjoint by construction, no coordination needed.
  * Mixing bins (English / African): a deterministic "credit" scheme (Bresenham-style) picks the
    bin for each micro-step from the current sampling weights. It is identical on every rank, tracks
    the weights exactly over time (including scheduled weights that change), and has no RNG state.
  * The whole state (epoch/position per bin + credits) is a small dict, persisted in
    trainer_state.json, so a resume continues mid-epoch instead of replaying or skipping data.

Pure numpy, no torch: unit-testable anywhere.
"""

from __future__ import annotations

import zlib
from typing import Optional, Sequence

import numpy as np


def n_blocks_for(n_tokens: int, block_size: int) -> int:
    """Blocks that fit with the +1 target token: block i needs data[i*bs : i*bs + bs + 1]."""
    return max(0, (n_tokens - 1) // block_size)


class EpochStream:
    """Infinite stream of block indices: perm_0, perm_1, ... (each a permutation of range(n))."""

    def __init__(self, n_blocks: int, seed: int, name: str):
        if n_blocks <= 0:
            raise ValueError(f"bin {name!r} has no complete block")
        self.n_blocks = n_blocks
        self.seed = seed
        self.name = name
        self.epoch = 0
        self.pos = 0  # blocks already consumed from this epoch's permutation
        self._perm_epoch: Optional[int] = None
        self._perm: Optional[np.ndarray] = None

    def _permutation(self, epoch: int) -> np.ndarray:
        if self._perm_epoch != epoch:
            rng = np.random.default_rng([self.seed, zlib.crc32(self.name.encode()), epoch])
            self._perm = rng.permutation(self.n_blocks).astype(np.int64)
            self._perm_epoch = epoch
        return self._perm  # type: ignore[return-value]

    def take(self, m: int) -> np.ndarray:
        """Next m block indices. Crosses epoch boundaries without dropping or repeating a block."""
        out = []
        need = m
        while need > 0:
            perm = self._permutation(self.epoch)
            chunk = perm[self.pos : self.pos + need]
            out.append(chunk)
            self.pos += len(chunk)
            need -= len(chunk)
            if self.pos >= self.n_blocks:
                self.epoch += 1
                self.pos = 0
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    @property
    def epochs_done(self) -> float:
        """Fractional epochs consumed, e.g. 1.37 = 37% through the second pass."""
        return self.epoch + self.pos / self.n_blocks

    def state_dict(self) -> dict:
        return {"n_blocks": self.n_blocks, "epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, state: dict) -> bool:
        """Restore; refuses (returns False) if the bin changed size since the state was saved."""
        if int(state.get("n_blocks", -1)) != self.n_blocks:
            return False
        self.epoch, self.pos = int(state["epoch"]), int(state["pos"])
        return True

    def reset(self) -> None:
        self.epoch, self.pos = 0, 0


class MixedBlockSampler:
    """Deterministic multi-bin sampler for `world_size` ranks; every rank runs the same object logic."""

    def __init__(self, names: Sequence[str], n_blocks: Sequence[int], *, seed: int, batch_size: int,
                 world_size: int = 1, rank: int = 0):
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.names = list(names)
        self.streams = [EpochStream(n, seed, name) for name, n in zip(names, n_blocks)]
        self.batch_size, self.world_size, self.rank = batch_size, world_size, rank
        self.credits = [0.0] * len(self.streams)

    def next_batch(self, weights: Sequence[float]) -> tuple[int, np.ndarray]:
        """(bin index, this rank's block indices) for one micro-step."""
        if len(self.streams) == 1:
            b = 0
        else:
            total = float(sum(weights))
            w = [1.0 / len(weights)] * len(weights) if total <= 0 else [x / total for x in weights]
            self.credits = [c + x for c, x in zip(self.credits, w)]
            b = max(range(len(self.credits)), key=lambda i: (self.credits[i], -i))  # ties -> lowest index
            self.credits[b] -= 1.0
        ids = self.streams[b].take(self.batch_size * self.world_size)
        lo = self.rank * self.batch_size
        return b, ids[lo : lo + self.batch_size]

    def epochs_done(self) -> dict[str, float]:
        return {n: round(s.epochs_done, 4) for n, s in zip(self.names, self.streams)}

    def state_dict(self) -> dict:
        return {"credits": list(self.credits), "streams": {n: s.state_dict() for n, s in zip(self.names, self.streams)}}

    def load_state_dict(self, state: dict) -> list[str]:
        """Restore what matches; returns the names of bins whose saved state could not be applied."""
        skipped = []
        for n, s in zip(self.names, self.streams):
            if n not in state.get("streams", {}) or not s.load_state_dict(state["streams"][n]):
                skipped.append(n)
        credits = state.get("credits", [])
        if len(credits) == len(self.credits) and not skipped:
            self.credits = [float(c) for c in credits]
        return skipped

    def reset(self) -> None:
        for s in self.streams:
            s.reset()
        self.credits = [0.0] * len(self.streams)


def read_blocks(data: np.ndarray, block_ids: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) int64 arrays of shape (len(block_ids), block_size) for the given blocks of a memmap."""
    starts = block_ids.astype(np.int64) * block_size
    x = np.stack([data[s : s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + block_size + 1] for s in starts]).astype(np.int64)
    return x, y
