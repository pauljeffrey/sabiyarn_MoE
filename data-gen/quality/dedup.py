"""Exact and near-duplicate detection, scoped per (task, language) cell.

Scoping to a cell (rather than the whole dataset) keeps this O(n^2)
shingle comparison cheap -- cells are a few hundred examples at most for
realistic run sizes -- while still catching the failure mode that
actually matters: the generation model falling into a repetitive pattern
within one (task, language) combination (e.g. reusing the same opening
phrase or the same toy numbers across many examples).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    return _WORD_RE.sub(lambda m: m.group(0).lower(), text)


def _shingles(text: str, k: int = 5) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def content_fingerprint(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def find_duplicates(
    records: list[dict[str, Any]],
    *,
    cell_key_fn,
    text_fn,
    near_dup_threshold: float = 0.8,
) -> tuple[set[int], dict[int, str]]:
    """Return (drop_indices, reasons) for records that are exact or
    near-duplicates of an earlier record in the same cell.

    `cell_key_fn(record) -> hashable` groups records (e.g. by (task, language)).
    `text_fn(record) -> str` extracts the text used for comparison (e.g. all
    user turns concatenated).
    """
    drop_indices: set[int] = set()
    reasons: dict[int, str] = {}

    cells: dict[Any, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        cells[cell_key_fn(rec)].append(idx)

    for cell, indices in cells.items():
        seen_hashes: dict[str, int] = {}
        kept_shingles: list[tuple[int, set[str]]] = []
        for idx in indices:
            text = text_fn(records[idx])
            fp = content_fingerprint(text)
            if fp in seen_hashes:
                drop_indices.add(idx)
                reasons[idx] = f"exact duplicate of index {seen_hashes[fp]} in cell {cell!r}"
                continue
            seen_hashes[fp] = idx

            shingles = _shingles(text)
            is_near_dup = False
            if shingles:
                for other_idx, other_shingles in kept_shingles:
                    if not other_shingles:
                        continue
                    union = shingles | other_shingles
                    if not union:
                        continue
                    jaccard = len(shingles & other_shingles) / len(union)
                    if jaccard >= near_dup_threshold:
                        drop_indices.add(idx)
                        reasons[idx] = (
                            f"near-duplicate (jaccard={jaccard:.2f}) of index "
                            f"{other_idx} in cell {cell!r}"
                        )
                        is_near_dup = True
                        break
            if not is_near_dup:
                kept_shingles.append((idx, shingles))

    return drop_indices, reasons


# ---------------------------------------------------------------------------
# Scalable variant for the corpus kinds (tens of thousands of records)
# ---------------------------------------------------------------------------


def _shingle_hashes(text: str, k: int) -> set[int]:
    """Stable 64-bit hashes of word k-shingles (not `hash()`, which is salted per run)."""
    return {
        int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
        for s in _shingles(text, k)
    }


def find_near_duplicates(
    records: list[dict[str, Any]],
    *,
    cell_key_fn,
    text_fn,
    near_dup_threshold: float = 0.8,
    shingle_size: int = 5,
    sketch_size: int = 48,
    label: str = "cell",
) -> tuple[set[int], dict[int, str]]:
    """Same contract as `find_duplicates`, but sub-quadratic.

    `find_duplicates` compares every record to every kept record in its cell,
    which is fine for a few hundred records but hopeless for 6,000 per
    language (or 72,000 across languages). Here each record keeps a bottom-k
    "sketch" (its `sketch_size` smallest shingle hashes); an inverted index
    over sketch elements proposes candidate pairs, and only candidates that
    share enough sketch elements are verified with the exact Jaccard score.
    For texts with fewer than `sketch_size` shingles the sketch is the whole
    shingle set, so short-text results are exact. Earlier records win.
    """
    drop: set[int] = set()
    reasons: dict[int, str] = {}

    cells: dict[Any, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        cells[cell_key_fn(rec)].append(idx)

    for cell, indices in cells.items():
        seen_fp: dict[str, int] = {}
        index: dict[int, list[int]] = defaultdict(list)  # sketch element -> kept record ids
        kept_shingles: dict[int, set[int]] = {}
        for idx in indices:
            text = text_fn(records[idx])
            fp = content_fingerprint(text)
            if fp in seen_fp:
                drop.add(idx)
                reasons[idx] = f"exact duplicate of index {seen_fp[fp]} in {label} {cell!r}"
                continue
            seen_fp[fp] = idx

            sh = _shingle_hashes(text, shingle_size)
            sketch = sorted(sh)[:sketch_size]
            dup_of = None
            if sketch:
                votes: Counter = Counter()
                for h in sketch:
                    for other in index.get(h, [])[-200:]:  # cap: ubiquitous shingles must not make this quadratic
                        votes[other] += 1
                need = max(1, int(0.25 * len(sketch)))
                for other, c in votes.most_common(20):
                    if c < need:
                        break
                    o = kept_shingles[other]
                    union = len(sh | o)
                    if union and len(sh & o) / union >= near_dup_threshold:
                        dup_of = (other, len(sh & o) / union)
                        break
            if dup_of is not None:
                drop.add(idx)
                reasons[idx] = f"near-duplicate (jaccard={dup_of[1]:.2f}) of index {dup_of[0]} in {label} {cell!r}"
                continue
            kept_shingles[idx] = sh
            for h in sketch:
                index[h].append(idx)

    return drop, reasons
