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
from collections import defaultdict
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
