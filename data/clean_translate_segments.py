"""
Core logic for removing complete <translate>...<marker>...</s> segments from a
tokenized .bin file, keeping only the first `keep_count` such segments and
deleting the rest, with every other token's relative order fully preserved.

Pure numpy/multiprocessing -- no Modal, network, or tokenizer dependency here,
so this is directly unit-testable (see tests/test_clean_translate_segments.py).
See data/clean_translate_segments_modal.py for the Modal entrypoint that
resolves real token ids and syncs the file to/from S3.

A "segment" is the token range [translate_pos, matched_eos] (inclusive), where
matched_eos is the first end-of-sequence token strictly after translate_pos. A
segment "qualifies" for deletion consideration only if at least one occurrence
of `marker_token` (e.g. <twi>) falls strictly between translate_pos and
matched_eos. Segments that don't qualify (e.g. <translate>...<eng>...</s>) are
never touched, regardless of keep_count. A trailing <translate> with no
following eos (an incomplete segment at the very end of the file) is left
untouched -- its extent can't be determined safely.

Of the qualifying segments, in stream order, the first `keep_count` are kept
and every later one is deleted entirely (including its own </s>).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence, Tuple

import numpy as np

DTYPE = np.uint16


def _find_positions_in_chunk(
    path: str, dtype, start: int, end: int, token_ids: Tuple[int, ...]
) -> Tuple[np.ndarray, ...]:
    """Positions of each id in token_ids within [start, end) of the memmap at
    path, as global (whole-file) offsets. Reads the chunk once regardless of
    how many token_ids are searched. Runs in a worker process -- opens its own
    memmap handle (memmaps aren't picklable across processes)."""
    arr = np.memmap(path, dtype=dtype, mode="r")
    chunk = arr[start:end]
    return tuple(np.flatnonzero(chunk == tok).astype(np.int64) + start for tok in token_ids)


def find_token_positions(
    path: str, total_len: int, token_ids: Tuple[int, ...], num_workers: int, dtype=DTYPE
) -> Tuple[np.ndarray, ...]:
    """Parallel whole-file scan for len(token_ids) token ids at once. Returns
    one sorted global-position array per token id, in file order."""
    num_workers = max(1, num_workers)
    chunk_size = math.ceil(total_len / num_workers) if total_len else 0
    bounds = [
        (i * chunk_size, min((i + 1) * chunk_size, total_len)) for i in range(num_workers)
    ]
    bounds = [b for b in bounds if b[0] < b[1]]
    if not bounds:
        return tuple(np.array([], dtype=np.int64) for _ in token_ids)

    parts: List[List[np.ndarray]] = [[] for _ in token_ids]
    if num_workers == 1 or len(bounds) == 1:
        for (s, e) in bounds:
            for i, pos in enumerate(_find_positions_in_chunk(path, dtype, s, e, token_ids)):
                parts[i].append(pos)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            # Submitted in chunk order; .result() on an earlier future blocks
            # until *that* one is ready regardless of completion order, so
            # collecting in submission order preserves global file order.
            futures = [ex.submit(_find_positions_in_chunk, path, dtype, s, e, token_ids) for (s, e) in bounds]
            for fut in futures:
                for i, pos in enumerate(fut.result()):
                    parts[i].append(pos)

    return tuple(np.concatenate(p) if p else np.array([], dtype=np.int64) for p in parts)


def _find_isin_positions_in_chunk(path: str, dtype, start: int, end: int, token_id_array: np.ndarray) -> np.ndarray:
    """Positions within [start, end) of the memmap at path whose value is ANY
    of token_id_array (e.g. every language/action tag id), as global offsets.
    Runs in a worker process -- opens its own memmap handle."""
    arr = np.memmap(path, dtype=dtype, mode="r")
    chunk = arr[start:end]
    return np.flatnonzero(np.isin(chunk, token_id_array)).astype(np.int64) + start


def find_any_token_positions(
    path: str, total_len: int, token_ids: Sequence[int], num_workers: int, dtype=DTYPE
) -> np.ndarray:
    """Parallel whole-file scan for positions matching ANY id in token_ids
    (e.g. the full set of language/action tags). Returns one sorted global
    position array, in file order."""
    token_id_array = np.asarray(sorted(set(int(t) for t in token_ids)), dtype=dtype)
    num_workers = max(1, num_workers)
    chunk_size = math.ceil(total_len / num_workers) if total_len else 0
    bounds = [
        (i * chunk_size, min((i + 1) * chunk_size, total_len)) for i in range(num_workers)
    ]
    bounds = [b for b in bounds if b[0] < b[1]]
    if not bounds:
        return np.array([], dtype=np.int64)

    parts: List[np.ndarray] = []
    if num_workers == 1 or len(bounds) == 1:
        for (s, e) in bounds:
            parts.append(_find_isin_positions_in_chunk(path, dtype, s, e, token_id_array))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_find_isin_positions_in_chunk, path, dtype, s, e, token_id_array) for (s, e) in bounds]
            for fut in futures:
                parts.append(fut.result())

    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def compute_delete_ranges(
    translate_pos: Sequence[int],
    marker_pos: Sequence[int],
    eos_pos: Sequence[int],
    keep_count: int,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Returns (delete_ranges, stats). delete_ranges is a sorted, non-overlapping
    list of (start, end_inclusive) token ranges to delete. stats reports counts
    useful for a sanity-check report before actually rewriting anything."""
    translate_pos = np.asarray(translate_pos, dtype=np.int64)
    marker_pos = np.asarray(marker_pos, dtype=np.int64)
    eos_pos = np.asarray(eos_pos, dtype=np.int64)

    stats = {
        "translate_tag_count": int(len(translate_pos)),
        "marker_tag_count": int(len(marker_pos)),
        "eos_tag_count": int(len(eos_pos)),
        "trailing_unterminated_translate": 0,
        "qualifying_segments": 0,
        "kept_segments": 0,
        "deleted_segments": 0,
        "malformed_overlapping_segments_merged": 0,
    }

    if len(translate_pos) == 0:
        return [], stats

    # Pair each <translate> with the first </s> strictly after it.
    eos_idx = np.searchsorted(eos_pos, translate_pos, side="right")
    has_match = eos_idx < len(eos_pos)
    stats["trailing_unterminated_translate"] = int((~has_match).sum())

    valid_translate = translate_pos[has_match]
    valid_eos = eos_pos[eos_idx[has_match]]

    # Does >=1 marker fall strictly inside (translate, eos)?
    lo = np.searchsorted(marker_pos, valid_translate, side="right")
    hi = np.searchsorted(marker_pos, valid_eos, side="left")
    qualifies = hi > lo

    q_starts = valid_translate[qualifies]  # already sorted: filtering preserves order
    q_ends = valid_eos[qualifies]
    stats["qualifying_segments"] = int(len(q_starts))

    if len(q_starts) <= keep_count:
        stats["kept_segments"] = int(len(q_starts))
        return [], stats

    del_starts = q_starts[keep_count:]
    del_ends = q_ends[keep_count:]
    stats["kept_segments"] = int(keep_count)
    stats["deleted_segments"] = int(len(del_starts))

    ranges: List[Tuple[int, int]] = list(zip(del_starts.tolist(), del_ends.tolist()))
    merged: List[Tuple[int, int]] = []
    for s, e in ranges:  # sorted by start already
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            stats["malformed_overlapping_segments_merged"] += 1
        else:
            merged.append((s, e))

    return merged, stats


def compute_delete_ranges_full(
    translate_pos: Sequence[int],
    marker_pos: Sequence[int],
    eos_pos: Sequence[int],
    action_tag_pos: Sequence[int],
    keep_count: int,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Like compute_delete_ranges, but adds a second, independent deletion
    criterion evaluated first for every [<translate> ... matched </s>]
    segment that has an identifiable language/action tag inside it (the
    first position in action_tag_pos strictly between translate and its
    matched eos):

    - "empty source": the tag immediately follows <translate>
      (<translate><LANG>...) -- there's no source text at all.
    - "empty target": </s> immediately follows the tag (...<LANG></s>) --
      there's no translated text at all.

    Either one gets the segment deleted UNCONDITIONALLY -- these are
    malformed/empty examples, not subject to keep_count, and they don't
    consume a "kept" slot even if marker_token also happens to appear in
    them (e.g. an empty-target segment whose tag *is* marker_token).

    Segments that aren't empty-source/target, and have marker_token
    somewhere inside, are thinning candidates exactly as in
    compute_delete_ranges: the first keep_count (in stream order) are kept,
    the rest deleted. Segments matching neither criterion are untouched.

    Ranges from both criteria are merged into one sorted, non-overlapping
    list before being returned. Adjacent (zero-gap) ranges get merged too --
    that's expected for densely packed corpora (one segment's </s>
    immediately followed by the next segment's <translate>), not a sign of
    malformed data; only genuinely overlapping ranges indicate that.
    """
    translate_pos = np.asarray(translate_pos, dtype=np.int64)
    marker_pos = np.asarray(marker_pos, dtype=np.int64)
    eos_pos = np.asarray(eos_pos, dtype=np.int64)
    action_tag_pos = np.asarray(action_tag_pos, dtype=np.int64)

    stats = {
        "translate_tag_count": int(len(translate_pos)),
        "marker_tag_count": int(len(marker_pos)),
        "eos_tag_count": int(len(eos_pos)),
        "action_tag_count": int(len(action_tag_pos)),
        "trailing_unterminated_translate": 0,
        "empty_source_segments": 0,
        "empty_target_segments": 0,
        "qualifying_thinning_segments": 0,
        "kept_thinning_segments": 0,
        "deleted_thinning_segments": 0,
        "adjacent_or_overlapping_ranges_merged": 0,
    }

    if len(translate_pos) == 0:
        return [], stats

    eos_idx = np.searchsorted(eos_pos, translate_pos, side="right")
    has_match = eos_idx < len(eos_pos)
    stats["trailing_unterminated_translate"] = int((~has_match).sum())

    t_pos = translate_pos[has_match]
    e_pos = eos_pos[eos_idx[has_match]]

    # First action/language tag strictly after t_pos; valid only if it also
    # falls before e_pos (otherwise it belongs to some later segment).
    if len(action_tag_pos) and len(t_pos):
        tag_idx = np.searchsorted(action_tag_pos, t_pos, side="right")
        has_tag_after = tag_idx < len(action_tag_pos)
        safe_idx = np.clip(tag_idx, 0, len(action_tag_pos) - 1)
        candidate_tag_pos = action_tag_pos[safe_idx]
        tag_inside = has_tag_after & (candidate_tag_pos < e_pos)
        lang_tag_pos = np.where(tag_inside, candidate_tag_pos, -1)
    else:
        tag_inside = np.zeros(len(t_pos), dtype=bool)
        lang_tag_pos = np.full(len(t_pos), -1, dtype=np.int64)

    is_empty_source = tag_inside & (lang_tag_pos == t_pos + 1)
    is_empty_target = tag_inside & (e_pos == lang_tag_pos + 1)
    always_delete = is_empty_source | is_empty_target
    stats["empty_source_segments"] = int(is_empty_source.sum())
    stats["empty_target_segments"] = int(is_empty_target.sum())

    # Marker-based thinning candidates, excluding segments already always_delete
    # (deleted regardless, so they must not consume a "kept" slot).
    lo = np.searchsorted(marker_pos, t_pos, side="right")
    hi = np.searchsorted(marker_pos, e_pos, side="left")
    has_marker = (hi > lo) & ~always_delete

    q_starts = t_pos[has_marker]
    q_ends = e_pos[has_marker]
    stats["qualifying_thinning_segments"] = int(len(q_starts))

    if len(q_starts) > keep_count:
        thin_del_starts = q_starts[keep_count:]
        thin_del_ends = q_ends[keep_count:]
        stats["kept_thinning_segments"] = int(keep_count)
        stats["deleted_thinning_segments"] = int(len(thin_del_starts))
    else:
        thin_del_starts = np.array([], dtype=np.int64)
        thin_del_ends = np.array([], dtype=np.int64)
        stats["kept_thinning_segments"] = int(len(q_starts))

    always_del_starts = t_pos[always_delete]
    always_del_ends = e_pos[always_delete]

    all_starts = np.concatenate([always_del_starts, thin_del_starts])
    all_ends = np.concatenate([always_del_ends, thin_del_ends])
    order = np.argsort(all_starts, kind="stable")
    all_starts = all_starts[order]
    all_ends = all_ends[order]

    merged: List[Tuple[int, int]] = []
    for s, e in zip(all_starts.tolist(), all_ends.tolist()):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            stats["adjacent_or_overlapping_ranges_merged"] += 1
        else:
            merged.append((s, e))

    return merged, stats


def compute_keep_spans(delete_ranges: List[Tuple[int, int]], total_len: int) -> List[Tuple[int, int]]:
    """Complement of delete_ranges (sorted, non-overlapping, inclusive-end) over
    [0, total_len). Returns half-open [start, end) spans to keep, in file order."""
    spans: List[Tuple[int, int]] = []
    prev_end = 0
    for (ds, de) in delete_ranges:
        if ds > prev_end:
            spans.append((prev_end, ds))
        prev_end = de + 1
    if prev_end < total_len:
        spans.append((prev_end, total_len))
    return spans


def _copy_spans_worker(
    in_path: str, out_path: str, dtype, spans_with_offsets: List[Tuple[int, int, int]]
) -> None:
    """Copies each [in_start, in_end) span from the input memmap to
    [out_start, out_start+len) in the output memmap. Runs in a worker process --
    opens its own handles, writes only to its own pre-assigned, disjoint region
    of the output file, so no locking is needed between workers."""
    in_arr = np.memmap(in_path, dtype=dtype, mode="r")
    out_arr = np.memmap(out_path, dtype=dtype, mode="r+")
    for (in_s, in_e, out_s) in spans_with_offsets:
        length = in_e - in_s
        out_arr[out_s : out_s + length] = in_arr[in_s:in_e]
    out_arr.flush()


def _partition_by_volume(
    spans_with_offsets: List[Tuple[int, int, int]], num_workers: int
) -> List[List[Tuple[int, int, int]]]:
    """Greedily groups spans so each worker gets roughly the same total token
    volume, not just the same span *count* -- avoids one worker stuck copying a
    few huge spans while others idle on many tiny ones."""
    total = sum(e - s for (s, e, _) in spans_with_offsets)
    target = max(1, total // num_workers)
    groups: List[List[Tuple[int, int, int]]] = []
    cur: List[Tuple[int, int, int]] = []
    cur_sum = 0
    for item in spans_with_offsets:
        cur.append(item)
        cur_sum += item[1] - item[0]
        if cur_sum >= target and len(groups) < num_workers - 1:
            groups.append(cur)
            cur, cur_sum = [], 0
    if cur:
        groups.append(cur)
    return groups


def write_cleaned_file(
    in_path: str,
    out_path: str,
    keep_spans: List[Tuple[int, int]],
    dtype=DTYPE,
    num_workers: int = 1,
) -> int:
    """Writes a new file at out_path containing exactly the tokens in
    keep_spans, in order. Returns the total number of tokens written."""
    lengths = [e - s for (s, e) in keep_spans]
    total_out = sum(lengths)
    offsets = np.cumsum([0] + lengths)[:-1].tolist() if lengths else []

    # Pre-allocate the output file at its exact final size before any writes.
    out_arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total_out,))
    del out_arr  # close; workers reopen with mode="r+"

    spans_with_offsets = [(s, e, off) for (s, e), off in zip(keep_spans, offsets)]
    num_workers = max(1, num_workers)

    if num_workers == 1 or len(spans_with_offsets) <= 1:
        _copy_spans_worker(in_path, out_path, dtype, spans_with_offsets)
    else:
        groups = _partition_by_volume(spans_with_offsets, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_copy_spans_worker, in_path, out_path, dtype, g) for g in groups if g]
            for fut in futures:
                fut.result()

    return total_out


def clean_translate_segments(
    in_path: str,
    out_path: str,
    translate_token: int,
    marker_token: int,
    eos_token: int,
    keep_count: int = 100_000,
    num_workers: int = 1,
    dtype=DTYPE,
    action_token_ids: Sequence[int] | None = None,
) -> dict:
    """End-to-end: scan in_path, compute what to delete, write the cleaned
    result to out_path (never touches in_path). Returns a stats dict.

    If action_token_ids is given (the full set of language/action tags),
    also deletes empty-source and empty-target segments unconditionally via
    compute_delete_ranges_full; otherwise uses the simpler marker-only
    compute_delete_ranges."""
    total_len = int(np.memmap(in_path, dtype=dtype, mode="r").shape[0])

    translate_pos, marker_pos, eos_pos = find_token_positions(
        in_path, total_len, (translate_token, marker_token, eos_token), num_workers, dtype
    )

    if action_token_ids:
        action_tag_pos = find_any_token_positions(in_path, total_len, action_token_ids, num_workers, dtype)
        delete_ranges, stats = compute_delete_ranges_full(
            translate_pos, marker_pos, eos_pos, action_tag_pos, keep_count
        )
    else:
        delete_ranges, stats = compute_delete_ranges(translate_pos, marker_pos, eos_pos, keep_count)

    keep_spans = compute_keep_spans(delete_ranges, total_len)
    total_out = write_cleaned_file(in_path, out_path, keep_spans, dtype, num_workers)

    stats["total_tokens_in"] = total_len
    stats["total_tokens_out"] = total_out
    stats["tokens_deleted"] = total_len - total_out
    return stats
