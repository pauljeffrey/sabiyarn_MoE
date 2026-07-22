import numpy as np
import pytest

from data.clean_translate_segments import (
    clean_translate_segments,
    compute_delete_ranges,
    compute_delete_ranges_full,
    compute_keep_spans,
    find_any_token_positions,
    find_token_positions,
    write_cleaned_file,
)

TRANSLATE = 50
TWI = 60
ENG = 70
IBO = 80
EOS = 99
ACTION_TAGS = (TWI, ENG, IBO)


def _write_bin(tmp_path, name, tokens):
    path = str(tmp_path / name)
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(tokens),))
    arr[:] = tokens
    arr.flush()
    del arr
    return path


# The user's own worked example:
#   "hello, how are you? ...... </s><translate> i love rice <twi>. ... </s>
#    <translate> mo feran iresi <eng> .... </s>"
# should become (keep_count=0, i.e. delete every qualifying segment):
#   "hello, how are you? ...... </s><translate> mo feran iresi <eng> .... </s>"
def _users_example_tokens():
    text1 = [1, 2, 3]  # "hello, how are you? ......"
    seg1 = [TRANSLATE, 10, 11, 12, TWI, 13, 14, 15, EOS]  # "<translate> i love rice <twi>. ... </s>"
    seg2 = [TRANSLATE, 20, 21, 22, ENG, 23, 24, 25, 26, EOS]  # "<translate> mo feran iresi <eng> .... </s>"
    return text1 + [EOS] + seg1 + seg2, text1 + [EOS] + seg2


@pytest.mark.parametrize("num_workers", [1, 4])
def test_users_worked_example_end_to_end(tmp_path, num_workers):
    full_tokens, expected_tokens = _users_example_tokens()
    in_path = _write_bin(tmp_path, "in.bin", full_tokens)
    out_path = str(tmp_path / "out.bin")

    stats = clean_translate_segments(
        in_path, out_path, TRANSLATE, TWI, EOS, keep_count=0, num_workers=num_workers,
    )

    result = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert result.tolist() == expected_tokens
    assert stats["qualifying_segments"] == 1
    assert stats["deleted_segments"] == 1
    assert stats["kept_segments"] == 0
    assert stats["tokens_deleted"] == len(full_tokens) - len(expected_tokens)


def test_non_qualifying_segment_never_touched(tmp_path):
    # Only the <eng>-tagged segment -- no <twi> anywhere -- must survive untouched.
    text1 = [1, 2, 3]
    seg2 = [TRANSLATE, 20, 21, 22, ENG, 23, 24, 25, 26, EOS]
    tokens = text1 + [EOS] + seg2
    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_path = str(tmp_path / "out.bin")

    stats = clean_translate_segments(in_path, out_path, TRANSLATE, TWI, EOS, keep_count=0)

    result = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert result.tolist() == tokens  # nothing deleted
    assert stats["qualifying_segments"] == 0
    assert stats["tokens_deleted"] == 0


def test_keep_count_keeps_first_n_qualifying_segments(tmp_path):
    # Three <twi>-tagged segments; keep_count=2 should keep the first two and
    # delete only the third.
    def seg(a, b):
        return [TRANSLATE, a, TWI, b, EOS]

    tokens = [1, 2] + seg(10, 11) + seg(20, 21) + seg(30, 31) + [9, 9]
    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_path = str(tmp_path / "out.bin")

    stats = clean_translate_segments(in_path, out_path, TRANSLATE, TWI, EOS, keep_count=2)

    expected = [1, 2] + seg(10, 11) + seg(20, 21) + [9, 9]
    result = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert result.tolist() == expected
    assert stats["qualifying_segments"] == 3
    assert stats["kept_segments"] == 2
    assert stats["deleted_segments"] == 1


def test_keep_count_larger_than_available_deletes_nothing(tmp_path):
    def seg(a, b):
        return [TRANSLATE, a, TWI, b, EOS]

    tokens = [1, 2] + seg(10, 11) + seg(20, 21)
    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_path = str(tmp_path / "out.bin")

    stats = clean_translate_segments(in_path, out_path, TRANSLATE, TWI, EOS, keep_count=100)

    result = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert result.tolist() == tokens
    assert stats["deleted_segments"] == 0
    assert stats["tokens_deleted"] == 0


def test_trailing_unterminated_translate_left_untouched(tmp_path):
    # A <translate>...<twi>... with no closing </s> at all (end of file) can't
    # be safely deleted -- its extent is unknown.
    tokens = [1, 2, EOS, TRANSLATE, 10, TWI, 11, 12]  # no trailing EOS
    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_path = str(tmp_path / "out.bin")

    stats = clean_translate_segments(in_path, out_path, TRANSLATE, TWI, EOS, keep_count=0)

    result = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert result.tolist() == tokens  # untouched
    assert stats["trailing_unterminated_translate"] == 1
    assert stats["qualifying_segments"] == 0


def test_order_preserved_with_multiple_workers_matches_single_worker(tmp_path):
    rng = np.random.default_rng(0)
    pieces = [rng.integers(1, 40, size=rng.integers(3, 12)).tolist() for _ in range(400)]

    tokens: list[int] = []
    for i, piece in enumerate(pieces):
        tokens.extend(piece)
        if i % 3 == 1:
            tag = TWI if i % 2 == 0 else ENG
            tokens.extend([TRANSLATE, *piece, tag, *piece, EOS])
        tokens.append(EOS)

    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_single = str(tmp_path / "out_single.bin")
    out_multi = str(tmp_path / "out_multi.bin")

    stats1 = clean_translate_segments(in_path, out_single, TRANSLATE, TWI, EOS, keep_count=5, num_workers=1)
    stats2 = clean_translate_segments(in_path, out_multi, TRANSLATE, TWI, EOS, keep_count=5, num_workers=6)

    r1 = np.memmap(out_single, dtype=np.uint16, mode="r")
    r2 = np.memmap(out_multi, dtype=np.uint16, mode="r")
    assert r1.tolist() == r2.tolist()
    assert stats1["qualifying_segments"] == stats2["qualifying_segments"]
    assert stats1["deleted_segments"] == stats2["deleted_segments"]


def test_find_token_positions_matches_across_worker_counts(tmp_path):
    tokens = [TRANSLATE, 1, TWI, 2, EOS] * 50
    in_path = _write_bin(tmp_path, "in.bin", tokens)
    p1 = find_token_positions(in_path, len(tokens), (TRANSLATE, TWI, EOS), num_workers=1)
    p8 = find_token_positions(in_path, len(tokens), (TRANSLATE, TWI, EOS), num_workers=8)
    for a, b in zip(p1, p8):
        assert a.tolist() == b.tolist()


def test_compute_keep_spans_complement():
    spans = compute_keep_spans([(4, 12), (20, 25)], total_len=30)
    assert spans == [(0, 4), (13, 20), (26, 30)]


def test_compute_keep_spans_no_deletions():
    assert compute_keep_spans([], total_len=10) == [(0, 10)]


def test_write_cleaned_file_respects_keep_spans(tmp_path):
    tokens = list(range(100, 120))
    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_path = str(tmp_path / "out.bin")
    spans = [(0, 5), (10, 15)]
    total = write_cleaned_file(in_path, out_path, spans, num_workers=3)
    result = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert total == 10
    assert result.tolist() == tokens[0:5] + tokens[10:15]


def test_compute_delete_ranges_merges_overlapping_malformed_segments():
    # Two <translate> tags sharing the same terminating </s> (malformed,
    # nested data) -- both qualify and both are past keep_count=0, so their
    # overlapping ranges must be merged into one, not double-counted.
    translate_pos = [2, 4]
    marker_pos = [3, 5]
    eos_pos = [10]
    ranges, stats = compute_delete_ranges(translate_pos, marker_pos, eos_pos, keep_count=0)
    assert ranges == [(2, 10)]
    assert stats["malformed_overlapping_segments_merged"] == 1


# --- empty-source / empty-target detection (compute_delete_ranges_full) ---

def test_find_any_token_positions_across_worker_counts(tmp_path):
    tokens = [TRANSLATE, 1, TWI, 2, ENG, 3, EOS] * 20
    in_path = _write_bin(tmp_path, "in.bin", tokens)
    p1 = find_any_token_positions(in_path, len(tokens), ACTION_TAGS, num_workers=1)
    p8 = find_any_token_positions(in_path, len(tokens), ACTION_TAGS, num_workers=8)
    assert p1.tolist() == p8.tolist()
    # every TWI and ENG position should show up, nothing else
    expected = sorted(i for i, t in enumerate(tokens) if t in ACTION_TAGS)
    assert p1.tolist() == expected


def test_empty_source_segment_deleted_unconditionally():
    # <translate> <ibo>: ... </s>  -- nothing between <translate> and <ibo>.
    translate_pos = [0]
    eos_pos = [5]
    action_tag_pos = [1]  # <ibo> immediately after <translate>
    marker_pos: list[int] = []  # no <twi> at all -- must still be deleted
    ranges, stats = compute_delete_ranges_full(
        translate_pos, marker_pos, eos_pos, action_tag_pos, keep_count=100_000,
    )
    assert ranges == [(0, 5)]
    assert stats["empty_source_segments"] == 1
    assert stats["empty_target_segments"] == 0


def test_empty_target_segment_deleted_unconditionally():
    # <translate> content <eng> </s>  -- nothing between <eng> and </s>.
    translate_pos = [0]
    action_tag_pos = [3]  # <eng> at position 3
    eos_pos = [4]  # </s> immediately after <eng>
    marker_pos: list[int] = []
    ranges, stats = compute_delete_ranges_full(
        translate_pos, marker_pos, eos_pos, action_tag_pos, keep_count=100_000,
    )
    assert ranges == [(0, 4)]
    assert stats["empty_source_segments"] == 0
    assert stats["empty_target_segments"] == 1


def test_non_empty_segment_with_no_marker_untouched():
    # <translate> content <eng> more_content </s> -- both sides non-empty,
    # no <twi> -- must survive untouched regardless of keep_count.
    translate_pos = [0]
    action_tag_pos = [3]
    eos_pos = [6]
    marker_pos: list[int] = []
    ranges, stats = compute_delete_ranges_full(
        translate_pos, marker_pos, eos_pos, action_tag_pos, keep_count=0,
    )
    assert ranges == []
    assert stats["empty_source_segments"] == 0
    assert stats["empty_target_segments"] == 0
    assert stats["qualifying_thinning_segments"] == 0


def test_empty_target_that_is_also_the_marker_does_not_consume_keep_slot():
    # <translate> content <twi> </s> -- empty target, and the tag IS the
    # marker token itself. Must be deleted via the empty-target rule and
    # must NOT be counted as a kept marker-thinning segment.
    translate_pos = [0]
    action_tag_pos = [2]  # <twi> at position 2, also used as marker_pos
    eos_pos = [3]  # </s> immediately after <twi>
    marker_pos = [2]
    ranges, stats = compute_delete_ranges_full(
        translate_pos, marker_pos, eos_pos, action_tag_pos, keep_count=100_000,
    )
    assert ranges == [(0, 3)]
    assert stats["empty_target_segments"] == 1
    assert stats["qualifying_thinning_segments"] == 0  # excluded, not double-counted
    assert stats["kept_thinning_segments"] == 0


def test_end_to_end_users_empty_source_and_target_examples(tmp_path):
    # Faithful to the two examples given: one empty-source (<translate><ibo>...)
    # and one empty-target (...<eng></s>), mixed with a normal, healthy segment
    # that must survive.
    empty_source = [TRANSLATE, IBO, 1, 2, 3, EOS]
    empty_target = [TRANSLATE, 4, 5, 6, ENG, EOS]
    healthy = [TRANSLATE, 7, 8, TWI, 9, 10, EOS]
    tokens = [100, 101] + empty_source + empty_target + healthy + [200]

    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_path = str(tmp_path / "out.bin")

    stats = clean_translate_segments(
        in_path, out_path, TRANSLATE, TWI, EOS, keep_count=100_000,
        num_workers=1, action_token_ids=ACTION_TAGS,
    )

    expected = [100, 101] + healthy + [200]
    result = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert result.tolist() == expected
    assert stats["empty_source_segments"] == 1
    assert stats["empty_target_segments"] == 1
    assert stats["kept_thinning_segments"] == 1  # the healthy <twi> segment, under keep_count


@pytest.mark.parametrize("num_workers", [1, 5])
def test_combined_empty_and_thinning_matches_across_worker_counts(tmp_path, num_workers):
    rng = np.random.default_rng(1)

    def make_segment(kind: str) -> list[int]:
        body = rng.integers(1, 40, size=rng.integers(2, 6)).tolist()
        if kind == "empty_source":
            return [TRANSLATE, IBO, *body, EOS]
        if kind == "empty_target":
            return [TRANSLATE, *body, ENG, EOS]
        if kind == "twi_thin":
            return [TRANSLATE, *body, TWI, *body, EOS]
        return [TRANSLATE, *body, ENG, *body, EOS]  # healthy, non-thinning

    kinds = (["empty_source", "empty_target", "twi_thin", "healthy"] * 30)
    rng.shuffle(kinds)  # type: ignore[arg-type]

    tokens: list[int] = []
    for k in kinds:
        tokens.extend(make_segment(k))

    in_path = _write_bin(tmp_path, "in.bin", tokens)
    out_single = str(tmp_path / "out_single.bin")
    out_multi = str(tmp_path / "out_multi.bin")

    stats1 = clean_translate_segments(
        in_path, out_single, TRANSLATE, TWI, EOS, keep_count=5,
        num_workers=1, action_token_ids=ACTION_TAGS,
    )
    stats2 = clean_translate_segments(
        in_path, out_multi, TRANSLATE, TWI, EOS, keep_count=5,
        num_workers=num_workers, action_token_ids=ACTION_TAGS,
    )

    r1 = np.memmap(out_single, dtype=np.uint16, mode="r")
    r2 = np.memmap(out_multi, dtype=np.uint16, mode="r")
    assert r1.tolist() == r2.tolist()
    assert stats1["empty_source_segments"] == stats2["empty_source_segments"] == kinds.count("empty_source")
    assert stats1["empty_target_segments"] == stats2["empty_target_segments"] == kinds.count("empty_target")
    assert stats1["kept_thinning_segments"] == stats2["kept_thinning_segments"] == 5
    assert stats1["deleted_thinning_segments"] == stats2["deleted_thinning_segments"] == kinds.count("twi_thin") - 5
