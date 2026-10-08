"""Stratified eval plan (training/eval_plan.py): torch-free, runs anywhere."""
import json

import numpy as np
import pytest

from training.eval_plan import (
    allocate_blocks, blocks_in, build_eval_plan, load_manifest, parse_manifest, read_windows,
    resolve_manifest_path, summarize_segments, systematic_indices,
)

B = 16  # block size used throughout


def manifest(spec, dtype="uint16"):
    """spec: [(name, n_tokens), ...] -> manifest dict with contiguous segments."""
    segs, cur = [], 0
    for name, n in spec:
        segs.append({"name": name, "start": cur, "end": cur + n})
        cur += n
    return {"dtype": dtype, "total_tokens": cur, "segments": segs}


def plan_for(spec, total=50, share=0.4, seed=7, block=B):
    return build_eval_plan(parse_manifest(manifest(spec)), block_size=block, total_blocks=total, old_share=share, seed=seed)


# ---------------------------------------------------------------- manifest

def test_manifest_round_trip_and_file_size_check(tmp_path):
    m = manifest([("old", 1000), ("zul", 200)])
    path = tmp_path / "m.json"
    path.write_text(json.dumps(m))
    got = load_manifest(str(path), file_tokens=1200)
    assert got.total_tokens == 1200 and [s.name for s in got.segments] == ["old", "zul"]
    with pytest.raises(ValueError, match="do not belong together"):
        load_manifest(str(path), file_tokens=1199)


@pytest.mark.parametrize("mutate", [
    lambda m: m["segments"].pop(0),                                  # does not start at 0
    lambda m: m["segments"][1].update(start=m["segments"][1]["start"] + 1),  # gap
    lambda m: m["segments"][1].update(name="old"),                   # duplicate name
    lambda m: m.update(total_tokens=m["total_tokens"] + 1),          # total disagrees with segments
    lambda m: m.update(dtype="uint32"),
    lambda m: m.update(segments=[]),
    lambda m: m["segments"][0].update(end=0),                        # empty segment
])
def test_bad_manifests_are_rejected(mutate):
    m = manifest([("old", 1000), ("zul", 200)])
    mutate(m)
    with pytest.raises(ValueError):
        parse_manifest(m)


def test_manifest_is_found_next_to_the_bin_when_the_configured_path_is_missing(tmp_path):
    (tmp_path / "validation_v2_manifest.json").write_text("{}")
    got = resolve_manifest_path("datasets/validation_v2_manifest.json", str(tmp_path / "validation_v2.bin"))
    assert got == str(tmp_path / "validation_v2_manifest.json")
    with pytest.raises(FileNotFoundError):
        resolve_manifest_path("nope.json", str(tmp_path / "validation_v2.bin"))


# ---------------------------------------------------------------- allocation

def test_allocation_splits_old_share_then_equal_languages():
    cap = {"old": 10_000, "zul": 100, "yor": 100, "hau": 100}
    alloc = allocate_blocks(500, 0.4, cap)
    assert alloc["old"] == 200 and sum(alloc.values()) == 500
    assert sorted(alloc[k] for k in ("zul", "yor", "hau")) == [100, 100, 100]  # all capped at 100 -> 300 total


def test_allocation_remainder_goes_to_the_first_languages_and_total_is_exact():
    cap = {"old": 1000, "a": 1000, "b": 1000, "c": 1000}
    alloc = allocate_blocks(100, 0.4, cap)  # 40 old, 60 / 3 = 20 each
    assert alloc == {"old": 40, "a": 20, "b": 20, "c": 20}
    alloc = allocate_blocks(101, 0.4, cap)  # round(40.4)=40 old, 61 -> 21,20,20
    assert alloc == {"old": 40, "a": 21, "b": 20, "c": 20}


def test_shortfall_of_a_small_language_is_redistributed_to_the_others():
    cap = {"old": 1000, "tiny": 5, "a": 1000, "b": 1000}
    alloc = allocate_blocks(100, 0.4, cap)  # 60 for languages; tiny holds 5 -> 55 split 28/27
    assert alloc["tiny"] == 5 and alloc["old"] == 40
    assert alloc["a"] + alloc["b"] == 55 and abs(alloc["a"] - alloc["b"]) <= 1
    assert sum(alloc.values()) == 100


def test_old_shortfall_goes_to_the_languages_and_full_languages_hand_back_to_old():
    assert allocate_blocks(100, 0.4, {"old": 10, "a": 1000, "b": 1000}) == {"old": 10, "a": 45, "b": 45}
    # languages can hold only 20 in total, so "old" takes the remaining 80 back
    assert allocate_blocks(100, 0.4, {"old": 1000, "a": 10, "b": 10}) == {"old": 80, "a": 10, "b": 10}


def test_allocation_caps_at_total_capacity_and_handles_missing_old_or_languages():
    assert allocate_blocks(100, 0.4, {"old": 7, "a": 3}) == {"old": 7, "a": 3}
    assert allocate_blocks(10, 0.4, {"a": 100, "b": 100}) == {"a": 5, "b": 5}
    assert allocate_blocks(10, 0.4, {"old": 100}) == {"old": 10}
    assert allocate_blocks(10, 0.0, {"old": 100, "a": 100})["old"] == 0
    assert allocate_blocks(10, 1.0, {"old": 100, "a": 100}) == {"old": 10, "a": 0}


@pytest.mark.parametrize("total,share,langs", [(500, 0.4, 13), (500, 0.4, 7), (37, 0.33, 5), (1, 0.4, 3), (500, 0.0, 4)])
def test_allocation_sums_to_total_when_capacity_allows(total, share, langs):
    cap = {"old": 100_000, **{f"l{i}": 40 + 3 * i for i in range(langs)}}
    alloc = allocate_blocks(total, share, cap)
    assert sum(alloc.values()) == total
    assert all(0 <= alloc[k] <= cap[k] for k in cap)


# ---------------------------------------------------------------- systematic spread

def test_systematic_indices_are_equally_spaced_across_the_whole_range():
    idx = systematic_indices(1000, 10, 0.37)
    assert len(idx) == len(set(idx)) == 10
    gaps = np.diff(idx)
    assert gaps.min() >= 99 and gaps.max() <= 101  # stride ~100, no clumps
    assert idx[0] < 100 and idx[-1] >= 900          # first and last decile both covered
    assert systematic_indices(5, 9, 0.5) == [0, 1, 2, 3, 4]  # k >= n -> everything
    assert systematic_indices(5, 0, 0.5) == [] and systematic_indices(0, 3, 0.5) == []


def test_plan_blocks_spread_over_each_segment_with_no_overlap():
    spec = [("old", B * 4000 + 1), ("zul", B * 300 + 1)]
    plan = plan_for(spec, total=100, share=0.4)
    old = np.array(plan.starts["old"])
    assert len(old) == 40
    assert np.all(np.diff(old) >= B)                                  # never overlapping
    assert np.diff(old).max() - np.diff(old).min() <= 2 * B           # equally spaced to within a block
    assert old.min() < 4000 * B / 40 and old.max() >= 4000 * B * 0.95  # spans the whole segment
    zul = np.array(plan.starts["zul"])
    assert len(zul) == 60 and zul.min() >= B * 4000 + 1 and np.all(np.diff(zul) >= B)


# ---------------------------------------------------------------- boundaries, determinism, totals

def test_blocks_never_cross_a_segment_boundary_including_the_target_token():
    spec = [("old", B * 50 + 5), ("zul", B * 20 + 1), ("yor", B * 9 + 15), ("hau", B * 3 + 2)]
    plan = plan_for(spec, total=400, share=0.4)  # asks for far more than the file holds -> every block
    cursor = 0
    for name, n in spec:
        start, end = cursor, cursor + n
        starts = plan.starts[name]
        assert len(starts) == blocks_in(n, B) == plan.capacity[name]
        assert all(start <= s and s + B + 1 <= end for s in starts)  # inputs AND the +1 target inside
        cursor = end


def test_plan_is_deterministic_in_the_seed_and_changes_with_it():
    spec = [("old", B * 5000 + 1), ("zul", B * 800 + 1), ("yor", B * 800 + 1)]
    a, b = plan_for(spec, seed=3), plan_for(spec, seed=3)
    assert a.starts == b.starts
    assert plan_for(spec, seed=4).starts != a.starts
    assert a.n_blocks == 50


def test_plan_total_is_exact_and_old_comes_first():
    spec = [("zul", B * 800 + 1), ("old", B * 5000 + 1), ("yor", B * 800 + 1)]  # old not first in the manifest
    plan = plan_for(spec, total=77, share=0.4)
    assert plan.segments[0] == "old" and plan.languages == ("zul", "yor")
    assert plan.n_blocks == 77 and plan.counts["old"] == 31


# ---------------------------------------------------------------- distribution over ranks

@pytest.mark.parametrize("world,bs", [(1, 4), (2, 4), (4, 12), (3, 5)])
def test_rank_steps_partition_the_plan_exactly_and_stay_in_lockstep(world, bs):
    spec = [("old", B * 900 + 1), ("zul", B * 70 + 1), ("yor", B * 31 + 1), ("hau", B * 400 + 1)]
    plan = plan_for(spec, total=123, share=0.4)
    per_rank = [plan.rank_steps(r, world, bs) for r in range(world)]
    assert len({len(s) for s in per_rank}) == 1  # same number of forwards on every rank
    seen = []
    for steps in per_rank:
        for seg, starts, real in steps:
            assert 1 <= len(starts) <= bs
            if real:
                seen += [(seg, s) for s in starts]
    expected = [(n, s) for n in plan.segments for s in plan.starts[n]]
    assert sorted(seen) == sorted(expected)  # every block scored exactly once, none twice
    for seg_steps in zip(*per_rank):  # step k is the same segment on every rank
        assert len({st.segment for st in seg_steps}) == 1


def test_a_rank_with_no_blocks_left_gets_a_zero_weight_dummy_step():
    plan = plan_for([("old", B * 100 + 1), ("zul", B * 3 + 1)], total=43, share=0.9)  # zul: 4 blocks? capacity 3
    steps = plan.rank_steps(rank=3, world=4, batch_size=2)
    zul_steps = [s for s in steps if s.segment == "zul"]
    assert zul_steps and not zul_steps[0].real and len(zul_steps[0].starts) == 1


# ---------------------------------------------------------------- reading + summary

def test_read_windows_uses_absolute_offsets_and_a_shifted_target():
    data = np.arange(1000, dtype=np.uint16)
    x, y = read_windows(data, [0, 333], B)
    assert x.shape == y.shape == (2, B) and np.all(y == x + 1) and x[1, 0] == 333


def test_summary_is_token_weighted_overall_and_unweighted_across_languages():
    # (loss x tok, ce x tok, tok, bytes)
    totals = {
        "old": (3.0 * 800, 2.9 * 800, 800, 4000),
        "zul": (5.0 * 100, 4.9 * 100, 100, 600),
        "yor": (7.0 * 100, 6.9 * 100, 100, 700),
    }
    s = summarize_segments(totals)
    assert s["val"] == pytest.approx((3.0 * 800 + 500 + 700) / 1000)
    assert s["val_old"] == pytest.approx(3.0)
    assert s["val_by_lang"] == {"zul": pytest.approx(5.0), "yor": pytest.approx(7.0)}
    assert s["val_lang_macro"] == pytest.approx(6.0)  # unweighted, not pulled toward "old"
    assert s["val_ce"] == pytest.approx((2.9 * 800 + 490 + 690) / 1000) and s["tokens"] == 1000


def test_summary_without_languages_or_old_or_with_empty_segments():
    s = summarize_segments({"old": (6.0, 5.0, 2, 10)})
    assert s["val"] == 3.0 and "val_lang_macro" not in s and s["val_by_lang"] == {}
    s = summarize_segments({"zul": (4.0, 4.0, 2, 8), "yor": (0.0, 0.0, 0, 0)})
    assert "val_old" not in s and s["val_by_lang"] == {"zul": 2.0}
