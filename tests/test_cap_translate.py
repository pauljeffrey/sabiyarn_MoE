import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data-gen"))
from sft_new import cap_translate as C  # noqa: E402


def _row(src, q, a="This is a sentence of normal length.", b="Hierdie is n sin van normale lengte."):
    return {"input": a, "response": b, "source_lang": "eng", "targ_lang": "afr", "source_dataset": src, "quality": q}


def test_curated_and_high_judge_scores_win_and_opus_ranks_like_a_seven():
    rows = [_row("African-Languages-Lab/multi-open", 6), _row("African-Languages-Lab/multi-open", 10), _row("sentence-transformers/parallel-sentences-opus-100", None),
            _row("HuggingFaceFW/finetranslations", 2.5), _row("African-Languages-Lab/multi-open", 8)]
    keep = C.select_top(C.rank_keys(rows, 0.0), 3)
    assert keep.tolist() == [False, True, False, True, True]


def test_inside_a_tier_misaligned_lengths_lose():
    ok = _row("African-Languages-Lab/multi-open", 10)
    bad = _row("African-Languages-Lab/multi-open", 10, b="x")
    keep = C.select_top(C.rank_keys([bad, ok], float(np.log(len(ok["response"]) / len(ok["input"])))), 1)
    assert keep.tolist() == [False, True]


def test_selection_is_deterministic():
    rows = [_row("African-Languages-Lab/multi-open", 10, a=f"sentence number {i} here", b=f"sin nommer {i} hier") for i in range(50)]
    k1 = C.select_top(C.rank_keys(rows, 0.0), 20)
    assert (k1 == C.select_top(C.rank_keys(rows, 0.0), 20)).all() and k1.sum() == 20
