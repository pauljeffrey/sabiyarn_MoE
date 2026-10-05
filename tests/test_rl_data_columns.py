"""The DPO trainer must be able to read data-gen's own records.

rl/data.py did `rec["chosen"]`, which KeyErrors on every record data-gen has ever produced -- it writes
`response_1`/`response_2` plus a `ranking`. The only bridge was judge_gen.py, a second paid pass that reduces
the candidates to chosen/rejected. With no judge model in the pipeline that bridge does not exist, so the
ordering the generator already gave us has to be read directly.

Torch is stubbed: these functions are pure dict logic and the test should not need a training environment.
"""
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.data import preference_pair, score_margin  # noqa: E402


def test_reads_data_gens_own_columns():
    rec = {"response_1": "good", "response_2": "bad", "ranking": ["best", "worst"]}
    assert preference_pair(rec) == ("good", "bad")


def test_ranking_is_authoritative_not_column_order():
    """postprocess_gen sorts the responses and writes the order it used, so a reversed ranking means the
    better answer is in response_2. Ignoring it would train on the pair backwards."""
    rec = {"response_1": "bad", "response_2": "good", "ranking": ["worst", "best"]}
    assert preference_pair(rec) == ("good", "bad")


def test_public_dpo_shape_still_works():
    """UltraFeedback, HH-RLHF and anything judged use chosen/rejected."""
    assert preference_pair({"chosen": "c", "rejected": "r"}) == ("c", "r")


def test_equal_quality_is_not_a_preference():
    assert preference_pair({"response_1": "a", "response_2": "b", "ranking": ["best", "best"]}) is None
    assert preference_pair({"response_1": "a", "response_2": "b", "ranking": ["partial", "partial"]}) is None


def test_a_missing_side_is_unusable():
    assert preference_pair({"response_1": "a"}) is None
    assert preference_pair({"response_2": "b"}) is None
    assert preference_pair({}) is None


def test_no_ranking_falls_back_to_column_order():
    """response_1 is the better side by construction when postprocess wrote no ranking."""
    assert preference_pair({"response_1": "a", "response_2": "b"}) == ("a", "b")


def test_score_margin_reads_the_generators_own_scores():
    """A pair whose sides scored within a point or two of each other teaches almost nothing and its gradient
    is noise. The generator already told us how far apart it thought they were."""
    assert score_margin({"scores": [8.0, 2.0]}) == 6.0
    assert score_margin({"scores": [5.0, 4.5]}) == 0.5
    assert score_margin({}) is None
    assert score_margin({"scores": [7.0]}) is None
    assert score_margin({"scores": ["x", "y"]}) is None
