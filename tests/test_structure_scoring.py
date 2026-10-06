"""Tests for rl/structure.py's scaffold scoring, and the contract with data-gen's columns.

The design question these pin down: the task_plan is a SET of things a turn does, so it is scored
order-agnostically by F-beta, with recall weighted above precision because missing a task is a worse error
than naming one too many. Order IS measured -- by Kendall tau over the shared verbs -- but earns nothing by
default, because folding it in would make a correct-set-wrong-order plan score like a wrong-set one.
"""
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.structure import (  # noqa: E402
    DEFAULT_WEIGHTS, format_score, plan_f1, plan_order, structure_of,
)

EXP = ("<|input_lang|><eng><task_plan><|is_financial|><|RAG|><qa></task_plan>"
       "<|target_lang|><zul><response>")


def test_missing_a_verb_costs_more_than_adding_one():
    """beta=2 weights recall 4x. "Not catching all the tasks is more important than the order" is a
    statement about recall, and F-beta is how it is said formally."""
    missing = plan_f1(["<a>", "<b>"], ["<a>", "<b>", "<c>"])
    extra = plan_f1(["<a>", "<b>", "<c>", "<d>"], ["<a>", "<b>", "<c>"])
    assert missing < extra, f"missing {missing:.3f} should cost more than extra {extra:.3f}"
    assert missing == pytest.approx(0.714, abs=0.01)
    assert extra == pytest.approx(0.938, abs=0.01)


def test_plan_score_ignores_order():
    assert plan_f1(["<a>", "<b>", "<c>"], ["<c>", "<a>", "<b>"]) == 1.0


def test_order_is_measured_separately():
    assert plan_order(["<a>", "<b>", "<c>"], ["<a>", "<b>", "<c>"]) == 1.0
    assert plan_order(["<c>", "<b>", "<a>"], ["<a>", "<b>", "<c>"]) == 0.0
    # and it is UNDEFINED, not zero, for a single shared verb -- scoring that as failure would punish a
    # correct one-verb plan
    assert plan_order(["<a>"], ["<a>"]) is None
    assert plan_order([], ["<a>", "<b>"]) is None


def test_order_earns_nothing_by_default():
    """Measured and reported, but not rewarded until there is evidence about what it is worth."""
    assert DEFAULT_WEIGHTS["plan_order"] == 0.0
    straight = format_score(EXP + "body", EXP)
    swapped = format_score(
        EXP.replace("<|is_financial|><|RAG|><qa>", "<qa><|RAG|><|is_financial|>") + "body", EXP)
    assert straight["score"] == pytest.approx(swapped["score"]), "order must not move the score yet"
    assert straight["plan_order"] == 1.0 and swapped["plan_order"] == 0.0, "but it must be reported"


def test_order_can_be_given_weight():
    straight = format_score(EXP + "body", EXP, weights={**DEFAULT_WEIGHTS, "plan_order": 0.2})
    swapped = format_score(
        EXP.replace("<|is_financial|><|RAG|><qa>", "<qa><|RAG|><|is_financial|>") + "body", EXP,
        weights={**DEFAULT_WEIGHTS, "plan_order": 0.2})
    assert straight["score"] > swapped["score"]


def test_the_wrong_target_language_closes_the_gate():
    """Without the gate a policy could earn content reward in the wrong language, which is the one failure
    that must not be partially rewarded."""
    right = format_score(EXP + "Izinga li-1,580.", EXP)
    wrong = format_score(EXP.replace("<|target_lang|><zul>", "<|target_lang|><yor>")
                         + "Ìwọ̀n náà jẹ́ 1,580.", EXP)
    assert right["gate"] is True
    assert wrong["gate"] is False
    assert wrong["score"] < right["score"]


def test_prose_with_no_scaffold_scores_zero():
    f = format_score("Izinga lokushintshanisa li-1,580.", EXP)
    assert f["score"] == 0.0 and f["gate"] is False


def test_structure_of_round_trips():
    """What data-gen writes into `expected_structure` must score as a match against the turn it came from."""
    full = ("<|input_lang|><eng><think>needs the rate</think><task_plan><|is_financial|><|RAG|><qa>"
            "</task_plan><|target_lang|><zul><response>Izinga li-1,580.")
    s = structure_of(full)
    assert "<<" not in s, "the language tags already carry their brackets"
    assert s == EXP
    assert format_score(full, s)["score"] > 0.95


def test_structure_of_handles_a_tool_calling_turn():
    """A tool call has no target language or response, so the scaffold stops where the turn does."""
    s = structure_of("<|input_lang|><eng><task_plan><|plan|><|RAG|></task_plan>"
                     '<tool_call>search_documents\n{"query":"x"}</tool_call>')
    assert "<|target_lang|>" not in s and "<response>" not in s
    assert "<task_plan><|plan|><|RAG|></task_plan>" in s


def test_every_plan_verb_is_a_real_token():
    """A verb outside the tokenizer costs junk sub-words and teaches a token the model cannot emit. <|hate|>
    is 52115 -- above vocab_size 52050 -- and <POS> does not exist at all."""
    import json
    import glob
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data-gen"))
    from assemble import VALID_VERBS, _UNUSABLE_TAGS
    snaps = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                                 "models--BeardedMonster--SabiYarn-32k/snapshots/*")))
    if not snaps:
        pytest.skip("tokenizer not cached")
    tk = json.loads((Path(snaps[-1]) / "tokenizer.json").read_text(encoding="utf-8"))
    ids = dict(tk["model"]["vocab"])
    ids.update({a["content"]: a["id"] for a in tk.get("added_tokens", [])})
    for v in VALID_VERBS:
        assert v in ids, f"{v} is not in the tokenizer"
        assert ids[v] < 52050, f"{v} is id {ids[v]}, at or above vocab_size 52050"
    assert "<|hate|>" in _UNUSABLE_TAGS and "<POS>" in _UNUSABLE_TAGS
    assert "<|hate|>" not in VALID_VERBS and "<POS>" not in VALID_VERBS
