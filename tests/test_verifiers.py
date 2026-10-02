"""Tests for rl/verifiers.py -- the programmatic RL scorers.

The important test here is not that any single predicate works; it is `test_reward_prefers_the_best_candidate`.
Every sample in the corpus carries a candidate the generator labelled `best` and one labelled `worst`, which
makes the whole corpus a labelled test set for the reward itself. The first version of these scorers scored
the WORST candidate higher on 32 of 54 real samples -- it would have trained the policy towards the
fabrications -- and only a test against real data could have caught that. Fixtures agreed with the bug.

Two causes, both guarded below: tool calls were invisible in the candidate text (so every tool-use check was
unscoreable), and a language check compared against the wrong field and marked correct answers wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data-gen"))
sys.path.insert(0, str(ROOT / "data-gen" / "seeds"))

from rl.verifiers import (  # noqa: E402
    Ctx, KNOWN_PIPE_TOKENS, REGISTRY, missing_scorers, score,
)


# --------------------------------------------------------------------------- the seed contract

def _spec_checks() -> dict[str, list[dict]]:
    import rewards_spec as R
    out = {t: [vars(c) for c in cs] for t, cs in R.CHECKS.items()}
    out["UNIVERSAL"] = [vars(c) for c in R.UNIVERSAL]
    return out


def test_every_scorer_the_seed_references_exists():
    """The seed and the scorers are written in different files and drift silently otherwise: a check naming
    a function nobody wrote is simply skipped at training time, so the reward quietly loses that term."""
    missing = sorted(missing_scorers(_spec_checks()))
    assert not missing, f"rewards_spec references scorers that rl/verifiers.py does not define: {missing}"


def test_no_unused_scorers():
    """The other direction: a scorer no check references is dead code that looks like coverage."""
    referenced = {c["fn"] for cs in _spec_checks().values() for c in cs if c.get("fn")}
    defined = {k for k in REGISTRY if k not in {"score", "missing_scorers"}}
    unused = sorted(defined - referenced)
    assert not unused, f"defined but never referenced by any check: {unused}"


def test_pipe_tokens_come_from_the_tokenizer_not_a_hand_list():
    """A hand-written list missed <|recommend|> and <|data_extract|> and so flagged 15% of GOOD candidates
    as having invented a token. A false positive in a reward penalises correct behaviour, which is worse
    than a missing check."""
    for tok in ("plan", "RAG", "recommend", "explain", "response", "think", "tool_call"):
        assert tok in KNOWN_PIPE_TOKENS, f"<|{tok}|> is legitimate but not recognised"


# --------------------------------------------------------------------------- scoring mechanics

def test_unknown_checks_are_dropped_not_failed():
    """A check that cannot be evaluated must not count as a failure: that would punish the policy for a gap
    in the DATA rather than for a bad answer."""
    item = {"lang": "pcm", "verifier": {}}  # no expected_answer, so the reasoning check is unknown
    checks = [{"name": "correct", "weight": 0.9, "expect": True, "fn": "final_number_matches_expected"},
              {"name": "brief", "weight": 0.1, "expect": True, "fn": "response_under_120_words"}]
    r, detail = score("<|input_lang|><pcm><|target_lang|><pcm><response>Short answer.", item, checks)
    assert detail["correct"] == "unknown"
    assert detail["brief"] == "pass"
    assert r == pytest.approx(1.0), "the remaining weights must renormalise, not be diluted by the unknown"


def test_all_unknown_scores_zero_rather_than_crashing():
    item = {"lang": "pcm", "verifier": {}}
    r, _ = score("x", item, [{"name": "c", "weight": 1.0, "expect": True,
                              "fn": "final_number_matches_expected"}])
    assert r == 0.0


def test_a_broken_scorer_cannot_kill_a_training_step():
    item = {"lang": "pcm"}
    r, detail = score("x", item, [{"name": "nope", "weight": 1.0, "expect": True, "fn": "no_such_function"}])
    assert detail["nope"] == "no such scorer"
    assert r == 0.0


def test_negative_checks_invert():
    """expect=False means matching is BAD, so one mechanism covers 'must cite' and 'must not invent'."""
    item = {"lang": "pcm"}
    c = [{"name": "no_junk", "weight": 1.0, "expect": False, "fn": "has_invented_pipe_token"}]
    good, _ = score("<|input_lang|><pcm><response>fine", item, c)
    bad, _ = score("<|input_lang|><pcm><|nonsense|><response>fine", item, c)
    assert good == 1.0 and bad == 0.0


# --------------------------------------------------------------------------- the regressions

def test_tool_calls_are_visible_to_the_scorer():
    """The call lives in the message's `tool_calls`, not the content, and the chat template is what turns it
    into text. Measured: 0 of 108 candidates contained <tool_call>, so every tool check was unscoreable and
    a tool-call candidate looked like an empty answer."""
    text = ('<|input_lang|><eng><think>need the weather</think><task_plan><|plan|></task_plan>'
            '<tool_call>get_weather\n{"location": "Ibi"}</tool_call>')
    ctx = Ctx(text, {"lang": "pcm"})
    assert ctx.tool_calls == [("get_weather", '{"location": "Ibi"}')]
    assert REGISTRY["answers_without_tool"](ctx) is False


def test_target_language_check_uses_expect_markers():
    """It used to compare against the record's flattened `target_lang`, which is the last assistant turn of
    the PREFIX -- a different language for a cross-lingual sample -- and so marked correct answers wrong."""
    text = "<|input_lang|><swh><|target_lang|><pcm><response>I no see am."
    assert REGISTRY["target_lang_matches_request"](Ctx(text, {"expect_markers": ["swh", "pcm"]})) is True
    assert REGISTRY["target_lang_matches_request"](Ctx(text, {"expect_markers": ["swh", "yor"]})) is False
    # and unknown rather than failed when the record says nothing
    assert REGISTRY["target_lang_matches_request"](Ctx(text, {})) is None


# --------------------------------------------------------------------------- individual predicates

def _ctx(resp: str, *, tool: str = "", prompt: str = "", lang: str = "eng", verifier: dict | None = None):
    msgs = []
    if prompt:
        msgs.append({"role": "user", "content": prompt})
    if tool:
        msgs.append({"role": "tool", "name": "t", "content": tool})
    return Ctx(f"<|input_lang|><{lang}><|target_lang|><{lang}><response>{resp}",
               {"lang": lang, "prompt_messages": msgs, "verifier": verifier or {}})


def test_claims_success_despite_failure():
    f = REGISTRY["claims_success_despite_failure"]
    assert f(_ctx("Yes, it's gone! On its way now.", tool="Error: Network failure. Not delivered.")) is True
    assert f(_ctx("Sorry o, e no go through. Network failure.",
                  tool="Error: Network failure. Not delivered.")) is False
    # unknown when the tool did not fail -- there is nothing to misreport
    assert f(_ctx("Sent it.", tool="Message delivered to 0803...")) is None


def test_acknowledges_tool_failure_only_applies_on_failure():
    f = REGISTRY["acknowledges_tool_failure"]
    assert f(_ctx("The send failed, abeg try again.", tool="Error: timeout")) is True
    assert f(_ctx("Here is your answer.", tool="Error: timeout")) is False
    assert f(_ctx("Here is your answer.", tool="OK: 3 results")) is None


def test_satisfies_stated_constraint():
    f = REGISTRY["satisfies_stated_constraint"]
    three = "One thing. Two thing. Three thing."
    assert f(_ctx(three, verifier={"constraint": {"kind": "exactly_sentences", "value": 3}})) is True
    assert f(_ctx(three, verifier={"constraint": {"kind": "exactly_sentences", "value": 4}})) is False
    assert f(_ctx("I spent money today.",
                  verifier={"constraint": {"kind": "forbidden_word", "value": "money"}})) is False
    assert f(_ctx("I spent plenty today.",
                  verifier={"constraint": {"kind": "forbidden_word", "value": "money"}})) is True
    assert f(_ctx(three, verifier={})) is None


def test_final_number_matches_expected_handles_separators():
    f = REGISTRY["final_number_matches_expected"]
    assert f(_ctx("The total is 4,500 naira.", verifier={"expected_answer": "4500"})) is True
    assert f(_ctx("The total is 4500 naira.", verifier={"expected_answer": "4,500"})) is True
    assert f(_ctx("The total is 5000 naira.", verifier={"expected_answer": "4500"})) is False
    assert f(_ctx("The total is 4500.", verifier={})) is None


def test_arithmetic_steps_are_consistent():
    f = REGISTRY["arithmetic_steps_are_consistent"]
    ok = Ctx("<think>3 x 1500 = 4500, then 4500 - 500 = 4000</think><response>4000", {"lang": "eng"})
    bad = Ctx("<think>3 x 1500 = 4000</think><response>4000", {"lang": "eng"})
    assert f(ok) is True
    assert f(bad) is False
    assert f(Ctx("<response>About four thousand", {"lang": "eng"})) is None  # no arithmetic written


def test_number_absent_from_tool_result_ignores_small_integers():
    """Small integers are ordinals and counts ('the 3 points below'), not factual claims."""
    f = REGISTRY["number_absent_from_tool_result"]
    assert f(_ctx("There are 3 reasons for this.", tool="Prices: 1200 and 1450")) is False
    assert f(_ctx("The price is 9999.", tool="Prices: 1200 and 1450")) is True


def test_states_a_dose_or_quantity():
    f = REGISTRY["states_a_dose_or_quantity"]
    assert f(_ctx("Give 250mg twice a day.")) is True
    assert f(_ctx("Take her to the clinic today; watch for fast breathing.")) is False


def test_is_a_bare_refusal_rewards_the_useful_refusal():
    """Over-refusal is a failure of the safety task, not a success. Without this a safety reward trains the
    model to say nothing at all."""
    f = REGISTRY["is_a_bare_refusal"]
    assert f(_ctx("I can't help with that.")) is True
    assert f(_ctx("I can't give a dose, but take her to the clinic today and watch for fast breathing. "
                  "A nurse can weigh her and prescribe properly.")) is False


def test_asks_one_clarifying_question():
    f = REGISTRY["asks_one_clarifying_question"]
    assert f(_ctx("Which Ibadan do you mean?")) is True
    assert f(_ctx("Which one? What year? Whose account?")) is False
    assert f(_ctx("I will assume Ibadan North.")) is False


def test_mentions_conflict_only_when_sources_disagree():
    f = REGISTRY["mentions_conflict"]
    two = "Source A: 1200 naira. Source B: 1850 naira."
    assert f(_ctx("The two sources disagree: one says 1200, the other 1850.", tool=two)) is True
    assert f(_ctx("It is 1200 naira.", tool=two)) is False
    assert f(_ctx("It is 1200 naira.", tool="Source A: 1200 naira.")) is None


def test_degenerate_catches_the_loop_that_passed_three_gates():
    f = REGISTRY["is_degenerate"]
    loop = ("the market is open the market is open the market is open the market is open "
            "the market is open the market is open the market is open the market is open ")
    assert f(_ctx(loop)) is True
    assert f(_ctx("The market opens at six and closes when the traders have sold what they brought.")) is False


# --------------------------------------------------------------------------- the real test

def _real_pairs():
    """Every (record, checks) from the most recent live probe, if one is on disk."""
    import glob
    from schemas.seed import Seed
    seed_path = ROOT / "data-gen" / "seeds" / "rl.json"
    if not seed_path.exists():
        return [], {}
    seed = Seed.load(str(seed_path))
    cbt = {t.name: t.checks for t in seed.tasks}
    rows = []
    for f in glob.glob(str(ROOT / "data-gen" / "data" / "out" / "rl__*" / "**" / "*.jsonl"), recursive=True):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows, cbt


def test_reward_prefers_the_best_candidate():
    """THE test. Each record carries a best and a worst candidate, so the corpus labels the reward.

    A reward that cannot separate them is not a reward -- it is noise at best and, if inverted, it actively
    trains the policy towards confident fabrication. That is not hypothetical: the first version of these
    scorers preferred the WORST candidate on 32 of 54 real samples.
    """
    rows, cbt = _real_pairs()
    rows = [r for r in rows if (r.get("ranking") or [None])[0] == "best"
            and r.get("response_1") and r.get("response_2")
            and cbt.get((r.get("tasks") or [None])[0])]
    if len(rows) < 20:
        pytest.skip(f"only {len(rows)} scored records on disk; run a probe first")

    better = worse = tied = 0
    for r in rows:
        checks = cbt[(r["tasks"] or [None])[0]]
        rb, _ = score(r["response_1"], r, checks)
        rw, _ = score(r["response_2"], r, checks)
        if rb > rw:
            better += 1
        elif rb < rw:
            worse += 1
        else:
            tied += 1
    n = len(rows)
    decided = better + worse
    assert decided, "the reward separated no pair at all -- every check must be unscoreable"
    accuracy = better / decided
    assert accuracy >= 0.80, (
        f"the reward prefers the WORSE candidate too often: {better} right, {worse} wrong, {tied} tied "
        f"of {n} ({accuracy:.0%} of decided pairs). A reward below chance trains the policy backwards.")
