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
    Ctx, KNOWN_PIPE_TOKENS, REGISTRY, missing_scorers, score, score_group,
)


# --------------------------------------------------------------------------- the seed contract

def _spec_checks() -> dict[str, list[dict]]:
    import rewards_spec as R
    out = {t: [vars(c) for c in cs] for t, cs in R.CHECKS.items()}
    out["UNIVERSAL"] = [vars(c) for c in R.UNIVERSAL]
    out["DISCRIMINATOR"] = [vars(c) for c in R.DISCRIMINATOR]
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
    # Any check that depends on seeing the call now sees it.
    assert REGISTRY["two_distinct_tools_called"](ctx) is False, "one call is one call"
    assert REGISTRY["think_precedes_tool_call"](ctx) is True


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

    Scoped to the tasks this reward CLAIMS to score. The judge-only tasks are excluded because they are
    excluded at training time too (VerifierReward returns `neutral` for them) -- measured at 40% there,
    below chance, which is exactly why that exclusion exists rather than a higher-effort check.
    """
    import rewards_spec as R
    rows, cbt = _real_pairs()
    rows = [r for r in rows if (r.get("ranking") or [None])[0] == "best"
            and r.get("response_1") and r.get("response_2")
            and cbt.get((r.get("tasks") or [None])[0])
            and not R.judge_required((r.get("tasks") or [None])[0])]
    if len(rows) < 20:
        pytest.skip(f"only {len(rows)} scored records on disk; run a probe first")

    better = worse = tied = 0
    for r in rows:
        checks = cbt[(r["tasks"] or [None])[0]]
        # score_group, not score: a check evaluable for one candidate and not the other is the CANDIDATE's
        # doing, and must not shrink its denominator into a higher score.
        (rb, rw), _ = score_group([r["response_1"], r["response_2"]], r, checks)
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


def test_nothing_is_judge_only_because_there_is_no_judge():
    """There is no judge model in this pipeline, so a task that cannot be scored programmatically cannot be
    trained on at all. Four tasks used to be excluded because the procedural checks scored them below chance;
    the discriminator fixed that at the DATA level -- the generator writes down the phrase that separates its
    two replies, postprocess verifies the claim holds, and scoring becomes a substring lookup."""
    import rewards_spec as R
    assert R.JUDGE_ONLY_TASKS == frozenset(), (
        f"these tasks would be untrainable without a judge: {sorted(R.JUDGE_ONLY_TASKS)}")
    # and the discriminator is actually present on every task's checks
    for task in R.CHECKS:
        names = {c["name"] for c in R.for_task(task)}
        assert "states_the_key_fact" in names, f"{task} has no answer key"
        assert "avoids_the_false_claim" in names


def test_discriminator_checks_score_the_right_candidate():
    """The sharpest signal available, and it applies to every task."""
    item = {"lang": "pcm", "verifier": {"discriminator": {"good": "4,500 naira", "bad": "about 6,000"}}}
    checks = [{"name": "key", "weight": 0.6, "expect": True, "fn": "states_the_key_fact"},
              {"name": "false", "weight": 0.4, "expect": False, "fn": "repeats_the_false_claim"}]
    best = "<|input_lang|><pcm><|target_lang|><pcm><response>Na 4,500 naira you go pay."
    worst = "<|input_lang|><pcm><|target_lang|><pcm><response>E be about 6,000 naira."
    (rb, rw), _ = score_group([best, worst], item, checks)
    assert rb == 1.0 and rw == 0.0


def test_discriminator_is_unknown_when_the_record_has_none():
    """Older records carry no discriminator; those checks must drop out rather than fail everything."""
    item = {"lang": "pcm", "verifier": {}}
    checks = [{"name": "key", "weight": 1.0, "expect": True, "fn": "states_the_key_fact"}]
    r, detail = score("<|input_lang|><pcm><response>anything", item, checks)
    assert detail["key"] == "unknown"


def test_behaviour_tasks_are_not_quietly_excluded():
    """The flip side: the exclusion must stay narrow. If a behaviour task ever lands in JUDGE_ONLY_TASKS the
    programmatic reward has silently stopped covering the thing this phase exists to teach."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "data-gen" / "seeds"))
    import rewards_spec as R
    from build_seeds import BEHAVIOUR_TASKS
    overlap = sorted(BEHAVIOUR_TASKS & R.JUDGE_ONLY_TASKS)
    assert not overlap, f"behaviour tasks must be programmatically scored, but these are judge-only: {overlap}"


def test_group_scoring_does_not_reward_being_unscoreable():
    """The flaw score_group exists to fix: dropping unknown checks per candidate let a candidate that did
    nothing checkable win. Measured on a real multi-tool sample -- worst 1.0, best 0.7 -- purely because the
    worst called no tool, so every tool check was unknown for it and its denominator shrank."""
    item = {"lang": "eng", "prompt_messages": [{"role": "user", "content": "price in naira then in cedis"}]}
    checks = [
        {"name": "two_tools", "weight": 0.6, "expect": True, "fn": "two_distinct_tools_called"},
        {"name": "brief", "weight": 0.4, "expect": True, "fn": "response_under_120_words"},
    ]
    acted = ('<|input_lang|><eng><task_plan><|plan|></task_plan>'
             '<tool_call>get_market_prices\n{"commodity":"maize"}</tool_call>'
             '<tool_call>get_exchange_rate\n{"base":"NGN"}</tool_call>')
    did_nothing = "<|input_lang|><eng><|target_lang|><eng><response>It is about 2000.</response>"

    (r_acted, r_idle), _ = score_group([acted, did_nothing], item, checks)
    assert r_acted > r_idle, (
        f"the candidate that did nothing scored {r_idle} against {r_acted} -- being unscoreable must not pay")


def test_a_check_unknown_for_every_candidate_is_dropped_for_everyone():
    """The other direction: a missing `verifier.expected_answer` is a gap in the DATA, so it must not
    penalise any candidate."""
    item = {"lang": "eng", "verifier": {}}
    checks = [{"name": "correct", "weight": 0.8, "expect": True, "fn": "final_number_matches_expected"},
              {"name": "brief", "weight": 0.2, "expect": True, "fn": "response_under_120_words"}]
    rs, details = score_group(["<|input_lang|><eng><response>Short.", "<|input_lang|><eng><response>Also."],
                              item, checks)
    assert all(r == pytest.approx(1.0) for r in rs)
    assert all(d["correct"] == "unknown (no candidate)" for d in details)


def test_a_failed_gate_zeroes_only_that_candidate():
    item = {"lang": "eng"}
    checks = [{"name": "not_degenerate", "weight": 1.0, "expect": False, "fn": "is_degenerate", "gate": True},
              {"name": "brief", "weight": 1.0, "expect": True, "fn": "response_under_120_words"}]
    loop = "<|input_lang|><eng><response>" + "the market is open " * 12
    fine = "<|input_lang|><eng><response>The market opens at six and shuts when the yams are gone."
    (r_loop, r_fine), _ = score_group([loop, fine], item, checks)
    assert r_loop == 0.0 and r_fine > 0.0
