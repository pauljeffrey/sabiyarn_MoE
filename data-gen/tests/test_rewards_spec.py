"""The RL seed carries its own scorers, so the thing to protect is the contract between them and the seed:
every RL task has checks, every check is well formed, and nothing references a scorer nobody will write.

These are cheap invariants but they are the ones that fail silently. A task added to _RL_WEIGHTS with no
CHECKS entry still generates data -- it just generates data that only the judge can score, which is the
expensive path we added rewards_spec.py to avoid.
"""
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "seeds"))

import rewards_spec as R  # noqa: E402
from build_seeds import _RL_WEIGHTS, RL_TASKS, SFT_TASKS  # noqa: E402


def test_every_rl_task_has_checks():
    """A task in RL without checks falls back entirely on the judge, which is slow and non-reproducible."""
    missing = sorted(t for t in _RL_WEIGHTS if t not in R.CHECKS)
    assert not missing, f"RL tasks with no programmatic scorers: {missing}"
    assert all(t.checks for t in RL_TASKS), "a task reached the seed with an empty checks list"


def test_checks_do_not_leak_into_sft():
    """SFT has one target per sample, so there is nothing to rank and the scorers would be dead weight."""
    assert not any(t.checks for t in SFT_TASKS)


@pytest.mark.parametrize("task", sorted(R.CHECKS))
def test_weights_sum_to_one(task):
    """So one task's programmatic score is directly comparable to another's."""
    total = sum(c.weight for c in R.CHECKS[task])
    assert total == pytest.approx(1.0), f"{task} weights sum to {total}"


@pytest.mark.parametrize("task", sorted(R.CHECKS))
def test_checks_are_well_formed(task):
    for c in R.CHECKS[task]:
        assert c.pattern or c.fn, f"{task}/{c.name} scores nothing"
        assert not (c.pattern and c.fn), f"{task}/{c.name} has both a pattern and a fn; pick one"
        assert c.weight > 0, f"{task}/{c.name} has a zero weight"
        if c.pattern:
            re.compile(c.pattern)  # raises on a bad regex
            assert c.compiled() is not None


def test_names_are_unique_within_a_task():
    """Two checks with one name silently collapse to one in any scorer keyed by name."""
    for task, checks in R.CHECKS.items():
        names = [c.name for c in checks]
        assert len(names) == len(set(names)), f"{task} has duplicate check names"


def test_every_task_can_fail_and_can_pass():
    """A task scored only by `expect=True` checks cannot distinguish a bad candidate that does the right
    things badly; one scored only by negatives rewards saying nothing. Each needs both."""
    for task, checks in R.CHECKS.items():
        kinds = {c.expect for c in checks}
        assert True in kinds, f"{task} has no positive check"
        if task not in {"instruction_following"}:  # its negative is a reward-hacking tell, not a harm
            assert False in kinds, f"{task} has no negative check"


def test_for_task_includes_the_universal_gates():
    out = R.for_task("summarization")
    names = {c["name"] for c in out}
    assert {c.name for c in R.UNIVERSAL} <= names
    assert "no_figure_absent_from_source" in names
    assert all(isinstance(c, dict) for c in out), "for_task must be JSON-serialisable for the seed"


def test_unknown_task_still_gets_the_gates_and_the_answer_key():
    """A task added to the seed before its own scorers are written is still gated for malformedness AND
    scored on the discriminator, which applies to every task regardless."""
    names = {c["name"] for c in R.for_task("no_such_task")}
    assert {c.name for c in R.UNIVERSAL} <= names
    assert {c.name for c in R.DISCRIMINATOR} <= names


def test_the_discriminator_does_not_crowd_out_procedure():
    """The discriminator is one phrase -- the sharpest signal, but not the whole story. Procedure must keep
    the majority, or "called a tool before answering" stops being worth doing."""
    for task in R.CHECKS:
        checks = [c for c in R.for_task(task) if not c.get("gate")]
        disc = sum(c["weight"] for c in checks if c["name"] in {x.name for x in R.DISCRIMINATOR})
        assert 0.0 < disc < 0.5, f"{task}: discriminator carries {disc:.0%} of the score"
        assert abs(sum(c["weight"] for c in checks) - 1.0) < 1e-6, f"{task} weights do not sum to 1"


def test_admission_phrases_cover_the_generated_languages():
    """An English-only admission cue teaches the model to say "I don't know" in English inside a Yoruba
    answer, which is the opposite of what the honesty tasks are for."""
    generating = {"pcm", "yor", "hau", "ibo", "fon", "ewe", "urh", "efi", "twi", "aka", "eng"}
    assert generating <= set(R.ADMISSION), f"no admission phrases for {sorted(generating - set(R.ADMISSION))}"
    for lang in generating:
        assert R.ADMISSION[lang], f"{lang} has an empty phrase list"
        pat = re.compile(R.admission_pattern(lang), re.I)
        assert pat.search(list(R.ADMISSION[lang])[0]), f"{lang} pattern does not match its own phrase"
        assert pat.search("i don't know"), f"{lang} should accept the English fallback -- mixed code is normal"


def test_objective_reasoning_check_is_weighted_highest():
    """final_number_matches_expected is the only fully objective check in the corpus. If it is ever not the
    dominant term for that task, the task has drifted into being judged on style."""
    checks = {c.name: c.weight for c in R.CHECKS["multi_step_reasoning"]}
    assert checks["final_answer_is_correct"] == max(checks.values())


def test_scorer_functions_are_named_consistently():
    fns = R.all_function_names()
    assert fns, "no scorer functions referenced at all"
    for fn in fns:
        assert re.fullmatch(r"[a-z][a-z0-9_]+", fn), f"{fn} is not a snake_case identifier"


def test_coverage_reports_every_task():
    assert set(R.coverage()) == set(R.CHECKS)
    assert all(n >= 3 for n in R.coverage().values()), "a task with fewer than 3 checks is barely scored"
