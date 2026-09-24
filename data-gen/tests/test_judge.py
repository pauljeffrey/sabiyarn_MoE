"""The judge pass: anonymised ranking -> response_1 (chosen) / response_2 (rejected)."""
import json

import pytest

from judge_gen import LABELS, apply_verdict, build_judge_request
from providers.base import Response
from schemas.seed import Seed


@pytest.fixture(scope="module")
def seed():
    return Seed.load("rl")


def _rl_record():
    return {"id": "rl__hau__tool_search_answer__000001", "lang": "hau", "tags": ["tool-calling"],
            "tasks": ["tool_search_answer"], "tools": ["search_internet", "set_reminder"],
            "distractor_tools": ["set_reminder"], "prompt_messages": [{"role": "user", "content": "Me ne AWS?"}],
            "prompt_text": "<s><|user|>Me ne AWS?</s><|assistant|>", "instruction": "Me ne AWS?",
            "input": "", "context": "", "domain": "tech_infrastructure", "subtopic": "cloud",
            "ranking": ["best", "partial", "worst"], "model": "openai/gpt-oss-120b",
            "response_1": "GROUNDED answer from the tool result.",
            "response_2": "HEDGED, technically true but useless.",
            "response_3": "INVENTED, confidently wrong."}


def test_judge_prompt_anonymises_and_shuffles(seed):
    req = build_judge_request(_rl_record(), seed)
    body = req.messages[1]["content"]
    # candidates relabelled, generator's own labels withheld
    for letter in "ABC":
        assert f"--- CANDIDATE {letter} ---" in body
    assert "best" not in body and "worst" not in body
    assert "deliberately irrelevant: set_reminder" in body
    # criteria are ranked, honesty first
    assert "HONESTY AT THE KNOWLEDGE BOUNDARY" in req.messages[0]["content"]
    assert "Over-refusal is a" in req.messages[0]["content"]


def test_judge_prompt_is_deterministic(seed):
    a = build_judge_request(_rl_record(), seed).messages[1]["content"]
    b = build_judge_request(_rl_record(), seed).messages[1]["content"]
    assert a == b


def _verdict(rec, seed, winner_text, loser_text, **extra):
    req = build_judge_request(rec, seed)
    order = req.metadata["order"]
    cands = [rec["response_1"], rec["response_2"], rec["response_3"]]
    w = LABELS[order.index(cands.index(winner_text))]
    l = LABELS[order.index(cands.index(loser_text))]
    payload = {"winner": w, "loser": l, "deciding_criterion": 1, "why": "grounded vs invented",
               "confidence": 0.9, "both_bad": False, **extra}
    return apply_verdict(Response(f"judge__{rec['id']}", json.dumps(payload), True,
                                  metadata=req.metadata), min_confidence=0.0)


def test_verdict_maps_letters_back_through_the_shuffle(seed):
    rec = _rl_record()
    out = _verdict(rec, seed, rec["response_1"], rec["response_3"])
    assert out["response_1"] == "GROUNDED answer from the tool result."
    assert out["response_2"] == "INVENTED, confidently wrong."
    assert out["deciding_criterion"] == 1 and out["judge_confidence"] == 0.9
    assert out["agrees_with_generator"] is True
    assert "response_3" not in out


def test_verdict_records_disagreement_with_the_generator(seed):
    rec = _rl_record()
    out = _verdict(rec, seed, rec["response_3"], rec["response_1"])
    assert out["response_1"] == "INVENTED, confidently wrong."
    assert out["agrees_with_generator"] is False


def test_both_bad_and_low_confidence_are_dropped(seed):
    rec = _rl_record()
    assert _verdict(rec, seed, rec["response_1"], rec["response_3"], both_bad=True) is None
    req = build_judge_request(rec, seed)
    payload = {"winner": "A", "loser": "B", "confidence": 0.2, "both_bad": False}
    assert apply_verdict(Response("x", json.dumps(payload), True, metadata=req.metadata),
                         min_confidence=0.5) is None


def test_malformed_verdicts_are_dropped(seed):
    req = build_judge_request(_rl_record(), seed)
    for bad in ("not json at all", '{"winner": "A", "loser": "A"}', '{"winner": "Z", "loser": "B"}'):
        assert apply_verdict(Response("x", bad, True, metadata=req.metadata)) is None


def test_single_candidate_is_not_judgeable(seed):
    rec = _rl_record()
    rec.pop("response_2"); rec.pop("response_3")
    assert build_judge_request(rec, seed) is None
