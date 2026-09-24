"""Seed schema, deterministic planning, prompt building and record post-processing -- all offline."""
import json

import pytest

from postprocess_gen import STATS, to_record
from prompts import build_request
from providers.base import Response
from schemas.seed import TAGS, Seed


@pytest.fixture(scope="module")
def seeds():
    return {k: Seed.load(k) for k in ("pretrain", "sft", "rl")}


# ----------------------------------------------------------------- schema / plan


def test_seeds_load_and_validate(seeds):
    assert seeds["pretrain"].total_samples() >= 435_000
    assert seeds["pretrain"].languages[0].code == "pcm" and seeds["pretrain"].languages[0].samples >= 60_000
    for l in seeds["pretrain"].languages:
        if l.code in ("urh", "efi"):
            assert l.samples >= 30_000, f"{l.code} below the 30k floor"
    for kind in ("sft", "rl"):
        s = seeds[kind]
        assert s.tools and s.tasks
        assert s.conversation["ends_with"] == "assistant"
        for t in s.tasks:
            assert set(t.tags) <= set(TAGS)


def test_plan_totals_match_per_language_volumes(seeds):
    for kind in ("sft", "rl"):
        s = seeds[kind]
        plan = s.plan()
        for lang in s.languages:
            assert sum(plan[lang.code].values()) == lang.samples, f"{kind}/{lang.code}"


def test_knowledge_boundary_is_the_biggest_block(seeds):
    """The behaviours the project exists to teach must dominate the mix, or this is just another chat corpus."""
    shares = seeds["sft"].task_shares()
    boundary = sum(shares[t.name] for t in seeds["sft"].tasks
                   if {"knowledge-boundary", "insufficient-context"} & set(t.tags))
    assert boundary > 0.20, f"knowledge-boundary tasks are only {boundary:.0%} of the mix"


def test_seed_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "sft", "details": "x", "languages": {}, "nonsense": 1}))
    with pytest.raises(ValueError, match="nonsense"):
        Seed.load(str(bad))


# ----------------------------------------------------------------- prompts


def test_prompt_building_is_deterministic_and_covers_the_taxonomy(seeds):
    row = {"custom_id": "sft__yor__rag_document_qa__000001", "lang": "yor",
           "task": "rag_document_qa", "index": 1}
    a = build_request(seeds["sft"], row)
    b = build_request(seeds["sft"], row)
    assert a.messages == b.messages and a.metadata == b.metadata

    # index walks distinct (domain, subtopic) pairs rather than resampling the same few
    pairs = {tuple(build_request(seeds["pretrain"],
                                 {"custom_id": f"pretrain__yor__doc__{i:06d}", "lang": "yor",
                                  "task": "doc", "index": i}).metadata[k] for k in ("domain", "subtopic"))
             for i in range(200)}
    assert len(pairs) == 200


def test_tools_referenced_by_a_task_are_in_scope(seeds):
    """A brief that says 'use get_exchange_rate' while omitting it from the tool list teaches nothing."""
    for i in range(60):
        row = {"custom_id": f"sft__hau__financial_analysis__{i:06d}", "lang": "hau",
               "task": "financial_analysis", "index": i}
        md = build_request(seeds["sft"], row).metadata
        assert "get_exchange_rate" in md["tools"] and "calculate" in md["tools"]
    for i in range(40):
        row = {"custom_id": f"sft__ibo__rag_document_qa__{i:06d}", "lang": "ibo",
               "task": "rag_document_qa", "index": i}
        assert "search_documents" in build_request(seeds["sft"], row).metadata["tools"]


def test_no_malformed_special_tokens_in_any_prompt(seeds):
    """Token 52043 is '|analyze|>' (missing its '<'), so '<|analyze|>' must never be asked for."""
    for kind in ("sft", "rl"):
        for t in seeds[kind].tasks:
            assert "<|analyze|>" not in t.task_plan, f"{kind}/{t.name}"


# ----------------------------------------------------------------- records


def _resp(custom_id, payload, md):
    return Response(custom_id, json.dumps(payload, ensure_ascii=False), True, metadata=md)


def test_pretrain_record(seeds):
    md = {"kind": "pretrain", "lang": "yor", "domain": "health_medicine", "subtopic": "malaria", "genre": "article"}
    r = to_record(seeds["pretrain"], _resp("pretrain__yor__doc__000001", {
        "title": "Ìbà", "text": " ".join(["ọ̀rọ̀"] * 120), "language_self_check": True}, md))
    assert r["lang"] == "yor" and r["title"] == "Ìbà" and len(r["text"].split()) == 120
    assert to_record(seeds["pretrain"], _resp("x", {"title": "t", "text": "too short",
                                                   "language_self_check": True}, md)) is None


def _sft_payload():
    return {"messages": [
        {"role": "system", "content": "You are helpful. Tools: [...]"},
        {"role": "user", "content": "Kini AWS?"},
        {"role": "assistant", "content": "<|input_lang|><yor><think>N kò tíì gbọ́ nǹkan yìí rí.</think>",
         "tool_calls": [{"function": {"name": "search_internet", "arguments": {"query": "what is AWS"}}}]},
        {"role": "tool", "name": "search_internet", "content": "AWS is Amazon's cloud computing platform."},
        {"role": "assistant",
         "content": "<|input_lang|><yor><task_plan><|explain|></task_plan><|target_lang|><yor><response>AWS jẹ́ ìpèsè kọ̀ǹpútà ti Amazon."},
    ], "tasks": ["tool_search_answer"], "tags": ["tool-calling", "knowledge-boundary", "qa"]}


def test_sft_record_keeps_both_representations(seeds):
    md = {"kind": "sft", "lang": "yor", "tasks": ["tool_search_answer"], "tags": ["tool-calling"],
          "tools": ["search_internet"]}
    r = to_record(seeds["sft"], _resp("sft__yor__tool_search_answer__000001", _sft_payload(), md))
    assert r is not None
    assert isinstance(r["messages"], list) and len(r["messages"]) == 5           # dict form
    assert r["text"].startswith("<s><|system|>") and r["text"].endswith("</s>")  # rendered form
    assert "<tool_call>search_internet" in r["text"] and "<tool_result>" in r["text"]
    assert "knowledge-boundary" in r["tags"]
    assert r["instruction"] == "Kini AWS?" and "Amazon" in r["context"]
    assert r["response"].startswith("AWS jẹ́")


def test_sft_rejects_broken_conversations(seeds):
    md = {"kind": "sft", "lang": "yor", "tasks": [], "tags": [], "tools": []}
    bad = _sft_payload()
    bad["messages"][-1]["content"] = "no response token at all"
    assert to_record(seeds["sft"], _resp("a", bad, md)) is None
    orphan = _sft_payload()
    orphan["messages"][2].pop("tool_calls")           # tool result with nothing that called it
    assert to_record(seeds["sft"], _resp("b", orphan, md)) is None
    unknown = _sft_payload()
    unknown["messages"][2]["tool_calls"][0]["function"]["name"] = "not_a_real_tool"
    assert to_record(seeds["sft"], _resp("c", unknown, md)) is None


def test_rl_record_orders_responses_and_rejects_paraphrases(seeds):
    md = {"kind": "rl", "lang": "hau", "tasks": ["tool_search_answer"], "tags": ["tool-calling"], "tools": []}
    prefix = _sft_payload()["messages"][:4] + [{"role": "user", "content": "To, me ke nan?"}]
    good = {"prompt_messages": prefix, "tasks": ["tool_search_answer"], "tags": ["knowledge-boundary"],
            "responses": [
                {"content": "<|input_lang|><hau><response>Ban sani ba.", "quality": "worst", "why": "invented"},
                {"content": "<|input_lang|><hau><response>AWS na Amazon ne.", "quality": "best", "why": "grounded"},
                {"content": "<|input_lang|><hau><response>Wataƙila.", "quality": "partial", "why": "hedged"}]}
    r = to_record(seeds["rl"], _resp("rl__hau__tool_search_answer__000001", good, md))
    assert r["ranking"] == ["best", "partial", "worst"]
    assert "Amazon" in r["response_1"] and r["response_3"].endswith("Ban sani ba.")
    assert r["prompt_text"].endswith("<|assistant|>")   # generation prompt, ready for the policy

    same = dict(good, responses=[dict(x, content="identical") for x in good["responses"]])
    assert to_record(seeds["rl"], _resp("d", same, md)) is None
    no_worst = dict(good, responses=[dict(x, quality="best") for x in good["responses"]])
    assert to_record(seeds["rl"], _resp("e", no_worst, md)) is None
