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


LOW_RESOURCE = ("efi", "urh", "fon", "ewe", "ful", "fuv")


def test_seeds_load_and_validate(seeds):
    assert seeds["pretrain"].total_samples() >= 435_000
    for l in seeds["pretrain"].languages:
        if l.code == "pcm":
            assert l.samples >= 60_000
        if l.code in ("urh", "efi"):
            assert l.samples >= 30_000, f"pretrain {l.code} below the 30k floor"
    for kind in ("sft", "rl"):
        s = seeds[kind]
        assert s.tools and s.tasks
        assert s.conversation["ends_with"] == "assistant"
        for t in s.tasks:
            assert set(t.tags) <= set(TAGS)


def test_requested_volumes_meet_the_agreed_floors(seeds):
    sft = {l.code: l.samples for l in seeds["sft"].languages}
    assert 150_000 <= seeds["sft"].total_samples() <= 200_000, seeds["sft"].total_samples()
    assert sft["pcm"] >= 50_000
    for c in LOW_RESOURCE + ("yor", "twi", "ibo", "hau"):
        assert sft[c] >= 10_000, f"sft {c} = {sft[c]}, below the 10k floor"
    assert 60_000 <= seeds["rl"].total_samples() <= 70_000, seeds["rl"].total_samples()


def test_yield_budget_over_requests_so_the_target_lands(seeds):
    """Low-resource conversations fail validation more often, so more must be requested."""
    for kind in ("pretrain", "sft", "rl"):
        s = seeds[kind]
        assert s.total_requests() > s.total_samples()
        low = next(l for l in s.languages if l.code == "fon")
        high = next(l for l in s.languages if l.code == "pcm")
        assert s.requests_for(low) / low.samples > s.requests_for(high) / high.samples


def test_every_tag_has_its_own_task(seeds):
    """Bundled tags cannot be counted or held out separately, so each gets a task."""
    for kind in ("sft",):
        covered = {t for task in seeds[kind].tasks for t in task.tags}
        assert covered == set(TAGS), f"{kind} missing {sorted(set(TAGS) - covered)}"


def test_translation_covers_english_and_interlanguage(seeds):
    names = {t.name for t in seeds["sft"].tasks}
    assert {"translation_english", "translation_interlanguage"} <= names
    inter = next(t for t in seeds["sft"].tasks if t.name == "translation_interlanguage")
    assert "without going through english" in inter.description.lower()


def test_tool_catalogue_spans_more_than_retrieval(seeds):
    names = {t.name for t in seeds["sft"].tools}
    for expected in ("search_internet", "search_db", "db_insert_record", "db_get_record",
                     "run_statistics", "calculate", "convert_units", "get_market_prices",
                     "lookup_crop_guidance", "find_health_facility", "get_weather_forecast"):
        assert expected in names, expected
    assert len(names) >= 18


def test_plan_totals_match_per_language_volumes(seeds):
    for kind in ("sft", "rl"):
        s = seeds[kind]
        plan = s.plan()
        for lang in s.languages:
            assert sum(plan[lang.code].values()) == s.requests_for(lang), f"{kind}/{lang.code}"


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


def test_genre_is_not_locked_to_the_domain_pair(seeds):
    """Regression: pair=(i+off)%636 with genre=(i*7+off)%28 locked each pair to ONE genre, because
    636*7 is a multiple of 28. That collapsed 17,808 (pair, genre) combinations to 636 -- every document
    about one sub-topic came out in the same genre. The walk must cover the full cross product."""
    import collections

    from prompts import _GENRES, _PAIRS

    n_combo = len(_PAIRS) * len(_GENRES)
    seen, per_pair = set(), collections.defaultdict(set)
    for i in range(n_combo + 200):
        md = build_request(seeds["pretrain"], {"custom_id": f"pretrain__yor__doc__{i:06d}",
                                              "lang": "yor", "task": "doc", "index": i}).metadata
        pair = (md["domain"], md["subtopic"])
        seen.add((pair, md["genre"]))
        per_pair[pair].add(md["genre"])
    assert len(seen) == n_combo, f"only {len(seen)} of {n_combo} (pair, genre) combinations reachable"
    assert min(len(v) for v in per_pair.values()) == len(_GENRES), "some pair never varies genre"
    # and no combination repeats before all of them have been used
    first_lap = set()
    for i in range(n_combo):
        md = build_request(seeds["pretrain"], {"custom_id": f"pretrain__hau__doc__{i:06d}",
                                              "lang": "hau", "task": "doc", "index": i}).metadata
        first_lap.add(((md["domain"], md["subtopic"]), md["genre"]))
    assert len(first_lap) == n_combo


def test_languages_get_different_slices_of_the_taxonomy(seeds):
    """Per-language offset: two languages must not march through the taxonomy in lockstep."""
    def first(lang, n=40):
        return [build_request(seeds["pretrain"], {"custom_id": f"pretrain__{lang}__doc__{i:06d}",
                                                 "lang": lang, "task": "doc", "index": i}
                              ).metadata["subtopic"] for i in range(n)]
    assert first("yor") != first("hau") != first("fon")


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


def test_thinking_is_english_and_task_plan_is_a_plan(seeds):
    req = build_request(seeds["sft"], {"custom_id": "sft__fon__tool_search_answer__000003",
                                       "lang": "fon", "task": "tool_search_answer", "index": 3})
    brief = req.messages[0]["content"]
    assert "<think> IS ALWAYS IN ENGLISH" in brief
    assert "never in Fon" in brief
    assert "<task_plan> is a PLAN, not a label" in brief
    assert "ALWAYS in Fon" in brief  # the response, unlike the thinking


def test_irrelevant_tools_are_in_scope_and_named(seeds):
    """Tool selection is only a skill if there is something wrong to select."""
    for i in range(30):
        req = build_request(seeds["sft"], {"custom_id": f"sft__ibo__action_tool_use__{i:06d}",
                                           "lang": "ibo", "task": "action_tool_use", "index": i})
        md = req.metadata
        d = md["distractor_tools"]
        assert 2 <= len(d) <= 3, d
        assert set(d) <= set(md["tools"])
        assert "IRRELEVANT TOOLS: " + ", ".join(d) in req.messages[1]["content"]
        # full definitions travel in metadata AND are attached natively for the provider
        assert len(md["tool_definitions"]) == len(md["tools"])
        assert req.tools and {t["function"]["name"] for t in req.tools} == set(md["tools"])


def test_confidence_is_requested_for_every_kind(seeds):
    for kind, row in (("pretrain", {"custom_id": "pretrain__yor__doc__000001", "lang": "yor",
                                    "task": "doc", "index": 1}),
                      ("sft", {"custom_id": "sft__yor__general_chat__000001", "lang": "yor",
                               "task": "general_chat", "index": 1}),
                      ("rl", {"custom_id": "rl__yor__world_knowledge_qa__000001", "lang": "yor",
                              "task": "world_knowledge_qa", "index": 1})):
        body = build_request(seeds[kind], row).messages[1]["content"]
        assert '"confidence"' in body, kind


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


def test_only_usable_special_tokens_are_asked_for(seeds):
    """Token 52043 was '|analyze|>' (missing its '<') and has since been fixed to '<|analyze|>', so the
    corrected form is now valid. The bare form never is. Separately, tokenizer ids >= the model's
    vocab_size (52050) can never be embedded, so tokens from that range must not appear either."""
    unusable = {"|analyze|>", "<|hate|>"}          # bare form; and 52115, above vocab_size
    for kind in ("sft", "rl"):
        for t in seeds[kind].tasks:
            for verb in t.task_plan:
                assert verb not in unusable, f"{kind}/{t.name}: {verb}"
                assert verb.startswith("<") and verb.endswith(">"), f"{kind}/{t.name}: {verb}"
    allowed = set(seeds["sft"].format["special_tokens"]["task_plan_verbs"])
    assert "<|analyze|>" in allowed or all(
        v in allowed for t in seeds["sft"].tasks for v in t.task_plan), "task_plan verb not in the declared list"
    for tok in unusable:
        assert tok not in allowed


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
    assert "<tool_call>search_internet" in r["text"] and "<tool_response>" in r["text"]
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
