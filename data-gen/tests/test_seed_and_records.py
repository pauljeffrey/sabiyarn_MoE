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


def test_pretrain_has_no_english(seeds):
    """Pretraining is for the 12 target languages only."""
    assert "eng" not in {l.code for l in seeds["pretrain"].languages}
    assert len(seeds["pretrain"].languages) == 12
    assert seeds["pretrain"].format["text_words"] == [300, 500]


def test_seeds_load_and_validate(seeds):
    assert seeds["pretrain"].total_samples() >= 420_000
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


def test_the_contract_asks_for_fields_not_marker_strings(seeds):
    """The generator got the finished marker string wrong in 88 of 108 published records, so it is no longer
    asked for one: it returns fields and assemble.py builds the scaffolding."""
    req = build_request(seeds["sft"], {"custom_id": "sft__fon__tool_search_answer__000003",
                                       "lang": "fon", "task": "tool_search_answer", "index": 3})
    brief = req.messages[0]["content"]
    assert "OUTPUT CONTRACT" in brief
    assert "never write <|input_lang|>" in brief
    assert "ALWAYS IN ENGLISH" in brief              # think
    assert "OMIT IT for simple turns" in brief       # think is optional, saving tokens
    assert "PLAIN TEXT" in brief
    # the old string-template instructions must be gone
    assert "<|input_lang|><__LANG__>" not in brief


def test_distractors_are_withheld_from_the_generator(seeds):
    """Distractors are injected at post-processing, not shown to the generator: it called them 12 times in
    35 once few-shot was added, and their definitions cost 300-1,200 input tokens for no generative benefit."""
    for i in range(30):
        req = build_request(seeds["sft"], {"custom_id": f"sft__ibo__action_tool_use__{i:06d}",
                                           "lang": "ibo", "task": "action_tool_use", "index": i})
        md = req.metadata
        d = md["distractor_tools"]
        assert 2 <= len(d) <= 3, d
        # withheld: not in the shown list, not in the attached defs, not mentioned in the prompt
        assert not (set(d) & set(md["tools"])), "distractor leaked into the shown tool list"
        assert req.tools and {t["function"]["name"] for t in req.tools} == set(md["tools"])
        body = req.messages[1]["content"] + req.messages[0]["content"]
        for name in d:
            assert name not in body, f"distractor {name} named in the prompt"
        # but their definitions ride along for injection
        assert len(md["distractor_definitions"]) == len(d)


def test_distractors_are_injected_into_the_finished_sample(seeds):
    """The finished catalogue must still contain them, or the sample stops teaching tool selection."""
    md = {"kind": "sft", "lang": "yor", "tasks": [], "tags": [],
          "tools": ["search_internet"], "distractor_tools": ["set_reminder", "convert_units"],
          "tool_definitions": [{"type": "function", "function": {"name": "search_internet",
                                                                "description": "d", "parameters": {}}}],
          "distractor_definitions": [
              {"type": "function", "function": {"name": "set_reminder", "description": "d", "parameters": {}}},
              {"type": "function", "function": {"name": "convert_units", "description": "d",
                                                "parameters": {}}}]}
    rec = to_record(seeds["sft"], _resp("sft__yor__x__000001", _sft_payload(), md))
    assert rec is not None
    sysmsg = rec["messages"][0]
    assert sysmsg["role"] == "system"
    names = [f["function"]["name"] for f in json.loads(sysmsg["content"].split("\n", 1)[1])]
    assert set(names) == {"search_internet", "set_reminder", "convert_units"}
    # position must not be a tell: over many ids a distractor lands somewhere other than last
    lasts = set()
    for i in range(30):
        r = to_record(seeds["sft"], _resp(f"sft__yor__x__{i:06d}", _sft_payload(), md))
        n = [f["function"]["name"] for f in json.loads(r["messages"][0]["content"].split("\n", 1)[1])]
        lasts.add(n[-1])
    assert len(lasts) > 1, "distractors always in the same position"


def test_system_message_is_canonical_not_the_generators(seeds):
    """Built by us, so every sample in the corpus has an identically-formatted catalogue."""
    md = {"kind": "sft", "lang": "yor", "tasks": [], "tags": [], "tools": ["search_internet"],
          "distractor_tools": [], "tool_definitions": [
              {"type": "function", "function": {"name": "search_internet", "description": "d",
                                               "parameters": {}}}], "distractor_definitions": []}
    payload = _sft_payload()
    payload["messages"][0]["content"] = "whatever prose the generator felt like writing"
    rec = to_record(seeds["sft"], _resp("sft__yor__y__000001", payload, md))
    # identity is varied across samples, so assert the SHAPE rather than one fixed string
    from postprocess_gen import IDENTITIES
    first = rec["messages"][0]["content"]
    assert any(first.startswith(i) for i in IDENTITIES), first[:60]
    assert "You have these tools:" in first
    assert "whatever prose" not in rec["text"]


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
        {"role": "user", "content": "Bawo ni?"},
        {"role": "assistant", "content": "<|input_lang|><yor><task_plan><|chat|></task_plan>"
                                         "<|target_lang|><yor><response>Mo wa daadaa."},
        {"role": "user", "content": "Kini AWS?"},
        {"role": "assistant", "content": "<|input_lang|><yor><think>N kò tíì gbọ́ nǹkan yìí rí.</think>",
         "tool_calls": [{"function": {"name": "search_internet", "arguments": {"query": "what is AWS"}}}]},
        {"role": "tool", "name": "search_internet", "content": "AWS is Amazon's cloud computing platform."},
        {"role": "assistant",
         "content": "<|input_lang|><yor><task_plan><|explain|></task_plan><|target_lang|><yor><response>AWS jẹ́ ìpèsè kọ̀ǹpútà ti Amazon."},
        {"role": "user", "content": "O se."},
        {"role": "assistant", "content": "<|input_lang|><yor><task_plan><|chat|></task_plan>"
                                         "<|target_lang|><yor><response>Ko si wahala."},
    ], "tasks": ["tool_search_answer"], "tags": ["tool-calling", "knowledge-boundary", "qa"],
        "confidence": 0.9}


def test_sft_record_keeps_both_representations(seeds):
    md = {"kind": "sft", "lang": "yor", "tasks": ["tool_search_answer"], "tags": ["tool-calling"],
          "tools": ["search_internet"]}
    r = to_record(seeds["sft"], _resp("sft__yor__tool_search_answer__000001", _sft_payload(), md))
    assert r is not None
    assert isinstance(r["messages"], list) and len(r["messages"]) == 9           # dict form
    assert r["confidence"] == 0.9
    assert r["text"].startswith("<s><|system|>") and r["text"].endswith("</s>")  # rendered form
    assert "<tool_call>search_internet" in r["text"] and "<tool_response>" in r["text"]
    assert "knowledge-boundary" in r["tags"]
    assert r["instruction"] == "O se." and "Amazon" in r["context"]


def test_sft_rejects_broken_conversations(seeds):
    md = {"kind": "sft", "lang": "yor", "tasks": [], "tags": [], "tools": []}
    bad = _sft_payload()
    bad["messages"][-1]["content"] = "no response token at all"
    assert to_record(seeds["sft"], _resp("a", bad, md)) is None
    orphan = _sft_payload()
    orphan["messages"][4].pop("tool_calls")           # tool result with nothing that called it
    assert to_record(seeds["sft"], _resp("b", orphan, md)) is None
    unknown = _sft_payload()
    unknown["messages"][4]["tool_calls"][0]["function"]["name"] = "not_a_real_tool"
    assert to_record(seeds["sft"], _resp("c", unknown, md)) is None


def test_rl_record_orders_responses_and_rejects_paraphrases(seeds):
    md = {"kind": "rl", "lang": "hau", "tasks": ["tool_search_answer"], "tags": ["tool-calling"], "tools": []}
    prefix = _sft_payload()["messages"][:6] + [{"role": "user", "content": "To, me ke nan?"}]
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


# ----------------------------------------------------------------- run namespacing / tool length


def test_run_tag_namespaces_output_so_models_do_not_collide(monkeypatch):
    """Two models must be able to generate the SAME plan rows; without a namespace, done_ids() would make
    the second skip everything the first produced and the comparison would be impossible."""
    from generate import namespace
    monkeypatch.delenv("DATA_GEN_RUN_TAG", raising=False)
    assert namespace("sft") == "sft"
    assert namespace("sft", "gemma4") == "sft__gemma4"
    assert namespace("sft", "google/gemma-4-31b-it") == "sft__google-gemma-4-31b-it"  # path-safe
    monkeypatch.setenv("DATA_GEN_RUN_TAG", "llama33")
    assert namespace("rl") == "rl__llama33"


def test_tool_conversations_get_more_room(seeds):
    """A tool call is an assistant turn and its result a tool turn, so the same number of exchanges is twice
    the raw messages. Tool samples get 6 user turns and a higher total bound; non-tool samples stay tight."""
    conv = seeds["sft"].conversation
    # 6-16 messages overall; tool conversations get more dialogue turns and a higher total bound, since a
    # call is an assistant turn and its result a tool turn.
    assert conv["min_messages"] == 6 and conv["max_messages"] == 16
    assert conv["max_user_turns"] == 6 and conv["max_user_turns_with_tools"] == 8
    assert conv["max_total_messages"] < conv["max_total_messages_with_tools"]
    assert conv["target_messages_with_tools"] == [10, 16]

    md = {"kind": "sft", "lang": "yor", "tasks": [], "tags": [], "tools": ["search_internet"]}

    def convo(n_user, with_tools):
        msgs = []
        for i in range(n_user):
            msgs.append({"role": "user", "content": f"q{i}"})
            if with_tools:
                msgs.append({"role": "assistant", "content": "<|input_lang|><yor><think>look it up</think>",
                             "tool_calls": [{"function": {"name": "search_internet",
                                                         "arguments": {"query": "x"}}}]})
                msgs.append({"role": "tool", "name": "search_internet", "content": "result"})
            msgs.append({"role": "assistant",
                         "content": "<|input_lang|><yor><task_plan><|chat|></task_plan>"
                                    "<|target_lang|><yor><response>a"})
        return {"messages": msgs, "tasks": [], "tags": [], "confidence": 0.9}

    assert to_record(seeds["sft"], _resp("t1", convo(5, True), md)) is not None
    # 7 user turns WITHOUT tools exceeds max_user_turns (6)
    assert to_record(seeds["sft"], _resp("t2", convo(7, False), md)) is None
    # 3 user turns, no tools, is the compact shape
    assert to_record(seeds["sft"], _resp("t3", convo(3, False), md)) is not None


def test_calling_a_distractor_tool_discards_the_sample(seeds):
    """Distractors exist to train tool SELECTION. A sample that calls one teaches the opposite, so it is a
    drop rather than a warning -- measured at 12 of 35 calls once few-shot exemplars were added."""
    md = {"kind": "sft", "lang": "yor", "tasks": [], "tags": [],
          "tools": ["search_internet", "set_reminder"], "distractor_tools": ["set_reminder"]}

    def convo(tool):
        return {"messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<|input_lang|><yor><think>look it up</think>",
             "tool_calls": [{"function": {"name": tool, "arguments": {"query": "x", "when": "now",
                                                                     "text": "t"}}}]},
            {"role": "tool", "name": tool, "content": "a substantive result of adequate length here"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "<|input_lang|><yor><task_plan><|chat|></task_plan>"
                                             "<|target_lang|><yor><response>ans"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "<|input_lang|><yor><task_plan><|chat|></task_plan>"
                                             "<|target_lang|><yor><response>ans2"},
        ], "tasks": [], "tags": [], "confidence": 0.9}

    assert to_record(seeds["sft"], _resp("legit", convo("search_internet"), md)) is not None
    assert to_record(seeds["sft"], _resp("bad", convo("set_reminder"), md)) is None


def test_reasoning_models_are_flagged(capsys):
    """gpt-oss routes its chain of thought to a `reasoning` field and strips <think> from the content, so it
    produced 0 of 14 think blocks on a pilot. A 200k-request run must not discover that at the end."""
    from providers.base import warn_if_reasoning_model
    warn_if_reasoning_model("openai/gpt-oss-120b")
    assert "reasoning model" in capsys.readouterr().out
    warn_if_reasoning_model("google/gemma-4-31b-it")
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------------- language direction / packing


def test_io_directions_are_exactly_even(seeds):
    """Sampled on its own odometer, so the distribution is exact rather than approximately even."""
    import collections
    for lang in ("yor", "fon"):
        c = collections.Counter(
            build_request(seeds["sft"], {"custom_id": f"sft__{lang}__general_chat__{i:06d}",
                                        "lang": lang, "task": "general_chat", "index": i}
                          ).metadata["io_direction"] for i in range(500))
        assert len(c) == 5, c
        assert max(c.values()) == min(c.values()) == 100, c


def test_io_direction_survives_the_domain_odometer(seeds):
    """The pair/genre odometer laps every 17,808 rows and 17,808 mod 5 = 3, coprime with 5, so a fixed
    (domain, genre) still sees all five directions rather than locking to one."""
    seen = set()
    for lap in range(6):
        i = lap * 636          # same domain pair every lap for sft (no genre dimension)
        md = build_request(seeds["sft"], {"custom_id": f"sft__ibo__general_chat__{i:06d}",
                                         "lang": "ibo", "task": "general_chat", "index": i}).metadata
        seen.add(md["io_direction"])
    assert len(seen) >= 3, seen


def test_expected_markers_match_the_direction(seeds):
    want = {"native_native": ["yor", "yor"], "english_english": ["eng", "eng"],
            "english_to_native": ["eng", "yor"], "native_to_english": ["yor", "eng"]}
    for i in range(60):
        md = build_request(seeds["sft"], {"custom_id": f"sft__yor__general_chat__{i:06d}",
                                        "lang": "yor", "task": "general_chat", "index": i}).metadata
        d = md["io_direction"]
        if d in want:
            assert md["expect_markers"] == want[d], (d, md["expect_markers"])
        else:  # crosslingual: other language in, target out, and never English on either side
            assert md["expect_markers"][1] == "yor"
            assert md["expect_markers"][0] not in ("eng", "yor")


def test_rag_context_is_specified_as_english(seeds):
    rag = next(t for t in seeds["sft"].tasks if t.name == "rag_document_qa")
    assert "ALWAYS IN ENGLISH" in rag.description
    req = build_request(seeds["sft"], {"custom_id": "sft__fon__rag_document_qa__000001",
                                      "lang": "fon", "task": "rag_document_qa", "index": 1})
    assert "ALWAYS IN ENGLISH" in req.messages[0]["content"]


def test_think_is_allowed_before_and_after_tool_use(seeds):
    brief = build_request(seeds["sft"], {"custom_id": "sft__yor__tool_search_answer__000001",
                                        "lang": "yor", "task": "tool_search_answer", "index": 1}
                          ).messages[0]["content"]
    # a tool round trip is two assistant turns, and the second gets its own think
    assert "takes ANOTHER turn" in brief and "fresh `think`" in brief
    # the plan belongs on the calling turn (114 published tool turns had none) and is revised as work proceeds
    assert "including a tool-calling turn" in brief
    assert "REVISED as work proceeds" in brief


def test_packing_shares_one_system_prompt_and_preserves_order(seeds):
    """The whole point of packing is paying the ~1,200-token system prompt once."""
    from prompts import build_packed_request
    rows = [{"custom_id": f"sft__hau__general_chat__{i:06d}", "lang": "hau",
             "task": "general_chat", "index": i} for i in range(4)]
    packed = build_packed_request(seeds["sft"], rows)
    singles = [build_request(seeds["sft"], r) for r in rows]
    assert packed.messages[0]["content"] == singles[0].messages[0]["content"]
    assert packed.metadata["custom_ids"] == [r["custom_id"] for r in rows]
    assert len(packed.metadata["members"]) == 4
    body = packed.messages[1]["content"]
    for i in range(1, 5):
        assert f"SAMPLE {i} of 4" in body
    # packing one request must cost less than four separate ones
    assert len(body) + len(packed.messages[0]["content"]) < sum(
        len(m["content"]) for s in singles for m in s.messages)


def test_packing_refuses_mixed_languages(seeds):
    from prompts import build_packed_request
    with pytest.raises(ValueError, match="single-language"):
        build_packed_request(seeds["sft"], [
            {"custom_id": "a", "lang": "yor", "task": "general_chat", "index": 0},
            {"custom_id": "b", "lang": "hau", "task": "general_chat", "index": 1}])


def test_packed_response_splits_into_records(seeds):
    from postprocess_gen import to_records
    from prompts import build_packed_request
    rows = [{"custom_id": f"sft__yor__general_chat__{i:06d}", "lang": "yor",
             "task": "general_chat", "index": i} for i in range(2)]
    packed = build_packed_request(seeds["sft"], rows)
    payload = {"samples": [_sft_payload(), _sft_payload()]}
    recs = to_records(seeds["sft"], Response(packed.custom_id, json.dumps(payload), True,
                                            metadata=packed.metadata))
    assert len(recs) == 2
    assert [r["id"] for r in recs] == [r["custom_id"] for r in rows]
    assert {r["lang"] for r in recs} == {"yor"}


def test_a_pack_that_loses_ordering_is_discarded(seeds):
    """More samples than specs means the model lost track and nothing can be trusted to match its spec."""
    from postprocess_gen import to_records
    from prompts import build_packed_request
    rows = [{"custom_id": f"sft__yor__general_chat__{i:06d}", "lang": "yor",
             "task": "general_chat", "index": i} for i in range(2)]
    packed = build_packed_request(seeds["sft"], rows)
    too_many = {"samples": [_sft_payload()] * 3}
    assert to_records(seeds["sft"], Response("x", json.dumps(too_many), True,
                                            metadata=packed.metadata)) == []
    # a SHORT pack is salvaged: take what lined up
    short = {"samples": [_sft_payload()]}
    assert len(to_records(seeds["sft"], Response("y", json.dumps(short), True,
                                                 metadata=packed.metadata))) == 1


# ----------------------------------------------------------------- few-shot / free-tier backoff


def test_fewshot_attaches_only_to_the_low_resource_tier(seeds, monkeypatch):
    """Exemplars cost ~1,600 tokens. They go where structure actually collapses, not everywhere."""
    monkeypatch.delenv("DATA_GEN_FEWSHOT", raising=False)
    from prompts import _fewshot_block
    if not _fewshot_block():
        pytest.skip("seeds/fewshot.json not built")

    def sysmsg(lang):
        return build_request(seeds["sft"], {"custom_id": f"sft__{lang}__rag_document_qa__000001",
                                           "lang": lang, "task": "rag_document_qa",
                                           "index": 1}).messages[0]["content"]

    assert "WORKED EXAMPLES" in sysmsg("efi")      # low
    assert "WORKED EXAMPLES" in sysmsg("fon")      # low
    assert "WORKED EXAMPLES" not in sysmsg("yor")  # medium
    assert "WORKED EXAMPLES" not in sysmsg("pcm")  # high
    monkeypatch.setenv("DATA_GEN_FEWSHOT", "1")
    build_request.__globals__["_fewshot_block"].cache_clear()
    assert "WORKED EXAMPLES" in sysmsg("yor")      # forced on


def test_fewshot_stays_inside_the_shared_prefix(seeds):
    """If exemplars varied per row they would break prefix caching and cost 1,600 tokens every time."""
    from prompts import _fewshot_block
    if not _fewshot_block():
        pytest.skip("seeds/fewshot.json not built")
    seen = {build_request(seeds["sft"], {"custom_id": f"sft__efi__general_chat__{i:06d}", "lang": "efi",
                                        "task": "general_chat", "index": i}).messages[0]["content"]
            for i in range(20)}
    assert len(seen) == 1


def test_fewshot_exemplars_are_structurally_perfect():
    """A wrong exemplar is worse than none -- the model copies it."""
    from pathlib import Path as _P
    p = _P(__file__).resolve().parents[1] / "seeds" / "fewshot.json"
    if not p.exists():
        pytest.skip("not built")
    from postprocess_gen import _degenerate_user_message, _invented_pipe_tokens
    data = json.loads(p.read_text(encoding="utf-8"))
    ex = data["exemplars"]
    assert ex, "no exemplars"
    for e in ex:
        msgs = e["messages"]
        assert _invented_pipe_tokens(msgs) == []
        assert not _degenerate_user_message(msgs)
        assert msgs[-1]["role"] == "assistant"
        assert any("<response>" in (m.get("content") or "") for m in msgs)
    assert any(e["uses_tools"] for e in ex), "need one tool-using exemplar"
    assert any(not e["uses_tools"] for e in ex), "need one plain exemplar"


def test_free_endpoints_get_minutes_of_backoff_and_low_concurrency(monkeypatch):
    """A ':free' endpoint rate-limits on a scale of minutes; a 1.5s->60s backoff just burns retries."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    from providers.openrouter import OpenRouterProvider
    free = OpenRouterProvider("google/gemma-4-31b-it:free", max_concurrency=16)
    paid = OpenRouterProvider("google/gemma-4-31b-it", max_concurrency=16)
    assert free.is_free and not paid.is_free
    assert free.retry_base_s >= 120 and free.retry_max_s >= 300
    assert free.max_concurrency <= 2 < paid.max_concurrency
    assert free.max_retries > paid.max_retries
    assert free.rates() == (0.0, 0.0)
    # an explicit override still wins
    custom = OpenRouterProvider("google/gemma-4-31b-it:free", retry_base_s=30, retry_max_s=90)
    assert custom.retry_base_s == 30 and custom.retry_max_s == 90
