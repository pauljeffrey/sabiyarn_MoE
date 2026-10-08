"""The three-agent loop: retrieval, the length guard, eviction, and the records that come out."""

from __future__ import annotations

import json

import pytest

from agents.loop import (MAX_LENGTH_RATIO, Conversation, Loop, Turn, _tidy_think, records_from,
                         sft_record)
from agents.retrieval import LocalRetriever, Passage, render, split_passages


# --------------------------------------------------------------------------- chunking and retrieval

DOC = """History

The Ampe game is played across Ghana and parts of Togo by children, usually girls, in a circle. Two players
face each other and jump, throwing one foot forward as they land. The scoring depends on whether the feet
thrown match.

Rules

A player nominates a foot before the round begins. If both players throw the same foot the nominator scores a
point; if they differ the opponent scores. Games run to a fixed number of points agreed before play.

Equipment

None is required. The game needs only a flat surface and two or more players, which is part of why it
survives in schoolyards where nothing can be bought."""


def test_passages_split_on_headings_and_keep_whole_paragraphs():
    """Paragraph-level, not sentence-level: a sentence pulled out of a paragraph loses the referent of every
    pronoun in it, and a model asked to answer from that either invents the referent or refuses."""
    ps = split_passages(DOC, target_words=60, doc_id="d1", title="Ampe")
    assert len(ps) >= 3
    assert {p.section for p in ps} >= {"History", "Rules", "Equipment"}
    # no passage starts mid-sentence
    for p in ps:
        assert p.text[0].isupper() or p.text[0].isdigit(), p.text[:40]
    assert all(p.doc_id == "d1" and p.title == "Ampe" for p in ps)


def test_retrieval_returns_a_ranked_result_with_cross_document_distractors():
    """A result is only a useful lesson if the near-misses are real near-misses.

    It also documents what the LOCAL retriever is: BM25 matches WORDS, not meaning. Asked "how does scoring
    work when the feet match" it ranks the History passage above the Rules passage, because History happens
    to contain the sentence "the scoring depends on whether the feet thrown match" while the passage that
    actually explains the rule says "nominates a foot" and "scores a point". That is a defensible lexical
    answer and a bad semantic one, and it is the concrete reason the Pinecone path with a multilingual
    embedding model is worth having -- doubly so for a Yoruba or Hausa query, where lexical overlap with an
    English-language source is near zero.
    """
    other = """Rules

Ludo is played on a cross-shaped board with four tokens each and a single die. A player must throw a six to
bring a token out of the yard before it can move."""
    ps = (split_passages(DOC, target_words=60, doc_id="ampe", title="Ampe") +
          split_passages(other, target_words=60, doc_id="ludo", title="Ludo"))
    r = LocalRetriever.build(ps)
    hits = r.search("how does scoring work in ampe when the feet match", k=4)
    assert hits, "nothing retrieved"
    assert hits[0].score == 1.0, "the top hit is normalised to 1.0 so lower ranks read as weaker"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    # the passage that really answers it is retrieved, just not first
    assert any("nominat" in h.text.lower() for h in hits)
    # and the distractors span DOCUMENTS, which chunking one article cannot give you
    assert len({h.doc_id for h in hits}) > 1, [h.doc_id for h in hits]


def test_render_shows_the_id_and_the_rank():
    ps = split_passages(DOC, target_words=60, doc_id="d1", title="Ampe")
    out = render(LocalRetriever.build(ps).search("scoring", k=2))
    assert "[d1#p" in out or "[p" in out
    assert "score=" in out
    assert render([]) == "No passages matched."


# --------------------------------------------------------------------------- the loop's own rules


class FakeAsk:
    """Scripted replies, so the loop can be driven with no provider."""

    def __init__(self, scripted):
        self.scripted, self.calls = list(scripted), []

    def __call__(self, messages, max_tokens=700):
        self.calls.append(messages)
        return self.scripted.pop(0) if self.scripted else ""


def _loop(scripted, **kw):
    ps = split_passages(DOC, target_words=60, doc_id="d1", title="Ampe")
    return Loop(FakeAsk(scripted), LocalRetriever.build(ps), lang="pcm", lang_name="Nigerian Pidgin",
                today="2026-10-08", **kw)


def test_the_refiner_may_not_make_the_reply_longer():
    """A preference dataset whose chosen side is reliably longer teaches "longer is better" -- the
    best-documented failure mode in preference learning, and ruinous for a model whose point is short,
    code-switched answers. The guard is enforced, not requested."""
    short = "Omo, e don happen."
    long = short + " " + "x" * int(len(short) * MAX_LENGTH_RATIO + 20)
    lp = _loop([json.dumps({"response": long})])
    assert lp.refine({"response": short}, "wetin happen") is None
    assert lp.stats["refine_too_long"] == 1
    # a rewrite of the same length is kept
    lp2 = _loop([json.dumps({"response": "Omo, e don land o."})])
    got = lp2.refine({"response": short}, "wetin happen")
    assert got and got["response"] == "Omo, e don land o."


def test_an_identical_or_empty_rewrite_is_not_a_pair():
    same = "Omo, e don happen."
    lp = _loop([json.dumps({"response": same})])
    assert lp.refine({"response": same}, "q") is None
    assert lp.stats["refine_identical"] == 1
    lp2 = _loop([json.dumps({"response": ""})])
    assert lp2.refine({"response": same}, "q") is None
    assert lp2.stats["refine_empty"] == 1


def test_the_refiner_keeps_the_plan_and_the_tool_call():
    """Only the register is up for revision. The plan and the call are substance."""
    lp = _loop([json.dumps({"response": "E don land."})])
    got = lp.refine({"response": "It has arrived.", "task_plan": ["<|RAG|>", "<qa>"],
                     "tool_call": {"name": "search_documents", "arguments": {"query": "x"}}}, "q")
    assert got["task_plan"] == ["<|RAG|>", "<qa>"]
    assert got["tool_call"]["name"] == "search_documents"


def test_think_drops_a_leading_restatement():
    assert _tidy_think("The user wants to know the fee. The fee changed in 2025 so look it up.") == \
        "The fee changed in 2025 so look it up."
    # a block that was ONLY a restatement is dropped -- see the test below
    assert _tidy_think("The user wants the balance.") == ""
    assert _tidy_think("") == ""


def test_retrieved_passages_are_evicted_from_the_agents_history():
    """Eviction is the whole cost argument: a document left in context is a 2x swing on the run. What a later
    turn refers back to is the assistant's ANSWER, so only a one-line note of the retrieval stays."""
    lp = _loop([])
    lp._turns = [Turn(user="wetin be ampe", assistant={"response": "Na game."},
                      tool_calls=[{"name": "search_documents"}],
                      retrieved=[Passage("d1#p1", "d1", "Ampe", "History", "A long passage " * 40)])]
    hist = lp._history()
    body = "".join(m["content"] for m in hist)
    assert "A long passage" not in body, "the passage body must not survive into the next turn's context"
    assert "retrieved 1 passages" in body and "d1#p1" in body
    assert "Na game." in body


# --------------------------------------------------------------------------- records


def _conv_with_pair():
    return Conversation(lang="pcm", persona="okada_rider", register="heavy", turns=[
        Turn(user="How much be the renewal fee?",
             assistant={"think": "The fee changed, look it up.", "task_plan": ["<|RAG|>", "<qa>"],
                        "response": "The fee is N8,000 according to the document."},
             refined={"task_plan": ["<|RAG|>", "<qa>"],
                      "response": "Omo, the fee na N8,000 based on wetin the document talk."},
             tool_calls=[{"name": "search_documents", "arguments": {"query": "renewal fee"}}],
             retrieved=[Passage("d1#p2", "d1", "Fees", "Schedule", "Renewal costs N8,000 from March.")]),
    ])


def test_one_record_per_paired_turn_with_the_context_written_back_in():
    """The generator evicts the passages; the RECORD must still contain them, or an SFT sample shows an
    answer with no visible source and teaches the model to answer from nowhere."""
    recs = records_from(_conv_with_pair(), row={"custom_id": "rl__pcm__x__000001", "domain": "transport"},
                        lang="pcm", target_lang="pcm", model="m", tasks=["tool_search_answer"],
                        tools=["search_documents"])
    assert len(recs) == 1
    r = recs[0]
    assert r["id"].endswith("__t1") and r["turn_index"] == 1
    assert r["ranking"] == ["best", "worst"]
    assert "Omo" in r["response_1"], "the refined reply is the chosen side"
    assert "according to the document" in r["response_2"]
    # the published prefix carries the call AND the retrieved text
    blob = json.dumps(r["prompt_messages"], ensure_ascii=False)
    assert "Renewal costs N8,000" in blob
    assert r["retrieved_context"] and r["retrieved_context"][0]["id"] == "d1#p2"
    # and the scaffold was extracted for the scorer
    assert r["expected_structure"].startswith("<|input_lang|><pcm>")
    assert "<task_plan>" in r["expected_structure"]
    assert r["persona"] == "okada_rider" and r["register"] == "heavy"


def test_a_turn_with_no_usable_rewrite_yields_no_record_but_stays_in_the_prefix():
    """A turn the refiner could not improve is not a pair -- but it is still conversation, and dropping it
    from the prefix would leave the next turn answering a question nobody asked."""
    conv = _conv_with_pair()
    conv.turns.append(Turn(user="And the surveyor?", assistant={"response": "N30,000."}, refined=None))
    conv.turns.append(Turn(user="Total?", assistant={"response": "N38,000 altogether."},
                           refined={"response": "Na N38,000 total."}))
    recs = records_from(conv, row={"custom_id": "c", "domain": "d"}, lang="pcm", target_lang="pcm",
                        model="m")
    assert [r["turn_index"] for r in recs] == [1, 3]
    blob = json.dumps(recs[-1]["prompt_messages"], ensure_ascii=False)
    assert "And the surveyor?" in blob and "N30,000" in blob


def test_sft_record_has_one_reply_per_turn_and_no_pair():
    conv = _conv_with_pair()
    rec = sft_record(conv, row={"custom_id": "sft__pcm__x__000001", "domain": "transport"}, lang="pcm",
                     target_lang="pcm", model="m", tasks=["tool_search_answer"])
    assert rec and "response_1" not in rec and "ranking" not in rec
    roles = [m.get("role") for m in rec["messages"]]
    assert roles[0] == "user" and "tool" in roles
    assert rec["retrieved_context"][0]["id"] == "d1#p2"
    assert rec["n_turns"] == 1


def test_an_empty_conversation_makes_no_record():
    empty = Conversation(lang="pcm", turns=[])
    assert records_from(empty, row={"custom_id": "c"}, lang="pcm", target_lang="pcm", model="m") == []
    assert sft_record(empty, row={"custom_id": "c"}, lang="pcm", target_lang="pcm", model="m") is None


def test_a_think_that_is_only_a_restatement_is_dropped_entirely():
    """Measured on the loop: 3 of 4 think blocks were nothing but "The user is asking for X" with no decision
    after it. The contract already says to omit <think> when nothing needs deciding."""
    assert _tidy_think("The user is asking for app recommendations for daily savings in Hausa.") == ""
    assert _tidy_think("The user wants to know if AI truly thinks independently.") == ""
    # a real decision survives, with or without a restatement in front of it
    assert _tidy_think("The user wants the fee. Look it up.") == "Look it up."
    assert _tidy_think("This is a legal matter that varies by jurisdiction, so it must be looked up.") == \
        "This is a legal matter that varies by jurisdiction, so it must be looked up."


def test_the_refiner_never_supplies_the_task_plan():
    """Asked for a plan, the refiner returns the list of EDITS it is about to make -- a plan, just not this
    one. It cost 11 of 17 pairs on a live run."""
    lp = _loop([json.dumps({"response": "E don land.",
                            "task_plan": ["Replace 'manhajoji' with 'apps'", "Insert particle 'dai'"]})])
    got = lp.refine({"response": "It has arrived now.", "task_plan": ["<|RAG|>", "<qa>"]}, "q")
    assert got["task_plan"] == ["<|RAG|>", "<qa>"], "the original's plan, not the refiner's edit list"


def test_a_plan_entry_that_is_not_a_token_is_dropped_not_fatal():
    """An unrecognised plan verb fails assembly and would cost the whole turn, so it is filtered and the
    task's canonical plan stands in."""
    from agents.loop import _assistant_turns
    t = _assistant_turns({"response": "a", "task_plan": ["Simplify the phrasing", "<qa>"]},
                         "hau", "hau", ["<|RAG|>"])
    assert t[0]["task_plan"] == ["<qa>"]
    t2 = _assistant_turns({"response": "a", "task_plan": ["not a token at all"]}, "hau", "hau", ["<|RAG|>"])
    assert t2[0]["task_plan"] == ["<|RAG|>"]
    t3 = _assistant_turns({"response": "a", "task_plan": []}, "hau", "hau", [])
    assert t3[0]["task_plan"] == ["<|chat|>"]


def test_a_tool_call_turns_plan_is_filtered_too():
    """The call turn was the one path that did not filter, and it is exactly where the next leak appeared:
    the agent put `search_internet(query='...')` in its task_plan and the pair was lost."""
    from agents.loop import _valid_plan
    assert _valid_plan(["search_internet(query='x')"], ["<|RAG|>"], "<|plan|>") == ["<|RAG|>"]
    assert _valid_plan(None, None, "<|plan|>") == ["<|plan|>"]
    assert _valid_plan(["<|RAG|>", "nonsense", "<qa>"], None) == ["<|RAG|>", "<qa>"]


def test_latex_is_repaired_not_rejected():
    """A live Hausa reply rendered a hierarchy as "AI $\\rightarrow$ Machine Learning". The arrow is right and
    the markup is not -- and these users have never seen a dollar-delimited equation."""
    from agents.loop import _clean_response
    assert _clean_response(r"AI $\rightarrow$ ML $\rightarrow$ DL") == "AI → ML → DL"
    assert _clean_response(r"Kudin $N8,000$ ne, \text{total} ya kai 10") == "Kudin N8,000 ne, total ya kai 10"
    assert _clean_response("plain text stays") == "plain text stays"


def test_the_refiner_does_not_invent_a_think_block():
    """Invited to "keep or shorten" the think, the refiner wrote a fresh one on EVERY turn -- including turns
    where the base agent had rightly omitted it. 17 of 17 chosen replies carried <think> against 5 of 18 base
    replies: the refiner was padding the side we reward."""
    lp = _loop([json.dumps({"response": "E don land.", "notes": ["dropped the calque"]})])
    got = lp.refine({"response": "It has arrived now."}, "q")
    assert got["think"] == "", "no think in the original means no think in the rewrite"
    # the edit list is kept as a DIAGNOSTIC, not folded into the sample
    assert got["_notes"] == ["dropped the calque"]
    lp2 = _loop([json.dumps({"response": "E don land."})])
    got2 = lp2.refine({"response": "It has arrived now.", "think": "The fee changed, so look it up."}, "q")
    assert got2["think"] == "The fee changed, so look it up."


def test_the_assistant_is_told_how_to_query_and_what_it_does_not_know():
    """Three things the agent was never told, each of which showed up as a defect in live output:

    * which language to query in -- measured, an English query returns the right document at 0.39-0.60
      similarity while the same question in Hausa or Yoruba returns the WRONG one at ~0.17, the noise floor,
      so a query is tagged with the language of the CORPUS and local-language corpora go to BM25;
    * that it should read evidence in any language and answer in the one asked for;
    * that its own knowledge is capped, which is what actually makes it reach for a tool instead of
      answering plausibly from memory.
    """
    from agents import agent_prompts as AP
    for tmpl in (AP.AI_AGENT, AP.SFT_AI_AGENT):
        body = tmpl.format(lang_name="Hausa", lang="hau", today="2026-10-08", tools_block="",
                           plan_hint='"<|RAG|>"', task_directive="", flavour_hint="x")
        assert "QUERY THE CORPUS IN ITS OWN LANGUAGE" in body
        assert '"lang": "hau"' in body, "the example must name THIS conversation's language"
        assert "READ WHATEVER COMES BACK" in body
        assert "16-year-old" in body and "LOOK UP with the tools" in body
        assert "THINK IN ENGLISH" in body


def test_the_refiner_may_correct_a_bad_think_but_not_add_one():
    """It may replace the reasoning when the original's was wrong; the common case costs nothing."""
    lp = _loop([json.dumps({"response": "E don land.", "think": "The 2025 figure is the one that applies."})])
    got = lp.refine({"response": "It has arrived now.", "think": "The user wants the figure."}, "q")
    assert got["think"] == "The 2025 figure is the one that applies."
    assert lp.stats["refine_fixed_think"] == 1


def test_the_edit_list_never_reaches_the_published_turn():
    """The published turn is a reply to the user, who never saw a draft. A model trained on "here is what I
    changed about my previous answer" learns to narrate revisions nobody asked for -- so the notes ride on
    the record as a diagnostic and `<|edit|>` is not appended to the plan."""
    conv = Conversation(lang="pcm", turns=[
        Turn(user="How much?", assistant={"task_plan": ["<|RAG|>"], "response": "It is N8,000."},
             refined={"task_plan": ["<|RAG|>"], "response": "Na N8,000.",
                      "_notes": ["dropped the calque", "added 'na'"]})])
    rec = records_from(conv, row={"custom_id": "c"}, lang="pcm", target_lang="pcm", model="m")[0]
    assert rec["refine_notes"] == ["dropped the calque", "added 'na'"]
    assert "dropped the calque" not in rec["response_1"]
    assert "<|edit|>" not in rec["response_1"]
    assert "<think>" not in rec["response_1"]


def test_the_web_is_two_steps_and_a_serp_is_not_evidence():
    """search_internet returns titles, urls and snippets; the assistant must pick one and fetch it. Handing
    back four full documents removes the decision, and the decision is the behaviour being taught."""
    from agents.retrieval import Library
    ps = split_passages(DOC, target_words=60, doc_id="d1", title="Ampe", url="https://x.test/ampe")
    lib = Library(by_url={"https://x.test/ampe": {"title": "Ampe", "text": DOC, "url": "https://x.test/ampe"}})
    lp = _loop([], library=lib)
    lp.retriever = LocalRetriever.build(ps)
    serp, cited = lp._run_tool({"name": "search_internet", "arguments": {"query": "ampe scoring"}}, "q")
    assert "https://x.test/ampe" in serp and "Fetch one of these" in serp
    assert cited == [], "a SERP is not evidence -- nothing is cited until a page is fetched"
    page, cited2 = lp._run_tool({"name": "fetch_page", "arguments": {"url": "https://x.test/ampe"}}, "q")
    assert "cookies" in page and "last edited" in page, "a real page carries clutter to read past"
    assert "nominates a foot" in page
    assert cited2 and cited2[0].url == "https://x.test/ampe"
    # an address that was never offered is a hallucinated url, and it must look like a failure
    missing, _ = lp._run_tool({"name": "fetch_page", "arguments": {"url": "https://nope.test/x"}}, "q")
    assert "404" in missing
    assert lp.stats["fetched_unoffered_url"] == 1


def test_queries_may_be_tagged_with_a_language_and_routed():
    """A question spanning two corpora sends two queries. Local-language corpora go to BM25, because the
    embedder is blind to them."""
    from agents.retrieval import HybridRetriever, EMBEDDABLE
    assert "eng" in EMBEDDABLE and "hau" not in EMBEDDABLE and "yor" not in EMBEDDABLE
    ps = (split_passages(DOC, target_words=60, doc_id="en1", title="Ampe", lang="eng") +
          split_passages("Wasa\n\nAmpe wasa ne na yara a Ghana da suke tsalle tare da buga hannu.",
                         target_words=60, doc_id="ha1", title="Ampe (Hausa)", lang="hau"))
    h = HybridRetriever(local=LocalRetriever.build(ps), remote=None)
    assert all(p.lang == "hau" for p in h.search("wasa yara Ghana", k=2, lang="hau"))
    lp = _loop([])
    lp.retriever = h
    qs = lp._queries({"queries": [{"query": "ampe scoring", "lang": "eng"},
                                  {"query": "wasa yara", "lang": "hau"}]}, "fallback")
    assert qs == [("ampe scoring", "eng"), ("wasa yara", "hau")]
    assert lp._queries({"query": "one thing"}, "fb") == [("one thing", "")]
    assert lp._find(qs, k=3), "a multi-language query must return something"


def test_a_failing_remote_index_does_not_stop_the_run():
    """The failure that actually happened: the organisation's monthly Pinecone embedding quota ran out
    mid-run -- 5M tokens for llama-text-embed-v2 against a corpus needing ~7.8M to ingest -- and the whole
    comparison crashed on a 429. A quota does not come back on the next call, so this is a circuit breaker
    rather than a retry: after a few failures the remote is off and BM25 carries the run."""
    from agents.retrieval import HybridRetriever

    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("[429 RESOURCE_EXHAUSTED] embedding token limit")

    ps = split_passages(DOC, target_words=60, doc_id="d1", title="Ampe", lang="eng")
    h = HybridRetriever(local=LocalRetriever.build(ps), remote=Broken(), max_remote_failures=2)
    assert h.search("ampe scoring", k=2), "BM25 must answer when the remote throws"
    assert h.remote is not None, "one failure is not enough to disable it"
    assert h.search("ampe scoring", k=2)
    assert h.remote is None, "repeated failure disables the remote for the rest of the run"
    assert h.search("ampe scoring", k=2), "and the run continues on BM25"
