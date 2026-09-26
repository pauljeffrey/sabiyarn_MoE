"""Two-stage long-document generation: planning, the splice, and the budget arithmetic.

The properties worth locking down are the ones that were WRONG before: the document's length (one-stage
generation produced 635 words against a 900+ target), the language split (70/10/20, which a non-coprime
stride would collapse), and the fact that the document is never re-emitted as output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import longdocs as L                                        # noqa: E402
from prompts import build_request, coverage_pick            # noqa: E402
from providers.base import Response                         # noqa: E402
from schemas.seed import Seed                               # noqa: E402


@pytest.fixture(scope="module")
def seed():
    return Seed.load("sft")


def _row(lang="yor", index=3, task="long_document_summarization"):
    return {"custom_id": f"sft__{lang}__{task}__{index:06d}", "lang": lang, "task": task, "index": index}


def _spec(seed, lang="yor", index=3):
    r = _row(lang, index)
    d, s, _ = coverage_pick(lang, index)
    return r, L.plan_document(seed, r, domain=d, subtopic=s)


def _doc(words=1500, lang="eng", mode="eng"):
    return {"title": "A Report", "text": " ".join(f"w{i}" for i in range(words)), "words": words,
            "doc_lang": lang, "doc_lang_mode": mode, "form": "field_report"}


# --------------------------------------------------------------------------- budget


def test_word_budget_scales_with_the_language():
    """One word figure cannot serve both: Yoruba costs ~2.5 tokens/word against English's ~1.15, so the same
    token budget is twice as many English words. A single figure would truncate Yoruba or waste the budget."""
    w = lambda lg: L.words_for_tokens(8000, lg)                      # noqa: E731
    assert w("eng") > w("pcm") > w("yor") > w("fon")


def test_token_band_covers_4k_to_16k(seed):
    """The owner's band. The stride must be coprime with the 10-bucket cycle or most bands are unreachable."""
    sizes = sorted({_spec(seed, "pcm", i)[1].tokens for i in range(200)})
    assert min(sizes) >= 4000 and max(sizes) <= 16000
    bands = {b for i in range(200) for b in [L._TOKEN_BANDS[(i * L._BAND_STRIDE
                                                            + L._offset("pcm")) % len(L._TOKEN_BANDS)]]}
    assert bands == set(L._TOKEN_BANDS)


def test_band_is_weighted_towards_shorter_documents():
    """A 16,000-token document costs ~7x a 4,000-token one and is ~7x rarer in practice."""
    assert L._TOKEN_BANDS.count((4000, 6000)) > L._TOKEN_BANDS.count((12000, 16000))


def test_pinning_the_budget_overrides_the_band(seed, monkeypatch):
    monkeypatch.setattr(L, "_PINNED", 5000)
    assert all(_spec(seed, "hau", i)[1].tokens == 5000 for i in range(10))


# --------------------------------------------------------------------------- planning


def test_planning_is_deterministic(seed):
    a = L.plan_document(seed, _row(), domain="health", subtopic="malaria").as_dict()
    b = L.plan_document(seed, _row(), domain="health", subtopic="malaria").as_dict()
    assert a == b


def test_document_language_split_is_70_10_20(seed):
    """The owner's split. The stride must be coprime with the 10-bucket cycle or most buckets are unreachable
    -- the arithmetic trap that once locked every sub-topic to one genre."""
    modes: dict[str, int] = {}
    for i in range(200):
        _, spec = _spec(seed, "yor", i)
        modes[spec.doc_lang_mode] = modes.get(spec.doc_lang_mode, 0) + 1
    assert modes["eng"] == 140 and modes["native"] == 40 and modes["mixed"] == 20


def test_english_target_language_never_asks_for_a_mixed_document(seed):
    for i in range(20):
        _, spec = _spec(seed, "eng", i)
        assert spec.doc_lang_mode == "eng" and spec.doc_lang == "eng"


def test_low_resource_sections_are_shorter(seed):
    """220 words of fluent Fon beats 420 that drift into French halfway through."""
    native = [s for s in (_spec(seed, "fon", i)[1] for i in range(40)) if s.doc_lang_mode == "native"]
    assert native and all(s.section_words < L._SECTION_WORDS["eng"] for s in native)


def test_section_count_follows_the_word_target(seed):
    for i in range(20):
        _, spec = _spec(seed, "pcm", i)
        assert spec.n_sections >= len(spec.stages)
        assert spec.n_sections * spec.section_words >= spec.words
        assert spec.parts == max(1, -(-spec.n_sections // L.SECTIONS_PER_PART))


def test_all_forms_are_reachable(seed):
    forms = {_spec(seed, "hau", i)[1].form for i in range(len(L._FORMS) * 3)}
    assert forms == {f[0] for f in L._FORMS}


# --------------------------------------------------------------------------- stage 0


def test_outline_request_asks_for_a_shared_cast_and_facts(seed):
    """Parallel section writing is only safe because every part gets the same people and numbers."""
    _, spec = _spec(seed)
    body = L.outline_request(seed, spec).messages[1]["content"]
    assert "cast" in body and "facts" in body and "tension" in body
    assert f"EXACTLY {spec.n_sections} sections" in body
    for stage in spec.stages:
        assert stage in body


def test_outline_is_rejected_when_it_is_far_too_short(seed):
    _, spec = _spec(seed)
    resp = Response("x", '{"title": "T", "sections": [{"heading": "One"}]}')
    assert L.parse_outline(resp, spec) is None


def test_outline_parses_and_drops_headingless_sections(seed):
    _, spec = _spec(seed)
    secs = [{"heading": f"H{i}", "brief": "b"} for i in range(spec.n_sections)] + [{"brief": "no heading"}]
    resp = Response("x", json.dumps({"title": "T", "setting": "S", "cast": [{"name": "A", "role": "r"}],
                                     "facts": ["f"], "tension": "t", "sections": secs}))
    got = L.parse_outline(resp, spec)
    assert len(got["sections"]) == spec.n_sections and got["cast"] == [{"name": "A", "role": "r"}]


def test_shared_facts_reach_every_part(seed):
    """The failure this prevents: part 3 inventing a different chairperson and a different budget."""
    _, spec = _spec(seed, "pcm", 4)
    outline = {"title": "T", "setting": "Ìbàdàn, 2025", "cast": [{"name": "Mrs Adébáyọ̀", "role": "chair"}],
               "facts": ["budget N4,200,000"], "tension": "the shortfall",
               "sections": [{"heading": f"H{i}", "brief": "b"} for i in range(13)]}
    bodies = [L.document_request(seed, spec, outline, p).messages[1]["content"] for p in range(3)]
    for body in bodies:
        assert "Mrs Adébáyọ̀" in body and "budget N4,200,000" in body and "the shortfall" in body


def test_a_later_part_is_told_not_to_reintroduce_the_document(seed):
    _, spec = _spec(seed, "pcm", 4)
    outline = {"title": "The Report", "sections": [{"heading": f"H{i}"} for i in range(13)],
               "cast": [], "facts": [], "setting": "", "tension": ""}
    later = L.document_request(seed, spec, outline, 2).messages[1]["content"]
    assert "PART 3 OF 3" in later and "do not recap earlier parts" in later
    first = L.document_request(seed, spec, outline, 0).messages[1]["content"]
    assert "PART 1 OF 3" in first


def test_single_part_documents_skip_the_outline_request(seed):
    """A document that cannot contradict itself is not worth a stage-0 round trip."""
    _, spec = _spec(seed)
    o = L.local_outline(spec)
    assert len(o["sections"]) == min(spec.n_sections, len(spec.stages))
    assert o["cast"] == [] and o["facts"] == []


# --------------------------------------------------------------------------- stage 1 request


def test_stage_one_request_asks_for_its_own_sections_and_their_length(seed):
    _, spec = _spec(seed)
    outline = {"title": "T", "cast": [], "facts": [], "setting": "", "tension": "",
               "sections": [{"heading": f"Heading {i}", "brief": f"brief {i}"} for i in range(9)]}
    req = L.document_request(seed, spec, outline, 0)
    body = req.messages[1]["content"]
    for i in range(L.SECTIONS_PER_PART):
        assert f"Heading {i}" in body and f"brief {i}" in body
    assert "Heading 6" not in body                              # belongs to part 2
    # the per-section budget is rescaled to the outline we actually got, not the one we asked for
    assert f"about {L.rescale(spec, outline)} words" in body
    # prose is the only thing in this completion, so it gets real headroom
    assert req.max_tokens >= L.rescale(spec, outline) * L.SECTIONS_PER_PART * 2


def test_stage_one_forbids_markdown_and_markup(seed):
    _, spec = _spec(seed)
    body = L.document_request(seed, spec, L.local_outline(spec)).messages[1]["content"]
    assert "NO markdown" in body and "NO html tags" in body


# --------------------------------------------------------------------------- cleaning / parsing


def test_clean_document_strips_markup_without_eating_words():
    raw = "<h1>Background</h1>\n**Important:** the clinic opened.\n- first point\n<div class='x'>text</div>"
    out = L.clean_document(raw)
    assert "<" not in out and "**" not in out
    for word in ("Background", "Important:", "clinic opened", "first point", "text"):
        assert word in out


def test_parse_document_counts_words_and_tolerates_a_fence():
    resp = Response("x", '```json\n{"title": "T", "text": "one two three"}\n```')
    assert L.parse_document(resp) == {"title": "T", "text": "one two three", "words": 3}


def test_parse_document_rejects_an_empty_body():
    assert L.parse_document(Response("x", '{"title": "T", "text": "  "}')) is None
    assert L.parse_document(Response("x", "", ok=False)) is None


# --------------------------------------------------------------------------- the splice


def test_splice_puts_the_document_in_the_user_turn():
    turns = [{"role": "user", "content": f"Check this: {L.PLACEHOLDER} -- summarise am"},
             {"role": "assistant", "input_lang": "eng", "target_lang": "pcm", "task_plan": ["<summarize>"],
              "response": "Di report talk say..."}]
    out, err = L.splice(turns, _doc(50))
    assert err is None
    assert L.PLACEHOLDER not in out[0]["content"]
    assert out[0]["content"].startswith("Check this: A Report")
    assert out[0]["content"].endswith("summarise am")


def test_splice_rejects_a_missing_placeholder():
    _, err = L.splice([{"role": "user", "content": "summarise the document"}], _doc())
    assert err == "placeholder_missing"


def test_splice_rejects_the_placeholder_in_an_assistant_response():
    """A summary that REFERS to the document instead of summarising it is the sample being worthless."""
    turns = [{"role": "user", "content": f"here {L.PLACEHOLDER}"},
             {"role": "assistant", "response": f"The document {L.PLACEHOLDER} says..."}]
    _, err = L.splice(turns, _doc())
    assert err == "placeholder_in_response"


def test_splice_rejects_the_placeholder_in_a_tool_result():
    turns = [{"role": "user", "content": f"here {L.PLACEHOLDER}"},
             {"role": "tool", "name": "search_documents", "content": f"passage: {L.PLACEHOLDER}"}]
    _, err = L.splice(turns, _doc())
    assert err == "placeholder_in_tool_turn"


# --------------------------------------------------------------------------- stage 2


def test_stage_two_carries_the_document_as_input_not_output(seed):
    """The whole point of two stages: the document is in the PROMPT and the completion budget holds only the
    conversation. One-stage generation had them competing, and the document lost."""
    doc = _doc(1800)
    req = build_request(seed, _row(), doc)
    body = req.messages[1]["content"]
    assert doc["text"][:40] in body                 # the document is input
    assert L.PLACEHOLDER in body                    # and the generator is told to reference it
    assert req.max_tokens <= 4096                   # no budget reserved for re-emitting it


def test_stage_two_markers_follow_the_document_language(seed):
    """An English report summarised into Yoruba is <|input_lang|><eng>, whatever io_direction would have said
    for a monolingual conversation."""
    req = build_request(seed, _row("yor", 3), _doc(1200, "eng", "eng"))
    src, tgt = req.metadata["expect_markers"]
    assert src == "eng" and tgt == "yor"


def test_stage_two_brief_does_not_contradict_the_document_language(seed):
    req = build_request(seed, _row("yor", 3), _doc(1200, "eng", "eng"))
    body = req.messages[1]["content"]
    assert "document the user pastes is in English" in body


def test_stage_two_keeps_the_document_the_subject_of_the_conversation(seed):
    for i in range(12):
        req = build_request(seed, _row("pcm", i), _doc(1500))
        assert 1 <= req.metadata["n_user_turns"] <= 3
        assert req.metadata["min_user_turns"] == 1 and req.metadata["max_user_turns"] == 3


def test_stage_two_covers_one_task_only(seed):
    req = build_request(seed, _row(), _doc())
    assert req.metadata["tasks"] == ["long_document_summarization"]
    assert "change subject at least once" not in req.messages[1]["content"]


def test_a_document_row_without_a_document_still_builds(seed):
    """generate.py defers these, but build_request must not raise -- a crash mid-run over a missing document
    would lose the whole batch."""
    req = build_request(seed, _row())
    assert req.custom_id == _row()["custom_id"]


# --------------------------------------------------------------------------- store


def test_document_store_round_trips(tmp_path):
    store = L.DocumentStore(tmp_path / "docs")
    assert "d1" not in store
    store.put({"doc_id": "d1", "title": "T", "text": "body", "words": 1})
    store.close()
    again = L.DocumentStore(tmp_path / "docs")
    assert "d1" in again and again.get("d1")["text"] == "body"


def test_document_store_ignores_a_half_written_line(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "documents.jsonl").write_text('{"doc_id":"a","text":"x","words":1}\n{"doc_id":"b","te',
                                       encoding="utf-8")
    store = L.DocumentStore(d)
    assert "a" in store and "b" not in store


def test_native_documents_are_capped_by_tier(seed):
    """4,775 words of Fon from a generator that half-knows Fon is 4,775 words of drift."""
    for lang, cap in (("fon", 6000), ("hau", 9000), ("pcm", 12000)):
        native = [s for s in (_spec(seed, lang, i)[1] for i in range(120))
                  if s.doc_lang_mode == "native"]
        assert native, lang
        assert max(s.tokens for s in native) <= cap, lang


def test_english_documents_keep_the_whole_band(seed):
    """The cap applies to the target language only: the long documents people actually paste are English."""
    eng = [s for s in (_spec(seed, "fon", i)[1] for i in range(200)) if s.doc_lang_mode == "eng"]
    assert max(s.tokens for s in eng) > 12000


def test_size_and_language_are_independent(seed):
    """They were not. Both cycles were 10 long, so they had the same period in `index` and only 10 of the 30
    (language, size) pairs ever occurred -- no Fon row ever drew a long English document. This is the same
    coupling that once locked every sub-topic to one genre, and it is caught the same way: by counting."""
    pairs = set()
    for i in range(400):
        _, s = _spec(seed, "fon", i)
        pairs.add((s.doc_lang_mode, L._TOKEN_BANDS[(i * L._BAND_STRIDE
                                                    + L._offset("fon")) % len(L._TOKEN_BANDS)]))
    assert len(pairs) == 3 * len(set(L._TOKEN_BANDS))


# --------------------------------------------------------------------------- robustness


def test_section_count_is_capped_and_words_are_preserved(seed):
    """More than MAX_SECTIONS in one outline and the model starts dropping them, so the word target is met by
    making each section longer instead."""
    for lg in ("eng", "pcm", "yor"):
        for i in range(60):
            _, s = _spec(seed, lg, i)
            assert s.n_sections <= L.MAX_SECTIONS
            assert s.section_words <= L._MAX_SECTION_WORDS
            assert s.n_sections * s.section_words >= min(s.words, L.MAX_SECTIONS * L._MAX_SECTION_WORDS)


def test_a_short_outline_is_rescaled_not_rejected(seed):
    """A model that returns 8 sections where 20 were asked for still has a usable plan; each section is simply
    asked to be longer. Rejecting it cost a whole document."""
    _, spec = _spec(seed, "eng", 2)
    outline = {"sections": [{"heading": f"H{i}"} for i in range(8)], "cast": [], "facts": [],
               "setting": "", "tension": "", "title": "T"}
    assert L.parse_outline(Response("x", json.dumps(outline)), spec) is not None
    assert L.rescale(spec, outline) > spec.section_words
    assert L.rescale(spec, outline) <= L._MAX_SECTION_WORDS


def test_an_outline_with_almost_no_sections_is_still_rejected(seed):
    _, spec = _spec(seed, "eng", 2)
    assert L.parse_outline(Response("x", '{"sections": [{"heading": "One"}]}'), spec) is None


def test_local_outline_is_the_fallback_for_a_failed_stage_zero(seed):
    """One transient outline failure used to cost an entire 9,700-word document."""
    _, spec = _spec(seed, "yor", 1)
    assert spec.parts > 1
    o = L.local_outline(spec)
    assert 4 <= len(o["sections"]) <= spec.n_sections


def test_word_floor_is_absolute_not_a_fraction_of_target(seed):
    """A 3,952-word document asked to be 6,686 is still an excellent long document -- 6x what one-stage
    generation managed. A 529-word one is a passage, not a document."""
    _, spec = _spec(seed, "ibo", 2)
    assert spec.min_words <= 1200
    assert 529 < spec.min_words < 3952


def test_a_gap_between_parts_is_rejected_but_a_short_prefix_is_kept(seed, tmp_path, monkeypatch):
    """Parts [0, 2] jump from the background to the recommendations, and a summary of that is a summary of
    something incoherent. Parts [0, 1] of 3 is just a document about less. The first version of this rule had
    it backwards and accepted two-part documents that had simply lost their ending."""
    _, spec = _spec(seed, "eng", 2)
    outline = {"title": "T", "cast": [], "facts": [], "setting": "", "tension": "",
               "sections": [{"heading": f"H{i}"} for i in range(18)]}
    def body(part):                     # distinct per part, or joining them looks like a loop
        return " ".join(f"p{part}w{i}" for i in range(3000))

    class FakeProvider:
        model = "fake"

        def __init__(self, keep):
            self.keep = keep

        def complete_many(self, requests, **kw):
            for r in requests:
                md = r.metadata
                if md["stage"] == "outline":
                    yield Response(r.custom_id, json.dumps(outline), metadata=md)
                elif md["part"] in self.keep:
                    yield Response(r.custom_id, json.dumps({"text": body(md["part"])}), metadata=md)
                else:
                    yield Response(r.custom_id, "", ok=False, metadata=md)

    store = L.DocumentStore(tmp_path / "gap")
    assert L.generate_documents(seed, [spec], FakeProvider({0, 2}), store) == {}   # gap -> rejected
    store.close()
    store2 = L.DocumentStore(tmp_path / "prefix")
    got = L.generate_documents(seed, [spec], FakeProvider({0, 1}), store2)         # prefix -> kept
    store2.close()
    assert got[spec.custom_id]["parts"] == 2 and got[spec.custom_id]["parts_wanted"] == 3


def test_a_looping_document_is_rejected_in_stage_one(seed):
    """Caught where it is the document's problem, not in stage 2 where it would discard a sound conversation
    for the sake of its input."""
    assert L._distinct_ngram("the clinic opened " * 200) < 0.55
    assert L._distinct_ngram(" ".join(f"w{i}" for i in range(500))) > 0.9
