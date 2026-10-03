"""Tests for doc_sources.py -- real documents for the RAG and summarisation tasks.

Network tests are skipped unless SABIYARN_NET=1, so the suite stays offline and fast. The scoring and pool
logic -- which is where the decisions live -- is tested without the network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from doc_sources import (  # noqa: E402
    MIN_TECHNICALITY, TOPIC_SEEDS, DocumentPool, clean, fill_store, technicality,
)

NET = os.environ.get("SABIYARN_NET") == "1"


# --------------------------------------------------------------------------- technicality

def test_numeric_reference_prose_scores_above_the_threshold():
    """The criterion is SPECIFIC, CHECKABLE, LOOKED-UP fact. A figure with a unit is the strongest signal,
    because it is precisely what a 306M model cannot hold and so must reach for a tool to get."""
    technical = (
        "Emitters are rated at a discharge rate of 2 l/h to 8 l/h at an operating pressure of 100 kPa. "
        "The recommended application rate is 120 kg/ha of nitrogen, split into two doses of 60 kg/ha each. "
        "Soil pH should be maintained between 5.5 and 6.8, and the lime requirement is 2.5 t/ha where pH "
        "falls below 5.0. Maturation takes 90 to 120 days depending on the variety. Moisture content must "
        "be reduced to 13% before storage, measured at 25 °C. " * 4)
    assert technicality(technical) >= MIN_TECHNICALITY


def test_general_interest_prose_scores_below_the_threshold():
    """A retrieved document about what everyone knows teaches the model to call a tool for things it
    already knows -- the over-retrieval failure the knowledge-boundary tasks exist to prevent."""
    general = (
        "He was born in Abeokuta and began his career as a singer in the late 1960s, recording his first "
        "album with the band shortly afterwards. His discography spans four decades and his filmography "
        "includes two documentaries. The novel's plot follows a character through the same period. He died "
        "in Lagos, and awards and nominations followed posthumously. " * 8)
    assert technicality(general) < MIN_TECHNICALITY


def test_list_articles_are_penalised():
    body = ("The table lists values of 12 mm, 16 mm and 20 mm at 100 kPa and 250 kPa, with ratings of "
            "4 kW and 6 kW. " * 20)
    assert technicality(body, "List of pump models") < technicality(body, "Pump sizing")


def test_short_documents_score_zero():
    """Too short to judge, and too short to be a retrieval context either."""
    assert technicality("2 l/h at 100 kPa. 120 kg/ha of nitrogen.") == 0.0


def test_score_is_bounded():
    extreme = "10 mg 20 mg 30 kg/ha 40 kPa dosage tolerance rating threshold 1-2 3-4 " * 200
    assert 0.0 <= technicality(extreme) <= 1.0
    assert 0.0 <= technicality("plot album discography " * 300) <= 1.0


# --------------------------------------------------------------------------- cleaning

def test_clean_drops_navigation_sections_only():
    raw = ("Intro paragraph about discharge rates.\n"
           "== Design ==\nThe emitter is rated at 4 l/h.\n"
           "== References ==\n[1] Someone, 2003.\n"
           "== External links ==\nhttp://example.org\n")
    out = clean(raw)
    assert "discharge rates" in out and "rated at 4 l/h" in out
    assert "Someone, 2003" not in out and "example.org" not in out
    assert "== Design ==" in out, "content headings must survive"


def test_clean_keeps_the_clutter_that_makes_it_realistic():
    """Over-cleaning defeats the purpose: the irrelevant-but-plausible paragraph is exactly what the model
    has to learn to read past in a real retrieval."""
    raw = ("== Etymology ==\nThe name derives from a local word.\n"
           "== Specifications ==\nRated 4 l/h at 100 kPa.\n")
    out = clean(raw)
    assert "Etymology" in out and "derives from a local word" in out


# --------------------------------------------------------------------------- the pool

def _doc(pid: int, words: int, domain: str = "agriculture_crops") -> dict:
    return {"title": f"Doc {pid}", "pageid": pid, "url": f"https://en.wikipedia.org/?curid={pid}",
            "text": "word " * words, "words": words, "technicality": 0.5, "domain": domain,
            "source": "wikipedia", "licence": "CC BY-SA 4.0"}


def test_pool_round_trips_and_deduplicates(tmp_path):
    p = DocumentPool(tmp_path / "pool.jsonl")
    assert p.extend([_doc(1, 2000), _doc(2, 3000)]) == 2
    assert p.extend([_doc(1, 2000), _doc(3, 4000)]) == 1, "page id 1 is already pooled"
    assert len(DocumentPool(tmp_path / "pool.jsonl")) == 3, "must survive a reload"


def test_pool_take_prefers_the_closest_length(tmp_path):
    p = DocumentPool(tmp_path / "pool.jsonl")
    p.extend([_doc(1, 2000), _doc(2, 6000), _doc(3, 3100)])
    assert p.take(words=3000)["pageid"] == 3


def test_pool_take_prefers_the_requested_domain(tmp_path):
    p = DocumentPool(tmp_path / "pool.jsonl")
    p.extend([_doc(1, 3000, "agriculture_crops"), _doc(2, 3000, "health_medicine")])
    assert p.take(words=3000, domain="health_medicine")["pageid"] == 2


def test_pool_never_hands_out_the_same_document_twice(tmp_path):
    p = DocumentPool(tmp_path / "pool.jsonl")
    p.extend([_doc(i, 3000) for i in range(3)])
    got = {p.take(words=3000)["pageid"] for _ in range(3)}
    assert len(got) == 3
    assert p.take(words=3000) is None, "an exhausted pool must say so, not repeat"


def test_pool_take_is_deterministic_for_a_key(tmp_path):
    """So a re-run picks the same document for the same sample and the corpus is reproducible."""
    docs = [_doc(i, 3000) for i in range(6)]
    a = DocumentPool(tmp_path / "a.jsonl"); a.extend(docs)
    b = DocumentPool(tmp_path / "b.jsonl"); b.extend(docs)
    assert a.take(words=3000, key="x")["pageid"] == b.take(words=3000, key="x")["pageid"]


# --------------------------------------------------------------------------- the bridge

class _Spec:
    def __init__(self, cid, doc_lang="eng", words=3000, domain="agriculture_crops", rag=True):
        self.custom_id, self.doc_lang, self.words = cid, doc_lang, words
        self.domain, self.rag, self.lang, self.tokens = domain, rag, "pcm", 4000


class _Store:
    def __init__(self):
        self.docs = {}

    def __contains__(self, k):
        return k in self.docs

    def put(self, d):
        self.docs[d["doc_id"]] = d


def test_fill_store_sources_english_specs(tmp_path):
    pool = DocumentPool(tmp_path / "pool.jsonl")
    pool.extend([_doc(1, 3000)])
    store = _Store()
    assert fill_store([_Spec("doc__a")], store, pool) == 1
    d = store.docs["doc__a"]
    assert d["source"] == "wikipedia" and d["doc_lang"] == "eng"
    assert d["licence"] == "CC BY-SA 4.0", "attribution metadata must travel with the text"
    assert d["source_url"].startswith("https://"), "a published corpus has to be able to attribute"
    assert d["rag"] is True and d["form"] == "encyclopaedia_article"


def test_fill_store_leaves_native_language_specs_alone(tmp_path):
    """The low-resource Wikipedias are stubs -- Yoruba's median article is 9 words -- so a native-language
    document still has to be generated. Sourcing one would silently swap the language."""
    pool = DocumentPool(tmp_path / "pool.jsonl")
    pool.extend([_doc(1, 3000)])
    store = _Store()
    assert fill_store([_Spec("doc__y", doc_lang="yor")], store, pool) == 0
    assert "doc__y" not in store.docs


def test_fill_store_does_not_overwrite_what_is_already_there(tmp_path):
    pool = DocumentPool(tmp_path / "pool.jsonl")
    pool.extend([_doc(1, 3000)])
    store = _Store()
    store.put({"doc_id": "doc__a", "text": "already generated"})
    assert fill_store([_Spec("doc__a")], store, pool) == 0
    assert store.docs["doc__a"]["text"] == "already generated"


def test_fill_store_stops_cleanly_when_the_pool_runs_dry(tmp_path):
    pool = DocumentPool(tmp_path / "pool.jsonl")
    pool.extend([_doc(1, 3000)])
    store = _Store()
    filled = fill_store([_Spec(f"doc__{i}") for i in range(4)], store, pool)
    assert filled == 1, "the rest must fall through to the generator, not fail"


def test_topic_seeds_cover_the_corpus_domains():
    """The pool is drawn across the corpus's OWN domains, so a retrieved document is about something a
    West African user would plausibly ask about rather than 'technical' in the abstract."""
    from prompts import DOMAINS
    unknown = sorted(set(TOPIC_SEEDS) - set(DOMAINS) - {"nutrition"})
    assert not unknown, f"TOPIC_SEEDS names domains the corpus does not have: {unknown}"
    assert len(TOPIC_SEEDS) >= 10
    assert all(v for v in TOPIC_SEEDS.values())


# --------------------------------------------------------------------------- relevance
# Technicality alone let "Toyota Prius (XW20)" (0.95), "BMW 7 Series (G70)" (0.93) and "Value-added tax in
# the United Kingdom" into the first harvest. All technical; all useless. Relevance is the second gate.

def test_a_foreign_jurisdiction_in_the_title_is_rejected_outright():
    """Not penalised -- rejected. "Canadian passport" is a perfect match for a query about identity
    enrolment and still the wrong answer: a document question answered from another country's procedure is
    worse than one answered with "I could not find it". A penalty left it above the gate at 0.50."""
    from doc_sources import MIN_RELEVANCE, relevance
    body = "passport enrolment identification national requirement procedure " * 30
    assert relevance(body, "Canadian passport", "national identification enrolment") == 0.0
    assert relevance(body, "Value-added tax in the United Kingdom", "tax rate threshold") == 0.0
    assert relevance(body, "Malaysian identity card", "national identification enrolment") == 0.0
    # and the same subject without a jurisdiction is kept
    civil = "civil birth registration procedure vital records certificate registrar " * 30
    assert relevance(civil, "Civil registration", "birth registration procedure") >= MIN_RELEVANCE


def test_a_product_or_model_in_the_title_is_rejected_outright():
    from doc_sources import relevance
    body = "hybrid battery charging profile kwh nickel metal hydride " * 30
    assert relevance(body, "Toyota Prius (XW20)", "battery charging profile") == 0.0
    assert relevance(body, "BMW 7 Series (G70)", "battery charging profile") == 0.0


def test_on_topic_local_documents_score_well():
    from doc_sources import MIN_RELEVANCE, relevance
    body = ("drip irrigation emitter discharge rate smallholder plots in Nigeria and Ghana, "
            "rated 2 l/h at 100 kPa. " * 25)
    assert relevance(body, "Drip irrigation", "drip irrigation emitter discharge rate") >= MIN_RELEVANCE


def test_a_passing_mention_is_not_relevance():
    """The lead section defines the subject; a query term appearing once near the end does not."""
    from doc_sources import MIN_RELEVANCE, relevance
    body = ("A long article about something else entirely. " * 200) + " tilapia stocking density pond"
    assert relevance(body, "Something Else", "tilapia stocking density pond") < MIN_RELEVANCE


def test_relevance_needs_a_query():
    from doc_sources import relevance
    assert relevance("anything at all", "Title", "") == 0.0


# --------------------------------------------------------------------------- network

@pytest.mark.skipif(not NET, reason="set SABIYARN_NET=1 to test against Wikipedia")
def test_live_search_and_fetch():
    from doc_sources import Wikipedia
    w = Wikipedia()
    hits = w.search("drip irrigation emitter discharge rate", limit=5)
    assert hits and all("wordcount" in h for h in hits), "wordcount is what makes selection cheap"
    pg = w.plaintext("Drip irrigation")
    assert pg and len(pg["text"].split()) > 1000
    assert technicality(clean(pg["text"]), pg["title"]) >= MIN_TECHNICALITY
