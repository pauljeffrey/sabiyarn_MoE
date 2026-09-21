"""Taxonomy integrity: no empty pools, every language has locales, rules are satisfiable."""

from __future__ import annotations

import pytest

from config.corpus_config import TARGET_LANGUAGE_CODES
from sampling import taxonomy as t
from sampling.locales import COUNTRIES, LOCALES, NAME_POOLS


def test_domain_count_and_subtopics():
    assert len(t.DOMAINS) >= 40
    for d in t.DOMAINS.values():
        assert len(d.subtopics) >= 8, d.key
        assert all(s.strip() for s in d.subtopics), d.key
        assert len(set(d.subtopics)) == len(d.subtopics), f"duplicate sub-topic in {d.key}"
        assert d.description.strip() and d.name.strip() and d.tags
    assert len(t.all_pairs()) >= 450
    assert len(set(t.all_pairs())) == len(t.all_pairs())


def test_domain_groups_and_tags_are_known():
    known_tags = set("technical commerce academic culture personal civic risk numeric historical timeless faith rural modern_only".split())
    for d in t.DOMAINS.values():
        assert d.tags <= known_tags, (d.key, d.tags - known_tags)
    assert len(t.DOMAIN_GROUPS) >= 5


def test_all_target_languages_have_locales_and_names():
    assert "eng" not in LOCALES and "eng" not in NAME_POOLS
    assert set(LOCALES) == set(TARGET_LANGUAGE_CODES)
    assert set(NAME_POOLS) == set(TARGET_LANGUAGE_CODES)
    for lang in TARGET_LANGUAGE_CODES:
        labels = [loc.label for loc in LOCALES[lang]]
        assert len(labels) >= 10, lang
        assert len(set(labels)) == len(labels), f"duplicate locale label in {lang}"
        assert all(loc.country in COUNTRIES for loc in LOCALES[lang])
        assert len(NAME_POOLS[lang]) >= 8


def test_locale_grounding_examples():
    def has(lang: str, needle: str) -> bool:
        return any(needle in loc.label for loc in LOCALES[lang])

    assert has("yor", "Ibadan") and has("yor", "Ogun") and has("yor", "Osun") and has("yor", "Benin")
    assert has("hau", "Kano") and has("hau", "Kaduna") and has("hau", "Sokoto") and has("hau", "Niger")
    assert has("twi", "Kumasi") and has("twi", "Accra") and has("twi", "Ashanti") and has("twi", "Akuapem")
    assert has("ewe", "Volta") and has("ewe", "Togo")
    assert has("fon", "Cotonou") and has("fon", "Benin")
    assert has("efi", "Calabar") and has("efi", "Cross River")
    assert has("urh", "Delta") and has("urh", "Warri") and has("urh", "Effurun")
    assert has("pcm", "Nigeria")
    assert has("ful", "Senegal") and has("fuv", "Adamawa") and has("fuv", "Cameroon") and has("fuv", "Sokoto")


@pytest.mark.parametrize("vocab", ["GENRES", "REGISTERS", "AUDIENCES", "LENGTH_BUCKETS", "PERSPECTIVES", "ERAS", "DIFFICULTIES",
                                   "RESPONSE_LENGTHS", "INSTRUCTION_STYLES", "USER_REGISTERS"])
def test_option_vocabularies_are_non_empty(vocab):
    options = getattr(t, vocab)
    assert len(options) >= 3
    for o in options.values():
        assert o.text.strip() and o.weight > 0


def test_genre_count_and_fields():
    assert len(t.GENRES) >= 20
    for g in t.GENRES.values():
        assert g.registers and g.perspectives, g.key
        assert set(g.registers) <= set(t.REGISTERS)
        assert set(g.perspectives) <= set(t.PERSPECTIVES)
        if g.lengths:
            assert set(g.lengths) <= set(t.LENGTH_BUCKETS)
        if g.eras:
            assert set(g.eras) <= set(t.ERAS)


def test_compatibility_rules_are_satisfiable_for_every_domain():
    """Every domain must admit at least one genre, and every genre at least one domain."""
    for d in t.DOMAINS.values():
        assert any(g.domain_ok(d.tags) for g in t.GENRES.values()), d.key
    for g in t.GENRES.values():
        assert any(g.domain_ok(d.tags) for d in t.DOMAINS.values()), g.key


def test_incompatible_combinations_are_rejected():
    def compat(domain: str, **chosen) -> dict:
        return {"domain": domain, "subtopic": "x", **chosen}

    assert not t.pretrain_compat(compat("history"), "genre", "product_description")
    assert not t.pretrain_compat(compat("mathematics"), "genre", "news_report")  # timeless
    assert not t.pretrain_compat(compat("agriculture_crops"), "genre", "sermon_devotional")
    assert t.pretrain_compat(compat("religion_ethics"), "genre", "sermon_devotional")
    assert t.pretrain_compat(compat("trade_markets"), "genre", "product_description")
    assert not t.pretrain_compat(compat("technology_digital"), "era", "traditional_heritage")
    assert not t.pretrain_compat(compat("health_medicine", register="formal"), "audience", "children")
    assert not t.pretrain_compat(compat("trade_markets", genre="poem_song_lyrics"), "length_bucket", "long")
    assert not t.pretrain_compat(compat("trade_markets", genre="dialogue"), "perspective", "first_person")
