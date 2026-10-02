"""Offline tests for the AfriCOMET translation-prompt builder (no network, no tokenizer download)."""

from __future__ import annotations

import random
import re
from collections import Counter

import pytest

from africomet_translation import instructions as ins
from africomet_translation.build import Window, make_records, rng_for
from africomet_translation.sources import AFRICOMET_SUPPORTED, LANGUAGE_ORDER, SOURCES
from africomet_translation.text_utils import (
    allocate, balance_by_k, dedupe_key, good_sentence, interleave_windows, normalize, partition_windows, split_sentences,
)


# ---------------------------------------------------------------- sentence splitting / filters

def test_splitter_keeps_abbreviations_and_initials_together():
    text = "Dr. Ehud Ur spoke at the U.S. embassy. J. R. Smith agreed. It rained!"
    assert split_sentences(text) == ["Dr. Ehud Ur spoke at the U.S. embassy.", "J. R. Smith agreed.", "It rained!"]


def test_splitter_handles_quotes_and_non_english_text():
    assert split_sentences('He said "stop." Then he left.') == ['He said "stop."', "Then he left."]
    hau = "Wannan rubutu ne. Na biyu kuma ya biyo baya."
    assert len(split_sentences(hau)) == 2


def test_normalize_collapses_whitespace_and_strips_controls():
    assert normalize("a  \n b​�  c") == "a b c"


def test_good_sentence_rejects_junk():
    assert good_sentence("This is a perfectly ordinary English sentence about rivers.", require_terminal=True)
    assert not good_sentence("See http://example.com for details about the thing.")
    assert not good_sentence("CHAPTER ONE THE BEGINNING OF EVERYTHING")
    assert not good_sentence("12 34 56 78 90 12 34 56 78 90 abc")
    assert not good_sentence("Two words")
    assert not good_sentence("A heading with no period at the end", require_terminal=True)
    assert good_sentence("A curated sentence with no final period", require_terminal=False)


# ---------------------------------------------------------------- windows

def test_partition_windows_cover_the_doc_exactly_once_with_sizes_1_to_5():
    sents = [f"S{i}." for i in range(103)]
    wins = partition_windows(sents, random.Random(0))
    assert [s for w in wins for s in w] == sents
    assert all(1 <= len(w) <= 5 for w in wins)


def test_partition_window_sizes_are_roughly_uniform_on_long_docs():
    sizes = Counter()
    rng = random.Random(1)
    for _ in range(300):
        for w in partition_windows([f"x{i}" for i in range(60)], rng)[:-1]:   # last window is the remainder
            sizes[len(w)] += 1
    total = sum(sizes.values())
    assert all(0.15 < sizes[k] / total < 0.25 for k in range(1, 6)), sizes


def test_interleave_spreads_over_documents_before_repeating_one():
    per_doc = [[["a1"], ["a2"], ["a3"]], [["b1"]], [["c1"], ["c2"]]]
    out = [w[0] for w in interleave_windows(per_doc, random.Random(0))]
    assert sorted(out) == ["a1", "a2", "a3", "b1", "c1", "c2"]
    first_three = {x[0] for x in out[:3]}
    assert first_three == {"a", "b", "c"}


def test_allocate_redistributes_from_capped_sources():
    got = allocate(100, {"a": 1, "b": 1, "c": 1}, {"a": 10, "b": 1000, "c": 1000})
    assert got["a"] == 10 and got["b"] + got["c"] == 90 and abs(got["b"] - got["c"]) <= 1
    assert sum(allocate(100, {"a": 1, "b": 1}, {"a": 5, "b": 7}).values()) == 12     # everything capped -> short


def test_dedupe_key_ignores_case_punctuation_and_spacing():
    assert dedupe_key("Hello,  World!") == dedupe_key("hello world")


# ---------------------------------------------------------------- instructions

def test_only_languages_with_authored_templates_are_instruction_languages():
    langs = ins.instruction_languages()
    assert {"eng", "pcm", "yor", "hau", "ibo"} <= set(langs)
    assert not {"fon", "efi", "ibb"} & set(langs)         # deliberately not authored
    assert set(langs) <= set(ins.CONFIDENCE)


@pytest.mark.parametrize("lang", sorted(ins.TEMPLATES))
def test_every_template_renders_without_leftover_placeholders(lang):
    rng = random.Random(3)
    for bank in ins.TEMPLATES[lang].values():
        for t in bank:
            assert "{target}" in t, f"{lang}: template must name the target language: {t}"
            out = t.format(target="Hausa", source="Yoruba")
            assert "{" not in out and "}" not in out


def test_position_selects_the_right_bank():
    rng = random.Random(0)
    for _ in range(200):
        pre = ins.make_instruction("eng", "yor", "hau", rng, position="prepend")
        post = ins.make_instruction("eng", "yor", "hau", rng, position="append")
        assert pre.position == "prepend" and post.position == "append"
        assert not re.search(r"\b(above|preceding)\b", pre.text.lower()), pre.text
        assert not re.search(r"\b(below|following|that follows)\b", post.text.lower()), post.text


def test_instructions_are_structurally_diverse():
    rng = random.Random(5)
    texts = [ins.make_instruction("eng", "yor", "hau", rng).text for _ in range(600)]
    assert len(set(texts)) > 120
    lengths = [len(t.split()) for t in texts]
    assert min(lengths) <= 3 and max(lengths) >= 25


def test_target_is_named_in_the_instruction():
    rng = random.Random(9)
    for lang in ins.instruction_languages():
        for _ in range(30):
            i = ins.make_instruction(lang, "eng", "ewe", rng)
            assert any(n in i.text for n in ins.DISPLAY_NAMES["ewe"]), i.text


def test_compose_prompt_orders_text_and_instruction_by_position():
    rng = random.Random(0)
    pre = ins.Instruction("Translate.", "prepend", "eng", "high")
    post = ins.Instruction("Translate.", "append", "eng", "high")
    assert ins.compose_prompt(pre, "TEXT", rng).startswith("Translate.")
    assert ins.compose_prompt(post, "TEXT", rng).endswith("Translate.")
    assert "TEXT" in ins.compose_prompt(pre, "TEXT", rng)


# ---------------------------------------------------------------- records

def _windows(n):
    return [Window(f"Sentence number {i}. Another one {i}.", 2, 14, "src", "CC-BY-4.0") for i in range(n)]


def test_records_never_target_the_source_language_and_cover_all_targets_evenly():
    recs = make_records("yor", _windows(900), seed=1, tag="t")
    assert all(r["target_language"] != "yor" for r in recs)
    c = Counter(r["target_language"] for r in recs)
    assert set(c) == set(LANGUAGE_ORDER) - {"yor"}
    assert max(c.values()) - min(c.values()) <= 1


def test_records_exclude_akan_and_urhobo_everywhere():
    recs = make_records("eng", _windows(200), seed=1, tag="t")
    assert not {"aka", "urh"} & ({r["target_language"] for r in recs} | set(LANGUAGE_ORDER) | set(SOURCES))


def test_ibibio_is_flagged_unsupported_by_africomet():
    recs = make_records("eng", _windows(300), seed=1, tag="t")
    for r in recs:
        assert r["africomet_supported"] == (r["target_language"] != "ibb")
    assert "ibb" not in AFRICOMET_SUPPORTED


def test_records_are_deterministic_and_have_the_requested_fields():
    a = make_records("hau", _windows(50), seed=7, tag="t")
    b = make_records("hau", _windows(50), seed=7, tag="t")
    assert a == b
    for k in ("instruction", "input", "source_language", "target_language"):
        assert a[0][k]
    assert a[0]["input"] in a[0]["prompt"] and a[0]["instruction"] in a[0]["prompt"]
    assert len({r["id"] for r in a}) == 50


def test_instruction_language_rotation_uses_every_authored_language():
    recs = make_records("yor", _windows(700), seed=2, tag="t")
    assert {r["instruction_language"] for r in recs} == set(ins.instruction_languages())


def test_registry_covers_exactly_the_ten_languages():
    assert set(SOURCES) == set(LANGUAGE_ORDER) == {"eng", "pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "ibb"}
    names = [s.name for srcs in SOURCES.values() for s in srcs]
    assert "howard-nlp/ibom-mt:eng-efi" in names and "howard-nlp/sib200-ibom:ibb_Latn" in names
    assert any("efik_speech_dataset" in n for n in names) and any("efik_audio_dataset" in n for n in names)
    assert any(s.name.startswith("wikimedia/wikipedia") for s in SOURCES["eng"])


def test_rng_for_is_stable():
    assert rng_for(1, "a").random() == rng_for(1, "a").random()
    assert rng_for(1, "a").random() != rng_for(1, "b").random()


def test_balance_by_k_is_as_uniform_as_the_pool_allows():
    pool = [Window(f"t{i}", k, 5, "s", "l") for i, k in enumerate([1] * 400 + [2] * 300 + [3] * 200 + [4] * 20 + [5] * 5)]
    got = Counter(w.n_sentences for w in balance_by_k(pool, 300))
    assert got[5] == 5 and got[4] == 20                    # scarce sizes are used up entirely
    assert abs(got[1] - got[2]) <= 1 and abs(got[2] - got[3]) <= 1
    assert sum(got.values()) == 300
    assert sum(Counter(w.n_sentences for w in balance_by_k(pool, 10_000)).values()) == len(pool)
