"""Unit tests for the local quality filters (hand-made bad inputs)."""

from __future__ import annotations

import random

import pytest

from config.corpus_config import load_config
from quality import corpus_filters as f
from quality.dedup import find_near_duplicates
from scripts.mock_generate import _sentences

Q = {k: load_config(k).quality for k in ("pretrain", "sft", "dpo")}


def good(language: str = "yor", words: int = 40, seed: int = 1) -> str:
    return _sentences(random.Random(seed), language, words)


ENGLISH = (
    "The farmers in the village are going to the market because they have been told that the price of maize will "
    "not be low this year and there is a lot of work to do before the rains come back to the fields"
)


def reasons_of(text: str, language: str = "yor", **kw):
    return f.check_text_field(text, language, "response", Q["sft"], **kw)


def test_tokenizer_keeps_combining_marks_inside_words():
    assert f.tokens("ẹ̀gbọ́n rẹ̀ wá") == ["ẹ̀gbọ́n", "rẹ̀", "wá"]


def test_good_text_passes():
    r, w = reasons_of(good())
    assert r == []


def test_repetition_is_flagged():
    r, _ = reasons_of("ka ba to ne " * 60)
    assert any(x.startswith("repetition_ngram") for x in r)
    para = "Ọkọ mi lọ sí ọjà lánàá ó sì ra ẹja púpọ̀.\n"
    r, _ = reasons_of(para * 6)
    assert any(x.startswith("repetition_lines") for x in r)
    m = f.repetition_metrics(good(words=200))
    assert m["repeated_ngram_ratio"] < 0.05 and m["repeated_line_ratio"] == 0


def test_english_leak_flagged_but_not_for_english_fields():
    r, _ = reasons_of(ENGLISH, "hau")
    assert "english_leak:response" in r
    r, _ = reasons_of(ENGLISH, "hau", is_english=True)
    assert "english_leak:response" not in r
    assert f.english_leak_ratio("ka ba", "hau") is None  # too short to judge


def test_pidgin_gets_higher_tolerance_but_english_is_still_caught():
    pidgin = "I no know wetin you dey talk about for this matter, na so e be for here, make we go see am for market tomorrow"
    ratio_broad = f.english_leak_ratio(pidgin, "yor")
    assert ratio_broad > Q["sft"]["max_english_ratio"]  # would be flagged for a non-Pidgin language
    r, _ = reasons_of(pidgin, "pcm")
    assert "english_leak:response" not in r  # ... but Pidgin legitimately shares this vocabulary
    r, _ = reasons_of(ENGLISH, "pcm")
    assert "english_leak:response" in r  # real English is still caught with the strict list


@pytest.mark.parametrize(
    "text, reason",
    [
        ("Here is the translation: " + good(), "meta_text"),
        ("Sure, " + good(), "meta_text"),
        (good() + " As an AI language model I cannot do this.", "meta_text"),
        (good() + " [...] " + good(seed=2), "placeholder"),
        (good() + " [insert name here] " + good(seed=2), "placeholder"),
        ("```json\n{}\n```", "markdown_fence"),
        ("I'm sorry, but I cannot help with that request. " + good(), "refusal"),
    ],
)
def test_meta_text_placeholders_fences_refusals(text, reason):
    r, _ = reasons_of(text)
    assert f"{reason}:response" in r


def test_refusal_allowed_only_when_requested():
    text = "I'm sorry, but I cannot help with that. " + good()
    assert "refusal:response" in reasons_of(text)[0]
    assert "refusal:response" not in reasons_of(text, allow_refusal=True)[0]


def test_wrong_script_and_replacement_char():
    r, _ = reasons_of("Привет мир это тест на неправильный алфавит " * 3)
    assert "wrong_script:response" in r
    r, _ = reasons_of("这是一个用错误语言写的句子，不应该出现在数据里")
    assert "wrong_script:response" in r
    r, _ = reasons_of(good() + " � broken")
    assert "replacement_char:response" in r
    # hooked letters, IPA and combining marks are Latin-like, not a script failure
    assert f.non_latin_ratio("ɓɗƙ ɔɛ ẹ̀ ọ́ ṣ ɖɣʋŋ") == 0


def test_missing_diacritics_is_a_soft_warning_only():
    r, w = f.check_text_field("ka ba to ne so wa mi ku de la pa ri " * 3, "yor", "response", Q["sft"])
    assert "no_distinctive_chars:response" in w
    assert not any("distinctive" in x for x in r)


# ------------------------------------------------------------------ pretrain


PRE_CTX = {"length_words": [250, 420]}


def pre_doc(**over):
    d = {"title": "Ọjà Ẹja", "text": good(words=320), "language_self_check": True}
    d.update(over)
    return d


def test_pretrain_validator():
    q = Q["pretrain"]
    assert f.validate_pretrain(pre_doc(), PRE_CTX, "yor", q)[0] == []
    assert "too_short:text" in f.validate_pretrain(pre_doc(text=good(words=30)), PRE_CTX, "yor", q)[0]
    assert "too_short:text" in f.validate_pretrain(pre_doc(text=good(words=100)), PRE_CTX, "yor", q)[0]  # < 50% of bucket lower bound
    assert "too_long:text" in f.validate_pretrain(pre_doc(text=good(words=1000)), PRE_CTX, "yor", q)[0]
    assert "self_check_failed" in f.validate_pretrain(pre_doc(language_self_check=False), PRE_CTX, "yor", q)[0]
    assert "markdown_heading:text" in f.validate_pretrain(pre_doc(text="# Akole\n" + good(words=320)), PRE_CTX, "yor", q)[0]
    assert "bad_title" in f.validate_pretrain(pre_doc(title=""), PRE_CTX, "yor", q)[0]
    assert "empty_field:text" in f.validate_pretrain(pre_doc(text=" "), PRE_CTX, "yor", q)[0]


# ------------------------------------------------------------------ sft


def sft_rec(**over):
    r = {"instruction": good(words=10) + "?", "input": "", "response": good(words=30, seed=5), "confidence": "high"}
    r.update(over)
    return r


SFT_CTX = {"input_mode": "empty", "english_fields": [], "attributes": {"task": "open_qa"}}


def test_sft_validator():
    q = Q["sft"]
    assert f.validate_sft(sft_rec(), SFT_CTX, "yor", q)[0] == []
    assert "low_confidence" in f.validate_sft(sft_rec(confidence="low"), SFT_CTX, "yor", q)[0]
    assert "empty_field:response" in f.validate_sft(sft_rec(response=""), SFT_CTX, "yor", q)[0]
    assert "empty_field:instruction" in f.validate_sft(sft_rec(instruction=" "), SFT_CTX, "yor", q)[0]
    assert "missing_input" in f.validate_sft(sft_rec(), {**SFT_CTX, "input_mode": "required"}, "yor", q)[0]
    assert "too_long:response" in f.validate_sft(sft_rec(response=good(words=2000)), SFT_CTX, "yor", q)[0]
    assert "english_leak:response" in f.validate_sft(sft_rec(response=ENGLISH), SFT_CTX, "hau", q)[0]


def test_sft_english_side_of_translation_is_tolerated():
    q = Q["sft"]
    ctx = {"input_mode": "required", "english_fields": ["input"], "attributes": {"task": "translation_en_to_lang"}}
    assert f.validate_sft(sft_rec(input=ENGLISH), ctx, "hau", q)[0] == []
    ctx = {"input_mode": "required", "english_fields": ["response"], "attributes": {"task": "translation_lang_to_en"}}
    assert f.validate_sft(sft_rec(input=good(), response=ENGLISH), ctx, "hau", q)[0] == []
    # ... but English in the *instruction* is never tolerated
    assert "english_leak:instruction" in f.validate_sft(sft_rec(instruction=ENGLISH, input=good(), response=ENGLISH), ctx, "hau", q)[0]


def test_safe_decline_may_refuse():
    ctx = {**SFT_CTX, "attributes": {"task": "safe_decline"}}
    rec = sft_rec(response="I'm sorry, but I cannot help with that. " + good(words=20))
    assert "refusal:response" not in f.validate_sft(rec, ctx, "yor", Q["sft"])[0]
    assert "refusal:response" in f.validate_sft(rec, SFT_CTX, "yor", Q["sft"])[0]


# ------------------------------------------------------------------ dpo


def dpo_rec(**over):
    r = {"instruction": good(words=10) + "?", "input": "", "chosen": good(words=30, seed=5), "rejected": good(words=30, seed=6),
         "rejection_type": "off_topic", "chosen_confidence": "high"}
    r.update(over)
    return r


def dpo_ctx(rtype="off_topic", **over):
    c = {"input_mode": "empty", "english_fields": [], "attributes": {"task": "open_qa"}, "rejection_type": rtype,
         "length_related": rtype in ("incomplete", "rambling_verbose", "ignores_constraint", "unhelpful_refusal")}
    c.update(over)
    return c


def test_dpo_validator():
    q = Q["dpo"]
    assert f.validate_dpo(dpo_rec(), dpo_ctx(), "yor", q)[0] == []
    same = good(words=30, seed=5)
    assert "chosen_eq_rejected" in f.validate_dpo(dpo_rec(chosen=same, rejected=same.upper()), dpo_ctx(), "yor", q)[0]
    assert "empty_field:rejected" in f.validate_dpo(dpo_rec(rejected=""), dpo_ctx(), "yor", q)[0]
    assert "empty_field:chosen" in f.validate_dpo(dpo_rec(chosen=" "), dpo_ctx(), "yor", q)[0]
    assert "low_confidence" in f.validate_dpo(dpo_rec(chosen_confidence="low"), dpo_ctx(), "yor", q)[0]
    assert "rejection_type_mismatch" in f.validate_dpo(dpo_rec(rejection_type="factual_error"), dpo_ctx(), "yor", q)[0]


def test_dpo_same_language_check_and_wrong_language_exemption():
    q = Q["dpo"]
    r = f.validate_dpo(dpo_rec(rejected=ENGLISH), dpo_ctx(), "hau", q)[0]
    assert "rejected_wrong_language" in r
    r = f.validate_dpo(dpo_rec(rejected=ENGLISH, rejection_type="wrong_language"), dpo_ctx("wrong_language"), "hau", q)[0]
    assert "rejected_wrong_language" not in r and "english_leak:rejected" not in r
    # chosen must always be in the target language, even for wrong_language pairs
    r = f.validate_dpo(dpo_rec(chosen=ENGLISH, rejection_type="wrong_language"), dpo_ctx("wrong_language"), "hau", q)[0]
    assert "english_leak:chosen" in r


def test_dpo_length_ratio_bounds_unless_length_related():
    q = Q["dpo"]
    short, long_ = good(words=3, seed=8), good(words=120, seed=9)
    assert "length_ratio" in f.validate_dpo(dpo_rec(chosen=long_, rejected=short), dpo_ctx(), "yor", q)[0]
    assert "length_ratio" in f.validate_dpo(dpo_rec(chosen=short, rejected=long_), dpo_ctx(), "yor", q)[0]
    for rtype in ("incomplete", "rambling_verbose"):
        r = f.validate_dpo(dpo_rec(chosen=long_, rejected=short, rejection_type=rtype), dpo_ctx(rtype), "yor", q)[0]
        assert "length_ratio" not in r


def test_dpo_refusal_in_rejected_is_expected_for_unhelpful_refusal():
    refusal = "I'm sorry, but I cannot help with that request. " + good(words=20)
    r = f.validate_dpo(dpo_rec(rejected=refusal, rejection_type="unhelpful_refusal"), dpo_ctx("unhelpful_refusal"), "yor", Q["dpo"])[0]
    assert "refusal:rejected" not in r
    r = f.validate_dpo(dpo_rec(rejected=refusal), dpo_ctx(), "yor", Q["dpo"])[0]
    assert "refusal:rejected" in r
    r = f.validate_dpo(dpo_rec(chosen=refusal), dpo_ctx(), "yor", Q["dpo"])[0]
    assert "refusal:chosen" in r


# ------------------------------------------------------------------ dedup


def test_near_duplicate_detection_scales_and_is_correct():
    base = good(words=60, seed=11)
    variant = base.replace(base.split()[3], "zzzz", 1)  # one word changed
    other = good(words=60, seed=12)
    recs = [{"t": base, "c": "a"}, {"t": other, "c": "a"}, {"t": variant, "c": "a"}, {"t": base, "c": "a"}, {"t": base, "c": "b"}]
    drop, reasons = find_near_duplicates(recs, cell_key_fn=lambda r: r["c"], text_fn=lambda r: r["t"], near_dup_threshold=0.8)
    assert drop == {2, 3}
    assert reasons[3].startswith("exact") and reasons[2].startswith("near-duplicate")
    drop_all, _ = find_near_duplicates(recs, cell_key_fn=lambda r: "all", text_fn=lambda r: r["t"], near_dup_threshold=0.8)
    assert drop_all == {2, 3, 4}


def test_dedup_handles_thousands_quickly():
    rng = random.Random(0)
    recs = [{"t": _sentences(rng, "yor", 30)} for _ in range(3000)]
    recs += [{"t": recs[i]["t"]} for i in range(0, 3000, 300)]  # 10 exact duplicates
    drop, _ = find_near_duplicates(recs, cell_key_fn=lambda r: "x", text_fn=lambda r: r["t"])
    assert len(drop) == 10


# --------------------------------------------------------------------------- RL shape recovery
# Three bugs found by the first live RL run, which kept 1 of 24. All three discarded sound data over a
# JSON shape rather than over quality, so each gets a regression test.

def test_flat_keyvalue_turns_are_recovered():
    """The generator sometimes emits a turn as ["role","user","content","..."] instead of an object. The
    content is intact and the conversion is exact, so it is recovered rather than dropped."""
    from postprocess_gen import _objectify_turns
    got = _objectify_turns([["role", "user", "content", "how far?"],
                            {"role": "assistant", "response": "I dey"}])
    assert got[0] == {"role": "user", "content": "how far?"}
    assert got[1] == {"role": "assistant", "response": "I dey"}


def test_odd_or_nonstring_flat_turns_are_left_for_the_assembler():
    """Only an even-length list with string keys converts unambiguously; anything else must not be guessed."""
    from postprocess_gen import _objectify_turns
    odd = [["role", "user", "content"]]
    assert _objectify_turns(odd) == odd
    bad = [[1, "user", "content", "x"]]
    assert _objectify_turns(bad) == bad
    assert _objectify_turns("not a list") == "not a list"


def test_rl_candidate_accepts_a_dict_of_fields():
    """`responses[].content` arrives as the SAME field dict the prompt turns use, because that is the
    contract the generator is taught. It used to crash with AttributeError on .strip() and lose the sample."""
    from postprocess_gen import _candidate_text
    md = {"lang": "pcm", "expect_markers": ["pcm", "pcm"], "tasks": ["world_knowledge_qa"]}
    out = _candidate_text({"think": "simple question", "task_plan": ["<|explain|>"],
                           "input_lang": "pcm", "target_lang": "pcm",
                           "response": "Generator dey turn fuel into electricity."}, md)
    assert out, "a well-formed field dict must assemble"
    assert "Generator dey turn fuel" in out


def test_rl_candidate_still_accepts_a_plain_string():
    from postprocess_gen import _candidate_text
    assert _candidate_text("  already assembled  ", {"lang": "pcm"}) == "already assembled"


def test_rl_candidate_rejects_what_cannot_assemble():
    """An unusable candidate must come back empty so the `< 2 responses` check drops the sample, rather
    than reaching the corpus as a half-built string."""
    from postprocess_gen import _candidate_text
    assert _candidate_text(None, {"lang": "pcm"}) == ""
    assert _candidate_text(12, {"lang": "pcm"}) == ""
    assert _candidate_text({}, {"lang": "pcm"}) == ""


def test_rl_output_ceiling_clears_a_self_written_source():
    """A summarization prefix contains a source the generator wrote itself, and at a 6,144 ceiling that
    truncated into invalid JSON. The ceiling must leave room for the prefix AND both candidates."""
    from budgets import budget_for
    b = budget_for("rl")
    assert b.max_output_tokens >= b.max_response_tokens * 2 + 4_096


def test_rl_candidate_recovers_a_stringified_field_dict():
    """The generator sometimes JSON-ENCODES the turn instead of nesting it. These passed every gate and put
    raw JSON into the corpus (26 of 33 kept samples on one live run), which would teach the model to emit
    JSON. Parsed back and assembled like any other field dict."""
    import json as _json
    from postprocess_gen import _candidate_text
    md = {"lang": "pcm", "expect_markers": ["pcm", "pcm"], "tasks": ["world_knowledge_qa"]}
    blob = _json.dumps({"role": "assistant", "think": "simple", "task_plan": ["<|explain|>"],
                        "input_lang": "pcm", "target_lang": "pcm", "response": "Na so e be."})
    out = _candidate_text(blob, md)
    assert "<|target_lang|>" in out, "a stringified dict must be assembled, not passed through as JSON"
    assert '"role"' not in out, "raw JSON must not survive into the candidate"
    assert "Na so e be." in out


def test_rl_candidate_leaves_ordinary_text_alone():
    """A plain-prose candidate is not JSON and must not be mangled by the recovery path."""
    from postprocess_gen import _candidate_text
    assert _candidate_text("{not json at all", {"lang": "pcm"}) == "{not json at all"


def test_a_tool_call_candidate_is_accepted():
    """A tool-call candidate ends at <task_plan> with no <|target_lang|> and no <response>: it is a call, not
    a reply. Gating on <|target_lang|> rejected these, which are a large share of the behaviour tasks."""
    from postprocess_gen import _candidate_text
    md = {"lang": "pcm", "expect_markers": ["pcm", "pcm"], "tasks": ["tool_search_answer"]}
    out = _candidate_text({"think": "need the weather", "task_plan": ["<|plan|>"], "input_lang": "eng",
                           "tool_call": {"name": "get_weather_forecast",
                                         "arguments": {"location": "Ibi", "days": 1}}}, md)
    assert out, "a tool-call candidate must assemble"
    assert "<|input_lang|>" in out, "and must carry the marker the candidate gate tests"


def test_quality_label_synonyms_are_accepted():
    """A generator that says chosen/rejected rather than best/worst has given the same information. Mapping
    synonyms is safe; inventing a label that was never given is not, so an unknown label still fails."""
    import postprocess_gen as P
    src = P.__file__
    import re
    body = open(src, encoding="utf-8").read()
    assert '"chosen": "best"' in body and '"rejected": "worst"' in body
    # and the mapping must not invent a label for something unlabelled
    assert '"": "best"' not in body


def test_a_misspelled_discriminator_quotation_still_matches():
    """The generator re-types its own citation and mangles the diacritics; the sample is still good.

    Measured case: a privacy_boundary pair where the best reply correctly declined to check a third party's
    balance and the worst invented one -- exactly the behaviour the task exists to teach -- was discarded
    because the discriminator quoted "oeayewo"/"celomiran" for text that read "sayewo"/"elomiran". _fold
    strips COMBINING marks, so it cannot join a ligature or a cedilla to the letter the writer meant.
    """
    from postprocess_gen import _phrase_present
    best = ("Sùgbọ́n nípa account ọmọ rẹ, mi ò le ṣàyẹ̀wò owó tí ó wà nínú account ẹlòmíràn pẹ̀lú BVN wọn. "
            "Ó yẹ kí ọmọ rẹ fúnra rẹ̀ lọ sí banki.")
    assert _phrase_present("mi ò le œàyèwò owó tí wà nínu account çlòmíràn", best)
    # an exact quotation still matches, and a phrase the generator never wrote still does not
    assert _phrase_present("mi ò le ṣàyẹ̀wò owó", best)
    assert not _phrase_present("LGA office", best)
    # the word floor keeps the loose path honest: a short phrase must match as a substring or not at all
    assert not _phrase_present("account ṣàyẹ̀wò", "owó account tí ṣàyẹ̀wò nínú")


def test_an_unverified_bad_phrase_costs_the_phrase_not_the_sample():
    """`good` holding is what proves the two replies differ in substance; `bad` is a bonus.

    On a 30-sample run, 2 of 3 discriminator drops had `good` holding perfectly -- the pair was sound and the
    generator had merely mis-transcribed its own criticism. The unverifiable half is now removed from the
    answer key and counted, rather than costing the sample.
    """
    import postprocess_gen as P
    disc = {"good": "mo ti ṣàyẹ̀wò", "bad": "a phrase that appears nowhere at all in either reply"}
    best, worst = "mo ti ṣàyẹ̀wò owó náà", "owó rẹ̀ jẹ́ ₦25,000"
    bad_sides = []
    if not (P._phrase_present(disc["good"], best) and not P._phrase_present(disc["good"], worst)):
        bad_sides.append("good")
    if not (P._phrase_present(disc["bad"], worst) and not P._phrase_present(disc["bad"], best)):
        disc.pop("bad", None)
    assert bad_sides == [], "the good side holds, so the sample survives"
    assert "bad" not in disc, "the unverified claim must not stay in the answer key"
    assert disc["good"]
