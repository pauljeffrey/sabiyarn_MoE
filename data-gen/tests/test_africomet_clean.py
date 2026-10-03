"""Offline tests for the AfriCOMET window cleaning rules (cases taken from real rows seen in the raw extraction)."""

from __future__ import annotations

import pytest

from africomet_translation.clean import clean_sentence, english_share, reject_reason


@pytest.mark.parametrize("raw,lang,expected", [
    ("12:20 Sugbon dipo, nibẹ ni o wa ọpọlọpọ awọn ẹya.", "yor", "Sugbon dipo, nibẹ ni o wa ọpọlọpọ awọn ẹya."),
    ("7.1 Awọn kemikali atunṣe gbọdọ ṣee lo.", "yor", "Awọn kemikali atunṣe gbọdọ ṣee lo."),
    ("14 Josef wee ziga kpọọ Jekọb nna ya.", "ibo", "Josef wee ziga kpọọ Jekọb nna ya."),
    ("Ati awọn èéfín ti won torment ascends lai (Ìṣí 14: 9-11 NKJ).", "yor", "Ati awọn èéfín ti won torment ascends lai."),
    ("Ẹ kú àárọ̀ , báwo ni gbogbo ẹ̀yin ṣe wà lónìí ?", "yor", "Ẹ kú àárọ̀, báwo ni gbogbo ẹ̀yin ṣe wà lónìí?"),
    ("7 Site n’ịghara iso mba ndị a,+ bụ́ ndị a fọdụrụ.", "ibo", "Site n’ịghara iso mba ndị a, bụ́ ndị a fọdụrụ."),
    ("6:2, 3) Nanso ne papa mu paa.—Monkenkan Ɔsɛnkafo 5:2.", "twi", "Nanso ne papa mu paa."),
    ("• Enye ama otono ntak edika . . . , oro edi", "efi", "Enye ama otono ntak edika..., oro edi"),
])
def test_clean_sentence_fixes_real_artifacts(raw, lang, expected):
    assert clean_sentence(raw, lang) == expected


def test_legitimate_leading_numbers_survive():
    assert clean_sentence("24 wakati ni ọjọ kan jẹ pupọ.", "yor") == "24 wakati ni ọjọ kan jẹ pupọ."     # lowercase follows
    assert clean_sentence("10 people were hurt in the blast.", "eng") == "10 people were hurt in the blast."


def test_rules_reject_the_junk_seen_in_raw_web_data():
    assert reject_reason("Ã©tait un jour â€™ hello world texte long ici pour tester", "hau") == "mojibake"
    assert reject_reason("The new software is available for all of your devices and will be updated from the store.", "ibo") == "english_leak"
    assert reject_reason("Ni Awọn Kẹhin Blue Print Awọn Ere Awọn Ti Ni Idagbasoke Orisirisi Creative Ere Fun Awọn", "yor", "web") == "title_case"
    assert reject_reason("Le ema megbe la, wode asi agbal8 te be Liberia kple h7 na3li ho kple Ghana ke wo5e ho3i3li la me.", "ewe", "encyclopedia") == "mixed_digits"
    assert reject_reason("buy now buy now buy now buy now buy now buy now buy now", "eng") == "repetition"


def test_rules_keep_good_text_including_tone_marked_languages():
    assert reject_reason("Awọn gomina yii ti waa rọ ọga ọlọpaa ko gbiyanju lati ṣiṣẹ papọ pẹlu awọn ẹgbẹ ajafẹtọọ.", "yor") is None
    assert reject_reason("Jezu ɖè acɛ kpodo wuvɛ̌sexámɛ tɔn kpo xlɛ́ ɖò ninɔmɛ wusyɛn dó lanmɛ nú mɛ tɔn enɛ mɛ.", "fon") is None
    assert reject_reason("Kallon shi nayi kadan sannan cikin sanyin murya nace.", "hau") is None
    assert reject_reason("Grevillea baueri is a species of flowering plant in the family Proteaceae and is endemic to Australia.", "eng") is None


def test_title_case_rule_only_applies_to_web_text():
    heading = "Ìlú Èkó Ati Ipinle Ogun Ati Ibadan Ni Orile Ede Naijiria Ati Ghana"
    assert reject_reason(heading, "yor", "web") == "title_case"
    assert reject_reason(heading, "yor", "encyclopedia") is None


def test_english_share_counts_only_lexicon_words():
    lex = frozenset({"software", "online"})
    assert english_share("ere online ati software tuntun fun awọn ọmọ", lex) == pytest.approx(2 / 8)
    assert english_share("", lex) == 0
