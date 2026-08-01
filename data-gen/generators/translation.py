"""Translation task generator.

The balanced `language` loop variable is the language the user's own
instruction is written in; the source/target languages being translated
between are sampled per-example (mostly involving the instruction
language on one end, since that's the realistic case, but sometimes a
third-party pair to cover local<->local translation broadly). We decide
the language pair ourselves in Python rather than asking the model to
choose, so every (task, language) cell gets predictable, even coverage of
pairs instead of the model gravitating toward English<->X only.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from config.languages import LANGUAGE_CODES, LANGUAGES, get_language
from config.settings import EXAMPLES_PER_LANGUAGE_PER_TASK
from generators.base import (
    GENERAL_DATA_QUALITY_RULES,
    BatchRequestSpec,
    make_custom_id,
    rng_for,
)
from generators.personas import sample_persona
from schemas.strict import to_strict_json_schema

TASK_NAME = "translation"

FINAL_SYSTEM_PROMPT = """\
You are a helpful multilingual assistant for West African language speakers. \
When asked to translate text, respond in whatever language the user asks \
in, giving an accurate, natural-sounding translation (not a stiff literal \
one) in the requested target language."""

TEXT_TOPIC_BANK = [
    "a short public health tip",
    "a text message about a family plan or errand",
    "a short market/business announcement",
    "a line from a school notice",
    "a short proverb-like piece of everyday advice",
    "a short weather/farming-related remark",
    "a line of everyday small talk",
    "a short instruction or safety notice",
    "a line about a upcoming community/church/mosque event",
    "a short customer service style message",
]


class TranslationSpec(BaseModel):
    source_text: str = Field(description="A short, natural piece of text (1-3 sentences) in the source language, about the given topic.")
    user_instruction: str = Field(description="The user's full message in the instruction language: naturally includes/quotes the source_text and asks for it to be translated into the target language. Vary phrasing -- don't always use the same template like 'translate X to Y'.")
    translated_text: str = Field(description="An accurate, natural (not word-for-word literal) translation of source_text into the target language.")
    has_followup: bool = Field(description="Whether there's a natural follow-up turn (e.g. asking to simplify the translation, make it more formal/informal, or translate one more short phrase).")
    followup_user_instruction: Optional[str] = Field(description="The follow-up request, in the instruction language, or null if has_followup is false.")
    followup_response: Optional[str] = Field(description="The assistant's response to the follow-up, or null if has_followup is false.")


def _sample_pair(rng, instruction_lang: str) -> tuple[str, str]:
    """Return (source_lang_code, target_lang_code)."""
    other_codes = [c for c in LANGUAGE_CODES if c != instruction_lang]
    roll = rng.random()
    if instruction_lang == "eng":
        # For an English instruction, pick some other local language as the
        # translation partner (translating either direction).
        other = other_codes[rng.randrange(len(other_codes))]
        return ("eng", other) if rng.random() < 0.5 else (other, "eng")
    if roll < 0.45:
        return "eng", instruction_lang  # translate English -> their language
    if roll < 0.75:
        return instruction_lang, "eng"  # translate their language -> English
    if roll < 0.90:
        other = [c for c in other_codes if c != "eng"]
        source = other[rng.randrange(len(other))] if other else "eng"
        return source, instruction_lang  # some other local language -> their language
    # third-party pair, unrelated to the instruction language
    a, b = rng.sample(other_codes, 2) if len(other_codes) >= 2 else (other_codes[0], "eng")
    return a, b


def _build_user_prompt(instruction_lang: str, persona: str, source_lang: str, target_lang: str, topic: str) -> str:
    inst = get_language(instruction_lang)
    src = get_language(source_lang)
    tgt = get_language(target_lang)
    return f"""\
Instruction language (write user_instruction, and the two followup fields \
if used, in this language): {inst.english_name} ({inst.endonym}), code \
'{inst.code}'.
{f"Style guidance for {inst.english_name}: {inst.style_notes}" if inst.style_notes else ""}

{persona}

Source text language: {src.english_name} (code '{src.code}'). \
{f"Style guidance: {src.style_notes}" if src.style_notes else ""}
Target translation language: {tgt.english_name} (code '{tgt.code}'). \
{f"Style guidance: {tgt.style_notes}" if tgt.style_notes else ""}

Write {"a" if topic[0] not in "aeiou" else "an"} {topic} as the source_text \
(in the source text language above), a natural user_instruction (in the \
instruction language above) that includes/quotes this text and asks for it \
to be translated into the target language, and the translated_text."""


def build_requests() -> list[BatchRequestSpec]:
    response_format = to_strict_json_schema(TranslationSpec, name="translation")
    specs: list[BatchRequestSpec] = []

    for lang in LANGUAGES:
        for i in range(EXAMPLES_PER_LANGUAGE_PER_TASK):
            rng = rng_for(TASK_NAME, lang.code, i)
            persona = sample_persona(rng)
            topic = rng.choice(TEXT_TOPIC_BANK)
            source_lang, target_lang = _sample_pair(rng, lang.code)

            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(TASK_NAME, lang.code, i),
                    task=TASK_NAME,
                    language=lang.code,
                    system_prompt=GENERAL_DATA_QUALITY_RULES,
                    user_prompt=_build_user_prompt(lang.code, persona, source_lang, target_lang, topic),
                    response_format=response_format,
                    context={
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "final_system_prompt": FINAL_SYSTEM_PROMPT,
                    },
                )
            )
    return specs
