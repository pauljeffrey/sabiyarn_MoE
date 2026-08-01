"""Language roster for synthetic SFT data generation.

Codes match the special-token tags already present in the SabiYarn-32k
tokenizer (see training/constant_tokens.py `action_tokens` in the parent
repo) so downstream language-tagging stays consistent with the model's
existing vocabulary. This list intentionally has 13 entries, not 15 -- the
project currently targets these languages; add more by appending a
`Language(...)` entry below, nothing else needs to change.

`resource_tier` is a rough signal for how reliable a strong LLM (e.g.
GPT-4o) is likely to be at *generating fluent text* in that language, not a
statement about speaker population. Use it to:
  - weight extra human/LLM-judge spot-checking (quality/validators.py)
  - decide whether to route generation through a higher-effort model
It is NOT used to skew how many examples each language gets -- generation
targets are equal-per-language by default (see settings.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResourceTier(str, Enum):
    HIGH = "high"  # model is fluent, plenty of training data existed
    MEDIUM = "medium"  # generally coherent, occasional errors
    LOW = "low"  # frequent errors, code-switching, or invented words likely


@dataclass(frozen=True)
class Language:
    code: str  # matches the tokenizer's language tag, e.g. "yor" -> <yor>
    english_name: str
    endonym: str  # name of the language in itself
    resource_tier: ResourceTier
    notes: str = ""
    # Extra generation guidance injected into meta-prompts for this language
    # specifically (orthography quirks, diacritics, common pitfalls a
    # non-fluent model tends to get wrong).
    style_notes: str = ""
    aliases: list[str] = field(default_factory=list)


LANGUAGES: list[Language] = [
    Language(
        code="eng",
        english_name="English",
        endonym="English",
        resource_tier=ResourceTier.HIGH,
        notes="Nigerian/West African English register preferred over American/British.",
        style_notes=(
            "Use West African English idiom and register where natural "
            "(not American or British English). Avoid sounding like a "
            "generic international assistant."
        ),
    ),
    Language(
        code="yor",
        english_name="Yoruba",
        endonym="Yorùbá",
        resource_tier=ResourceTier.MEDIUM,
        style_notes=(
            "Use correct tonal diacritics (Ìwé kíkọ) consistently -- underdot "
            "(ẹ, ọ, ṣ) and tone marks (à, á, è, é, etc.). Do not silently drop "
            "diacritics even though many casual Yoruba texts online do."
        ),
    ),
    Language(
        code="hau",
        english_name="Hausa",
        endonym="Hausa",
        resource_tier=ResourceTier.MEDIUM,
        style_notes=(
            "Use standard Boko (Latin) orthography with hooked letters where "
            "appropriate (ɓ, ɗ, ƙ, 'y). Prefer everyday spoken register over "
            "formal literary Hausa unless the persona calls for it."
        ),
    ),
    Language(
        code="ibo",
        english_name="Igbo",
        endonym="Igbo",
        resource_tier=ResourceTier.MEDIUM,
        style_notes=(
            "Use correct subdots (ị, ọ, ụ) and digraphs (gb, gh, kp, nw, ny, "
            "sh) consistently. Prefer Central/Standard Igbo unless a persona "
            "implies a dialect."
        ),
    ),
    Language(
        code="efi",
        english_name="Efik",
        endonym="Efịk",
        resource_tier=ResourceTier.LOW,
        notes="Very low-resource for LLMs; expect higher error rates -- flag for extra spot-check.",
        style_notes=(
            "Efik has limited digital text available to language models. "
            "Keep sentences short and grammatically simple to reduce the "
            "chance of invented or incorrect constructions. If truly unsure "
            "of a word, prefer a simpler paraphrase over a fabricated term."
        ),
    ),
    Language(
        code="urh",
        english_name="Urhobo",
        endonym="Urhobo",
        resource_tier=ResourceTier.LOW,
        notes="Very low-resource for LLMs; expect higher error rates -- flag for extra spot-check.",
        style_notes=(
            "Urhobo has very limited digital text available to language "
            "models. Keep sentences short and grammatically simple. If "
            "unsure of a word, prefer a simpler paraphrase over a "
            "fabricated term."
        ),
    ),
    Language(
        code="twi",
        english_name="Twi",
        endonym="Twi",
        resource_tier=ResourceTier.MEDIUM,
        notes="Akan dialect cluster alongside 'aka' below; kept distinct because the tokenizer has separate tags.",
        style_notes=(
            "Use Akuapem/Asante Twi orthography consistently within a single "
            "conversation (do not mix). Use nasalized vowels and digraphs "
            "correctly (ɔ, ɛ)."
        ),
    ),
    Language(
        code="fon",
        english_name="Fon",
        endonym="Fɔ̀ngbè",
        resource_tier=ResourceTier.LOW,
        notes="Very low-resource for LLMs; expect higher error rates -- flag for extra spot-check.",
        style_notes=(
            "Fon has limited digital text available to language models. "
            "Keep sentences short and grammatically simple. If unsure of a "
            "word, prefer a simpler paraphrase over a fabricated term."
        ),
    ),
    Language(
        code="pcm",
        english_name="Nigerian Pidgin",
        endonym="Naija",
        resource_tier=ResourceTier.HIGH,
        style_notes=(
            "Use natural West African Pidgin as actually spoken/written "
            "(e.g. on Naija social media), not a caricature or overly "
            "anglicized version. Avoid mixing in Ghanaian/Sierra Leonean "
            "Pidgin vocabulary."
        ),
    ),
    Language(
        code="ewe",
        english_name="Ewe",
        endonym="Èʋegbe",
        resource_tier=ResourceTier.LOW,
        notes="Very low-resource for LLMs; expect higher error rates -- flag for extra spot-check.",
        style_notes=(
            "Ewe has limited digital text available to language models. Use "
            "correct special characters (ɖ, ɣ, ʋ, ŋ, ɔ, ɛ) and keep "
            "sentences short and grammatically simple."
        ),
    ),
    Language(
        code="aka",
        english_name="Akan",
        endonym="Akan",
        resource_tier=ResourceTier.MEDIUM,
        notes="Broader Akan macrolanguage; keep distinct in style from 'twi' when both appear in a run.",
        style_notes=(
            "Use general/formal Akan rather than a specific dialect unless "
            "the persona implies one. Use nasalized vowels and digraphs "
            "correctly (ɔ, ɛ)."
        ),
    ),
    Language(
        code="ful",
        english_name="Fulah",
        endonym="Pulaar/Fulfulde",
        resource_tier=ResourceTier.LOW,
        notes="General/Pular Fula variant; distinct tokenizer tag from 'fuv' (Nigerian Fulfulde) below.",
        style_notes=(
            "Fulah has limited digital text available to language models. "
            "Keep sentences short and grammatically simple. If unsure of a "
            "word, prefer a simpler paraphrase over a fabricated term."
        ),
    ),
    Language(
        code="fuv",
        english_name="Fulfulde",
        endonym="Fulfulde",
        resource_tier=ResourceTier.LOW,
        notes="Nigerian/Adamawa Fulfulde variant; distinct tokenizer tag from 'ful' (general Fulah) above.",
        style_notes=(
            "Use the Nigerian/Adamawa Fulfulde variant specifically, not "
            "Pular or Futa Jallon Fula. Keep sentences short and "
            "grammatically simple; prefer a paraphrase over a fabricated "
            "term when unsure."
        ),
    ),
]

LANGUAGE_BY_CODE: dict[str, Language] = {lang.code: lang for lang in LANGUAGES}

LANGUAGE_CODES: list[str] = [lang.code for lang in LANGUAGES]


def get_language(code: str) -> Language:
    try:
        return LANGUAGE_BY_CODE[code]
    except KeyError as e:
        raise KeyError(
            f"Unknown language code {code!r}. Known codes: {LANGUAGE_CODES}"
        ) from e
