"""Pretraining-document generator (plain-text corpus documents).

One Batch API request per document. The attributes of every document
(domain, sub-topic, genre, register, audience, locale, length bucket,
perspective, era, reading level) come from the `CoverageSampler`, never from
the model, so coverage is balanced by construction and auditable from the
manifest afterwards.

Quality stance (see PRETRAIN_SYSTEM_PROMPT): pretraining text is only useful
if it is fluent, factually careful and free of English leakage; a weak
generator that "sounds fluent" in a low-resource language can do net harm.
So the prompt pushes hard for hedged general statements over invented
specifics, and for short grammatically safe sentences in lower-resource
languages, and the postprocess filters then re-check what they can locally.
"""

from __future__ import annotations

from typing import Iterator, Optional

from config.corpus_config import CorpusConfig
from generators.base import BatchRequestSpec, language_style_block
from generators.corpus_common import (
    attribute_spec,
    corpus_custom_id,
    domain_block,
    locale_spec,
    locale_text,
    resolve_languages,
    sample_names,
)
from sampling.sampler import CoverageSampler, pair_weights_from_groups
from sampling.taxonomy import (
    AUDIENCES,
    DIFFICULTIES,
    DOMAINS,
    ERAS,
    GENRES,
    LENGTH_BUCKETS,
    LENGTH_WORD_RANGES,
    PERSPECTIVES,
    REGISTERS,
    all_pairs,
    option_weights,
    pretrain_compat,
)
from schemas.corpus import PRETRAIN_FORMAT_NAME, PretrainDoc, strict_response_format

KIND = "pretrain"

PRETRAIN_SYSTEM_PROMPT = """\
You are a native-level writer and editor of West African languages who produces ORIGINAL documents for a language-model pretraining corpus. You write directly in the target language; you never translate from English.

Mandatory quality rules:
1. LANGUAGE: The entire title and text must be in the target language named in the request. It must sound like a fluent native speaker wrote it (natural word order, idioms, connectors), not like translated English. No English words except unavoidable proper nouns, brand names or standard units.
2. ORTHOGRAPHY: Use the language's correct alphabet, special letters and diacritics/tone marks exactly as described in the language notes, consistently. Do not drop diacritics to make typing easier.
3. FACTS: Be careful. Prefer widely known general knowledge and clearly hedged statements ("many farmers...", "usually...", "often..."). NEVER invent statistics, percentages, dates, laws, studies, quotations, or claims about real named people, companies or organisations. Invented (local) names are allowed only for clearly fictional characters in stories, dialogues, letters, case studies and similar fictional genres.
4. LOCAL GROUNDING: Use the requested setting: local names, places, foods, currency, units, customs and everyday realities. Avoid Western defaults (dollars, snow, supermarkets-only, "Mr. Smith").
5. STRUCTURE: Coherent beginning, development and ending. Every paragraph adds new information. Never repeat a paragraph, and do not start many sentences the same way.
6. FORM: Follow the genre's format hint. Plain text only: no markdown headings, no bold/asterisks, no code fences, no emojis, no bracketed placeholders such as [name] or [...].
7. NO META: Never mention that you are an AI, the prompt, the word count or the language. Never write "Here is ..." or add notes. The text must read like a real document.
8. LENGTH: Stay inside the requested word range (words in the target language).
9. HONEST SIMPLICITY: If you are unsure of a word or construction in this language, use a simpler common word or rephrase; never invent words. For lower-resource languages prefer short, grammatically safe sentences over ambitious ones.
10. Set language_self_check to false if you doubt that the text is natural, entirely in the target language, and free of invented facts.
Return only the JSON object required by the schema."""


def build_pretrain_sampler(cfg: CorpusConfig, language: str) -> CoverageSampler:
    """One sampler per (pretrain, language); identical procedure for every language."""
    pairs = all_pairs()
    attributes = [
        attribute_spec("genre", option_weights(GENRES), cfg),
        attribute_spec("register", option_weights(REGISTERS), cfg),
        attribute_spec("audience", option_weights(AUDIENCES), cfg),
        attribute_spec("length_bucket", option_weights(LENGTH_BUCKETS), cfg),
        attribute_spec("perspective", option_weights(PERSPECTIVES), cfg),
        attribute_spec("era", option_weights(ERAS), cfg),
        attribute_spec("difficulty", option_weights(DIFFICULTIES), cfg),
        locale_spec(language, cfg),
    ]
    group_of = {k: d.group for k, d in DOMAINS.items()}
    return CoverageSampler(
        kind=KIND,
        language=language,
        seed=cfg.seed,
        pairs=pairs,
        attributes=attributes,
        compat=pretrain_compat,
        pair_weights=pair_weights_from_groups(pairs, group_of, cfg.domain_group_weights),
    )


def build_user_prompt(language: str, attrs: dict[str, str], index: int) -> str:
    genre = GENRES[attrs["genre"]]
    lo, hi = LENGTH_WORD_RANGES[attrs["length_bucket"]]
    names = ", ".join(sample_names(language, index, KIND))
    return f"""\
{language_style_block(language)}

Write ONE document with these properties.
{domain_block(attrs["domain"], attrs["subtopic"])}
Genre: {genre.text}
Format hint: {genre.format_hint}
Register: {REGISTERS[attrs["register"]].text}
Audience: {AUDIENCES[attrs["audience"]].text}
Setting: {locale_text(language, attrs["locale"])}
Perspective: {PERSPECTIVES[attrs["perspective"]].text}
Time frame: {ERAS[attrs["era"]].text}
Reading level: {DIFFICULTIES[attrs["difficulty"]].text}
Length: {lo}-{hi} words.
Local given names you may use for fictional characters (fictional genres only): {names}.

Write a fitting title and the document text. Stay on the specific topic and make it concrete and locally grounded."""


def iter_requests(cfg: CorpusConfig, languages: Optional[list[str]] = None) -> Iterator[BatchRequestSpec]:
    response_format = strict_response_format(PretrainDoc, PRETRAIN_FORMAT_NAME)
    for language in resolve_languages(cfg, languages):
        sampler = build_pretrain_sampler(cfg, language)
        for i in range(cfg.samples_per_language[language]):
            attrs = sampler.draw()
            lo, hi = LENGTH_WORD_RANGES[attrs["length_bucket"]]
            yield BatchRequestSpec(
                custom_id=corpus_custom_id(KIND, language, i),
                task=KIND,
                language=language,
                system_prompt=PRETRAIN_SYSTEM_PROMPT,
                user_prompt=build_user_prompt(language, attrs, i),
                response_format=response_format,
                context={"kind": KIND, "index": i, "attributes": attrs, "length_words": [lo, hi]},
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )


def build_requests(cfg: CorpusConfig, languages: Optional[list[str]] = None) -> list[BatchRequestSpec]:
    return list(iter_requests(cfg, languages))
