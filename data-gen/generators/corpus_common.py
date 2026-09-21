"""Helpers shared by the pretrain / SFT / DPO generators.

Kept separate from `generators/base.py` (which the six original task
generators depend on) so nothing here can affect them.
"""

from __future__ import annotations

import random
from typing import Mapping, Optional

from config.corpus_config import TARGET_LANGUAGE_CODES, CorpusConfig
from generators.base import make_custom_id, rng_for
from sampling.sampler import AttributeSpec, merge_weights
from sampling.taxonomy import DOMAINS, NAME_POOLS, Option, locale_info, locale_labels


def corpus_custom_id(kind: str, language: str, index: int) -> str:
    """`<kind>__<lang>__<index>__base` -- unique per (kind, language, index)."""
    return make_custom_id(kind, language, index)


def attribute_spec(name: str, defaults: Mapping[str, float], cfg: CorpusConfig) -> AttributeSpec:
    """Taxonomy defaults + the yaml `attribute_weights.<name>` overrides."""
    return AttributeSpec(name, merge_weights(defaults, cfg.attribute_weights.get(name), name=name))


def locale_spec(language: str, cfg: CorpusConfig) -> AttributeSpec:
    """Locale weights are uniform over the language's pool unless overridden in yaml."""
    defaults = {label: 1.0 for label in locale_labels(language)}
    overrides = cfg.attribute_weights.get("locale")
    return AttributeSpec("locale", merge_weights(defaults, overrides, name="locale") if overrides else defaults)


def options_text(options: Mapping[str, Option], key: str) -> str:
    return options[key].text


def locale_text(language: str, label: str) -> str:
    """Prompt text describing the setting, including currency/local realities."""
    _, country = locale_info(language, label)
    return f"{label} ({country.name}). Money is {country.currency}. Local realities: {country.notes}."


def sample_names(language: str, index: int, kind: str, k: int = 4) -> list[str]:
    """A few local given names, varied per example (deterministic)."""
    pool = NAME_POOLS[language]
    rng = rng_for(kind, "names", language, index)
    return rng.sample(pool, min(k, len(pool)))


def domain_block(domain_key: str, subtopic: str) -> str:
    d = DOMAINS[domain_key]
    return f"Domain: {d.name} -- {d.description}.\nSpecific topic: {subtopic}."


def language_names(language: str) -> str:
    from config.languages import get_language

    lang = get_language(language)
    return f"{lang.english_name} ({lang.endonym})"


def resolve_languages(cfg: CorpusConfig, languages: Optional[list[str]]) -> list[str]:
    langs = list(languages) if languages else cfg.languages
    bad = [c for c in langs if c not in TARGET_LANGUAGE_CODES]
    if bad:
        raise ValueError(f"Unsupported language(s) {bad}. Corpus kinds cover only {TARGET_LANGUAGE_CODES}.")
    return langs


def rng_for_example(kind: str, language: str, index: int) -> random.Random:
    return rng_for(kind, "example", language, index)
