"""The shipped yaml presets load, cover exactly the 12 languages, and carry the agreed counts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config.corpus_config import TARGET_LANGUAGE_CODES, load_config, parse_config
from config.languages import LANGUAGE_CODES
from config.settings import CONFIGS_DIR
from generators import dpo, pretrain, sft_tasks

PRESET_COUNTS = {"pretrain": 4000, "sft": 6000, "dpo": 1500}


def test_target_languages_are_the_twelve_non_english():
    assert len(TARGET_LANGUAGE_CODES) == 12
    assert "eng" not in TARGET_LANGUAGE_CODES
    assert set(TARGET_LANGUAGE_CODES) == set(LANGUAGE_CODES) - {"eng"}


@pytest.mark.parametrize("kind", PRESET_COUNTS)
def test_yaml_preset_loads_with_exact_languages_and_counts(kind):
    cfg = load_config(CONFIGS_DIR / f"{kind}.yaml")
    assert cfg.kind == kind
    assert cfg.model == "gpt-4o-mini"
    assert set(cfg.samples_per_language) == set(TARGET_LANGUAGE_CODES)
    assert all(n == PRESET_COUNTS[kind] for n in cfg.samples_per_language.values())
    assert cfg.total_requests == 12 * PRESET_COUNTS[kind]
    # all 12 languages are listed explicitly in the file (not only via `default`)
    raw = yaml.safe_load(Path(CONFIGS_DIR / f"{kind}.yaml").read_text(encoding="utf-8"))
    assert set(raw["samples_per_language"]) == set(TARGET_LANGUAGE_CODES) | {"default"}
    assert "eng" not in raw["samples_per_language"]


@pytest.mark.parametrize("kind", PRESET_COUNTS)
def test_load_by_bare_kind_name_and_relative_path(kind):
    assert load_config(kind).kind == kind
    assert load_config(f"configs/{kind}.yaml").kind == kind


@pytest.mark.parametrize("kind", PRESET_COUNTS)
def test_preset_weights_are_valid_for_every_language(kind):
    """Building each language's sampler validates every weight key against the taxonomy."""
    cfg = load_config(kind)
    build = {"pretrain": pretrain.build_pretrain_sampler, "sft": sft_tasks.build_sft_sampler, "dpo": dpo.build_dpo_sampler}[kind]
    for lang in TARGET_LANGUAGE_CODES:
        build(cfg, lang).draw_many(5)


def test_quality_sections_present():
    for kind in PRESET_COUNTS:
        q = load_config(kind).quality
        for key in ("max_repetition_ratio", "max_english_ratio", "max_english_ratio_by_language", "drop_low_confidence"):
            assert key in q
        assert "pcm" in q["max_english_ratio_by_language"]
    assert {"min_words", "max_words"} <= set(load_config("pretrain").quality)
    assert {"min_instruction_chars", "max_response_chars"} <= set(load_config("sft").quality)
    assert {"min_length_ratio", "max_length_ratio"} <= set(load_config("dpo").quality)
    assert load_config("sft").dedup["cross_language"] is True


def test_costs_are_documented_next_to_presets():
    for kind in PRESET_COUNTS:
        text = (CONFIGS_DIR / f"{kind}.yaml").read_text(encoding="utf-8")
        assert "COST ESTIMATE" in text and "$" in text


def test_rejects_english_and_unknown_languages_and_keys():
    base = {"kind": "sft", "samples_per_language": {"default": 5}}
    with pytest.raises(ValueError, match="English is excluded|Unsupported"):
        parse_config({**base, "samples_per_language": {"default": 5, "eng": 5}})
    with pytest.raises(ValueError):
        parse_config({**base, "samples_per_language": {"default": 5, "xxx": 5}})
    with pytest.raises(ValueError, match="Unknown top-level"):
        parse_config({**base, "bogus": 1})
    with pytest.raises(ValueError, match="Unknown key"):
        parse_config({**base, "quality": {"not_a_threshold": 1}})
    with pytest.raises(ValueError):
        parse_config({"kind": "nope", "samples_per_language": {"default": 5}})
    with pytest.raises(ValueError):
        parse_config({**base, "samples_per_language": {"yor": 5}})  # no default


def test_per_language_overrides_and_with_overrides():
    cfg = parse_config({"kind": "dpo", "samples_per_language": {"default": 10, "yor": 99}})
    assert cfg.samples_per_language["yor"] == 99 and cfg.samples_per_language["hau"] == 10
    small = cfg.with_overrides(per_language=2, languages=["yor", "hau"])
    assert small.samples_per_language == {"yor": 2, "hau": 2}
    with pytest.raises(ValueError):
        cfg.with_overrides(languages=["eng"])
