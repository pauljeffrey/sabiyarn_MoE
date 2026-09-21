"""Cost estimator sanity (offline; no API calls)."""

from __future__ import annotations

import pytest

from config.settings import BATCH_PRICING_USD_PER_M_TOKENS
from conftest import tiny
from pipeline.estimate import estimate_kind, format_estimate, main, pricing_for, tokens_per_word

PRESET_REQUESTS = {"pretrain": 48000, "sft": 72000, "dpo": 18000}


@pytest.mark.parametrize("kind", PRESET_REQUESTS)
def test_preset_estimate_is_sane(presets, kind):
    est = estimate_kind(presets[kind], sample_per_language=8)
    t = est["totals"]
    assert t["requests"] == PRESET_REQUESTS[kind]
    assert len(est["rows"]) == 12 and all(r["requests"] == PRESET_REQUESTS[kind] // 12 for r in est["rows"])
    assert t["input_tokens"] > t["requests"] * 300  # prompts are hundreds of tokens
    assert t["output_tokens"] > t["requests"] * 30
    (pin, pout), fallback = pricing_for("gpt-4o-mini")
    assert not fallback
    recomputed = t["input_tokens"] / 1e6 * pin + t["output_tokens"] / 1e6 * pout
    assert t["usd"] == pytest.approx(recomputed, rel=0.01)
    assert 1.0 < t["usd"] < 100.0  # a small-budget preset
    assert not est["truncation_risk"]
    assert t["batch_files"] >= 1
    assert "TOTAL" in format_estimate(est)


def test_pidgin_is_cheaper_than_yoruba_output(presets):
    est = estimate_kind(presets["pretrain"], sample_per_language=10)
    by_lang = {r["language"]: r for r in est["rows"]}
    assert by_lang["pcm"]["output_tokens"] < by_lang["yor"]["output_tokens"]
    assert tokens_per_word("yor") > tokens_per_word("pcm") >= 1.35


def test_cost_scales_linearly_with_volume(presets):
    small = estimate_kind(tiny(presets["sft"], 100), sample_per_language=10)["totals"]["usd"]
    big = estimate_kind(tiny(presets["sft"], 200), sample_per_language=10)["totals"]["usd"]
    assert big == pytest.approx(2 * small, rel=0.02)


def test_unknown_model_uses_flagged_fallback_price(presets):
    cfg = tiny(presets["dpo"], 5)
    cfg.model = "some-future-model"
    est = estimate_kind(cfg, sample_per_language=5)
    assert est["pricing_usd_per_m_tokens"]["fallback_used"] is True
    assert "FALLBACK" in format_estimate(est)


def test_pricing_constants_live_in_settings():
    assert "gpt-4o-mini" in BATCH_PRICING_USD_PER_M_TOKENS
    import config.settings as s

    src = open(s.__file__, encoding="utf-8").read()
    assert "verify current pricing" in src.lower()


def test_cli_prints_per_language_table_without_api(capsys):
    main(["--kind", "dpo", "--per-language", "3", "--sample", "3"])
    out = capsys.readouterr().out
    assert "== dpo" in out and "yor" in out and "fuv" in out and "No API calls were made" in out
