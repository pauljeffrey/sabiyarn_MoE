"""At the shipped preset volumes every domain, (domain, sub-topic) pair, genre, SFT task type and DPO flaw type
is requested in every one of the 12 languages (no API calls; ~8s)."""
import pytest

from config.corpus_config import TARGET_LANGUAGE_CODES
from scripts.audit_coverage import audit


@pytest.mark.parametrize("kind", ["pretrain", "sft", "dpo"])
def test_presets_cover_the_whole_taxonomy_in_every_language(kind):
    result = audit(kind)
    assert set(result) == set(TARGET_LANGUAGE_CODES)
    for lang, attrs in result.items():
        for name, r in attrs.items():
            assert not r["missing"], f"[{kind}/{lang}] {name} never requested: {r['missing']}"
            assert not r["unknown"], f"[{kind}/{lang}] {name} outside the vocabulary: {r['unknown']}"
            assert r["min"] >= 1
