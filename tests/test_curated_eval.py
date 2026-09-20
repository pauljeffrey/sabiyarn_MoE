import json
import math
from collections import Counter
from pathlib import Path

import pytest

from training.curated_eval import LN2, CuratedTotals, build_sequence, load_curated_samples, resolve_path

DATA = Path(__file__).resolve().parents[1] / "data" / "curated_eval.jsonl"


def test_build_sequence_mirrors_bin_layout_and_strips_leading_bos():
    assert build_sequence([1, 5, 6], eos=2, bos=1) == [2, 2, 5, 6, 2, 2]
    assert build_sequence([5, 6], eos=2, bos=None) == [2, 2, 5, 6, 2, 2]
    assert build_sequence([5, 6], eos=2, bos=1) == [2, 2, 5, 6, 2, 2]  # no bos to strip


def test_totals_weight_by_tokens_and_convert_to_bpb():
    t = CuratedTotals()
    t.add("eng", ce_mean=2.0, n_tokens=10, n_bytes=40)
    t.add("eng", ce_mean=4.0, n_tokens=30, n_bytes=100)
    t.add("yor", ce_mean=6.0, n_tokens=20, n_bytes=50)
    s = t.summary()
    assert s["by_language"]["eng"]["ce"] == pytest.approx((20 + 120) / 40)
    assert s["by_language"]["eng"]["n"] == 2
    assert s["by_language"]["yor"]["bpb"] == pytest.approx(120 / (LN2 * 50))
    assert s["overall"]["ce"] == pytest.approx((140 + 120) / 60)
    assert s["overall"]["n"] == 3


def test_loader_rejects_rows_without_lang_or_text(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"lang": "eng"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_curated_samples(str(bad))


def test_resolve_path():
    assert resolve_path("data/x.jsonl", "/repo") == "/repo/data/x.jsonl"
    assert resolve_path("/abs/x.jsonl", "/repo") == "/abs/x.jsonl"


def test_shipped_curated_file_is_well_formed():
    rows = load_curated_samples(str(DATA))
    assert len(rows) == 100
    assert Counter(r["lang"] for r in rows) == {"eng": 20, "pcm": 20, "yor": 20, "hau": 20, "ibo": 20}
    assert len({r["text"] for r in rows}) == 100
    assert all(10 <= len(r["text"]) <= 400 for r in rows)
