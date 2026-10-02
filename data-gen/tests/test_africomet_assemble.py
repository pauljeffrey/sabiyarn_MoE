"""Offline tests for the large-scale AfriCOMET assembly logic (dedupe, target balancing, rendering)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from africomet_translation import assemble as A
from africomet_translation.text_utils import sent_hash

SCHEMA = pa.schema([("text", pa.string()), ("n_sentences", pa.int8()), ("tokens", pa.int16()),
                    ("hashes", pa.list_(pa.int64())), ("source", pa.string()), ("license", pa.string()), ("kind", pa.string())])


def _write_windows(out, lang, slug, windows, kind="web"):
    d = out / "windows" / lang
    d.mkdir(parents=True, exist_ok=True)
    cols = {"text": [], "n_sentences": [], "tokens": [], "hashes": [], "source": [], "license": [], "kind": []}
    for sents in windows:
        cols["text"].append(" ".join(sents))
        cols["n_sentences"].append(len(sents))
        cols["tokens"].append(10 * len(sents))
        cols["hashes"].append([sent_hash(s) for s in sents])
        cols["source"].append(slug)
        cols["license"].append("CC0-1.0")
        cols["kind"].append(kind)
    pq.write_table(pa.table(cols, schema=SCHEMA), d / f"{slug}-000.parquet")


def test_dedupe_drops_windows_that_reuse_an_earlier_sentence_and_keeps_priority_order(tmp_path):
    # flores_plus (priority 10) must beat HPLT (priority 50) for the shared sentence, whatever the file order.
    _write_windows(tmp_path, "yor", "HPLT_HPLT2.0_cleaned_yor_Latn_train-00000-of-00001.parquet",
                   [["Shared sentence number one is here.", "Only in hplt sentence."], ["Another hplt only sentence now."]])
    _write_windows(tmp_path, "yor", "openlanguagedata_flores_plus_yor_Latn",
                   [["Shared sentence number one is here."], ["Flores only sentence goes here."]], kind="curated")
    res = A.dedupe_language(tmp_path, "yor")
    assert res["in"] == 4 and res["kept"] == 3
    kept = [r["text"] for p in sorted((tmp_path / "kept" / "yor").glob("[0-9]*.parquet")) for r in pq.read_table(p).to_pylist()]
    assert "Shared sentence number one is here." in kept                      # the flores copy survived
    assert not any("Only in hplt" in t for t in kept)                        # the hplt window containing it is gone


def test_dedupe_removes_duplicates_inside_one_shard(tmp_path):
    _write_windows(tmp_path, "hau", "masakhane_masakhanews_hau",
                   [["Same sentence appears twice here."], ["Same sentence appears twice here.", "Plus another one."]], kind="news")
    assert A.dedupe_language(tmp_path, "hau")["kept"] == 1


def test_split_counts_is_exact_and_proportional():
    got = A.split_counts(1001, {"a": 1, "b": 1, "c": 2})
    assert sum(got.values()) == 1001 and got["c"] in (500, 501)


def test_english_weights_equalise_every_target_total():
    non_eng = {"yor": 400_000, "hau": 1_500_000, "ibo": 300_000, "fon": 30_000, "ewe": 60_000, "twi": 90_000,
               "efi": 20_000, "ibb": 12_000, "pcm": 50_000}
    E = 6_000_000
    w = A.english_weights(non_eng, E)
    assert abs(sum(w.values()) - 1) < 1e-9 and "eng" not in w
    totals = Counter()
    for s, n in non_eng.items():
        tw = A.target_weights_non_eng(s)
        z = sum(tw.values())
        for t, x in tw.items():
            totals[t] += n * x / z
    for t, x in w.items():
        totals[t] += E * x
    tg = [totals[t] for t in w]
    assert max(tg) / min(tg) < 1.001                                          # equalised to within 0.1%


def test_english_weights_only_feed_the_targets_that_are_behind():
    # 9M Hausa rows give every OTHER target ~1M rows but give Hausa itself none, so a small English top-up must all
    # go to Hausa, and no weight may be negative.
    w = A.english_weights({"hau": 9_000_000}, 1_000)
    assert all(v >= 0 for v in w.values())
    assert w["hau"] == pytest.approx(1.0, abs=1e-6)


def test_non_english_sources_weight_english_target_double():
    w = A.target_weights_non_eng("yor")
    assert w["eng"] == 2.0 and "yor" not in w and all(v == 1.0 for t, v in w.items() if t != "eng")


def test_full_pipeline_on_synthetic_windows(tmp_path):
    for lang in ("yor", "eng"):
        wins = [[f"{lang} sentence {i} alpha.", f"{lang} sentence {i} beta."] for i in range(300)]
        _write_windows(tmp_path, lang, "masakhane_masakhanews_" + lang, wins, kind="news")
    stats = A.assemble(tmp_path, total=500, seed=1, tag="t")
    assert stats["rows"]["yor"] == 300 and 195 <= stats["rows"]["eng"] <= 205         # English tops up to ~total
    rows = [r for p in (tmp_path / "final").glob("*/*.parquet") for r in pq.read_table(p).to_pylist()]
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["source_language"] != r["target_language"] for r in rows)
    assert all(r["input"] in r["prompt"] and r["instruction"] in r["prompt"] for r in rows)
    assert all(r["target_language"] != "eng" for r in rows if r["source_language"] == "eng")
    assert {"akan", "aka", "urh"}.isdisjoint({r["target_language"] for r in rows})
