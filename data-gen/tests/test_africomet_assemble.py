"""Offline tests for the AfriCOMET assembly logic (dedupe, target planning, reference rows, rendering)."""

from __future__ import annotations

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from africomet_translation import assemble as A
from africomet_translation.text_utils import sent_hash

SCHEMA = pa.schema([("text", pa.string()), ("n_sentences", pa.int8()), ("tokens", pa.int16()),
                    ("hashes", pa.list_(pa.int64())), ("source", pa.string()), ("license", pa.string()),
                    ("kind", pa.string()), ("ref", pa.string()), ("ref_lang", pa.string())])


def _write_windows(out, lang, slug, windows, kind="web", refs=None, ref_lang=""):
    d = out / "windows" / lang
    d.mkdir(parents=True, exist_ok=True)
    cols = {k: [] for k in ("text", "n_sentences", "tokens", "hashes", "source", "license", "kind", "ref", "ref_lang")}
    for i, sents in enumerate(windows):
        cols["text"].append(" ".join(sents))
        cols["n_sentences"].append(len(sents))
        cols["tokens"].append(10 * len(sents))
        cols["hashes"].append([sent_hash(s) for s in sents])
        cols["source"].append(slug)
        cols["license"].append("CC0-1.0")
        cols["kind"].append(kind)
        cols["ref"].append(refs[i] if refs else "")
        cols["ref_lang"].append(ref_lang if refs else "")
    pq.write_table(pa.table(cols, schema=SCHEMA), d / f"{slug}-000.parquet")


def _rows(out):
    return [r for p in sorted((out / "final").glob("*/*.parquet")) for r in pq.read_table(p).to_pylist()]


SRC = {l: "windows" for l in A.LANGS}


def test_dedupe_keeps_the_higher_priority_source_and_counts_references(tmp_path):
    _write_windows(tmp_path, "yor", "HPLT_HPLT2.0_cleaned_yor_Latn_train-00000-of-00001.parquet",
                   [["Shared sentence number one is here.", "Only in hplt sentence."], ["Another hplt only sentence now."]])
    _write_windows(tmp_path, "yor", "openlanguagedata_flores_plus_yor_Latn",
                   [["Shared sentence number one is here."], ["Flores only sentence goes here."]], kind="curated",
                   refs=["Shared English one.", "Flores English two."], ref_lang="eng")
    res = A.dedupe_language(tmp_path, "yor")
    assert res["in"] == 4 and res["kept"] == 3
    assert sum(sum(p["ref"].values()) for p in res["parts"]) == 2                  # both flores rows carry a reference
    kept = [r["text"] for p in sorted((tmp_path / "kept_windows" / "yor").glob("[0-9]*.parquet")) for r in pq.read_table(p).to_pylist()]
    assert "Shared sentence number one is here." in kept and not any("Only in hplt" in t for t in kept)


def test_dedupe_accepts_old_windows_without_reference_columns(tmp_path):
    d = tmp_path / "windows" / "hau"
    d.mkdir(parents=True)
    old = pa.schema([f for f in SCHEMA if f.name not in ("ref", "ref_lang")])
    pq.write_table(pa.table({"text": ["Wannan jimla ce ta gwaji."], "n_sentences": pa.array([1], pa.int8()),
                             "tokens": pa.array([8], pa.int16()), "hashes": [[sent_hash("Wannan jimla ce ta gwaji.")]],
                             "source": ["masakhane_masakhanews_hau"], "license": ["AFL-3.0"], "kind": ["news"]}, schema=old),
                   d / "masakhane_masakhanews_hau-000.parquet")
    assert A.dedupe_language(tmp_path, "hau")["kept"] == 1


def test_split_counts_is_exact_and_proportional():
    got = A.split_counts(1001, {"a": 1, "b": 1, "c": 2})
    assert sum(got.values()) == 1001 and got["c"] in (500, 501)


def test_english_weights_equalise_every_target_including_fixed_reference_rows():
    base = {"yor": 400_000.0, "hau": 90_000.0, "ibo": 300_000.0, "fon": 900_000.0, "ewe": 700_000.0, "twi": 650_000.0,
            "efi": 60_000.0, "pcm": 80_000.0, "eng": 2_000_000.0}
    base.update({t: 150_000.0 + 20_000.0 * i for i, t in enumerate(A.NEW_LANGS)})
    E = 30_000_000
    w = A.english_weights(base, E)
    assert abs(sum(w.values()) - 1) < 1e-9 and "eng" not in w
    totals = {t: base[t] + E * w[t] for t in w}
    assert max(totals.values()) / min(totals.values()) < 1.001


def test_english_weights_only_feed_the_targets_that_are_behind():
    base = {t: 1_000_000.0 for t in A.LANGS if t != "eng"}
    base["hau"] = 0.0
    w = A.english_weights(base, 1_000)
    assert all(v >= 0 for v in w.values()) and w["hau"] == pytest.approx(1.0, abs=1e-6)


def test_free_rows_weight_english_target_double_and_exclude_ibibio_urhobo():
    w = A.target_weights_free("yor")
    assert w["eng"] == 2.0 and "yor" not in w and "zul" in w and not {"ibb", "urh"} & set(w)
    assert all(v == 1.0 for t, v in w.items() if t != "eng")


def test_rows_with_a_reference_get_that_language_as_target_and_response(tmp_path):
    _write_windows(tmp_path, "efi", "efik_pairs", [[f"Efik sentence number {i} alpha beta."] for i in range(120)], kind="jw",
                   refs=[f"English reference translation number {i}." for i in range(120)], ref_lang="eng")
    _write_windows(tmp_path, "zul", "zul_mono", [[f"Zulu sentence number {i} gamma delta."] for i in range(60)])
    stats = A.assemble(tmp_path, total=10_000, seed=1, tag="t", src_dirs=SRC)
    rows = _rows(tmp_path)
    efi = [r for r in rows if r["source_lang"] == "efi"]
    assert len(efi) == 120 and all(r["targ_lang"] == "eng" for r in efi)
    assert all(r["response"].startswith("English reference translation") for r in efi)
    zul = [r for r in rows if r["source_lang"] == "zul"]
    assert len(zul) == 60 and all(r["response"] == "" for r in zul)
    assert all(r["targ_lang"] != "zul" for r in zul) and len({r["targ_lang"] for r in zul}) > 3
    assert stats["with_response"]["efi"] == 120 and stats["with_response"]["zul"] == 0
    assert not {"ibb", "urh"} & {r["targ_lang"] for r in rows}


def test_required_columns_and_values_in_the_final_schema(tmp_path):
    for lang in ("yor", "eng"):
        _write_windows(tmp_path, lang, "masakhane_masakhanews_" + lang, [[f"{lang} sentence {i} alpha.", f"{lang} sentence {i} beta."] for i in range(300)], kind="news")
    A.assemble(tmp_path, total=500, seed=1, tag="t", src_dirs=SRC)
    rows = _rows(tmp_path)
    for k in ("instruction", "input", "response", "source_lang", "targ_lang"):
        assert k in rows[0]
    assert list(rows[0])[:5] == ["instruction", "input", "response", "source_lang", "targ_lang"]
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["source_lang"] != r["targ_lang"] for r in rows)
    assert all(r["input"] in r["prompt"] and r["instruction"] in r["prompt"] for r in rows)
    assert {"akan", "aka"}.isdisjoint({r["targ_lang"] for r in rows})
    assert 195 <= sum(r["source_lang"] == "eng" for r in rows) <= 205                # English tops up to ~total


def test_english_rows_with_references_are_kept_whole_and_not_counted_as_free(tmp_path):
    _write_windows(tmp_path, "eng", "pairs_rev", [[f"English pair sentence {i} one two."] for i in range(50)], kind="mined",
                   refs=[f"Ewe aligned sentence {i} kple nu." for i in range(50)], ref_lang="ewe")
    _write_windows(tmp_path, "eng", "free_eng", [[f"English free sentence {i} three four."] for i in range(400)])
    A.assemble(tmp_path, total=100, seed=1, tag="t", src_dirs=SRC)
    rows = _rows(tmp_path)
    with_ref = [r for r in rows if r["response"]]
    assert len(with_ref) == 50 and all(r["targ_lang"] == "ewe" for r in with_ref)
    assert 45 <= len(rows) - 50 <= 55                                                  # total=100 -> ~50 free rows


def test_rows_carry_the_expected_scaffold_and_the_tokenized_prompt_text(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sft_new.render import render

    _write_windows(tmp_path, "zul", "zul_mono", [[f"Zulu sentence number {i} gamma delta."] for i in range(80)])
    _write_windows(tmp_path, "eng", "eng_mono", [[f"English sentence {i} alpha beta."] for i in range(400)])
    A.assemble(tmp_path, total=300, seed=1, tag="t", src_dirs=SRC)
    rows = _rows(tmp_path)
    assert rows and all("output_structure" in r and "text" in r for r in rows)
    for r in rows[:50]:
        from africomet_translation.instructions import input_lang_tags
        want = input_lang_tags(r["source_lang"], r["instruction_language"], r["instruction_position"])      # every language of the user turn, in reading order
        assert r["output_structure"] == f"<|input_lang|>{want}<task_plan><translate></task_plan><|target_lang|><{r['targ_lang']}><response>"
    multi = [r for r in rows if r["instruction_language"] != r["source_lang"]]
    assert multi and all(r["output_structure"].startswith("<|input_lang|><") and r["output_structure"].count("><") >= 1 for r in multi)
    assert any(r["output_structure"].split("<task_plan>")[0].count("<") == 3 for r in multi)               # at least one two-language field
    for r in rows[:50]:
        assert r["text"] == render([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True)
