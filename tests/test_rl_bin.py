import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rl import bin_data as B


def _fake_tokens(texts):
    return [[1] + [10 + (ord(c) % 40) for c in t[:60]] for t in texts]


def _make_shards(tmp_path, n=300):
    langs = ["zul", "xho", "eng", "hau"]
    rows = {"text": [], "input": [], "source_lang": [], "targ_lang": [], "instruction_language": [], "instruction_position": []}
    for i in range(n):
        s, t = langs[i % 4], langs[(i + 1) % 4]
        src = f"source sentence {i} " * (1 + i % 5)
        rows["text"].append(f"<s><|user|>Translate {src}</s><|assistant|>")
        rows["input"].append(src)
        rows["source_lang"].append(s)
        rows["targ_lang"].append(t)
        rows["instruction_language"].append(["ibo", "zul", "eng", "yor"][i % 4])
        rows["instruction_position"].append("append" if i % 3 == 0 else "prepend")
    for k in range(3):
        sl = slice(k * (n // 3), (k + 1) * (n // 3))
        pq.write_table(pa.table({c: v[sl] for c, v in rows.items()}), tmp_path / f"s{k}.parquet")
    return sorted(tmp_path.glob("s*.parquet")), rows


def test_csr_bin_roundtrip_shuffle_and_lazy_views(tmp_path):
    shards, rows = _make_shards(tmp_path)
    m = B.build(shards, tmp_path / "p", _fake_tokens, pad_id=3, eos_id=2, tokenizer_name="fake", eval_rows=20, buckets=8, seed=1)
    ds = B.RLPromptBin(tmp_path / "p")
    assert len(ds) == 300 and m["eval_start"] == 280 and m["max_prompt_tokens"] == max(len(_fake_tokens([t])[0]) for t in rows["text"])
    seen = sorted(ds[i]["source"] for i in range(len(ds)))
    assert seen == sorted(rows["input"])                                   # every row is there exactly once
    order = [ds[i]["source"] for i in range(len(ds))]
    assert order != rows["input"]                                         # and shuffled
    it = ds[7]
    from rl.structure import input_lang_tags
    want = input_lang_tags(it["source_lang"], it["instruction_lang"], it["instruction_position"])
    assert it["expected_structure"] == f"<|input_lang|>{want}<task_plan><translate></task_plan><|target_lang|><{it['targ_lang']}><response>"
    assert want.count("<") in (1, 2)
    multi = [ds[j] for j in range(len(ds)) if ds[j]["instruction_lang"] not in (None, ds[j]["source_lang"])]
    assert multi and all(m["expected_structure"].startswith("<|input_lang|><") and m["expected_structure"].count("<") >= 8 for m in multi)
    assert it["africomet_src"] and it["prompt_ids"][0] == 1
    assert len(ds.train_view()) == 280 and len(ds.eval_view()) == 20 and len(ds.eval_view(5)) == 5
    assert ds.eval_view()[0]["index"] == 280


def test_padded_layout_matches_csr_and_collate_pads_on_the_left(tmp_path):
    shards, _ = _make_shards(tmp_path)
    B.build(shards, tmp_path / "c", _fake_tokens, 3, 2, "fake", layout="csr", buckets=4, seed=2)
    B.build(shards, tmp_path / "d", _fake_tokens, 3, 2, "fake", layout="padded", buckets=4, seed=2)
    c, d = B.RLPromptBin(tmp_path / "c"), B.RLPromptBin(tmp_path / "d")
    assert all(c[i]["prompt_ids"] == d[i]["prompt_ids"] and c[i]["source"] == d[i]["source"] for i in range(0, 300, 17))
    batch = B.collate_left_pad([c[0], c[1], c[2]], pad_id=3, as_torch=False)
    lens = [len(c[i]["prompt_ids"]) for i in range(3)]
    L = max(lens)
    assert batch["input_ids"].shape == (3, L) and (batch["attention_mask"].sum(1) == np.array(lens)).all()
    for r, n in enumerate(lens):
        assert (batch["input_ids"][r, : L - n] == 3).all() and batch["input_ids"][r, L - n] == 1 and batch["attention_mask"][r, : L - n].sum() == 0
