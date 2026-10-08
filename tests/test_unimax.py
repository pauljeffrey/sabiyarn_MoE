import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("pu", Path(__file__).resolve().parents[1] / "data" / "prepare_unimax.py")
pu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pu)


def test_unimax_matches_the_paper_algorithm():
    counts = {"a": 100, "b": 1000, "c": 100_000}
    u = pu.unimax(counts, budget=30_000, max_repeat=4)
    assert u["a"] == 400 and u["b"] == 4000
    assert u["c"] == pytest.approx(30_000 - 400 - 4000)
    assert sum(u.values()) <= 30_000 + 1e-6


def test_unimax_is_uniform_when_every_language_is_large_enough():
    u = pu.unimax({"a": 1e6, "b": 2e6, "c": 3e6}, budget=3e6, max_repeat=4)
    assert u["a"] == u["b"] == u["c"] == pytest.approx(1e6)


def test_unimax_never_exceeds_max_repeat_and_leaves_budget_unused_when_data_is_short():
    assert pu.unimax({"a": 10, "b": 20}, budget=1e9, max_repeat=4) == {"a": 40, "b": 80}


def test_validation_size_rule_and_hash_split_are_deterministic():
    assert pu.calculate_test_size(40) == 0 and pu.calculate_test_size(1000) == 100 and pu.calculate_test_size(500_000) == 5000
    flags = [pu.is_val(f"doc {i}", 0.1) for i in range(5000)]
    assert flags == [pu.is_val(f"doc {i}", 0.1) for i in range(5000)]
    assert 0.08 < sum(flags) / len(flags) < 0.12


def test_rule_check_flags_the_usual_noise_and_passes_ordinary_text():
    import random as _r
    rr = _r.Random(3)
    syl = ["ba", "ko", "mu", "ta", "ne", "li", "sa", "ro", "du", "ki", "ze", "po"]
    good = " ".join("".join(rr.choice(syl) for _ in range(rr.randint(2, 5))) for _ in range(120))
    assert pu.rule_check(good) is None
    assert pu.rule_check("a b c " * 40) == "mean_word_length"
    assert pu.rule_check(" ".join(["the same repeated phrase here"] * 60)) in ("top_2gram", "top_3gram", "top_4gram")
    assert pu.rule_check("short text only") == "too_few_words"
    junk = " ".join(f"{i}{i}12" for i in range(100))
    assert pu.rule_check(junk) == "alpha_word_ratio"
    dup = "\n".join(["This is one repeated line of text, again and again for sure."] * 8 + ["x y z w v u t s r q p o n m l k j i h g f e d c b a"])
    assert pu.rule_check(dup) in ("duplicate_lines", "top_2gram", "top_3gram", "top_4gram", "dup_5gram", "dup_8gram", "dup_10gram")


def test_prior_select_removes_the_documents_farthest_from_the_medians_and_hits_the_token_target():
    rng = np.random.default_rng(0)
    mu = rng.normal(-9, 0.2, 1000).astype(np.float32)
    sg = rng.normal(0.01, 0.002, 1000).astype(np.float32)
    mu[:5] = -20.0                                           # five obvious outliers in mu
    sg[5:10] = 0.5                                           # five obvious outliers in sigma
    lens = np.full(1000, 100, dtype=np.int64)
    keep = pu.prior_select(mu, sg, lens, discard_tokens=1000)
    assert not keep[:10].any()                               # all ten outliers are gone
    removed = int(lens[~keep].sum())
    assert 1000 <= removed <= 1000 + 100 * 2 * 1             # the union just reaches the target
    assert pu.prior_select(mu, sg, lens, 0).all()


def test_doc_prior_stats_and_token_priors():
    class D:
        def __init__(self, arrs): self.a = arrs
        def tokens(self, i): return self.a[i]
    docs = D([np.array([1, 1, 1, 2, 3, 9], dtype=np.uint16), np.array([1, 1, 2, 9], dtype=np.uint16)])
    p = pu.token_priors(docs, np.array([True, True]), np.random.default_rng(0), sample=1.0)
    assert p[1] > p[2] > p[7] and abs(p.sum() - 1) < 1e-9
    mu, sg = pu.doc_prior_stats(docs.a[0], p, np.log(p))
    assert mu < 0 and sg > 0


def test_final_split_caps_validation_tokens_and_returns_the_surplus_to_training(tmp_path):
    d = tmp_path / "aaa"
    d.mkdir()
    rng = np.random.default_rng(0)
    for side, n in (("train", 20), ("val", 30)):
        lens = rng.integers(40, 60, size=n)
        np.concatenate([np.full(l, 7, dtype=np.uint16) for l in lens]).tofile(d / f"{side}.part00000.bin")
        np.save(d / f"{side}.part00000.offsets.npy", np.cumsum(lens).astype(np.uint64))
    docs = pu.LangDocs(d)
    is_v = pu.final_split(docs, np.random.default_rng(1), cap=500)
    assert not is_v[:20].any()                                # training documents never become validation
    got = int(docs.len[is_v].sum())
    assert 500 <= got < 500 + 60
    assert (~is_v).sum() == len(docs) - is_v.sum()


def test_assemble_shuffles_documents_across_languages_and_repeats_small_ones(tmp_path):
    out = tmp_path
    rng = np.random.default_rng(0)
    for lang, n_docs, base in (("aaa", 10, 100), ("bbb", 60, 1000)):
        d = out / lang
        d.mkdir()
        (d / "state.json").write_text("{}")
        for side, k in (("train", n_docs), ("val", 2)):
            lens = rng.integers(20, 40, size=k)
            toks = np.concatenate([np.full(l, base + i, dtype=np.uint16) for i, l in enumerate(lens)])
            toks.tofile(d / f"{side}.part00000.bin")
            np.save(d / f"{side}.part00000.offsets.npy", np.cumsum(lens).astype(np.uint64))
    pu.EOT_ID = 2
    for lang in ("aaa", "bbb"):                      # give every synthetic document a closing end-of-text id, like the real tokenizer output
        for side in ("train", "val"):
            f = out / lang / f"{side}.part00000.bin"
            offs = np.load(out / lang / f"{side}.part00000.offsets.npy")
            arr = np.fromfile(f, dtype=np.uint16)
            arr[(offs - 1).astype(int)] = 2
            arr.tofile(f)
    pu.LANGS[:] = ["aaa", "bbb"]
    plan = pu.make_plan(out, budget=1e9, max_repeat=4, langs=["aaa", "bbb"])
    m = pu.assemble(out, "t")
    assert plan["languages"]["aaa"]["epochs"] > 1                                    # the small language is repeated
    data = np.fromfile(out / "t.bin", dtype=np.uint16)
    assert data.size == m["train_total"]
    langs_in_order = (data >= 1000)                                                  # token ids encode the language here
    switches = int(np.count_nonzero(langs_in_order[1:] != langs_in_order[:-1]))
    assert switches > 20                                                             # documents of both languages are interleaved
    assert m["eot_tokens_in_file"] == m["n_docs_written"] and m["file_ends_with_eot"]   # an end-of-text between every pair of documents
    segs = pu.write_new_val(out, out / "val.bin")
    assert set(segs) == {"aaa", "bbb"} and segs["aaa"][1] == segs["bbb"][0]


def test_tail_documents_are_appended_after_the_shuffled_body(tmp_path):
    out = tmp_path
    rng = np.random.default_rng(1)
    for lang, n_docs, base in (("aaa", 20, 100), ("eng-tail", 15, 5000)):
        d = out / lang
        d.mkdir()
        (d / "state.json").write_text("{}")
        for side, k in (("train", n_docs), ("val", 2)):
            lens = rng.integers(20, 40, size=k)
            toks = np.concatenate([np.full(l, base + i, dtype=np.uint16) for i, l in enumerate(lens)])
            toks[np.cumsum(lens) - 1] = 2
            toks.tofile(d / f"{side}.part00000.bin")
            np.save(d / f"{side}.part00000.offsets.npy", np.cumsum(lens).astype(np.uint64))
    pu.EOT_ID = 2
    pu.make_plan(out, budget=1e9, max_repeat=1, langs=["aaa"])
    m = pu.assemble(out, "t")
    data = np.fromfile(out / "t.bin", dtype=np.uint16)
    ids = data[data != 2]
    first_tail = int(np.flatnonzero(ids >= 5000)[0])
    assert (ids[first_tail:] >= 5000).all() and (ids[:first_tail] < 5000).all()             # every tail token comes after every body token
    assert m["tail"]["eng-tail"]["docs"] == 17 and m["n_docs_written"] == m["eot_tokens_in_file"] and m["file_ends_with_eot"]
    assert m["train_total"] == data.size
