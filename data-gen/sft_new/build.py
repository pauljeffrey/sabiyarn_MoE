"""Translation / instruction SFT rows in the SabiYarn chat structure.

Columns: instruction, input, response, source_lang, targ_lang, messages, source_dataset, quality.
`messages` is ready for tokenizer.apply_chat_template: the user turn is the instruction + text (placement varies,
instruction language varies) and the assistant turn follows the corpus convention
`<|input_lang|><src><task_plan><translate></task_plan><|target_lang|><tgt><response>TEXT`.

    nice -n 10 python -m sft_new.build --source multi-open --out OUT [--push]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from africomet_translation import lid
from africomet_translation.instructions import (compose_prompt, instruction_languages, make_instruction)
from africomet_translation.text_utils import good_sentence, normalize

REPO = "BeardedMonster/sft-new"
SCHEMA = pa.schema([("instruction", pa.string()), ("input", pa.string()), ("response", pa.string()),
                    ("source_lang", pa.string()), ("targ_lang", pa.string()),
                    ("messages", pa.string()), ("source_dataset", pa.string()), ("quality", pa.float32())])

# multi-open config -> (code, minimum translation_quality_score). Scores cluster at 1, 3-4, 6-7 and 10. Measured with
# GlotLID on 600 sampled rows per (language, score): scores 1-3 fail the language check 30-80% of the time, 4-5 still
# misses 15-40% (Twi, Wolof, Fon worst) and shows loose alignment on inspection, 6+ is clean. 6 is the cut for all.
MULTI_OPEN = {"afrikaans": ("afr", 6), "chewa": ("nya", 6), "ewe": ("ewe", 6), "fon": ("fon", 6), "hausa": ("hau", 6),
              "igbo": ("ibo", 6), "kinyarwanda": ("kin", 6), "oromo": ("orm", 6), "shona": ("sna", 6), "somali": ("som", 6),
              "southern_sotho": ("sot", 6), "swahili": ("swh", 6), "tswana": ("tsn", 6), "twi": ("twi", 6),
              "wolof": ("wol", 6), "xhosa": ("xho", 6), "yoruba": ("yor", 6), "zulu": ("zul", 6)}
# FineTranslations: real FineWeb-2 text (source side) + Gemma-3-27B English translation. LLM-written, so it is gated on
# the columns that measure it: edu_score (0-5, English side), og_language_score and early_stop (truncated translations).
FINETRANS = {"hau": "hau_Latn", "ibo": "ibo_Latn", "yor": "yor_Latn", "kin": "kin_Latn", "nya": "nya_Latn", "orm": "gaz_Latn",
             "pcm": "pcm_Latn", "sna": "sna_Latn", "som": "som_Latn", "sot": "sot_Latn", "swh": "swh_Latn", "wol": "wol_Latn",
             "xho": "xho_Latn", "zul": "zul_Latn", "afr": "afr_Latn"}
FT_MIN_EDU, FT_MIN_LANG, FT_CAP = 2.0, 0.95, 100_000
SWIM_IR = {"sw": "swh", "yo": "yor"}
SWIM_CAP = 150_000
OPUS100 = {"af": "afr", "ha": "hau", "ig": "ibo", "yo": "yor", "xh": "xho", "zu": "zul", "rw": "kin"}
MAX_PER_ENGLISH = 2
INSTR_LANGS = None


def _instr_langs():
    global INSTR_LANGS
    if INSTR_LANGS is None:
        INSTR_LANGS = instruction_languages()
    return INSTR_LANGS


def pair_ok(eng: str, other: str, long: bool = False) -> bool:
    lim_e, lim_o, mw = (2600, 3400, 700) if long else (700, 900, 120)
    if not (12 <= len(eng) <= lim_e and 8 <= len(other) <= lim_o):
        return False
    r = (len(other) + 1) / (len(eng) + 1)
    if not 0.4 <= r <= 3.2:
        return False
    return good_sentence(eng, min_words=3, max_words=mw, require_terminal=False, lenient=True) and \
        good_sentence(other, min_words=2, max_words=mw + 20, require_terminal=False, lenient=True)


def make_row(src: str, tgt: str, a: str, b: str, source: str, quality: float | None, rng: random.Random) -> dict:
    """a is in `src`, b is its translation into `tgt`."""
    il = rng.choice(_instr_langs()) if rng.random() < 0.5 else "eng"
    ins = make_instruction(il, src, tgt, rng)
    prompt = compose_prompt(ins, a, rng)
    reply = f"<|input_lang|><{src}><task_plan><translate></task_plan><|target_lang|><{tgt}><response>{b}"
    return {"instruction": ins.text, "input": a, "response": b, "source_lang": src, "targ_lang": tgt,
            "messages": json.dumps([{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}], ensure_ascii=False),
            "source_dataset": source, "quality": quality}


class Shards:
    def __init__(self, out: Path, name: str, rows_per_shard: int = 100_000):
        self.dir = out / "data" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n, self.rows, self.max, self.total = 0, [], rows_per_shard, 0

    def add(self, row: dict):
        self.rows.append(row)
        self.total += 1
        if len(self.rows) >= self.max:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        cols = {k: [r[k] for r in self.rows] for k in SCHEMA.names}
        pq.write_table(pa.table(cols, schema=SCHEMA), self.dir / f"part-{self.n:05d}.parquet", compression="zstd")
        self.n += 1
        self.rows = []


def _h(*parts: str) -> int:
    return int.from_bytes(hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest(), "little")


def lid_filter(pairs: list[tuple[str, str]], code: str) -> list[bool]:
    en = lid.top_labels([p[0] for p in pairs])
    ok = lid.EXPECTED[code]
    ot = lid.top_labels([p[1] for p in pairs])
    return [e == "eng_Latn" and o in ok for e, o in zip(en, ot)]


def process_pairs(code: str, raw, source: str, seen: set, per_eng: dict, rng, shards: Shards, stats: dict, quality=None,
                  long: bool = False):
    """raw: iterable of (english, other, quality). Applies cleaning, LID on both sides, dedupe; emits rows in both
    directions (chosen at random, 50/50)."""
    batch: list[tuple[str, str, float | None]] = []

    def flush():
        if not batch:
            return
        keep = lid_filter([(a, b) for a, b, _ in batch], code)
        for (a, b, q), k in zip(batch, keep):
            if not k:
                stats["lid_out"] += 1
                continue
            key = _h(a.lower(), b.lower())
            if key in seen:
                stats["dup"] += 1
                continue
            ek = (code, _h(a.lower()))
            if per_eng.get(ek, 0) >= MAX_PER_ENGLISH:
                stats["dup_en"] += 1
                continue
            seen.add(key)
            per_eng[ek] = per_eng.get(ek, 0) + 1
            if rng.random() < 0.5:
                shards.add(make_row("eng", code, a, b, source, q, rng))
            else:
                shards.add(make_row(code, "eng", b, a, source, q, rng))
            stats["kept"] += 1
        batch.clear()

    for a, b, q in raw:
        stats["seen"] += 1
        a, b = normalize(a), normalize(b)
        if not pair_ok(a, b, long):
            stats["filtered"] += 1
            continue
        batch.append((a, b, q))
        if len(batch) >= 2000:
            flush()
    flush()


def multi_open_pairs(cfg: str, minscore: int, hf_token: str, scratch: Path):
    from huggingface_hub import hf_hub_download
    cache = Path(tempfile.mkdtemp(dir=scratch))
    try:
        p = hf_hub_download("African-Languages-Lab/multi-open", f"data/english-{cfg}/train-00000.parquet", repo_type="dataset",
                            token=hf_token, cache_dir=str(cache))
        pf = pq.ParquetFile(p)
        other = [c for c in pf.schema_arrow.names if c not in ("english", "translation_quality_score")
                 and not c.endswith("_token_count")][0]           # the column name is not always the config name
        for batch in pf.iter_batches(batch_size=20000, columns=["english", other, "translation_quality_score"]):
            d = batch.to_pydict()
            for e, o, s in zip(d["english"], d[other], d["translation_quality_score"]):
                if s is None or int(s) < minscore or not e or not o:
                    continue
                yield e, o, float(s)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def opus100_pairs(two: str, hf_token: str, scratch: Path):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=hf_token)
    files = [f.path for f in api.list_repo_tree("sentence-transformers/parallel-sentences-opus-100", path_in_repo=f"en-{two}",
                                                repo_type="dataset") if f.path.endswith(".parquet")]
    for path in files:
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            p = hf_hub_download("sentence-transformers/parallel-sentences-opus-100", path, repo_type="dataset", token=hf_token,
                                cache_dir=str(cache))
            for batch in pq.ParquetFile(p).iter_batches(batch_size=20000):
                d = batch.to_pydict()
                for e, o in zip(d["english"], d["non_english"]):
                    yield e, o, None
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def finetrans_pairs(label: str, hf_token: str, stats: dict):
    """Yield (english, other, edu_score) from FineTranslations, reading only the needed columns over HTTP ranges."""
    from huggingface_hub import HfApi, HfFileSystem
    api = HfApi(token=hf_token)
    files = sorted(f.path for f in api.list_repo_tree("HuggingFaceFW/finetranslations", path_in_repo=f"data/{label}",
                                                      repo_type="dataset", recursive=True) if f.path.endswith(".parquet"))
    fs = HfFileSystem(token=hf_token)
    cols = ["og_chunks", "translated_chunks", "edu_score", "og_language_score", "early_stop"]
    for path in files:
        with fs.open(f"datasets/HuggingFaceFW/finetranslations/{path}", "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=2000, columns=cols):
                d = batch.to_pydict()
                for og, tr, edu, ls, early in zip(d["og_chunks"], d["translated_chunks"], d["edu_score"], d["og_language_score"], d["early_stop"]):
                    stats["seen"] += 1
                    if early or (edu or 0) < FT_MIN_EDU or (ls or 0) < FT_MIN_LANG or not og or len(og) != len(tr):
                        stats["gated"] += 1
                        continue
                    for o, t in zip(og, tr):
                        yield t, o, float(edu)


def swim_ir_rows(two: str, code: str, hf_token: str, scratch: Path, rng: random.Random, stats: dict):
    """nthakur/swim-ir-cross-lingual: an LLM-written question in the language plus the English Wikipedia passage it was
    written from. Rendered as question -> passage (a cross-lingual QA / retrieval-style answer)."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=hf_token)
    files = sorted(f.path for f in api.list_repo_tree("nthakur/swim-ir-cross-lingual", path_in_repo=two, repo_type="dataset")
                   if f.path.endswith(".parquet"))
    seen: set[int] = set()
    keep_p = 0.35                                    # ~900k rows per language -> about SWIM_CAP kept after filters
    for path in files:
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            p = hf_hub_download("nthakur/swim-ir-cross-lingual", path, repo_type="dataset", token=hf_token, cache_dir=str(cache))
            for batch in pq.ParquetFile(p).iter_batches(batch_size=20000, columns=["query", "title", "text"]):
                d = batch.to_pydict()
                qs, ps = [], []
                for q, ti, tx in zip(d["query"], d["title"], d["text"]):
                    stats["seen"] += 1
                    if rng.random() > keep_p:
                        continue
                    q, tx = normalize(q or ""), normalize(tx or "")
                    w = q.split()
                    if not (4 <= len(w) <= 60) or len(set(x.lower() for x in w)) / len(w) < 0.7:
                        stats["bad_query"] += 1
                        continue
                    if not (150 <= len(tx) <= 1800) or not (tx[0].isupper() or tx[0].isdigit()):
                        stats["bad_passage"] += 1
                        continue
                    qs.append(q)
                    ps.append(f"{normalize(ti or '')}\n{tx}".strip() if ti else tx)
                if not qs:
                    continue
                ok = lid.verify(qs, code)
                for q, ps_, good in zip(qs, ps, ok):
                    h = _h(q.lower())
                    if not good:
                        stats["lid_out"] += 1
                    elif h in seen:
                        stats["dup"] += 1
                    else:
                        seen.add(h)
                        yield q, ps_
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def qa_row(code: str, q: str, passage: str, source: str) -> dict:
    reply = f"<|input_lang|><{code}><task_plan><qa></task_plan><|target_lang|><eng><response>{passage}"
    return {"instruction": q, "input": "", "response": passage, "source_lang": code, "targ_lang": "eng",
            "messages": json.dumps([{"role": "user", "content": q}, {"role": "assistant", "content": reply}], ensure_ascii=False),
            "source_dataset": source, "quality": None}


def push(out: Path, name: str, token: str, n: int):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(out / "data" / name), path_in_repo=f"data/{name}",
                      commit_message=f"{name}: {n:,} rows")
    shutil.rmtree(out / "data" / name, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["multi-open", "opus100", "swim-ir", "finetranslations"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--langs", nargs="*")
    a = ap.parse_args(argv)
    from africomet_translation.build import hf_token
    token = hf_token()
    os.environ["HF_TOKEN"] = token
    out = Path(a.out)
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    (out / "stats").mkdir(exist_ok=True)
    if a.source == "finetranslations":
        for code, label in FINETRANS.items():
            name = f"translation-finetranslations-{code}"
            if (out / "stats" / f"{name}.json").exists():
                continue
            rng = random.Random(f"ft-{code}")
            stats, shards, seen, per_eng = defaultdict(int), Shards(out, name), set(), {}

            def capped(it):
                for x in it:
                    if stats["kept"] >= FT_CAP:
                        return
                    yield x
            process_pairs(code, capped(finetrans_pairs(label, token, stats)), "HuggingFaceFW/finetranslations", seen, per_eng, rng,
                          shards, stats, long=True)
            shards.flush()
            (out / "stats" / f"{name}.json").write_text(json.dumps(dict(stats), indent=1))
            print(name, dict(stats), file=sys.stderr, flush=True)
            if a.push and stats["kept"]:
                push(out, name, token, stats["kept"])
        return
    if a.source == "swim-ir":
        for two, code in SWIM_IR.items():
            name = f"instruct-swim-ir-{code}"
            if (out / "stats" / f"{name}.json").exists():
                continue
            rng = random.Random(f"swim-{code}")
            stats, shards = defaultdict(int), Shards(out, name)
            for q, passage in swim_ir_rows(two, code, token, scratch, rng, stats):
                shards.add(qa_row(code, q, passage, "nthakur/swim-ir-cross-lingual"))
                stats["kept"] += 1
                if stats["kept"] >= SWIM_CAP:
                    break
            shards.flush()
            (out / "stats" / f"{name}.json").write_text(json.dumps(dict(stats), indent=1))
            print(name, dict(stats), file=sys.stderr, flush=True)
            if a.push and stats["kept"]:
                push(out, name, token, stats["kept"])
        return
    jobs = ([(cfg, code, ms) for cfg, (code, ms) in MULTI_OPEN.items()] if a.source == "multi-open"
            else [(two, code, 0) for two, code in OPUS100.items()])
    # opus-100 is deduplicated against multi-open per language, so the language order and the shared hash set matter
    for cfg, code, ms in jobs:
        if a.langs and code not in a.langs:
            continue
        name = f"translation-{a.source}-{code}"
        if (out / "stats" / f"{name}.json").exists():
            print(name, "done", file=sys.stderr)
            continue
        rng = random.Random(f"{a.source}-{code}")
        seen, per_eng = set(), {}
        stats = defaultdict(int)
        shards = Shards(out, name)
        raw = (multi_open_pairs(cfg, ms, token, scratch) if a.source == "multi-open" else opus100_pairs(cfg, token, scratch))
        src = "African-Languages-Lab/multi-open" if a.source == "multi-open" else "sentence-transformers/parallel-sentences-opus-100"
        process_pairs(code, raw, src, seen, per_eng, rng, shards, stats)
        shards.flush()
        (out / "stats" / f"{name}.json").write_text(json.dumps(dict(stats), indent=1))
        print(name, dict(stats), file=sys.stderr, flush=True)
        if a.push and stats["kept"]:
            push(out, name, token, stats["kept"])


if __name__ == "__main__":
    main()
