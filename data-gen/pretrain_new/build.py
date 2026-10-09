"""Collect, clean, deduplicate and push the monolingual pretrain set (`text`, `lang`, `score`).

One process, one language at a time, constant memory: sources are streamed, rows are written to ~50k-row parquet
shards, each finished language is uploaded and its local files are deleted.

    nice -n 10 python -m pretrain_new.build --langs wol tsn --out OUT --cap 5000000 [--push]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from africomet_translation import lid
from pretrain_new import docclean, sources

REPO = "BeardedMonster/pretrain-new"
SCHEMA = pa.schema([("text", pa.string()), ("lang", pa.string()), ("score", pa.float32())])
SHARD_ROWS = 50_000
LID_SOURCES = {"mbazaNLP_kinyarwanda_monolingual_v01.1", "dsfsi_PuoData", "dsfsi_vukuzenzele-monolingual", "allenai_madlad-400", "allenai_c4-multilingual", "AfriSpeech_african-corpus-jw", "HuggingFaceFW_finepdfs", "AfriSpeech_africa-corpus", "uctnlp_mzansi-text-deduplicated", "omarkamali_wikipedia-monthly",
               "CircuitNotion_kinyarwanda_corpus", "ik_twi-combined-corpus", "ghananlpcommunity_ghana-topics-jw", "Lahad_wolof_only-dataset",
               "suru8-ai_malawi-chichewa-tumbuka-corpus", "ogaufi_setswana_new", "AnalyticsIntelligence_pidgin_corpus",
               "abukar123_somali-news-dataset", "google_WaxalNLP", "Africanvoice_naija_african_voices",
               "Professor_wolof-speech-data", "offiongbassey_efik_speech_dataset",
               "essogbe_mms-tts-fon-accentfine"}          # sources without a per-segment LID of their own get checked here
_rng = random.Random(0)


def lid_pass(text: str, lang: str, need: float = 0.6, n: int = 6) -> bool:
    lines = [l for l in text.split("\n") if len(l) >= 40]
    if not lines:
        return False
    if len(lines) > n:
        step = len(lines) / n
        lines = [lines[int(i * step)] for i in range(n)]
    ok = lid.verify(lines, lang)
    return sum(ok) / len(ok) >= need


class Writer:
    def __init__(self, out: Path, lang: str, start_part: int = 0):
        self.dir = out / "data" / lang
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[tuple[str, float]] = []
        self.lang, self.n_shards, self.n = lang, start_part, 0

    def add(self, text: str, score: float):
        self.rows.append((text, score))
        self.n += 1
        if len(self.rows) >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        tb = pa.table({"text": [r[0] for r in self.rows], "lang": [self.lang] * len(self.rows),
                       "score": pa.array([r[1] for r in self.rows], pa.float32())}, schema=SCHEMA)
        pq.write_table(tb, self.dir / f"part-{self.n_shards:05d}.parquet", compression="zstd")
        self.n_shards += 1
        self.rows = []


PASSES = {
    "base": lambda lang, token, scratch: _chain(sources.hplt3(lang), sources.finepdfs(lang, token, scratch)),
    "bible": lambda lang, token, scratch: sources.africa_corpus(lang, token, scratch),
    "extra": lambda lang, token, scratch: sources.extra(lang, token, scratch),
    "jw": lambda lang, token, scratch: sources.jw_corpus(lang, token, scratch),
    "legacy": lambda lang, token, scratch: sources.legacy(lang, token, scratch),
    "legacy2": lambda lang, token, scratch: sources.legacy2(lang, token, scratch),
    "speech": lambda lang, token, scratch: sources.speech(lang, token, scratch),
}


def _chain(*its):
    for it in its:
        yield from it


def run_language(lang: str, out: Path, cap: int, token: str, scratch: Path, push: bool, log, pass_name: str = "base",
                exact_only: bool = False) -> dict:
    """One pass over one source group. The dedupe tables and shard counter persist in out/state so that later passes
    (other source groups) deduplicate against everything already kept for the language."""
    state_dir = out / "state"
    state_dir.mkdir(exist_ok=True)
    mf = state_dir / f"{lang}.json"
    prev = json.loads(mf.read_text()) if mf.exists() else {"kept": 0, "n_shards": 0, "chars": 0}
    ns = 2.0 if exact_only else 0.5          # 2.0 can never be reached: only exact-duplicate documents are dropped
    dd = docclean.Deduper.load(state_dir / f"{lang}.npz", near_share=ns) if (state_dir / f"{lang}.npz").exists() else docclean.Deduper(near_share=ns)
    d0e, d0n = dd.dropped_exact, dd.dropped_near
    stats = {"lang": lang, "pass": pass_name, "seen": 0, "cleaned_out": 0, "lid_out": 0, "dup_exact": 0, "dup_near": 0,
             "kept": 0, "by_source": {}, "chars": 0}
    w = Writer(out, lang, prev["n_shards"])
    t0 = time.time()
    room = cap - prev["kept"]
    if room > 0:
        for text, score, src in PASSES[pass_name](lang, token, scratch):
            stats["seen"] += 1
            if stats["seen"] % 20000 == 0:
                log(f"{lang}/{pass_name}: seen {stats['seen']:,} kept {stats['kept']:,} ({time.time()-t0:.0f}s)")
            t = docclean.clean_document(text, lang)
            if t is None:
                stats["cleaned_out"] += 1
                continue
            if src in LID_SOURCES and not lid_pass(t, lang):
                stats["lid_out"] += 1
                continue
            if dd.is_duplicate(t):
                continue
            w.add(t, score)
            stats["kept"] += 1
            stats["chars"] += len(t)
            stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
            if stats["kept"] >= room:
                break
    w.flush()
    stats["dup_exact"], stats["dup_near"] = dd.dropped_exact - d0e, dd.dropped_near - d0n
    stats["seconds"] = round(time.time() - t0)
    dd.save(state_dir / f"{lang}.npz")
    mf.write_text(json.dumps({"kept": prev["kept"] + stats["kept"], "n_shards": w.n_shards, "chars": prev["chars"] + stats["chars"]}))
    (out / "stats").mkdir(exist_ok=True)
    (out / "stats" / f"{lang}.{pass_name}.json").write_text(json.dumps(stats, indent=1))
    if push and stats["kept"]:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(w.dir), path_in_repo=f"data/{lang}",
                          commit_message=f"{lang} [{pass_name}]: {stats['kept']:,} documents")
        shutil.rmtree(w.dir, ignore_errors=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=5_000_000)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--exact-only", action="store_true", help="no near-duplicate removal (languages under 100k documents)")
    ap.add_argument("--pass", dest="pass_name", default="base", choices=list(PASSES))
    a = ap.parse_args(argv)
    from africomet_translation.build import hf_token
    token = hf_token()
    os.environ["HF_TOKEN"] = token
    out = Path(a.out)
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, file=sys.stderr, flush=True)
    for lang in a.langs:
        if (out / "stats" / f"{lang}.{a.pass_name}.json").exists():
            log(f"{lang}: already done")
            continue
        s = run_language(lang, out, a.cap, token, scratch, a.push, log, a.pass_name, a.exact_only)
        log(f"{lang}: DONE kept {s['kept']:,} / seen {s['seen']:,}  dup {s['dup_exact']:,}+{s['dup_near']:,}  "
            f"clean-out {s['cleaned_out']:,}  lid-out {s['lid_out']:,}  {s['chars']/1e6:.0f}M chars  {s['seconds']}s")


if __name__ == "__main__":
    main()
