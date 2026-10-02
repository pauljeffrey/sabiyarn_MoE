"""v2 source registry: every dataset used for the large-scale AfriCOMET prompt build.

Skipped on purpose (with the evidence):
  CohereLabs/xP3x, bigscience/xP3mt   every file for these languages is derived from facebook/flores, so they
                                      only re-add FLORES plus prompt templates.
  lelapa/Inkuba-instruct              a bundle of MAFAND / SIB-200 / MasakhaNER etc. that is already covered.
  agentlans/bluesky                   no license, personal posts (handles / PII), only a few KB in yo/ig/ha.
  Aletheia-ng/GlotCC-V1               just a 21k-document Hausa subset of cis-lmu/GlotCC-V1.
Off by default (CC-BY-NC, and not needed for coverage): lelapa/Inkuba-Mono (`include_nc=True`).
"""

from __future__ import annotations

import gzip
import itertools
import json
import re
from typing import Iterator

from .sources import (
    Ctx, Doc, Source, _efik_audio_transcripts, _flores, _ibom_mt, _masakhanews, _sib200_ibom, _smol,
)
from .text_utils import normalize, split_sentences

CC0, ODC, BY, SA = "CC0-1.0", "ODC-By-1.0", "CC-BY-4.0", "CC-BY-SA-4.0"
WIKI_LIC = "CC-BY-SA-3.0 / GFDL"
BIBLE_LIC = "unspecified (Bible translation; Bible-society copyright not verified)"


def _dl(ctx: Ctx, repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", cache_dir=str(ctx.cache), token=ctx.token)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# ----------------------------------------------------------------------------------------------------------------
# loaders (each yields Docs: ordered sentences of ONE paragraph / chapter, so windows stay coherent)
# ----------------------------------------------------------------------------------------------------------------

def _paragraph_docs(text: str, min_chars: int = 40) -> Iterator[Doc]:
    for line in text.split("\n"):
        line = line.strip()
        if len(line) >= min_chars:
            sents = split_sentences(line)
            if sents:
                yield Doc(sents, "web")


def _long_paragraphs(text: str, max_paras: int = 2) -> Iterator[Doc]:
    """English: up to `max_paras` prose paragraphs (>=200 chars, >=2 sentences) per document, for diversity."""
    kept = 0
    for line in text.split("\n"):
        if kept >= max_paras:
            return
        if 200 <= len(line) <= 2500:
            sents = split_sentences(line)
            if len(sents) >= 2:
                kept += 1
                yield Doc(sents, "web")


def _parquet_rows(path: str, columns: list[str], batch_size: int = 2000) -> Iterator[dict]:
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def _hplt(label: str, filename: str, min_score: float = 6.0):
    """HPLT carries a language label per line (`seg_langs`) and a document quality score; both are used, and
    GlotLID then re-checks every sentence (about half of the lines HPLT calls Yoruba are not Yoruba)."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        for row in _parquet_rows(_dl(ctx, "HPLT/HPLT2.0_cleaned", filename), ["text", "seg_langs", "doc_scores", "pii"]):
            sc = row["doc_scores"]
            if row["pii"] or not len(sc) or sc[0] < min_score:
                continue
            lines, langs = row["text"].split("\n"), row["seg_langs"]
            if len(lines) != len(langs):
                continue
            for line, lg in zip(lines, langs):
                if lg == label and len(line) >= 40:
                    sents = split_sentences(line)
                    if sents:
                        yield Doc(sents, "web")
    return load


def _glotcc(code: str, min_prob: float = 0.9):
    def load(ctx: Ctx) -> Iterator[Doc]:
        for row in _parquet_rows(_dl(ctx, "cis-lmu/GlotCC-V1", f"v1.0/{code}-Latn/{code}-Latn_0.parquet"),
                                 ["content", "identification-prob"]):
            if row["identification-prob"] >= min_prob:
                yield from _paragraph_docs(row["content"])
    return load


def _mc4(filename: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        with gzip.open(_dl(ctx, "allenai/c4", filename), "rt", encoding="utf-8") as f:
            for line in f:
                yield from _paragraph_docs(json.loads(line)["text"])
    return load


def _wiki(code: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        for row in _parquet_rows(_dl(ctx, "wikimedia/wikipedia", f"20231101.{code}/train-00000-of-00001.parquet"), ["text"]):
            yield from _paragraph_docs(row["text"], min_chars=60)
    return load


def _wikianc(code: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        for split in ("train", "validation"):
            for row in _parquet_rows(_dl(ctx, "cyanic-selkie/wikianc", f"data/{code}/{split}.parquet"), ["paragraph_text"]):
                sents = split_sentences(row["paragraph_text"] or "")
                if sents:
                    yield Doc(sents, split)
    return load


def _bible(filename: str):
    """A chapter is one document; verses are its sentences (verse_key is BOOK.chapter.verse)."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        import pandas as pd
        df = pd.read_csv(_dl(ctx, "AfriSpeech/africa-corpus", filename)).dropna(subset=["local"])
        parts = df["verse_key"].astype(str).str.split(".", expand=True)
        df = df.assign(_ch=parts[0] + "." + parts[1], _v=pd.to_numeric(parts[2].str.extract(r"(\d+)")[0], errors="coerce"))
        for _, g in df.dropna(subset=["_v"]).groupby("_ch", sort=False):
            yield Doc([normalize(t) for t in g.sort_values("_v")["local"]], "bible")
    return load


def _opus(pair: str, lang: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        for split in ("train", "validation", "test"):
            try:
                path = _dl(ctx, "Helsinki-NLP/opus-100", f"{pair}/{split}-00000-of-00001.parquet")
            except Exception:
                continue
            for row in _parquet_rows(path, ["translation"]):
                t = (row["translation"] or {}).get(lang)
                if t:
                    yield Doc([normalize(t)], split)
    return load


def _inkuba(code: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        with open(_dl(ctx, "lelapa/Inkuba-Mono", f"{code}/data.txt"), encoding="utf-8") as f:
            while True:
                chunk = [l.strip() for l in itertools.islice(f, 5)]
                if not chunk:
                    return
                yield Doc([c for c in chunk if c], "web")
    return load


def _c4_en(i: int):
    def load(ctx: Ctx) -> Iterator[Doc]:
        with gzip.open(_dl(ctx, "allenai/c4", f"en/c4-train.{i:05d}-of-01024.json.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                yield from _long_paragraphs(json.loads(line)["text"])
    return load


def _wiki_en(i: int):
    def load(ctx: Ctx) -> Iterator[Doc]:
        for row in _parquet_rows(_dl(ctx, "wikimedia/wikipedia", f"20231101.en/train-{i:05d}-of-00041.parquet"), ["text"]):
            yield from _long_paragraphs(row["text"])
    return load


def _edu_en(i: int):
    def load(ctx: Ctx) -> Iterator[Doc]:
        for row in _parquet_rows(_dl(ctx, "HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet"), ["text", "int_score"]):
            if row["int_score"] >= 3:
                yield from _long_paragraphs(row["text"])
    return load


# ----------------------------------------------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------------------------------------------

def _web(name, lic, loader, lang_codes=None, prio=50, **kw) -> Source:
    return Source(name, lic, 1.0, loader, kind="web", lid=True, priority=prio, min_words=5, max_words=80,
                  require_terminal=True, **kw)


def build_registry(eng: dict | None = None, include_nc: bool = False) -> dict[str, list[Source]]:
    """`eng` = {"c4": n_shards, "wiki": n_shards, "edu": n_shards, "quota": windows_per_shard}."""
    R: dict[str, list[Source]] = {}

    def news(code, **kw):
        return Source(f"masakhane/masakhanews:{code}", "AFL-3.0", 1.0, _masakhanews(code), min_words=4, max_words=90,
                      require_terminal=True, kind="news", priority=12, **kw)

    def flores(variant):
        return Source(f"openlanguagedata/flores_plus:{variant}", SA, 1.0, _flores(variant), priority=10)

    def smol(cfg):
        return Source(f"google/smol:smoldoc__en_{cfg}", BY, 1.0, _smol(cfg), priority=11)

    def wiki(code):
        return Source(f"wikimedia/wikipedia:20231101.{code}", WIKI_LIC, 1.0, _wiki(code), kind="encyclopedia",
                      priority=20, min_words=4, max_words=90, require_terminal=True)

    def wikianc(code):
        return Source(f"cyanic-selkie/wikianc:{code}", SA, 1.0, _wikianc(code), kind="encyclopedia", priority=21,
                      min_words=5, max_words=90, require_terminal=True)

    def bible(fn):
        return Source(f"AfriSpeech/africa-corpus:{fn}", BIBLE_LIC, 1.0, _bible(fn), kind="bible", priority=30)

    def hplt(label, fn):
        return _web(f"HPLT/HPLT2.0_cleaned:{fn}", CC0, _hplt(label, fn), prio=50)

    def glotcc(code):
        return _web(f"cis-lmu/GlotCC-V1:{code}-Latn", CC0, _glotcc(code), prio=40)

    def mc4(fn):
        return _web(f"allenai/c4:{fn}", ODC, _mc4(fn), prio=60)

    def opus(pair, lang):
        return Source(f"Helsinki-NLP/opus-100:{pair}", "unknown (OPUS; mixed)", 1.0, _opus(pair, lang), priority=70,
                      min_words=4, max_words=60)

    R["pcm"] = [smol("pcm"), news("pcm"), wiki("pcm"), wikianc("pcm"), bible("Pidgin,_Nigerian_pcm_v2516.csv"), glotcc("pcm")]
    R["yor"] = [flores("yor_Latn"), smol("yo"), news("yor"), wiki("yo"), wikianc("yo"),
                *[bible(f"Yoruba_yor_v{v}.csv") for v in (207, 2754, 911)], glotcc("yor"),
                hplt("yor_Latn", "yor_Latn/train-00000-of-00001.parquet"),
                *[mc4(f"multilingual/c4-yo.tfrecord-0000{i}-of-00002.json.gz") for i in range(2)], opus("en-yo", "yo")]
    R["hau"] = [flores("hau_Latn"), smol("ha"), news("hau"), wiki("ha"), wikianc("ha"),
                *[bible(f"Hausa_hau_v{v}.csv") for v in (1614, 71)], glotcc("hau"),
                *[hplt("hau_Latn", f"hau_Latn/train-0000{i}-of-00003.parquet") for i in range(3)],
                *[mc4(f"multilingual/c4-ha.tfrecord-0000{i}-of-00008.json.gz") for i in range(8)], opus("en-ha", "ha")]
    R["ibo"] = [flores("ibo_Latn"), smol("ig"), news("ibo"), wiki("ig"), wikianc("ig"),
                *[bible(f"Igbo_ibo_v{v}.csv") for v in (1624, 77)], glotcc("ibo"),
                hplt("ibo_Latn", "ibo_Latn/train-00000-of-00001.parquet"),
                *[mc4(f"multilingual/c4-ig.tfrecord-0000{i}-of-00004.json.gz") for i in range(4)], opus("en-ig", "ig")]
    R["fon"] = [flores("fon_Latn"), smol("fon"), wiki("fon"), *[bible(f"Fon_fon_v{v}.csv") for v in (813, 817)],
                glotcc("fon"), hplt("fon_Latn", "fon_Latn/train-00000-of-00001.parquet")]
    R["ewe"] = [flores("ewe_Latn"), smol("ee"), wiki("ee"), wikianc("ee"), *[bible(f"Éwé_ewe_v{v}.csv") for v in (1613, 2259, 3306)],
                glotcc("ewe"), hplt("ewe_Latn", "ewe_Latn/train-00000-of-00001.parquet")]
    R["twi"] = [flores("twi_Latn_akua1239"), flores("twi_Latn_asan1239"), wiki("tw"), wikianc("tw"),
                *[bible(f"Twi_twi_v{v}.csv") for v in (1461, 1631, 1861, 3439, 3440)],
                glotcc("twi"), hplt("twi_Latn", "twi_Latn/train-00000-of-00001.parquet")]
    R["efi"] = [
        Source("howard-nlp/ibom-mt:eng-efi", "CC-BY-NC-4.0", 1.0, _ibom_mt("eng-efi", "efi"), priority=10),
        smol("efi"),
        Source("offiongbassey/efik_speech_dataset", BY, 1.0, _efik_audio_transcripts("offiongbassey/efik_speech_dataset"), priority=14),
        Source("offiongbassey/efik_audio_dataset", "unspecified", 1.0, _efik_audio_transcripts("offiongbassey/efik_audio_dataset"), priority=15),
        Source("howard-nlp/sib200-ibom:efi_Latn", SA, 1.0, _sib200_ibom("efi_Latn"), priority=13),
        *[bible(f"Efik_efi_v{v}.csv") for v in (2570, 4541)], glotcc("efi"),
    ]
    R["ibb"] = [
        Source("howard-nlp/ibom-mt:eng-ibb", "CC-BY-NC-4.0", 1.0, _ibom_mt("eng-ibb", "ibb"), priority=10),
        Source("howard-nlp/sib200-ibom:ibb_Latn", SA, 1.0, _sib200_ibom("ibb_Latn"), priority=13),
        bible("Ibibio_ibb_v2619.csv"),
    ]
    if include_nc:
        R["hau"].append(_web("lelapa/Inkuba-Mono:hau", "CC-BY-NC-4.0", _inkuba("hau"), prio=80))
        R["yor"].append(_web("lelapa/Inkuba-Mono:yor", "CC-BY-NC-4.0", _inkuba("yor"), prio=80))

    e = eng or {"c4": 0, "wiki": 0, "edu": 0, "quota": 0}
    q = e["quota"]
    R["eng"] = [news("eng")]
    R["eng"] += [Source(f"allenai/c4:en/c4-train.{i:05d}", ODC, 1.0, _c4_en(i), kind="web", priority=50,
                        min_words=7, max_words=80, require_terminal=True, stream=True, quota=q) for i in range(e["c4"])]
    R["eng"] += [Source(f"wikimedia/wikipedia:20231101.en/{i:05d}", WIKI_LIC, 1.0, _wiki_en(i), kind="encyclopedia", priority=20,
                        min_words=7, max_words=80, require_terminal=True, stream=True, quota=q) for i in range(e["wiki"])]
    R["eng"] += [Source(f"HuggingFaceFW/fineweb-edu:sample/10BT/{i:03d}", ODC, 1.0, _edu_en(i), kind="web", priority=40,
                        min_words=7, max_words=80, require_terminal=True, stream=True, quota=q) for i in range(e["edu"])]
    return R
