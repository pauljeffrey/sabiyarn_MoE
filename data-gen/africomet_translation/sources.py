"""Per-language source registry and loaders.

A loader yields `Doc`s: an ordered list of sentences that belong together (one article, one FLORES article,
one SMOL document, ...), so a window of 1-5 consecutive sentences is always coherent. Sources whose natural
unit is a single sentence are grouped back into their FLORES articles where that grouping can be recovered.

Everything is downloaded into `ctx.cache` (an isolated directory), never the shared HF cache, so cleanup is
one targeted delete. English web sources are streamed and leave nothing on disk.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from .text_utils import dedupe_key, normalize, split_sentences

PILOT_ENV = "AFRICOMET_STREAM_SCAN_LIMIT"


@dataclass
class Doc:
    sentences: list[str]
    split: str = ""


@dataclass
class Ctx:
    cache: Path
    token: Optional[str]
    seed: int = 42


@dataclass
class Source:
    name: str                       # shown in the `source_dataset` field
    license: str
    weight: float
    load: Callable[[Ctx], Iterator[Doc]]
    streaming: bool = False
    min_words: int = 3
    max_words: int = 120
    require_terminal: bool = False  # web text must look like prose; curated sentence sets need not


# ----------------------------------------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------------------------------------

def _masakhanews(code: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        from datasets import load_dataset
        ds = load_dataset("masakhane/masakhanews", code, cache_dir=str(ctx.cache), token=ctx.token)
        for split, rows in ds.items():
            for row in rows:
                sents = split_sentences(row["text"])
                if sents:
                    yield Doc(sents, split)
    return load


def _flores(variant: str):
    """openlanguagedata/flores_plus: one jsonl per language per split; consecutive ids in one `url` are one article."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        from huggingface_hub import hf_hub_download
        for split in ("dev", "devtest"):
            path = hf_hub_download("openlanguagedata/flores_plus", f"{split}/{variant}.jsonl", repo_type="dataset",
                                   cache_dir=str(ctx.cache), token=ctx.token)
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            rows.sort(key=lambda r: r["id"])
            for _, grp in itertools.groupby(rows, key=lambda r: r["url"]):
                yield Doc([normalize(r["text"]) for r in grp], split)
    return load


def _smol(cfg: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        from datasets import load_dataset
        ds = load_dataset("google/smol", f"smoldoc__en_{cfg}", split="train", cache_dir=str(ctx.cache), token=ctx.token)
        for row in ds:
            yield Doc([normalize(s) for s in row["trgs"]], "train")
    return load


def _flores_url_by_eng(ctx: Ctx) -> dict[str, str]:
    """English FLORES sentence -> article url, used to recover article boundaries for FLORES-aligned sets."""
    from huggingface_hub import hf_hub_download
    out: dict[str, str] = {}
    for split in ("dev", "devtest"):
        path = hf_hub_download("openlanguagedata/flores_plus", f"{split}/eng_Latn.jsonl", repo_type="dataset",
                               cache_dir=str(ctx.cache), token=ctx.token)
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[dedupe_key(r["text"])] = r["url"]
    return out


def _group_by_article(rows: list[tuple[int, str, str]], urls: dict[str, str], split: str) -> Iterator[Doc]:
    """rows = (id, english, target) in id order. Consecutive rows from one FLORES article become one Doc;
    an English sentence not found in FLORES stands alone."""
    keyed = [(urls.get(dedupe_key(eng)) or f"solo-{i}", tgt) for i, eng, tgt in rows]
    for _, grp in itertools.groupby(keyed, key=lambda kv: kv[0]):
        yield Doc([normalize(t) for _, t in grp], split)


def _ibom_mt(cfg: str, col: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        urls = _flores_url_by_eng(ctx)
        for split in ("dev", "test", "train"):
            path = hf_hub_download("howard-nlp/ibom-mt", f"{cfg}/{split}.csv", repo_type="dataset",
                                   cache_dir=str(ctx.cache), token=ctx.token)
            df = pd.read_csv(path).dropna(subset=["eng", col]).sort_values("id")
            rows = [(int(r.id), str(r.eng), str(getattr(r, col))) for r in df.itertuples()]
            if split == "train":    # ordered Wikipedia biography sentences, no FLORES article boundaries to recover
                yield Doc([normalize(t) for _, _, t in rows], split)
            else:
                yield from _group_by_article(rows, urls, split)
    return load


def _sib200_ibom(lang_cfg: str):
    def load(ctx: Ctx) -> Iterator[Doc]:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        urls = _flores_url_by_eng(ctx)

        def read(cfg: str) -> "pd.DataFrame":
            parts = []
            for split in ("train", "dev", "test"):
                p = hf_hub_download("howard-nlp/sib200-ibom", f"data/{cfg}/{split}.tsv", repo_type="dataset",
                                    cache_dir=str(ctx.cache), token=ctx.token)
                parts.append(pd.read_csv(p, sep="\t"))
            return pd.concat(parts).drop_duplicates("index_id").set_index("index_id")

        eng, tgt = read("eng_Latn"), read(lang_cfg)
        ids = sorted(set(eng.index) & set(tgt.index))
        rows = [(int(i), str(eng.loc[i, "text"]), str(tgt.loc[i, "text"])) for i in ids]
        yield from _group_by_article(rows, urls, "all")
    return load


def _efik_audio_transcripts(repo: str):
    """Only the `text` column is read, via parquet range requests -- the audio (~1 GB) is never downloaded."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem(token=ctx.token)
        texts: list[str] = []
        for f in sorted(fs.glob(f"datasets/{repo}/data/*.parquet")):
            with fs.open(f, "rb") as fh:
                texts += pq.read_table(fh, columns=["text"]).column("text").to_pylist()
        yield Doc([normalize(t) for t in texts if t], "train")     # original order is the narrative order
    return load


def _stream_limit() -> int:
    return int(os.environ.get(PILOT_ENV, "100000"))


def _wikipedia(ctx: Ctx) -> Iterator[Doc]:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=ctx.token)
    for art in itertools.islice(ds.shuffle(seed=ctx.seed, buffer_size=5000), _stream_limit()):
        kept = 0
        for line in art["text"].split("\n"):
            if kept >= 2:
                break
            if 200 <= len(line) <= 2500:
                sents = split_sentences(line)
                if len(sents) >= 2:
                    kept += 1
                    yield Doc(sents, "train")


def _fineweb_edu(ctx: Ctx) -> Iterator[Doc]:
    """Paragraphs are single-newline separated. Pages hard-wrapped at ~80 columns have no line >= 200 chars,
    so the length floor below also discards them (they would otherwise split sentences mid-line)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True, token=ctx.token)
    for row in itertools.islice(ds.shuffle(seed=ctx.seed, buffer_size=5000), _stream_limit()):
        if row.get("int_score", 0) < 3:
            continue
        kept = 0
        for line in row["text"].split("\n"):
            if kept >= 2:
                break
            if 200 <= len(line) <= 2500:
                sents = split_sentences(line)
                if len(sents) >= 2:
                    kept += 1
                    yield Doc(sents, "train")


# ----------------------------------------------------------------------------------------------------------------
# Registry. ORDER MATTERS: sentences are de-duplicated across a language's sources in this order, so list the
# preferred (first-class) source first.
# ----------------------------------------------------------------------------------------------------------------

FLORES_LIC = "CC-BY-SA-4.0"
SMOL_LIC = "CC-BY-4.0"
NEWS_LIC = "AFL-3.0"


def _news(code: str, w: float = 1.0, min_words: int = 4) -> Source:
    return Source(f"masakhane/masakhanews:{code}", NEWS_LIC, w, _masakhanews(code), min_words=min_words,
                  max_words=90, require_terminal=True)


def _fl(variant: str, w: float = 1.0) -> Source:
    return Source(f"openlanguagedata/flores_plus:{variant}", FLORES_LIC, w, _flores(variant))


def _sm(cfg: str, w: float = 1.0) -> Source:
    return Source(f"google/smol:smoldoc__en_{cfg}", SMOL_LIC, w, _smol(cfg))


SOURCES: dict[str, list[Source]] = {
    "eng": [
        Source("wikimedia/wikipedia:20231101.en", "CC-BY-SA-3.0 / GFDL", 0.4, _wikipedia, streaming=True,
               min_words=7, max_words=80, require_terminal=True),
        Source("HuggingFaceFW/fineweb-edu:sample-10BT", "ODC-By-1.0", 0.4, _fineweb_edu, streaming=True,
               min_words=7, max_words=80, require_terminal=True),
        _news("eng", 0.2, min_words=6),
    ],
    "pcm": [_news("pcm"), _sm("pcm")],
    "yor": [_news("yor"), _fl("yor_Latn"), _sm("yo")],
    "hau": [_news("hau"), _fl("hau_Latn"), _sm("ha")],
    "ibo": [_news("ibo"), _fl("ibo_Latn"), _sm("ig")],
    "fon": [_fl("fon_Latn"), _sm("fon")],
    "ewe": [_fl("ewe_Latn"), _sm("ee")],
    "twi": [_fl("twi_Latn_akua1239"), _fl("twi_Latn_asan1239")],
    "efi": [
        Source("howard-nlp/ibom-mt:eng-efi", "CC-BY-NC-4.0", 1.0, _ibom_mt("eng-efi", "efi")),
        _sm("efi"),
        Source("offiongbassey/efik_speech_dataset", "CC-BY-4.0", 1.0, _efik_audio_transcripts("offiongbassey/efik_speech_dataset")),
        Source("offiongbassey/efik_audio_dataset", "unspecified", 0.3, _efik_audio_transcripts("offiongbassey/efik_audio_dataset")),
        Source("howard-nlp/sib200-ibom:efi_Latn", "CC-BY-SA-4.0", 0.3, _sib200_ibom("efi_Latn")),
    ],
    "ibb": [
        Source("howard-nlp/ibom-mt:eng-ibb", "CC-BY-NC-4.0", 1.0, _ibom_mt("eng-ibb", "ibb")),
        Source("howard-nlp/sib200-ibom:ibb_Latn", "CC-BY-SA-4.0", 0.3, _sib200_ibom("ibb_Latn")),
    ],
}

LANGUAGE_ORDER = ["pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "ibb", "eng"]
# Languages the AfriCOMET-1.1 encoder (Davlan/afro-xlmr-large-76L) lists. Ibibio is NOT among them.
AFRICOMET_SUPPORTED = {"eng", "pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi"}
