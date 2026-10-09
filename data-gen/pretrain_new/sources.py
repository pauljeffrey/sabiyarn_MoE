"""Streaming monolingual sources. Nothing is stored: HPLT shards are piped through zstd, parquet files are
downloaded one at a time and deleted as soon as they have been read."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import requests

# output code -> HPLT 3.0 / FinePDFs label
LABEL = {"hau": "hau_Latn", "ibo": "ibo_Latn", "yor": "yor_Latn", "twi": "twi_Latn", "ewe": "ewe_Latn", "fon": "fon_Latn",
         "kin": "kin_Latn", "sna": "sna_Latn", "zul": "zul_Latn", "swh": "swh_Latn", "orm": "gaz_Latn", "wol": "wol_Latn",
         "xho": "xho_Latn", "tsn": "tsn_Latn", "nya": "nya_Latn", "afr": "afr_Latn", "som": "som_Latn", "sot": "sot_Latn",
         "pcm": "pcm_Latn", "efi": "efi_Latn", "urh": "urh_Latn"}
# labels HPLT's own segment-level LID may use for the same language
HPLT_OK = {"twi_Latn": {"twi_Latn", "aka_Latn"}}

# Doc is (text, score on a 0-10 scale, source slug)
Doc = tuple[str, float, str]


def hplt3(lang: str, min_bin: int = 5) -> Iterator[Doc]:
    """HPLT 3.0 documents, best quality bin first. `score` is HPLT's document quality score (WDS, 0-10)."""
    label = LABEL.get(lang)
    if not label:
        return
    ok = HPLT_OK.get(label, {label})
    r = requests.get(f"https://data.hplt-project.org/three/sorted/{label}.map", timeout=60)
    if r.status_code != 200:
        return
    urls = [u for u in r.text.split() if u.startswith("http")]

    def key(u):
        b, n = u.rsplit("/", 1)[1].split(".")[0].split("_")
        return (-int(b), int(n))

    for url in sorted(urls, key=key):
        if -key(url)[0] < min_bin:
            continue
        p = subprocess.Popen(f"curl -sSL --fail --retry 4 --retry-delay 3 --max-time 7200 '{url}' | zstd -dc",
                             shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                             errors="replace", bufsize=1 << 20)
        try:
            for line in p.stdout:
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("filter") != "keep" or j.get("pii"):
                    continue
                segs = j.get("seg_langs") or []
                if len(segs) < 3 or sum(s in ok for s in segs) / len(segs) < 0.8:
                    continue
                sc = j.get("doc_scores") or [0]
                yield j["text"], float(sc[0]), "HPLT_HPLT3.0"
        finally:
            p.kill()
            p.wait()


def finepdfs(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    """FinePDFs documents (created 2025-09). Each parquet file is deleted after it has been read."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    label = LABEL.get(lang)
    if not label:
        return
    api = HfApi(token=hf_token)
    try:
        files = [f.path for sp in ("train", "test")
                 for f in api.list_repo_tree("HuggingFaceFW/finepdfs", path_in_repo=f"data/{label}/{sp}", repo_type="dataset")
                 if f.path.endswith(".parquet")]
    except Exception:
        return
    for path in files:
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            local = hf_hub_download("HuggingFaceFW/finepdfs", path, repo_type="dataset", token=hf_token, cache_dir=str(cache))
            pf = pq.ParquetFile(local)
            cols = [c for c in ("text", "full_doc_lid_score", "ocr_quality_scores", "is_truncated") if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=1000, columns=cols):
                d = batch.to_pydict()
                for i, t in enumerate(d["text"]):
                    lid = d["full_doc_lid_score"][i] if "full_doc_lid_score" in d else 0.9
                    if (lid or 0) < 0.9:
                        continue
                    ocr = d.get("ocr_quality_scores", [None] * len(d["text"]))[i]
                    sc = min(10 * (lid or 0), 10 * (sum(ocr) / len(ocr)) if ocr else 10)
                    yield t, round(sc, 2), "HuggingFaceFW_finepdfs"
        finally:
            shutil.rmtree(cache, ignore_errors=True)


# ISO codes as they appear in AfriSpeech/africa-corpus file names (<Name>_<iso>_v<id>.csv)
AFRICA_CORPUS_ISO = {"efi": {"efi"}, "ibb": {"ibb"}, "urh": {"urh"}, "fon": {"fon"}, "pcm": {"pcm"}, "wol": {"wol", "wof"},
                     "twi": {"twi", "aka"}, "ewe": {"ewe"}, "hau": {"hau"}, "ibo": {"ibo"}, "yor": {"yor"}, "kin": {"kin"},
                     "sna": {"sna"}, "zul": {"zul"}, "swh": {"swh"}, "orm": {"gaz"}, "tsn": {"tsn"}, "nya": {"nya"},
                     "xho": {"xho"}, "sot": {"sot"}, "som": {"som"}, "afr": {"afr"}}


def africa_corpus(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    """AfriSpeech/africa-corpus (Bible text, one CSV per translation). One document per chapter, one verse per line,
    so near-duplicate detection can see shared verses across translations."""
    import csv
    from huggingface_hub import HfApi, hf_hub_download

    isos = AFRICA_CORPUS_ISO.get(lang)
    if not isos:
        return
    api = HfApi(token=hf_token)
    files = [f.path for f in api.list_repo_tree("AfriSpeech/africa-corpus", repo_type="dataset")
             if f.path.endswith(".csv") and "/" not in f.path and not f.path.startswith("english")
             and "Arabic_script" not in f.path and f.path.rsplit("_v", 1)[0].rsplit("_", 1)[-1] in isos]
    csv.field_size_limit(1 << 24)
    for path in sorted(files):
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            local = hf_hub_download("AfriSpeech/africa-corpus", path, repo_type="dataset", token=hf_token, cache_dir=str(cache))
            chapter, verses = None, []
            with open(local, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    ch = row["verse_key"].rsplit(".", 1)[0]
                    if ch != chapter and verses:
                        yield "\n".join(verses), 7.0, "AfriSpeech_africa-corpus"
                        verses = []
                    chapter = ch
                    v = (row.get("local") or "").replace(";.", ".").strip()
                    if v:
                        verses.append(v)
            if verses:
                yield "\n".join(verses), 7.0, "AfriSpeech_africa-corpus"
        finally:
            shutil.rmtree(cache, ignore_errors=True)


WIKI_CODE = {"hau": "ha", "ibo": "ig", "yor": "yo", "twi": "tw", "ewe": "ee", "fon": "fon", "kin": "rw", "sna": "sn",
             "zul": "zu", "swh": "sw", "orm": "om", "wol": "wo", "xho": "xh", "tsn": "tn", "nya": "ny", "afr": "af",
             "som": "so", "sot": "st", "pcm": "pcm"}
WIKI_ALLOWED = {"afr", "zul", "sna", "nya", "xho", "sot", "kin", "tsn", "orm", "wol", "som", "swh"}   # the newly added languages
MZANSI_CODE = {"zul": "zul", "xho": "xho", "tsn": "tsn", "sot": "sot", "afr": "afr"}


def _parquet_stream(repo: str, files: list[str], cols: list[str], hf_token: str, scratch: Path):
    """Yield row dicts from parquet files, downloading and deleting one file at a time."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for path in files:
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            local = hf_hub_download(repo, path, repo_type="dataset", token=hf_token, cache_dir=str(cache))
            for batch in pq.ParquetFile(local).iter_batches(batch_size=2000, columns=cols):
                d = batch.to_pydict()
                for i in range(len(d[cols[0]])):
                    yield {c: d[c][i] for c in cols}
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def _tree(repo: str, path: str, hf_token: str, suffix: str = ".parquet") -> list[str]:
    from huggingface_hub import HfApi
    try:
        return sorted(f.path for f in HfApi(token=hf_token).list_repo_tree(repo, path_in_repo=path, repo_type="dataset")
                      if f.path.endswith(suffix))
    except Exception:
        return []


def extra(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    """Other post-May-2025 monolingual datasets that survived vetting (see README)."""
    # omarkamali/wikipedia-monthly (repo created 2025-07-13): allowed by the user for the newly added languages only
    code = WIKI_CODE.get(lang)
    if code and lang in WIKI_ALLOWED:
        files = _tree("omarkamali/wikipedia-monthly", f"20261001/{code}/train", hf_token)
        for r in _parquet_stream("omarkamali/wikipedia-monthly", files, ["text"], hf_token, scratch):
            if len(r["text"] or "") >= 500:
                yield r["text"], 6.5, "omarkamali_wikipedia-monthly"
    yield from extra_small(lang, hf_token, scratch)
    # khaledyusuf44/somaliweb-v1 (created 2026-04): deduplicated Somali web + news + Wikipedia
    if lang == "som":
        files = _tree("khaledyusuf44/somaliweb-v1", "", hf_token) or []
        files = [f for f in files if "/" not in f or f.startswith("data/")]
        for r in _parquet_stream("khaledyusuf44/somaliweb-v1", files, ["text", "quality_score"], hf_token, scratch):
            q = float(r["quality_score"] or 0)
            yield r["text"], round(10 * q, 2), "khaledyusuf44_somaliweb-v1"
    # uctnlp/mzansi-text-deduplicated (created 2026-08): South African documents, labels are noisy so LID-checked
    mz = MZANSI_CODE.get(lang)
    if mz:
        files = _tree("uctnlp/mzansi-text-deduplicated", "data", hf_token)
        files = [f for f in files if "train-" in f]
        for r in _parquet_stream("uctnlp/mzansi-text-deduplicated", files, ["text", "lang"], hf_token, scratch):
            if r["lang"] == mz:
                yield r["text"], 6.0, "uctnlp_mzansi-text-deduplicated"


# --------------------------------------------------------------------------------------------------------------
# small post-2025-08 datasets that survived vetting. Most are sentence-level, so consecutive rows are packed into
# pseudo-documents of PACK_N lines (the cleaner then enforces the length floor).
PACK_N = 12


def _stream(repo: str, config, split: str, col: str, hf_token: str, where=None):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True, token=hf_token)
    for r in ds:
        if where and not where(r):
            continue
        v = r.get(col)
        if isinstance(v, str) and v.strip():
            yield v.strip()


def _pack(lines, n: int = PACK_N):
    buf = []
    for ln in lines:
        buf.append(ln)
        if len(buf) >= n:
            yield "\n".join(buf)
            buf = []
    if buf:
        yield "\n".join(buf)


def extra_small(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    if lang == "som":      # abukar123/somali-news-dataset (created 2026-09): scraped Somali news articles
        import csv
        from huggingface_hub import hf_hub_download
        csv.field_size_limit(1 << 24)
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            p = hf_hub_download("abukar123/somali-news-dataset", "combined_cleaned.csv", repo_type="dataset", token=hf_token, cache_dir=str(cache))
            with open(p, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    t = ((r.get("headline") or "") + "\n" + (r.get("body") or "")).strip()
                    yield t, 6.5, "abukar123_somali-news-dataset"
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    if lang == "kin":      # CircuitNotion/kinyarwanda_corpus (created 2026-05): news sentences and comments
        for d in _pack(_stream("CircuitNotion/kinyarwanda_corpus", None, "train", "text", hf_token)):
            yield d, 5.5, "CircuitNotion_kinyarwanda_corpus"
    if lang == "twi":      # ik/twi-combined-corpus (2026-04, dirty: filtered hard) and ghana-topics JW split (2026-09)
        for d in _pack(_stream("ik/twi-combined-corpus", None, "train", "text", hf_token)):
            yield d, 5.0, "ik_twi-combined-corpus"
        for d in _pack(_stream("ghananlpcommunity/ghana-topics", "Twi_twi", "jw", "local", hf_token), 8):
            yield d, 6.0, "ghananlpcommunity_ghana-topics-jw"
    if lang == "ewe":
        for d in _pack(_stream("ghananlpcommunity/ghana-topics", "Éwé_ewe", "jw", "local", hf_token), 8):
            yield d, 6.0, "ghananlpcommunity_ghana-topics-jw"
    if lang == "wol":      # Lahad/wolof_only-dataset (2026-01)
        for d in _pack(_stream("Lahad/wolof_only-dataset", None, "train", "texte_wolof", hf_token)):
            yield d, 5.5, "Lahad_wolof_only-dataset"
    if lang == "nya":      # suru8-ai/malawi-chichewa-tumbuka-corpus (2026-07): Bible, Chichewa rows only
        for d in _pack(_stream("suru8-ai/malawi-chichewa-tumbuka-corpus", None, "train", "text", hf_token,
                               where=lambda r: r.get("language_code") == "nya"), 20):
            yield d, 6.0, "suru8-ai_malawi-chichewa-tumbuka-corpus"
    if lang == "tsn":      # ogaufi/setswana_new (2026-03): Daily News (Dikgang), whole articles
        for sp in ("train", "validation", "test"):
            for t in _stream("ogaufi/setswana_new", None, sp, "text", hf_token):
                yield t, 6.0, "ogaufi_setswana_new"
    if lang == "pcm":      # AnalyticsIntelligence/pidgin_corpus (2025-10): BBC Pidgin news, sentence level
        for d in _pack(_stream("AnalyticsIntelligence/pidgin_corpus", None, "train", "text", hf_token), 15):
            yield d, 6.0, "AnalyticsIntelligence_pidgin_corpus"


# --------------------------------------------------------------------------------------------------------------
# speech-transcript text (low-resource languages only). The audio is never fetched: parquet files are read with
# column projection over HTTP range requests, so only the text column crosses the network.
def _remote_text(repo: str, files: list[str], col: str, hf_token: str):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=hf_token)
    import time
    for path in files:
        for attempt in range(3):                                                # network resets happen; retry the whole shard
            got = []
            try:
                with fs.open(f"datasets/{repo}/{path}", "rb") as fh:
                    for batch in pq.ParquetFile(fh).iter_batches(batch_size=4000, columns=[col]):
                        got += [t.strip() for t in batch.to_pydict()[col] if isinstance(t, str) and t.strip()]
                yield from got
                break
            except Exception:
                time.sleep(5 * (attempt + 1))


def _files(repo: str, hf_token: str, pattern: str) -> list[str]:
    import re
    import time
    from huggingface_hub import HfApi
    for attempt in range(4):
        try:
            return sorted(f.path for f in HfApi(token=hf_token).list_repo_tree(repo, repo_type="dataset", recursive=True)
                          if hasattr(f, "size") and re.search(pattern, f.path))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return []


def _speech_all(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    """Transcripts of human-recorded speech datasets created after 2025-08 (see README for the vetting)."""
    W = "google/WaxalNLP"
    waxal = {"twi": [("ASR/aka", "transcription"), ("TTS/twi", "text")], "ewe": [("ASR/ewe", "transcription"), ("TTS/ewe", "text")],
             "orm": [("ASR/orm", "transcription")], "sna": [("ASR/sna", "transcription")], "pcm": [("TTS/pcm", "text")]}
    for sub, col in waxal.get(lang, []):
        files = [f for f in _files(W, hf_token, rf"data/{sub}/") if "unlabeled" not in f]
        for d in _pack(_remote_text(W, files, col, hf_token), 10):
            yield d, 6.5, "google_WaxalNLP"
    if lang == "pcm":      # Africanvoice/naija_african_voices: human-transcribed spontaneous Pidgin (lowercased)
        r = "Africanvoice/naija_african_voices"
        for d in _pack(_remote_text(r, _files(r, hf_token, r"data/.*\.parquet"), "transcript", hf_token), 10):
            yield d, 4.5, "Africanvoice_naija_african_voices"
    # b1n1yam/Sagalee (CC-BY-NC) and the LDS-talk sentences (CC-BY-NC) were removed: non-commercial licences.
    if lang == "wol":      # Professor/wolof-speech-data manifest
        r = "Professor/wolof-speech-data"
        for d in _pack(_remote_text(r, ["manifest.parquet"], "text", hf_token), 10):
            yield d, 5.5, "Professor_wolof-speech-data"
    if lang == "efi":      # offiongbassey/efik_speech_dataset: manually transcribed, reviewed
        r = "offiongbassey/efik_speech_dataset"
        for d in _pack(_remote_text(r, _files(r, hf_token, r"data/.*\.parquet"), "text", hf_token), 10):
            yield d, 7.0, "offiongbassey_efik_speech_dataset"
    if lang == "fon":      # essogbe/mms-tts-fon-accentfine-part-1..8: tone-marked Fon sentences
        for i in range(1, 9):
            r = f"essogbe/mms-tts-fon-accentfine-part-{i}"
            for d in _pack(_remote_text(r, _files(r, hf_token, r"data/.*\.parquet"), "text", hf_token), 10):
                yield d, 5.0, "essogbe_mms-tts-fon-accentfine"


# AfriSpeech/african-corpus-jw (created 2026-09, Watch Tower publications; the user explicitly allowed this text).
# File names are <Name>_<iso>_v1.csv with 2-letter codes for the larger languages; rows are paragraphs, verse_key =
# "<document id>#<paragraph>", so consecutive rows with the same id form one document.
JW_FILES = {"twi": ["Twi_tw_v1.csv"], "ewe": ["Ewe_ee_v1.csv"], "hau": ["Hausa_ha_v1.csv"], "ibo": ["Igbo_ig_v1.csv"],
            "yor": ["Yoruba_yo_v1.csv"], "wol": ["Wolof_wo_v1.csv"], "efi": ["Efik_efi_v1.csv"], "fon": ["Fon_fon_v1.csv"],
            "kin": ["Kinyarwanda_rw_v1.csv"], "sna": ["Shona_sn_v1.csv"], "zul": ["Zulu_zu_v1.csv"], "swh": ["Swahili_sw_v1.csv"],
            "orm": ["Oromo_om_v1.csv"], "tsn": ["Setswana_tn_v1.csv"], "nya": ["Chichewa_ny_v1.csv", "Cinyanja_nya_v1.csv"],
            "xho": ["Xhosa_xh_v1.csv"], "sot": ["Sesotho_Lesotho_st_v1.csv", "Sesotho_South_Africa_st_v1.csv"]}


def jw_corpus(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    import csv
    from huggingface_hub import hf_hub_download
    csv.field_size_limit(1 << 24)
    for name in JW_FILES.get(lang, []):
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            p = hf_hub_download("AfriSpeech/african-corpus-jw", name, repo_type="dataset", token=hf_token, cache_dir=str(cache))
            cur, paras = None, []
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    doc = row["verse_key"].split("#", 1)[0]
                    if doc != cur and paras:
                        yield "\n".join(paras), 6.0, "AfriSpeech_african-corpus-jw"
                        paras = []
                    cur = doc
                    t = (row.get("local") or "").strip()
                    if t:
                        paras.append(t)
            if paras:
                yield "\n".join(paras), 6.0, "AfriSpeech_african-corpus-jw"
        finally:
            shutil.rmtree(cache, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------------
# Older web corpora, allowed for the 12 newly added languages only (cutoff 2021-05): WURA (2023, Apache-2.0),
# FineWeb-2 (2024, ODC-By), GlotCC-V1 (2024, CC0), HPLT 2.0 cleaned (2024, CC0). Licences are all commercial-friendly.
LEGACY_LANGS = {"afr", "zul", "sna", "nya", "xho", "sot", "kin", "tsn", "orm", "wol", "som", "swh"}
WURA_CODE = {"afr": "afr", "zul": "zul", "sna": "sna", "nya": "nya", "xho": "xho", "sot": "sot", "kin": "kin", "orm": "orm",
             "som": "som", "swh": "swa"}                      # no tsn / wol in WURA
GLOTCC_CODE = {"afr": "afr-Latn", "zul": "zul-Latn", "sna": "sna-Latn", "nya": "nya-Latn", "xho": "xho-Latn", "sot": "sot-Latn",
               "kin": "kin-Latn", "tsn": "tsn-Latn", "orm": "gaz-Latn", "wol": "wol-Latn", "som": "som-Latn", "swh": "swh-Latn"}


def _long_enough(path: str, col: str, keep_fraction: float) -> int:
    """Length (characters) such that about `keep_fraction` of the documents in a local parquet file are at least that long."""
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    if keep_fraction >= 1.0:
        return 0
    lens = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=[col]):
        lens.append(pc.utf8_length(b.column(0)).to_numpy(zero_copy_only=False).astype(np.int32))
    arr = np.concatenate(lens) if lens else np.zeros(1, np.int32)
    return int(np.quantile(arr, 1.0 - keep_fraction))


def _big_parquet(repo: str, files: list[str], cols: list[str], text_col: str, hf_token: str, scratch: Path, room: int,
                 keep_filter=None):
    """Stream rows of large parquet files one file at a time (download, read, delete). When the source holds clearly more
    documents than there is room for, only documents above a length threshold are yielded (longest first)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    total_bytes = None
    thr = None
    for path in files:
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            local = hf_hub_download(repo, path, repo_type="dataset", token=hf_token, cache_dir=str(cache))
            pf = pq.ParquetFile(local)
            if thr is None:                                   # decided once per source, from its first file
                rows_first = pf.metadata.num_rows
                est_total = rows_first * max(1, len(files))
                frac = min(1.0, 1.3 * room / max(1, est_total))
                thr = _long_enough(local, text_col, frac)
            for batch in pf.iter_batches(batch_size=2000, columns=cols):
                d = batch.to_pydict()
                for i in range(len(d[text_col])):
                    t = d[text_col][i]
                    if not isinstance(t, str) or len(t) < thr:
                        continue
                    row = {c: d[c][i] for c in cols}
                    if keep_filter and not keep_filter(row):
                        continue
                    yield row
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def legacy(lang: str, hf_token: str, scratch: Path, room: int = 8_000_000) -> Iterator[Doc]:
    if lang not in LEGACY_LANGS:
        return
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=hf_token)

    def ls(repo, path, suffix=".parquet"):
        try:
            return sorted(f.path for f in api.list_repo_tree(repo, path_in_repo=path, repo_type="dataset", recursive=True)
                          if f.path.endswith(suffix))
        except Exception:
            return []

    # WURA: curated African news/web documents, one jsonl per language
    wc = WURA_CODE.get(lang)
    if wc:
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            p = hf_hub_download("castorini/wura", f"documents-v1.0/train/{wc}.jsonl", repo_type="dataset", token=hf_token, cache_dir=str(cache))
            import json
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        j = json.loads(line)
                    except ValueError:
                        continue
                    yield ((j.get("headline") or "") + "\n" + (j.get("content") or "")).strip(), 7.0, "castorini_wura"
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    # FineWeb-2
    label = LABEL.get(lang)
    files = ls("HuggingFaceFW/fineweb-2", f"data/{label}/train")
    for r in _big_parquet("HuggingFaceFW/fineweb-2", files, ["text", "language_score"], "text", hf_token, scratch, room,
                          keep_filter=lambda r: (r["language_score"] or 0) >= 0.9):
        yield r["text"], round(10 * (r["language_score"] or 0), 2), "HuggingFaceFW_fineweb-2"
    # GlotCC-V1: only documents without quality warnings
    gc = GLOTCC_CODE.get(lang)
    if gc:
        files = ls("cis-lmu/GlotCC-V1", f"v1.0/{gc}")
        for r in _big_parquet("cis-lmu/GlotCC-V1", files, ["content", "quality-warnings", "identification-prob"], "content", hf_token,
                              scratch, room, keep_filter=lambda r: not r["quality-warnings"] and (r["identification-prob"] or 0) >= 0.9):
            yield r["content"], round(10 * (r["identification-prob"] or 0), 2), "cis-lmu_GlotCC-V1"
    # HPLT 2.0 cleaned
    files = ls("HPLT/HPLT2.0_cleaned", label)
    for r in _big_parquet("HPLT/HPLT2.0_cleaned", files, ["text", "filter", "pii", "doc_scores"], "text", hf_token, scratch, room,
                          keep_filter=lambda r: r["filter"] == "keep" and not r["pii"]):
        sc = r["doc_scores"] or [0]
        yield r["text"], float(sc[0]), "HPLT_HPLT2.0_cleaned"


def speech(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    try:
        yield from _speech_all(lang, hf_token, scratch)
    except Exception as e:                                                      # keep what was read; the run continues
        import sys
        print(f"speech[{lang}] stopped early: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------------------------
# Second set of older crawls for the thinnest of the newly added languages: MADLAD-400 (2023, ODC-By; clean and noisy
# docs, the noisy ones are language-checked here) and mC4 (allenai/c4 multilingual, 2022, ODC-By).
MADLAD_CODE = {"sna": "sn", "nya": "ny", "sot": "st", "xho": "xh", "orm": "om", "tsn": "tn", "zul": "zu"}
MC4_CODE = {"sna": "sn", "nya": "ny", "sot": "st", "xho": "xh", "zul": "zu"}


def _gz_jsonl(repo: str, files: list[str], hf_token: str, scratch: Path):
    import gzip
    import json
    from huggingface_hub import hf_hub_download
    for path in files:
        cache = Path(tempfile.mkdtemp(dir=scratch))
        try:
            local = hf_hub_download(repo, path, repo_type="dataset", token=hf_token, cache_dir=str(cache))
            with gzip.open(local, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def legacy2(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    yield from legacy3(lang, hf_token, scratch)
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    def ls(repo, path):
        try:
            return sorted(f.path for f in api.list_repo_tree(repo, path_in_repo=path, repo_type="dataset", recursive=True)
                          if hasattr(f, "size") and f.path.endswith(".gz"))
        except Exception:
            return []

    c = MADLAD_CODE.get(lang)
    if c:
        files = ls("allenai/madlad-400", f"data-v1p5/{c}")
        files = sorted(files, key=lambda f: 0 if "clean_docs" in f else 1)          # clean first
        for j in _gz_jsonl("allenai/madlad-400", files, hf_token, scratch):
            t = j.get("text")
            if t:
                yield t, 6.0, "allenai_madlad-400"
    c = MC4_CODE.get(lang)
    if c:
        files = [f for f in ls("allenai/c4", "multilingual") if f"/c4-{c}." in f and "-train." in f or f"/c4-{c}." in f and "validation" not in f]
        for j in _gz_jsonl("allenai/c4", sorted(set(files)), hf_token, scratch):
            t = j.get("text")
            if t:
                yield t, 5.5, "allenai_c4-multilingual"


# Added after vetting the post-2021 search (commercial-friendly licences only).
_KIN_SPAM = ("Ubushinwa", "Inganda", "Diaper", "ibicuruzwa", "info@", "Alibaba", "Tel:", "WhatsApp", "Pipe ")


def _detok(t: str) -> str:
    import re
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def legacy3(lang: str, hf_token: str, scratch: Path) -> Iterator[Doc]:
    if lang == "kin":      # mbazaNLP/kinyarwanda_monolingual_v01.1 (CC-BY-4.0): scraped paragraphs, ~40% spam -> filtered
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        fs = HfFileSystem(token=hf_token)

        def rows():
            with fs.open("datasets/mbazaNLP/kinyarwanda_monolingual_v01.1/data/train-00000-of-00001.parquet", "rb") as fh:
                for b in pq.ParquetFile(fh).iter_batches(batch_size=5000, columns=["text", "nr_words"]):
                    d = b.to_pydict()
                    for t, n in zip(d["text"], d["nr_words"]):
                        if t and (n or 0) >= 20 and not any(x in t for x in _KIN_SPAM):
                            yield t.strip()
        for d in _pack(rows(), 6):
            yield d, 5.5, "mbazaNLP_kinyarwanda_monolingual_v01.1"
    if lang == "tsn":      # dsfsi/PuoData (CC-BY-SA-4.0): curated Setswana sentences from many sources
        for sp in ("train", "test"):
            lines = (_detok(t) for t in _stream("dsfsi/PuoData", None, sp, "text", hf_token))
            for d in _pack((l for l in lines if len(l.split()) >= 4), 12):
                yield d, 6.0, "dsfsi_PuoData"
    if lang in ("sot", "tsn", "xho", "zul"):      # dsfsi/vukuzenzele-monolingual (CC-BY-4.0): government magazine articles
        for sp in ("train", "test", "eval"):
            try:
                for t in _stream("dsfsi/vukuzenzele-monolingual", lang if lang != "tsn" else "tsn", sp, "text", hf_token):
                    yield t.replace("•", "\n"), 6.0, "dsfsi_vukuzenzele-monolingual"
            except Exception:
                continue
