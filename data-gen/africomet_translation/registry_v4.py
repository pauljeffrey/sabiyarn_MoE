"""v4 registry: wide Hugging Face sweep for pcm, efi, ibb, fon, urh, twi, ewe (+ parallel data with references).

Parallel sources register BOTH directions: the column of language A becomes a source with the aligned language-B
column as `response` (when B is one of our languages), and vice versa. A parallel dataset whose partner language is
not one of ours (e.g. Ewe-French) contributes only our language's column, as monolingual text.

Excluded on purpose (provenance checked on the dataset cards / samples):
  ghananlpcommunity/pristine-twi*, ghok-parallel-en-twi, twi-english-generation-prompts, *-synth, Medical-Reasoning-*
      LLM-generated.
  ghananlpcommunity/twi-english-paragraph-dataset_news     Twi side is NLLB output.
  KwekuAinoo/eng_twi, kweku2088/twi_to_english             unknown provenance, English side 22% non-English.
  *emotions-corpus*, *sentiments-corpus*, *farmer-qa*, alpaca-* , health-asr-*   translated/LLM-made task data.
  speech/TTS/ASR datasets                                    short read-aloud snippets; the text duplicates Bible/news.
  Cameroonian Pidgin (michsethowusu/english-cameroon-pidgin)  a different language from Nigerian Pidgin.
  FineWeb-2 pcm_Latn (23 GB)                                 almost entirely misidentified English.
Mined pair sets (michsethowusu / ghananlpcommunity "sentence-pairs") are sorted by NLLB margin score; quality collapses
after rank ~200k (mojibake, random names), so only the top rows above a score floor are read.
"""

from __future__ import annotations

import itertools
from typing import Iterator, Optional

from .registry import BIBLE_LIC, BY, CC0, ODC, SA, WIKI_LIC, _paragraph_docs, build_registry
from .sources import Ctx, Doc, Source
from .text_utils import normalize, split_sentences

NC = "CC-BY-NC-4.0"
NC_SA = "CC-BY-NC-SA-4.0"
JW_LIC = "unspecified (jw.org text; copyright not cleared)"
UNSPEC = "unspecified"

LENIENT_LANGS = ("pcm", "efi", "ibb", "fon", "urh", "twi", "ewe")


def _dl(ctx: Ctx, repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", cache_dir=str(ctx.cache), token=ctx.token)


def _cell(batch: dict, path: str, i: int):
    """batch['col'][i], or batch['col'][i]['key'] for a dotted path."""
    head, _, rest = path.partition(".")
    v = batch[head][i]
    return v.get(rest) if rest and isinstance(v, dict) else v


# ----------------------------------------------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------------------------------------------

def hf_loader(repo: str, col: str, *, ref_col: str = "", ref_lang: str = "", config: Optional[str] = None,
              split: str = "train", score_col: str = "", min_score: float = 0.0, max_rows: int = 0, stride: int = 1,
              text: bool = False, files: Optional[str] = None):
    """Read one column (and optionally an aligned one) of a Hub dataset.

    text=False: every row is one unit (a sentence / short paragraph) -> a one-sentence Doc, with its aligned `ref`.
    text=True : every row is a document; it is split into paragraphs and sentences (monolingual only).
    rows are read in file order; for score-sorted sets `max_rows` + `min_score` keep only the best rows."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        from datasets import load_dataset
        kw = {"data_files": files} if files else {}
        ds = load_dataset(repo, config, split=split, cache_dir=str(ctx.cache), token=ctx.token, **kw)
        if max_rows:
            ds = ds.select(range(0, min(len(ds), max_rows * stride), stride))
        elif stride > 1:
            ds = ds.select(range(0, len(ds), stride))
        use_score = bool(score_col) and score_col in ds.column_names
        need = {c.split(".")[0] for c in (col, ref_col, score_col if use_score else "") if c}
        ds = ds.select_columns(sorted(need))
        for batch in ds.iter(batch_size=4096):
            n = len(batch[col.split(".")[0]])
            for i in range(n):
                if use_score and float(batch[score_col][i]) < min_score:
                    continue
                t = _cell(batch, col, i)
                if not isinstance(t, str) or not t.strip():
                    continue
                if text:
                    yield from _paragraph_docs(t, min_chars=12)
                    continue
                if not ref_col:
                    yield Doc(split_sentences(t) or [t], split)
                    continue
                r = _cell(batch, ref_col, i)
                has = isinstance(r, str) and bool(r.strip())
                yield Doc([t], split, refs=[r] if has else None, ref_lang=ref_lang if has else "")
    return load


def _pretrain_new(lang: str):
    """Documents of BeardedMonster/pretrain-new (our own cleaned, deduplicated post-cutoff monolingual collection).
    One parquet file at a time is downloaded and deleted again."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        import shutil
        import tempfile
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
        repo = "BeardedMonster/pretrain-new"
        files = sorted(f.path for f in HfApi(token=ctx.token).list_repo_tree(repo, path_in_repo=f"data/{lang}", repo_type="dataset")
                       if f.path.endswith(".parquet"))
        for path in files:
            tmp = tempfile.mkdtemp(dir=str(ctx.cache))
            try:
                local = hf_hub_download(repo, path, repo_type="dataset", cache_dir=tmp, token=ctx.token)
                for batch in pq.ParquetFile(local).iter_batches(batch_size=500, columns=["text"]):
                    for t in batch.to_pydict()["text"]:
                        yield from _paragraph_docs(t, min_chars=12)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return load


def _jw(filename: str):
    """AfriSpeech/african-corpus-jw: paragraphs of jw.org articles, aligned to an English anchor on `docid#pid`."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        import pandas as pd
        loc = pd.read_csv(_dl(ctx, "AfriSpeech/african-corpus-jw", filename)).dropna(subset=["local"])
        eng = pd.read_csv(_dl(ctx, "AfriSpeech/african-corpus-jw", "english_cache.csv")).dropna()
        m = loc.merge(eng, on="verse_key", how="left")
        m["_doc"] = m["verse_key"].astype(str).str.split("#").str[0]
        for _, g in m.groupby("_doc", sort=False):
            yield Doc([normalize(t) for t in g["local"]], "jw",
                      refs=[normalize(e) if isinstance(e, str) else "" for e in g["eng"]], ref_lang="eng")
    return load


def _bible_ref(filename: str, repo: str = "AfriSpeech/africa-corpus"):
    """A Bible chapter is a document; its verses are the sentences, aligned to an English Bible on the verse key."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        import pandas as pd
        df = pd.read_csv(_dl(ctx, repo, filename)).dropna(subset=["local"])
        eng = pd.read_csv(_dl(ctx, repo, "english_cache.csv")).dropna().drop_duplicates("verse_key")
        df = df.merge(eng, on="verse_key", how="left")
        parts = df["verse_key"].astype(str).str.split(".", expand=True)
        df = df.assign(_ch=parts[0] + "." + parts[1], _v=pd.to_numeric(parts[2].str.extract(r"(\d+)")[0], errors="coerce"))
        for _, g in df.dropna(subset=["_v"]).groupby("_ch", sort=False):
            g = g.sort_values("_v")
            yield Doc([normalize(t) for t in g["local"]], "bible",
                      refs=[normalize(e) if isinstance(e, str) else "" for e in g["eng"]], ref_lang="eng")
    return load


def _madlad(filename: str):
    """AfriSpeech/madlad-91-african: MADLAD-400 documents, one per CSV row."""
    def load(ctx: Ctx) -> Iterator[Doc]:
        import pandas as pd
        for chunk in pd.read_csv(_dl(ctx, "AfriSpeech/madlad-91-african", filename), chunksize=20_000):
            for t in chunk["local"].dropna():
                yield from _paragraph_docs(str(t), min_chars=20)
    return load


def _english_efik_bible(ctx: Ctx) -> Iterator[Doc]:
    """7,978 consecutive verses, columns `efik` / `english`; groups of 8 verses act as one document."""
    import pyarrow.parquet as pq
    t = pq.read_table(_dl(ctx, "Aletheia-ng/English_efik_bible_translation", "data/train-00000-of-00001.parquet"),
                      columns=["efik", "english"]).to_pylist()
    for i in range(0, len(t), 8):
        grp = [r for r in t[i:i + 8] if isinstance(r["efik"], str) and isinstance(r["english"], str)]
        if grp:
            yield Doc([normalize(r["efik"]) for r in grp], "bible", refs=[normalize(r["english"]) for r in grp], ref_lang="eng")


# ----------------------------------------------------------------------------------------------------------------
# source builders
# ----------------------------------------------------------------------------------------------------------------

def _src(name, lic, loader, *, kind, prio, lid=True, ref_lang="", lid_ref_only=False, min_words=2, max_words=120, quota=None):
    return Source(name, lic, 1.0, loader, kind=kind, lid=lid, priority=prio, min_words=min_words, max_words=max_words,
                  require_terminal=False, quota=quota, ref_lang=ref_lang, lid_ref_only=lid_ref_only)


def pair(R, repo, a, col_a, b, col_b, *, lic, config=None, split="train", score_col="", min_score=0.0, max_rows=0,
         stride=1, kind="mined", prio=60, lid=True, files=None):
    """Register both directions of a parallel dataset. `b` may be a language outside our set (monolingual then)."""
    ours = set(R)
    tag = f"{repo}:{config or 'default'}"
    if a in ours:
        loader = hf_loader(repo, col_a, ref_col=col_b if b in ours else "", ref_lang=b if b in ours else "", config=config,
                           split=split, score_col=score_col, min_score=min_score, max_rows=max_rows, stride=stride, files=files)
        R[a].append(_src(f"{tag}:{col_a}->{b}", lic, loader, kind=kind, prio=prio, lid=lid, ref_lang=b if b in ours else ""))
    if b in ours and a in ours:
        loader = hf_loader(repo, col_b, ref_col=col_a, ref_lang=a, config=config, split=split, score_col=score_col,
                           min_score=min_score, max_rows=max_rows, stride=stride, files=files)
        R[b].append(_src(f"{tag}:{col_b}->{a}", lic, loader, kind=kind, prio=prio, lid=lid, ref_lang=a,
                         lid_ref_only=(b == "eng")))


def mono(R, lang, repo, col, *, lic, config=None, split="train", max_rows=0, stride=1, text=False, kind="web", prio=65,
         lid=True, files=None, min_words=3):
    R[lang].append(_src(f"{repo}:{config or 'default'}:{col}", lic,
                        hf_loader(repo, col, config=config, split=split, max_rows=max_rows, stride=stride, text=text, files=files),
                        kind=kind, prio=prio, lid=lid, min_words=min_words))


def build_registry_v4(eng: dict | None = None, include_nc: bool = False) -> dict[str, list[Source]]:
    base = build_registry(eng=eng, include_nc=include_nc)          # strict languages (yor hau ibo eng) keep their v3 lists
    NEW = ("afr", "zul", "sna", "nya", "xho", "sot", "kin", "tsn", "orm", "wol", "som", "swh")
    R: dict[str, list[Source]] = {l: [] for l in ("pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "ibb", "urh", "eng", *NEW)}
    for l in ("yor", "hau", "ibo", "eng"):
        R[l] = list(base[l])

    def reuse(lang, pred=lambda s: True):
        R[lang] += [s for s in base.get(lang, []) if pred(s)]

    def curated(lang):     # flores, smol, news, wikipedia, wikianc, glotcc, hplt, opus, ibom, sib200, efik audio sets
        reuse(lang, lambda s: s.kind != "bible")

    # ---------------------------------------------------------------- Nigerian Pidgin
    curated("pcm")
    pair(R, "Tommy0201/pidgin-to-english", "pcm", "pidgin", "eng", "english", lic=UNSPEC, kind="curated", prio=15, lid=True)
    pair(R, "Okwu/african-language-parallel-corpus", "pcm", "source_text", "eng", "english_translation", config="pcm-eng",
         lic="CDLA-Permissive-2.0", kind="curated", prio=16)
    pair(R, "michsethowusu/english-nigerian-pidgin_sentence-pairs_mt560", "pcm", "pcm", "eng", "eng", lic=BY, prio=60)
    pair(R, "Tommy0201/Pidgin_English_AmazonMTurk", "pcm", "text", "eng", "label", lic=UNSPEC, kind="curated", prio=17)
    mono(R, "pcm", "AnalyticsIntelligence/pidgin_corpus", "text", lic=UNSPEC, kind="news", prio=18)   # mirror of the AfriBERTa Pidgin (BBC) corpus; the original is a loader script
    mono(R, "pcm", "Guavacoderepo/gclm-pidgin-text-corpus", "text", lic=UNSPEC, kind="news", prio=19)
    mono(R, "pcm", "rufatronics/mt5_nigerian_pidgin", "input", lic=UNSPEC, kind="web", prio=66)
    mono(R, "pcm", "cis-lmu/Glot500", "text", config="pcm_Latn", lic="other (Glot500 mixture)", prio=67)
    mono(R, "pcm", "goldfish-models/fish-food", "text", config="pcm_latn", lic=NC_SA, prio=66)
    mono(R, "pcm", "HuggingFaceFW/finepdfs", "text", config="pcm_Latn", lic=ODC, text=True, kind="web", prio=70, max_rows=60_000)
    mono(R, "pcm", "Africanvoice/naija_african_voices", "transcript", lic="CC-BY-4.0", kind="speech_text", prio=72, max_rows=120_000)
    R["pcm"].append(_src("AfriSpeech/africa-corpus:Pidgin,_Nigerian_pcm_v2516.csv", BIBLE_LIC,
                         _bible_ref("Pidgin,_Nigerian_pcm_v2516.csv"), kind="bible", prio=30, lid=False, ref_lang="eng"))

    # ---------------------------------------------------------------- Efik
    curated("efi")
    pair(R, "michsethowusu/english-efik_sentence-pairs_mt560", "efi", "efi", "eng", "eng", lic=BY, kind="jw", prio=36)
    R["efi"].append(_src("AfriSpeech/african-corpus-jw:Efik_efi_v1.csv", JW_LIC, _jw("Efik_efi_v1.csv"), kind="jw", prio=35,
                         lid=False, ref_lang="eng"))
    R["efi"].append(_src("Aletheia-ng/English_efik_bible_translation", UNSPEC, _english_efik_bible, kind="bible", prio=29, lid=False, ref_lang="eng"))
    mono(R, "efi", "HuggingFaceFW/fineweb-2", "text", config="efi_Latn", lic=ODC, text=True, kind="web", prio=42)
    mono(R, "efi", "openbmb/DCAD-2000", "text", config="efi_Latn", lic="other (DCAD-2000)", text=True, kind="web", prio=43)
    mono(R, "efi", "HuggingFaceFW/finepdfs", "text", config="efi_Latn", lic=ODC, text=True, kind="web", prio=44)
    for v in (2570, 4541):
        R["efi"].append(_src(f"AfriSpeech/africa-corpus:Efik_efi_v{v}.csv", BIBLE_LIC, _bible_ref(f"Efik_efi_v{v}.csv"),
                             kind="bible", prio=30, lid=False, ref_lang="eng"))

    # ---------------------------------------------------------------- Ibibio
    curated("ibb")
    R["ibb"].append(_src("AfriSpeech/africa-corpus:Ibibio_ibb_v2619.csv", BIBLE_LIC, _bible_ref("Ibibio_ibb_v2619.csv"),
                         kind="bible", prio=30, lid=False, ref_lang="eng"))
    for f, pr in (("Ibibio_ibb_v1.csv", 40), ("Ibibio_ibb_v2.csv", 70)):
        R["ibb"].append(_src(f"AfriSpeech/madlad-91-african:{f}", ODC, _madlad(f), kind="web", prio=pr, lid=True, min_words=3))
    mono(R, "ibb", "openbmb/DCAD-2000", "text", config="ibb_Latn", lic="other (DCAD-2000)", text=True, kind="web", prio=43)

    # ---------------------------------------------------------------- Urhobo
    pair(R, "michsethowusu/english-urhobo_sentence-pairs_mt560", "urh", "urh", "eng", "eng", lic=BY, kind="jw", prio=36)
    mono(R, "urh", "HuggingFaceFW/fineweb-2", "text", config="urh_Latn", lic=ODC, text=True, kind="web", prio=42)
    mono(R, "urh", "openbmb/DCAD-2000", "text", config="urh_Latn", lic="other (DCAD-2000)", text=True, kind="web", prio=43)
    R["urh"].append(_src("AfriSpeech/africa-corpus:Urhobo_urh_v2616.csv", BIBLE_LIC, _bible_ref("Urhobo_urh_v2616.csv"),
                         kind="bible", prio=30, lid=False, ref_lang="eng"))

    # ---------------------------------------------------------------- Fon
    curated("fon")
    pair(R, "michsethowusu/english-fon_sentence-pairs", "fon", "Fon", "eng", "English", lic=UNSPEC, score_col="similarity",
         min_score=1.09, max_rows=300_000)
    pair(R, "African-Languages-Lab/multi-open", "fon", "fon", "eng", "english", config="english-fon", lic="other (multi-open)",
         score_col="translation_quality_score", min_score=8, kind="curated", prio=50, max_rows=270_000)
    pair(R, "michsethowusu/english-fon_sentence-pairs_mt560", "fon", "fon", "eng", "eng", lic=BY, kind="jw", prio=36)
    pair(R, "beethogedeon/Helsinki-en-fon-opus-20k", "fon", "translated_text", "eng", "source_text", lic=BY, kind="curated", prio=50)
    mono_fr = [("michsethowusu/fon-french_sentence-pairs", "Fon", 250_000), ("jonathansuru/fr_fon", "fon", 0),
               ("FrancophonIA/Fon_French_Daily_Dialogues_Parallel_Data", "Fon_Sentence", 0), ("amaniopia/fon_french_translation", "Fon_Sentence", 0)]
    for repo, col, cap in mono_fr:
        mono(R, "fon", repo, col, lic=UNSPEC, kind="mined", prio=62, max_rows=cap)
    R["fon"].append(_src("AfriSpeech/african-corpus-jw:Fon_fon_v1.csv", JW_LIC, _jw("Fon_fon_v1.csv"), kind="jw", prio=35, lid=False, ref_lang="eng"))
    for v in (813, 817):
        R["fon"].append(_src(f"AfriSpeech/africa-corpus:Fon_fon_v{v}.csv", BIBLE_LIC, _bible_ref(f"Fon_fon_v{v}.csv"),
                             kind="bible", prio=30, lid=False, ref_lang="eng"))
    for f, pr in (("Fon_fon_v1.csv", 40), ("Fon_fon_v2.csv", 70)):
        R["fon"].append(_src(f"AfriSpeech/madlad-91-african:{f}", ODC, _madlad(f), kind="web", prio=pr, min_words=3))
    mono(R, "fon", "HuggingFaceFW/fineweb-2", "text", config="fon_Latn", lic=ODC, text=True, kind="web", prio=42)
    mono(R, "fon", "cis-lmu/Glot500", "text", config="fon_Latn", lic="other (Glot500 mixture)", prio=67)
    mono(R, "fon", "openbmb/DCAD-2000", "text", config="fon_Latn", lic="other (DCAD-2000)", text=True, kind="web", prio=43)

    # ---------------------------------------------------------------- Ewe
    curated("ewe")
    pair(R, "ghananlpcommunity/english-ewe-sentence-pairs-4m", "ewe", "Ewe", "eng", "English", lic=NC, score_col="similarity",
         min_score=1.10, max_rows=350_000)
    pair(R, "African-Languages-Lab/multi-open", "ewe", "ewe", "eng", "english", config="english-ewe", lic="other (multi-open)",
         score_col="translation_quality_score", min_score=8, kind="curated", prio=50)
    pair(R, "Ghana-NLP/EWE_ENGLISH_PARALLEL_TEXT", "ewe", "label", "eng", "text", lic=UNSPEC, kind="curated", prio=15)
    pair(R, "ghanaopenai/ewe-twi_sentence-pairs-200k", "ewe", "Ewe", "twi", "Twi", lic=NC, max_rows=150_000)
    mono(R, "ewe", "michsethowusu/ewe-french_sentence-pairs", "Ewe", lic=UNSPEC, kind="mined", prio=62, max_rows=250_000)
    mono(R, "ewe", "goldfish-models/fish-food", "text", config="ewe_latn", lic=NC_SA, prio=66, stride=5)
    R["ewe"].append(_src("AfriSpeech/african-corpus-jw:Ewe_ee_v1.csv", JW_LIC, _jw("Ewe_ee_v1.csv"), kind="jw", prio=35, lid=False, ref_lang="eng"))
    for v in (1613, 2259, 3306):
        R["ewe"].append(_src(f"AfriSpeech/africa-corpus:Éwé_ewe_v{v}.csv", BIBLE_LIC, _bible_ref(f"Éwé_ewe_v{v}.csv"),
                             kind="bible", prio=30, lid=False, ref_lang="eng"))
    for f, pr in (("Ewe_ee_v1.csv", 40), ("Ewe_ee_v2.csv", 70)):
        R["ewe"].append(_src(f"AfriSpeech/madlad-91-african:{f}", ODC, _madlad(f), kind="web", prio=pr, min_words=3))
    mono(R, "ewe", "HuggingFaceFW/fineweb-2", "text", config="ewe_Latn", lic=ODC, text=True, kind="web", prio=42)
    mono(R, "ewe", "cis-lmu/Glot500", "text", config="ewe_Latn", lic="other (Glot500 mixture)", prio=67)

    # ---------------------------------------------------------------- Twi
    curated("twi")
    pair(R, "ghananlpcommunity/english-twi_sentence-pairs-4m", "twi", "Twi", "eng", "English", lic=NC, score_col="similarity",
         min_score=1.10, max_rows=350_000)
    pair(R, "michsethowusu/akan-english_sentence-pairs", "twi", "Akan", "eng", "English", lic=UNSPEC, score_col="similarity", min_score=1.05)
    pair(R, "African-Languages-Lab/multi-open", "twi", "twi", "eng", "english", config="english-twi", lic="other (multi-open)",
         score_col="translation_quality_score", min_score=8, kind="curated", prio=50)
    pair(R, "Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT", "twi", "text", "eng", "label", lic=UNSPEC, kind="curated", prio=15)
    mono(R, "twi", "ghanaopenai/twi_sentences_320k", "sentence", lic=NC, kind="web", prio=64, max_rows=300_000)
    mono(R, "twi", "goldfish-models/fish-food", "text", config="twi_latn", lic=NC_SA, prio=66, stride=6)
    R["twi"].append(_src("AfriSpeech/african-corpus-jw:Twi_tw_v1.csv", JW_LIC, _jw("Twi_tw_v1.csv"), kind="jw", prio=35, lid=False, ref_lang="eng"))
    for v in (1461, 1631, 1861, 3439, 3440):
        R["twi"].append(_src(f"AfriSpeech/africa-corpus:Twi_twi_v{v}.csv", BIBLE_LIC, _bible_ref(f"Twi_twi_v{v}.csv"),
                             kind="bible", prio=30, lid=False, ref_lang="eng"))
    for f, pr in (("Akan_ak_v1.csv", 40), ("Akan_ak_v2.csv", 70)):
        R["twi"].append(_src(f"AfriSpeech/madlad-91-african:{f}", ODC, _madlad(f), kind="web", prio=pr, min_words=3))
    mono(R, "twi", "HuggingFaceFW/fineweb-2", "text", config="twi_Latn", lic=ODC, text=True, kind="web", prio=42)
    mono(R, "twi", "cis-lmu/Glot500", "text", config="twi_Latn", lic="other (Glot500 mixture)", prio=67)
    mono(R, "twi", "openbmb/DCAD-2000", "text", config="aka_Latn", lic="other (DCAD-2000)", text=True, kind="web", prio=43)

    # ---------------------------------------------------------------- pairs between our own languages
    # (michsethowusu = NLLB-mined; 'similarity' sorted descending; cap rows per set)
    for repo, a, ca, b, cb, cap in [
        ("michsethowusu/fon-hausa_sentence-pairs", "fon", "Fon", "hau", "Hausa", 80_000),
        ("michsethowusu/fon-yoruba_sentence-pairs", "fon", "Fon", "yor", "Yoruba", 70_000),
        ("michsethowusu/fon-igbo_sentence-pairs", "fon", "Fon", "ibo", "Igbo", 50_000),
        ("michsethowusu/fon-twi_sentence-pairs", "fon", "Fon", "twi", "Twi", 70_000),
        ("michsethowusu/ewe-fon_sentence-pairs", "ewe", "Ewe", "fon", "Fon", 70_000),
        ("michsethowusu/ewe-hausa_sentence-pairs", "ewe", "Ewe", "hau", "Hausa", 100_000),
        ("michsethowusu/ewe-yoruba_sentence-pairs", "ewe", "Ewe", "yor", "Yoruba", 100_000),
        ("michsethowusu/ewe-igbo_sentence-pairs", "ewe", "Ewe", "ibo", "Igbo", 80_000),
        ("michsethowusu/akan-yoruba_sentence-pairs", "twi", "Akan", "yor", "Yoruba", 90_000),
        ("michsethowusu/akan-hausa_sentence-pairs", "twi", "Akan", "hau", "Hausa", 80_000),
        ("michsethowusu/akan-igbo_sentence-pairs", "twi", "Akan", "ibo", "Igbo", 39_000),
        ("michsethowusu/akan-fon_sentence-pairs", "twi", "Akan", "fon", "Fon", 46_000),
        ("michsethowusu/akan-ewe_sentence-pairs", "twi", "Akan", "ewe", "Ewe", 60_000),
    ]:
        pair(R, repo, a, ca, b, cb, lic=UNSPEC, score_col="similarity", min_score=1.07, max_rows=cap, prio=61)

    # ---------------------------------------------------------------- our own post-cutoff monolingual collection
    # one deficit-balanced window per paragraph (stream) so the quota is spread over many documents, not the first few
    for l in (*NEW, "pcm", "efi", "fon", "ewe", "twi", "yor", "hau", "ibo"):
        R[l].append(Source("BeardedMonster/pretrain-new", "mixed (see pretrain-new card)", 1.0, _pretrain_new(l), kind="web", lid=True,
                           priority=45, min_words=3, max_words=120, require_terminal=False, quota=400_000, stream=True))
    return R
