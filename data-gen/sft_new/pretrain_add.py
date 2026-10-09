"""Extra monolingual sources for BeardedMonster/pretrain-new (the text/lang/score table behind other_african_pretrain.bin).

    nice -n 10 python -m sft_new.pretrain_add --preview [NAME ...]      # 3 documents per source + language/drop statistics
    HF_HOME=SCRATCH nice -n 10 python -m sft_new.pretrain_add --run NAME ... --out DIR [--push]   # clean, language-check, dedupe, write shards, push, delete

Each source says where the text is and which language it is in. A language the repo states is CHECKED, not trusted: a document
whose text is clearly another supported language is dropped. A source with mixed languages is routed by detection (certain/likely only).

Cleaning and exact + near-duplicate removal reuse pretrain_new.docclean. The deduper is built from the language's existing
pretrain-new documents first, so nothing added here repeats what is already in the repo.

Sources (all named explicitly by the user, so the date rule does not apply):
    lrln, monolingual, pidgin-corpus-synth (title + content), yoruba-corpus-synth (title + content), yo_wiki_processed, Glot500-swahili,
    polynews-swahili, naijaweb, naijaweb-edu, naijaweb-edu2, text_ratings (text column only), data-gen pretrain, WebScraped_Igbo_To_Eng
    and english_to_igbo in the `<translate> A <eng>\\nB` form.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from sft_new import hubio  # noqa: E402
from sft_new import langdetect as L  # noqa: E402
from sft_new.textutil import clean, num  # noqa: E402

TARGET_REPO = "BeardedMonster/pretrain-new"
SHARD_ROWS = 200_000


@dataclass
class Source:
    repo: str
    text: Callable[[dict], str]
    lang: str | None                       # None = detect per document
    score: Callable[[dict], float] = lambda r: 5.0
    keep: Callable[[dict], bool] = lambda r: True
    note: str = ""
    pack: int = 0                          # join this many consecutive rows into one document (sentence-level sources)
    min_chars: int = 100
    lang_of: Callable[[dict], str | None] | None = None      # language stated per row by the source itself
    pair: tuple | None = None              # (col_a, lang_a, col_b, lang_b): emit "<translate> A <lang_b>\nB" documents in both directions
    match: str | None = None               # only files whose path contains this
    folder: str | None = None              # destination folder under data/ when it is not the language code (eng-tail: appended to the END of the bin)


def _tc(r):                                # title + content
    t, c = clean(r.get("title")), clean(r.get("content"))
    return f"{t}\n{c}" if t and c else (c or t)


_DG = {"aka": "twi", "twi": "twi", "yor": "yor", "hau": "hau", "ibo": "ibo", "pcm": "pcm", "ewe": "ewe", "fon": "fon", "efi": "efi", "urh": "urh", "eng": "eng"}

SOURCES: dict[str, Source] = {
    "lrln": Source("BeardedMonster/lrln", lambda r: clean(r.get("text")), None, keep=lambda r: str(r.get("is_bad")).lower() in ("false", "0", "none", ""), note="is_bad rows removed; language by detection"),
    "monolingual": Source("BeardedMonster/monolingual", lambda r: clean(r.get("text")) or _tc(r), None, note="language by detection"),
    "pidgin_synth": Source("Aletheia-ng/pidgin-corpus-synth", _tc, "pcm", note="LLM-written Nigerian Pidgin; title and content merged"),
    "yoruba_synth": Source("Aletheia-ng/yoruba-corpus-synth", _tc, "yor", note="LLM-written Yoruba; title and content merged"),
    "yo_wiki": Source("Aletheia-ng/yo_wiki_processed", lambda r: clean(r.get("raw_text")), "yor", score=lambda r: 3.0, note="lower-cased, unpunctuated wiki text"),
    "glot500_swh": Source("Aletheia-ng/Glot500-swahili", lambda r: clean(r.get("text")), "swh", pack=12, note="sentence-level: 12 consecutive sentences per document"),
    "polynews_swh": Source("Aletheia-ng/polynews-swahili", lambda r: clean(r.get("text")), "swh", pack=12, note="headline-level: 12 consecutive headlines per document"),
    "naijaweb": Source("Aletheia-ng/naijaweb", lambda r: clean(r.get("text")), "eng", score=lambda r: 2.0 * num(r.get("int_score")), note="Nigerian news in English; goes to data/eng-tail = the end of the bin", folder="eng-tail"),
    "naijaweb_edu": Source("Aletheia-ng/naijaweb-edu", lambda r: clean(r.get("text")), "eng", score=lambda r: 2.0 * num(r.get("int_score")), folder="eng-tail"),
    "naijaweb_edu2": Source("Aletheia-ng/naijaweb-edu2", lambda r: clean(r.get("text")), "eng", score=lambda r: 2.0 * num(r.get("int_score")), folder="eng-tail"),
    "datagen_pretrain": Source("BeardedMonster/data-gen", lambda r: clean(r.get("text")), None, match="pretrain/", lang_of=lambda r: _DG.get(str(r.get("lang"))),
                               keep=lambda r: float(r.get("confidence") or 0) >= 0.7, note="our own LLM-written documents; aka -> twi; Fulfulde dropped"),
    "igbo_pairs_webscraped": Source("Aletheia-ng/WebScraped_Igbo_To_Eng", lambda r: "", "ibo", min_chars=40, pair=("igbo", "ibo", "english", "eng"), note="<translate> A <tgt> B documents"),
    "igbo_pairs_english_to_igbo": Source("Aletheia-ng/english_to_igbo", lambda r: "", "ibo", min_chars=40, pair=("Igbo", "ibo", "English", "eng"), note="<translate> A <tgt> B documents"),
    "text_ratings": Source("Aletheia-ng/text_ratings", lambda r: clean(r.get("text")), None, score=lambda r: min(10.0, num(r.get("rating_float")) / 10.0), note="text column only; language by detection"),
}


def _detect_batch(texts: list[str]):
    return L.detect_many([t[:1500] for t in texts])


def _docs_from_rows(s: Source, rows: list[dict]):
    """Rows -> candidate documents (text, forced language or None, score)."""
    if s.pair:
        ca, la, cb, lb = s.pair
        for r in rows:
            a, b = clean(r.get(ca)), clean(r.get(cb))
            if a and b:
                yield f"<translate> {a} <{lb}>\n{b}", la, 5.0
                yield f"<translate> {b} <{la}>\n{a}", la, 5.0          # both directions belong to the African language's folder
        return
    if s.pack:
        for i in range(0, len(rows), s.pack):
            grp = rows[i:i + s.pack]
            yield "\n".join(t for t in (s.text(r) for r in grp) if t), s.lang, sum(s.score(r) for r in grp) / len(grp)
        return
    for r in rows:
        yield s.text(r), (s.lang_of(r) if s.lang_of else s.lang), s.score(r)


def iter_docs(name: str, limit: int | None) -> Iterator[tuple[str, str, float, str]]:
    """-> (text, lang, score, drop_reason_or_''). A drop reason means the document was rejected at routing time."""
    s = SOURCES[name]
    buf: list[dict] = []

    def flush():
        cands = [(t, lg, sc) for t, lg, sc in _docs_from_rows(s, buf)]
        buf.clear()
        det = _detect_batch([t for t, _, _ in cands]) if cands and not s.pair else [None] * len(cands)
        for (t, lg, sc), d in zip(cands, det):
            if s.lang_of is not None and lg is None:
                yield "", "", 0.0, "language_not_supported"
            elif len(t) < s.min_chars:
                yield "", "", 0.0, "too_short"
            elif s.pair:
                yield t, lg, sc, ""
            elif lg is None:
                if d.code and d.level != "none":
                    yield t, d.code, sc, ""
                else:
                    yield "", "", 0.0, "language_unsupported_or_undetermined"
            elif d.code and d.code != lg and d.level == "certain" and d.code != "eng" and lg != "pcm":
                yield "", "", 0.0, f"is_{d.code}_not_{lg}"
            else:
                yield t, lg, sc, ""

    for r in hubio.iter_repo(s.repo, limit=limit, match=s.match):
        if not s.keep(r):
            yield "", "", 0.0, "filtered_by_source_flag"
            continue
        buf.append(r)
        if len(buf) >= 256 + (s.pack or 0) * 21:
            yield from flush()
    yield from flush()


def preview(names: list[str], limit: int, show: int = 3):
    import collections
    for name in names:
        s = SOURCES[name]
        kept, drops, langs = 0, collections.Counter(), collections.Counter()
        ex = []
        for text, lang, score, why in iter_docs(name, limit):
            if why:
                drops[why] += 1
                continue
            kept += 1
            langs[lang] += 1
            if len(ex) < show:
                ex.append((lang, score, text))
        print(f"\n=== {name}: {s.repo}  [{s.note}]\n    {limit} rows read -> {kept} usable  languages={dict(langs.most_common(6))}  dropped={dict(drops.most_common(5))}")
        for lang, score, text in ex:
            print(f"    [{lang} score={score:.1f}] {text[:300]!r}")


def existing_deduper(lang: str, dd):
    """Fill `dd` with every document already in pretrain-new for `lang` (range-read, nothing stored)."""
    n = 0
    for r in hubio.iter_repo(TARGET_REPO, columns=["text"], match=f"data/{lang}/"):
        dd.is_duplicate(r.get("text") or "")
        n += 1
    return n


def run(names: list[str], out: Path, push: bool, limit: int | None = None):
    import collections
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from pretrain_new import docclean
    from sft_new.combine import DiskGuard
    guard = DiskGuard(out)
    dedupers: dict = {}
    api = HfApi(token=hubio.token())
    for name in names:
        stats = collections.Counter()
        writers: dict = {}
        for text, lang, score, why in iter_docs(name, limit):
            if why:
                stats["drop:" + why] += 1
                continue
            key = SOURCES[name].folder or lang                          # destination folder under data/
            if key not in dedupers:
                dedupers[key] = docclean.Deduper(near_share=0.5)
                print(f"loading existing {key} documents for de-duplication ...", file=sys.stderr, flush=True)
                existing_deduper(key, dedupers[key])
            t = text if SOURCES[name].pair else docclean.clean_document(text, lang)
            if t is None:
                stats["drop:cleaned_out"] += 1
                continue
            if dedupers[key].is_duplicate(t):
                stats["drop:duplicate"] += 1
                continue
            w = writers.setdefault(key, [])
            w.append((t, lang, score))
            stats[f"kept:{key}"] += 1
            if len(w) >= SHARD_ROWS:
                _flush(out, name, key, writers, stats)
                guard.check()
        for key in list(writers):
            _flush(out, name, key, writers, stats)
        print(f"{name}: {dict(stats)}", file=sys.stderr, flush=True)
        if push:
            for lang in sorted({k[5:] for k in stats if k.startswith("kept:")}):
                folder = out / "data" / lang
                api.upload_folder(repo_id=TARGET_REPO, repo_type="dataset", folder_path=str(folder), path_in_repo=f"data/{lang}",
                                  commit_message=f"{lang}: add {name}")
                shutil.rmtree(folder, ignore_errors=True)
        hubio.clear_cache()


def _flush(out: Path, name: str, folder: str, writers: dict, stats):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = writers.pop(folder, [])
    if not rows:
        return
    d = out / "data" / folder
    d.mkdir(parents=True, exist_ok=True)
    k = len(list(d.glob(f"add-{name}-*.parquet")))
    pq.write_table(pa.table({"text": [r[0] for r in rows], "lang": [r[1] for r in rows], "score": pa.array([r[2] for r in rows], pa.float32())}),
                   d / f"add-{name}-{k:05d}.parquet", compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args()
    names = a.names or list(SOURCES)
    if a.preview:
        preview(names, a.limit)
    elif a.run:
        if not a.out or not os.environ.get("HF_HOME"):
            sys.exit("--run needs --out DIR and HF_HOME set to a scratch folder")
        run(names, a.out, a.push)


if __name__ == "__main__":
    main()
