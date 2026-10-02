"""Stage B: candidate windows -> final prompt shards.

1. DEDUPE, per source language, in source-priority order (curated > encyclopedia > bible > web). A window is dropped
   if any of its sentences already appeared in an earlier window (vectorised over 64-bit sentence hashes).
2. PLAN. Non-English rows are all kept. English rows fill up to `--total`, and are distributed over the nine other
   languages so every TARGET language ends up with about the same number of rows (English as a target gets double
   weight from the non-English sources). English is the augmenter you asked for: it never reduces the other rows.
3. SCATTER/RENDER. Rows of each language are shuffled across output shards of ~200k rows (so every shard mixes every
   source), then instruction language, wording, position and target are assigned and parquet is written.
4. PUSH one commit per batch of shards.
"""

from __future__ import annotations

import glob
import json
import re
import shutil
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROWS_PER_OUT_SHARD = 200_000
COLS = ["text", "n_sentences", "tokens", "source", "license", "kind"]
LANGS = ["pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "ibb", "eng"]


def _priority_by_slug() -> dict[str, int]:
    from .registry import build_registry, slug
    reg = build_registry(eng={"c4": 1024, "wiki": 41, "edu": 14, "quota": 0}, include_nc=True)
    return {slug(s.name): s.priority for srcs in reg.values() for s in srcs}


def _stem_slug(path: Path) -> str:
    return re.sub(r"-\d{3}$", "", path.stem)


# ------------------------------------------------------------------------------------------------ 1. dedupe

def dedupe_language(out: Path, lang: str, force: bool = False) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    kept_dir = out / "kept" / lang
    marker = kept_dir / "_done.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    shutil.rmtree(kept_dir, ignore_errors=True)
    kept_dir.mkdir(parents=True)
    prio = _priority_by_slug()
    files = sorted((out / "windows" / lang).glob("*.parquet"), key=lambda p: (prio.get(_stem_slug(p), 99), p.name))
    seen = np.empty(0, dtype=np.int64)
    parts, total_in, total_kept = [], 0, 0
    for i, f in enumerate(files):
        t = pq.read_table(f)
        n = t.num_rows
        if not n:
            continue
        col = t.column("hashes").combine_chunks()
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.int64)
        lens = np.diff(col.offsets.to_numpy())
        owner = np.repeat(np.arange(n), lens)
        _, first = np.unique(flat, return_index=True)
        is_first = np.zeros(len(flat), dtype=bool)
        is_first[first] = True
        bad = ~is_first | np.isin(flat, seen)
        bad_win = np.bincount(owner[bad], minlength=n) > 0
        keep = ~bad_win
        seen = np.union1d(seen, flat[np.repeat(keep, lens)])
        kt = t.select(COLS).filter(pa.array(keep))
        pq.write_table(kt, kept_dir / f"{i:04d}.parquet", compression="zstd")
        parts.append({"file": f"{i:04d}.parquet", "source": f.stem, "rows": int(keep.sum())})
        total_in += n
        total_kept += int(keep.sum())
    res = {"lang": lang, "in": total_in, "kept": total_kept, "parts": parts}
    marker.write_text(json.dumps(res))
    return res


# ------------------------------------------------------------------------------------------------ 2. plan

def target_weights_non_eng(src: str) -> dict[str, float]:
    return {t: (2.0 if t == "eng" else 1.0) for t in LANGS if t != src}


def english_weights(non_eng_counts: dict[str, int], english_rows: int) -> dict[str, float]:
    """Distribute English rows over the nine other languages so every target's total is as equal as possible."""
    base = {t: 0.0 for t in LANGS}
    for s, n in non_eng_counts.items():
        w = target_weights_non_eng(s)
        z = sum(w.values())
        for t, wt in w.items():
            base[t] += n * wt / z
    tg = [t for t in LANGS if t != "eng"]
    lo, hi = 0.0, max(base[t] for t in tg) + english_rows
    for _ in range(200):                                   # water-filling: find level C with sum(max(0, C-base)) = E
        mid = (lo + hi) / 2
        need = sum(max(0.0, mid - base[t]) for t in tg)
        lo, hi = (mid, hi) if need < english_rows else (lo, mid)
    level = (lo + hi) / 2
    x = {t: max(0.0, level - base[t]) for t in tg}
    z = sum(x.values()) or 1.0
    return {t: v / z for t, v in x.items()}


def split_counts(n: int, weights: dict[str, float]) -> dict[str, int]:
    z = sum(weights.values())
    raw = {k: n * v / z for k, v in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[: n - sum(base.values())]:
        base[k] += 1
    return base


# ------------------------------------------------------------------------------------------------ 3. scatter + render

def scatter(out: Path, lang: str, keep_per_part: dict[str, int], n_out: int, seed: int) -> None:
    import pyarrow.parquet as pq
    tmp = out / "tmp" / lang
    shutil.rmtree(tmp, ignore_errors=True)
    for p in sorted((out / "kept" / lang).glob("[0-9]*.parquet")):
        t = pq.read_table(p)
        want = keep_per_part.get(p.name, t.num_rows)
        rng = np.random.default_rng([seed, zlib.crc32(f"{lang}/{p.name}".encode())])
        idx = rng.permutation(t.num_rows)[:want]
        for i, chunk in enumerate(np.array_split(idx, n_out)):
            d = tmp / f"{i:05d}"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(t.take(np.sort(chunk)), d / p.name, compression="zstd")


def render_shard(out: Path, lang: str, i: int, weights: dict[str, float], seed: int, tag: str, stats: dict) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .build import rng_for
    from .instructions import compose_prompt, instruction_languages, make_instruction
    from .sources import AFRICOMET_SUPPORTED
    tables = [pq.read_table(p) for p in sorted((out / "tmp" / lang / f"{i:05d}").glob("*.parquet"))]
    t = pa.concat_tables(tables)
    n = t.num_rows
    rng = rng_for(seed, "render", lang, i)
    order = list(range(n))
    rng.shuffle(order)
    t = t.take(order)
    targets = [k for k, c in split_counts(n, weights).items() for _ in range(c)]
    rng.shuffle(targets)
    il = instruction_languages()
    ins_langs = [il[j % len(il)] for j in range(n)]
    rng.shuffle(ins_langs)
    text, kind, lic, src = (t.column(c).to_pylist() for c in ("text", "kind", "license", "source"))
    ks, toks = t.column("n_sentences").to_pylist(), t.column("tokens").to_pylist()
    cols = {k: [] for k in ("id", "instruction", "input", "source_language", "target_language", "prompt",
                            "instruction_position", "instruction_language", "template_confidence", "n_sentences",
                            "src_tokens", "source_dataset", "source_license", "source_kind", "africomet_supported")}
    for j in range(n):
        tgt = targets[j]
        ins = make_instruction(ins_langs[j], lang, tgt, rng)
        sup = lang in AFRICOMET_SUPPORTED and tgt in AFRICOMET_SUPPORTED
        for k, v in (("id", f"{lang}-{tag}-{i:04d}-{j:06d}"), ("instruction", ins.text), ("input", text[j]),
                     ("source_language", lang), ("target_language", tgt), ("prompt", compose_prompt(ins, text[j], rng)),
                     ("instruction_position", ins.position), ("instruction_language", ins.language),
                     ("template_confidence", ins.confidence), ("n_sentences", ks[j]), ("src_tokens", toks[j]),
                     ("source_dataset", src[j]), ("source_license", lic[j]), ("source_kind", kind[j]),
                     ("africomet_supported", sup)):
            cols[k].append(v)
        stats["pairs"][f"{lang}>{tgt}"] += 1
    stats["rows"][lang] += n
    for k, name in (("source_kind", "kind"), ("source_license", "license"), ("source_dataset", "source"),
                    ("instruction_language", "ins_lang"), ("instruction_position", "position"),
                    ("n_sentences", "k"), ("target_language", "target"), ("africomet_supported", "supported")):
        stats[name].update(cols[k])
    dest = out / "final" / lang / f"train-{tag}-{i:05d}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dest, compression="zstd")
    return dest


def new_stats() -> dict:
    return {k: Counter() for k in ("pairs", "rows", "kind", "license", "source", "ins_lang", "position", "k", "target", "supported")}


def stats_to_json(stats: dict) -> dict:
    return {k: {str(a): b for a, b in v.items()} for k, v in stats.items()}


# ------------------------------------------------------------------------------------------------ orchestration

def assemble(out: Path, total: int, seed: int, tag: str, langs: list[str] | None = None, limit_per_lang: int | None = None,
             on_language_done=None, max_non_eng_per_lang: int | None = None) -> dict:
    langs = langs or LANGS
    print("== 1/3 de-duplicating", flush=True)
    dd = {}
    for lang in langs:
        if (out / "windows" / lang).exists():
            dd[lang] = dedupe_language(out, lang)
            print(f"   {lang}: {dd[lang]['in']:,} -> {dd[lang]['kept']:,} windows after de-duplication", flush=True)
    non_eng = {l: min(d["kept"], max_non_eng_per_lang or 10**12, limit_per_lang or 10**12) for l, d in dd.items() if l != "eng"}
    eng_avail = dd.get("eng", {}).get("kept", 0)
    english_rows = 0
    if "eng" in dd:
        english_rows = max(0, min(eng_avail, total - sum(non_eng.values()), limit_per_lang or 10**12))
    weights_eng = english_weights(non_eng, english_rows) if english_rows else {}
    print(f"== 2/3 plan: non-English {sum(non_eng.values()):,} + English {english_rows:,} = {sum(non_eng.values()) + english_rows:,} "
          f"(English available {eng_avail:,})", flush=True)

    stats = new_stats()
    for lang in langs:
        if lang not in dd:
            continue
        n_keep = english_rows if lang == "eng" else non_eng[lang]
        if n_keep == 0:
            continue
        parts = dd[lang]["parts"]
        total_rows = sum(p["rows"] for p in parts)
        keep = {p["file"]: int(round(p["rows"] * n_keep / total_rows)) for p in parts} if n_keep < total_rows else {}
        n_out = max(1, -(-n_keep // ROWS_PER_OUT_SHARD))
        print(f"== 3/3 {lang}: {n_keep:,} rows -> {n_out} shard(s)", flush=True)
        scatter(out, lang, keep, n_out, seed)
        w = weights_eng if lang == "eng" else target_weights_non_eng(lang)
        files = [render_shard(out, lang, i, w, seed, tag, stats) for i in range(n_out)]
        shutil.rmtree(out / "tmp" / lang, ignore_errors=True)
        if on_language_done:
            on_language_done(lang, files)
    (out / "final").mkdir(exist_ok=True)
    (out / "final" / f"stats-{tag}.json").write_text(json.dumps(stats_to_json(stats), indent=1))
    return stats
