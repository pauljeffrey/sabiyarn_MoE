"""Stage B: candidate windows -> final prompt shards.

1. DEDUPE, per source language, in source-priority order (curated > encyclopedia > Bible > JW > web > mined). A window
   is dropped if any of its sentences already appeared in an earlier window (vectorised over 64-bit sentence hashes).
2. PLAN. Non-English rows are all kept. Rows that carry an aligned reference (`ref`) have a FIXED target: the language of
   the reference, which becomes `response`. All other rows get a target drawn so that, once English free rows are
   added, every TARGET language ends up with about the same number of rows.
3. SCATTER/RENDER. Rows of each language are shuffled across output shards of ~200k rows (so every shard mixes every
   source), then instruction language, wording, position and target are assigned and parquet is written.
4. PUSH one commit per batch of shards.

Final columns: instruction, input, response, source_lang, targ_lang (+ provenance / instruction metadata).
"""

from __future__ import annotations

import json
import re
import shutil
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROWS_PER_OUT_SHARD = 200_000
COLS = ["text", "n_sentences", "tokens", "source", "license", "kind", "ref", "ref_lang"]
# Ibibio and Urhobo are NOT in AfriCOMET's language list, so they are not part of the RL set at all.
NEW_LANGS = ["afr", "zul", "sna", "nya", "xho", "sot", "kin", "tsn", "orm", "wol", "som", "swh"]
LANGS = ["pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", *NEW_LANGS, "eng"]
FINAL_COLUMNS = ["instruction", "input", "response", "source_lang", "targ_lang", "id", "prompt", "instruction_position",
                 "instruction_language", "template_confidence", "n_sentences", "src_tokens", "source_dataset",
                 "source_license", "source_kind", "africomet_supported", "output_structure", "text"]


def _priority_by_slug() -> dict[str, int]:
    from .registry import slug
    from .registry_v4 import build_registry_v4
    reg = build_registry_v4(eng={"c4": 1024, "wiki": 41, "edu": 14, "quota": 0}, include_nc=True)
    return {slug(s.name): s.priority for srcs in reg.values() for s in srcs}


def _stem_slug(path: Path) -> str:
    return re.sub(r"-\d{3}$", "", path.stem)


def _with_ref_columns(t):
    import pyarrow as pa
    for c in ("ref", "ref_lang"):
        if c not in t.column_names:
            t = t.append_column(c, pa.array([""] * t.num_rows, type=pa.string()))
    return t


# ------------------------------------------------------------------------------------------------ 1. dedupe

def dedupe_language(out: Path, lang: str, force: bool = False, src_dir: str = "windows") -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    kept_dir = out / f"kept_{src_dir}" / lang
    marker = kept_dir / "_done.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    shutil.rmtree(kept_dir, ignore_errors=True)
    kept_dir.mkdir(parents=True)
    prio = _priority_by_slug()
    files = sorted((out / src_dir / lang).glob("*.parquet"), key=lambda p: (prio.get(_stem_slug(p), 99), p.name))
    seen = np.empty(0, dtype=np.int64)
    parts, total_in, total_kept = [], 0, 0
    for i, f in enumerate(files):
        t = _with_ref_columns(pq.read_table(f))
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
        keep = ~(np.bincount(owner[bad], minlength=n) > 0)
        seen = np.union1d(seen, flat[np.repeat(keep, lens)])
        kt = t.select(COLS).filter(pa.array(keep))
        pq.write_table(kt, kept_dir / f"{i:04d}.parquet", compression="zstd")
        refs = Counter(r for r in kt.column("ref_lang").to_pylist() if r)
        parts.append({"file": f"{i:04d}.parquet", "source": f.stem, "rows": int(keep.sum()), "ref": dict(refs)})
        total_in += n
        total_kept += int(keep.sum())
    res = {"lang": lang, "in": total_in, "kept": total_kept, "parts": parts}
    marker.write_text(json.dumps(res))
    return res


# ------------------------------------------------------------------------------------------------ 2. plan

def target_weights_free(src: str) -> dict[str, float]:
    """Target distribution for non-English rows WITHOUT a reference: English double-weighted."""
    return {t: (2.0 if t == "eng" else 1.0) for t in LANGS if t != src}


def english_weights(base_totals: dict[str, float], english_rows: int) -> dict[str, float]:
    """Spread `english_rows` free English rows over the non-English targets so every target total is as equal as
    possible (water-filling on the totals the other rows already produce)."""
    tg = [t for t in LANGS if t != "eng"]
    lo, hi = 0.0, max(base_totals.get(t, 0.0) for t in tg) + english_rows
    for _ in range(200):
        mid = (lo + hi) / 2
        need = sum(max(0.0, mid - base_totals.get(t, 0.0)) for t in tg)
        lo, hi = (mid, hi) if need < english_rows else (lo, mid)
    level = (lo + hi) / 2
    x = {t: max(0.0, level - base_totals.get(t, 0.0)) for t in tg}
    z = sum(x.values()) or 1.0
    return {t: v / z for t, v in x.items()}


def split_counts(n: int, weights: dict[str, float]) -> dict[str, int]:
    z = sum(weights.values())
    raw = {k: n * v / z for k, v in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[: n - sum(base.values())]:
        base[k] += 1
    return base


def part_counts(part: dict) -> tuple[int, Counter]:
    """(free rows, fixed-target rows by ref_lang) of one kept part."""
    fixed = Counter(part.get("ref", {}))
    return part["rows"] - sum(fixed.values()), fixed


# ------------------------------------------------------------------------------------------------ 3. scatter + render

def scatter(out: Path, lang: str, keep_free: dict[str, int], n_out: int, seed: int, kept_name: str) -> None:
    """Shuffle a language's rows across `n_out` shards. Rows with a reference are always kept; free rows are
    sub-sampled to `keep_free[file]` when that is set."""
    import pyarrow.parquet as pq
    tmp = out / "tmp" / lang
    shutil.rmtree(tmp, ignore_errors=True)
    for p in sorted((out / kept_name / lang).glob("[0-9]*.parquet")):
        t = pq.read_table(p)
        rng = np.random.default_rng([seed, zlib.crc32(f"{lang}/{p.name}".encode())])
        has_ref = np.array([bool(r) for r in t.column("ref").to_pylist()], dtype=bool)
        idx_ref, idx_free = np.flatnonzero(has_ref), np.flatnonzero(~has_ref)
        want = keep_free.get(p.name, len(idx_free))
        idx_free = rng.permutation(idx_free)[:want]
        idx = rng.permutation(np.concatenate([idx_ref, idx_free]))
        for i, chunk in enumerate(np.array_split(idx, n_out)):
            d = tmp / f"{i:05d}"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(t.take(np.sort(chunk)), d / p.name, compression="zstd")


def render_shard(out: Path, lang: str, i: int, free_weights: dict[str, float], seed: int, tag: str, stats: dict) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .build import rng_for
    from .instructions import compose_prompt, input_lang_tags, instruction_languages, make_instruction
    from .sources import AFRICOMET_SUPPORTED
    t = pa.concat_tables([pq.read_table(p) for p in sorted((out / "tmp" / lang / f"{i:05d}").glob("*.parquet"))])
    n = t.num_rows
    rng = rng_for(seed, "render", lang, i)
    order = list(range(n))
    rng.shuffle(order)
    t = t.take(order)
    text, kind, lic, src = (t.column(c).to_pylist() for c in ("text", "kind", "license", "source"))
    ref, ref_lang = t.column("ref").to_pylist(), t.column("ref_lang").to_pylist()
    ks, toks = t.column("n_sentences").to_pylist(), t.column("tokens").to_pylist()
    free_idx = [j for j in range(n) if not ref[j]]
    pool = [k for k, c in split_counts(len(free_idx), free_weights).items() for _ in range(c)] if free_idx else []
    rng.shuffle(pool)
    targets = [ref_lang[j] if ref[j] else None for j in range(n)]
    for j, tg in zip(free_idx, pool):
        targets[j] = tg
    il = instruction_languages()
    ins_langs = [il[j % len(il)] for j in range(n)]
    rng.shuffle(ins_langs)
    cols = {k: [] for k in FINAL_COLUMNS}
    for j in range(n):
        tgt = targets[j]
        ins = make_instruction(ins_langs[j], lang, tgt, rng)
        sup = lang in AFRICOMET_SUPPORTED and tgt in AFRICOMET_SUPPORTED
        prompt_j = compose_prompt(ins, text[j], rng)
        for k, v in (("instruction", ins.text), ("input", text[j]), ("response", ref[j] or ""), ("source_lang", lang),
                     ("targ_lang", tgt), ("id", f"{lang}-{tag}-{i:04d}-{j:06d}"),
                     ("prompt", prompt_j), ("instruction_position", ins.position),
                     ("instruction_language", ins.language), ("template_confidence", ins.confidence),
                     ("n_sentences", ks[j]), ("src_tokens", toks[j]), ("source_dataset", src[j]),
                     ("source_license", lic[j]), ("source_kind", kind[j]), ("africomet_supported", sup),
                     # what the policy must emit before the translation, and the exact string that is tokenized as the prompt
                     ("output_structure", f"<|input_lang|>{input_lang_tags(lang, ins.language, ins.position)}<task_plan><translate></task_plan><|target_lang|><{tgt}><response>"),
                     ("text", f"<s><|user|>{prompt_j}</s><|assistant|>")):
            cols[k].append(v)
        stats["pairs"][f"{lang}>{tgt}"] += 1
    stats["rows"][lang] += n
    stats["with_response"][lang] += sum(1 for r in ref if r)
    for k, name in (("source_kind", "kind"), ("source_license", "license"), ("source_dataset", "source"),
                    ("instruction_language", "ins_lang"), ("instruction_position", "position"),
                    ("n_sentences", "k"), ("targ_lang", "target"), ("africomet_supported", "supported")):
        stats[name].update(cols[k])
    dest = out / "final" / lang / f"train-{tag}-{i:05d}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dest, compression="zstd")
    return dest


def new_stats() -> dict:
    return {k: Counter() for k in ("pairs", "rows", "with_response", "kind", "license", "source", "ins_lang", "position",
                                   "k", "target", "supported")}


def stats_to_json(stats: dict) -> dict:
    return {k: {str(a): b for a, b in v.items()} for k, v in stats.items()}


# ------------------------------------------------------------------------------------------------ orchestration

def assemble(out: Path, total: int, seed: int, tag: str, src_dirs: dict[str, str], langs: list[str] | None = None,
             limit_per_lang: int | None = None, on_language_done=None, max_non_eng_per_lang: int | None = None) -> dict:
    """`src_dirs` maps language -> 'windows' (lenient languages, already final) or 'clean' (strictly cleaned)."""
    langs = langs or LANGS
    print("== 1/3 de-duplicating", flush=True)
    dd = {}
    for lang in langs:
        d = src_dirs.get(lang, "windows")
        if (out / d / lang).exists():
            dd[lang] = dedupe_language(out, lang, src_dir=d)
            n_ref = sum(sum(p.get("ref", {}).values()) for p in dd[lang]["parts"])
            print(f"   {lang}: {dd[lang]['in']:,} -> {dd[lang]['kept']:,} windows after de-duplication ({n_ref:,} with a reference)", flush=True)

    cap = lambda n: min(n, max_non_eng_per_lang or 10**12, limit_per_lang or 10**12)
    scale = {l: (cap(d["kept"]) / d["kept"] if d["kept"] else 1.0) for l, d in dd.items() if l != "eng"}
    fixed = Counter()                       # rows whose target is fixed by their reference, per target language
    free = {}                               # rows to be assigned a target, per non-English source language
    keep_free: dict[str, dict[str, int]] = {l: {} for l in dd}
    for l, d in dd.items():
        for p in d["parts"]:
            f_free, f_fixed = part_counts(p)
            if l == "eng":
                fixed.update(f_fixed)       # English rows with references are kept whole
                continue
            sc = scale[l]
            fixed.update({k: int(round(v * sc)) for k, v in f_fixed.items()})
            free[l] = free.get(l, 0) + int(round(f_free * sc))
            if sc < 1.0:
                keep_free[l][p["file"]] = int(round(f_free * sc))
    non_eng_rows = sum(int(round((d["kept"]) * scale[l])) for l, d in dd.items() if l != "eng")
    eng = dd.get("eng")
    eng_fixed_rows = sum(sum(part_counts(p)[1].values()) for p in eng["parts"]) if eng else 0
    eng_free_avail = sum(part_counts(p)[0] for p in eng["parts"]) if eng else 0
    eng_free = max(0, min(eng_free_avail, total - non_eng_rows - eng_fixed_rows, limit_per_lang or 10**12)) if eng else 0

    base = Counter({t: float(v) for t, v in fixed.items()})
    for l, n in free.items():
        w = target_weights_free(l)
        z = sum(w.values())
        for t, wt in w.items():
            base[t] += n * wt / z
    eng_w = english_weights(base, eng_free) if eng_free else {}
    print(f"== 2/3 plan: non-English {non_eng_rows:,} + English with reference {eng_fixed_rows:,} + English free {eng_free:,} "
          f"= {non_eng_rows + eng_fixed_rows + eng_free:,} (English free available {eng_free_avail:,})", flush=True)
    if eng and eng_free < eng_free_avail:
        frac = eng_free / eng_free_avail
        for p in eng["parts"]:
            f_free, _ = part_counts(p)
            keep_free["eng"][p["file"]] = int(round(f_free * frac))

    stats = new_stats()
    for lang in langs:
        if lang not in dd:
            continue
        d = dd[lang]
        kept_name = f"kept_{src_dirs.get(lang, 'windows')}"
        if lang == "eng":
            n_rows = eng_fixed_rows + eng_free
            w = eng_w
        else:
            n_rows = int(round(d["kept"] * scale[lang]))
            w = target_weights_free(lang)
        if n_rows == 0:
            continue
        n_out = max(1, -(-n_rows // ROWS_PER_OUT_SHARD))
        print(f"== 3/3 {lang}: {n_rows:,} rows -> {n_out} shard(s)", flush=True)
        scatter(out, lang, keep_free[lang], n_out, seed, kept_name)
        files = [render_shard(out, lang, i, w, seed, tag, stats) for i in range(n_out)]
        shutil.rmtree(out / "tmp" / lang, ignore_errors=True)
        if on_language_done:
            on_language_done(lang, files)
    (out / "final").mkdir(exist_ok=True)
    (out / "final" / f"stats-{tag}.json").write_text(json.dumps(stats_to_json(stats), indent=1))
    return stats
