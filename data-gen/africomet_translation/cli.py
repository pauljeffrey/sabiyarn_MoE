"""Large-scale AfriCOMET prompt build (v2).  Run from data-gen/:

    python -m africomet_translation.cli extract  --out OUT --cache CACHE [--languages yor,hau] [--only REGEX]
    python -m africomet_translation.cli status   --out OUT
    python -m africomet_translation.cli assemble --out OUT --total 10000000 [--push]
    python -m africomet_translation.cli cleanup  --out OUT --cache CACHE

`extract` is resumable (finished sources are skipped). Language-ID workers each hold a 1.6 GB model, so
`--lid-workers` is the knob to turn down if memory is tight.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _env(cache: str) -> str:
    from .build import hf_token
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["HF_DATASETS_CACHE"] = cache
    token = hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def _registry_args(a) -> dict:
    return {"eng": {"c4": a.eng_c4, "wiki": a.eng_wiki, "edu": a.eng_edu, "quota": a.eng_quota},
            "include_nc": a.include_nc}


def cmd_extract(a) -> int:
    out, cache = Path(a.out).resolve(), Path(a.cache).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    _env(str(cache))
    from .registry import build_registry, slug
    reg_args = _registry_args(a)
    registry = build_registry(**reg_args)
    langs = a.languages.split(",") if a.languages else list(registry)
    tasks = []
    for lang in langs:
        for s in registry[lang]:
            if a.only and not re.search(a.only, s.name):
                continue
            done = out / "windows" / lang / f"{slug(s.name)}.done.json"
            if done.exists() and not a.force:
                continue
            tasks.append((lang, s))
    print(f"{len(tasks)} source task(s) to run ({sum(1 for _, s in tasks if s.lid)} with language-ID)", flush=True)
    if not tasks:
        return 0
    from .extract import run_task
    cfg = {"cache": str(cache), "out": str(out), "seed": a.seed, "cap": a.max_src_tokens, "registry_args": reg_args}
    ctx = mp.get_context("spawn")
    t0, finished, errors = time.time(), 0, 0
    with ProcessPoolExecutor(a.lid_workers, mp_context=ctx) as lid_pool, ProcessPoolExecutor(a.workers, mp_context=ctx) as pool:
        futs = {(lid_pool if s.lid else pool).submit(run_task, lang, s.name, cfg): (lang, s.name) for lang, s in tasks}
        for f in as_completed(futs):
            r = f.result()
            finished += 1
            if "error" in r:
                errors += 1
                print(f"[{finished}/{len(tasks)}] FAILED {r['lang']} {r['source']}: {r['error']}\n{r.get('trace', '')}", flush=True)
            else:
                lid_note = f" lid_rejected={r.get('lid_rejected', 0)}/{r.get('lid_checked', 0)}" if r.get("lid_checked") else ""
                print(f"[{finished}/{len(tasks)}] {r['lang']} {r['source']}: {r['rows']:,} windows in {r['seconds']}s{lid_note}", flush=True)
    print(f"extract finished in {time.time() - t0:.0f}s, {errors} failed", flush=True)
    return 1 if errors else 0


def cmd_status(a) -> int:
    out = Path(a.out)
    by_lang: dict[str, int] = {}
    for p in sorted(out.glob("windows/*/*.done.json")):
        r = json.loads(p.read_text())
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + r["rows"]
        if a.verbose:
            print(f"  {r['lang']:4s} {r['rows']:>10,}  {r['source']}")
    for l, n in sorted(by_lang.items()):
        print(f"{l:4s} {n:>12,} candidate windows")
    print(f"{'all':4s} {sum(by_lang.values()):>12,}")
    return 0


def _clean_worker(args):
    from .clean import clean_file
    return clean_file(*args)


def cmd_clean(a) -> int:
    import pickle
    out = Path(a.out).resolve()
    cache = Path(a.cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    _env(str(cache))
    from .clean import THRESHOLD, train_model
    langs = a.languages.split(",") if a.languages else ["pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "ibb", "eng"]
    (out / "clean_models").mkdir(exist_ok=True)
    for lang in langs:
        p = out / "clean_models" / f"{lang}.pkl"
        if p.exists() and not a.retrain:
            continue
        t0 = time.time()
        m = train_model(out, lang)
        if m is None:
            print(f"[{lang}] too little curated or web data for a classifier; rules only", flush=True)
            continue
        p.write_bytes(pickle.dumps(m))
        print(f"[{lang}] classifier: {m['n_pos']:,} clean + {m['n_neg']:,} web; "
              f"{m['heldout_pass']:.0%} of held-out clean windows score >= {THRESHOLD} ({time.time() - t0:.0f}s)", flush=True)
    from .clean import build_lexicon
    for lang in langs:
        lp = out / "clean_models" / f"{lang}.lex"
        if lp.exists() and not a.retrain:
            continue
        lx = build_lexicon(out, lang)
        if lx is None:
            continue
        lp.write_bytes(pickle.dumps(lx))
        print(f"[{lang}] english-word lexicon: {len(lx['lexicon']):,} words, reject above {lx['threshold']:.1%} "
              f"(99.5th percentile of {lx['n_clean_ref']:,} clean windows)", flush=True)
    if a.train_only:
        return 0
    files = [(str(f), lang, str(out), a.threshold) for lang in langs for f in sorted((out / "windows" / lang).glob("*.parquet"))
             if not (out / "clean" / lang / f.name).exists() or a.force]
    print(f"cleaning {len(files)} window file(s) with {a.workers} worker(s)", flush=True)
    ctx = mp.get_context("spawn")
    agg: dict[str, Counter] = {}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as pool:
        futs = [pool.submit(_clean_worker, f) for f in files]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            agg.setdefault(r["lang"], Counter()).update({k: v for k, v in r.items() if isinstance(v, int)})
            print(f"[{i}/{len(files)}] {r['lang']} {r['file'][:50]}: {r['in']:,} -> {r['out']:,}", flush=True)
    for lang, c in agg.items():
        rej = {k[4:]: v for k, v in c.items() if k.startswith("rej:")}
        print(f"{lang}: kept {c['out']:,} of {c['in']:,} ({c['out'] / max(1, c['in']):.0%}); rejected by: "
              + ", ".join(f"{k} {v:,}" for k, v in sorted(rej.items(), key=lambda kv: -kv[1])[:5]), flush=True)
    (out / "clean").mkdir(exist_ok=True)
    (out / "clean" / "_stats.json").write_text(json.dumps({l: dict(c) for l, c in agg.items()}, indent=1))
    print(f"clean finished in {time.time() - t0:.0f}s", flush=True)
    return 0


def build_card_v2(stats: dict, tag: str, max_src_tokens: int) -> str:
    from .instructions import CONFIDENCE
    total = sum(stats["rows"].values())
    def table(counter, h1, h2="rows", top=None):
        items = sorted(counter.items(), key=lambda kv: -kv[1])[: top or None]
        return ["| %s | %s |" % (h1, h2), "|---|---|", *[f"| {k} | {v:,} |" for k, v in items], ""]
    lic = {}
    for src, n in stats["source"].items():
        lic[src] = n
    lines = [
        "---", "license: other", "license_name: mixed-per-record-see-card",
        "language:", *[f"- {c}" for c in ("en", "yo", "ha", "ig", "pcm", "ee", "fon", "tw", "efi", "ibb")],
        "pretty_name: AfriCOMET translation prompts", "size_categories:", f"- {'10M<n<100M' if total >= 10_000_000 else '1M<n<10M'}",
        "task_categories:", "- translation", "- text-generation",
        "tags:", "- rlhf", "- africomet", "- translation", "- african-languages", "- prompts",
        "configs:", "- config_name: default", "  data_files:", "  - split: train", "    path: data/*/*.parquet",
        "---", "",
        "# AfriCOMET translation prompts", "",
        f"{total:,} translation prompts for RLHF with [AfriCOMET](https://huggingface.co/masakhane/africomet-qe-stl-1.1) "
        "(reference-free QE) as the reward. Each row is a passage of 1-5 consecutive sentences from one document in a "
        "source language plus an instruction asking for a translation into a different target language. There is no "
        "reference translation: the policy translates and AfriCOMET scores the translation against `input`.", "",
        "No language model wrote any text here. Passages come from the public corpora listed below; instructions come "
        "from hand-written template banks.", "",
        "## Fields", "", "| field | meaning |", "|---|---|",
        "| `instruction`, `input` | the request (in `instruction_language`) and the source passage |",
        "| `source_language`, `target_language` | `eng pcm yor hau ibo fon ewe twi efi ibb` |",
        "| `prompt` | `instruction` and `input` joined in the order given by `instruction_position` (`prepend` / `append`) |",
        "| `template_confidence` | reliability of the instruction wording: `high` (English), `medium` (pcm/yor/hau/ibo), `low` (twi/ewe) |",
        "| `n_sentences`, `src_tokens` | sentences in `input`; length in `Davlan/afro-xlmr-large-76L` tokens (capped at "
        f"{max_src_tokens}; AfriCOMET truncates each segment at 512) |",
        "| `source_dataset`, `source_license`, `source_kind` | provenance. kind: `curated` `news` `encyclopedia` `web` `bible` |",
        "| `africomet_supported` | both languages are in AfriCOMET-1.1's encoder language list (Ibibio is not) |", "",
        "## How it was built", "",
        "* **Web text is only used after sentence-level language ID** (GlotLID v3). Measured on HPLT 'Yoruba', only ~46% of "
        "lines the corpus labels Yoruba were Yoruba. Sentences with e-mail/phone/URL, spam-like repetition, markup or "
        "non-prose shapes are dropped; sentences are de-duplicated across all sources.",
        "* **Cleaned, then filtered.** Each sentence is cleaned (verse / section numbers, scripture references, footnote "
        "marks, bullets and spacing removed), then windows are dropped for mojibake, symbols/emoji, too few letters, digits "
        "or punctuation, repetition, Title-Case spam, mixed letter-digit corruption, and English function- or content-word "
        "leakage (threshold calibrated per language on its own clean news). Web Yoruba, Igbo and English are additionally "
        "scored by a per-language classifier trained to separate hand-translated/edited prose from random web text; "
        "windows below 0.5 are dropped. This removed most machine-translated web text: **only 13% of raw Yoruba, 24% of "
        "Igbo and 67% of English windows survived**. The classifier is deliberately NOT used for Hausa, Twi and the smaller "
        "languages, where it rejected genuine news and religious prose, so those rely on the rules alone and are noisier.",
        "* **English is the augmenter.** Non-English source text is limited by what exists, so English passages fill the "
        "rest and are routed to the other nine languages so each *target* language gets about the same number of rows.",
        "* **Targets** for non-English sources cycle evenly over the nine other languages, with English as target weighted double.",
        "* Window sizes are drawn uniformly from 1-5 sentences, limited by paragraph length.", "",
        "## Rows by source language", ""] + table(stats["rows"], "language") + [
        "## Rows by target language", ""] + table(stats["target"], "target") + [
        "## Rows by source kind", ""] + table(stats["kind"], "kind") + [
        "## Sources", "", "Licenses differ per row; the dataset as a whole is only as permissive as its most restrictive rows.", "",
        "| source | rows |", "|---|---|", *[f"| `{k}` | {v:,} |" for k, v in sorted(stats["source"].items(), key=lambda kv: -kv[1])[:60]], "",
        "## Licenses", ""] + table(stats["license"], "license") + [
        "## Limitations", "",
        "* **Licenses are mixed and some are restrictive:** `howard-nlp/ibom-mt` is CC-BY-NC-4.0; Wikipedia/FLORES/SIB-200/"
        "wikianc content is share-alike; **Bible rows (`source_kind=bible`) come from translations whose Bible-society "
        "copyright was not verified**; web rows are Common Crawl derivatives whose copyright is not cleared by the "
        "corpora's CC0 label. Filter on `source_license` / `source_kind` for a stricter release.",
        "* **Ibibio is not in AfriCOMET-1.1's language list**; rows involving it have `africomet_supported=false`.",
        "* **Instruction templates are not native-reviewed** and **Fon, Efik and Ibibio have none** (never an instruction "
        "language). Confidence: " + ", ".join(f"`{l}` {c}" for l, c in CONFIDENCE.items()) + ".",
        "* The scarce languages (Fon, Efik, Ibibio, Ewe, Twi, Pidgin) are dominated by Bible text; English rows carry "
        "most of the volume. Web text for these languages may include machine-translated pages that passed language ID.",
        "* FLORES-200 / SIB-200 sentences are standard evaluation sets: do not evaluate on FLORES after training on this.",
        "* Efik and Ibibio sources mix `n̄` and `ñ`; text is NFC-normalised only. No toxicity filtering was applied.", "",
        "## Reproduction", "", "`data-gen/africomet_translation/` in the SabiYarn repository: `cli extract`, then `cli assemble`.", "",
    ]
    return "\n".join(lines)


def cmd_assemble(a) -> int:
    out = Path(a.out).resolve()
    token = _env(str(Path(a.cache).resolve())) if a.cache else _env(str(out / ".hfcache"))
    from .assemble import assemble
    from .build import REPO_ID, _api, push_files
    api = _api(token) if a.push else None
    langs = a.languages.split(",") if a.languages else None

    def on_done(lang, files):
        if not api:
            return
        for i in range(0, len(files), 6):
            batch = {f"data/{lang}/{f.name}": f for f in files[i:i + 6]}
            n = push_files(api, REPO_ID, batch, f"Add {lang} prompts {a.tag} ({len(batch)} shard(s))", private=not a.public)
            print(f"[hub] {lang}: pushed {n} shard(s)", flush=True)

    stats = assemble(out, a.total, a.seed, a.tag, langs, a.limit_per_lang, on_done, a.max_non_eng_per_lang, a.windows_dir)
    card = build_card_v2(stats, a.tag, a.max_src_tokens)
    (out / "final" / "README.md").write_text(card)
    if api:
        push_files(api, REPO_ID, {"README.md": card.encode("utf-8")}, f"Update dataset card ({a.tag})", private=not a.public)
        print(f"[hub] card updated -> https://huggingface.co/datasets/{REPO_ID}", flush=True)
    print(f"done: {sum(stats['rows'].values()):,} rows", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--out", required=True)
    e.add_argument("--cache", required=True)
    e.add_argument("--languages", default="")
    e.add_argument("--only", default="", help="regex on the source name")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--max-src-tokens", type=int, default=256)
    e.add_argument("--workers", type=int, default=5)
    e.add_argument("--lid-workers", type=int, default=3)
    e.add_argument("--force", action="store_true")
    e.add_argument("--include-nc", action="store_true", help="also use CC-BY-NC sources that are not needed for coverage")
    e.add_argument("--eng-c4", type=int, default=0)
    e.add_argument("--eng-wiki", type=int, default=0)
    e.add_argument("--eng-edu", type=int, default=0)
    e.add_argument("--eng-quota", type=int, default=300_000, help="windows per English shard")
    m = sub.add_parser("assemble")
    m.add_argument("--out", required=True)
    m.add_argument("--cache", default="")
    m.add_argument("--total", type=int, default=10_000_000)
    m.add_argument("--tag", default="v2")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--max-src-tokens", type=int, default=256)
    m.add_argument("--languages", default="")
    m.add_argument("--limit-per-lang", type=int, default=0)
    m.add_argument("--max-non-eng-per-lang", type=int, default=0)
    m.add_argument("--windows-dir", default="windows", help="'clean' to assemble from the cleaned windows")
    m.add_argument("--push", action="store_true")
    m.add_argument("--public", action="store_true")
    c = sub.add_parser("clean")
    c.add_argument("--out", required=True)
    c.add_argument("--cache", required=True)
    c.add_argument("--languages", default="")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--threshold", type=float, default=0.5)
    c.add_argument("--retrain", action="store_true")
    c.add_argument("--train-only", action="store_true")
    c.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--out", required=True)
    s.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        return cmd_extract(a)
    if a.cmd == "status":
        return cmd_status(a)
    if a.cmd == "clean":
        return cmd_clean(a)
    if a.cmd == "assemble":
        a.limit_per_lang = a.limit_per_lang or None
        a.max_non_eng_per_lang = a.max_non_eng_per_lang or None
        return cmd_assemble(a)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
