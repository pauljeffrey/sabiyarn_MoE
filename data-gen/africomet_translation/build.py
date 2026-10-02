"""Build translation prompts for AfriCOMET-scored RLHF and push them to the Hub as they are produced.

    python -m africomet_translation.build build --per-language 12 --out OUT --cache CACHE              # pilot, local only
    python -m africomet_translation.build build --per-language 5000 --out OUT --cache CACHE --push     # full run
    python -m africomet_translation.build card  --out OUT                                              # print the dataset card
    python -m africomet_translation.build cleanup --out OUT --cache CACHE                              # verify push, then delete the downloads

Run from data-gen/. No LLM is involved anywhere: the text is real, the instructions come from the template banks.
All source downloads go into --cache (never the shared HF cache) so `cleanup` is one targeted delete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "BeardedMonster/africomet-translation"
TOKENIZER = "Davlan/afro-xlmr-large-76L"       # the encoder AfriCOMET-1.1 (stl and qe-stl) is built on
SHARD_SIZE = 1000


def rng_for(*parts) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def hf_token() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")


class Tok:
    def __init__(self, token: str | None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER, token=token)
        self.tok.model_max_length = 10**9      # we enforce our own cap; silence the 512 warning

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=True, truncation=False)["input_ids"])


@dataclass
class Window:
    text: str
    n_sentences: int
    tokens: int
    source: str
    license: str


def fit_window(sentences: list[str], tok: Tok, cap: int):
    """Longest prefix of the window that fits `cap` AfriCOMET-encoder tokens; None if even sentence 1 does not."""
    s = list(sentences)
    while s:
        text = " ".join(s)
        n = tok.count(text)
        if n <= cap:
            return s, text, n
        s.pop()
    return None


def clean_runs(doc, src, seen: set[str]) -> list[list[str]]:
    """Normalise a doc's sentences and cut it into runs of consecutive good, previously-unseen sentences.
    A dropped sentence ends a run, so a window never silently skips over a hole."""
    from .text_utils import dedupe_key, good_sentence, normalize
    runs, cur = [], []
    for raw in doc.sentences:
        s = normalize(raw)
        key = dedupe_key(s)
        ok = bool(s) and key not in seen and good_sentence(
            s, min_words=src.min_words, max_words=src.max_words, require_terminal=src.require_terminal)
        if ok:
            seen.add(key)
            cur.append(s)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def finite_windows(src, ctx, seed: int, lang: str, seen: set[str], tok: Tok, cap: int) -> list[Window]:
    from .text_utils import interleave_windows, partition_windows
    rng = rng_for(seed, "windows", lang, src.name)
    per_doc = [partition_windows(run, rng) for doc in src.load(ctx) for run in clean_runs(doc, src, seen)]
    out = []
    for w in interleave_windows(per_doc, rng):
        fitted = fit_window(w, tok, cap)
        if fitted:
            s, text, n = fitted
            out.append(Window(text, len(s), n, src.name, src.license))
    return out


def stream_windows(src, ctx, seed: int, lang: str, seen: set[str], tok: Tok, cap: int, take: int):
    """Streamed web text is plentiful, so choose window sizes to be uniform over 1..5 (largest deficit first,
    among the sizes the paragraph is long enough for) and spread the quota over as many documents as possible."""
    import math
    from collections import Counter as _C
    rng = rng_for(seed, "stream", lang, src.name)
    per_k, counts, emitted = math.ceil(take / 5), _C(), 0
    for doc in src.load(ctx):
        for run in clean_runs(doc, src, seen):
            if emitted >= take:
                return
            avail = [k for k in range(1, min(5, len(run)) + 1) if counts[k] < per_k]
            if not avail:
                continue
            low = min(counts[k] for k in avail)
            k = rng.choice([k for k in avail if counts[k] == low])
            start = rng.randint(0, len(run) - k)
            fitted = fit_window(run[start:start + k], tok, cap)
            if fitted:
                s, text, n = fitted
                counts[len(s)] += 1
                emitted += 1
                yield Window(text, len(s), n, src.name, src.license)


def build_language(lang: str, n: int, seed: int, ctx, tok: Tok, cap: int) -> tuple[list[Window], dict]:
    from .sources import SOURCES
    from .text_utils import allocate, balance_by_k
    sources = SOURCES[lang]
    seen: set[str] = set()
    finite = {s.name: finite_windows(s, ctx, seed, lang, seen, tok, cap) for s in sources if not s.streaming}
    caps = {s.name: (len(finite[s.name]) if not s.streaming else n) for s in sources}
    alloc = allocate(n, {s.name: s.weight for s in sources}, caps)
    chosen: list[Window] = []
    for s in sources:
        take = alloc[s.name]
        if s.streaming:
            chosen += list(stream_windows(s, ctx, seed, lang, seen, tok, cap, take))
        else:
            chosen += balance_by_k(finite[s.name], take)
    rng_for(seed, "shuffle", lang).shuffle(chosen)
    stats = {
        "language": lang, "requested": n, "produced": len(chosen),
        "available_per_source": {k: len(v) for k, v in finite.items()},
        "taken_per_source": dict(Counter(w.source for w in chosen)),
    }
    return chosen, stats


def _cycle(items: list[str], rng: random.Random):
    while True:
        batch = items[:]
        rng.shuffle(batch)
        yield from batch


def make_records(lang: str, windows: list[Window], seed: int, tag: str) -> list[dict]:
    from .instructions import compose_prompt, instruction_languages, make_instruction
    from .sources import AFRICOMET_SUPPORTED, LANGUAGE_ORDER
    rng = rng_for(seed, "records", lang)
    targets = _cycle([l for l in LANGUAGE_ORDER if l != lang], rng)      # every pair gets even coverage
    ins_langs = _cycle(instruction_languages(), rng)
    records = []
    for i, w in enumerate(windows):
        tgt = next(targets)
        ins = make_instruction(next(ins_langs), lang, tgt, rng)
        records.append({
            "id": f"{lang}-{tag}-{i:06d}",
            "instruction": ins.text,
            "input": w.text,
            "source_language": lang,
            "target_language": tgt,
            "prompt": compose_prompt(ins, w.text, rng),
            "instruction_position": ins.position,
            "instruction_language": ins.language,
            "template_confidence": ins.confidence,
            "n_sentences": w.n_sentences,
            "src_tokens": w.tokens,
            "source_dataset": w.source,
            "source_license": w.license,
            "africomet_supported": lang in AFRICOMET_SUPPORTED and tgt in AFRICOMET_SUPPORTED,
        })
    return records


def write_shards(out: Path, lang: str, tag: str, records: list[dict]) -> list[Path]:
    d = out / "data" / lang
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(records), SHARD_SIZE):
        p = d / f"shard-{tag}-{i // SHARD_SIZE:03d}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in records[i:i + SHARD_SIZE]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------------------------------------------
# Hub
# ----------------------------------------------------------------------------------------------------------------

def _api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def remote_files(api, repo_id: str) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception:
        return set()


def push_files(api, repo_id: str, files: dict[str, Path | bytes], message: str, private: bool = True) -> int:
    from huggingface_hub import CommitOperationAdd
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    have = remote_files(api, repo_id)
    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src) if isinstance(src, Path) else src)
           for dest, src in files.items() if dest not in have or dest == "README.md"]
    if not ops:
        return 0
    api.create_commit(repo_id, repo_type="dataset", operations=ops, commit_message=message)
    return len(ops)


def push_language(api, repo_id: str, out: Path, lang: str, private: bool) -> int:
    files = {f"data/{lang}/{p.name}": p for p in sorted((out / "data" / lang).glob("shard-*.jsonl"))}
    n = push_files(api, repo_id, files, f"Add {lang} prompts ({len(files)} shard(s))", private)
    print(f"[hub] {lang}: {n} new shard(s) pushed to {repo_id}")
    return n


# ----------------------------------------------------------------------------------------------------------------
# Dataset card
# ----------------------------------------------------------------------------------------------------------------

def load_all_stats(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((out / "stats").glob("*.json"))]


def build_card(out: Path, max_src_tokens: int) -> str:
    from .instructions import CONFIDENCE, instruction_languages
    stats = load_all_stats(out)
    recs_by_lang, lic_by_src, k_hist = {}, {}, Counter()
    ins_lang, ins_pos, pairs = Counter(), Counter(), Counter()
    total = 0
    for p in sorted((out / "data").glob("*/shard-*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            total += 1
            recs_by_lang[r["source_language"]] = recs_by_lang.get(r["source_language"], 0) + 1
            lic_by_src[(r["source_dataset"], r["source_license"])] = lic_by_src.get((r["source_dataset"], r["source_license"]), 0) + 1
            k_hist[r["n_sentences"]] += 1
            ins_lang[r["instruction_language"]] += 1
            ins_pos[r["instruction_position"]] += 1
            pairs[(r["source_language"], r["target_language"])] += 1
    lines = [
        "---", "license: other", "license_name: mixed-per-record-see-card",
        "language:", *[f"- {c}" for c in ("en", "yo", "ha", "ig", "pcm", "ee", "fon", "tw", "efi", "ibb")],
        "pretty_name: AfriCOMET translation prompts",
        "task_categories:", "- translation", "- text-generation",
        "tags:", "- rlhf", "- africomet", "- translation", "- african-languages", "- prompts",
        "configs:", "- config_name: default", "  data_files:", "  - split: train", "    path: data/*/*.jsonl",
        "---", "",
        "# AfriCOMET translation prompts", "",
        f"{total:,} translation prompts for RLHF with [AfriCOMET](https://huggingface.co/masakhane/africomet-qe-stl-1.1) as the "
        "reward. Each row is a real passage of 1-5 consecutive sentences in a source language plus a translation "
        "instruction naming a different target language. There is **no reference translation**: the policy "
        "translates, AfriCOMET (reference-free QE variant) scores `input` against the translation.", "",
        "No language model wrote any text here. Passages come from the datasets listed below; instructions come "
        "from hand-written template banks (`africomet_translation/instructions.py` in the SabiYarn repo).", "",
        "## Fields", "",
        "| field | meaning |", "|---|---|",
        "| `instruction` | the translation request, in `instruction_language`, naming the target language |",
        "| `input` | the source passage (1-5 consecutive sentences, same document) |",
        "| `source_language`, `target_language` | codes: `eng pcm yor hau ibo fon ewe twi efi ibb` |",
        "| `prompt` | `instruction` and `input` already joined, in the position given by `instruction_position` |",
        "| `instruction_position` | `prepend` (instruction first) or `append` (instruction after the text) |",
        "| `instruction_language`, `template_confidence` | language the instruction is written in, and how reliable that bank is (`high`/`medium`/`low`) |",
        "| `n_sentences`, `src_tokens` | sentences in `input`; its length in `Davlan/afro-xlmr-large-76L` tokens |",
        "| `source_dataset`, `source_license` | provenance of `input`; **licenses differ per row** |",
        "| `africomet_supported` | true only when both languages are in AfriCOMET-1.1's encoder language list |", "",
        "## Sources and licenses", "",
        "The dataset as a whole is only as permissive as its most restrictive rows. **Rows from `howard-nlp/ibom-mt` "
        "are CC-BY-NC-4.0 (non-commercial)**; filter on `source_license` to exclude them.", "",
        "| source | license | rows |", "|---|---|---|",
        *[f"| `{s}` | {l} | {c:,} |" for (s, l), c in sorted(lic_by_src.items())], "",
        "## Rows per source language", "", "| language | rows |", "|---|---|",
        *[f"| `{l}` | {c:,} |" for l, c in sorted(recs_by_lang.items())], "",
        "## Design notes", "",
        f"* **Length.** AfriCOMET truncates each of source / translation at 512 tokens. `input` is capped at "
        f"{max_src_tokens} tokens of the AfriCOMET-1.1 tokenizer, leaving headroom for the policy's translation, which "
        "tokenizes longer in several of these languages.",
        "* **Window size** is drawn uniformly from 1-5 sentences, then limited by the document: sources made of very "
        "short articles (FLORES-style, 2-5 sentences) cannot supply long windows, so their size distribution skews "
        "small. Observed overall: " + ", ".join(f"{k} sentence(s): {v:,}" for k, v in sorted(k_hist.items())) + ".",
        "* **Targets** cycle evenly over the nine other languages, so every source/target pair is covered about equally.",
        f"* **Instruction placement:** {dict(ins_pos)}. **Instruction language:** {dict(ins_lang)}.", "",
        "## Limitations", "",
        "* **Ibibio (`ibb`) is not in AfriCOMET-1.1's language list** (Efik is). Rows involving it have "
        "`africomet_supported=false`; the reward there is unvalidated.",
        "* **Instruction templates were not reviewed by native speakers.** Confidence: "
        + ", ".join(f"`{l}` {CONFIDENCE.get(l, 'none')}" for l in ("eng", "pcm", "yor", "hau", "ibo", "twi", "ewe"))
        + ". **Fon, Efik and Ibibio have no instruction templates at all** (never used as `instruction_language`). "
        "Filter on `template_confidence` before relying on the low tier.",
        "* **FLORES-200 / SIB-200 sentences** (`flores_plus`, `ibom-mt` dev/test, `sib200-ibom`) are standard "
        "evaluation sets. Do not evaluate on FLORES after training on these prompts without accounting for the overlap.",
        "* Efik and Ibibio sources mix the `n̄` and `ñ` spellings of the same nasal; text is NFC-normalised only.",
        "* `ibom-mt` and the Efik speech transcripts contain occasional translation or typing errors (e.g. a "
        "mistyped year); nothing here was proofread.",
        "* Twi comes from FLORES in both Akuapem and Asante varieties, i.e. the same sentences twice.", "",
        "## Reproduction", "",
        "`cd data-gen && python -m africomet_translation.build build --per-language 5000 --push` "
        "(SabiYarn repository). Deterministic given `--seed`.", "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------

def cmd_build(a) -> int:
    out, cache = Path(a.out), Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    from .sources import LANGUAGE_ORDER, SOURCES, Ctx
    token = hf_token()
    ctx = Ctx(cache=cache, token=token, seed=a.seed)
    langs = a.languages.split(",") if a.languages else LANGUAGE_ORDER
    bad = [l for l in langs if l not in SOURCES]
    if bad:
        raise SystemExit(f"unknown language(s) {bad}; known: {list(SOURCES)}")
    api = _api(token) if a.push else None
    tok = Tok(token)
    (out / "stats").mkdir(exist_ok=True)
    for lang in langs:
        marker = out / "stats" / f"{lang}.json"
        if marker.exists() and not a.force:
            print(f"[{lang}] already built (use --force to rebuild)")
        else:
            t = time.time()
            windows, stats = build_language(lang, a.per_language, a.seed, ctx, tok, a.max_src_tokens)
            records = make_records(lang, windows, a.seed, a.run_tag)
            write_shards(out, lang, a.run_tag, records)
            stats.update(seconds=round(time.time() - t, 1), max_src_tokens=a.max_src_tokens, run_tag=a.run_tag,
                         sentences_hist=dict(Counter(r["n_sentences"] for r in records)))
            marker.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
            flag = "" if stats["produced"] >= a.per_language else f"  (SHORT of {a.per_language})"
            print(f"[{lang}] {stats['produced']} prompts in {stats['seconds']}s{flag}  {stats['taken_per_source']}")
        if api:
            push_language(api, REPO_ID, out, lang, private=not a.public)
    if api:
        card = build_card(out, a.max_src_tokens).encode("utf-8")
        push_files(api, REPO_ID, {"README.md": card}, "Update dataset card", private=not a.public)
        print(f"[hub] dataset card updated -> https://huggingface.co/datasets/{REPO_ID}")
    return 0


def cmd_cleanup(a) -> int:
    out, cache = Path(a.out), Path(a.cache)
    token = hf_token()
    api = _api(token)
    have = remote_files(api, REPO_ID)
    local = [f"data/{p.parent.name}/{p.name}" for p in sorted((out / "data").glob("*/shard-*.jsonl"))]
    missing = [f for f in local if f not in have]
    if not local:
        raise SystemExit("nothing built locally; refusing to clean up")
    if missing:
        raise SystemExit(f"{len(missing)} local shard(s) are NOT on the Hub yet (e.g. {missing[:3]}); not deleting anything")
    print(f"verified: all {len(local)} local shards are on {REPO_ID}")
    if cache.exists():
        size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 2**20
        shutil.rmtree(cache)
        print(f"deleted source-dataset cache {cache} ({size:.0f} MB)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--languages", default="", help="comma list; default all")
    b.add_argument("--per-language", type=int, default=5000)
    b.add_argument("--max-src-tokens", type=int, default=256)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--run-tag", default="v1")
    b.add_argument("--out", required=True)
    b.add_argument("--cache", required=True)
    b.add_argument("--push", action="store_true")
    b.add_argument("--public", action="store_true", help="create the Hub repo public (default private)")
    b.add_argument("--force", action="store_true")
    c = sub.add_parser("card")
    c.add_argument("--out", required=True)
    c.add_argument("--max-src-tokens", type=int, default=256)
    d = sub.add_parser("cleanup")
    d.add_argument("--out", required=True)
    d.add_argument("--cache", required=True)
    a = ap.parse_args(argv)

    if a.cmd in ("build", "cleanup"):                       # must happen before datasets/huggingface_hub are imported
        cache = os.path.abspath(a.cache)
        os.environ["HF_HUB_CACHE"] = cache
        os.environ["HF_DATASETS_CACHE"] = cache
    if a.cmd == "build":
        return cmd_build(a)
    if a.cmd == "cleanup":
        return cmd_cleanup(a)
    print(build_card(Path(a.out), a.max_src_tokens))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
