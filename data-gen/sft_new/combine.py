"""Run every converter, deduplicate, and publish ONE combined SFT repository (BeardedMonster/sft-all).

    nice -n 10 python -m sft_new.combine --list
    nice -n 10 python -m sft_new.combine --run [NAME ...] --out DIR [--limit N] [--push] [--repo BeardedMonster/sft-all]
    nice -n 10 python -m sft_new.combine --stats DIR                     # language x task statistics of what was written

Per converter: stream -> finish (assemble + render) -> exact dedupe against everything already written (sqlite hash table) -> parquet shards
of 100k rows under DIR/data/<converter>/ -> push that folder in ONE commit -> delete it. Nothing is kept on disk but the hash table and
the small stats files. The Hugging Face cache is emptied after every converter (`hubio.clear_cache`, and the process runs with
HF_HOME pointed at a scratch folder, so the user's own caches are never touched).

Disk guard: before and while writing, the folder is compared with GUARD (35%) of the free space measured at start; the run stops
before it would exceed that, so the machine is never filled.

Columns: id, task, source_dataset, source_lang, targ_lang, language, instruction, input, response, messages, text, output_structure, metadata
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

GUARD = 0.35
SHARD_ROWS = 100_000
REPO = "BeardedMonster/sft-all"


def folder_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


class DiskGuard:
    def __init__(self, root: Path, fraction: float = GUARD):
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.budget = root, int(shutil.disk_usage(root).free * fraction)
        print(f"disk guard: {shutil.disk_usage(root).free / 1e9:.0f} GB free, budget {self.budget / 1e9:.0f} GB ({fraction:.0%})", file=sys.stderr)

    def check(self):
        used = folder_bytes(self.root)
        if used > self.budget:
            raise RuntimeError(f"disk guard: {used / 1e9:.1f} GB written under {self.root}, budget {self.budget / 1e9:.1f} GB")


class Shards:
    def __init__(self, folder: Path):
        self.dir, self.rows, self.n, self.total = folder, [], 0, 0
        folder.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict):
        self.rows.append(row)
        self.total += 1
        if len(self.rows) >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        from sft_new.compose import COLUMNS
        tb = pa.table({c: [r[c] for r in self.rows] for c in COLUMNS}, schema=pa.schema([(c, pa.string()) for c in COLUMNS]))
        pq.write_table(tb, self.dir / f"part-{self.n:05d}.parquet", compression="zstd")
        self.n += 1
        self.rows = []


class LateShards:
    """Plain-text leftovers (text/lang/score, the pretrain-new schema) written per language under DIR/late/<lang>/."""

    def __init__(self, folder: Path, rows: int = 100_000):
        self.dir, self.buf, self.n, self.rows, self.rows_total = folder, {}, collections.Counter(), rows, 0

    def add(self, item, name: str):
        self.buf.setdefault(item.lang, []).append(item)
        self.rows_total += 1
        if len(self.buf[item.lang]) >= self.rows:
            self.flush(item.lang, name)

    def flush(self, lang: str, name: str):
        rows = self.buf.pop(lang, [])
        if not rows:
            return
        d = self.dir / lang
        d.mkdir(parents=True, exist_ok=True)
        tb = pa.table({"text": [r.text for r in rows], "lang": [lang] * len(rows), "score": pa.array([r.score for r in rows], pa.float32())})
        pq.write_table(tb, d / f"late-{name}-{self.n[(name, lang)]:05d}.parquet", compression="zstd")
        self.n[(name, lang)] += 1

    def flush_all(self, name: str):
        for lang in list(self.buf):
            self.flush(lang, name)


class Reservoir:
    """Uniform reservoir sample of `per_key` items for every key (here: source dataset + language pair)."""

    def __init__(self, per_key: int):
        self.k, self.items, self.seen = per_key, {}, collections.Counter()

    def offer(self, key: str, item: dict, rng: random.Random):
        self.seen[key] += 1
        buf = self.items.setdefault(key, [])
        if len(buf) < self.k:
            buf.append(item)
        else:
            j = rng.randrange(self.seen[key])
            if j < self.k:
                buf[j] = item

    def dump(self, path: Path):
        if not self.items:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for key, buf in self.items.items():
                for it in buf:
                    f.write(json.dumps({**it, "population": self.seen[key]}, ensure_ascii=False) + "\n")


def split_name(name: str) -> tuple[str, str | None]:
    base, _, fam = name.partition("@")
    return base, fam or None


def safe(name: str) -> str:
    """A folder/file name for a build step: readable prefix + short hash (family names contain brackets, commas and quotes)."""
    import hashlib
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80] + "-" + hashlib.sha1(name.encode()).hexdigest()[:6]


def run_one(name: str, out: Path, seen, guard: DiskGuard, limit: int | None, push: bool, repo: str, seed: int = 0) -> dict:
    """One build step: stream -> convert -> dedupe -> shards -> push -> delete. `base@family` runs one source family of a naija-style converter."""
    from sft_new import converters as C
    from sft_new import hubio
    from sft_new.compose import Late, Skip, finish
    base, fam = split_name(name)
    fn, desc = C.CONVERTERS[base]
    tag = safe(name)
    rng = random.Random(f"{seed}-{name}")
    seen.forget(tag)                                        # a re-run first removes what an interrupted attempt had already recorded
    shutil.rmtree(out / "data" / tag, ignore_errors=True)
    sh = Shards(out / "data" / tag)
    late = LateShards(out / "late")
    st = collections.Counter()
    sample = Reservoir(per_key=40)                         # translation rows kept for the AfriCOMET-QE audit (sft_new.audit_translation)
    t0 = time.time()
    C.FILE_FAMILY = fam
    try:
        for item in fn(rng, limit):
            if isinstance(item, Skip):
                st["drop:" + str(item)[:60]] += 1
                continue
            if isinstance(item, Late):
                h = item.text[:4000]
                if not seen.add(__import__("hashlib").blake2b(h.encode(), digest_size=8).hexdigest(), tag):
                    st["drop:late_duplicate"] += 1
                    continue
                late.add(item, tag)
                st[f"late:{item.lang}"] += 1
                if late.rows_total % 20_000 == 0:
                    guard.check()
                continue
            try:
                row = finish(item, rng)
            except Skip as e:
                st["drop:" + str(e)[:60]] += 1
                continue
            if not seen.add(row["id"], tag):
                st["drop:duplicate"] += 1
                continue
            st[f"kept:{row['task']}"] += 1
            st[f"lang:{row['source_lang']}>{row['targ_lang']}"] += 1
            st[f"ninput:{len(json.loads(row['metadata']).get('input_langs', [])) or 1}"] += 1
            if row["task"] == "translation" and len(row["input"]) <= 500 and len(row["response"]) <= 500:
                sample.offer(f"{row['source_dataset']}|{row['source_lang']}>{row['targ_lang']}", {k: row[k] for k in ("source_dataset", "source_lang", "targ_lang", "input", "response")}, rng)
            sh.add(row)
            if sh.total % 20_000 == 0:
                guard.check()
    finally:
        C.FILE_FAMILY = None
    sh.flush()
    late.flush_all(tag)
    kept = sh.total
    res = {"converter": name, "description": desc, "kept": kept, "late_rows": late.rows_total, "seconds": round(time.time() - t0), "counts": dict(st)}
    if push and kept:
        from huggingface_hub import HfApi
        api = HfApi(token=hubio.token())
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(sh.dir), path_in_repo=f"data/{tag}", commit_message=f"{name}: {kept:,} rows")
    if push:
        shutil.rmtree(sh.dir, ignore_errors=True)
    if push and late.rows_total:
        from huggingface_hub import HfApi
        api = HfApi(token=hubio.token())
        for lang_dir in sorted((out / "late").glob("*")):
            api.upload_folder(repo_id="BeardedMonster/pretrain-new", repo_type="dataset", folder_path=str(lang_dir), path_in_repo=f"late/{lang_dir.name}",
                              commit_message=f"late/{lang_dir.name}: plain text left over from {name}")
            shutil.rmtree(lang_dir, ignore_errors=True)
    (out / "stats").mkdir(exist_ok=True)
    sample.dump(out / "audit_samples" / f"{tag}.jsonl")
    (out / "stats" / f"{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))       # written LAST: its presence means "pushed, done"
    hubio.clear_cache()
    print(f"{name}: kept {kept:,} (+{late.rows_total:,} late) in {res['seconds']}s   top drops: {[(k, v) for k, v in st.most_common(40) if k.startswith('drop')][:3]}",
          file=sys.stderr, flush=True)
    return res


def plan(with_late: bool) -> list[str]:
    """Build order. Sources whose quality we vouch for come first so that, when two sources hold the same pair, the better provenance wins the dedupe;
    the naija/naija2 families (one step each) come last, the plain-text *_monolingual families (pretraining leftovers) after everything else."""
    from sft_new import converters as C
    first = ["sft_new_existing", "datagen_sft", "old_hate", "old_intent", "old_topic", "old_pos", "old_tasks", "old_summarization", "old_african_sft", "old_eng_sft",
             "rag_synthetic", "global_mmlu_dev", "mmlu_prox_validation", "mmmlu_yoruba", "igbo_sentiment", "hausa_voa_topics", "gpt_teacher_yo", "nvidia_steer_yo",
             "roleplay_hausa", "apollo_moe", "wiki_yo", "gsm8k", "tinystories", "storyteller", "coqa_stories", "english_to_igbo", "webscraped_igbo", "efik_bible",
             "low_resource_multilingual", "aya_collection", "aya_translate", "aya_interlanguage", "tds"]
    rest = [n for n in C.CONVERTERS if n not in first and not n.startswith("naija") and n not in ("xp3x_kongo", "sumnews")]            # xp3x_kongo / sumnews contain no supported-language rows
    names = [n for n in first + rest if n in C.CONVERTERS] + C.families("naija") + C.families("naija2")
    if with_late:
        names += C.families("naija_late") + C.families("naija2_late")
    return names


def main():
    from sft_new import converters as C
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the build order and exit")
    ap.add_argument("--all", action="store_true", help="run the whole plan (resumable: finished steps are skipped)")
    ap.add_argument("--with-late", action="store_true", help="also run the *_monolingual (pretraining leftover) families, last")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    if a.list:
        for k, (_, d) in C.CONVERTERS.items():
            print(f"{k:26s} {d}")
        return
    if a.plan:
        for i, n in enumerate(plan(a.with_late)):
            print(i, n)
        return
    if a.stats:
        return stats(a.stats)
    if not ((a.run or a.all) and a.out):
        ap.error("--run / --all need --out")
    from sft_new.dedup import Seen
    if not os.environ.get("HF_HOME"):
        sys.exit("set HF_HOME to a scratch folder first, so the Hugging Face cache can be deleted without touching your own")
    guard = DiskGuard(a.out)
    seen = Seen(a.out / "seen.sqlite")
    names = a.names or (plan(a.with_late) if a.all else list(C.CONVERTERS))
    failed = []
    try:
        for name in names:
            guard.check()
            if (a.out / "stats" / f"{safe(name)}.json").exists() and not a.force:
                print(f"{name}: already done", file=sys.stderr, flush=True)
                continue
            for attempt in range(3):
                try:
                    run_one(name, a.out, seen, guard, a.limit, a.push, a.repo)
                    break
                except RuntimeError:
                    raise                                           # the disk guard: stop everything
                except Exception as e:                              # network hiccup etc.: retry the step from its start, then move on
                    print(f"{name}: attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
                    time.sleep(30 * (attempt + 1))
            else:
                failed.append(name)
    finally:
        seen.close()
        if failed:
            (a.out / "FAILED_STEPS.json").write_text(json.dumps(failed, indent=1))
            print("failed steps:", failed, file=sys.stderr)


def stats(out: Path):
    tasks, langs, ds = collections.Counter(), collections.Counter(), collections.Counter()
    for f in sorted((out / "stats").glob("*.json")):
        j = json.loads(f.read_text())
        for k, v in j["counts"].items():
            if k.startswith("kept:"):
                tasks[k[5:]] += v
            elif k.startswith("lang:"):
                langs[k[5:]] += v
        ds[j["converter"]] = j["kept"]
    print(json.dumps({"total": sum(ds.values()), "by_converter": dict(ds.most_common()), "by_task": dict(tasks.most_common()),
                      "by_language_pair": dict(langs.most_common(60))}, indent=1))


if __name__ == "__main__":
    main()
