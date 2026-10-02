"""Stage A: turn each source into shards of candidate windows (1-5 consecutive sentences of one document).

One task = one (language, source). Tasks are independent processes, resumable (a `.done.json` marker per task),
and write parquet rows: text, n_sentences, tokens, sentence hashes (for cross-source de-duplication in stage B),
source, license, kind. Web text is verified sentence by sentence with GlotLID before it can become a window.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

ROWS_PER_SHARD = 200_000
LID_CHUNK = 4000
_TOK = None


def _schema():
    import pyarrow as pa
    return pa.schema([
        ("text", pa.string()), ("n_sentences", pa.int8()), ("tokens", pa.int16()), ("hashes", pa.list_(pa.int64())),
        ("source", pa.string()), ("license", pa.string()), ("kind", pa.string()),
    ])


class ShardWriter:
    def __init__(self, out_dir: Path, slug: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir, self.slug, self.files, self.total = out_dir, slug, 0, 0
        self.cols = {k: [] for k in ("text", "n_sentences", "tokens", "hashes", "source", "license", "kind")}

    def add(self, **row):
        for k, v in row.items():
            self.cols[k].append(v)
        if len(self.cols["text"]) >= ROWS_PER_SHARD:
            self.flush()

    def flush(self):
        n = len(self.cols["text"])
        if not n:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.out_dir / f"{self.slug}-{self.files:03d}.parquet"
        tmp = path.with_suffix(".tmp")
        pq.write_table(pa.table(self.cols, schema=_schema()), tmp, compression="zstd")
        tmp.rename(path)
        self.files += 1
        self.total += n
        self.cols = {k: [] for k in self.cols}

    def close(self) -> int:
        self.flush()
        return self.total


def _tok():
    global _TOK
    if _TOK is None:
        from .build import Tok
        _TOK = Tok(os.environ.get("HF_TOKEN"))
    return _TOK


def token_lengths(texts: list[str]) -> list[int]:
    return [len(x) for x in _tok().tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]


def run_task(lang: str, name: str, cfg: dict) -> dict:
    """Process one source. Returns stats; never raises (errors are reported in the result)."""
    t0 = time.time()
    try:
        return _run_task(lang, name, cfg, t0)
    except Exception as e:  # noqa: BLE001 -- one bad source must not kill the batch
        import traceback
        return {"lang": lang, "source": name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}


def _run_task(lang: str, name: str, cfg: dict, t0: float) -> dict:
    os.environ["HF_HUB_CACHE"] = cfg["cache"]
    os.environ["HF_DATASETS_CACHE"] = cfg["cache"]
    from . import lid
    from .build import rng_for
    from .registry import build_registry, slug
    from .sources import Ctx
    from .text_utils import good_sentence, normalize, partition_windows, sent_hash

    src = next(s for s in build_registry(**cfg["registry_args"])[lang] if s.name == name)
    ctx = Ctx(cache=Path(cfg["cache"]), token=os.environ.get("HF_TOKEN"), seed=cfg["seed"])
    rng = rng_for(cfg["seed"], "extract", lang, name)
    cap = cfg["cap"]
    out_dir = Path(cfg["out"]) / "windows" / lang
    writer = ShardWriter(out_dir, slug(name))
    seen: set[int] = set()
    pending: list[list] = []         # per doc: list of (sentence, hash) | None  (None = a sentence that was rejected)
    stats = Counter()
    per_k = math.ceil(src.quota / 5) if src.quota else 0
    k_counts: Counter = Counter()
    state = {"emitted": 0}

    def quota_hit() -> bool:
        return bool(src.quota) and state["emitted"] >= src.quota

    def windows_from(doc) -> list[list]:
        runs, cur = [], []
        for item in doc:
            if item is None:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(item)
        if cur:
            runs.append(cur)
        out = []
        for run in runs:
            if src.stream:
                avail = [k for k in range(1, min(5, len(run)) + 1) if k_counts[k] < per_k]
                if not avail:
                    continue
                low = min(k_counts[k] for k in avail)
                k = rng.choice([k for k in avail if k_counts[k] == low])
                start = rng.randint(0, len(run) - k)
                out.append(run[start:start + k])
            else:
                for w in partition_windows(run, rng):
                    out.append(w)
        return out

    def flush():
        if not pending:
            return
        if src.lid:
            flat = [it[0] for d in pending for it in d if it is not None]
            ok = iter(lid.verify(flat, lang)) if flat else iter(())
            stats["lid_checked"] += len(flat)
            for d in pending:
                for j, it in enumerate(d):
                    if it is not None and not next(ok):
                        d[j] = None
                        stats["lid_rejected"] += 1
        wins = [w for d in pending for w in windows_from(d)]
        pending.clear()
        if not wins:
            return
        texts = [" ".join(s for s, _ in w) for w in wins]
        lens = token_lengths(texts)
        for w, text, n_tok in zip(wins, texts, lens):
            while n_tok > cap and len(w) > 1:                 # drop trailing sentences until it fits
                w = w[:-1]
                text = " ".join(s for s, _ in w)
                n_tok = token_lengths([text])[0]
            if n_tok > cap or quota_hit():
                continue
            k_counts[len(w)] += 1
            state["emitted"] += 1
            writer.add(text=text, n_sentences=len(w), tokens=n_tok, hashes=[h for _, h in w],
                       source=src.name, license=src.license, kind=src.kind)

    n_pend = 0
    for doc in src.load(ctx):
        stats["docs"] += 1
        items = []
        for raw in doc.sentences:
            stats["sentences"] += 1
            s = normalize(raw)
            if not s or not good_sentence(s, min_words=src.min_words, max_words=src.max_words,
                                          require_terminal=src.require_terminal):
                items.append(None)
                continue
            h = sent_hash(s)
            if h in seen:
                items.append(None)
                stats["local_dups"] += 1
                continue
            seen.add(h)
            items.append((s, h))
        if any(it is not None for it in items):
            pending.append(items)
            n_pend += len(items)
        if n_pend >= LID_CHUNK:
            flush()
            n_pend = 0
            if quota_hit():
                break
    flush()
    rows = writer.close()
    res = {"lang": lang, "source": name, "rows": rows, "seconds": round(time.time() - t0, 1),
           "k_hist": dict(sorted(k_counts.items())), **dict(stats)}
    (out_dir / f"{slug(name)}.done.json").write_text(json.dumps(res))
    return res
