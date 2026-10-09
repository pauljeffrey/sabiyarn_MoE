"""Stage A: turn each source into shards of candidate windows (1-5 consecutive sentences of one document).

One task = one (language, source). Tasks are independent processes, resumable (a `.done.json` marker per task),
and write parquet rows: text, n_sentences, tokens, sentence hashes (for cross-source de-duplication in stage B),
source, license, kind. Web text is verified sentence by sentence with GlotLID before it can become a window.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
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
        ("ref", pa.string()), ("ref_lang", pa.string()),
    ])


class ShardWriter:
    def __init__(self, out_dir: Path, slug: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir, self.slug, self.files, self.total = out_dir, slug, 0, 0
        self.cols = {k: [] for k in ("text", "n_sentences", "tokens", "hashes", "source", "license", "kind", "ref", "ref_lang")}

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


REF_CAP_TOKENS = 480            # AfriCOMET truncates each segment at 512; keep the reference safely inside
_URL = re.compile(r"https?://|www\.", re.I)


def _ref_ok(ref: str, src: str) -> bool:
    """Cheap sanity for an aligned translation: present, not a URL, and of a plausible length relative to the source."""
    if not (5 <= len(ref) <= 4000) or _URL.search(ref):
        return False
    r = len(ref) / max(1, len(src))
    return 0.25 <= r <= 4.0 and sum(c.isalpha() for c in ref) / len(ref) > 0.4


def _run_task(lang: str, name: str, cfg: dict, t0: float) -> dict:
    os.environ["HF_HUB_CACHE"] = cfg["cache"]
    os.environ["HF_DATASETS_CACHE"] = cfg["cache"]
    from . import lid
    from .build import rng_for
    from .clean import LENIENT, clean_sentence, reject_reason
    from .registry import slug
    from .registry_v4 import build_registry_v4
    from .sources import Ctx
    from .text_utils import good_sentence, normalize, partition_windows, sent_hash

    src = next(s for s in build_registry_v4(**cfg["registry_args"])[lang] if s.name == name)
    lenient = lang in LENIENT
    ctx = Ctx(cache=Path(cfg["cache"]), token=os.environ.get("HF_TOKEN"), seed=cfg["seed"])
    rng = rng_for(cfg["seed"], "extract", lang, name)
    cap = cfg["cap"]
    out_dir = Path(cfg["out"]) / "windows" / lang
    writer = ShardWriter(out_dir, slug(name))
    seen: set[int] = set()
    pending: list[list] = []         # per doc: list of (sentence, hash, ref) | None  (None = a rejected sentence)
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
            if flat and not src.lid_ref_only:
                ok = iter(lid.verify(flat, lang))
            else:
                ok = itertools.repeat(True)
            stats["lid_checked"] += 0 if src.lid_ref_only else len(flat)
            refs_flat = [(d_i, j) for d_i, d in enumerate(pending) for j, it in enumerate(d) if it is not None and it[2]]
            ref_ok = {}
            if src.ref_lang and src.ref_lang != "eng" and refs_flat:
                for key, good in zip(refs_flat, lid.verify([pending[a][b][2] for a, b in refs_flat], src.ref_lang)):
                    ref_ok[key] = good
            for d_i, d in enumerate(pending):
                for j, it in enumerate(d):
                    if it is None:
                        continue
                    good = next(ok) and ref_ok.get((d_i, j), True)
                    if not good:
                        d[j] = None
                        stats["lid_rejected"] += 1
        wins = [w for d in pending for w in windows_from(d)]
        pending.clear()
        if not wins:
            return
        texts = [" ".join(s for s, _, _ in w) for w in wins]
        refs = [" ".join(r for _, _, r in w) if all(r for _, _, r in w) else "" for w in wins]
        lens = token_lengths(texts)
        ref_lens = token_lengths([r for r in refs if r]) if any(refs) else []
        ref_it = iter(ref_lens)
        for w, text, ref, n_tok in zip(wins, texts, refs, lens):
            n_ref = next(ref_it) if ref else 0
            while (n_tok > cap or n_ref > REF_CAP_TOKENS) and len(w) > 1:       # drop trailing sentences until it fits
                w = w[:-1]
                text = " ".join(s for s, _, _ in w)
                ref = " ".join(r for _, _, r in w) if ref else ""
                n_tok = token_lengths([text])[0]
                n_ref = token_lengths([ref])[0] if ref else 0
            if n_tok > cap or n_ref > REF_CAP_TOKENS or quota_hit():
                continue
            if lenient and reject_reason(text, lang, src.kind):
                stats["rej:window_rules"] += 1
                continue
            k_counts[len(w)] += 1
            state["emitted"] += 1
            writer.add(text=text, n_sentences=len(w), tokens=n_tok, hashes=[h for _, h, _ in w],
                       source=src.name, license=src.license, kind=src.kind, ref=ref, ref_lang=src.ref_lang if ref else "")

    n_pend = 0
    for doc in src.load(ctx):
        stats["docs"] += 1
        if cfg.get("max_docs") and stats["docs"] > cfg["max_docs"]:
            break
        items = []
        refs = doc.refs or [""] * len(doc.sentences)
        for raw, raw_ref in zip(doc.sentences, refs):
            stats["sentences"] += 1
            s = normalize(raw)
            if lenient:
                s = clean_sentence(s, lang)
            if not s or not good_sentence(s, min_words=src.min_words, max_words=src.max_words,
                                          require_terminal=src.require_terminal and not lenient, lenient=lenient):
                items.append(None)
                continue
            ref = normalize(raw_ref) if raw_ref else ""
            if ref and not _ref_ok(ref, s):
                items.append(None)
                stats["bad_ref"] += 1
                continue
            h = sent_hash(s)
            if h in seen:
                items.append(None)
                stats["local_dups"] += 1
                continue
            seen.add(h)
            items.append((s, h, ref))
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
