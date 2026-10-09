"""Build `other_african_pretrain.bin` (+ the validation additions) from BeardedMonster/pretrain-new.

Resumable stages, one language at a time, constant memory:

  tokenize   stream each language's parquet shards from the Hub; rule-based filter (Gopher/C4-style, see `rule_check`); tokenize with
             the SabiYarn tokenizer with ONE end-of-text token per document; candidate validation documents are chosen at document
             level by a hash of the text (same size rule as data/prepare.py::calculate_test_size). Output: token parts + offsets.
  score      per language: token priors p(x) = f(x) / sum f from a 10% sample of the language's training documents (Seo et al., ICLR 2026,
             "Prior-based Noisy Text Data Filtering", arXiv:2509.18577), then for every document mu_d = mean(log p) and sigma_d = std(p),
             the latter computed per 512-token block and averaged (the paper scores 512-token blocks).
  plan       finalise the validation split (<= 1M tokens per language; the surplus goes back to training), run UniMax (Chung et al.,
             ICLR 2023) on the per-language training token counts for the total budget C with at most `max_repeat` epochs per language,
             and, for every language whose share is below one epoch, select the BEST documents: documents whose mu_d / sigma_d are
             furthest from the language's medians are discarded (the paper's rule: grow k until the union of the k largest
             |mu_d - M_mu| and the k largest |sigma_d - M_sigma| removes the tokens that do not fit in the budget).
  assemble   materialise the plan (whole epochs, each reshuffled, plus a random part of the next) and write ONE file in which the
             documents of all languages are globally shuffled at document level, so any 4096-token block mixes several languages;
             write the validation segments (one contiguous segment per language) as a separate file.
  merge-val  old validation.bin (from S3) + the new validation segments -> validation_v2.bin and validation_v2_manifest.json.
  upload     send the files to the S3 bucket from training/train_config.yaml under new names (never overwrites).

The order in which documents were read or tokenized does not matter: the final order is a uniform random permutation of documents
(drawn in `assemble`), which is the same thing as shuffling before tokenization, without a second copy of the text on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = "BeardedMonster/pretrain-new"
TOKENIZER = "BeardedMonster/SabiYarn-32k"
# Text for CONTINUED PRETRAINING must be tokenised with the tokenizer as it was before the tool-calling vocabulary was added: the model has 52,050
# embedding rows until the SFT-time growth, and ordinary web text contains strings such as "name" or "type" that the current tokenizer turns into ids
# >= 52,119. Override with PRETRAIN_TOKENIZER_REVISION=<commit or 'main'> only for SFT/RL text.
PRETRAIN_TOKENIZER_REVISION = os.environ.get("PRETRAIN_TOKENIZER_REVISION", "0c711cb2f7af669b2782c90c1d0d4d097df21422")
LANGS = ["afr", "swh", "som", "zul", "hau", "xho", "kin", "sot", "sna", "nya", "ibo", "yor", "orm", "tsn", "pcm", "ewe", "twi",
         "wol", "efi", "fon", "ibb", "urh"]
SMALL = {"tsn", "pcm", "ewe", "twi", "wol", "efi", "fon", "ibb", "urh", "orm"}      # rule thresholds are relaxed for these
MAX_VAL_TOKENS = 1_000_000
BLOCK = 512
VOCAB_SIZE = 65536


# ------------------------------------------------------------------------------------------------ UniMax
def unimax(counts: dict[str, float], budget: float, max_repeat: float = 4.0) -> dict[str, float]:
    """UniMax (Chung et al., 2023, Algorithm 1). counts: tokens per language; budget: total to distribute.
    -> U_l, the tokens each language contributes (sum <= budget; each <= max_repeat * count)."""
    order = sorted(counts, key=lambda l: counts[l])
    remaining, out = float(budget), {}
    for i, l in enumerate(order):
        b_l = remaining / (len(order) - i)
        u = counts[l] * max_repeat if b_l > counts[l] * max_repeat else b_l
        out[l] = u
        remaining -= u
    return out


def sampling_distribution(u: dict[str, float]) -> dict[str, float]:
    z = sum(u.values())
    return {l: v / z for l, v in u.items()}


# ------------------------------------------------------------------------------------------------ rule-based filters
_ALPHA_WORD = re.compile(r"\S*[^\W\d_]\S*")


def _top_ngram_frac(words: list[str], n: int, total_chars: int) -> float:
    if len(words) < n or total_chars == 0:
        return 0.0
    c = Counter(zip(*[words[i:] for i in range(n)]))
    g, k = c.most_common(1)[0]
    return (sum(len(w) for w in g) * k) / total_chars


def _dup_ngram_frac(words: list[str], n: int, total_chars: int) -> float:
    """Gopher: share of characters covered by word n-grams that occur more than once (non-overlapping count)."""
    if len(words) < n or total_chars == 0:
        return 0.0
    seen: set = set()
    covered = 0
    i = 0
    while i + n <= len(words):
        g = tuple(words[i:i + n])
        if g in seen:
            covered += sum(len(w) for w in g)
            i += n
        else:
            seen.add(g)
            i += 1
    return covered / total_chars


def rule_check(text: str, relax: float = 1.0) -> str | None:
    """Gopher / C4 style quality rules. -> None if the document passes, else the name of the first rule it breaks.
    `relax` > 1 loosens the repetition thresholds (languages with very little data and heavily formulaic text, e.g. scripture)."""
    words = text.split()
    n = len(words)
    if n < 20:
        return "too_few_words"
    if n > 200_000:
        return "too_many_words"
    chars = sum(len(w) for w in words)
    mwl = chars / n
    if not (2.2 <= mwl <= 14.0):
        return "mean_word_length"
    if (text.count("#") + text.count("...") + text.count("…")) / n > 0.10:
        return "symbol_ratio"
    if len(_ALPHA_WORD.findall(text)) / n < 0.75:
        return "alpha_word_ratio"
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        if sum(l.rstrip().endswith(("...", "…")) for l in lines) / len(lines) > 0.30:
            return "ellipsis_lines"
        if sum(l.lstrip().startswith(("•", "-", "*")) for l in lines) / len(lines) > 0.90:
            return "bullet_lines"
        seen, dup_chars = set(), 0
        for l in lines:
            if l in seen:
                dup_chars += len(l)
            seen.add(l)
        if len(lines) > 3 and dup_chars / max(1, sum(len(l) for l in lines)) > 0.30 * relax:
            return "duplicate_lines"
    w = words[:1000]                                     # repetition rules look at the first 1,000 words (cost control)
    wc = sum(len(x) for x in w)
    for k, lim in ((2, 0.20), (3, 0.18), (4, 0.16)):
        if _top_ngram_frac(w, k, wc) > lim * relax:
            return f"top_{k}gram"
    for k, lim in ((5, 0.15), (8, 0.12), (10, 0.10)):
        if _dup_ngram_frac(w, k, wc) > lim * relax:
            return f"dup_{k}gram"
    return None


def _rule_batch(args):
    texts, relax = args
    return [rule_check(t, relax) for t in texts]


# ------------------------------------------------------------------------------------------------ validation split
def calculate_test_size(n_docs: int) -> int:
    """Same rule as data/prepare.py::calculate_test_size (copied: importing prepare.py loads configs and lmdb)."""
    if n_docs <= 50:
        return 0
    if n_docs <= 80_000:
        return int(0.1 * n_docs)
    if n_docs <= 800_000:
        return int(0.01 * n_docs)
    if n_docs <= 3_500_000:
        return int(0.0025 * n_docs)
    return int(0.0005 * n_docs)


def is_val(text: str, frac: float, seed: int = 2357) -> bool:
    h = hashlib.blake2b(text.strip().encode("utf-8"), digest_size=8, key=seed.to_bytes(8, "little")).digest()
    return int.from_bytes(h, "little") / 2**64 < frac


# ------------------------------------------------------------------------------------------------ stage: tokenize
def _hf_token():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data-gen"))
    from africomet_translation.build import hf_token
    return hf_token()


def tokenize_language(lang: str, out: Path, token: str, tk, eot: int, pool, log=print) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    d = out / lang
    d.mkdir(parents=True, exist_ok=True)
    state_p = d / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"done": [], "train_tokens": 0, "val_tokens": 0, "train_docs": 0,
                                                                         "val_docs": 0, "chars": 0, "rule_dropped": {}, "docs_in": 0}
    files = sorted(f.path for f in HfApi(token=token).list_repo_tree(REPO, path_in_repo=f"data/{lang}", repo_type="dataset")
                   if f.path.endswith(".parquet"))
    fs = HfFileSystem(token=token)
    n_docs = 0
    for p in files:
        with fs.open(f"datasets/{REPO}/{p}", "rb") as fh:
            n_docs += pq.ParquetFile(fh).metadata.num_rows
    frac = calculate_test_size(n_docs) / max(1, n_docs)
    relax = 1.5 if lang in SMALL else 1.0
    state["docs_total"], state["val_frac"] = n_docs, frac
    for p in files:
        if p in state["done"]:
            continue
        tmp = Path(tempfile.mkdtemp(dir=out))
        try:
            local = hf_hub_download(REPO, p, repo_type="dataset", token=token, cache_dir=str(tmp))
            texts = [t for t in pq.read_table(local, columns=["text"]).column(0).to_pylist() if t and t.strip()]
            state["docs_in"] += len(texts)
            verdicts = [v for part in pool.imap(_rule_batch, [(texts[i:i + 200], relax) for i in range(0, len(texts), 200)])
                        for v in part]
            kept = []
            for t, v in zip(texts, verdicts):
                if v is None:
                    kept.append(t)
                else:
                    state["rule_dropped"][v] = state["rule_dropped"].get(v, 0) + 1
            tag = f"part{len(state['done']):05d}"
            sides = {"train": [], "val": []}
            for t in kept:
                sides["val" if is_val(t, frac) else "train"].append(t)
            for side, ts in sides.items():
                if not ts:
                    continue
                arrs, lens = [], []
                for i in range(0, len(ts), 2000):
                    for e in tk.encode_batch(ts[i:i + 2000]):
                        ids = e.ids + [eot]                            # ONE end-of-text token per document
                        arrs.append(np.asarray(ids, dtype=np.uint16))
                        lens.append(len(ids))
                flat = np.concatenate(arrs)
                np.save(d / f"{side}.{tag}.offsets.npy", np.cumsum(np.asarray(lens, dtype=np.uint64)))
                flat.tofile(d / f"{side}.{tag}.bin.tmp")
                os.replace(d / f"{side}.{tag}.bin.tmp", d / f"{side}.{tag}.bin")
                state[f"{side}_tokens"] += int(flat.size)
                state[f"{side}_docs"] += len(ts)
            state["chars"] += sum(len(t) for t in kept)
            state["done"].append(p)
            state_p.write_text(json.dumps(state))
            log(f"{lang} {p.split('/')[-1]} kept {len(kept)}/{len(texts)} train={state['train_tokens']:,} val={state['val_tokens']:,}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return state


# ------------------------------------------------------------------------------------------------ documents of a language
class LangDocs:
    """All documents of one language (train + candidate-validation parts) as numpy arrays; ~30 bytes per document."""

    def __init__(self, lang_dir: Path):
        self.dir = lang_dir
        self.files, fid, st, en, side = [], [], [], [], []
        for s in ("train", "val"):
            for off_p in sorted(lang_dir.glob(f"{s}.part*.offsets.npy")):
                ends = np.load(off_p)
                self.files.append(off_p.with_name(off_p.name.replace(".offsets.npy", ".bin")))
                fid.append(np.full(ends.size, len(self.files) - 1, dtype=np.uint16))
                st.append(np.concatenate([[0], ends[:-1]]).astype(np.uint64))
                en.append(ends.astype(np.uint64))
                side.append(np.full(ends.size, s == "val", dtype=bool))
        z = np.zeros(0, dtype=np.uint64)
        self.fid = np.concatenate(fid) if fid else np.zeros(0, dtype=np.uint16)
        self.start = np.concatenate(st) if st else z
        self.end = np.concatenate(en) if en else z
        self.cand_val = np.concatenate(side) if side else np.zeros(0, dtype=bool)
        self.len = (self.end - self.start).astype(np.int64)
        self._maps: dict[int, np.memmap] = {}

    def __len__(self):
        return self.fid.size

    def tokens(self, i: int) -> np.ndarray:
        k = int(self.fid[i])
        if k not in self._maps:
            self._maps[k] = np.memmap(self.files[k], dtype=np.uint16, mode="r")
        return self._maps[k][int(self.start[i]): int(self.end[i])]

    def write(self, f, idx) -> int:
        n = 0
        for i in idx:
            seg = self.tokens(int(i))
            f.write(seg.tobytes())
            n += seg.size
        return n


def final_split(docs: LangDocs, rng: np.random.Generator, cap: int = MAX_VAL_TOKENS) -> np.ndarray:
    """-> bool array, True = validation. Candidate validation docs (hash split) are kept in random order until `cap` tokens; the surplus
    goes back to training so no language contributes more than ~cap validation tokens."""
    is_v = np.zeros(len(docs), dtype=bool)
    cand = np.flatnonzero(docs.cand_val)
    rng.shuffle(cand)
    got = 0
    for i in cand:
        if got >= cap:
            break
        is_v[i] = True
        got += int(docs.len[i])
    return is_v


# ------------------------------------------------------------------------------------------------ stage: score (priors)
def token_priors(docs: LangDocs, train_mask: np.ndarray, rng: np.random.Generator, sample: float = 0.10) -> np.ndarray:
    """p_prior(x) = f_D(x) / sum f_D, with f from a `sample` fraction of the training documents; +0.5 smoothing so unseen ids have a prior."""
    idx = np.flatnonzero(train_mask)
    idx = idx[rng.random(idx.size) < sample] if idx.size > 2000 else idx
    f = np.zeros(VOCAB_SIZE, dtype=np.float64)
    for i in idx:
        t = docs.tokens(int(i))[:-1]                                   # not the end-of-text token
        f += np.bincount(t, minlength=VOCAB_SIZE)
    return (f + 0.5) / (f.sum() + 0.5 * VOCAB_SIZE)


def doc_prior_stats(tokens: np.ndarray, p: np.ndarray, logp: np.ndarray) -> tuple[float, float]:
    """(mu_d, sigma_d): mu = mean log prior of the document; sigma = std of the (non-log) prior, averaged over 512-token blocks."""
    t = tokens[:-1]
    if t.size == 0:
        return 0.0, 0.0
    mu = float(logp[t].mean())
    pp = p[t]
    if pp.size <= BLOCK:
        return mu, float(pp.std())
    nb = pp.size // BLOCK                                              # full blocks; the tail joins the last block
    blocks = pp[: nb * BLOCK].reshape(nb, BLOCK)
    return mu, float(blocks.std(axis=1).mean())


def score_language(docs: LangDocs, train_mask: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    p = token_priors(docs, train_mask, rng)
    logp = np.log(p)
    mu = np.zeros(len(docs), dtype=np.float32)
    sg = np.zeros(len(docs), dtype=np.float32)
    for i in range(len(docs)):
        if train_mask[i]:
            mu[i], sg[i] = doc_prior_stats(docs.tokens(i), p, logp)
    return {"mu": mu, "sigma": sg}


# ------------------------------------------------------------------------------------------------ stage: select
def prior_select(mu: np.ndarray, sigma: np.ndarray, lens: np.ndarray, discard_tokens: float) -> np.ndarray:
    """Seo et al. (2025): outliers are the documents farthest from the medians of mu_d and sigma_d. Grow k until the union of the k
    farthest by delta_mu and the k farthest by delta_sigma holds at least `discard_tokens` tokens. -> bool keep mask."""
    n = mu.size
    if discard_tokens <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    d_mu = np.abs(mu - np.median(mu))
    d_sg = np.abs(sigma - np.median(sigma))
    rank_mu = np.empty(n, dtype=np.int64)
    rank_mu[np.argsort(-d_mu, kind="stable")] = np.arange(n)
    rank_sg = np.empty(n, dtype=np.int64)
    rank_sg[np.argsort(-d_sg, kind="stable")] = np.arange(n)

    def removed(k: int) -> int:
        return int(lens[(rank_mu < k) | (rank_sg < k)].sum())

    lo, hi = 0, n
    while lo < hi:                                                      # smallest k whose union covers the discard target
        mid = (lo + hi) // 2
        if removed(mid) >= discard_tokens:
            hi = mid
        else:
            lo = mid + 1
    return ~((rank_mu < lo) | (rank_sg < lo))


# ------------------------------------------------------------------------------------------------ stage: plan
def make_plan(out: Path, budget: float, max_repeat: float, langs: list[str], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    info = {}
    for l in langs:
        if not (out / l / "state.json").exists():
            continue
        docs = LangDocs(out / l)
        is_v = final_split(docs, rng)
        tr = ~is_v
        sc_p = out / l / "scores.npz"
        if sc_p.exists():
            z = np.load(sc_p)
            sc = {"mu": z["mu"], "sigma": z["sigma"]}
        else:
            sc = score_language(docs, tr)
        np.savez(sc_p, **sc)
        np.save(out / l / "is_val.npy", is_v)
        info[l] = {"docs": docs, "train": tr, "val_mask": is_v, "sc": sc, "tokens": int(docs.len[tr].sum()), "val_tokens": int(docs.len[is_v].sum())}
    counts = {l: v["tokens"] for l, v in info.items()}
    u = unimax(counts, budget, max_repeat)
    p = sampling_distribution(u)
    plan = {"budget_tokens": float(budget), "max_repeat": max_repeat, "available_train_tokens": int(sum(counts.values())), "languages": {}}
    for l, v in info.items():
        keep = v["train"].copy()
        discarded_tokens = 0
        if u[l] < counts[l]:                                            # less than one epoch fits: keep the best documents
            idx = np.flatnonzero(v["train"])
            k = prior_select(v["sc"]["mu"][idx], v["sc"]["sigma"][idx], v["docs"].len[idx], counts[l] - u[l])
            keep[idx[~k]] = False
            discarded_tokens = int(v["docs"].len[idx[~k]].sum())
        np.save(out / l / "keep.npy", keep)
        kept_tokens = int(v["docs"].len[keep].sum())
        plan["languages"][l] = {"train_tokens": counts[l], "val_tokens": v["val_tokens"], "target_tokens": int(u[l]),
                                "selected_tokens": kept_tokens, "discarded_tokens_by_prior_filter": discarded_tokens,
                                "epochs": u[l] / counts[l], "share": p[l], "natural_share": counts[l] / sum(counts.values())}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    return plan


# ------------------------------------------------------------------------------------------------ stage: assemble
def _epoch_sequence(lens: np.ndarray, pool_idx: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Document indices (into the language's table) covering `target` tokens: whole epochs (fresh shuffle each) plus a random part."""
    if pool_idx.size == 0 or target <= 0:
        return np.zeros(0, dtype=np.int64)
    total = int(lens[pool_idx].sum())
    seqs, got = [], 0
    while got < target:
        perm = rng.permutation(pool_idx)
        if got + total <= target:
            seqs.append(perm); got += total
        else:
            c = np.cumsum(lens[perm])
            k = int(np.searchsorted(c, target - got)) + 1
            seqs.append(perm[:k]); got = target
    return np.concatenate(seqs)


EOT_ID = 2          # </s> in BeardedMonster/SabiYarn-32k
# Language folders whose documents are NOT part of the UniMax mix and not shuffled into it: they are appended, whole and once, AFTER the shuffled
# body of the bin (Nigerian English news: naijaweb, naijaweb-edu, naijaweb-edu2 from data/eng-tail/ of pretrain-new). They get no validation split,
# no prior filtering, no repeats; documents are shuffled inside the tail, each closed by one end-of-text token like everything else.
TAIL_DIRS = ("eng-tail",)


def count_token(path: Path, token: int, chunk: int = 1 << 27) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            a = np.fromfile(f, dtype=np.uint16, count=chunk)
            if a.size == 0:
                return n
            n += int(np.count_nonzero(a == token))


def append_tails(out: Path, name: str, rng: np.random.Generator) -> dict:
    """Append every document of each tail folder (train and candidate-validation sides alike) to the end of the bin, shuffled within the tail."""
    info = {}
    for t in TAIL_DIRS:
        if not (out / t).exists():
            continue
        d = LangDocs(out / t)
        idx = rng.permutation(len(d))
        with open(out / f"{name}.bin", "ab") as f:
            n = d.write(f, idx)
        info[t] = {"docs": int(len(d)), "tokens": int(n), "position": "end of file, after the shuffled body"}
    return info


def assemble(out: Path, name: str, seed: int = 0) -> dict:
    plan = json.loads((out / "plan.json").read_text())
    rng = np.random.default_rng(seed)
    docs, seqs = {}, {}
    for l, info in plan["languages"].items():
        d = LangDocs(out / l)
        keep = np.load(out / l / "keep.npy")
        pool_idx = np.flatnonzero(keep)
        target = info["target_tokens"]
        # below one epoch the prior-filtered pool IS the selection (it may sit a hair under the target); above it, repeat the pool
        seqs[l] = pool_idx if info["epochs"] < 1 else _epoch_sequence(d.len, pool_idx, target, rng)
        docs[l] = d
    names = list(seqs)
    lang_id = np.concatenate([np.full(seqs[l].size, i, dtype=np.uint8) for i, l in enumerate(names)])
    doc_id = np.concatenate([seqs[l] for l in names]).astype(np.uint32)
    order = rng.permutation(lang_id.size)                               # a uniform random permutation of ALL documents of ALL languages
    written = {l: 0 for l in names}
    with open(out / f"{name}.bin", "wb") as f:
        for j in order:
            l = names[lang_id[j]]
            written[l] += docs[l].write(f, [int(doc_id[j])])
    tail = append_tails(out, name, rng)
    eot_in_file = count_token(out / f"{name}.bin", EOT_ID)          # one end-of-text token closes every document, across languages too
    last = np.fromfile(out / f"{name}.bin", dtype=np.uint16, count=1, offset=max(0, (out / f"{name}.bin").stat().st_size - 2))
    vtok = {}
    for l in plan["languages"]:
        d = LangDocs(out / l)
        is_v = np.load(out / l / "is_val.npy")
        vtok[l] = int(d.len[is_v].sum())
    manifest = {"name": name, "tokenizer": TOKENIZER, "dtype": "uint16", "eot_per_doc": 1, "eot_id": EOT_ID, "n_docs_written": int(order.size) + sum(t["docs"] for t in tail.values()),
                "eot_tokens_in_file": eot_in_file, "file_ends_with_eot": bool(last.size and int(last[0]) == EOT_ID), "plan": plan, "train_tokens_written": written,
                "tail": tail, "train_total": sum(written.values()) + sum(t["tokens"] for t in tail.values()), "val_tokens": vtok, "val_total": sum(vtok.values())}
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ------------------------------------------------------------------------------------------------ stage: merge validation
def write_new_val(out: Path, path: Path, seed: int = 1) -> dict:
    """One contiguous validation segment per language (documents shuffled inside it). -> {lang: (start, end)} relative to `path`."""
    rng = np.random.default_rng(seed)
    segs, pos = {}, 0
    with open(path, "wb") as f:
        for l in LANGS:
            if not (out / l / "is_val.npy").exists():
                continue
            d = LangDocs(out / l)
            idx = np.flatnonzero(np.load(out / l / "is_val.npy"))
            rng.shuffle(idx)
            n = d.write(f, idx)
            segs[l] = (pos, pos + n)
            pos += n
    return segs


def merge_val(out: Path, old_validation: Path, new_name: str = "validation_v2"):
    new_part = out / "new_val_segments.bin"
    segs = write_new_val(out, new_part)
    n_old = old_validation.stat().st_size // 2
    target = out / f"{new_name}.bin"
    with open(target, "wb") as f:
        with open(old_validation, "rb") as g:
            shutil.copyfileobj(g, f, 64 << 20)
        with open(new_part, "rb") as g:
            shutil.copyfileobj(g, f, 64 << 20)
    manifest = {"dtype": "uint16", "total_tokens": n_old + sum(e - s for s, e in segs.values()),
                "segments": [{"name": "old", "start": 0, "end": n_old}] +
                            [{"name": l, "start": n_old + s, "end": n_old + e} for l, (s, e) in segs.items()]}
    (out / f"{new_name}_manifest.json").write_text(json.dumps(manifest, indent=1))
    new_part.unlink()
    return manifest


# ------------------------------------------------------------------------------------------------ stage: upload
def upload(out: Path, files: list[str]):
    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    cfg = OmegaConf.to_container(OmegaConf.load(root / "training" / "train_config.yaml"), resolve=True)
    s3 = cfg.get("s3", {})
    sys.path.insert(0, str(root))
    from training.s3_utils import upload_if_absent

    for fn in files:
        upload_if_absent(str(out / fn), f"datasets/{fn}", bucket=s3["s3_bucket_name"], endpoint=s3["s3_endpoint"],
                         access_key=os.environ["S3_ACCESS_KEY_ID"], secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
                         prefix="", override=False)
        print("uploaded", fn, flush=True)


def download_old_validation(dest: Path):
    from dotenv import load_dotenv
    import boto3

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    from omegaconf import OmegaConf
    s3 = OmegaConf.to_container(OmegaConf.load(root / "training" / "train_config.yaml"), resolve=True)["s3"]
    c = boto3.client("s3", endpoint_url=s3["s3_endpoint"], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                     aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"])
    c.download_file(s3["s3_bucket_name"], "datasets/validation.bin", str(dest))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("tokenize", "plan", "assemble", "merge-val", "upload"):
        p = sub.add_parser(c)
        p.add_argument("--out", required=True)
        if c == "tokenize":
            p.add_argument("--langs", nargs="*", default=LANGS)
            p.add_argument("--workers", type=int, default=3)
        if c == "plan":
            p.add_argument("--budget-tokens", type=float, default=12.5e9)
            p.add_argument("--max-repeat", type=float, default=4.0)
        if c == "assemble":
            p.add_argument("--name", default="other_african_pretrain")
        if c == "merge-val":
            p.add_argument("--old-validation", default=None, help="local copy of datasets/validation.bin (default: download from S3)")
        if c == "upload":
            p.add_argument("--files", nargs="+", default=["other_african_pretrain.bin", "other_african_pretrain_manifest.json",
                                                          "validation_v2.bin", "validation_v2_manifest.json"])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "tokenize":
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        token = _hf_token()
        tk = Tokenizer.from_file(hf_hub_download(TOKENIZER, "tokenizer.json", token=token, revision=PRETRAIN_TOKENIZER_REVISION))
        eot = tk.token_to_id("</s>")
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for l in a.langs:
                tokenize_language(l, out, token, tk, eot, pool, log=lambda m: print(m, file=sys.stderr, flush=True))
    elif a.cmd == "plan":
        plan = make_plan(out, a.budget_tokens, a.max_repeat, LANGS)
        print(f"budget {plan['budget_tokens']:,.0f} of {plan['available_train_tokens']:,} available train tokens")
        for l, i in sorted(plan["languages"].items(), key=lambda kv: -kv[1]["target_tokens"]):
            print(f"{l:4} {i['train_tokens']:>14,} -> {i['target_tokens']:>14,}  epochs {i['epochs']:.2f}  share {i['share']:.1%} "
                  f"(natural {i['natural_share']:.1%})  prior-filter dropped {i['discarded_tokens_by_prior_filter']:,}")
    elif a.cmd == "assemble":
        m = assemble(out, a.name)
        print(f"train {m['train_total']:,} tokens, val {m['val_total']:,} tokens")
    elif a.cmd == "merge-val":
        old = Path(a.old_validation) if a.old_validation else out / "validation_old.bin"
        if not old.exists():
            download_old_validation(old)
        m = merge_val(out, old)
        print(f"validation_v2: {m['total_tokens']:,} tokens, {len(m['segments'])} segments")
    else:
        upload(out, a.files)


if __name__ == "__main__":
    main()
