"""Rejection-sampling fine-tuning: an alternative to running RL (`algo: rs`).

Reward-guided RL (rl/policy_gradient.py) updates the policy with a gradient that pushes up the probability of high-reward samples
and down that of low-reward ones. Rejection sampling gets most of that benefit offline and with plain supervised learning: use the
same reward (AfriCOMET-QE, plus the scaffold score) only to DECIDE WHICH TEXT TO TRAIN ON, then run ordinary SFT on the survivors
(rl/sft.py). Two modes, chosen with `rs_mode`:

  filter       score the (source, translation) pairs of an existing SFT translation set with AfriCOMET-QE and keep the best ones
               (rank inside each language pair, so a language the metric scores low is not wiped out; plus an absolute floor).
               No model, no sampling: it cleans the data. Cheapest option and often the right first one.
  best_of_n    sample `rs_n` completions per prompt from the current policy, score each with the structured reward (scaffold +
               QE content, unshaped), keep the best `rs_keep_top` that clear `rs_min_reward`, write them as SFT records. Train on
               that file, then repeat with the new model (rounds). This is rejection-sampling fine-tuning / expert iteration /
               ReST (Anthony et al. 2017; Gulcehre et al. 2023; Yuan et al. 2023 "RFT").

Why it works: RL's policy-gradient estimate is E[(R - baseline) * grad log pi]. Replace the advantage by an indicator "this sample
is in the top-k and above the bar" and the gradient is just the maximum-likelihood gradient on the accepted samples -- the policy is
pulled toward its own best outputs. What it gives up: it never pushes DOWN on bad samples and it only sees what the model
already samples, so it cannot learn from rare successes the way RL's weighted updates can; repeated rounds narrow diversity.
What it gains: no KL tuning, no reward-hacking dynamics inside one run (a bad reward only mislabels one round of data, and you can
read the file before training), trivially resumable, runs on one GPU.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import structlog

from rl.config import RLConfig

LOG = structlog.get_logger()


# ------------------------------------------------------------------------------------------- pure selection logic
def filter_by_score(rows: list[dict], scores: list[float], keep_frac: float, min_score: float,
                    key: Callable[[dict], tuple] = lambda r: (r.get("source_lang"), r.get("targ_lang"))) -> list[dict]:
    """Keep rows with score >= min_score that rank in the top `keep_frac` of their group (language pair). Adds `qe_score`."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[key(r)].append(i)
    keep: set[int] = set()
    for idxs in groups.values():
        idxs = sorted(idxs, key=lambda i: -scores[i])
        n_keep = max(1, math.ceil(keep_frac * len(idxs))) if keep_frac > 0 else 0
        keep.update(i for i in idxs[:n_keep] if scores[i] >= min_score)
    return [{**rows[i], "qe_score": float(scores[i])} for i in sorted(keep)]


def select_best_of_n(rewards: list[float], n: int, keep_top: int, min_reward: float) -> list[list[int]]:
    """rewards are grouped consecutively (n per prompt). -> for each prompt, indices (into `rewards`) of the kept completions."""
    out = []
    for p in range(0, len(rewards), n):
        idx = sorted(range(p, p + n), key=lambda i: -rewards[i])[:keep_top]
        out.append([i for i in idx if rewards[i] >= min_reward])
    return out


# ------------------------------------------------------------------------------------------- IO helpers
def _iter_input(spec: str) -> Iterable[dict]:
    """Local jsonl/parquet globs, or  hub://<repo>/<glob>  (e.g. hub://BeardedMonster/sft-new/data/translation-multi-open-zul/*.parquet)."""
    import pyarrow.parquet as pq

    if spec.startswith("hub://"):
        from huggingface_hub import HfFileSystem

        from rl.common import hf_token

        fs = HfFileSystem(token=hf_token())
        repo_and_glob = spec[len("hub://"):]
        for path in sorted(fs.glob("datasets/" + repo_and_glob)):
            with fs.open(path, "rb") as fh:
                yield from pq.read_table(fh).to_pylist()
        return
    for path in sorted(glob.glob(spec)):
        if path.endswith(".parquet"):
            yield from pq.read_table(path).to_pylist()
        else:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def _write_jsonl(path: str, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ------------------------------------------------------------------------------------------- mode: filter
def run_filter(cfg: RLConfig, scorer: Optional[Callable] = None, rows: Optional[Iterable[dict]] = None) -> str:
    """AfriCOMET-QE filter/rank of existing translation SFT rows. -> path of the filtered jsonl."""
    if scorer is None:
        from rl.rewards import _africomet

        if not cfg.reward_qe_id:
            raise SystemExit("rs_mode=filter needs reward_qe_id (a reference-free AfriCOMET)")
        scorer = _africomet(cfg, "cpu", cfg.reward_qe_id)
    rows = rows if rows is not None else _iter_input(cfg.rs_input)
    translate = [r for r in rows if "<translate>" in (r.get("output_structure") or "<translate>") and r.get("input") and r.get("response")]
    scores: list[float] = []
    for i in range(0, len(translate), 4000):
        chunk = translate[i: i + 4000]
        scores += scorer([{"source": r["input"], "reference": ""} for r in chunk], [r["response"] for r in chunk])
    kept = filter_by_score(translate, scores, cfg.rs_filter_keep_frac, cfg.rs_filter_min_score)
    LOG.info("rs_filter", rows=len(translate), kept=len(kept), keep_frac=cfg.rs_filter_keep_frac, min_score=cfg.rs_filter_min_score)
    out = os.path.join(cfg.out_dir, "rs_filter.jsonl")
    _write_jsonl(out, kept)
    return out


# ------------------------------------------------------------------------------------------- mode: best_of_n
def run_best_of_n(cfg: RLConfig, records: Optional[list[dict]] = None, reward_fn: Optional[Callable] = None) -> str:
    """Sample rs_n completions per prompt, score, keep the best -> SFT records (`messages`, `text`, `reward`)."""
    import torch

    from rl.common import load_model, load_tokenizer
    from rl.data import render_prompt
    from rl.policy_gradient import sample_completions
    from rl.rewards import build_reward

    tok = load_tokenizer(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(cfg, cfg.model_path, trainable=False).to(device).eval()
    if reward_fn is None:
        reward_fn = build_reward(cfg.__class__(**{**cfg.__dict__, "reward_shaping": "none"}), device=device)   # raw [0, 1] scale
    wants_raw = getattr(reward_fn, "wants_raw", False)
    if records is None:
        from rlhf.translation import load_translation_records

        records = load_translation_records(cfg)
    kept_rows, seen = [], 0
    B = max(1, cfg.batch_size * cfg.grad_accum_steps)
    for start in range(0, len(records), B):
        batch = records[start: start + B]
        prompts = [render_prompt(tok, it["prompt_messages"]) for it in batch]
        with torch.no_grad():
            comps = sample_completions(model, tok, prompts, cfg, k=cfg.rs_n, device=device)
        texts = [tok.decode(c, skip_special_tokens=not wants_raw) for c in comps]
        expanded = [it for it in batch for _ in range(cfg.rs_n)]
        rewards = reward_fn(expanded, texts)
        for p, idxs in enumerate(select_best_of_n(rewards, cfg.rs_n, cfg.rs_keep_top, cfg.rs_min_reward)):
            for i in idxs:
                answer = texts[i] if wants_raw else tok.decode(comps[i], skip_special_tokens=True)
                answer = answer.replace("</s>", "").replace("<pad>", "").strip()
                msgs = batch[p]["prompt_messages"] + [{"role": "assistant", "content": answer}]
                kept_rows.append({"messages": msgs, "reward": float(rewards[i]), "source": batch[p].get("source"),
                                  "text": tok.apply_chat_template(msgs, tokenize=False)})
        seen += len(batch)
        LOG.info("rs_best_of_n", prompts=seen, kept=len(kept_rows))
    out = os.path.join(cfg.out_dir, "rs_best_of_n.jsonl")
    _write_jsonl(out, kept_rows)
    return out


def run_rs(cfg: RLConfig, **kw) -> str:
    if cfg.rs_mode == "filter":
        return run_filter(cfg, **kw)
    if cfg.rs_mode == "best_of_n":
        return run_best_of_n(cfg, **kw)
    raise ValueError(f"unknown rs_mode {cfg.rs_mode!r} (filter | best_of_n)")
