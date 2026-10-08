"""Datasets for SFT / DPO / RL, rendered with the SabiYarn chat template.

Inputs are data-gen's records (data-gen/data/processed/{sft,dpo}.jsonl) or any Hub dataset with the same columns:

  dpo : prompt_messages (list of {role, content}), and EITHER chosen/rejected
        OR data-gen's own response_1/response_2 + ranking (see preference_pair)
  sft : messages (full conversation ending in the assistant turn)   -- or prompt_messages + response
  rl  : prompt_messages (+ whatever the reward needs: `reference`, `source`, ...)

A training text is exactly what the chat template renders for the finished conversation, i.e.
`<s><|system|>..</s><|user|>..</s><|assistant|>{answer}</s>`; the prompt is the same text up to and including
`<|assistant|>` (add_generation_prompt=True), which tests/test_chat_template.py guarantees is a strict prefix.
Only completion tokens (answer + the closing `</s>`) get a loss.

The model has learned absolute positions and never saw text longer than the prompt+answer it is given here, so
over-long examples are DROPPED (and counted), never truncated from the left: cutting the prompt would remove
`<s>`/the system turn and change what the model conditions on.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import structlog
import torch

from rl.config import RLConfig

LOG = structlog.get_logger()


# ------------------------------------------------------------------------------------------- loading


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(cfg: RLConfig) -> list[dict]:
    if cfg.dataset_id:
        from datasets import load_dataset

        from rl.common import hf_token

        ds = load_dataset(cfg.dataset_id, split=cfg.dataset_split, token=hf_token())
        rows = [dict(r) for r in ds]
    else:
        rows = _read_jsonl(cfg.data_path)
    if cfg.languages:
        keep = set(cfg.languages)
        rows = [r for r in rows if r.get("language") in keep]
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)  # deterministic: the same split on every rank and every restart
    if cfg.max_samples > 0:
        rows = rows[: cfg.max_samples]
    return rows


def split_records(rows: list[dict], eval_samples: int) -> tuple[list[dict], list[dict]]:
    n_eval = min(max(0, eval_samples), max(0, len(rows) // 10))  # never let eval eat more than 10%
    return rows[n_eval:], rows[:n_eval]


# ------------------------------------------------------------------------------------------- rendering


def render_prompt(tok, messages: list[dict]) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]  # the template already contains <s>


def render_answer(tok, answer: str) -> list[int]:
    return tok(answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]


def _prompt_messages(rec: dict) -> list[dict]:
    if "prompt_messages" in rec:
        return rec["prompt_messages"]
    msgs = rec["messages"]
    return msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs


@dataclass
class Encoded:
    prompt_ids: list[int]
    answer_ids: list[int]


def encode_answer(tok, rec: dict, answer: str, cfg: RLConfig) -> Optional[Encoded]:
    prompt = render_prompt(tok, _prompt_messages(rec))
    ans = render_answer(tok, answer)
    if len(prompt) > cfg.max_prompt_len or len(prompt) + len(ans) > cfg.max_seq_len:
        return None
    return Encoded(prompt, ans)


def preference_pair(rec: dict) -> Optional[tuple[str, str]]:
    """(chosen, rejected) from a record, in either of the two shapes this project produces.

    `chosen`/`rejected` is what a judged or public DPO set looks like. data-gen writes `response_1` and
    `response_2` plus a `ranking` -- and reading only the first shape meant the trainer KeyError'd on every
    record data-gen has ever produced. The only bridge was judge_gen.py, a second paid pass that reduces the
    candidates to chosen/rejected; with no judge model in the pipeline that bridge does not exist, so the
    ordering the generator already gave us is used directly.

    `ranking` is authoritative where present, because postprocess_gen sorts the responses by quality and
    writes the order it used. Falling back to response_1-is-better matches that sort.
    """
    if rec.get("chosen") is not None and rec.get("rejected") is not None:
        return str(rec["chosen"]), str(rec["rejected"])

    a, b = rec.get("response_1"), rec.get("response_2")
    if not a or not b:
        return None
    ranking = rec.get("ranking") or []
    if len(ranking) >= 2:
        order = {"best": 0, "partial": 1, "worst": 2}
        ra, rb = order.get(str(ranking[0]).lower(), 1), order.get(str(ranking[1]).lower(), 1)
        if ra == rb:
            return None  # same quality on both sides is not a preference
        return (str(a), str(b)) if ra < rb else (str(b), str(a))
    return str(a), str(b)


def score_margin(rec: dict) -> Optional[float]:
    """The generator's own 0-10 gap between the two replies, where it recorded one.

    Used to drop pairs that are nearly as good as each other: those teach almost nothing and their gradient is
    noise. postprocess_gen already drops anything under 2.0 at generation time, so this is a second filter for
    corpora built before that existed.
    """
    sc = rec.get("scores")
    if not isinstance(sc, (list, tuple)) or len(sc) < 2:
        return None
    try:
        lo, hi = float(sc[0]), float(sc[-1])
    except (TypeError, ValueError):
        return None
    return abs(lo - hi)


def encode_dpo(tok, records: Iterable[dict], cfg: RLConfig) -> list[tuple[Encoded, Encoded]]:
    out, dropped, unusable, narrow = [], 0, 0, 0
    min_margin = float(getattr(cfg, "min_score_margin", 0.0) or 0.0)
    for rec in records:
        pair = preference_pair(rec)
        if pair is None:
            unusable += 1
            continue
        if min_margin:
            m = score_margin(rec)
            if m is not None and m < min_margin:
                narrow += 1
                continue
        c, r = encode_answer(tok, rec, pair[0], cfg), encode_answer(tok, rec, pair[1], cfg)
        if c is None or r is None:
            dropped += 1
            continue
        out.append((c, r))
    if dropped:
        LOG.warning("dpo_examples_dropped_too_long", dropped=dropped, kept=len(out),
                    max_prompt_len=cfg.max_prompt_len, max_seq_len=cfg.max_seq_len)
    if unusable:
        LOG.warning("dpo_examples_without_a_preference", dropped=unusable,
                    hint="need chosen/rejected, or response_1/response_2 with a ranking that differs")
    if narrow:
        LOG.info("dpo_examples_dropped_narrow_margin", dropped=narrow, min_margin=min_margin)
    return out


def encode_sft(tok, records: Iterable[dict], cfg: RLConfig) -> list[Encoded]:
    out, dropped = [], 0
    for rec in records:
        answer = rec["response"] if "response" in rec else rec["messages"][-1]["content"]
        e = encode_answer(tok, rec, answer, cfg)
        if e is None:
            dropped += 1
        else:
            out.append(e)
    if dropped:
        LOG.warning("sft_examples_dropped_too_long", dropped=dropped, kept=len(out))
    return out


# ------------------------------------------------------------------------------------------- batching


def pad_batch(seqs: list[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-padded (prompt + answer) batch; loss_mask is 1 on answer tokens only."""
    width = max(len(s.prompt_ids) + len(s.answer_ids) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    loss = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        n_p, n = len(s.prompt_ids), len(s.prompt_ids) + len(s.answer_ids)
        ids[i, :n] = torch.tensor(s.prompt_ids + s.answer_ids)
        attn[i, :n] = 1
        loss[i, n_p:n] = 1
    return {"input_ids": ids, "attention_mask": attn, "loss_mask": loss}


class DPOCollator:
    """A batch of B pairs -> one batch of 2B sequences (chosen first, then rejected): one forward for both,
    which is what DDP needs (a single forward per backward)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, pairs: list[tuple[Encoded, Encoded]]) -> dict[str, torch.Tensor]:
        return pad_batch([c for c, _ in pairs] + [r for _, r in pairs], self.pad_id)


class SFTCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, items: list[Encoded]) -> dict[str, torch.Tensor]:
        return pad_batch(items, self.pad_id)


def identity_collate(items: list[Any]) -> list[Any]:
    return items
