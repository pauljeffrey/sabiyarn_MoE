"""Reward-based RL with a group baseline (REINFORCE with a leave-in group mean; the GRPO family).

Per optimizer step, every rank
  1. takes `batch_size * grad_accum_steps` prompts and samples `group_size` completions for each (the current
     policy, KV-cache generation, left-padded batches -- the model handles per-row positions);
  2. scores them with a reward function (rl/rewards.py: chrF++, AfriCOMET, or your own);
  3. turns rewards into advantages  A = r - mean_group(r)  (optionally / std): no learned value function, and a
     prompt whose completions all score the same contributes nothing (A = 0);
  4. does ONE gradient step on   -A * log pi(completion)  +  kl_coef * KL(pi || ref)   with the per-token k3
     estimator  exp(d) - d - 1,  d = log ref - log pi.

Because each batch is used once, right after it is sampled, the update is exactly on-policy: no importance ratio,
no clipping and no stored old log-probs are needed. Sampling temperature is also applied when scoring, which keeps
the estimator unbiased (top_p < 1 would bias it; the default is 1.0).

Why this and not PPO: a ~300M model with a sentence-level reward does not need a critic, and the group mean is a
low-variance baseline that costs nothing. Why the KL term matters here: a learned reward (AfriCOMET) can be gamed;
KL to the reference bounds how far the policy can wander to do it.

Reward and prompts: records need `prompt_messages` (or `messages`) and `reference`; `source` is optional (AfriCOMET
uses it). data-gen sft.jsonl records work as they are (reference = the response, source = the input); rlhf/
builds translation records from its dataset.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import structlog
import torch

from rl.common import Run, build_optimizer, load_model, load_tokenizer, pad_id_of, token_logps
from rl.config import RLConfig
from rl.data import Encoded, encode_sft, load_records, pad_batch, render_prompt, split_records
from rl.rewards import build_reward

LOG = structlog.get_logger()


def to_rl_item(rec: dict) -> dict:
    """Normalise a record to {prompt_messages, reference, source, ...}."""
    out = dict(rec)
    if "prompt_messages" not in out:
        msgs = out["messages"]
        out["prompt_messages"] = msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs
    if "reference" not in out:
        out["reference"] = out.get("response") or out["messages"][-1]["content"]
    out.setdefault("source", out.get("input", ""))
    return out


def group_advantages(rewards: list[float], group_size: int, std_norm: bool) -> tuple[torch.Tensor, float]:
    """(advantages (N,), fraction of groups with zero reward variance). Rewards are grouped consecutively."""
    r = torch.tensor(rewards, dtype=torch.float32).view(-1, group_size)
    adv = r - r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    flat_groups = (std.squeeze(1) < 1e-8).float().mean().item()
    if std_norm:
        adv = adv / (std + 1e-4)
    return adv.flatten(), flat_groups


@torch.no_grad()
def sample_completions(model, tok, prompts: list[list[int]], cfg: RLConfig, *, greedy: bool = False, k: Optional[int] = None,
                       device="cpu") -> list[list[int]]:
    """`k` completions per prompt, as token-id lists (cut after the first EOS, EOS included). Rows come out grouped
    by prompt: prompt i owns rows i*k .. i*k+k-1."""
    k = 1 if greedy else (k or cfg.group_size)
    pad, eos = pad_id_of(tok), tok.eos_token_id
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):  # LEFT-pad: generation appends on the right
        ids[i, width - len(p):] = torch.tensor(p)
        mask[i, width - len(p):] = 1
    kwargs = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=0)
    out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=cfg.max_new_tokens,
                         num_return_sequences=k, pad_token_id=pad, eos_token_id=eos, use_cache=True, **kwargs)
    comps = []
    for row in out[:, width:].tolist():
        if eos in row:
            row = row[: row.index(eos) + 1]
        comps.append(row)
    return comps


def _chunks(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def rl_update(run: Run, policy, ref, optimizer, seqs: list[Encoded], advantages: torch.Tensor, cfg: RLConfig, lr: float) -> dict:
    """One optimizer step over all sampled sequences of this rank (chunked to `rl_micro_batch`)."""
    acc = run.accelerator
    total_tokens = float(sum(len(s.answer_ids) for s in seqs))
    stats = {"kl": 0.0, "pg": 0.0}
    chunk_list = list(_chunks(len(seqs), cfg.rl_micro_batch))
    optimizer.zero_grad(set_to_none=True)
    for ci, sl in enumerate(chunk_list):
        batch = {k: v.to(run.device) for k, v in pad_batch(seqs[sl], run.pad_id).items()}
        adv = advantages[sl].to(run.device).unsqueeze(1)
        sync = ci == len(chunk_list) - 1
        ctx = acc.no_sync(policy) if not sync else _null()
        with ctx:
            lp = token_logps(policy, batch["input_ids"], batch["attention_mask"], cfg.temperature)
            with torch.no_grad():
                ref_lp = token_logps(ref, batch["input_ids"], batch["attention_mask"], cfg.temperature)
            m = batch["loss_mask"][:, 1:].to(lp.dtype)
            d = ref_lp - lp
            kl = torch.exp(d) - d - 1.0  # k3, >= 0, unbiased for KL(pi||ref) under samples from pi
            per_tok = -(adv * lp) + cfg.kl_coef * kl
            loss = (per_tok * m).sum() / max(total_tokens, 1.0)
            acc.backward(loss)
        stats["kl"] += (kl.detach() * m).sum().item()
        stats["pg"] += (-(adv * lp).detach() * m).sum().item()
    grad_norm = float(acc.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"train/kl": stats["kl"] / max(total_tokens, 1.0), "train/pg_loss": stats["pg"] / max(total_tokens, 1.0),
            "train/lr": lr, "train/grad_norm": grad_norm}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def evaluate(run: Run, policy, tok, items: list[dict], reward_fn: Callable, cfg: RLConfig) -> dict[str, float]:
    """Greedy-decode a held-out slice; mean reward (same metric the policy is trained on)."""
    acc = run.accelerator
    mine = items[acc.process_index :: acc.num_processes][: max(1, 32)]
    if not mine:
        return {}
    prompts = [render_prompt(tok, it["prompt_messages"]) for it in mine]
    with acc.autocast():
        comps = sample_completions(acc.unwrap_model(policy), tok, prompts, cfg, greedy=True, device=run.device)
    texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
    r = reward_fn(mine, texts)
    return run.mean_over_ranks({"eval/reward": sum(r) / len(r)})


def train_rl(cfg: RLConfig, reward_fn: Optional[Callable] = None, run: Optional[Run] = None,
             records: Optional[list[dict]] = None) -> str:
    if cfg.max_prompt_len + cfg.max_new_tokens > cfg.max_seq_len:
        raise SystemExit("max_prompt_len + max_new_tokens must be <= max_seq_len")
    run = run or Run(cfg, grad_accum=1)
    acc = run.accelerator
    tok = load_tokenizer(cfg)
    run.pad_id = pad_id_of(tok)
    rows = records if records is not None else load_records(cfg)
    train_rows, eval_rows = split_records([to_rl_item(r) for r in rows], cfg.eval_samples)
    train_items = [r for r in train_rows if len(render_prompt(tok, r["prompt_messages"])) <= cfg.max_prompt_len]
    eval_items = [r for r in eval_rows if len(render_prompt(tok, r["prompt_messages"])) <= cfg.max_prompt_len]
    if len(train_items) < acc.num_processes:
        raise SystemExit("not enough prompts within max_prompt_len for this many ranks")
    if reward_fn is None:
        reward_fn = build_reward(cfg, device=str(run.device))

    policy = load_model(cfg, trainable=True)
    ref = load_model(cfg, cfg.reference_path or cfg.model_path, trainable=False).to(run.device)
    optimizer = build_optimizer(policy, cfg)
    policy, optimizer = acc.prepare(policy, optimizer)

    prompts_per_step = cfg.batch_size * cfg.grad_accum_steps
    steps_per_epoch = max(1, len(train_items) // (prompts_per_step * acc.num_processes))
    total = int(math.ceil(steps_per_epoch * cfg.epochs))
    total = min(total, cfg.max_steps) if cfg.max_steps > 0 else total
    cfg.set_schedule(total)
    run.start_tracking({"train_prompts": len(train_items), "total_steps": total, "prompts_per_step_per_rank": prompts_per_step})
    LOG.info("rl_start", prompts=len(train_items), steps=total, group_size=cfg.group_size, reward=cfg.reward,
             world_size=acc.num_processes)

    try:
        cursor, epoch, order = 0, 0, []
        while run.step < total:
            if cursor + prompts_per_step > len(order):  # new epoch: same permutation on every rank, disjoint slices
                order = random.Random(cfg.seed + epoch).sample(range(len(train_items)), len(train_items))
                order = order[acc.process_index :: acc.num_processes]
                cursor, epoch = 0, epoch + 1
            batch_items = [train_items[i] for i in order[cursor : cursor + prompts_per_step]]
            cursor += prompts_per_step

            prompts = [render_prompt(tok, it["prompt_messages"]) for it in batch_items]
            with acc.autocast():
                comps = sample_completions(acc.unwrap_model(policy), tok, prompts, cfg, device=run.device)
            texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
            expanded = [it for it in batch_items for _ in range(cfg.group_size)]
            rewards = reward_fn(expanded, texts)
            adv, flat = group_advantages(rewards, cfg.group_size, cfg.advantage_std_norm)
            seqs = [Encoded(prompts[i // cfg.group_size], c) for i, c in enumerate(comps) if c]
            keep = [i for i, c in enumerate(comps) if c]
            adv = adv[keep]

            lr = run.apply_lr(optimizer, run.step)
            with acc.autocast():
                m = rl_update(run, policy, ref, optimizer, seqs, adv, cfg, lr)
            run.step += 1

            if run.step % cfg.log_every == 0 or run.step == 1:
                r = torch.tensor(rewards)
                eos = tok.eos_token_id
                m.update({"train/reward": r.mean().item(), "train/reward_std": r.std(unbiased=False).item(),
                          "train/flat_group_frac": flat,
                          "train/completion_len": sum(len(c) for c in comps) / len(comps),
                          "train/eos_rate": sum(1 for c in comps if c and c[-1] == eos) / len(comps)})
                run.log(run.mean_over_ranks(m), run.step, echo=True)
                if run.master:
                    LOG.info("sample", prompt=tok.decode(prompts[0], skip_special_tokens=False)[-160:], completion=texts[0][:200],
                             reward=round(rewards[0], 4))
            if eval_items and cfg.eval_every and run.step % cfg.eval_every == 0:
                run.log(evaluate(run, policy, tok, eval_items, reward_fn, cfg), run.step, echo=True)
            if cfg.save_every and run.step % cfg.save_every == 0:
                run.save(policy, tok, f"step_{run.step}")
        if eval_items:
            run.log(evaluate(run, policy, tok, eval_items, reward_fn, cfg), run.step, echo=True)
        final = run.save(policy, tok, "final")
        run.publish(final)
        run.finish()
        return final
    except BaseException:
        run.finish(status="FAILED")
        raise
