"""Direct Preference Optimisation on data-gen's dpo.jsonl (Rafailov et al. 2023).

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )

Variants (config `dpo_loss`): `ipo` (Azar et al. 2023: squared loss on the length-normalised log-ratio margin,
robust to over-fitting deterministic preferences) and `hinge` (SLiC). `label_smoothing` gives conservative DPO;
`sft_alpha` adds an NLL term on the chosen answer (RPO), which stops both likelihoods from drifting down.

Design notes specific to this repo
  * The data-gen rejected answers are OFF-POLICY (a model was asked to write a flawed answer), so DPO learns
    "what a typical flaw looks like" more than it improves the policy's own mistakes. Keep beta moderate and the
    learning rate small (5e-7 by default), and watch `reward_accuracy`: ~1.0 within a few hundred steps means the
    pairs are separable by surface cues (length, language, refusals) rather than by quality.
  * Chosen and rejected are one 2B-row batch -> one forward per backward (DDP needs that).
  * Policy and frozen reference are both scored in eval mode (no MoE router noise), so the very first loss is
    ln 2 = 0.6931 and `init_logratio_absmax` ~ 0; anything else means the two models differ.
"""

from __future__ import annotations

import math
from typing import Optional

import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl.common import (
    Run, build_optimizer, load_model, load_tokenizer, masked_sum_and_count, pad_id_of, steps_for, token_logps,
)
from rl.config import RLConfig
from rl.data import DPOCollator, encode_dpo, load_records, split_records

LOG = structlog.get_logger()


def dpo_loss(pi_c, pi_r, ref_c, ref_r, *, beta: float, loss_type: str = "sigmoid", label_smoothing: float = 0.0):
    """Per-pair loss and the implicit rewards. All inputs are (B,) sequence log-probs (or their per-token means)."""
    z = (pi_c - pi_r) - (ref_c - ref_r)
    if loss_type == "sigmoid":
        loss = -(1 - label_smoothing) * F.logsigmoid(beta * z) - label_smoothing * F.logsigmoid(-beta * z)
    elif loss_type == "ipo":
        loss = (z - 1.0 / (2.0 * beta)) ** 2
    elif loss_type == "hinge":
        loss = F.relu(1.0 - beta * z)
    else:
        raise ValueError(f"unknown dpo_loss {loss_type!r}")
    return loss, beta * (pi_c - ref_c).detach(), beta * (pi_r - ref_r).detach()


def sequence_scores(policy, ref, batch, *, length_normalize: bool):
    """(policy sums/means (2B,), ref sums/means (2B,), completion token counts (2B,))."""
    ids, attn, lmask = batch["input_ids"], batch["attention_mask"], batch["loss_mask"]
    pi_s, n = masked_sum_and_count(token_logps(policy, ids, attn), lmask)
    with torch.no_grad():
        ref_s, _ = masked_sum_and_count(token_logps(ref, ids, attn), lmask)
    if length_normalize:
        pi_s, ref_s = pi_s / n.clamp(min=1), ref_s / n.clamp(min=1)
    return pi_s, ref_s, n


def _pair_terms(cfg: RLConfig, policy, ref, batch):
    pi, rf, n = sequence_scores(policy, ref, batch, length_normalize=cfg.length_normalize)
    b = pi.size(0) // 2
    loss, rc, rr = dpo_loss(pi[:b], pi[b:], rf[:b], rf[b:], beta=cfg.beta, loss_type=cfg.dpo_loss,
                            label_smoothing=cfg.label_smoothing)
    if cfg.sft_alpha > 0:
        nll_chosen = -(pi[:b] if cfg.length_normalize else pi[:b] / n[:b].clamp(min=1))
        loss = loss + cfg.sft_alpha * nll_chosen
    stats = {
        "loss": loss.detach().sum().item(),
        "reward_acc": (rc > rr).float().sum().item(),
        "margin": (rc - rr).sum().item(),
        "chosen_reward": rc.sum().item(),
        "rejected_reward": rr.sum().item(),
        "chosen_logp": pi[:b].detach().sum().item(),
        "rejected_logp": pi[b:].detach().sum().item(),
        "init_logratio_absmax": (pi - rf).detach().abs().max().item(),
        "n": float(b),
    }
    return loss.mean(), stats


@torch.no_grad()
def evaluate(run: Run, policy, ref, loader) -> dict[str, float]:
    tot: dict[str, float] = {}
    for batch in loader:
        _, s = _pair_terms(run.cfg, policy, ref, batch)
        for k in ("loss", "reward_acc", "margin", "n"):
            tot[k] = tot.get(k, 0.0) + s[k]
    t = torch.tensor([tot.get(k, 0.0) for k in ("loss", "reward_acc", "margin", "n")], device=run.device)
    t = run.accelerator.reduce(t, reduction="sum")
    n = max(t[3].item(), 1.0)
    return {"eval/loss": t[0].item() / n, "eval/reward_accuracy": t[1].item() / n, "eval/reward_margin": t[2].item() / n}


def train_dpo(cfg: RLConfig, run: Optional[Run] = None) -> str:
    run = run or Run(cfg)
    acc = run.accelerator
    tok = load_tokenizer(cfg)
    train_pairs, eval_pairs = (encode_dpo(tok, part, cfg) for part in split_records(load_records(cfg), cfg.eval_samples))
    if not train_pairs:
        raise SystemExit("no DPO training pairs after filtering -- check data_path / max_seq_len / languages")
    collate = DPOCollator(pad_id_of(tok))
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_pairs, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate,
                        generator=gen)
    eval_loader = (DataLoader(eval_pairs, batch_size=cfg.batch_size, collate_fn=collate) if eval_pairs else None)

    policy = load_model(cfg, trainable=True)
    ref = load_model(cfg, cfg.reference_path or cfg.model_path, trainable=False).to(run.device)
    optimizer = build_optimizer(policy, cfg)
    policy, optimizer, loader = acc.prepare(policy, optimizer, loader)
    if eval_loader is not None:
        eval_loader = acc.prepare(eval_loader)

    total = steps_for(cfg, len(loader))
    cfg.set_schedule(total)
    run.start_tracking({"train_pairs": len(train_pairs), "eval_pairs": len(eval_pairs), "total_steps": total})
    LOG.info("dpo_start", train_pairs=len(train_pairs), eval_pairs=len(eval_pairs), steps=total,
             world_size=acc.num_processes, beta=cfg.beta, loss=cfg.dpo_loss)

    window: dict[str, float] = {}
    grad_norm = 0.0
    try:
        for epoch in range(max(1, math.ceil(cfg.epochs))):
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            for batch in loader:
                with acc.accumulate(policy):
                    lr = run.apply_lr(optimizer, run.step)
                    loss, stats = _pair_terms(cfg, policy, ref, batch)
                    acc.backward(loss)
                    if acc.sync_gradients:
                        grad_norm = float(acc.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for k, v in stats.items():
                    window[k] = max(window.get(k, 0.0), v) if k == "init_logratio_absmax" else window.get(k, 0.0) + v
                if not acc.sync_gradients:
                    continue
                run.step += 1
                if run.step % cfg.log_every == 0 or run.step == 1:
                    n = max(window.pop("n"), 1.0)
                    init = window.pop("init_logratio_absmax")
                    m = {f"train/{k}": v / n for k, v in window.items()}
                    m = {**run.mean_over_ranks(m), "train/lr": lr, "train/grad_norm": grad_norm}
                    if run.step == 1:
                        m["train/init_logratio_absmax"] = init  # ~0 or the reference differs from the policy
                    run.log(m, run.step, echo=True)
                    window = {}
                if eval_loader is not None and cfg.eval_every and run.step % cfg.eval_every == 0:
                    run.log(evaluate(run, policy, ref, eval_loader), run.step, echo=True)
                if cfg.save_every and run.step % cfg.save_every == 0:
                    run.save(policy, tok, f"step_{run.step}")
                if run.step >= total:
                    break
            if run.step >= total:
                break
        if eval_loader is not None:
            run.log(evaluate(run, policy, ref, eval_loader), run.step, echo=True)
        final = run.save(policy, tok, "final")
        run.publish(final)
        run.finish()
        return final
    except BaseException:
        run.finish(status="FAILED")
        raise
