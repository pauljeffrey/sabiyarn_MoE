"""Small supervised warm-start on chat-formatted data (data-gen's sft.jsonl, or a Hub dataset such as the
translation set rlhf/ uses). Full pretraining-scale SFT belongs in training/new_train.py (`mode: sft`, packed
bins, document masks); this is for fine-tuning-sized jobs that start from an already SFT'd checkpoint and need to
be aligned to a narrow task (e.g. translation) before RL. Loss: mean token NLL over answer tokens only."""

from __future__ import annotations

import math
from typing import Optional

import structlog
import torch
from torch.utils.data import DataLoader

from rl.common import Run, build_optimizer, load_model, load_tokenizer, pad_id_of, steps_for, token_logps
from rl.config import RLConfig
from rl.data import SFTCollator, encode_sft, load_records, split_records

LOG = structlog.get_logger()


def _nll(policy, batch) -> tuple[torch.Tensor, torch.Tensor]:
    lp = token_logps(policy, batch["input_ids"], batch["attention_mask"])
    m = batch["loss_mask"][:, 1:].to(lp.dtype)
    return -(lp * m).sum(), m.sum()


@torch.no_grad()
def evaluate(run: Run, policy, loader) -> dict[str, float]:
    tot = torch.zeros(2, device=run.device)
    for batch in loader:
        s, n = _nll(policy, batch)
        tot += torch.stack([s, n])
    tot = run.accelerator.reduce(tot, reduction="sum")
    return {"eval/loss": (tot[0] / tot[1].clamp(min=1)).item()}


def train_sft(cfg: RLConfig, run: Optional[Run] = None) -> str:
    run = run or Run(cfg)
    acc = run.accelerator
    tok = load_tokenizer(cfg)
    train, held = (encode_sft(tok, part, cfg) for part in split_records(load_records(cfg), cfg.eval_samples))
    if not train:
        raise SystemExit("no SFT examples after filtering -- check data_path / dataset_id / max_seq_len")
    collate = SFTCollator(pad_id_of(tok))
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(cfg.seed))
    eval_loader = DataLoader(held, batch_size=cfg.batch_size, collate_fn=collate) if held else None

    policy = load_model(cfg, trainable=True)
    optimizer = build_optimizer(policy, cfg)
    policy, optimizer, loader = acc.prepare(policy, optimizer, loader)
    if eval_loader is not None:
        eval_loader = acc.prepare(eval_loader)
    total = steps_for(cfg, len(loader))
    cfg.set_schedule(total)
    run.start_tracking({"train_examples": len(train), "total_steps": total})
    LOG.info("sft_start", examples=len(train), steps=total, world_size=acc.num_processes)

    loss_sum, tok_sum, grad_norm = 0.0, 0.0, 0.0
    try:
        for epoch in range(max(1, math.ceil(cfg.epochs))):
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            for batch in loader:
                with acc.accumulate(policy):
                    lr = run.apply_lr(optimizer, run.step)
                    s, n = _nll(policy, batch)
                    acc.backward(s / n.clamp(min=1))
                    if acc.sync_gradients:
                        grad_norm = float(acc.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum, tok_sum = loss_sum + s.item(), tok_sum + n.item()
                if not acc.sync_gradients:
                    continue
                run.step += 1
                if run.step % cfg.log_every == 0 or run.step == 1:
                    m = run.mean_over_ranks({"train/loss": loss_sum / max(tok_sum, 1.0)})
                    run.log({**m, "train/lr": lr, "train/grad_norm": grad_norm}, run.step, echo=True)
                    loss_sum, tok_sum = 0.0, 0.0
                if eval_loader is not None and cfg.eval_every and run.step % cfg.eval_every == 0:
                    run.log(evaluate(run, policy, eval_loader), run.step, echo=True)
                if cfg.save_every and run.step % cfg.save_every == 0:
                    run.save(policy, tok, f"step_{run.step}")
                if run.step >= total:
                    break
            if run.step >= total:
                break
        if eval_loader is not None:
            run.log(evaluate(run, policy, eval_loader), run.step, echo=True)
        final = run.save(policy, tok, "final")
        run.publish(final)
        run.finish()
        return final
    except BaseException:
        run.finish(status="FAILED")
        raise
