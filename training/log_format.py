"""Console formatting for the training log: train/val loss, perplexity and bits-per-byte, easy to read at a glance (torch-free).

    [step    1200 |  3.14B tokens]  TRAIN  loss 2.4310  ppl  11.37  bpb 0.842 | lr 3.00e-04 | 52.3k tok/s | MFU 31.2%

    ======================== EVAL  step 2000 | 4.10B tokens ========================
                    loss       ppl       bpb
      train       2.4010     11.03     0.8301
      val         2.5502     12.81     0.8892
        old-val   2.3101     10.07     0.8011
        lang-macro 2.9000    18.17     1.0102
      ---- per language (val) ----
      zul         3.1000     22.20     1.0200
    =============================================================================

ppl = exp(cross-entropy in nats) (the CE WITHOUT the MoE auxiliary loss when the trainer provides it), bpb = bits per UTF-8 byte of the text
the supervised tokens decode to, so languages and tokenizers are comparable.
"""

from __future__ import annotations

import math


def ppl(ce: float | None) -> float | None:
    return None if ce is None else math.exp(min(float(ce), 50.0))


def _f(x, w: int, p: int) -> str:
    return " " * (w - 1) + "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{w}.{p}f}"


def _tokens(n: float) -> str:
    return f"{n / 1e9:6.2f}B" if n >= 1e8 else f"{n / 1e6:6.1f}M"


def format_step_line(iter_num: int, tokens_seen: float, loss: float, ce: float | None, bpb: float | None, lr: float,
                     tokens_per_sec: float | None = None, mfu: float | None = None) -> str:
    c = ce if ce is not None else loss
    parts = [f"[step {iter_num:>7} | {_tokens(tokens_seen)} tokens]  TRAIN  loss {_f(loss, 7, 4)}  ppl {_f(ppl(c), 7, 2)}  bpb {_f(bpb, 6, 3)}",
             f"lr {lr:.2e}"]
    if tokens_per_sec:
        parts.append(f"{tokens_per_sec / 1e3:.1f}k tok/s")
    if mfu is not None:
        parts.append(f"MFU {mfu * 100:.1f}%")
    return " | ".join(parts)


def format_eval_block(iter_num: int, tokens_seen: float, losses: dict) -> str:
    """`losses` is what Trainer.estimate_loss returns: train, val, train_ce, val_ce, train_bpb, val_bpb and, with a stratified eval,
    val_old, val_lang_macro, val_by_lang, val_bpb_by_lang."""
    head = f" EVAL  step {iter_num} | {_tokens(tokens_seen).strip()} tokens "
    width = max(78, len(head) + 8)
    lines = ["=" * ((width - len(head)) // 2) + head + "=" * (width - len(head) - (width - len(head)) // 2),
             f"  {'':14}{'loss':>10}{'ppl':>10}{'bpb':>10}"]

    def row(name, loss, ce, bpb):
        c = ce if ce is not None else loss
        lines.append(f"  {name:14}{_f(loss, 10, 4)}{_f(ppl(c), 10, 2)}{_f(bpb, 10, 4)}")

    row("train", losses.get("train"), losses.get("train_ce"), losses.get("train_bpb"))
    row("val", losses.get("val"), losses.get("val_ce"), losses.get("val_bpb"))
    if losses.get("val_old") is not None:
        row("  old-val", losses["val_old"], losses["val_old"], None)
    if losses.get("val_lang_macro") is not None:
        row("  lang-macro", losses["val_lang_macro"], losses["val_lang_macro"], None)
    by, by_bpb = losses.get("val_by_lang") or {}, losses.get("val_bpb_by_lang") or {}
    if by:
        lines.append("  " + "-" * 8 + " per language (val) " + "-" * 8)
        for lang in sorted(by, key=lambda l: by[l]):
            row(f"  {lang}", by[lang], by[lang], by_bpb.get(lang))
    lines.append("=" * width)
    return "\n".join(lines)
