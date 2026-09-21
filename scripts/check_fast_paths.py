#!/usr/bin/env python3
"""Verify (and benchmark) sparse MoE and FlexAttention on real hardware, on real weights and data.

    python scripts/check_fast_paths.py --model out_280M/<run>/ckpt_best            # a checkpoint dir
    python scripts/check_fast_paths.py --model Aletheia-ng/SabiYarn_MoE-280M --batch-size 4
    python scripts/check_fast_paths.py --tiny                                       # CPU smoke test, no GPU/data

It runs the SAME real batch through four configurations of the model --

    dense MoE + dense SDPA mask   (baseline: the original code path)
    sparse MoE + dense SDPA mask
    dense MoE + flex block mask
    sparse MoE + flex block mask

-- and for each reports the loss, gradient norm, throughput and peak GPU memory, and whether the loss
and gradients agree with the baseline within bf16 tolerance. Sparse MoE and flex are only "safe to
turn on" if this prints PASS for them on YOUR GPU; it also tells you which config to put in
train_config.yaml and how much larger a micro-batch that buys (raise --batch-size until the
baseline OOMs to see the headroom directly).

Uses the trainer's own precision setup (fp32 parameters + bf16 autocast) and the same
document-causal rule, so a PASS here means the trainer will compute the same thing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabiyarn.model.modeling import MoE  # noqa: E402
from training.data_sampler import n_blocks_for, read_blocks  # noqa: E402
from training.training_attention_mask import build_document_block_mask, build_document_causal_mask  # noqa: E402

CONFIGS = [
    ("dense  + sdpa_mask", False, "sdpa_mask"),  # baseline
    ("sparse + sdpa_mask", True, "sdpa_mask"),
    ("dense  + flex", False, "flex"),
    ("sparse + flex", True, "flex"),
]
LOSS_RTOL = 2e-2  # bf16 autocast: two correct implementations differ by rounding noise
GRAD_RTOL = 5e-2
GRAD_MIN_COS = 0.99   # cosine similarity of the full flattened gradient vs the baseline
GRAD_MAX_REL_L2 = 0.15  # ||g - g_base|| / ||g_base||


def set_sparse(model, sparse: bool) -> None:
    for m in model.modules():
        if isinstance(m, MoE):
            m.sparse_dispatch = sparse


def run_config(model, x, y, eos: int, sparse: bool, attn: str, steps: int, device: str):
    """(loss, grad_norm, tokens/s, peak GiB, grads) for `steps` timed fwd+bwd passes; forward-only where
    the backend can't differentiate (FlexAttention on CPU). `grads` is {param name: fp32 CPU gradient}
    of the last pass (None when there is no backward)."""
    set_sparse(model, sparse)
    model.eval()  # no router noise: the comparison must be deterministic
    backward = not (attn == "flex" and device == "cpu")
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)

    def one_pass():
        mask = build_document_block_mask(x, eos) if attn == "flex" else build_document_causal_mask(x, eos)
        model.zero_grad(set_to_none=True)
        with amp, torch.set_grad_enabled(backward):
            loss = model(input_ids=x, attention_mask=mask, targets=y).loss
        if backward:
            loss.backward()
        return loss.detach().float().item()

    loss = one_pass()  # warmup (also compiles flex on the first call)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_pass()
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    gnorm = None
    if backward:
        gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
    grads = ({n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
             if backward else None)
    return loss, gnorm, x.numel() / dt, peak, grads


def grad_agreement(base: dict, other: dict) -> dict:
    """How closely two gradients agree, over ALL parameters at once (and the worst single tensor).

    The total norm alone can hide a wrong gradient of the right size, so this also reports the cosine
    similarity and relative L2 error of the concatenated gradient vectors. Two correct implementations
    differ only by bf16 rounding: cosine ~0.999+, relative error a few percent."""
    dot = na = nb = err = 0.0
    worst_name, worst_rel = "", 0.0
    for name, a in base.items():
        b = other.get(name)
        if b is None:
            return {"cos": 0.0, "rel_l2": float("inf"), "worst": (name, float("inf"))}
        a64, b64 = a.double().flatten(), b.double().flatten()
        dot += float(a64 @ b64)
        na += float(a64 @ a64)
        nb += float(b64 @ b64)
        d = float(((a64 - b64) ** 2).sum())
        err += d
        rel = (d ** 0.5) / max(float(a64 @ a64) ** 0.5, 1e-12)
        if float(a64 @ a64) > 1e-12 and rel > worst_rel:
            worst_name, worst_rel = name, rel
    cos = dot / max((na ** 0.5) * (nb ** 0.5), 1e-30)
    return {"cos": cos, "rel_l2": (err ** 0.5) / max(na ** 0.5, 1e-30), "worst": (worst_name, worst_rel)}


def compare(model, x, y, eos: int, steps: int, device: str) -> list[dict]:
    rows = []
    base_grads = None
    for i, (name, sparse, attn) in enumerate(CONFIGS):
        loss, gnorm, tps, peak, grads = run_config(model, x, y, eos, sparse, attn, steps, device)
        row = {"name": name, "loss": loss, "gnorm": gnorm, "tps": tps, "peak": peak, "agree": None}
        if i == 0:
            base_grads = grads
        elif grads is not None and base_grads is not None:
            row["agree"] = grad_agreement(base_grads, grads)
        rows.append(row)
    base = rows[0]
    for r in rows:
        r["loss_ok"] = abs(r["loss"] - base["loss"]) <= LOSS_RTOL * max(1e-6, abs(base["loss"]))
        if r["gnorm"] is None or base["gnorm"] is None:
            r["grad_ok"] = None  # not differentiable on this backend (flex on CPU)
        else:
            r["grad_ok"] = abs(r["gnorm"] - base["gnorm"]) <= GRAD_RTOL * max(1e-6, base["gnorm"])
            if r["agree"] is not None:
                r["grad_ok"] &= r["agree"]["cos"] >= GRAD_MIN_COS and r["agree"]["rel_l2"] <= GRAD_MAX_REL_L2
        r["speedup"] = r["tps"] / base["tps"]
    return rows


def report(rows: list[dict]) -> bool:
    print(f"\n{'config':22s}{'loss':>10s}{'|grad|':>10s}{'tok/s':>12s}{'speedup':>9s}{'peak GiB':>10s}   verdict")
    all_ok = True
    for r in rows:
        ok = r["loss_ok"] and r["grad_ok"] is not False
        all_ok &= ok
        g = "n/a" if r["gnorm"] is None else f"{r['gnorm']:.3f}"
        note = "baseline" if r is rows[0] else ("PASS" if ok else "FAIL (loss/grad disagree with baseline)")
        if r["grad_ok"] is None and r is not rows[0]:
            note += " (forward only: no CPU backward for flex)"
        print(f"{r['name']:22s}{r['loss']:10.4f}{g:>10s}{r['tps']:12.0f}{r['speedup']:8.2f}x{r['peak']:10.2f}   {note}")
        if r["agree"] is not None:
            a = r["agree"]
            print(f"{'':22s}  gradient vs baseline: cosine {a['cos']:.5f} (need >= {GRAD_MIN_COS}), "
                  f"rel. L2 error {a['rel_l2']:.4f} (need <= {GRAD_MAX_REL_L2}), worst tensor {a['worst'][0]} @ {a['worst'][1]:.3f}")
    return all_ok


def advise(rows: list[dict]) -> None:
    by = {r["name"]: r for r in rows}
    sparse_ok = by["sparse + sdpa_mask"]["loss_ok"] and by["sparse + sdpa_mask"]["grad_ok"] is not False
    flex_ok = by["dense  + flex"]["loss_ok"] and by["dense  + flex"]["grad_ok"] is not False
    print("\nrecommendation:")
    print(f"  model.moe_dispatch:        {'sparse' if sparse_ok else 'dense   # sparse disagreed with dense -- do not use'}")
    print(f"  training.attention_impl:   {'flex' if flex_ok else 'sdpa_mask   # flex failed or disagreed on this GPU'}")
    best = by["sparse + flex"] if (sparse_ok and flex_ok) else by["sparse + sdpa_mask"] if sparse_ok else rows[0]
    print(f"  expected gain: {best['speedup']:.2f}x tokens/s and peak memory {rows[0]['peak']:.1f} -> {best['peak']:.1f} GiB "
          "(then raise TRAIN_BATCH_SIZE until peak_mem_gib in the step log nears the card's limit)")


def load_real_batch(args, block_size: int, device: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.data:
        data_path = args.data
    else:
        from training.load_config import load_train_config

        data_path = load_train_config().eval_data_path
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    n = n_blocks_for(len(data), block_size)
    ids = np.random.default_rng(0).choice(n, size=args.batch_size, replace=False)
    x, y = read_blocks(data, ids, block_size)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), tok.eos_token_id, data_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="checkpoint dir or HF repo id (weights)")
    ap.add_argument("--tokenizer", default="BeardedMonster/SabiYarn-32k")
    ap.add_argument("--data", help="a tokenized .bin (default: the eval bin from train_config.yaml / VAL_DATA_PATH)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=3, help="timed passes per config (after one warmup)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="random tiny model + synthetic data (CPU smoke test)")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.tiny:
        from sabiyarn.model.configuration import GPTJXMoEConfig
        from sabiyarn.model.modeling import GPTJXMoEForCausalLM

        cfg = GPTJXMoEConfig(block_size=128, vocab_size=200, n_layer=2, n_heads=2, n_embd=32, use_moe=True,
                             num_experts=4, num_experts_per_tok=2, moe_dim=64, expert_per_layer={"0": 2, "1": 4})
        model = GPTJXMoEForCausalLM(cfg)
        x = torch.randint(3, 200, (args.batch_size, 128))
        x[:, 40] = 2
        y = torch.randint(3, 200, (args.batch_size, 128))
        eos = 2
        print("tiny synthetic model/data (CPU smoke test)")
    else:
        if not args.model:
            ap.error("--model is required (or use --tiny)")
        from sabiyarn.model.modeling import GPTJXMoEForCausalLM

        model = GPTJXMoEForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        x, y, eos, data_path = load_real_batch(args, args.seq, args.device)
        print(f"model {args.model} | data {data_path} | batch {tuple(x.shape)} | eos-token positions per row: "
              f"{[(row == eos).sum().item() for row in x]}")
    model = model.to(args.device)
    x, y = x.to(args.device), y.to(args.device)

    rows = compare(model, x, y, eos, args.steps, args.device)
    ok = report(rows)
    advise(rows)
    if args.device == "cpu":
        print("\nnote: CPU timings/memory are not meaningful -- this run only checks that the paths agree numerically.\n"
              "Run it on the GPU box for speed, memory and the flex-attention check.")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
