#!/usr/bin/env python
"""Calibrate the reward-shaping anchors per language pair from a human-translated benchmark, with AfriCOMET-QE.

    python rlhf/scripts/calibrate_shaping.py --bench bench.jsonl --out outputs/shaping.json [--model outputs/translate-sft/final]
        [--reward-python .venv-comet/bin/python] [--max-per-pair 500]

`bench` rows: source_lang, targ_lang, input (source text), response (HUMAN reference translation) -- e.g. FLORES+ dev/devtest for every
pair you train on (interlanguage pairs included). For every pair it reports and writes
  * anchors  [t_fail, t_mid, t_great] = P10 / P50 / P90 of QE(source, human reference): "even a good human translation rarely scores
             below t_fail, and t_great is what a good human translation scores"
  * auc      how well QE separates the human reference from a MISMATCHED reference (the same pair, another sentence): the check that
             the metric means something for THIS pair. Near 0.5 means the metric cannot see translation quality there -- do not
             train on it for that pair. Pairs under --min-auc are flagged and left out of the file.
  * model    (with --model) the same quantiles for the SFT model's greedy translations, and the share of its outputs below t_fail,
             so you can see where it sits relative to the human anchors before RL starts.
Use the result with  reward_shaping=calibrated reward_calibration_path=<out>  (or adaptive).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl.config import load_rl_config  # noqa: E402
from rl.rejection_sampling import _iter_input  # noqa: E402
from rl.shaping import auc, compute_anchors, pair_anchors, reliability_from_auc  # noqa: E402


def calibrate(rows: list[dict], scorer, translate=None, max_per_pair: int = 500, min_auc: float = 0.7, seed: int = 0) -> dict:
    """scorer(items, completions) -> QE scores; translate(items) -> model outputs (optional)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        if r.get("input") and r.get("response") and r.get("source_lang") and r.get("targ_lang"):
            by[f"{r['source_lang']}>{r['targ_lang']}"].append(r)
    out = {}
    for pair, rs in sorted(by.items()):
        rng.shuffle(rs)
        rs = rs[:max_per_pair]
        items = [{"source": r["input"], "reference": ""} for r in rs]
        human = scorer(items, [r["response"] for r in rs])
        shuffled = [r["response"] for r in rs[1:]] + [rs[0]["response"]]
        mism = scorer(items, shuffled)
        copy = scorer(items, [r["input"] for r in rs])                                    # "translating" by copying the source
        trunc = scorer(items, [" ".join(r["response"].split()[: max(1, len(r["response"].split()) // 2)]) for r in rs])   # omission
        entry = {"n": len(rs), "anchors": list(pair_anchors(human, mism)), "human_quantiles_p10_p50_p90": list(compute_anchors(human)),
                 "mismatched_p90": float(sorted(mism)[int(0.9 * (len(mism) - 1))]), "auc_vs_mismatched": auc(human, mism),
                 "auc_vs_copy": auc(human, copy), "auc_vs_half_omitted": auc(human, trunc),
                 "mean_human": sum(human) / len(human), "mean_mismatched": sum(mism) / len(mism), "mean_copy": sum(copy) / len(copy),
                 "mean_half_omitted": sum(trunc) / len(trunc), "share_human_below_0.46": sum(h < 0.46 for h in human) / len(human)}
        if translate is not None:
            mt = translate(rs)
            ms = scorer(items, mt)
            entry["model"] = {"anchors": list(compute_anchors(ms)), "mean": sum(ms) / len(ms),
                              "below_t_fail": sum(s < entry["anchors"][0] for s in ms) / len(ms)}
        entry["usable"] = bool(entry["auc_vs_mismatched"] >= min_auc)
        entry["reliability"] = reliability_from_auc(entry["auc_vs_mismatched"]) if entry["usable"] else 0.0
        out[pair] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="jsonl/parquet glob (or hub://repo/glob)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reward-python", default=None)
    ap.add_argument("--max-per-pair", type=int, default=500)
    ap.add_argument("--min-auc", type=float, default=0.7)
    a = ap.parse_args()
    cfg = load_rl_config(str(ROOT / "rlhf" / "configs" / "default.yaml"), **({"reward_python": a.reward_python} if a.reward_python else {}))
    from rl.rewards import _africomet

    scorer = _africomet(cfg, "cpu", cfg.reward_qe_id)
    translate = None
    if a.model:
        import torch

        from rl.common import load_model, load_tokenizer
        from rl.data import render_prompt
        from rl.policy_gradient import sample_completions
        from rlhf.translation import to_record

        tok, device = load_tokenizer(cfg), ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(cfg, a.model, trainable=False).to(device).eval()

        def translate(rs):
            recs = [to_record({**r, "output_structure": r.get("output_structure") or ""}) for r in rs]
            outs = []
            for i in range(0, len(recs), 16):
                prompts = [render_prompt(tok, x["prompt_messages"]) for x in recs[i:i + 16]]
                comps = sample_completions(model, tok, prompts, cfg, greedy=True, device=device)
                outs += [tok.decode(c, skip_special_tokens=False) for c in comps]
            from rl.structure import parse
            return [parse(t).response for t in outs]

    res = calibrate(list(_iter_input(a.bench)), scorer, translate, a.max_per_pair, a.min_auc)
    for pair, e in res.items():
        m = e.get("model")
        print(f"{pair:10} n={e['n']:4} t_fail/t_mid/t_great={[round(x, 3) for x in e['anchors']]} rel={e['reliability']:.2f} auc: mismatched={e['auc_vs_mismatched']:.3f} copy={e['auc_vs_copy']:.3f} half-omitted={e['auc_vs_half_omitted']:.3f}"
              + (f" | model P50={m['anchors'][1]:.3f} below_t_fail={m['below_t_fail']:.0%}" if m else "")
              + ("" if e["usable"] else "  <-- QE does not separate real from mismatched here; excluded"))
    def group(pred):
        g = [v for k, v in res.items() if pred(*k.split(">"))]
        return g

    for name, pred in (("English-centric (eng on either side)", lambda a_, b_: "eng" in (a_, b_)),
                       ("interlanguage (neither side English)", lambda a_, b_: "eng" not in (a_, b_))):
        g = group(pred)
        if g:
            f = lambda key: sum(v[key] for v in g) / len(g)
            print(f"\n{name}: {len(g)} pairs | mean AUC vs mismatched {f('auc_vs_mismatched'):.3f}, vs copy {f('auc_vs_copy'):.3f}, vs half-omitted "
                  f"{f('auc_vs_half_omitted'):.3f} | mean QE human {f('mean_human'):.3f}, mismatched {f('mean_mismatched'):.3f}, copy {f('mean_copy'):.3f}, "
                  f"half-omitted {f('mean_half_omitted'):.3f} | human refs below 0.46: {f('share_human_below_0.46'):.1%}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))          # every pair, with usable / reliability; the shaper skips the unusable ones
    n_ok = sum(v["usable"] for v in res.values())
    print(f"wrote {len(res)} pairs ({n_ok} usable, {len(res) - n_ok} excluded) to {a.out}")


if __name__ == "__main__":
    main()
