"""CLI: evaluate a checkpoint on translation / topic / sentiment / NER / MMLU.

    python -m eval_suite.run --model out_280M/<run>/ckpt_best --tasks all --langs all --limit 200 --name base_10k
    python -m eval_suite.run --model <ckpt> --tasks translation --langs yor,hau --style chat   # SFT model

Writes eval_results/<name>/{results.json, summary.md, predictions/<task>_<lang>.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from eval_suite import tasks
from eval_suite.data import DataUnavailable
from eval_suite.langs import LANGS, parse_langs

ALL_TASKS = ["translation", "topic", "sentiment", "ner", "mmlu"]


def make_africomet(model_name: str, batch_size: int = 16):
    """Optional reference-based AfriCOMET score (needs `pip install unbabel-comet`). Also the metric
    you would use as an RL reward; report it next to chrF++ so reward hacking shows up as a gap."""
    from comet import download_model, load_from_checkpoint

    import torch

    model = load_from_checkpoint(download_model(model_name))

    def score(srcs, hyps, refs):
        rows = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        return float(model.predict(rows, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0).system_score)

    return score


def headline(task: str, m: dict):
    if task == "translation":
        vals = [v["chrf++"] for k, v in m.items() if "->" in k]
        return sum(vals) / len(vals) if vals else None
    if task == "ner":
        return 100 * m["f1"]
    if task in ("mmlu",):
        return 100 * m["accuracy"]
    return 100 * m["macro_f1"]


def summary_markdown(results: dict) -> str:
    cols = list(results)
    langs = sorted({l for t in results.values() for l in t}, key=lambda c: list(LANGS).index(c))
    lines = ["| lang | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    means = {c: [] for c in cols}
    for lang in langs:
        cells = []
        for c in cols:
            m = results[c].get(lang)
            if m is None:
                cells.append("")
            elif "skipped" in m:
                cells.append("n/a")
            else:
                h = headline(c.split("_")[0], m)
                cells.append("" if h is None else f"{h:.1f}")
                if h is not None:
                    means[c].append(h)
        lines.append(f"| {lang} | " + " | ".join(cells) + " |")
    lines.append("| **mean** | " + " | ".join(f"{sum(v) / len(v):.1f}" if v else "" for v in means.values()) + " |")
    lines += ["", "translation = mean chrF++ over both directions; topic/sentiment = macro-F1; ner = entity F1; mmlu = accuracy (chance 25). All x100."]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint dir or HF repo id (weights)")
    ap.add_argument("--tokenizer", default="BeardedMonster/SabiYarn-32k")
    ap.add_argument("--model-code", default="local", choices=["local", "hub"])
    ap.add_argument("--tasks", default="all", help=f"comma list of {ALL_TASKS} or 'all'")
    ap.add_argument("--langs", default="all", help="comma list of language codes or 'all' (English is always the pivot)")
    ap.add_argument("--limit", type=int, default=200, help="examples per (task, language); 0 = full test set")
    ap.add_argument("--style", default="tag", choices=["tag", "chat"], help="tag = pretraining prompts, chat = SFT chat template")
    ap.add_argument("--few-shot", type=int, default=5, help="MMLU few-shot examples")
    ap.add_argument("--topic-source", default="both", choices=["sib200", "masakhanews", "both"])
    ap.add_argument("--translation-source", default="auto", choices=["auto", "flores", "mafand"])
    ap.add_argument("--custom-translation", default=None, help="jsonl of {lang, eng, xx} for languages without a benchmark (efi, urh)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--africomet", default=None, help="e.g. masakhane/africomet-stl (verify the id); adds an AfriCOMET score")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)  # hf:// parquet fallback reads this
    task_list = ALL_TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    langs = parse_langs(args.langs)
    limit = args.limit or None
    name = args.name or f"{Path(args.model.rstrip('/')).name}_{args.style}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out_dir) / name
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    from eval_suite.model import ModelRunner

    runner = ModelRunner(args.model, args.tokenizer, model_code=args.model_code, batch_size=args.batch_size)
    africomet = make_africomet(args.africomet) if args.africomet else None
    gen = dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, repetition_penalty=args.repetition_penalty)

    jobs = []  # (result key, lang code, callable)
    for t in task_list:
        for code in langs:
            lang = LANGS[code]
            if t == "translation":
                jobs.append(("translation", code, lambda l=lang: tasks.run_translation(
                    runner, l, args.style, limit, args.translation_source, token, africomet=africomet,
                    custom_path=args.custom_translation, **gen)))
            elif t == "topic":
                for src in (["sib200", "masakhanews"] if args.topic_source == "both" else [args.topic_source]):
                    jobs.append((f"topic_{src}", code, lambda l=lang, s=src: tasks.run_classification(
                        runner, "topic", l, args.style, limit, s, token)))
            elif t == "sentiment":
                jobs.append(("sentiment", code, lambda l=lang: tasks.run_classification(
                    runner, "sentiment", l, args.style, limit, "", token)))
            elif t == "ner":
                jobs.append(("ner", code, lambda l=lang: tasks.run_ner(runner, l, args.style, limit, token, **gen)))
            elif t == "mmlu":
                jobs.append(("mmlu", code, lambda l=lang: tasks.run_mmlu(runner, l, args.style, limit, token, args.few_shot)))
            else:
                raise SystemExit(f"unknown task {t!r}; choose from {ALL_TASKS}")
    if "mmlu" in task_list:  # English MMLU as the reference point
        jobs.append(("mmlu", "eng", lambda: tasks.run_mmlu(runner, LANGS["eng"], args.style, limit, token, args.few_shot)))

    results: dict = {}
    for key, code, fn in jobs:
        t0 = time.time()
        try:
            metrics_, rows = fn()
        except DataUnavailable as exc:
            results.setdefault(key, {})[code] = {"skipped": str(exc)[:300]}
            print(f"[skip] {key:22s} {code}: {str(exc)[:110]}")
            continue
        results.setdefault(key, {})[code] = metrics_
        with open(out / "predictions" / f"{key}_{code}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = headline(key.split("_")[0], metrics_)
        print(f"[done] {key:22s} {code}: {'' if h is None else f'{h:.1f}'} ({time.time() - t0:.0f}s, n={metrics_.get('n', '')})")

    meta = {"model": args.model, "style": args.style, "limit": limit, "few_shot": args.few_shot,
            "translation_source": args.translation_source, "generation": gen, "tokenizer": args.tokenizer}
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2))
    md = summary_markdown(results)
    (out / "summary.md").write_text(md)
    print("\n" + md + f"\n\nwritten to {out}/")


if __name__ == "__main__":
    main()
