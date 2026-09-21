#!/usr/bin/env python3
"""OPTIONAL LLM-judge stage: score processed records, then filter by score.

Two steps, both free/local until you submit the built batch yourself:

  build  -> reads `processed/<kind>.jsonl`, samples a stratified fraction (or
            all) per language, and writes `batch_input/judge_<kind>.jsonl`
            (+ manifest). Submit/fetch it with the usual `submit --tasks
            judge_<kind> --confirm` / `fetch`.
  apply  -> reads the judge outputs, applies the score thresholds from the
            config's `judge:` section (override on the CLI), and writes
            `processed/<kind>.judged.jsonl` plus a per-language pass-rate
            report (`reports/judge_<kind>_summary.json`).

Be honest about what this measures: the judge is an LLM (default
gpt-4o-mini) scoring text in languages it is weak at. Its language-correctness
scores for Efik, Urhobo, Fon, Ewe and Fulah/Fulfulde are noisy and biased
towards leniency. Treat scores as a triage signal, prefer a stronger judge
(`--model gpt-4o`) for a sample, and confirm with native speakers.

Usage:
    python pipeline/judge.py build --kind sft --fraction 0.2
    python pipeline/judge.py apply --kind sft --min language_correctness=4 --min fluency=3
    python run.py judge build --kind dpo
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.corpus_config import CorpusConfig, load_config
from config.languages import get_language
from config.settings import CORPUS_KINDS, DataPaths
from generators.base import BatchRequestSpec, make_custom_id
from pipeline.build_corpus import write_split_batches
from pipeline.postprocess_corpus import load_manifest, load_outputs, parse_response_row
from sampling.sampler import derive_seed
from schemas.corpus import DPO_JUDGE_FORMAT_NAME, JUDGE_FORMAT_NAME, DPOJudgeScores, JudgeScores, strict_response_format

SCORE_KEYS = ["language_correctness", "fluency", "factuality", "instruction_following", "usefulness"]

JUDGE_SYSTEM_PROMPT = """\
You are a strict, careful reviewer of synthetic training data written in West African languages. You score ONE record on a 1-5 scale for each criterion.

Scale: 5 = a native speaker would accept it without any edit; 4 = minor blemishes; 3 = noticeable problems but usable; 2 = serious problems; 1 = unusable.

Criteria:
- language_correctness: it is in the intended language (not English, not another language), with correct orthography, diacritics/special letters and grammar.
- fluency: natural, idiomatic, native-sounding phrasing, not translationese or invented words.
- factuality: no factual errors, wrong arithmetic, invented statistics/quotes/named entities, or unsupported claims.
- instruction_following: does what was asked / matches the requested topic, genre and every explicit constraint.
- usefulness: would genuinely help train a good assistant / language model (clear, complete, not trivial, not repetitive).

Be critical; do not inflate scores. If you cannot reliably judge this language, score language_correctness and fluency no higher than 3 rather than guessing high. Put the single most important problem in `issues` (one short English sentence) or "none"."""


def _kind_prompt(kind: str, rec: dict, language: str) -> str:
    lang = get_language(language)
    head = f"Intended language: {lang.english_name} ({lang.endonym}), code '{language}'.\n"
    if kind == "pretrain":
        return (
            f"{head}Record type: pretraining document.\n"
            f"Requested topic: {rec['subtopic']} (domain: {rec['domain']}); genre: {rec['genre']}; register: {rec['register']}.\n"
            f"Title: {rec['title']}\nText:\n{rec['text']}"
        )
    base = f"{head}Record type: {'instruction-response' if kind == 'sft' else 'preference pair'} example; task type: {rec['task']}.\n"
    base += f"Instruction: {rec['instruction']}\nInput: {rec['input'] or '(none)'}\n"
    if kind == "sft":
        return base + f"Response: {rec['response']}"
    return (
        base + f"Chosen answer: {rec['chosen']}\nRejected answer (deliberately flawed: {rec['rejection_type']}): {rec['rejected']}\n"
        "Score the CHOSEN answer on the five criteria, and say whether it is clearly better than the rejected one."
    )


def build_judge_requests(kind: str, records: list[dict], cfg: CorpusConfig, fraction: float, seed: int) -> list[BatchRequestSpec]:
    """Stratified (per-language) deterministic sample -> one judge request per record."""
    fmt_name, model_cls = (DPO_JUDGE_FORMAT_NAME, DPOJudgeScores) if kind == "dpo" else (JUDGE_FORMAT_NAME, JudgeScores)
    response_format = strict_response_format(model_cls, fmt_name)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lang[r["language"]].append(r)

    task = f"judge_{kind}"
    specs: list[BatchRequestSpec] = []
    for language in sorted(by_lang):
        recs = sorted(by_lang[language], key=lambda r: r["id"])
        k = len(recs) if fraction >= 1 else min(len(recs), max(1, math.ceil(len(recs) * fraction)))
        rng = random.Random(derive_seed(seed, task, language))
        chosen = sorted(rng.sample(recs, k), key=lambda r: r["id"]) if k < len(recs) else recs
        for i, rec in enumerate(chosen):
            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(task, language, i),
                    task=task,
                    language=language,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_prompt=_kind_prompt(kind, rec, language),
                    response_format=response_format,
                    context={"kind": kind, "record_id": rec["id"]},
                    model=str(cfg.judge["model"]),
                    temperature=float(cfg.judge["temperature"]),
                    max_tokens=int(cfg.judge["max_tokens"]),
                )
            )
    return specs


def load_processed(kind: str, data_dir: Optional[Path], suffix: str = ".jsonl") -> list[dict]:
    path = DataPaths(data_dir).processed / f"{kind}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run postprocess for kind {kind!r} first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_build(cfg: CorpusConfig, data_dir: Optional[Path], fraction: Optional[float], seed: Optional[int]) -> list[Path]:
    records = load_processed(cfg.kind, data_dir)
    frac = float(cfg.judge["sample_fraction"] if fraction is None else fraction)
    specs = build_judge_requests(cfg.kind, records, cfg, frac, cfg.seed if seed is None else seed)
    return write_split_batches(f"judge_{cfg.kind}", specs, DataPaths(data_dir).batch_input)


def passes(scores: dict[str, Any], thresholds: dict[str, float], kind: str) -> tuple[bool, list[str]]:
    failed = [k for k, t in thresholds.items() if k in scores and scores[k] < t]
    if kind == "dpo" and scores.get("chosen_better_than_rejected") is False:
        failed.append("chosen_better_than_rejected")
    return (not failed), failed


def run_apply(cfg: CorpusConfig, data_dir: Optional[Path], thresholds: dict[str, float], drop_unjudged: Optional[bool] = None) -> tuple[list[dict], dict]:
    kind = cfg.kind
    paths = DataPaths(data_dir)
    records = load_processed(kind, data_dir)
    task = f"judge_{kind}"
    manifest = load_manifest(task, paths.batch_input)
    outputs = load_outputs(task, paths.batch_output)
    drop_unjudged = bool(cfg.judge["drop_unjudged"]) if drop_unjudged is None else drop_unjudged
    model_cls = DPOJudgeScores if kind == "dpo" else JudgeScores

    verdict: dict[str, dict[str, Any]] = {}  # record_id -> {"scores":..., "ok":..., "failed":[...]}
    unparsable: dict[str, int] = defaultdict(int)
    for cid, entry in manifest.items():
        row = outputs.get(cid)
        value, _err = parse_response_row(row) if row else (None, "missing_output")
        if value is not None:
            try:
                value = model_cls.model_validate(value).model_dump()
            except Exception:  # noqa: BLE001
                value = None  # schema-invalid judge output counts as unparsable
        if value is None:
            unparsable[entry["language"]] += 1
            continue
        ok, failed = passes(value, thresholds, kind)
        verdict[entry["context"]["record_id"]] = {"scores": value, "ok": ok, "failed": failed, "language": entry["language"]}

    kept: list[dict] = []
    per_lang: dict[str, dict[str, Any]] = defaultdict(lambda: {"records": 0, "judged": 0, "passed": 0, "failed": 0, "unjudged_dropped": 0, "fail_reasons": defaultdict(int), "score_sums": defaultdict(float)})
    for rec in records:
        st = per_lang[rec["language"]]
        st["records"] += 1
        v = verdict.get(rec["id"])
        if v is None:
            if drop_unjudged:
                st["unjudged_dropped"] += 1
            else:
                kept.append(rec)
            continue
        st["judged"] += 1
        for k in SCORE_KEYS:
            st["score_sums"][k] += v["scores"][k]
        if v["ok"]:
            st["passed"] += 1
            kept.append({**rec, "judge_scores": {k: v["scores"][k] for k in SCORE_KEYS}})
        else:
            st["failed"] += 1
            for f in v["failed"]:
                st["fail_reasons"][f] += 1

    report = {"kind": kind, "thresholds": thresholds, "drop_unjudged": drop_unjudged, "languages": {}}
    for lang, st in sorted(per_lang.items()):
        judged = st["judged"]
        report["languages"][lang] = {
            "records": st["records"], "judged": judged, "passed": st["passed"], "failed": st["failed"],
            "pass_rate": round(st["passed"] / judged, 4) if judged else None,
            "unjudged_dropped": st["unjudged_dropped"], "unparsable_judge_outputs": unparsable.get(lang, 0),
            "fail_reasons": dict(st["fail_reasons"]),
            "mean_scores": {k: round(st["score_sums"][k] / judged, 3) for k in SCORE_KEYS} if judged else {},
        }
    out = paths.processed / f"{kind}.judged.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (paths.reports / f"judge_{kind}_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return kept, report


def _parse_min(pairs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in pairs:
        k, _, v = p.partition("=")
        if k not in SCORE_KEYS or not v:
            raise argparse.ArgumentTypeError(f"--min expects <criterion>=<score> with criterion in {SCORE_KEYS}, got {p!r}")
        out[k] = float(v)
    return out


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--kind", required=True, choices=CORPUS_KINDS)
        p.add_argument("--config", default=None)
        p.add_argument("--data-dir", default=None)
        p.add_argument("--model", default=None, help="judge model override (e.g. gpt-4o for a more reliable judge)")
        if name == "build":
            p.add_argument("--fraction", type=float, default=None, help="share of records to judge per language (default: config judge.sample_fraction; 1.0 = all)")
            p.add_argument("--seed", type=int, default=None)
        else:
            p.add_argument("--min", action="append", default=[], metavar="CRITERION=SCORE", help="override a threshold (repeatable)")
            p.add_argument("--drop-unjudged", action="store_true", help="also drop records that were not in the judged sample")
    args = parser.parse_args(argv)

    cfg = load_config(args.config or args.kind)
    if args.model:
        cfg.judge["model"] = args.model
    data_dir = Path(args.data_dir) if args.data_dir else None

    if args.cmd == "build":
        run_build(cfg, data_dir, args.fraction, args.seed)
        print(f"Submit with: python run.py submit --tasks judge_{args.kind} --confirm   (costs money), then `fetch`, then `judge apply`.")
    else:
        thresholds = {**{k: float(v) for k, v in cfg.judge["min_scores"].items()}, **_parse_min(args.min)}
        kept, report = run_apply(cfg, data_dir, thresholds, True if args.drop_unjudged else None)
        print(f"kept {len(kept)} records -> {DataPaths(data_dir).processed / (args.kind + '.judged.jsonl')}")
        for lang, r in report["languages"].items():
            pr = f"{100 * r['pass_rate']:.0f}%" if r["pass_rate"] is not None else "-"
            print(f"  {lang}: judged={r['judged']} passed={r['passed']} pass_rate={pr}")


if __name__ == "__main__":
    main()
