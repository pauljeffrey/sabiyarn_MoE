#!/usr/bin/env python3
"""Offline volume / token / cost estimator for the corpus kinds.

Makes NO API calls. It builds a small sample of the real requests for each
language (so prompt sizes are measured, not guessed), estimates output size
from the sampled length buckets, and applies the pricing constants in
`config/settings.py` (marked "verify current pricing").

Token accounting notes:
  * Input tokens: prompt + schema characters / CHARS_PER_TOKEN_ENGLISH (the
    meta-prompts are English).
  * Output tokens: words x tokens-per-word for the *target language*.
    OpenAI tokenisers were trained mostly on English, so text in Yoruba,
    Efik, Fon, ... splits into far more tokens per word; the multipliers in
    settings.py are deliberately conservative. Real cost is likely at or below
    the estimate, rarely above.

Usage:
    python pipeline/estimate.py                       # all three kinds, preset counts
    python pipeline/estimate.py --kind sft --per-language 200
    python run.py estimate --kind dpo
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.corpus_config import CorpusConfig, load_config
from config.settings import (
    AFRICAN_LANG_TOKENS_PER_WORD_DEFAULT,
    BATCH_PRICING_USD_PER_M_TOKENS,
    CHARS_PER_TOKEN_ENGLISH,
    CORPUS_KINDS,
    ENGLISH_TOKENS_PER_WORD,
    FALLBACK_BATCH_PRICING_USD_PER_M_TOKENS,
    MAX_BATCH_FILE_BYTES,
    TOKENS_PER_WORD_BY_LANGUAGE,
)
from pipeline.build_batch import MAX_REQUESTS_PER_BATCH_FILE
from sampling.taxonomy import RESPONSE_WORD_RANGES

GENERATOR_MODULES = {"pretrain": "generators.pretrain", "sft": "generators.sft_tasks", "dpo": "generators.dpo"}

# Rough word counts of the non-response fields of an SFT/DPO example.
_INSTRUCTION_WORDS = 25
_INPUT_WORDS = {"required": 90, "optional": 30, "empty": 0}
_JSON_SCAFFOLD_TOKENS = 35  # keys, quotes, braces, confidence field


def tokens_per_word(language: str) -> float:
    return TOKENS_PER_WORD_BY_LANGUAGE.get(language, AFRICAN_LANG_TOKENS_PER_WORD_DEFAULT)


def pricing_for(model: str) -> tuple[tuple[float, float], bool]:
    """((input, output) USD per 1M tokens, is_fallback)."""
    if model in BATCH_PRICING_USD_PER_M_TOKENS:
        return BATCH_PRICING_USD_PER_M_TOKENS[model], False
    return FALLBACK_BATCH_PRICING_USD_PER_M_TOKENS, True


def _spec_input_tokens(spec) -> float:
    chars = len(spec.system_prompt) + len(spec.user_prompt) + len(json.dumps(spec.response_format, ensure_ascii=False))
    return chars / CHARS_PER_TOKEN_ENGLISH + 12  # +12: chat-format overhead


def _spec_output_tokens(kind: str, spec) -> float:
    ctx = spec.context
    tpw = tokens_per_word(spec.language)
    if kind == "pretrain":
        lo, hi = ctx["length_words"]
        return ((lo + hi) / 2 + 12) * tpw + _JSON_SCAFFOLD_TOKENS
    attrs = ctx["attributes"]
    lo, hi = RESPONSE_WORD_RANGES[attrs["response_length"]]
    resp_words = (lo + hi) / 2
    english = set(ctx.get("english_fields", []))

    def toks(words: float, field: str) -> float:
        return words * (ENGLISH_TOKENS_PER_WORD if field in english else tpw)

    base = toks(_INSTRUCTION_WORDS, "instruction") + toks(_INPUT_WORDS[ctx["input_mode"]], "input") + _JSON_SCAFFOLD_TOKENS
    if kind == "sft":
        return base + toks(resp_words, "response")
    # dpo: chosen + rejected of comparable length
    return base + 2 * toks(resp_words, "response")


def estimate_kind(cfg: CorpusConfig, *, sample_per_language: int = 40, languages: Optional[list[str]] = None) -> dict[str, Any]:
    kind = cfg.kind
    gen = importlib.import_module(GENERATOR_MODULES[kind])
    (price_in, price_out), fallback = pricing_for(cfg.model)

    rows = []
    total_in = total_out = total_bytes = 0.0
    total_n = 0
    truncation_risk = False
    langs = languages or cfg.languages
    for lang in langs:
        n = cfg.samples_per_language[lang]
        if n == 0:
            rows.append({"language": lang, "requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
            continue
        m = min(n, sample_per_language)
        sample = list(gen.iter_requests(cfg.with_overrides(per_language=m, languages=[lang]), [lang]))
        avg_in = sum(_spec_input_tokens(s) for s in sample) / len(sample)
        raw_out = [_spec_output_tokens(kind, s) for s in sample]
        avg_out_uncapped = sum(raw_out) / len(raw_out)
        avg_out = sum(min(x, cfg.max_tokens) for x in raw_out) / len(raw_out)
        if avg_out_uncapped > 0.85 * cfg.max_tokens or max(raw_out) > cfg.max_tokens:
            truncation_risk = True
        avg_bytes = sum(len(json.dumps(s.to_batch_line(), ensure_ascii=False).encode("utf-8")) for s in sample) / len(sample)
        in_tok, out_tok = avg_in * n, avg_out * n
        usd = in_tok / 1e6 * price_in + out_tok / 1e6 * price_out
        rows.append({"language": lang, "requests": n, "input_tokens": round(in_tok), "output_tokens": round(out_tok), "usd": round(usd, 2)})
        total_in += in_tok
        total_out += out_tok
        total_bytes += avg_bytes * n
        total_n += n

    total_usd = total_in / 1e6 * price_in + total_out / 1e6 * price_out
    files_by_count = math.ceil(total_n / MAX_REQUESTS_PER_BATCH_FILE) if total_n else 0
    files_by_size = math.ceil(total_bytes / MAX_BATCH_FILE_BYTES) if total_n else 0
    return {
        "kind": kind,
        "model": cfg.model,
        "pricing_usd_per_m_tokens": {"input": price_in, "output": price_out, "fallback_used": fallback},
        "rows": rows,
        "totals": {
            "requests": total_n,
            "input_tokens": round(total_in),
            "output_tokens": round(total_out),
            "usd": round(total_usd, 2),
            "approx_input_file_mb": round(total_bytes / 1e6, 1),
            "batch_files": max(files_by_count, files_by_size),
        },
        "max_tokens": cfg.max_tokens,
        "truncation_risk": truncation_risk,
    }


def format_estimate(est: dict[str, Any]) -> str:
    p = est["pricing_usd_per_m_tokens"]
    lines = [
        f"== {est['kind']} | model={est['model']} | batch price ${p['input']}/M in, ${p['output']}/M out"
        + ("  (model not in price table: FALLBACK price used)" if p["fallback_used"] else ""),
        f"{'lang':<6}{'requests':>10}{'input_tok':>14}{'output_tok':>14}{'est_usd':>10}",
    ]
    for r in est["rows"]:
        lines.append(f"{r['language']:<6}{r['requests']:>10,}{r['input_tokens']:>14,}{r['output_tokens']:>14,}{r['usd']:>10.2f}")
    t = est["totals"]
    lines.append(f"{'TOTAL':<6}{t['requests']:>10,}{t['input_tokens']:>14,}{t['output_tokens']:>14,}{t['usd']:>10.2f}")
    lines.append(f"input file size ~{t['approx_input_file_mb']} MB -> {t['batch_files']} batch file(s) (limits: {MAX_REQUESTS_PER_BATCH_FILE:,} requests / {MAX_BATCH_FILE_BYTES // 10**6} MB each)")
    if est["truncation_risk"]:
        lines.append(f"WARNING: some outputs may exceed max_tokens={est['max_tokens']} and be truncated; raise max_tokens in the config.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", default="all", help=f"one of {CORPUS_KINDS} or 'all' (default)")
    parser.add_argument("--config", default=None, help="config yaml (only with a single --kind); default configs/<kind>.yaml")
    parser.add_argument("--per-language", type=int, default=None, help="override samples per language")
    parser.add_argument("--languages", default=None, help="comma-separated subset of language codes")
    parser.add_argument("--model", default=None, help="override the model (changes the price used)")
    parser.add_argument("--sample", type=int, default=40, help="requests per language to build for measuring prompt size")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)

    kinds = CORPUS_KINDS if args.kind == "all" else [args.kind]
    if any(k not in CORPUS_KINDS for k in kinds):
        parser.error(f"--kind must be one of {CORPUS_KINDS} or 'all'")
    if args.config and len(kinds) != 1:
        parser.error("--config requires a single --kind")
    langs = [c.strip() for c in args.languages.split(",")] if args.languages else None

    estimates = []
    for kind in kinds:
        cfg = load_config(args.config or kind)
        if args.model:
            cfg.model = args.model
        cfg = cfg.with_overrides(per_language=args.per_language, languages=langs)
        estimates.append(estimate_kind(cfg, sample_per_language=args.sample))

    if args.json:
        print(json.dumps(estimates, indent=2))
        return
    for est in estimates:
        print(format_estimate(est))
        print()
    if len(estimates) > 1:
        print(f"GRAND TOTAL: {sum(e['totals']['requests'] for e in estimates):,} requests, ~${sum(e['totals']['usd'] for e in estimates):,.2f} (Batch API, gpt-4o-mini prices in config/settings.py -- verify current pricing)")
    print(
        "Notes: African-language text tokenises poorly under OpenAI tokenisers (conservative multipliers in config/settings.py). "
        "Batch API queue limits (enqueued tokens per model, by usage tier) may force you to submit files sequentially. "
        "No API calls were made."
    )


if __name__ == "__main__":
    main()
