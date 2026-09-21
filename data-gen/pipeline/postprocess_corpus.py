#!/usr/bin/env python3
"""Parse, validate, filter, dedup and write the pretrain / sft / dpo datasets.

Pipeline per kind:
  1. Join Batch API outputs (`batch_output/<kind>[__partN].output.jsonl`) to the
     manifests written at build time (`batch_input/<kind>[__partN].manifest.jsonl`).
     Every manifest entry is accounted for: kept, or dropped with a reason
     (including `missing_output` for requests that never came back).
  2. Parse + re-validate the JSON against the pydantic schema (the API's strict
     mode guarantees shape, never content).
  3. Local quality filters (`quality/corpus_filters.py`): length bounds,
     repetition, English leakage, meta text/placeholders/refusals, script sanity,
     confidence, and DPO pair checks.
  4. Dedup: exact + near-duplicate (`quality/dedup.py`) within each language and
     then across languages of the same kind.
  5. Write `processed/<kind>.jsonl` and `reports/<kind>_summary.json`.

Usage:
    python pipeline/postprocess_corpus.py --kind sft
    python run.py postprocess --kind all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, ValidationError

from config.corpus_config import CorpusConfig, load_config
from config.settings import CORPUS_KINDS, DataPaths
from quality.corpus_filters import tokens, validate_dpo, validate_pretrain, validate_sft
from quality.dedup import find_near_duplicates
from rendering.render import render_messages
from schemas.corpus import DPOPair, PretrainDoc, SFTExample
from schemas.messages import Message
from stats.corpus_report import CorpusStats

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful multilingual assistant for speakers of West African languages. "
    "Reply in the language the user writes in, clearly and accurately."
)

SCHEMAS: dict[str, type[BaseModel]] = {"pretrain": PretrainDoc, "sft": SFTExample, "dpo": DPOPair}
VALIDATORS = {"pretrain": validate_pretrain, "sft": validate_sft, "dpo": validate_dpo}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def kind_files(directory: Path, kind: str, suffix: str) -> list[Path]:
    """`<kind><suffix>` plus split `<kind>__part*<suffix>` files, in order.

    Exact-prefix globbing (kind + '.' / kind + '__part') so `sft` never picks up
    `judge_sft` files.
    """
    return sorted(directory.glob(f"{kind}{suffix}")) + sorted(directory.glob(f"{kind}__part*{suffix}"))


def load_manifest(kind: str, batch_dir: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in kind_files(batch_dir, kind, ".manifest.jsonl"):
        for entry in _jsonl(path):
            by_id[entry["custom_id"]] = entry
    return by_id


def load_outputs(kind: str, out_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in kind_files(out_dir, kind, ".output.jsonl"):
        for row in _jsonl(path):
            rows[row["custom_id"]] = row
    return rows


def parse_response_row(row: dict) -> tuple[Optional[dict], Optional[str]]:
    """Extract the JSON object from one Batch output line, or (None, reason)."""
    if row.get("error"):
        return None, "api_error"
    resp = row.get("response") or {}
    if resp.get("status_code") != 200:
        return None, "api_error"
    try:
        message = resp["body"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None, "malformed_response"
    if message.get("refusal"):
        return None, "model_refusal"
    try:
        value = json.loads(message["content"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None, "invalid_json"
    if not isinstance(value, dict):
        return None, "invalid_json"
    return value, None


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def _user_content(instruction: str, inp: str) -> str:
    return f"{instruction.strip()}\n\n{inp.strip()}" if inp.strip() else instruction.strip()


def build_record(kind: str, custom_id: str, language: str, value: dict, attrs: dict[str, str], cfg: CorpusConfig) -> tuple[dict, str]:
    """Return (record, dedup_text)."""
    if kind == "pretrain":
        text = value["text"].strip()
        rec = {
            "id": custom_id, "language": language, "text": text, "title": value["title"].strip(),
            "domain": attrs["domain"], "subtopic": attrs["subtopic"], "genre": attrs["genre"],
            "register": attrs["register"], "audience": attrs["audience"], "locale": attrs["locale"],
            "length_bucket": attrs["length_bucket"], "perspective": attrs["perspective"], "era": attrs["era"],
            "difficulty": attrs["difficulty"], "n_words": len(tokens(text)),
        }
        return rec, text

    system = cfg.system_prompt or DEFAULT_SYSTEM_PROMPT
    instruction, inp = value["instruction"].strip(), value["input"].strip()
    prompt = [
        Message(role="system", content=system).to_template_dict(),
        Message(role="user", content=_user_content(instruction, inp)).to_template_dict(),
    ]
    common = {
        "id": custom_id, "language": language, "task": attrs["task"], "domain": attrs["domain"], "subtopic": attrs["subtopic"],
        "instruction": instruction, "input": inp,
    }
    extra = {k: attrs[k] for k in ("register", "instruction_style", "response_length", "difficulty", "locale")}
    if kind == "sft":
        response = value["response"].strip()
        messages = prompt + [Message(role="assistant", content=response).to_template_dict()]
        rec = {**common, "response": response, "messages": messages, "text": render_messages(messages),
               "confidence": value["confidence"], **extra}
        return rec, f"{instruction} {inp}"
    rec = {**common, "prompt_messages": prompt, "chosen": value["chosen"].strip(), "rejected": value["rejected"].strip(),
           "rejection_type": value["rejection_type"], "chosen_confidence": value["chosen_confidence"], **extra}
    return rec, f"{instruction} {inp}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_kind(cfg: CorpusConfig, data_dir: Optional[Path] = None) -> tuple[list[dict], CorpusStats]:
    kind = cfg.kind
    paths = DataPaths(data_dir)
    manifest = load_manifest(kind, paths.batch_input)
    outputs = load_outputs(kind, paths.batch_output)
    stats = CorpusStats(kind)
    validator = VALIDATORS[kind]
    schema = SCHEMAS[kind]
    q = cfg.quality

    candidates: list[dict[str, Any]] = []
    for custom_id in sorted(manifest):
        entry = manifest[custom_id]
        language, ctx = entry["language"], entry["context"]
        stats.record_generated(language)

        row = outputs.get(custom_id)
        if row is None:
            stats.record_drop(language, ["missing_output"], custom_id)
            continue
        value, err = parse_response_row(row)
        if err:
            stats.record_drop(language, [err], custom_id)
            continue
        try:
            value = schema.model_validate(value).model_dump()
        except ValidationError:
            stats.record_drop(language, ["schema_invalid"], custom_id)
            continue

        reasons, warnings = validator(value, ctx, language, q)
        stats.record_warnings(language, warnings)
        if reasons:
            stats.record_drop(language, reasons, custom_id)
            continue
        try:
            rec, dedup_text = build_record(kind, custom_id, language, value, ctx["attributes"], cfg)
        except Exception as e:  # noqa: BLE001 - one bad record must not abort a 70k-record run
            stats.record_drop(language, [f"render_error:{type(e).__name__}"], custom_id)
            continue
        candidates.append({"rec": rec, "text": dedup_text, "language": language, "index": ctx.get("index", 0), "attrs": ctx["attributes"]})

    kept = _dedup(candidates, cfg, stats)
    for c in kept:
        stats.record_kept(c["language"], c["attrs"])
    records = [c["rec"] for c in kept]

    out_path = paths.processed / f"{kind}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.write(paths.reports / f"{kind}_summary.json")
    return records, stats


def _dedup(candidates: list[dict[str, Any]], cfg: CorpusConfig, stats: CorpusStats) -> list[dict[str, Any]]:
    """Within-language pass, then cross-language pass, earliest-index-wins.

    Sorting by (index, language) makes ties across languages resolve fairly
    instead of always favouring the alphabetically first language.
    """
    d = cfg.dedup
    ordered = sorted(candidates, key=lambda c: (c["index"], c["language"], c["rec"]["id"]))
    passes = []
    if d["within_language"]:
        passes.append(("language", lambda c: c["language"], "duplicate"))
    if d["cross_language"]:
        passes.append(("all-languages", lambda c: "all", "cross_language_duplicate"))

    current = ordered
    for label, key_fn, reason_prefix in passes:
        drop, reasons = find_near_duplicates(
            current, cell_key_fn=key_fn, text_fn=lambda c: c["text"],
            near_dup_threshold=float(d["near_dup_threshold"]), shingle_size=int(d["shingle_size"]), label=label,
        )
        nxt = []
        for i, c in enumerate(current):
            if i in drop:
                kind_of = "exact" if reasons[i].startswith("exact") else "near"
                stats.record_drop(c["language"], [f"{reason_prefix}_{kind_of}"], c["rec"]["id"])
            else:
                nxt.append(c)
        current = nxt
    return current


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, help=f"one of {CORPUS_KINDS} or 'all'")
    parser.add_argument("--config", default=None, help="config yaml (single --kind only); default configs/<kind>.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    kinds = CORPUS_KINDS if args.kind == "all" else [args.kind]
    if any(k not in CORPUS_KINDS for k in kinds):
        parser.error(f"--kind must be one of {CORPUS_KINDS} or 'all'")
    if args.config and len(kinds) != 1:
        parser.error("--config requires a single --kind")
    data_dir = Path(args.data_dir) if args.data_dir else None

    for kind in kinds:
        cfg = load_config(args.config or kind)
        records, stats = process_kind(cfg, data_dir)
        paths = DataPaths(data_dir)
        print(f"\n[{kind}] kept {len(records)} -> {paths.processed / (kind + '.jsonl')}")
        print(stats.as_table())
        print(f"report: {paths.reports / (kind + '_summary.json')}")


if __name__ == "__main__":
    main()
