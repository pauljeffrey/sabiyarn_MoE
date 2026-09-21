#!/usr/bin/env python3
"""Synthesize structurally-valid (but linguistically fake) batch outputs
for every request in data/batch_input/*.jsonl, without calling the OpenAI
API. This exists purely to pipeline-test postprocess.py -- parsing,
rendering, validation, dedup -- for free, before spending real money on
submit_batch.py. Output is NOT meant to be used as actual training data.

The original six tasks get generic schema-driven fake values. The corpus
kinds (pretrain / sft / dpo) and the judge stage get task-aware fakes:
pseudo-words built from per-language syllables (so they pass the local
quality filters and exercise every downstream code path), sensible JSON for
extraction tasks, flaw-shaped rejected answers for DPO, and judge scores.

Usage:
    python pipeline/build_batch.py            # (with a small DATA_GEN_PER_CELL)
    python scripts/mock_generate.py
    python pipeline/postprocess.py

    python run.py build --kind sft --per-language 5
    python run.py mock
    python run.py postprocess --kind sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BATCH_INPUT_DIR, BATCH_OUTPUT_DIR, DataPaths
from schemas.corpus import DPO_FORMAT_NAME, DPO_JUDGE_FORMAT_NAME, JUDGE_FORMAT_NAME, PRETRAIN_FORMAT_NAME, SFT_FORMAT_NAME


def _fake_value(schema: dict, key_hint: str = ""):
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        for sub in schema["anyOf"]:
            if sub.get("type") != "null":
                return _fake_value(sub, key_hint)
        return None
    if t == "string":
        return f"mock {key_hint or 'text'} sample"
    if t == "integer":
        return 3
    if t == "number":
        return 3.5
    if t == "boolean":
        return False
    if t == "array":
        item_schema = schema.get("items", {"type": "string"})
        n = schema.get("minItems", 2)
        return [_fake_value(item_schema, key_hint) for _ in range(n)]
    if t == "object":
        return {k: _fake_value(v, k) for k, v in schema.get("properties", {}).items()}
    return None


def _fake_content_for(response_format: dict) -> str:
    schema = response_format["json_schema"]["schema"]
    value = _fake_value(schema)
    return json.dumps(value, ensure_ascii=False)


# Task-specific overrides so mock data is at least *structurally* sensible
# (e.g. a RAG relevant_chunk_id of 0, not a random hallucinated int) instead
# of exercising failure paths every time.
def _patch(task: str, spec: dict) -> dict:
    if task == "rag":
        for turn in spec.get("turns", []):
            if turn.get("answerable"):
                turn["relevant_chunk_id"] = 0
            else:
                turn["relevant_chunk_id"] = None
    if task == "edge_action":
        if spec.get("needs_tool_call"):
            spec["tool_arguments_json"] = "{}"
            spec["tool_result_json"] = '{"status": "success"}'
    if task == "math_stats":
        if spec.get("uses_calculator"):
            spec["calculator_expression"] = "2+2"
            spec["calculator_result"] = "4"
    return spec


# ---------------------------------------------------------------------------
# Corpus-kind fakes
# ---------------------------------------------------------------------------

_CONSONANTS = list("bdfghklmnprstwyjz")
_VOWELS = list("aeiou")
# Language-distinctive letters so mock text also passes the soft diacritic sanity check.
_EXTRA_LETTERS = {
    "yor": ["ẹ", "ọ", "ṣ"], "hau": ["ɓ", "ɗ", "ƙ"], "ibo": ["ị", "ọ", "ụ"], "ewe": ["ɖ", "ɣ", "ʋ", "ŋ", "ɔ", "ɛ"],
    "twi": ["ɔ", "ɛ"], "aka": ["ɔ", "ɛ"], "fon": ["ɖ", "ɛ", "ɔ"],
}
# English content words (no stopwords) for the English side of translation-style tasks.
_ENGLISH_WORDS = (
    "market farmer village maize water river school teacher clinic nurse mother children morning harvest season price basket "
    "goat chicken road bridge festival music story drum cloth kitchen soup yam cassava rice fish trader bank phone light rain "
    "dry garden health family neighbour elder chief town city bus driver shop money saving lesson book pencil song dance"
).split()
_ENGLISH_STOP = "the and of are with from that this".split()


def _rng(custom_id: str, salt: str = "") -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(f"{custom_id}|{salt}".encode()).digest()[:8], "big"))


def _pseudo_word(rng: random.Random, language: str) -> str:
    vowels = _VOWELS + _EXTRA_LETTERS.get(language, [])
    n_syl = rng.choice((1, 2, 2, 3))
    return "".join(rng.choice(_CONSONANTS) + rng.choice(vowels) for _ in range(n_syl))


def _sentences(rng: random.Random, language: str, n_words: int, *, english: bool = False, paragraphs: bool = False) -> str:
    words_left = max(n_words, 1)
    sents: list[str] = []
    while words_left > 0:
        k = min(words_left, rng.randint(7, 13))
        ws = [rng.choice(_ENGLISH_WORDS) if english else _pseudo_word(rng, language) for _ in range(k)]
        if english and k > 4:
            ws[1] = rng.choice(_ENGLISH_STOP)
        ws[0] = ws[0].capitalize()
        sents.append(" ".join(ws) + ".")
        words_left -= k
    if not paragraphs:
        return " ".join(sents)
    return "\n\n".join(" ".join(sents[i : i + 4]) for i in range(0, len(sents), 4))


_RESPONSE_WORDS = {"brief": 14, "short": 32, "medium": 90, "long": 170}


def _mock_pretrain(cid: str, language: str, ctx: dict) -> dict:
    rng = _rng(cid)
    lo, hi = ctx.get("length_words", [150, 250])
    return {
        "title": " ".join(w.capitalize() if i == 0 else w for i, w in enumerate(_pseudo_word(rng, language) for _ in range(4))),
        "text": _sentences(rng, language, (lo + hi) // 2, paragraphs=True),
        "language_self_check": True,
    }


def _extraction_json(rng: random.Random, task: str, source: str, *, flawed: bool = False) -> str:
    """Valid JSON for extraction tasks; `flawed` (a DPO rejected answer) drops the values."""
    words = [w.strip(".") for w in source.split() if len(w) > 3] or ["ka"]
    if flawed:
        words = ["ka"]
    if task == "ner_extraction":
        return json.dumps({"persons": words[:1], "locations": words[1:2], "organizations": [], "dates": []}, ensure_ascii=False)
    return json.dumps({"name": words[0], "place": words[-1], "date": None}, ensure_ascii=False)


def _mock_sft_like(cid: str, language: str, ctx: dict, *, dpo: bool) -> dict:
    rng = _rng(cid)
    attrs = ctx.get("attributes", {})
    task = attrs.get("task", "open_qa")
    english = set(ctx.get("english_fields", []))
    length = _RESPONSE_WORDS.get(attrs.get("response_length", "short"), 32)

    instruction = _sentences(rng, language, rng.randint(9, 16)).rstrip(".") + "?"
    if "input" in english:
        inp = _sentences(rng, language, 14, english=True)
    elif ctx.get("input_mode") == "required":
        inp = _sentences(rng, language, 24)
    else:
        inp = ""

    def answer(words: int, salt: str, *, force_english: bool = False) -> str:
        r = _rng(cid, salt)
        if task in ("ner_extraction", "info_extraction_json"):
            return _extraction_json(r, task, inp or instruction, flawed=(salt == "rej"))
        return _sentences(r, language, words, english=("response" in english) or force_english)

    if not dpo:
        return {"instruction": instruction, "input": inp, "response": answer(length, "resp"), "confidence": "high"}

    rtype = ctx.get("rejection_type", "off_topic")
    chosen = answer(length, "chosen")
    if rtype == "incomplete":
        rejected = answer(max(length // 2, 12), "rej")
    elif rtype == "rambling_verbose":
        rejected = answer(length * 2 + 10, "rej")
    elif rtype == "wrong_language":
        rejected = _sentences(_rng(cid, "rej"), language, length, english=True)
    else:
        rejected = answer(length, "rej")
    return {"instruction": instruction, "input": inp, "chosen": chosen, "rejected": rejected,
            "rejection_type": rtype, "chosen_confidence": "high"}


def _mock_judge(cid: str, dpo: bool) -> dict:
    rng = _rng(cid, "judge")
    scores = {k: rng.choice((4, 4, 5, 5, 5)) for k in ("language_correctness", "fluency", "factuality", "instruction_following", "usefulness")}
    if rng.random() < 0.15:  # a deterministic slice of "bad" records so `apply` has something to filter
        scores["fluency"] = 2
        scores["language_correctness"] = 2
    out: dict[str, Any] = {**scores, "issues": "none"}
    if dpo:
        out["chosen_better_than_rejected"] = True
    return out


def _corpus_fake(fmt_name: str, cid: str, language: str, ctx: dict) -> Optional[dict]:
    if fmt_name == PRETRAIN_FORMAT_NAME:
        return _mock_pretrain(cid, language, ctx)
    if fmt_name == SFT_FORMAT_NAME:
        return _mock_sft_like(cid, language, ctx, dpo=False)
    if fmt_name == DPO_FORMAT_NAME:
        return _mock_sft_like(cid, language, ctx, dpo=True)
    if fmt_name in (JUDGE_FORMAT_NAME, DPO_JUDGE_FORMAT_NAME):
        return _mock_judge(cid, dpo=fmt_name == DPO_JUDGE_FORMAT_NAME)
    return None


def _load_manifest(batch_path: Path) -> dict[str, dict]:
    manifest_path = batch_path.with_name(f"{batch_path.stem}.manifest.jsonl")
    out: dict[str, dict] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    out[e["custom_id"]] = e
    return out


def run(input_dir: Path, output_dir: Path, only: Optional[list[str]] = None) -> list[Path]:
    """Write a mock `<stem>.output.jsonl` for every batch input file. `only`
    restricts to file stems whose base name (before `__part`) is listed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for batch_path in sorted(input_dir.glob("*.jsonl")):
        if batch_path.name.endswith(".manifest.jsonl"):
            continue
        task = batch_path.stem.split("__part")[0]
        if only and task not in only:
            continue
        manifest = _load_manifest(batch_path)
        out_path = output_dir / f"{batch_path.stem}.output.jsonl"
        n = 0
        with batch_path.open(encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                response_format = req["body"]["response_format"]
                fmt_name = response_format["json_schema"]["name"]
                entry = manifest.get(req["custom_id"], {})
                spec = _corpus_fake(fmt_name, req["custom_id"], entry.get("language", ""), entry.get("context", {}))
                if spec is None:
                    spec = _patch(task, json.loads(_fake_content_for(response_format)))
                content_str = json.dumps(spec, ensure_ascii=False)

                out_line = {
                    "id": f"mock-{req['custom_id']}",
                    "custom_id": req["custom_id"],
                    "response": {
                        "status_code": 200,
                        "request_id": "mock",
                        "body": {"choices": [{"message": {"content": content_str}}]},
                    },
                    "error": None,
                }
                f_out.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                n += 1
        print(f"[{task}] wrote {n} mock responses -> {out_path.name}")
        written.append(out_path)
    return written


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="data root (default: DATA_GEN_OUTPUT_DIR or ./data)")
    parser.add_argument("--only", default=None, help="comma-separated task/kind names to mock (default: every file in batch_input)")
    args = parser.parse_args(argv)
    if args.data_dir:
        paths = DataPaths(args.data_dir)
        input_dir, output_dir = paths.batch_input, paths.batch_output
    else:
        input_dir, output_dir = BATCH_INPUT_DIR, BATCH_OUTPUT_DIR
    only = [t.strip() for t in args.only.split(",") if t.strip()] if args.only else None
    run(input_dir, output_dir, only)


if __name__ == "__main__":
    main()
