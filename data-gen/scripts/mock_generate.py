#!/usr/bin/env python3
"""Synthesize structurally-valid (but linguistically fake) batch outputs
for every request in data/batch_input/*.jsonl, without calling the OpenAI
API. This exists purely to pipeline-test postprocess.py -- parsing,
rendering, validation, dedup -- for free, before spending real money on
submit_batch.py. Output is NOT meant to be used as actual training data.

Usage:
    python pipeline/build_batch.py            # (with a small DATA_GEN_PER_CELL)
    python scripts/mock_generate.py
    python pipeline/postprocess.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BATCH_INPUT_DIR, BATCH_OUTPUT_DIR


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


def main() -> None:
    for batch_path in sorted(BATCH_INPUT_DIR.glob("*.jsonl")):
        if batch_path.name.endswith(".manifest.jsonl"):
            continue
        task = batch_path.stem.split("__part")[0]
        out_path = BATCH_OUTPUT_DIR / f"{batch_path.stem}.output.jsonl"
        n = 0
        with batch_path.open(encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                response_format = req["body"]["response_format"]
                content_str = _fake_content_for(response_format)
                spec = _patch(task, json.loads(content_str))
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


if __name__ == "__main__":
    main()
