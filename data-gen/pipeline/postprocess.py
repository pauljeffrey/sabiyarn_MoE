#!/usr/bin/env python3
"""Turn raw Batch API output into final SFT-ready training examples.

For each task, joins data/batch_output/<task>.output.jsonl (model
responses) back to data/batch_input/<task>.manifest.jsonl (generation
context, keyed by custom_id), reconstructs a Conversation per task's
own message layout (see render_<task> below -- this is the ONE place that
decides how each task's structured fields become actual chat turns),
validates it, deduplicates within each (task, language) cell, renders it
through the real chat template, and writes:

  - data/processed/<task>.jsonl   -- per-task training examples
  - data/processed/all.jsonl      -- every task combined
  - data/reports/summary.json     -- counts + warnings per (task, language)

Usage:
    python pipeline/postprocess.py
    python pipeline/postprocess.py --tasks rag,translation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BATCH_OUTPUT_DIR, BATCH_INPUT_DIR, PROCESSED_DIR, REPORTS_DIR, TASK_NAMES
from documents.loader import documents_by_id, load_default_documents
from quality.dedup import find_duplicates
from quality.validators import check_tool_arguments, language_sanity_warnings
from rendering.render import render_conversation
from schemas.messages import Conversation, Message, ToolCall, ToolFunctionCall, tool_result_content
from schemas.tools import EDGE_DEVICE_TOOLS_BY_NAME
from stats.report import RunStats


def _safe_json_loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Per-task rendering: structured GPT output + generation context -> messages
# ---------------------------------------------------------------------------


def render_rag(spec: dict, context: dict, docs_by_id: dict) -> list[Message]:
    doc = docs_by_id[context["doc_id"]]
    messages = [Message(role="system", content=context["final_system_prompt"])]
    for turn in spec["turns"]:
        messages.append(Message(role="user", content=turn["user_message"]))
        messages.append(
            Message(
                role="assistant",
                tool_calls=[ToolCall(function=ToolFunctionCall(name="search_document", arguments={"query": turn["search_query"]}))],
            )
        )
        if turn.get("answerable") and turn.get("relevant_chunk_id") is not None:
            chunk_text = doc.chunk_text(int(turn["relevant_chunk_id"]))  # raises KeyError if hallucinated id
            payload = {"chunk_id": turn["relevant_chunk_id"], "found": True, "text": chunk_text}
        else:
            payload = {"chunk_id": None, "found": False, "text": "No relevant passage found in the document."}
        messages.append(Message(role="tool", name="search_document", content=tool_result_content(payload)))
        messages.append(Message(role="assistant", content=turn["assistant_response"]))
    return messages


def render_summarization(spec: dict, context: dict, docs_by_id: dict) -> list[Message]:
    messages = [Message(role="system", content=context["final_system_prompt"])]
    if context["variant"] == "doc":
        messages.append(Message(role="user", content=spec["user_instruction"]))
        messages.append(Message(role="assistant", content=spec["summary"]))
        if spec.get("has_followup") and spec.get("followup_user_instruction") and spec.get("followup_summary"):
            messages.append(Message(role="user", content=spec["followup_user_instruction"]))
            messages.append(Message(role="assistant", content=spec["followup_summary"]))
    else:
        for turn in spec["prior_turns"]:
            messages.append(Message(role="user", content=turn["user_message"]))
            messages.append(Message(role="assistant", content=turn["assistant_message"]))
        messages.append(Message(role="user", content=spec["summarize_request"]))
        messages.append(Message(role="assistant", content=spec["summary"]))
    return messages


def render_edge_action(spec: dict, context: dict, docs_by_id: dict) -> list[Message]:
    messages = [Message(role="system", content=context["final_system_prompt"])]
    messages.append(Message(role="user", content=spec["user_instruction"]))

    reasoning_block = f"<reason>{spec['reasoning']}</reason><task_plan>{spec['task_plan']}</task_plan>"

    if spec.get("needs_tool_call") and spec.get("tool_name"):
        tool_name = spec["tool_name"]
        available = {t["function"]["name"]: t for t in context.get("available_tools", [])}
        tool_def = available.get(tool_name) or EDGE_DEVICE_TOOLS_BY_NAME.get(tool_name)
        if tool_def is None:
            raise ValueError(f"model called unknown tool {tool_name!r} not in available_tools")

        args = _safe_json_loads(spec.get("tool_arguments_json"), {})
        errors = check_tool_arguments(tool_def, args)
        if errors:
            raise ValueError(f"invalid arguments for {tool_name}: {errors}")

        result = _safe_json_loads(spec.get("tool_result_json"), {"status": "success"})

        messages.append(Message(role="assistant", content=reasoning_block))
        messages.append(
            Message(role="assistant", tool_calls=[ToolCall(function=ToolFunctionCall(name=tool_name, arguments=args))])
        )
        messages.append(Message(role="tool", name=tool_name, content=tool_result_content(result)))
        messages.append(Message(role="assistant", content=spec["final_response"]))
    else:
        messages.append(Message(role="assistant", content=f"{reasoning_block}{spec['final_response']}"))

    return messages


def render_structured_output(spec: dict, context: dict, docs_by_id: dict) -> list[Message]:
    final_system_prompt = context["final_system_prompt_template"].replace(
        "__SOURCE_TEXT_PLACEHOLDER__", spec["source_text"]
    )
    messages = [Message(role="system", content=final_system_prompt)]
    messages.append(Message(role="user", content=spec["user_instruction"]))
    extracted_json_str = json.dumps(spec["extracted"], ensure_ascii=False)
    messages.append(Message(role="assistant", content=extracted_json_str))
    return messages


def render_math_stats(spec: dict, context: dict, docs_by_id: dict) -> list[Message]:
    messages = [Message(role="system", content=context["final_system_prompt"])]
    messages.append(Message(role="user", content=spec["problem_statement"]))

    if spec.get("uses_calculator") and spec.get("calculator_expression"):
        messages.append(Message(role="assistant", content=spec["reasoning"]))
        messages.append(
            Message(
                role="assistant",
                tool_calls=[ToolCall(function=ToolFunctionCall(name="calculate", arguments={"expression": spec["calculator_expression"]}))],
            )
        )
        result_val = spec.get("calculator_result") or ""
        messages.append(Message(role="tool", name="calculate", content=tool_result_content({"result": result_val})))
        messages.append(Message(role="assistant", content=spec["final_answer"]))
    else:
        combined = f"{spec['reasoning']} {spec['final_answer']}".strip()
        messages.append(Message(role="assistant", content=combined))

    return messages


def render_translation(spec: dict, context: dict, docs_by_id: dict) -> list[Message]:
    messages = [Message(role="system", content=context["final_system_prompt"])]
    messages.append(Message(role="user", content=spec["user_instruction"]))
    messages.append(Message(role="assistant", content=spec["translated_text"]))
    if spec.get("has_followup") and spec.get("followup_user_instruction") and spec.get("followup_response"):
        messages.append(Message(role="user", content=spec["followup_user_instruction"]))
        messages.append(Message(role="assistant", content=spec["followup_response"]))
    return messages


RENDERERS = {
    "rag": render_rag,
    "summarization": render_summarization,
    "edge_action": render_edge_action,
    "structured_output": render_structured_output,
    "math_stats": render_math_stats,
    "translation": render_translation,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_manifest(task: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in sorted(BATCH_INPUT_DIR.glob(f"{task}.manifest.jsonl")) + sorted(BATCH_INPUT_DIR.glob(f"{task}__part*.manifest.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                by_id[entry["custom_id"]] = entry
    return by_id


def _iter_output_lines(task: str):
    for path in sorted(BATCH_OUTPUT_DIR.glob(f"{task}.output.jsonl")) + sorted(BATCH_OUTPUT_DIR.glob(f"{task}__part*.output.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def process_task(task: str, docs_by_id: dict, stats: RunStats) -> list[dict]:
    manifest = _load_manifest(task)
    renderer = RENDERERS[task]
    records: list[dict] = []

    for row in _iter_output_lines(task):
        custom_id = row.get("custom_id")
        entry = manifest.get(custom_id)
        if entry is None:
            continue  # output for a request we have no manifest context for; skip
        language = entry["language"]
        context = entry["context"]
        cell = (task, language)
        stats.generated[cell] += 1

        if row.get("error"):
            stats.parse_failed[cell] += 1
            stats.record_warning(task, language, custom_id, f"batch-level error: {row['error']}")
            continue

        body = (row.get("response") or {}).get("body") or {}
        if (row.get("response") or {}).get("status_code") != 200:
            stats.parse_failed[cell] += 1
            stats.record_warning(task, language, custom_id, f"non-200 status: {row.get('response')}")
            continue

        try:
            choice = body["choices"][0]["message"]
        except (KeyError, IndexError):
            stats.parse_failed[cell] += 1
            stats.record_warning(task, language, custom_id, "malformed response body")
            continue

        if choice.get("refusal"):
            stats.parse_failed[cell] += 1
            stats.record_warning(task, language, custom_id, f"model refused: {choice['refusal']}")
            continue

        try:
            spec = json.loads(choice["content"])
        except (json.JSONDecodeError, TypeError, KeyError):
            stats.parse_failed[cell] += 1
            stats.record_warning(task, language, custom_id, "response content was not valid JSON")
            continue

        try:
            messages = renderer(spec, context, docs_by_id)
            conversation = Conversation(example_id=custom_id, task=task, language=language, messages=messages, meta=context)
        except Exception as e:  # noqa: BLE001 - deliberately broad: any bad model output should be skipped, not crash the run
            stats.validation_failed[cell] += 1
            stats.record_warning(task, language, custom_id, f"validation failed: {e}")
            continue

        for msg in messages:
            if msg.role in ("user", "assistant") and msg.content:
                for w in language_sanity_warnings(msg.content, language):
                    stats.record_warning(task, language, custom_id, w)

        records.append(
            {
                "custom_id": custom_id,
                "task": task,
                "language": language,
                "conversation": conversation,
            }
        )

    return records


def dedup_and_write(task: str, records: list[dict], stats: RunStats) -> list[dict]:
    def cell_key(rec: dict) -> tuple[str, str]:
        return (rec["task"], rec["language"])

    def text_fn(rec: dict) -> str:
        conv: Conversation = rec["conversation"]
        return " ".join(m.content for m in conv.messages if m.role == "user" and m.content)

    drop_indices, reasons = find_duplicates(records, cell_key_fn=cell_key, text_fn=text_fn)

    kept: list[dict] = []
    for idx, rec in enumerate(records):
        cell = (rec["task"], rec["language"])
        if idx in drop_indices:
            stats.dedup_dropped[cell] += 1
            stats.record_warning(rec["task"], rec["language"], rec["custom_id"], reasons[idx])
            continue
        stats.kept[cell] += 1
        kept.append(rec)

    out_path = PROCESSED_DIR / f"{task}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            conv: Conversation = rec["conversation"]
            text = render_conversation(conv)
            f.write(json.dumps(conv.to_jsonl_record(text), ensure_ascii=False) + "\n")
    print(f"[{task}] kept {len(kept)}/{len(records)} -> {out_path}")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=str, default=",".join(TASK_NAMES))
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    docs_by_id = documents_by_id(load_default_documents())
    stats = RunStats()

    all_path = PROCESSED_DIR / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as all_f:
        for task in tasks:
            records = process_task(task, docs_by_id, stats)
            kept = dedup_and_write(task, records, stats)
            for rec in kept:
                conv: Conversation = rec["conversation"]
                text = render_conversation(conv)
                all_f.write(json.dumps(conv.to_jsonl_record(text), ensure_ascii=False) + "\n")

    report_path = REPORTS_DIR / "summary.json"
    stats.write(report_path)
    print("\n" + stats.as_table())
    print(f"\nTotals: {stats.totals()}")
    print(f"Full report: {report_path}")
    print(f"Combined output: {all_path}")


if __name__ == "__main__":
    main()
