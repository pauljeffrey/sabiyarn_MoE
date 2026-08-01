#!/usr/bin/env python3
"""Build OpenAI Batch API input files (+ side-car manifests) from all task
generators, or a subset selected with --tasks.

This step makes NO network calls and costs nothing -- it only writes
JSONL files to data/batch_input/. Review the output before running
submit_batch.py, which is the step that actually spends money.

Usage:
    python pipeline/build_batch.py
    python pipeline/build_batch.py --tasks rag,translation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import TASK_NAMES
from generators.base import BatchRequestSpec, write_batch_and_manifest

# OpenAI Batch API hard limit at time of writing; kept as a local constant
# (not settings.py) since it's an API constraint, not a tuning knob.
MAX_REQUESTS_PER_BATCH_FILE = 50_000

GENERATOR_MODULES = {
    "rag": "generators.rag",
    "summarization": "generators.summarization",
    "edge_action": "generators.edge_action",
    "structured_output": "generators.structured_output",
    "math_stats": "generators.math_stats",
    "translation": "generators.translation",
}


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_task(task_name: str) -> None:
    import importlib

    module = importlib.import_module(GENERATOR_MODULES[task_name])
    specs: list[BatchRequestSpec] = module.build_requests()

    chunks = _chunk(specs, MAX_REQUESTS_PER_BATCH_FILE)
    for part_idx, chunk in enumerate(chunks):
        name = task_name if len(chunks) == 1 else f"{task_name}__part{part_idx}"
        batch_path, manifest_path, n = write_batch_and_manifest(name, chunk)
        print(f"[{task_name}] wrote {n} requests -> {batch_path.name} (+ {manifest_path.name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASK_NAMES),
        help=f"Comma-separated task names to build. Available: {', '.join(TASK_NAMES)}",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in GENERATOR_MODULES]
    if unknown:
        parser.error(f"Unknown task(s): {unknown}. Available: {list(GENERATOR_MODULES)}")

    for task_name in tasks:
        build_task(task_name)
    print("Done. Review data/batch_input/*.jsonl before running submit_batch.py.")


if __name__ == "__main__":
    main()
