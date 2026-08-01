#!/usr/bin/env python3
"""Upload batch input files and create OpenAI Batch API jobs.

THIS STEP SPENDS MONEY. It is intentionally inert without --confirm: by
default it only prints what it WOULD submit (files, request counts, an
approximate cost order-of-magnitude) so you can review before committing.
Batch API pricing is typically ~50% of standard sync pricing, and jobs run
within the completion window (currently 24h) rather than immediately.

Usage:
    python pipeline/submit_batch.py                       # dry run, no API calls
    python pipeline/submit_batch.py --tasks rag --confirm  # actually submits
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BATCH_COMPLETION_WINDOW, BATCH_INPUT_DIR, MANIFEST_PATH, TASK_NAMES


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def _find_batch_files(tasks: list[str]) -> list[Path]:
    files = []
    for task in tasks:
        # Matches both "<task>.jsonl" and split "<task>__partN.jsonl"
        files.extend(sorted(BATCH_INPUT_DIR.glob(f"{task}.jsonl")))
        files.extend(sorted(BATCH_INPUT_DIR.glob(f"{task}__part*.jsonl")))
    return files


def _append_manifest(entry: dict) -> None:
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=str, default=",".join(TASK_NAMES))
    parser.add_argument("--confirm", action="store_true", help="Actually call the OpenAI API and submit. Omit for a dry run.")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    files = _find_batch_files(tasks)
    if not files:
        print(f"No batch input files found for tasks {tasks} in {BATCH_INPUT_DIR}. Run build_batch.py first.")
        return

    print(f"Found {len(files)} batch input file(s):")
    total_requests = 0
    for f in files:
        n = _count_lines(f)
        total_requests += n
        print(f"  {f.name}: {n} requests")
    print(f"Total: {total_requests} requests across {len(files)} file(s).")

    if not args.confirm:
        print(
            "\nDry run only (no API calls made). Re-run with --confirm to "
            "actually upload these files and create Batch API jobs -- this "
            "will incur OpenAI API cost."
        )
        return

    from openai import OpenAI

    client = OpenAI()

    for f in files:
        task_name = f.stem.split("__part")[0]
        print(f"Uploading {f.name} ...")
        with f.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
        print(f"  file_id={uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window=BATCH_COMPLETION_WINDOW,
            metadata={"task": task_name, "source_file": f.name},
        )
        print(f"  batch_id={batch.id} status={batch.status}")

        _append_manifest(
            {
                "task": task_name,
                "source_file": f.name,
                "input_file_id": uploaded.id,
                "batch_id": batch.id,
                "status": batch.status,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    print(f"\nSubmitted. Tracking info appended to {MANIFEST_PATH}.")
    print("Use pipeline/fetch_results.py to poll and download results.")


if __name__ == "__main__":
    main()
