#!/usr/bin/env python3
"""Poll tracked Batch API jobs and download results for completed ones.

Reads data/batch_manifest.jsonl (written by submit_batch.py), checks each
batch's current status, and for any batch whose status is "completed",
downloads its output (and error, if any) file into data/batch_output/.

Usage:
    python pipeline/fetch_results.py               # check status + download completed
    python pipeline/fetch_results.py --check-only   # just print status, no downloads
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BATCH_OUTPUT_DIR, MANIFEST_PATH


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    entries = []
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Print status only, do not download.")
    args = parser.parse_args()

    entries = _load_manifest()
    if not entries:
        print(f"No submitted batches found in {MANIFEST_PATH}. Run submit_batch.py first.")
        return

    from openai import OpenAI

    client = OpenAI()

    for entry in entries:
        batch = client.batches.retrieve(entry["batch_id"])
        counts = batch.request_counts
        print(
            f"[{entry['task']}] batch_id={batch.id} status={batch.status} "
            f"completed={counts.completed}/{counts.total} failed={counts.failed}"
        )

        if args.check_only or batch.status != "completed":
            continue

        out_path = BATCH_OUTPUT_DIR / f"{entry['task']}.output.jsonl"
        if batch.output_file_id:
            content = client.files.content(batch.output_file_id)
            out_path.write_bytes(content.read())
            print(f"  wrote {out_path}")

        if batch.error_file_id:
            err_path = BATCH_OUTPUT_DIR / f"{entry['task']}.errors.jsonl"
            content = client.files.content(batch.error_file_id)
            err_path.write_bytes(content.read())
            print(f"  wrote {err_path} (request-level errors -- review these)")

    print("\nOnce all relevant batches show status=completed and are downloaded, run pipeline/postprocess.py.")


if __name__ == "__main__":
    main()
