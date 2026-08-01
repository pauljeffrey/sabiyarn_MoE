#!/usr/bin/env python3
"""Single entrypoint for the data-gen pipeline.

    python run.py build      [--tasks ...]              # free, local only
    python run.py mock       [--tasks ...]               # free, local only (pipeline-tests postprocess without the API)
    python run.py submit     [--tasks ...] [--confirm]   # costs money once --confirm is passed
    python run.py fetch      [--check-only]              # polls/downloads
    python run.py postprocess [--tasks ...]               # free, local only

Each subcommand is also runnable directly, e.g. `python pipeline/build_batch.py`.
This wrapper exists just for convenience/discoverability.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUBCOMMANDS = {
    "build": ROOT / "pipeline" / "build_batch.py",
    "submit": ROOT / "pipeline" / "submit_batch.py",
    "fetch": ROOT / "pipeline" / "fetch_results.py",
    "postprocess": ROOT / "pipeline" / "postprocess.py",
    "mock": ROOT / "scripts" / "mock_generate.py",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(__doc__)
        sys.exit(1 if len(sys.argv) >= 2 else 0)

    cmd = sys.argv[1]
    sys.argv = [str(SUBCOMMANDS[cmd])] + sys.argv[2:]
    runpy.run_path(str(SUBCOMMANDS[cmd]), run_name="__main__")


if __name__ == "__main__":
    main()
