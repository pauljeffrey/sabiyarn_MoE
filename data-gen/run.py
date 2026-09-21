#!/usr/bin/env python3
"""Single entrypoint for the data-gen pipeline.

Original six SFT tasks (unchanged):
    python run.py build      [--tasks ...]              # free, local only
    python run.py mock       [--tasks ...]               # free, local only (pipeline-tests postprocess without the API)
    python run.py submit     [--tasks ...] [--confirm]   # costs money once --confirm is passed
    python run.py fetch      [--check-only]              # polls/downloads
    python run.py postprocess [--tasks ...]               # free, local only

New corpus kinds (pretrain / sft / dpo, 12 non-English languages, gpt-4o-mini):
    python run.py estimate    [--kind pretrain|sft|dpo|all]                    # volume/token/$ estimate, no API calls
    python run.py build       --kind pretrain --config configs/pretrain.yaml   # free, local only (add --dry-run to just estimate)
    python run.py mock                                                          # fake outputs for every built file
    python run.py submit      --kind pretrain [--confirm]                       # dry run unless --confirm (costs money)
    python run.py fetch                                                         # polls/downloads all tracked batches
    python run.py postprocess --kind pretrain [--config ...]                    # parse, filter, dedup, report
    python run.py judge       build|apply --kind sft                            # optional LLM-judge stage

`--kind X` on `build`/`postprocess` selects the corpus pipeline; on `submit` it
is shorthand for `--tasks X`. Each subcommand is also runnable directly, e.g.
`python pipeline/build_batch.py`. This wrapper exists just for convenience.
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
    "estimate": ROOT / "pipeline" / "estimate.py",
    "judge": ROOT / "pipeline" / "judge.py",
}

# With --kind these two switch to the corpus pipeline.
CORPUS_VARIANTS = {
    "build": ROOT / "pipeline" / "build_corpus.py",
    "postprocess": ROOT / "pipeline" / "postprocess_corpus.py",
}


def _has_kind(args: list[str]) -> bool:
    return any(a == "--kind" or a.startswith("--kind=") for a in args)


def _kind_to_tasks(args: list[str]) -> list[str]:
    """`submit --kind sft` -> `submit --tasks sft` (submit_batch.py's own flag)."""
    out: list[str] = []
    it = iter(args)
    for a in it:
        if a == "--kind":
            out += ["--tasks", next(it, "")]
        elif a.startswith("--kind="):
            out += ["--tasks", a.split("=", 1)[1]]
        else:
            out.append(a)
    return out


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(__doc__)
        sys.exit(1 if len(sys.argv) >= 2 else 0)

    cmd, rest = sys.argv[1], sys.argv[2:]
    script = SUBCOMMANDS[cmd]
    if cmd in CORPUS_VARIANTS and _has_kind(rest):
        script = CORPUS_VARIANTS[cmd]
    elif cmd == "submit" and _has_kind(rest):
        rest = _kind_to_tasks(rest)
    sys.argv = [str(script)] + rest
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
