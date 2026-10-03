#!/usr/bin/env python3
"""Drive the RL generation run language by language, through the provider's batch queue.

    python scripts/run_rl_batches.py --alloc /tmp/rl_alloc.json            # submit everything outstanding
    python scripts/run_rl_batches.py --alloc ... --fetch-only              # just collect what is finished
    python scripts/run_rl_batches.py --alloc ... --langs fon,ewe           # one or two languages

WHY A DRIVER AND NOT A LOOP IN THE SHELL
----------------------------------------
Three things make this run awkward to do by hand, and all three are state:

  1. The budget is per LANGUAGE, in kept samples, while `generate.py --limit` counts REQUESTS. The conversion
     is the measured yield, which differs per tier by more than 2x -- so Fon needs 5,263 requests for 2,000
     samples where English needs 7,500 for 6,000.
  2. Batches are asynchronous with a 24h window. Submitting returns an id that must outlive the process, or
     the work is paid for and lost.
  3. The document pre-stage is SYNCHRONOUS even in batch mode (longdocs runs before the batch is built,
     because a sample's prompt has to contain its document). At 17% of the mix that is the slowest part of
     the run by a wide margin, and it must not be redone on a restart.

State lives in <state>.json next to the allocation, so an interrupted run resumes instead of re-paying.
generate.py already skips custom_ids present on disk, so re-running a language is safe and cheap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "seeds"))

# Measured yields, same source as scripts/allocate_budget.py. Requests = samples / yield, with headroom,
# because under-requesting silently delivers fewer samples than the budget bought.
from allocate_budget import _TIER, _YIELD  # noqa: E402

HEADROOM = 1.10


def requests_for(lang: str, samples: int) -> int:
    y = _YIELD[_TIER.get(lang, "low")]
    return int(samples / y * HEADROOM) + 1


class State:
    """Resumable record of what has been submitted and fetched."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"langs": {}}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def lang(self, code: str) -> dict:
        return self.data["langs"].setdefault(code, {"batches": [], "fetched": [], "requested": 0})


def _run(cmd: list[str], env: dict) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def submit(lang: str, n_req: int, args, env: dict, st: State) -> None:
    rec = st.lang(lang)
    cmd = [sys.executable, "generate.py", "--kind", "rl", "--provider", args.provider,
           "--langs", lang, "--limit", str(n_req), "--batch",
           "--concurrency", str(args.concurrency)]
    if args.model:
        cmd += ["--model", args.model]
    print(f"[{lang}] submitting up to {n_req:,} requests "
          f"(documents are generated synchronously first -- this is the slow part)", flush=True)
    code, out = _run(cmd, env)
    ids = re.findall(r"batch ([0-9a-f-]{16,})", out)
    for b in ids:
        if b not in rec["batches"]:
            rec["batches"].append(b)
    rec["requested"] = max(rec["requested"], n_req)
    st.save()
    tail = "\n".join(out.strip().splitlines()[-4:])
    if ids:
        print(f"[{lang}] batch(es): {', '.join(ids)}", flush=True)
    else:
        print(f"[{lang}] NO BATCH ID RETURNED (exit {code}). Last output:\n{tail}", flush=True)


def fetch(lang: str, args, env: dict, st: State) -> bool:
    """Fetch every outstanding batch for `lang`. Returns True when none are left pending."""
    rec = st.lang(lang)
    pending = [b for b in rec["batches"] if b not in rec["fetched"]]
    if not pending:
        return True
    done = True
    for b in pending:
        cmd = [sys.executable, "generate.py", "--kind", "rl", "--provider", args.provider, "--fetch", b]
        if args.push:
            cmd.append("--push")
        code, out = _run(cmd, env)
        last = "\n".join(out.strip().splitlines()[-3:])
        if re.search(r"\bkept \d+ of \d+", out) or "completed" in out:
            if "in_progress" not in out:
                rec["fetched"].append(b)
                st.save()
                print(f"[{lang}] {b}: {last}", flush=True)
                continue
        done = False
        print(f"[{lang}] {b}: still running -- {last}", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True, help="json written by scripts/allocate_budget.py")
    ap.add_argument("--state", default="", help="defaults to <alloc>.state.json")
    ap.add_argument("--provider", default="doubleword")
    ap.add_argument("--model", default="")
    ap.add_argument("--langs", default="", help="comma list; default is every language in the allocation")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="applies to the SYNCHRONOUS document pre-stage, which is the bottleneck. The "
                         "provider returns empty 200s when pushed too hard; those are retried, so a high "
                         "value costs wall-clock rather than data.")
    ap.add_argument("--push", action="store_true", help="push each fetched shard to the Hub")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--poll-minutes", type=int, default=0,
                    help="keep polling until every batch is collected (0 = one pass)")
    a = ap.parse_args()

    alloc = json.loads(Path(a.alloc).read_text())["alloc"]
    if a.langs:
        want = [s.strip() for s in a.langs.split(",") if s.strip()]
        alloc = {k: v for k, v in alloc.items() if k in want}
    st = State(Path(a.state or (a.alloc + ".state.json")))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    # Cheapest-per-sample first: if the run has to be cut short, the money already spent has bought the most.
    order = sorted(alloc.items(), key=lambda kv: (_TIER.get(kv[0], "low") != "high", -kv[1]))

    if not a.fetch_only:
        print(f"plan: {sum(alloc.values()):,} samples across {len(alloc)} languages", flush=True)
        for lang, samples in order:
            n = requests_for(lang, samples)
            rec = st.lang(lang)
            if rec["batches"]:
                print(f"[{lang}] already submitted ({len(rec['batches'])} batch(es)) -- skipping", flush=True)
                continue
            submit(lang, n, a, env, st)

    deadline = time.time() + a.poll_minutes * 60
    while True:
        outstanding = [l for l, _ in order if not fetch(l, a, env, st)]
        if not outstanding:
            print("all batches collected", flush=True)
            return 0
        if time.time() >= deadline:
            print(f"still outstanding: {', '.join(outstanding)}\n"
                  f"re-run with --fetch-only to collect them.", flush=True)
            return 0
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
