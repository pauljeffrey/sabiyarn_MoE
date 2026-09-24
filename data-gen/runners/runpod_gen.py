#!/usr/bin/env python3
"""Run data generation on RunPod, or on any box you can SSH into (vast.ai included).

Two ways to use it:

  ON the pod (simplest -- also exactly what you do on vast.ai):
      git clone <repo> && cd sabiyarn_MoE/data-gen
      pip install -r requirements.txt
      cp ../.env .env.local          # or export the keys
      python runners/runpod_gen.py --kind sft --limit 20000 --shards 4

    `--shards N` forks N local worker processes, which is how you saturate a box: this workload is
    network-bound, so 4-8 workers x 24 concurrent requests each is usually the sweet spot.

  FROM your laptop, creating the pod for you (needs RUNPOD_API_KEY and the `runpod` package):
      python runners/runpod_gen.py --launch --kind sft --limit 50000

Note RunPod bills GPU pods by the GPU even when, as here, the work is pure API calls. If you are paying
rather than burning credit, a CPU pod (or Modal, or a vast CPU box) is much cheaper for this particular job.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GEN = HERE / "generate.py"


def run_local(kind: str, provider: str, model: str, limit: int, langs: str, concurrency: int,
              shards: int, push: bool, repo_id: str) -> int:
    procs = []
    for i in range(max(1, shards)):
        cmd = [sys.executable, str(GEN), "--kind", kind, "--provider", provider,
               "--concurrency", str(concurrency), "--repo-id", repo_id]
        if model:
            cmd += ["--model", model]
        if limit:
            cmd += ["--limit", str(limit // max(1, shards))]
        if langs:
            cmd += ["--langs", langs]
        if push:
            cmd += ["--push"]
        env = {**os.environ, "PYTHONPATH": str(HERE),
               "DATA_GEN_SHARD_INDEX": str(i), "DATA_GEN_SHARDS": str(shards)}
        print(f"[worker {i}] {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(HERE), env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    return rc


def launch_pod(args) -> int:
    try:
        import runpod
    except ImportError:
        raise SystemExit("pip install runpod  (or run this script ON the pod instead of --launch)")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("set RUNPOD_API_KEY in the repo-root .env")
    runpod.api_key = key
    repo = os.environ.get("DATA_GEN_GIT_REPO", "https://github.com/pauljeffrey/sabiyarn_MoE")
    keys = " ".join(f"{k}={os.environ[k]}" for k in
                    ("OPENROUTER_API_KEY", "TOGETHER_API_KEY", "HF_TOKEN", "HF_WRITE_TOKEN")
                    if os.environ.get(k))
    cmd = (f"bash -lc 'git clone {repo} /work && cd /work/data-gen && "
           f"pip install -q -r requirements.txt && export {keys} && "
           f"python runners/runpod_gen.py --kind {args.kind} --provider {args.provider} "
           f"--limit {args.limit} --shards {args.shards} --push'")
    pod = runpod.create_pod(name=f"sabiyarn-datagen-{args.kind}",
                            image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                            gpu_type_id=args.gpu_type, cloud_type="SECURE", gpu_count=1,
                            docker_args=cmd)
    print(f"launched pod {pod.get('id')}. Watch it in the RunPod console; it pushes to {args.repo_id} "
          f"and you can terminate it as soon as the push lands.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["pretrain", "sft", "rl"])
    ap.add_argument("--provider", default="openrouter", choices=["together", "openrouter"])
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--langs", default="")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--shards", type=int, default=1, help="local worker processes")
    ap.add_argument("--push", action="store_true", default=True)
    ap.add_argument("--no-push", dest="push", action="store_false")
    ap.add_argument("--repo-id", default="BeardedMonster/data-gen")
    ap.add_argument("--launch", action="store_true", help="create a RunPod pod instead of running here")
    ap.add_argument("--gpu-type", default="NVIDIA RTX A4000")
    a = ap.parse_args()
    if a.launch:
        return launch_pod(a)
    return run_local(a.kind, a.provider, a.model, a.limit, a.langs, a.concurrency, a.shards,
                     a.push, a.repo_id)


if __name__ == "__main__":
    raise SystemExit(main())
