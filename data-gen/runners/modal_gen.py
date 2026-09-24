#!/usr/bin/env python3
"""Run data generation on Modal.

    modal run data-gen/runners/modal_gen.py --kind sft --limit 2000
    modal run data-gen/runners/modal_gen.py --kind pretrain --provider together --batch
    modal run data-gen/runners/modal_gen.py --kind sft --shards 8 --limit 40000    # 8 workers in parallel

This is API-call work, not GPU work: the containers are CPU-only and cheap. Output goes straight to the
Hugging Face dataset repo rather than a Modal volume, so nothing has to be fetched afterwards and several
workers (or several platforms at once) can contribute to the same corpus.

Sharding is by `--shards N`: worker i takes every Nth row of the plan. Because plan rows are deterministic
and shard files are named per worker, N workers never collide and a failed worker is just re-run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
DATA_GEN = ROOT / "data-gen"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.40", "huggingface_hub>=0.34", "pydantic>=2.7", "jinja2>=3.1",
                 "pyyaml>=6.0", "python-dotenv>=1.0", "tqdm", "rich")
    .add_local_dir(str(DATA_GEN), remote_path="/app/data-gen",
                   ignore=["**/__pycache__", "*.pyc", ".venv*", "data/out/**", ".env"])
)

app = modal.App("sabiyarn-data-gen")


@app.function(image=image, cpu=4.0, timeout=86400, secrets=[modal.Secret.from_dotenv(str(ROOT))])
def generate(kind: str, provider: str = "openrouter", model: str = "", limit: int = 0,
             langs: str = "", concurrency: int = 24, batch: bool = False,
             shard_index: int = 0, shards: int = 1, repo_id: str = "BeardedMonster/data-gen"):
    import subprocess

    cmd = [sys.executable, "generate.py", "--kind", kind, "--provider", provider,
           "--concurrency", str(concurrency), "--push", "--repo-id", repo_id]
    if model:
        cmd += ["--model", model]
    if limit:
        cmd += ["--limit", str(limit)]
    if langs:
        cmd += ["--langs", langs]
    if batch:
        cmd += ["--batch"]
    env = {**os.environ, "PYTHONPATH": "/app/data-gen",
           # Each worker takes a disjoint slice of the plan; see generate.py --shard.
           "DATA_GEN_SHARD_INDEX": str(shard_index), "DATA_GEN_SHARDS": str(shards),
           "DATA_GEN_OUTPUT_DIR": "/tmp/data-gen"}
    print(f"[modal shard {shard_index}/{shards}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd="/app/data-gen", env=env, check=True)
    return True


@app.local_entrypoint()
def main(kind: str = "sft", provider: str = "openrouter", model: str = "", limit: int = 0,
         langs: str = "", concurrency: int = 24, batch: bool = False, shards: int = 1,
         repo_id: str = "BeardedMonster/data-gen"):
    if shards <= 1:
        generate.remote(kind, provider, model, limit, langs, concurrency, batch, 0, 1, repo_id)
        return
    args = [(kind, provider, model, limit, langs, concurrency, batch, i, shards, repo_id)
            for i in range(shards)]
    for _ in generate.starmap(args):
        pass
    print(f"all {shards} shards finished")
