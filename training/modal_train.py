#!/usr/bin/env python3
"""Modal launcher for YAML-driven pretrain / SFT / RL training.

GPU topology (num_nodes, gpus_per_node, gpu_type) is read from train_config.yaml's
`modal:` section at *deploy* time -- Modal's cluster size and GPU shape are fixed
per Function, not a runtime argument. Edit the yaml and redeploy to change
topology. `mode` and `override` remain true runtime arguments.

Every training run syncs and mixes *both* the eng and afr bins configured under
`data.<mode>` -- there's no `--data-type` flag here (that only makes sense for
`data/prepare_modal.py`, which tokenizes one language's raw datasets at a time).
The eng/afr mix ratio is controlled by `data.sampling` in train_config.yaml.

Multi-node rendezvous uses modal.experimental.clustered()/get_cluster_info(),
which provisions `num_nodes` networked containers per invocation and hands back
each container's rank + peer IPs -- the rank-0 container's IP becomes
--master_addr for torchrun on every node.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up MODAL_TOKEN_ID/SECRET, S3/HF/wandb keys for local `modal run`

import modal
import modal.experimental
import yaml

# Modal mounts add_local_dir's data at /app inside the container, but re-imports
# *this entrypoint script itself* from a separate location (observed at
# /root/modal_train.py) when hydrating a function for remote execution --
# Path(__file__).resolve().parents[1] is only correct when running locally.
# Detect which context we're in and resolve ROOT (and therefore the `training`
# package on sys.path) accordingly.
_APP_MOUNT = Path("/app")
ROOT = _APP_MOUNT if (_APP_MOUNT / "training").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.load_config import load_train_config
from training.s3_utils import sync_training_files

DATA_DIR = "/data"
CONFIG_PATH = str(ROOT / "training" / "train_config.yaml")
MASTER_PORT = "29500"

with open(CONFIG_PATH, "r", encoding="utf-8") as _fh:
    _raw_cfg = yaml.safe_load(_fh) or {}
_modal_cfg = _raw_cfg.get("modal", {}) or {}
NUM_NODES = max(1, int(_modal_cfg.get("num_nodes", 1)))
GPUS_PER_NODE = max(1, int(_modal_cfg.get("gpus_per_node", 1)))
GPU_TYPE = str(_modal_cfg.get("gpu_type", "A100"))
GPU_SPEC = f"{GPU_TYPE}:{GPUS_PER_NODE}" if GPUS_PER_NODE > 1 else GPU_TYPE
NODE_CPU = max(16, 8 * GPUS_PER_NODE)

# training.out_dir names this run's checkpoint subdirectory on the persistent
# volume (e.g. "out_280M") -- keeping it config-driven, rather than a single
# fixed "checkpoints" folder shared by every config, means different model
# sizes/experiments don't collide in the same directory and confuse
# new_train.py's latest-checkpoint auto-discovery with an incompatible run.
_training_cfg = _raw_cfg.get("training", {}) or {}
CFG_OUT_DIR = str(_training_cfg.get("out_dir", "checkpoints"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.55.0",
        "accelerate>=0.34.0",
        "wandb",
        "structlog",
        "numpy",
        "omegaconf",
        "pyyaml",
        "boto3",
        "datasets",
        "huggingface_hub",
        "python-dotenv",
        "lmdb",
        "bitsandbytes",
        "psutil",
    )
    .add_local_dir(
        str(ROOT), remote_path="/app",
        ignore=[
            ".git", "__pycache__", "*.pyc", "out/", ".env",
            # Local-only dev state: Modal installs its own deps via pip_install
            # above, and .venv can contain Unix-style symlinks (e.g. from uv)
            # that Windows can't read when walking the directory to upload it.
            ".venv", ".pytest_cache", ".claude",
        ],
    )
)

app = modal.App("sabiyarn-modal-training")
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)


def _build_env(mode: str, override: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env["TRAIN_CONFIG_PATH"] = CONFIG_PATH
    env["TRAIN_MODE"] = mode
    # Persist checkpoints on the mounted volume (not the training container's ephemeral
    # disk) so they survive preemption and are visible to eval/test_generation containers.
    # Namespaced under CFG_OUT_DIR (training.out_dir) so separate configs/experiments
    # get separate checkpoint trees on the shared volume instead of overwriting/
    # colliding with each other's runs.
    env["TRAIN_OUT_DIR"] = os.path.join(DATA_DIR, CFG_OUT_DIR)
    env["OVERRIDE_DATA"] = "1" if override else "0"
    return env


def _sync_data(env: dict[str, str]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    # Each node independently syncs its own local copy of the training data from
    # S3 (no-op if already present, e.g. via the shared Modal Volume). Both the
    # eng and afr bins are always synced -- training mixes them per-batch via
    # data.sampling in train_config.yaml, it doesn't train on a single language.
    cfg = load_train_config(CONFIG_PATH)
    s3_fields = {
        "s3_bucket": cfg.s3_bucket, "s3_endpoint": cfg.s3_endpoint,
        "s3_access_key": bool(cfg.s3_access_key), "s3_secret_key": bool(cfg.s3_secret_key),
    }
    if not all(s3_fields.values()):
        missing = [k for k, v in s3_fields.items() if not v]
        print(
            f"[data sync] SKIPPED -- missing/empty config: {missing}. "
            "Training will fail to find local data files unless they're already "
            "present (e.g. from a previous run's Modal Volume). Check that the "
            "s3-secret Modal secret actually has non-empty S3_ACCESS_KEY_ID/"
            "S3_SECRET_ACCESS_KEY values (`modal secret list` only shows names, "
            "not contents)."
        )
        return

    local_train, local_eval = sync_training_files(
        cfg.train_data_paths,
        DATA_DIR,
        bucket=cfg.s3_bucket,
        endpoint=cfg.s3_endpoint,
        access_key=cfg.s3_access_key,
        secret_key=cfg.s3_secret_key,
        prefix=cfg.s3_prefix,
        eval_path=cfg.eval_data_path,
    )
    print(f"[data sync] train files: {local_train}, eval file: {local_eval}")
    if local_train:
        env["TRAIN_DATA_PATHS_LOCAL"] = ",".join(local_train)
    if local_eval:
        env["VAL_DATA_PATH"] = local_eval


_common_kwargs = dict(
    image=image,
    gpu=GPU_SPEC,
    cpu=NODE_CPU,
    timeout=86400,
    volumes={DATA_DIR: volume},
    # Reads your local .env at `modal run` time and injects it through Modal's
    # normal encrypted secret mechanism -- no `modal secret create` step, and
    # .env itself is still never uploaded/baked into the image (it's excluded
    # from add_local_dir above; from_dotenv reads it locally, not from /app).
    secrets=[modal.Secret.from_dotenv(__file__)],
)


@app.function(**_common_kwargs)
def train_single_node(mode: str = "pretrain", override: bool = False):
    """Single node, `GPUS_PER_NODE` GPUs -- a plain Function, not clustered().

    modal.experimental.clustered() reserves whole physical hosts and, on this
    platform, requires the full GPU device count per node for the selected
    GPU type (e.g. 8 for A100) even when num_nodes=1 -- it rejects partial
    counts like 2. Single-node multi-GPU doesn't need that machinery at all;
    --standalone torchrun on one container handles it directly.
    """
    env = _build_env(mode, override)
    _sync_data(env)

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={GPUS_PER_NODE}",
        "-m", "training.new_train",
    ]
    print(f"[single node, {GPUS_PER_NODE} GPU(s)] launching: {' '.join(cmd)}")
    subprocess.run(cmd, cwd="/app", env=env, check=True)
    return True


# Only define train_cluster at all when it's actually needed. Modal validates
# a clustered() function's GPU spec against the full-per-node-device-count
# rule at *deploy* time -- as soon as the function exists in the app, whether
# or not it's ever invoked. If this were defined unconditionally, a
# single-node config with gpus_per_node < the full device count for GPU_TYPE
# (e.g. 2 A100s instead of 8) would fail the whole `modal run` before
# train_single_node ever got a chance to be scheduled, even though nothing
# actually calls train_cluster in that case.
if NUM_NODES > 1:
    @app.function(**_common_kwargs)
    @modal.experimental.clustered(size=NUM_NODES)
    def train_cluster(mode: str = "pretrain", override: bool = False):
        """True multi-node (num_nodes > 1). Modal requires gpus_per_node to be the
        full device count for GPU_TYPE here (e.g. 8 for A100) -- partial counts
        are rejected for clustered functions."""
        cluster_info = modal.experimental.get_cluster_info()
        rank = cluster_info.rank
        master_addr = cluster_info.container_ips[0]

        env = _build_env(mode, override)
        _sync_data(env)

        cmd = [
            "torchrun",
            f"--nnodes={NUM_NODES}",
            f"--node_rank={rank}",
            f"--nproc_per_node={GPUS_PER_NODE}",
            f"--master_addr={master_addr}",
            f"--master_port={MASTER_PORT}",
            "-m", "training.new_train",
        ]
        print(f"[node {rank}/{NUM_NODES}] launching: {' '.join(cmd)}")
        subprocess.run(cmd, cwd="/app", env=env, check=True)
        return True


@app.local_entrypoint()
def main(mode: str = "pretrain", override: bool = False):
    if NUM_NODES > 1:
        train_cluster.remote(mode=mode, override=override)
    else:
        train_single_node.remote(mode=mode, override=override)
