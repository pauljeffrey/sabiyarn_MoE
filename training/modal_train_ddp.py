#!/usr/bin/env python3
"""Modal launcher for training/new_train_ddp.py -- the plain-DDP diagnostic
variant of new_train.py's Trainer (see that module's docstring for why).
Mirrors modal_train.py closely, just pointed at a different entrypoint
module and a separate Modal app name, so it never collides with or
modifies your normal FSDP training app/deployment. Single-node only --
this is meant as a short, targeted comparison run, not a permanent
multi-node setup.

Reads the same train_config.yaml (data, model, S3, HF, sampling, etc. all
identical to the FSDP path) -- only how the model gets wrapped for compute
differs. See training/new_train_ddp.py's module docstring for the memory
caveat: DDP needs a full, unsharded model + optimizer per GPU, so you may
need a smaller train_batch_size/block_size than your usual FSDP config to
avoid OOM.

Usage:
    modal run training/modal_train_ddp.py --mode pretrain
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import modal
import yaml

_APP_MOUNT = Path("/app")
ROOT = _APP_MOUNT if (_APP_MOUNT / "training").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.load_config import load_train_config
from training.s3_utils import sync_training_files

DATA_DIR = "/data"
CONFIG_PATH = str(ROOT / "training" / "train_config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as _fh:
    _raw_cfg = yaml.safe_load(_fh) or {}
_modal_cfg = _raw_cfg.get("modal", {}) or {}
GPUS_PER_NODE = max(1, int(_modal_cfg.get("gpus_per_node", 1)))
GPU_TYPE = str(_modal_cfg.get("gpu_type", "A100"))
GPU_SPEC = f"{GPU_TYPE}:{GPUS_PER_NODE}" if GPUS_PER_NODE > 1 else GPU_TYPE
NODE_CPU = max(16, 8 * GPUS_PER_NODE)

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
            ".venv", ".pytest_cache", ".claude",
        ],
    )
)

# Separate app name from modal_train.py's "sabiyarn-modal-training" --
# this is a standalone diagnostic run, never meant to collide with or
# affect your normal FSDP training deployment.
app = modal.App("sabiyarn-modal-training-ddp")
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)


def _build_env(mode: str, override: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env["TRAIN_CONFIG_PATH"] = CONFIG_PATH
    env["TRAIN_MODE"] = mode
    env["TRAIN_OUT_DIR"] = os.path.join(DATA_DIR, CFG_OUT_DIR)
    env["OVERRIDE_DATA"] = "1" if override else "0"
    return env


def _sync_data(env: dict[str, str]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = load_train_config(CONFIG_PATH)
    s3_fields = {
        "s3_bucket": cfg.s3_bucket, "s3_endpoint": cfg.s3_endpoint,
        "s3_access_key": bool(cfg.s3_access_key), "s3_secret_key": bool(cfg.s3_secret_key),
    }
    if not all(s3_fields.values()):
        missing = [k for k, v in s3_fields.items() if not v]
        print(f"[data sync] SKIPPED -- missing/empty config: {missing}.")
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


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=NODE_CPU,
    timeout=86400,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def train_single_node_ddp(mode: str = "pretrain", override: bool = False):
    """Single node, GPUS_PER_NODE GPUs, plain DDP -- see module docstring."""
    env = _build_env(mode, override)
    _sync_data(env)

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={GPUS_PER_NODE}",
        "-m", "training.new_train_ddp",
    ]
    print(f"[single node, {GPUS_PER_NODE} GPU(s), DDP] launching: {' '.join(cmd)}")
    subprocess.run(cmd, cwd="/app", env=env, check=True)
    return True


@app.local_entrypoint()
def main(mode: str = "pretrain", override: bool = False):
    train_single_node_ddp.remote(mode=mode, override=override)
