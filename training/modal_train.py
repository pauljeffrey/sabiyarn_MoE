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
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up MODAL_TOKEN_ID/SECRET, S3/HF/wandb keys for local `modal run`

import modal
import modal.experimental
import yaml

from training.load_config import load_train_config
from training.s3_utils import sync_training_files

ROOT = Path(__file__).resolve().parents[1]
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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libaio-dev")  # required by DeepSpeed's async-io op
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.55.0",
        "accelerate>=0.34.0",
        "deepspeed>=0.15.0",
        "ninja",
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
    .add_local_dir(str(ROOT), remote_path="/app", ignore=[".git", "__pycache__", "*.pyc", "out/", ".env"])
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
    env["TRAIN_OUT_DIR"] = os.path.join(DATA_DIR, "checkpoints")
    env["OVERRIDE_DATA"] = "1" if override else "0"
    return env


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=NODE_CPU,
    timeout=86400,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret"), modal.Secret.from_name("s3-secret")],
)
@modal.experimental.clustered(size=NUM_NODES)
def train_cluster(mode: str = "pretrain", override: bool = False):
    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank
    master_addr = cluster_info.container_ips[0]

    env = _build_env(mode, override)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Each node independently syncs its own local copy of the training data from
    # S3 (no-op if already present, e.g. via the shared Modal Volume). Both the
    # eng and afr bins are always synced -- training mixes them per-batch via
    # data.sampling in train_config.yaml, it doesn't train on a single language.
    cfg = load_train_config(CONFIG_PATH)
    if cfg.s3_bucket and cfg.s3_access_key and cfg.s3_secret_key and cfg.s3_endpoint:
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
        if local_train:
            env["TRAIN_DATA_PATHS_LOCAL"] = ",".join(local_train)
        if local_eval:
            env["VAL_DATA_PATH"] = local_eval

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
    train_cluster.remote(mode=mode, override=override)
