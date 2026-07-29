#!/usr/bin/env python3
"""Modal entrypoint: pre-download the configured pretrain/SFT .bin files
(train + eval) onto the persistent sabiyarn-data volume using a CPU-only
container -- no GPU requested at all.

Run this BEFORE training/modal_train.py: the same files land at the same
paths under the volume's DATA_DIR that training.new_train.py.new_train's
_sync_data() checks first (download_if_missing skips anything already
present with nonzero size), so a subsequent GPU training launch finds
everything already cached and goes straight into training instead of
burning billed GPU time waiting on downloads.

Usage:
    modal run data/prefetch_bins_modal.py --mode pretrain
    modal run data/prefetch_bins_modal.py --mode sft
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up MODAL_TOKEN_ID/SECRET, S3 keys for local `modal run`

import modal

# See training/modal_train.py for why: Modal re-imports this entrypoint script
# from a separate location when hydrating a function remotely, so
# Path(__file__).resolve().parents[1] is only correct when running locally.
_APP_MOUNT = Path("/app")
ROOT = _APP_MOUNT if (_APP_MOUNT / "training").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = "/data"
CONFIG_PATH = str(ROOT / "training" / "train_config.yaml")

# Deliberately lightweight -- training.load_config and training.s3_utils only
# need pyyaml/boto3/python-dotenv, nothing GPU-related, so this image never
# installs torch/transformers and the function below never requests a GPU.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3", "omegaconf", "pyyaml", "python-dotenv", "structlog")
    .add_local_dir(
        str(ROOT), remote_path="/app",
        ignore=[".git", "__pycache__", "*.pyc", "out/", ".env", ".venv", ".pytest_cache", ".claude"],
    )
)

app = modal.App("sabiyarn-prefetch-bins")
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)


@app.function(
    image=image, cpu=8, memory=8192, timeout=86400,
    volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_dotenv(__file__)],
)
def prefetch(mode: str = "pretrain") -> dict:
    os.environ["TRAIN_MODE"] = mode  # load_train_config reads this over yaml's training.mode

    from training.load_config import load_train_config
    from training.s3_utils import sync_training_files

    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = load_train_config(CONFIG_PATH)

    s3_fields = {
        "s3_bucket": cfg.s3_bucket, "s3_endpoint": cfg.s3_endpoint,
        "s3_access_key": bool(cfg.s3_access_key), "s3_secret_key": bool(cfg.s3_secret_key),
    }
    missing = [k for k, v in s3_fields.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing/empty S3 config: {missing}. Check that the s3-secret Modal secret "
            "actually has non-empty S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY values "
            "(`modal secret list` only shows names, not contents)."
        )

    print(f"prefetching mode={mode}: train={cfg.train_data_paths} eval={cfg.eval_data_path}")
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
    volume.commit()
    result = {"mode": mode, "train_data_paths": local_train, "eval_data_path": local_eval}
    print(f"done: {result}")
    return result


@app.local_entrypoint()
def main(mode: str = "pretrain"):
    prefetch.remote(mode=mode)
