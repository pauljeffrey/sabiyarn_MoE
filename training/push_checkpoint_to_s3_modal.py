#!/usr/bin/env python3
"""Modal entrypoint: push a training checkpoint from the persistent
sabiyarn-data volume (see training/modal_train.py) to S3-compatible storage.

By default, pushes the LATEST checkpoint for --mode: the folder
new_train.py's own trainer_state.json (under the most recently modified
run directory for that mode) currently points at via its "latest_ckpt"
field -- i.e. exactly what a fresh `init_from: "resume"` run would load
weights from. Pass --folder to push something else instead: a path
relative to the volume's <out_dir> root, e.g. "20260722_143012_pretrain"
for a whole run directory (HF weights + resume_state, i.e. a full
resumable-training backup) or "20260722_143012_pretrain/ckpt_800" for one
specific checkpoint's HF weights only.

Never touches the original training data objects in the bucket -- this
only ever writes new keys under --dest-prefix (default "checkpoints"),
mirroring the local path under the volume's DATA_DIR.

Usage:
    # Push whatever new_train.py currently considers "latest" for pretrain.
    modal run training/push_checkpoint_to_s3_modal.py --mode pretrain

    # Push one specific checkpoint folder (HF weights only).
    modal run training/push_checkpoint_to_s3_modal.py \
        --folder 20260722_143012_pretrain/ckpt_800

    # Push a whole run directory (weights + optimizer/resume state).
    modal run training/push_checkpoint_to_s3_modal.py \
        --folder 20260722_143012_pretrain
"""

from __future__ import annotations

import json
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

# Deliberately NOT depending on torch/transformers/accelerate (training.new_train
# pulls those in) -- this script only needs to scan directories and talk to S3,
# so _find_latest_run_dir's tiny scan is duplicated below rather than imported.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3", "pyyaml", "python-dotenv", "structlog")
    .add_local_dir(
        str(ROOT), remote_path="/app",
        ignore=[".git", "__pycache__", "*.pyc", "out/", ".env", ".venv", ".pytest_cache", ".claude"],
    )
)

app = modal.App("sabiyarn-push-checkpoint")
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)


def _nested_list_dict(items) -> dict:
    """Merge a plain dict, or YAML list-of-mapping sections like
    `[{k: v}, {k2: v2}, ...]`, into one dict."""
    if items is None:
        return {}
    if isinstance(items, dict):
        return dict(items)
    out: dict = {}
    for item in items or []:
        if isinstance(item, dict):
            out.update(item)
    return out


def _load_yaml_cfg() -> dict:
    import yaml

    config_path = str(ROOT / "training" / "train_config.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _find_latest_run_dir(out_dir: str, mode: str) -> str | None:
    """Same scan as training/new_train.py's _find_latest_run_dir -- kept as an
    independent copy here so this script stays free of torch/transformers/
    accelerate, which that module imports at load time."""
    if not os.path.isdir(out_dir):
        return None
    suffix = f"_{mode}"
    candidates = []
    for name in os.listdir(out_dir):
        if not name.endswith(suffix):
            continue
        full = os.path.join(out_dir, name)
        if os.path.isfile(os.path.join(full, "trainer_state.json")):
            candidates.append(full)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _resolve_checkpoint_dir(mode: str, folder: str) -> str:
    """Absolute local path (on the mounted volume) of the folder to push."""
    raw_cfg = _load_yaml_cfg()
    out_dir_name = str((raw_cfg.get("training", {}) or {}).get("out_dir", "checkpoints"))
    out_dir = os.path.join(DATA_DIR, out_dir_name)

    if folder:
        path = folder if os.path.isabs(folder) else os.path.join(out_dir, folder)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"folder not found on volume: {path}")
        return path

    run_dir = _find_latest_run_dir(out_dir, mode)
    if run_dir is None:
        raise FileNotFoundError(f"no checkpoint run directory found under {out_dir} for mode={mode!r}")

    meta_path = os.path.join(run_dir, "trainer_state.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    latest_ckpt = meta.get("latest_ckpt")
    if not latest_ckpt or not os.path.isdir(latest_ckpt):
        raise FileNotFoundError(
            f"trainer_state.json at {meta_path} has no valid latest_ckpt "
            f"(got {latest_ckpt!r}) -- pass --folder explicitly instead"
        )
    return latest_ckpt


@app.function(
    image=image, cpu=8, memory=8192, timeout=86400,
    volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_dotenv(__file__)],
)
def push_checkpoint(
    mode: str = "pretrain",
    folder: str = "",
    dest_prefix: str = "checkpoints",
    override: bool = False,
) -> dict:
    from training.s3_utils import upload_folder

    local_dir = _resolve_checkpoint_dir(mode, folder)
    raw_cfg = _load_yaml_cfg()
    s3_cfg = _nested_list_dict(raw_cfg.get("s3", {}))

    rel_to_data = os.path.relpath(local_dir, DATA_DIR).replace(os.sep, "/")
    remote_prefix = f"{dest_prefix.rstrip('/')}/{rel_to_data}" if dest_prefix else rel_to_data
    bucket = s3_cfg.get("s3_bucket_name") or os.environ["S3_BUCKET"]

    print(f"pushing {local_dir} -> s3://{bucket}/{remote_prefix}/...")
    uploaded = upload_folder(
        local_dir, remote_prefix,
        bucket=bucket,
        endpoint=s3_cfg.get("s3_endpoint") or os.environ["S3_ENDPOINT"],
        access_key=os.environ["S3_ACCESS_KEY_ID"],
        secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
        prefix=str(s3_cfg.get("prefix", "")),
        override=override,
    )
    print(f"uploaded/verified {len(uploaded)} file(s)")
    return {
        "local_dir": local_dir,
        "bucket": bucket,
        "remote_prefix": remote_prefix,
        "file_count": len(uploaded),
        "keys": uploaded,
    }


@app.local_entrypoint()
def main(
    mode: str = "pretrain",
    folder: str = "",
    dest_prefix: str = "checkpoints",
    override: bool = False,
):
    push_checkpoint.remote(mode=mode, folder=folder, dest_prefix=dest_prefix, override=override)
