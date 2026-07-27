#!/usr/bin/env python3
"""Modal entrypoint: push a training checkpoint from the persistent
sabiyarn-data volume (see training/modal_train.py) to S3-compatible storage.

By default, pushes the WHOLE latest run directory for --mode (weights for
every ckpt_N saved so far, resume_state/ optimizer+RNG state, and
trainer_state.json) -- the exact same directory structure
training/new_train.py's Trainer._resolve_resume_dir_local_or_s3 looks for
and downloads when comparing local vs. S3 recency, so a push here is always
fully resumable (weights + optimizer + step count) from any Modal
account/machine that shares this S3 bucket, not just a snapshot of the
latest weights. Pass --folder to push something else instead: a path
relative to the volume's <out_dir> root, e.g.
"20260722_143012_pretrain/ckpt_800" for one specific checkpoint's HF weights
only (NOT independently resumable -- no resume_state/trainer_state.json --
useful only if you specifically want to publish a single snapshot's weights).

Never touches the original training data objects in the bucket -- this
only ever writes new keys under --dest-prefix (default "checkpoints"),
mirroring the local path under the volume's DATA_DIR.

Usage:
    # Push the whole latest run directory for pretrain (weights + full resume state).
    modal run training/push_checkpoint_to_s3_modal.py --mode pretrain

    # Push one specific checkpoint's weights only (not independently resumable).
    modal run training/push_checkpoint_to_s3_modal.py \
        --folder 20260722_143012_pretrain/ckpt_800
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
    return run_dir


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
    from training.s3_utils import is_mutable_checkpoint_file, upload_folder

    local_dir = _resolve_checkpoint_dir(mode, folder)
    raw_cfg = _load_yaml_cfg()
    s3_cfg = _nested_list_dict(raw_cfg.get("s3", {}))

    rel_to_data = os.path.relpath(local_dir, DATA_DIR).replace(os.sep, "/")
    remote_prefix = f"{dest_prefix.rstrip('/')}/{rel_to_data}" if dest_prefix else rel_to_data
    bucket = s3_cfg.get("s3_bucket_name") or os.environ["S3_BUCKET"]

    print(f"pushing {local_dir} -> s3://{bucket}/{remote_prefix}/...")
    # trainer_state.json and resume_state/ are rewritten in place on every
    # save -- always re-upload them regardless of --override, or a later
    # push would silently skip them (their S3 key never changes) and leave
    # the run's actual latest iter_num/optimizer state frozen at whatever
    # existed the first time this run_dir was pushed, even as ckpt_N/
    # folders for newer iters keep getting added correctly alongside them.
    uploaded = upload_folder(
        local_dir, remote_prefix,
        bucket=bucket,
        endpoint=s3_cfg.get("s3_endpoint") or os.environ["S3_ENDPOINT"],
        access_key=os.environ["S3_ACCESS_KEY_ID"],
        secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
        prefix=str(s3_cfg.get("prefix", "")),
        override=override,
        force_override_paths=is_mutable_checkpoint_file,
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


@app.function(
    image=image, cpu=2, memory=1024, timeout=300,
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def reset_resume_state_to_checkpoint(
    run_dir: str,
    ckpt_name: str,
    iter_num: int,
    best_val_loss: float = 1e9,
    dest_prefix: str = "checkpoints",
) -> dict:
    """Corrects a run_dir's S3 state to warm-start from a specific
    checkpoint whose own resume_state was never pushed -- e.g. ckpt_name
    came from a different training process/account than whichever last
    pushed trainer_state.json for this run_dir, so the resume_state
    currently on S3 has mismatched optimizer/RNG momentum for it.

    Overwrites trainer_state.json to point at ckpt_name/iter_num/
    best_val_loss, and DELETES resume_state/ entirely, so the next training
    launch loads ckpt_name's weights (if init_from="resume") and correctly
    resumes iter_num for the LR/sampling schedule, but starts with a FRESH
    optimizer rather than silently reusing a different checkpoint's
    momentum under a new iter_num label. Irreversible for resume_state/ --
    only run this when you're sure the existing resume_state doesn't
    actually match ckpt_name.
    """
    from training.s3_utils import delete_prefix, write_remote_json

    raw_cfg = _load_yaml_cfg()
    s3_cfg = _nested_list_dict(raw_cfg.get("s3", {}))
    bucket = s3_cfg.get("s3_bucket_name") or os.environ["S3_BUCKET"]
    endpoint = s3_cfg.get("s3_endpoint") or os.environ["S3_ENDPOINT"]
    access_key = os.environ["S3_ACCESS_KEY_ID"]
    secret_key = os.environ["S3_SECRET_ACCESS_KEY"]
    bucket_prefix = str(s3_cfg.get("prefix", ""))

    out_dir_name = str((raw_cfg.get("training", {}) or {}).get("out_dir", "checkpoints"))
    run_prefix = f"{dest_prefix.rstrip('/')}/{out_dir_name}/{run_dir.strip('/')}"

    def _full_key(rel: str) -> str:
        remote_key = f"{run_prefix}/{rel}"
        return f"{bucket_prefix.rstrip('/')}/{remote_key.lstrip('/')}" if bucket_prefix else remote_key.lstrip("/")

    state = {
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "latest_ckpt": f"/data/{out_dir_name}/{run_dir}/{ckpt_name}",
        "last_hf_push_loss": None,
        "last_hf_push_iter": 0,
    }
    trainer_state_key = _full_key("trainer_state.json")
    print(f"writing corrected trainer_state.json -> s3://{bucket}/{trainer_state_key}: {state}")
    write_remote_json(trainer_state_key, state, bucket=bucket, endpoint=endpoint, access_key=access_key, secret_key=secret_key)

    deleted = delete_prefix(
        f"{run_prefix}/resume_state", bucket=bucket, endpoint=endpoint,
        access_key=access_key, secret_key=secret_key, prefix=bucket_prefix,
    )
    print(f"deleted {len(deleted)} stale resume_state object(s)")

    return {"trainer_state_key": trainer_state_key, "state": state, "deleted_resume_state_keys": deleted}


@app.local_entrypoint()
def reset_main(
    run_dir: str,
    ckpt_name: str,
    iter_num: int,
    best_val_loss: float = 1e9,
    dest_prefix: str = "checkpoints",
):
    reset_resume_state_to_checkpoint.remote(
        run_dir=run_dir, ckpt_name=ckpt_name, iter_num=iter_num,
        best_val_loss=best_val_loss, dest_prefix=dest_prefix,
    )


@app.function(
    image=image, cpu=2, memory=1024, timeout=300,
    volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_dotenv(__file__)],
)
def clear_local_checkpoints(mode: str = "pretrain", run_dir: str = "") -> dict:
    """Deletes local checkpoint run director(ies) on THIS Modal account's
    volume under out_dir -- for forcing the next training launch on this
    account to fall back to S3 (via Trainer._resolve_resume_dir_local_or_s3)
    instead of resuming from local state known to be stale/wrong, e.g.
    after correcting S3 with reset_resume_state_to_checkpoint. Irreversible.

    run_dir: delete just this one run directory name (relative to out_dir).
    Leave blank to delete every `_{mode}`-suffixed run directory found
    under out_dir on this account's volume.
    """
    import shutil

    raw_cfg = _load_yaml_cfg()
    out_dir_name = str((raw_cfg.get("training", {}) or {}).get("out_dir", "checkpoints"))
    out_dir = os.path.join(DATA_DIR, out_dir_name)

    if not os.path.isdir(out_dir):
        return {"out_dir": out_dir, "deleted": []}

    if run_dir:
        targets = [run_dir]
    else:
        suffix = f"_{mode}"
        targets = [name for name in os.listdir(out_dir) if name.endswith(suffix)]

    deleted = []
    for name in targets:
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            deleted.append(path)
            print(f"deleted {path}")

    volume.commit()
    return {"out_dir": out_dir, "deleted": deleted}


@app.local_entrypoint()
def clear_main(mode: str = "pretrain", run_dir: str = ""):
    result = clear_local_checkpoints.remote(mode=mode, run_dir=run_dir)
    print(result)
