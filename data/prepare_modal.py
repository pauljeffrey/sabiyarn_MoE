#!/usr/bin/env python3
"""Modal entrypoint for preparing S3-backed training data into local memmap bins."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up MODAL_TOKEN_ID/SECRET, S3/HF keys for local `modal run`

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "numpy",
        "omegaconf",
        "pyyaml",
        "python-dotenv",
        "structlog",
        "tqdm",
        "lmdb",
        "boto3",
    )
    .add_local_dir(str(ROOT), remote_path="/app", ignore=[".git", "__pycache__", "*.pyc", "out/", ".env"])
)

app = modal.App("sabiyarn-data-prepare")
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)


def _nested_list_dict(items) -> dict:
    """Merge YAML list-of-mapping sections like `[{k: v}, {k2: v2}, ...]` into one dict."""
    out: dict = {}
    for item in items or []:
        if isinstance(item, dict):
            out.update(item)
    return out


def _resolve_paths(mode: str, data_type: str, override: bool) -> tuple[str, str, dict]:
    import yaml

    config_path = str(ROOT / "training" / "train_config.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    data_cfg = config.get("data", {})
    mode_cfg = data_cfg.get(mode, {})
    if isinstance(mode_cfg, list):
        mode_cfg = _nested_list_dict(mode_cfg)

    if data_type == "eng":
        dataset_names = data_cfg.get(f"{mode}_datasets", {}).get("english", [])
        train_path = mode_cfg.get("eng_train_data_path", "")
        eval_path = mode_cfg.get("eval_data_path", "")
    elif data_type == "african":
        dataset_names = data_cfg.get(f"{mode}_datasets", {}).get("african", [])
        train_path = mode_cfg.get("afr_train_data_path", "")
        eval_path = mode_cfg.get("eval_data_path", "")
    else:
        raise ValueError("data_type must be 'eng' or 'african'")

    if override:
        train_path = os.path.join(DATA_DIR, os.path.basename(train_path))
        eval_path = os.path.join(DATA_DIR, os.path.basename(eval_path))
    else:
        train_path = os.path.join(DATA_DIR, os.path.basename(train_path)) if train_path else os.path.join(DATA_DIR, f"{mode}_{data_type}.bin")
        eval_path = os.path.join(DATA_DIR, os.path.basename(eval_path)) if eval_path else os.path.join(DATA_DIR, f"{mode}_{data_type}_eval.bin")

    os.environ["TRAIN_MODE"] = mode
    os.environ["TRAIN_CONFIG_PATH"] = config_path
    os.environ["TRAIN_DATA_PATH"] = train_path
    os.environ["VAL_DATA_PATH"] = eval_path
    os.environ["PREP_STATE_PATH"] = os.path.join(DATA_DIR, f"prep_state_{mode}_{data_type}.json")
    os.environ["OVERRIDE_DATA"] = "1" if override else "0"
    return train_path, eval_path, {"datasets": dataset_names, "mode": mode, "data_type": data_type}


@app.function(image=image, cpu=8, timeout=86400, volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_name("hf-secret")])
def prepare_data(mode: str = "pretrain", data_type: str = "eng", override: bool = False):
    import yaml
    from data.prepare import run

    train_path, eval_path, info = _resolve_paths(mode, data_type, override)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)

    with open(os.environ["TRAIN_CONFIG_PATH"], "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    datasets = config.get("data", {}).get(f"{mode}_datasets", {}).get("english" if data_type == "eng" else "african", [])
    run(datasets_list=datasets, registry_cache=os.path.join(DATA_DIR, f"registry_{mode}_{data_type}.lmdb"))
    print(f"prepared {mode}/{data_type} -> train={train_path} eval={eval_path}")
    volume.commit()


@app.local_entrypoint()
def main(mode: str = "pretrain", data_type: str = "eng", override: bool = False):
    prepare_data.remote(mode=mode, data_type=data_type, override=override)


@app.function(image=image, cpu=4, timeout=3600, volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_name("s3-secret")])
def count_tag(mode: str = "pretrain", data_type: str = "african", tag: str = "<twi>") -> int:
    """Count occurrences of a tag's token id in the tokenized data.<mode>.<data_type>_train_data_path bin."""
    import yaml

    from data.prepare import count_tag_occurrences
    from training.s3_utils import download_if_missing

    config_path = str(ROOT / "training" / "train_config.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    mode_cfg = config.get("data", {}).get(mode, [])
    if isinstance(mode_cfg, list):
        mode_cfg = _nested_list_dict(mode_cfg)
    key = "afr_train_data_path" if data_type == "african" else "eng_train_data_path"
    remote_path = mode_cfg.get(key, "")
    if not remote_path:
        raise ValueError(f"No {key} configured for data.{mode}")

    s3_cfg = _nested_list_dict(config.get("s3", []))
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, os.path.basename(remote_path))
    download_if_missing(
        remote_path,
        local_path,
        bucket=s3_cfg.get("s3_bucket_name") or os.environ["S3_BUCKET"],
        endpoint=s3_cfg.get("s3_endpoint") or os.environ["S3_ENDPOINT"],
        access_key=os.environ["S3_ACCESS_KEY_ID"],
        secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
        prefix=str(s3_cfg.get("prefix", "")),
    )

    count = count_tag_occurrences(local_path, tag)
    print(f"tag_count:{tag}:{count} (path={local_path})")
    return count


@app.local_entrypoint()
def count_tag_main(mode: str = "pretrain", data_type: str = "african", tag: str = "<twi>"):
    count_tag.remote(mode=mode, data_type=data_type, tag=tag)
