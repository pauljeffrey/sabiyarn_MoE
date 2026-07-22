#!/usr/bin/env python3
"""Modal entrypoint: remove excess <translate>...<twi>...</s> segments from a
tokenized .bin file, keeping only the first `keep_count` and deleting the
rest -- see data/clean_translate_segments.py for the exact algorithm and its
test suite (tests/test_clean_translate_segments.py).

Does NOT touch the original S3 object -- writes the cleaned result to a new
local file and a new S3 key, so you can compare/verify before ever pointing
train_config.yaml at it or overwriting the original.

Usage:
    # Fast, no rewrite -- just report how many segments would be deleted.
    modal run data/clean_translate_segments_modal.py::report_main \
        --data-type african --marker-tag "<twi>" --keep-count 100000

    # Full run: downloads, cleans, uploads to a new S3 key.
    modal run data/clean_translate_segments_modal.py::clean_main \
        --data-type african --marker-tag "<twi>" --keep-count 100000 \
        --out-s3-key datasets/training_cleaned.bin
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up MODAL_TOKEN_ID/SECRET, S3/HF keys for local `modal run`

import modal

# See training/modal_train.py for why: Modal re-imports this entrypoint script
# from a separate location when hydrating a function remotely, so
# Path(__file__).resolve().parents[1] is only correct when running locally.
_APP_MOUNT = Path("/app")
ROOT = _APP_MOUNT if (_APP_MOUNT / "training").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = "/data"
CPU_COUNT = 32
MEMORY_MB = 65536  # 64 GB -- generous for ~24B-token files; the chunked
# design (see clean_translate_segments.py) never materializes the whole file
# in RAM at once, so this is comfortable headroom, not a hard requirement.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "huggingface_hub",
        "numpy",
        "omegaconf",
        "pyyaml",
        "python-dotenv",
        "structlog",
        "boto3",
    )
    .add_local_dir(
        str(ROOT), remote_path="/app",
        ignore=[".git", "__pycache__", "*.pyc", "out/", ".env", ".venv", ".pytest_cache", ".claude"],
    )
)

app = modal.App("sabiyarn-clean-translate-segments")
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


def _resolve_remote_path(config: dict, mode: str, data_type: str) -> str:
    mode_cfg = config.get("data", {}).get(mode, [])
    if isinstance(mode_cfg, list):
        mode_cfg = _nested_list_dict(mode_cfg)
    key = "afr_train_data_path" if data_type == "african" else "eng_train_data_path"
    remote_path = mode_cfg.get(key, "")
    if not remote_path:
        raise ValueError(f"No {key} configured for data.{mode}")
    return remote_path


def _fetch_input(mode: str, data_type: str) -> str:
    """Downloads (if not already cached on the Volume) the configured bin
    file and returns its local path."""
    import yaml

    from training.s3_utils import download_if_missing

    config_path = str(ROOT / "training" / "train_config.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    remote_path = _resolve_remote_path(config, mode, data_type)
    s3_cfg = _nested_list_dict(config.get("s3", {}))
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
    return local_path


def _resolve_tag_ids(translate_tag: str, marker_tag: str, eos_tag: str) -> tuple[int, int, int]:
    from data.prepare import _tokenizer_cfg, get_tokenizer_and_eot

    tok_name = _tokenizer_cfg.get("name", "BeardedMonster/SabiYarn-32k")
    enc, _ = get_tokenizer_and_eot(tok_name)

    def _single_id(tag: str) -> int:
        ids = enc.encode(tag, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"{tag!r} tokenizes to {len(ids)} ids ({ids}), not 1 -- this script assumes "
                "single-token tags. Adjust clean_translate_segments.py to handle multi-token "
                "tags before proceeding."
            )
        return ids[0]

    return _single_id(translate_tag), _single_id(marker_tag), _single_id(eos_tag)


@app.function(image=image, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=86400, volumes={DATA_DIR: volume})
def report_stats(
    mode: str = "pretrain",
    data_type: str = "african",
    translate_tag: str = "<translate>",
    marker_tag: str = "<twi>",
    eos_tag: str = "</s>",
    keep_count: int = 100_000,
) -> dict:
    """Scans and reports segment counts WITHOUT writing any output file --
    a fast sanity check before committing to the full (slower) clean run."""
    import numpy as np

    from data.clean_translate_segments import compute_delete_ranges, find_token_positions

    local_path = _fetch_input(mode, data_type)
    translate_id, marker_id, eos_id = _resolve_tag_ids(translate_tag, marker_tag, eos_tag)
    print(f"tag ids: {translate_tag}={translate_id} {marker_tag}={marker_id} {eos_tag}={eos_id}")

    total_len = int(np.memmap(local_path, dtype=np.uint16, mode="r").shape[0])
    print(f"input: {local_path} total_tokens={total_len}")

    translate_pos, marker_pos, eos_pos = find_token_positions(
        local_path, total_len, (translate_id, marker_id, eos_id), num_workers=CPU_COUNT
    )
    _, stats = compute_delete_ranges(translate_pos, marker_pos, eos_pos, keep_count)
    print(f"stats: {stats}")
    return stats


@app.local_entrypoint()
def report_main(
    mode: str = "pretrain",
    data_type: str = "african",
    translate_tag: str = "<translate>",
    marker_tag: str = "<twi>",
    eos_tag: str = "</s>",
    keep_count: int = 100_000,
):
    report_stats.remote(
        mode=mode, data_type=data_type, translate_tag=translate_tag,
        marker_tag=marker_tag, eos_tag=eos_tag, keep_count=keep_count,
    )


@app.function(image=image, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=86400, volumes={DATA_DIR: volume})
def clean_bin_file(
    mode: str = "pretrain",
    data_type: str = "african",
    translate_tag: str = "<translate>",
    marker_tag: str = "<twi>",
    eos_tag: str = "</s>",
    keep_count: int = 100_000,
    out_s3_key: str = "",
    upload: bool = True,
) -> dict:
    """Downloads the configured bin file, removes all but the first
    keep_count <translate>...<marker>...</s> segments, and writes the result
    to a NEW local file + (if upload=True) a NEW S3 key. Never modifies or
    overwrites the original file/S3 object."""
    import yaml

    from data.clean_translate_segments import clean_translate_segments
    from training.s3_utils import upload_if_absent

    local_in = _fetch_input(mode, data_type)
    translate_id, marker_id, eos_id = _resolve_tag_ids(translate_tag, marker_tag, eos_tag)
    print(f"tag ids: {translate_tag}={translate_id} {marker_tag}={marker_id} {eos_tag}={eos_id}")

    base, ext = os.path.splitext(os.path.basename(local_in))
    local_out = os.path.join(DATA_DIR, f"{base}_cleaned{ext}")

    stats = clean_translate_segments(
        local_in, local_out, translate_id, marker_id, eos_id,
        keep_count=keep_count, num_workers=CPU_COUNT,
    )
    print(f"cleaned: {local_in} -> {local_out}")
    print(f"stats: {stats}")
    volume.commit()

    if upload:
        config_path = str(ROOT / "training" / "train_config.yaml")
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        s3_cfg = _nested_list_dict(config.get("s3", {}))
        remote_in = _resolve_remote_path(config, mode, data_type)
        key = out_s3_key or f"{os.path.splitext(remote_in)[0]}_cleaned{os.path.splitext(remote_in)[1]}"
        uploaded_key = upload_if_absent(
            local_out, key,
            bucket=s3_cfg.get("s3_bucket_name") or os.environ["S3_BUCKET"],
            endpoint=s3_cfg.get("s3_endpoint") or os.environ["S3_ENDPOINT"],
            access_key=os.environ["S3_ACCESS_KEY_ID"],
            secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
            prefix=str(s3_cfg.get("prefix", "")),
        )
        stats["uploaded_s3_key"] = uploaded_key
        print(f"uploaded to s3 key: {uploaded_key}")

    return stats


@app.local_entrypoint()
def clean_main(
    mode: str = "pretrain",
    data_type: str = "african",
    translate_tag: str = "<translate>",
    marker_tag: str = "<twi>",
    eos_tag: str = "</s>",
    keep_count: int = 100_000,
    out_s3_key: str = "",
    upload: bool = True,
):
    clean_bin_file.remote(
        mode=mode, data_type=data_type, translate_tag=translate_tag,
        marker_tag=marker_tag, eos_tag=eos_tag, keep_count=keep_count,
        out_s3_key=out_s3_key, upload=upload,
    )
