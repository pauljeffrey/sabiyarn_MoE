#!/usr/bin/env python3
"""Modal entrypoint: diagnose pretrain-mode label masking
(training/utils.py's process_labels_optimized, used via
training/label_masking.py's process_pretrain_labels) against REAL sampled
windows from a configured pretraining .bin file.

The pretrain corpus mixes monolingual documents (no tags) with tagged task
documents -- translation (<translate> + language tag), sentiment
(<classify> ... <sentiment>), topic (<classify> ... <topic>), NER, etc. --
all packed together and separated by <|end_of_text|>. process_labels_optimized
finds every action/prompting tag anywhere in a window and masks between
consecutive PAIRS of them, with no awareness of document boundaries. This
script checks, on real data, whether that pairing ever spans across a
document boundary (</s>) -- which would mean a document (e.g. a tagless
monolingual one, or any other unrelated document) got swept into a masked
span that was only ever meant to bound a different document's prompt/
response split.

Read-only: only downloads (if not already cached) and analyzes samples,
never writes or modifies anything in S3/the bin file.

Usage:
    modal run data/inspect_label_masking_modal.py \
        --mode pretrain --data-type african --num-windows 200

    modal run data/inspect_label_masking_modal.py \
        --mode pretrain --data-type english --num-windows 200
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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0", "transformers>=4.55.0", "numpy", "omegaconf", "pyyaml",
        "boto3", "python-dotenv", "structlog", "huggingface_hub",
    )
    .add_local_dir(
        str(ROOT), remote_path="/app",
        ignore=[".git", "__pycache__", "*.pyc", "out/", ".env", ".venv", ".pytest_cache", ".claude"],
    )
)

app = modal.App("sabiyarn-inspect-label-masking")
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


@app.function(
    image=image, cpu=4, memory=8192, timeout=3600,
    volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_dotenv(__file__)],
)
def inspect_masking(
    mode: str = "pretrain",
    data_type: str = "african",
    num_windows: int = 200,
    window_size: int = 4096,
    seed: int = 0,
) -> dict:
    import numpy as np
    import torch

    from training.constant_tokens import action_tokens, end_of_text_token, prompting_tokens
    from training.utils import process_labels_optimized

    local_path = _fetch_input(mode, data_type)
    data = np.memmap(local_path, dtype=np.uint16, mode="r")
    total_len = len(data)
    print(f"input: {local_path} total_tokens={total_len} eos_token={end_of_text_token}")

    rng = np.random.default_rng(seed)
    all_tag_ids = {int(t) for t in action_tokens} | {int(t) for t in prompting_tokens}

    overall_masked = 0
    overall_total = 0
    windows_with_any_mask = 0
    total_masked_runs = 0
    cross_doc_masked_runs = 0
    cross_doc_masked_tokens = 0
    tag_hits_total = 0
    example_cross_doc = []

    starts = rng.integers(0, max(1, total_len - window_size - 1), size=num_windows)
    for s in starts:
        s = int(s)
        window = data[s: s + window_size].astype(np.int64)
        tokens = torch.from_numpy(window)
        window_list = window.tolist()

        eos_positions = [i for i, t in enumerate(window_list) if t == end_of_text_token]
        tag_positions = [i for i, t in enumerate(window_list) if t in all_tag_ids]
        tag_hits_total += len(tag_positions)

        result = process_labels_optimized(tokens.clone(), mask=-100)
        masked_bool = (result == -100).numpy()
        overall_masked += int(masked_bool.sum())
        overall_total += window_size
        if masked_bool.any():
            windows_with_any_mask += 1

        # Contiguous masked runs.
        runs = []
        i = 0
        while i < window_size:
            if masked_bool[i]:
                j = i
                while j < window_size and masked_bool[j]:
                    j += 1
                runs.append((i, j - 1))
                i = j
            else:
                i += 1

        for (rs, re) in runs:
            total_masked_runs += 1
            crosses = any(rs < e < re for e in eos_positions)
            if crosses:
                cross_doc_masked_runs += 1
                cross_doc_masked_tokens += (re - rs + 1)
                if len(example_cross_doc) < 5:
                    example_cross_doc.append({
                        "window_start": s, "run": [rs, re],
                        "eos_inside": [e for e in eos_positions if rs < e < re],
                    })

    stats = {
        "num_windows": num_windows,
        "window_size": window_size,
        "overall_masked_fraction": overall_masked / max(1, overall_total),
        "windows_with_any_mask_fraction": windows_with_any_mask / num_windows,
        "avg_tag_hits_per_window": tag_hits_total / num_windows,
        "total_masked_runs": total_masked_runs,
        "cross_document_masked_runs": cross_doc_masked_runs,
        "cross_document_masked_run_fraction": cross_doc_masked_runs / max(1, total_masked_runs),
        "cross_document_masked_tokens": cross_doc_masked_tokens,
        "cross_document_masked_token_fraction_of_all_masked": cross_doc_masked_tokens / max(1, overall_masked),
        "example_cross_document_runs": example_cross_doc,
    }
    print(f"stats: {stats}")
    return stats


@app.local_entrypoint()
def main(
    mode: str = "pretrain",
    data_type: str = "african",
    num_windows: int = 200,
    window_size: int = 4096,
    seed: int = 0,
):
    inspect_masking.remote(
        mode=mode, data_type=data_type, num_windows=num_windows,
        window_size=window_size, seed=seed,
    )
