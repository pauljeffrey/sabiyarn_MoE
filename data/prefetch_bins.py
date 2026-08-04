#!/usr/bin/env python3
"""Plain, non-Modal counterpart to data/prefetch_bins_modal.py: pre-download
the configured pretrain/SFT .bin files (train + eval) from S3 onto local
disk -- for vast.ai or any other bare GPU box, where there's no Modal
Volume and no separate data-sync step before training starts.

training.new_train's Trainer._setup_data only CHECKS that the data files
configured in train_config.yaml already exist locally -- it never
downloads them (that's Modal's job, via modal_train.py's _sync_data).
Run this once before training on a bare box, then use the env vars it
prints so the training process picks up exactly what got downloaded.

Usage:
    python -m data.prefetch_bins --mode pretrain
    python -m data.prefetch_bins --mode sft
    python -m data.prefetch_bins --mode pretrain --local-dir /data/bins
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = str(ROOT / "training" / "train_config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="pretrain", choices=["pretrain", "sft"])
    parser.add_argument(
        "--local-dir", default=str(ROOT / "data" / "bins"),
        help="Where to download the .bin files (default: ./data/bins). "
             "Downloads are flattened to basenames here, same as Modal's DATA_DIR -- "
             "use the printed env vars below rather than assuming train_config.yaml's "
             "own relative paths point here.",
    )
    args = parser.parse_args()

    os.environ["TRAIN_MODE"] = args.mode  # load_train_config reads this over yaml's training.mode

    from training.load_config import load_train_config
    from training.s3_utils import sync_training_files

    os.makedirs(args.local_dir, exist_ok=True)
    cfg = load_train_config(CONFIG_PATH)

    s3_fields = {
        "s3_bucket": cfg.s3_bucket, "s3_endpoint": cfg.s3_endpoint,
        "s3_access_key": bool(cfg.s3_access_key), "s3_secret_key": bool(cfg.s3_secret_key),
    }
    missing = [k for k, v in s3_fields.items() if not v]
    if missing:
        raise SystemExit(
            f"Missing/empty S3 config: {missing}. Set S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY "
            "in your .env (see HOW_TO_RUN.md section 0)."
        )

    print(f"prefetching mode={args.mode}: train={cfg.train_data_paths} eval={cfg.eval_data_path}")
    local_train, local_eval = sync_training_files(
        cfg.train_data_paths,
        args.local_dir,
        bucket=cfg.s3_bucket,
        endpoint=cfg.s3_endpoint,
        access_key=cfg.s3_access_key,
        secret_key=cfg.s3_secret_key,
        prefix=cfg.s3_prefix,
        eval_path=cfg.eval_data_path,
    )
    print(f"done: train={local_train} eval={local_eval}")
    print()
    print("Add these lines to your .env before running training/new_train.py (or")
    print("training/new_train_ddp.py) -- load_config.py reads them over train_config.yaml's")
    print("own (relative) data paths, same override modal_train.py uses internally:")
    print()
    print(f"TRAIN_DATA_PATHS_LOCAL={','.join(local_train)}")
    print(f"VAL_DATA_PATH={local_eval}")


if __name__ == "__main__":
    main()
