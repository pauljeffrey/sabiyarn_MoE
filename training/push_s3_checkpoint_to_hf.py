#!/usr/bin/env python3
"""Download a checkpoint's model weights from S3 and push them to the HF
checkpoint repo (training.hf_chkpt_path in train_config.yaml).

Reads the run directories training/new_train.py pushes to S3 under
checkpoints/<training.out_dir>/<run_dir>/ and picks one of the two
checkpoints tracked there (see training.resume_from in train_config.yaml):
  - latest (default): trainer_state.json's latest_ckpt, i.e. ckpt_<iter_num>/
  - best (--best):    trainer_state.json's best_ckpt, i.e. ckpt_best/
Only that one weights folder is downloaded -- never resume_state/ (optimizer
/RNG state), which HF has no use for. Its contents are uploaded to the repo
root, the same way Trainer._push_checkpoint_to_hf does during training, so
the repo's trust_remote_code modeling.py etc. are left untouched.

The run dir defaults to the most recent one on S3 for training.mode (same
lookup the trainer's S3 auto-resume uses); pass --run-dir to pick another.

Credentials (from the environment / .env, never the config file):
    S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and a Hub WRITE token as
    HF_WRITE_TOKEN (or HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HF_API_KEY).

Usage -- plain Python (vast.ai, bare GPU box, laptop):
    python training/push_s3_checkpoint_to_hf.py                 # latest checkpoint
    python training/push_s3_checkpoint_to_hf.py --best          # best checkpoint
    python training/push_s3_checkpoint_to_hf.py --dry-run       # resolve + print, push nothing
    python training/push_s3_checkpoint_to_hf.py --best \
        --run-dir 20260722_143012_pretrain --repo Aletheia-ng/other-repo

Usage -- Modal (runs in a CPU container, nothing downloaded to your machine):
    modal run training/push_s3_checkpoint_to_hf.py
    modal run training/push_s3_checkpoint_to_hf.py --best
    modal run training/push_s3_checkpoint_to_hf.py --best --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# See training/modal_train.py for why: Modal re-imports this entrypoint script
# from a separate location when hydrating a function remotely, so
# Path(__file__).resolve().parents[1] is only correct when running locally.
_APP_MOUNT = Path("/app")
ROOT = _APP_MOUNT if (_APP_MOUNT / "training").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



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


def _load_yaml_cfg(config_path: str = "") -> dict:
    import yaml

    path = config_path or os.environ.get("TRAIN_CONFIG_PATH") or str(ROOT / "training" / "train_config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_WRITE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_API_KEY")  # last resort: the read token in some setups
    )


def _ckpt_iter(name: str) -> int:
    suffix = name[len("ckpt_"):]
    return int(suffix) if suffix.isdigit() else -1


def _resolve_ckpt_name(run_prefix: str, meta: dict, best: bool, s3_kwargs: dict) -> str:
    """Name of the weights folder inside run_prefix to push. trainer_state.json
    stores latest_ckpt/best_ckpt as absolute paths from whichever machine
    saved them, so only their basenames are meaningful here."""
    from training.s3_utils import list_immediate_subfolders

    folders = [
        f.rstrip("/").rsplit("/", 1)[-1]
        for f in list_immediate_subfolders(run_prefix, prefix="", **s3_kwargs)
    ]
    recorded = meta.get("best_ckpt" if best else "latest_ckpt")
    if recorded:
        name = os.path.basename(recorded.rstrip("/"))
        if name in folders:
            return name
        print(f"warning: trainer_state.json points at {name!r}, which isn't on S3 -- falling back")

    if best:
        if "ckpt_best" in folders:
            return "ckpt_best"
        raise FileNotFoundError(f"no ckpt_best/ under s3 run dir {run_prefix} -- nothing to push with --best")

    numbered = [f for f in folders if _ckpt_iter(f) >= 0]
    if not numbered:
        raise FileNotFoundError(f"no ckpt_<iter>/ folder under s3 run dir {run_prefix}")
    return max(numbered, key=_ckpt_iter)


def push_s3_checkpoint_to_hf(
    best: bool = False,
    run_dir: str = "",
    mode: str = "",
    repo: str = "",
    local_dir: str = "",
    config_path: str = "",
    dry_run: bool = False,
) -> dict:
    # Imported here, not at module level, so `modal run` can load this file
    # locally without boto3/structlog installed. Deliberately free of
    # torch/transformers (training.new_train pulls those in).
    from training.s3_utils import download_folder, find_latest_remote_run_dir, read_remote_json

    raw_cfg = _load_yaml_cfg(config_path)
    train_cfg = raw_cfg.get("training", {}) or {}
    s3_cfg = _nested_list_dict(raw_cfg.get("s3", {}))

    mode = mode or str(train_cfg.get("mode", "pretrain"))
    repo = repo or train_cfg.get("hf_chkpt_path") or ""
    if not repo:
        raise ValueError("no HF repo: set training.hf_chkpt_path in train_config.yaml or pass --repo")

    # Same remote layout Trainer._push_checkpoint_to_s3 writes:
    # [<s3 prefix>/]checkpoints/<basename(out_dir)>/<run_dir>/
    out_dir_name = os.path.basename(
        str(os.getenv("TRAIN_OUT_DIR") or train_cfg.get("out_dir", "out")).rstrip("/")
    )
    bucket_prefix = str(s3_cfg.get("prefix", ""))
    s3_kwargs = dict(
        bucket=s3_cfg.get("s3_bucket_name") or os.environ["S3_BUCKET"],
        endpoint=s3_cfg.get("s3_endpoint") or os.environ["S3_ENDPOINT"],
        access_key=os.environ["S3_ACCESS_KEY_ID"],
        secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )
    remote_root = f"checkpoints/{out_dir_name}"

    # Full keys (bucket prefix already folded in) from here on, so every
    # s3_utils call below passes prefix="".
    if run_dir:
        rel = f"{remote_root}/{run_dir.strip('/')}/"
        run_prefix = f"{bucket_prefix.rstrip('/')}/{rel}" if bucket_prefix else rel
    else:
        run_prefix = find_latest_remote_run_dir(remote_root, mode, prefix=bucket_prefix, **s3_kwargs)
        if run_prefix is None:
            raise FileNotFoundError(
                f"no run dir ending in _{mode} with a trainer_state.json under "
                f"s3://{s3_kwargs['bucket']}/{remote_root}/"
            )

    meta = read_remote_json(f"{run_prefix}trainer_state.json", **s3_kwargs) or {}
    ckpt_name = _resolve_ckpt_name(run_prefix, meta, best, s3_kwargs)
    iter_num = meta.get("best_iter_num") if best else meta.get("iter_num")
    if not best and ckpt_name != os.path.basename(str(meta.get("latest_ckpt", "")).rstrip("/")):
        iter_num = _ckpt_iter(ckpt_name)

    remote_ckpt = f"{run_prefix}{ckpt_name}"
    which = "best" if best else "latest"
    print(f"run dir:    s3://{s3_kwargs['bucket']}/{run_prefix}")
    print(f"checkpoint: {ckpt_name} ({which}, iter {iter_num}, best_val_loss {meta.get('best_val_loss')})")
    print(f"target:     https://huggingface.co/{repo}")

    summary = {
        "run_prefix": run_prefix, "checkpoint": ckpt_name, "which": which,
        "iter_num": iter_num, "repo": repo, "pushed": False,
    }
    if dry_run:
        print("dry run -- nothing downloaded or pushed")
        return summary

    token = _hf_token()
    if not token:
        raise RuntimeError("missing HF token: set HF_WRITE_TOKEN (or HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HF_API_KEY)")

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="s3_ckpt_") as tmp:
        dest = os.path.join(local_dir, ckpt_name) if local_dir else os.path.join(tmp, ckpt_name)
        # ckpt_best/ is rewritten in place on every improvement, so a cached
        # copy in --local-dir can be stale; ckpt_<iter>/ never changes once saved.
        files = download_folder(
            remote_ckpt, dest, prefix="",
            force_redownload_paths=(lambda _rel: True) if best else None,
            **s3_kwargs,
        )
        if not files:
            raise FileNotFoundError(f"s3://{s3_kwargs['bucket']}/{remote_ckpt}/ is empty")
        print(f"downloaded {len(files)} file(s) -> {dest}")

        api = HfApi(token=token)
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            folder_path=dest,
            repo_id=repo,
            repo_type="model",
            commit_message=f"{which} checkpoint ({ckpt_name}) at iter {iter_num} from s3",
        )
    print(f"pushed {ckpt_name} -> {repo}: {getattr(commit, 'commit_url', commit)}")
    summary["pushed"] = True
    return summary


# ---------------------------------------------------------------------------
# Modal entrypoint (`modal run training/push_s3_checkpoint_to_hf.py ...`).
# Optional: plain `python` use works without modal installed.
# ---------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("boto3", "pyyaml", "python-dotenv", "structlog", "huggingface_hub[hf_transfer]")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .add_local_dir(
            str(ROOT), remote_path="/app",
            ignore=[".git", "__pycache__", "*.pyc", "out/", ".env", ".venv", ".pytest_cache", ".claude"],
        )
    )
    app = modal.App("sabiyarn-push-s3-checkpoint-to-hf")

    @app.function(
        image=image, cpu=4, memory=8192, timeout=4 * 3600,
        secrets=[modal.Secret.from_dotenv(__file__)],
    )
    def push_remote(
        best: bool = False, run_dir: str = "", mode: str = "", repo: str = "", dry_run: bool = False,
    ) -> dict:
        return push_s3_checkpoint_to_hf(best=best, run_dir=run_dir, mode=mode, repo=repo, dry_run=dry_run)

    @app.local_entrypoint()
    def main(best: bool = False, run_dir: str = "", mode: str = "", repo: str = "", dry_run: bool = False):
        print(push_remote.remote(best=best, run_dir=run_dir, mode=mode, repo=repo, dry_run=dry_run))


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best", action="store_true", help="Push ckpt_best/ instead of the latest ckpt_<iter>/.")
    parser.add_argument("--run-dir", default="", help="Run dir name under checkpoints/<out_dir>/ (default: latest for --mode).")
    parser.add_argument("--mode", default="", help="pretrain | sft (default: training.mode from the config).")
    parser.add_argument("--repo", default="", help="HF repo id (default: training.hf_chkpt_path from the config).")
    parser.add_argument("--local-dir", default="", help="Keep the downloaded weights here instead of a temp dir.")
    parser.add_argument("--config", default="", help="Config path (default: TRAIN_CONFIG_PATH or training/train_config.yaml).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print what would be pushed; push nothing.")
    args = parser.parse_args()
    push_s3_checkpoint_to_hf(
        best=args.best, run_dir=args.run_dir, mode=args.mode, repo=args.repo,
        local_dir=args.local_dir, config_path=args.config, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()
