#!/usr/bin/env python3
"""Push sabiyarn/model/modeling.py (and configuration.py) to a HF model
repo's trust_remote_code files, overwriting what's there.

Training (training/new_train.py) never reads sabiyarn/model/modeling.py
directly -- AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
always downloads modeling.py fresh from the HF repo named by
model.repo_name/hf_chkpt_path in train_config.yaml. So a fix made to the
local copy under sabiyarn/model/ has NO EFFECT on real training runs until
it's pushed here, overwriting the file that repo currently serves.

Local, no Modal/GPU needed -- just huggingface_hub. Defaults to a DRY RUN
(prints what would be uploaded, uploads nothing); pass --confirm to
actually push.

Usage:
    # Preview only -- uploads nothing.
    python training/push_model_code_to_hf.py

    # Actually push modeling.py (+ configuration.py if present) to the repo.
    python training/push_model_code_to_hf.py --confirm

    # Push to a different repo, or push only one file.
    python training/push_model_code_to_hf.py --repo Aletheia-ng/sabiyarn-ref --confirm
    python training/push_model_code_to_hf.py --files modeling.py --confirm
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC_DIR = ROOT / "sabiyarn" / "model"
DEFAULT_REPO = "Aletheia-ng/SabiYarn_MoE-280M"
DEFAULT_FILES = ["modeling.py", "configuration.py"]


def _resolve_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_API_KEY")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF model repo to push to (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--files", nargs="+", default=DEFAULT_FILES,
        help=f"Filenames under sabiyarn/model/ to push (default: {DEFAULT_FILES})",
    )
    parser.add_argument("--confirm", action="store_true", help="Actually upload. Without this, only prints a preview.")
    args = parser.parse_args()

    paths = []
    for name in args.files:
        p = MODEL_SRC_DIR / name
        if not p.is_file():
            print(f"SKIP (not found): {p}")
            continue
        paths.append(p)

    if not paths:
        print("Nothing to push -- no valid files found.")
        return

    print(f"Target repo: {args.repo}")
    for p in paths:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)  ->  {p.name} in repo root")

    if not args.confirm:
        print("\nDry run only -- nothing uploaded. Re-run with --confirm to actually push.")
        return

    token = _resolve_token()
    if not token:
        raise SystemExit(
            "No HF token found (checked HF_TOKEN, HUGGING_FACE_HUB_TOKEN, HF_API_KEY). "
            "Set one in your environment or .env before pushing."
        )

    from huggingface_hub import HfApi

    api = HfApi()
    for p in paths:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=args.repo,
            repo_type="model",
            token=token,
            commit_message=f"Update {p.name}: gate MoE router noise on torch.is_grad_enabled() too, "
                            "not just self.training",
        )
        print(f"Pushed {p.name} -> {args.repo}")


if __name__ == "__main__":
    main()
