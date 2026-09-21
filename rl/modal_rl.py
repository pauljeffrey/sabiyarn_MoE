#!/usr/bin/env python3
"""Modal launcher for rl/ (DPO, RL, SFT warm-start). Same image, GPU shape, volume and .env secrets as
training/modal_train.py -- edit `modal:` in training/train_config.yaml to change the GPUs.

    modal run rl/modal_rl.py --config rl/config.yaml                                   # DPO
    modal run rl/modal_rl.py --config rlhf/configs/default.yaml --overrides "reward=chrf max_steps=50"
    modal run rl/modal_rl.py --script rlhf/scripts/run_rlhf.py --overrides "model_path=/data/out_280M/run/ckpt_best"

The model to start from must be reachable inside the container: a Hub id, or a path on the volume
(`/data/...`, where training/ writes its checkpoints). Outputs and MLflow runs go to the same volume
(/data/rl/...; `modal volume get sabiyarn-data rl/ ./rl_out`). AfriCOMET (`reward=africomet`) needs the second
Python environment described in HOW_TO_RUN.md, so on Modal use `reward=chrf`, or bake a COMET venv into the image.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

_APP_MOUNT = Path("/app")
ROOT = _APP_MOUNT if (_APP_MOUNT / "training").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# image / volume / GPU spec / secrets are the training launcher's, so both stay in step
from training.modal_train import DATA_DIR, GPUS_PER_NODE, GPU_SPEC, NODE_CPU, image, volume  # noqa: E402

app = modal.App("sabiyarn-modal-rl")


@app.function(image=image, gpu=GPU_SPEC, cpu=NODE_CPU, timeout=86400, volumes={DATA_DIR: volume},
              secrets=[modal.Secret.from_dotenv(__file__)])
def run_rl(config: str = "rl/config.yaml", script: str = "", overrides: str = ""):
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env.setdefault("MLFLOW_TRACKING_URI", f"file:{DATA_DIR}/mlruns")
    env.setdefault("MLFLOW_UI_ENABLED", "0")
    kv = overrides.split()
    if not any(a.startswith("out_dir=") for a in kv):
        kv.append(f"out_dir={DATA_DIR}/rl/{Path(config).stem}")
    entry = ["-m", "rl.run", "--config", config] if not script else [script]
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={GPUS_PER_NODE}", *entry, *kv]
    print("launching:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd="/app", env=env, check=True)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "rl/config.yaml", script: str = "", overrides: str = ""):
    run_rl.remote(config=config, script=script, overrides=overrides)
