#!/usr/bin/env python3
"""Modal launcher for post-training: DPO, reward-based RL and the SFT warm-start (`rl/`).

    modal run rl/modal_rl.py                                              # DPO, rl/config.yaml
    modal run rl/modal_rl.py --config rlhf/configs/default.yaml --overrides "reward=chrf max_steps=50"
    modal run rl/modal_rl.py --config rlhf/configs/default.yaml           # AfriCOMET RL (comet image)
    modal run rl/modal_rl.py --overrides "model_path=/data/out_280M/20260101_120000_pretrain/ckpt_best"

Same GPU shape, persistent volume and `.env` secrets as training/modal_train.py -- the `modal:` section of
training/train_config.yaml decides the GPUs for both. Deliberately NOT importing modal_train: this file needs
its own image (the AfriCOMET one carries a second virtualenv) and one `modal.App` per file keeps `modal run`
unambiguous.

Three things this handles that a bare `modal run` would not:

  * Training data. `data-gen/data/` is excluded from the image upload (it is large and regenerated), so a local
    `data_path` is uploaded to the volume at `/data/rl_data/<name>` before launch and the container is pointed
    at it. A `data_path` that is already a `/data/...` path, or a `dataset_id`, is left alone.
  * The starting checkpoint. `model_path` is usually a Hub id, but a path under `/data/...` (where
    modal_train.py writes) works too. A LOCAL directory would not exist in the container -- that is an error
    here rather than a confusing failure 20 minutes in.
  * AfriCOMET. `unbabel-comet` cannot coexist with transformers 5 (see rlhf/requirements.txt), so the `comet`
    image bakes it into /opt/comet-venv and `reward_python` is pointed there automatically.

Outputs, checkpoints and MLflow runs land on the volume; fetch them with
`modal volume get sabiyarn-data rl/ ./rl_out` and `modal volume get sabiyarn-data mlruns ./mlruns`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import modal
import yaml

_APP_MOUNT = Path("/app")
ROOT = _APP_MOUNT if (_APP_MOUNT / "training").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = "/data"
RL_DATA_DIR = f"{DATA_DIR}/rl_data"
COMET_VENV = "/opt/comet-venv"
TRAIN_CONFIG = ROOT / "training" / "train_config.yaml"

# GPU topology comes from the same place as training (`modal:` in train_config.yaml), so a post-training run
# lands on the hardware you already sized for this model.
_modal_cfg = (yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8")) or {}).get("modal", {}) or {}
GPUS_PER_NODE = max(1, int(_modal_cfg.get("gpus_per_node", 1)))
GPU_TYPE = str(_modal_cfg.get("gpu_type", "A100"))
GPU_SPEC = f"{GPU_TYPE}:{GPUS_PER_NODE}" if GPUS_PER_NODE > 1 else GPU_TYPE
NODE_CPU = max(8, 4 * GPUS_PER_NODE)

_IGNORE = [
    ".git", "**/__pycache__", "*.pyc", ".env", "**/.venv*", ".pytest_cache", ".claude",
    "out/", "out_*/", "mlruns/", "eval_results/", "data/bins/", "data-gen/data/", "*.bin",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .add_local_dir(str(ROOT), remote_path="/app", ignore=_IGNORE)
)


def _comet_pins() -> list[str]:
    """Requirement lines of rlhf/requirements.txt (comments stripped), read at image-build time."""
    text = (ROOT / "rlhf" / "requirements.txt").read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


# AfriCOMET image: the same environment plus a second virtualenv holding unbabel-comet. --system-site-packages
# reuses the ~2.5 GB torch already in the image; comet's transformers<5 / numpy<2 are installed into the venv
# and shadow the outer ones only for the worker process (rl/comet_worker.py).
comet_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .run_commands(
        f"python -m venv --system-site-packages {COMET_VENV}",
        f"{COMET_VENV}/bin/pip install --no-cache-dir " + " ".join(f"'{p}'" for p in _comet_pins()),
        f"{COMET_VENV}/bin/python -c 'import comet, transformers; print(\"comet ok\", transformers.__version__)'",
    )
    .add_local_dir(str(ROOT), remote_path="/app", ignore=_IGNORE)
)

app = modal.App("sabiyarn-modal-rl")
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)
_fn_kwargs = dict(
    cpu=NODE_CPU, timeout=86400, volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_dotenv(__file__)],
)


def _launch(config: str, overrides: str, comet: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env.setdefault("MLFLOW_TRACKING_URI", f"file:{DATA_DIR}/mlruns")
    env.setdefault("MLFLOW_UI_ENABLED", "0")  # a UI inside the container is unreachable

    kv = overrides.split()
    given = {a.split("=", 1)[0] for a in kv if "=" in a}
    if "out_dir" not in given:
        kv.append(f"out_dir={DATA_DIR}/rl/{Path(config).stem}")
    if comet and "reward_python" not in given:
        kv.append(f"reward_python={COMET_VENV}/bin/python")

    cmd = ["torchrun", "--standalone", f"--nproc_per_node={GPUS_PER_NODE}", "-m", "rl.run", "--config", config, *kv]
    print("launching:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd="/app", env=env, check=True)
    finally:
        volume.commit()  # flush checkpoints + MLflow runs even if the run crashed


@app.function(image=image, gpu=GPU_SPEC, **_fn_kwargs)
def run_rl(config: str = "rl/config.yaml", overrides: str = ""):
    _launch(config, overrides, comet=False)


@app.function(image=comet_image, gpu=GPU_SPEC, **_fn_kwargs)
def run_rl_comet(config: str = "rlhf/configs/default.yaml", overrides: str = ""):
    _launch(config, overrides, comet=True)


def _resolve(config: str, overrides: str):
    """Load the config exactly as the container will (also a local fail-fast on a bad config)."""
    from rl.config import load_rl_config

    over = dict(a.split("=", 1) for a in overrides.split() if "=" in a)
    return load_rl_config(str(ROOT / config), **over), over


def _stage_data(cfg, over: dict) -> str:
    """Upload a local jsonl the config points at to the volume; return the extra override (or "")."""
    if cfg.dataset_id:
        return ""  # a Hub dataset: the container downloads it itself
    if not cfg.data_path or cfg.data_path.startswith(DATA_DIR):
        return ""
    local = ROOT / cfg.data_path
    if not local.is_file():
        raise SystemExit(
            f"data_path {cfg.data_path!r} does not exist locally and is not a {DATA_DIR}/... path on the volume.\n"
            f"Generate it (data-gen), or upload it yourself:\n"
            f"    modal volume put sabiyarn-data <file.jsonl> rl_data/<file.jsonl>\n"
            f"    modal run rl/modal_rl.py --overrides 'data_path={RL_DATA_DIR}/<file.jsonl>'"
        )
    remote = f"rl_data/{local.name}"
    print(f"uploading {local} ({local.stat().st_size / 2**20:.1f} MB) -> volume:{remote}")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local), f"/{remote}")
    return f"data_path={DATA_DIR}/{remote}"


@app.local_entrypoint()
def main(config: str = "rl/config.yaml", overrides: str = "", comet: bool = False):
    """--config <yaml> --overrides "key=value key=value" [--comet]

    `--comet` (or a config whose reward is africomet) selects the image with the AfriCOMET virtualenv.
    """
    cfg, over = _resolve(config, overrides)
    if cfg.model_path.startswith((".", "/")) and not cfg.model_path.startswith(DATA_DIR):
        raise SystemExit(
            f"model_path {cfg.model_path!r} is a local path; the container cannot see it. Use a Hub id, or a "
            f"checkpoint on the volume ({DATA_DIR}/<training.out_dir>/<run>/ckpt_best -- "
            f"`modal volume ls sabiyarn-data` to find it)."
        )
    staged = _stage_data(cfg, over)
    if staged:
        overrides = f"{overrides} {staged}".strip()

    print(f"algo={cfg.algo} reward={cfg.reward} model={cfg.model_path} gpus={GPU_SPEC}")
    if comet or cfg.reward == "africomet":
        print("image: comet (AfriCOMET virtualenv baked in)")
        run_rl_comet.remote(config=config, overrides=overrides)
    else:
        run_rl.remote(config=config, overrides=overrides)
