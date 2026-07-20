import modal
from modal import App, Image, Secret, Volume
from pathlib import Path
import structlog

sabiyarn = Image.debian_slim(python_version="3.12").pip_install(
    "transformers[torch]==4.41.2",
    "triton",
    "bitsandbytes",
    "datasets",
    "wandb",
    "structlog",
    "PyYAML",
    "simple-parsing==0.0.3rc1",
    "sentencepiece",
    "scikit-learn",
    force_build=True,
)

LOG = structlog.stdlib.get_logger()
VOL_MOUNT_PATH = Path("/vol")
stub = App(name="sabiyarn-ablation-tests", image=sabiyarn)
output_vol = Volume.from_name("sabiyarn_v2", create_if_missing=True)

restart_tracker_dict = modal.Dict.from_name("sabiyarn-ablation", create_if_missing=True)


def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count


def run_eval(vol: modal.Volume):
    from . import eval

    eval.run_all(vol)


@stub.function(
    gpu="T4",
    timeout=60 * 60 * 20,
    cpu=4.0,
    secrets=[Secret.from_name("wandb-api"), Secret.from_name("hf-secret")],
    volumes={VOL_MOUNT_PATH: output_vol},
)
def run():
    LOG.info("modal instance running..")
    run_eval(output_vol)
    output_vol.commit()
