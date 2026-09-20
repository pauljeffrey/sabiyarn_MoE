"""Which of these three weight sets actually hold the trained model?

  A. <run_dir>/ckpt_<iter>/model.safetensors        (save_pretrained output)
  B. <run_dir>/resume_state/pytorch_model_fsdp.bin  (what accelerator.load_state reads)
  C. the HF repo weights (init_from: "hf")

Prints relative L2 distance between each pair, per parameter group. Read it as:
  A~B, both far from C  -> resume_state is fine; accelerator.load_state() is a silent no-op
  A far from B, B~C     -> resume_state itself holds stale weights (save-side problem):
                           never rely on it for the model, use init_from: "resume" and skip its model restore
  A~C                   -> the "checkpoint" was never trained
Also prints the dtype each file is stored in. If B (resume_state, fp32 masters written by FSDP) is
measurably different from A (bf16-rounded save_pretrained output) the training progress lives only in B.

CPU only, reads the sabiyarn-data volume. Usage:
    modal run scripts/compare_resume_weights_modal.py --run-dir /data/out_280M/20260723_031904_pretrain --iter 17250
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # picks up MODAL_TOKEN_ID/SECRET (and HF token) from .env, same as training/modal_train.py

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "safetensors", "huggingface_hub", "numpy"
)
app = modal.App("sabiyarn-compare-resume-weights")
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=False)


def _norm(key: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "_orig_mod.", "module."):
        key = key.replace(prefix, "")
    return key


def _group(key: str) -> str:
    if "wte" in key or "lm_head" in key:
        return "embeddings (wte/lm_head)"
    if "wpe" in key:
        return "wpe"
    if key.startswith("transformer.h.") or ".h." in key:
        return "blocks"
    return "other (ln_f, ...)"


@app.function(
    image=image, cpu=4, memory=16384, timeout=1800, volumes={"/data": volume},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def compare(run_dir: str, iter: int, hf_repo: str = "Aletheia-ng/SabiYarn_MoE-280M"):
    import glob
    import os

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    def load_safetensors_dir(path: str) -> dict:
        sd: dict = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(f))
        return sd

    sets = {
        "A ckpt": load_safetensors_dir(f"{run_dir}/ckpt_{iter}"),
        "B resume_state": torch.load(f"{run_dir}/resume_state/pytorch_model_fsdp.bin", map_location="cpu", weights_only=True),
        "C hf": load_safetensors_dir(snapshot_download(hf_repo, allow_patterns=["*.safetensors"])),
    }
    for name, sd in sets.items():
        dtypes: dict[str, int] = {}
        for v in sd.values():
            if hasattr(v, "dtype"):
                dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
        print(f"{name}: {len(sd)} tensors, stored dtypes {dtypes}")  # fp32 vs bf16 on disk matters, see docstring
    sets = {name: {_norm(k): v.float() for k, v in sd.items() if hasattr(v, "float")} for name, sd in sets.items()}

    def rel_l2(a: dict, b: dict) -> dict[str, float]:
        num: dict[str, float] = {}
        den: dict[str, float] = {}
        for k in a.keys() & b.keys():
            if a[k].shape != b[k].shape:
                continue
            g = _group(k)
            num[g] = num.get(g, 0.0) + float((a[k] - b[k]).pow(2).sum())
            den[g] = den.get(g, 0.0) + float(b[k].pow(2).sum())
        return {g: (num[g] / den[g]) ** 0.5 if den[g] else float("nan") for g in num}

    for left, right in (("A ckpt", "B resume_state"), ("A ckpt", "C hf"), ("B resume_state", "C hf")):
        print(f"\nrel L2  {left}  vs  {right}")
        for group, value in sorted(rel_l2(sets[left], sets[right]).items()):
            print(f"  {group:28s} {value:.6f}")


@app.local_entrypoint()
def main(run_dir: str, iter: int, hf_repo: str = "Aletheia-ng/SabiYarn_MoE-280M"):
    compare.remote(run_dir=run_dir, iter=iter, hf_repo=hf_repo)
