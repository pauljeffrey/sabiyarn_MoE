#!/usr/bin/env python3
"""
Test generation using a checkpoint saved by training/new_train.py on Modal.

Loads via the standard transformers AutoModelForCausalLM/AutoTokenizer interface --
GPTJXMoEForCausalLM (sabiyarn/model/modeling.py) is a regular HF PreTrainedModel, so
checkpoints saved with save_pretrained() round-trip through from_pretrained() directly,
no manual state_dict surgery required.

Usage:
  modal run test_generation.py::main
  modal run test_generation.py::main --checkpoint-dir /data/checkpoints/<run>/ckpt_<iter>
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "transformers>=4.55.0", "numpy", "structlog", "pyyaml", "omegaconf", "python-dotenv")
    .add_local_dir(
        str(ROOT),
        remote_path="/app",
        ignore=[".git", "*.pyc", "__pycache__", ".pytest_cache", "*.egg-info", "out/", "*.bin", ".env"],
    )
)

app = modal.App("sabiyarn-generation-test")

# Same volume training writes checkpoints to (see training/modal_train.py TRAIN_OUT_DIR).
volume = modal.Volume.from_name("sabiyarn-data", create_if_missing=True)


def _latest_checkpoint_dir(checkpoints_root: str) -> str:
    """Most recently modified ckpt_* directory under any run dir on the volume."""
    import os

    candidates = []
    if os.path.isdir(checkpoints_root):
        for run in os.scandir(checkpoints_root):
            if not run.is_dir():
                continue
            for ckpt in os.scandir(run.path):
                if ckpt.is_dir() and ckpt.name.startswith("ckpt_"):
                    candidates.append(ckpt.path)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {checkpoints_root}. Train first, or pass --checkpoint-dir explicitly."
        )
    return max(candidates, key=lambda p: os.path.getmtime(p))


@app.function(
    gpu="A100-40GB",
    timeout=1200,
    image=image,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def test_generation(
    prompts: list[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    checkpoint_dir: str = "",
):
    """Generate text from a checkpoint via the standard HF generate() API."""
    import os
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.chdir("/app")
    sys.path.insert(0, "/app")

    from training.load_config import load_train_config

    cfg = load_train_config()
    ckpt_dir = checkpoint_dir or _latest_checkpoint_dir(os.path.join(DATA_DIR, "checkpoints"))
    print(f"Loading checkpoint: {ckpt_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    emb_vocab = model.get_input_embeddings().weight.size(0)
    if tokenizer.vocab_size != model.config.vocab_size or emb_vocab != model.config.vocab_size:
        print(
            f"Warning: vocab mismatch -- tokenizer={tokenizer.vocab_size}, "
            f"model.config.vocab_size={model.config.vocab_size}, embeddings={emb_vocab}. "
            "Make sure this is the exact tokenizer the checkpoint was trained with."
        )

    if prompts is None:
        prompts = ["The boy", "I am a"]

    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            if input_ids.size(1) > model.config.block_size:
                input_ids = input_ids[:, -model.config.block_size :]

            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
            )

            text_full = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            text_input = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            outputs.append({"prompt": prompt, "completion": text_full[len(text_input):], "full": text_full})

    return outputs


@app.local_entrypoint()
def main(checkpoint_dir: str = ""):
    """Run generation on Modal and print results."""
    print("Generating from checkpoint...")
    results = test_generation.remote(checkpoint_dir=checkpoint_dir)
    for r in results:
        print(f"\n=== Prompt ===\n{r['prompt']}\n=== Completion ===\n{r['completion']}")
    return True


if __name__ == "__main__":
    main()
