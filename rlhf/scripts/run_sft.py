#!/usr/bin/env python
"""Supervised warm-start on the translation dataset (optional: skip it if the checkpoint already translates well).

    python rlhf/scripts/run_sft.py                       # 1 GPU
    torchrun --standalone --nproc_per_node=2 rlhf/scripts/run_sft.py
    python rlhf/scripts/run_sft.py learning_rate=1e-5 max_steps=500
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from rl.config import load_rl_config  # noqa: E402
from rl.sft import train_sft  # noqa: E402
from rlhf.translation import load_translation_records  # noqa: E402

if __name__ == "__main__":
    load_dotenv()
    over = dict(kv.split("=", 1) for kv in sys.argv[1:])
    cfg = load_rl_config(str(ROOT / "rlhf" / "configs" / "default.yaml"), algo="sft", **{"out_dir": "outputs/translate-sft", **over})
    # rl.sft reads records through rl.data; hand it the chat-formatted translation records instead
    import rl.sft as _sft

    _sft.load_records = lambda c: load_translation_records(c)
    train_sft(cfg)
