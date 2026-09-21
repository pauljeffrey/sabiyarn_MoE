#!/usr/bin/env python
"""AfriCOMET-guided RL on the translation dataset.

    python rlhf/scripts/run_rlhf.py model_path=outputs/translate-sft/final           # after run_sft.py
    torchrun --standalone --nproc_per_node=2 rlhf/scripts/run_rlhf.py
    python rlhf/scripts/run_rlhf.py reward=chrf max_steps=20                         # cheap smoke test, no COMET
    python rlhf/scripts/run_rlhf.py reward_python=.venv-comet/bin/python            # AfriCOMET in its own env

`key=value` pairs are RLConfig fields (rl/config.py). AfriCOMET needs transformers<5, so it runs in a second
virtualenv (see HOW_TO_RUN.md "AfriCOMET RL"); `reward=chrf` needs nothing extra.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from rl.config import load_rl_config  # noqa: E402
from rl.policy_gradient import train_rl  # noqa: E402
from rlhf.translation import load_translation_records  # noqa: E402

if __name__ == "__main__":
    load_dotenv()
    over = dict(kv.split("=", 1) for kv in sys.argv[1:])
    cfg = load_rl_config(str(ROOT / "rlhf" / "configs" / "default.yaml"), **over)
    train_rl(cfg, records=load_translation_records(cfg))
