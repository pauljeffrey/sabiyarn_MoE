#!/usr/bin/env python3
"""Entry point:  python -m rl.run [--config rl/config.yaml] [--algo dpo|rl|sft] [key=value ...]

    python -m rl.run                                              # DPO with rl/config.yaml
    torchrun --standalone --nproc_per_node=2 -m rl.run --config rlhf/configs/default.yaml
    python -m rl.run learning_rate=1e-6 max_steps=200 out_dir=outputs/dpo-test

`key=value` pairs are RLConfig fields (same as the RL_<FIELD> environment variables, which they override).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import load_rl_config  # noqa: E402


def main(argv: list[str] | None = None) -> str:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: RL_CONFIG_PATH or rl/config.yaml)")
    ap.add_argument("--algo", choices=["sft", "dpo", "rl"], default=None)
    ap.add_argument("overrides", nargs="*", help="RLConfig fields as key=value")
    args = ap.parse_args(argv)

    over = dict(kv.split("=", 1) for kv in args.overrides)
    if args.algo:
        over["algo"] = args.algo
    cfg = load_rl_config(args.config, **over)

    if cfg.algo == "dpo":
        from rl.dpo import train_dpo

        return train_dpo(cfg)
    if cfg.algo == "sft":
        from rl.sft import train_sft

        return train_sft(cfg)
    from rl.policy_gradient import train_rl

    return train_rl(cfg)


if __name__ == "__main__":
    main()
