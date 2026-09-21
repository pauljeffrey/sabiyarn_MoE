#!/usr/bin/env python
"""Translate one sentence with a trained checkpoint.

    python rlhf/scripts/translate.py --instruction "Translate to Hausa:" --input "Good morning" --model-path outputs/rlhf/final
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from rl.config import load_rl_config  # noqa: E402
from rlhf.translation import translate  # noqa: E402

if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instruction", required=True, help="e.g. 'Translate to Hausa:'")
    ap.add_argument("--input", required=True, help="source sentence")
    ap.add_argument("--model-path", default="outputs/rlhf/final")
    args = ap.parse_args()
    cfg = load_rl_config(str(ROOT / "rlhf" / "configs" / "default.yaml"))
    print(translate(args.instruction, args.input, args.model_path, cfg))
