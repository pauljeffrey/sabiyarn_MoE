"""
Manual sanity check for SFT label masking against the REAL tokenizer.

training/label_masking.py's <|system|>/<|user|>/<|assistant|> masking is
covered by tests/test_label_masking.py using synthetic token ids (the
sandbox that wrote this had no huggingface.co access). Run this script
yourself, wherever you have real HF access, to confirm the real tokenizer's
token ids behave the same way. Not part of the pytest suite -- it needs
network access and prints for manual inspection rather than asserting.

Usage (from repo root):
    python scripts/verify_real_tokenizer_masking.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.constant_tokens import MASK, assistant_token, system_token, tokenizer, user_token
from training.label_masking import process_sft_labels

EXAMPLES = {
    "user_then_assistant": (
        "<s>\n<|user|>yaw to malin so accra\nPlease translate the above sentence to English.\n</s>\n\n\n"
        "<|assistant|>\n\n<|input_lang|><unk><task_plan><translate></task_plan><|target_lang|>"
        "     <eng><response> Yaw lay malin but accra</s>\n\n"
    ),
    "system_then_user_then_assistant": (
        "<s>\n<|system|>You are a helpful translation assistant.\n"
        "<|user|>yaw to malin so accra\nPlease translate the above sentence to English.\n</s>\n\n\n"
        "<|assistant|>\n\n<eng><response> Yaw lay malin but accra</s>\n\n"
    ),
}


def main():
    for label, text in EXAMPLES.items():
        print("=" * 80)
        print(label)
        print("=" * 80)
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = torch.tensor(ids)
        result = process_sft_labels(tokens, user_token, assistant_token, system_token=system_token, mask=MASK)

        for tok_id, masked in zip(ids, result.tolist()):
            piece = tokenizer.decode([tok_id])
            flag = "MASKED" if masked == MASK else "kept  "
            print(f"  [{flag}] {tok_id:>7}  {piece!r}")
        print()


if __name__ == "__main__":
    main()
