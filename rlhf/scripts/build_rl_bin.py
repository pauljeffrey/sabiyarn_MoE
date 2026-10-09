#!/usr/bin/env python
"""Build the RL prompt bin from the assembled AfriCOMET RL dataset and (optionally) push it to the S3 bucket.

    python rlhf/scripts/build_rl_bin.py --shards "OUT/final/*/*.parquet" --prefix work/rl_translation_prompts [--layout csr|padded]
        [--eval-rows 5000] [--push]

Shuffles thoroughly, tokenizes the `text` column (the chat-template prompt) with the SabiYarn tokenizer, writes the files described in
rl/bin_data.py and prints the maximum prompt length (use it as `max_prompt_len`). `--push` uploads <prefix>.* to datasets/ in the bucket
configured in training/train_config.yaml (never overwrites an existing object).
"""
import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data-gen"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--layout", default="csr", choices=["csr", "padded"])
    ap.add_argument("--eval-rows", type=int, default=5000)
    ap.add_argument("--max-tokens-cap", type=int, default=0, help="drop prompts longer than this (0 = keep all)")
    ap.add_argument("--tokenizer", default="BeardedMonster/SabiYarn-32k")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    from africomet_translation.build import hf_token
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from rl.bin_data import build

    tk = Tokenizer.from_file(hf_hub_download(a.tokenizer, "tokenizer.json", token=hf_token()))
    eos, pad = tk.token_to_id("</s>"), tk.token_to_id("<pad>")
    shards = sorted(Path(p) for p in glob.glob(a.shards))
    print(len(shards), "shards")
    m = build(shards, Path(a.prefix), lambda texts: [e.ids for e in tk.encode_batch(texts, add_special_tokens=False)], pad, eos, a.tokenizer,
              layout=a.layout, eval_rows=a.eval_rows, max_tokens_cap=a.max_tokens_cap or None)
    print({k: m[k] for k in ("n_rows", "eval_start", "max_prompt_tokens", "mean_prompt_tokens", "p99_prompt_tokens", "dropped_over_cap")})
    if a.push:
        from dotenv import load_dotenv
        from omegaconf import OmegaConf

        load_dotenv(ROOT / ".env")
        s3 = OmegaConf.to_container(OmegaConf.load(ROOT / "training" / "train_config.yaml"), resolve=True)["s3"]
        from training.s3_utils import upload_if_absent

        for f in sorted(Path(a.prefix).parent.glob(Path(a.prefix).name + ".*")):
            upload_if_absent(str(f), f"datasets/{f.name}", bucket=s3["s3_bucket_name"], endpoint=s3["s3_endpoint"],
                             access_key=os.environ["S3_ACCESS_KEY_ID"], secret_key=os.environ["S3_SECRET_ACCESS_KEY"], prefix="", override=False)
            print("uploaded", f.name)


if __name__ == "__main__":
    main()
