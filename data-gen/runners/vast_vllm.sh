#!/usr/bin/env bash
# Bootstrap a rented GPU box and generate the corpus with vLLM. Paste into a fresh instance's shell.
#
#   export HF_TOKEN=...            # required: to download the generator's weights
#   export HF_WRITE_TOKEN=...      # optional: only needed for PUSH=1
#   bash runners/vast_vllm.sh sft
#
# Everything is an environment variable so one script covers every phase and every box:
#
#   KIND     pretrain | sft | rl | judge      (also the first positional argument)
#   LANGS    comma list, e.g. yor,hau,ibo     (empty = every language in the seed)
#   CONTEXT  16384 | 32768                    (default 32768, the model's own block_size)
#   MODEL    default google/gemma-4-31b-it
#   QUANT    auto | throughput | none | fp8 | bitsandbytes   (default auto: highest precision that fits)
#   TP       GPUs to shard the model across   (default: every GPU the box reports)
#   LIMIT    cap this run                     (0 = everything outstanding)
#   GPU_COST your hourly rate, for the cost lines it prints
#   PUSH     1 to push shards to the Hub when the phase finishes
#   SHARDS   run N workers over one row list (set SHARD_INDEX per worker)
#
# CHOOSING A BOX. gemma-4-31b is dense 30.8B: ~62 GB of bf16 weights, ~31 GB fp8, ~17 GB 4-bit. Two traps
# worth knowing before you bid:
#   * fp8 needs compute capability >= 8.9. Ada (L40S, RTX 4090/5090), Hopper and Blackwell have it; AMPERE
#     DOES NOT, so on an A100 or A800 the only quantized option is bitsandbytes.
#   * a listing's headline GB is sometimes SYSTEM RAM, not VRAM. A "CMP 170HX 64GB" has 8 GB of VRAM; the 64
#     is the host. Check the VRAM figure, and `--plan-only` below prints what the box actually reports before
#     you spend anything.
# The script is billed by the second, so it pushes incrementally: if the box dies or you are outbid you lose
# only the chunk in flight, and re-running skips every row already written.
set -euo pipefail

KIND="${1:-${KIND:-sft}}"
MODEL="${MODEL:-google/gemma-4-31b-it}"
CONTEXT="${CONTEXT:-32768}"
QUANT="${QUANT:-auto}"
LIMIT="${LIMIT:-0}"
LANGS="${LANGS:-}"
GPU_COST="${GPU_COST:-1.00}"
SHARDS="${SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PUSH="${PUSH:-0}"
REPO="${DATA_GEN_GIT_REPO:-https://github.com/pauljeffrey/sabiyarn_MoE}"
WORK="${WORK:-/workspace/sabiyarn}"

case "$KIND" in
  pretrain|sft|rl|judge) ;;
  *) echo "KIND must be pretrain, sft, rl or judge (got '$KIND')" >&2; exit 2 ;;
esac
if [ "$CONTEXT" != "16384" ] && [ "$CONTEXT" != "32768" ]; then
  echo "CONTEXT must be 16384 or 32768 (got '$CONTEXT')" >&2; exit 2
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set: the generator's weights cannot be downloaded." >&2; exit 2
fi

echo "=== 1/5 system deps"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq git python3-pip >/dev/null
fi
python3 -m pip install -qq --upgrade pip >/dev/null

echo "=== 2/5 repo"
if [ -d "$WORK/.git" ]; then
  git -C "$WORK" pull --ff-only
else
  git clone --depth 1 "$REPO" "$WORK"
fi
cd "$WORK/data-gen"

echo "=== 3/5 python deps"
# vLLM pulls its own torch build; installing torch separately is the usual way to get a mismatched pair.
python3 -m pip install -qq vllm huggingface_hub pyyaml jinja2 python-dotenv
# bitsandbytes is only needed for the 4-bit path, but installing it up front means `auto` can choose it on a
# small card without a second pip round trip mid-rental.
python3 -m pip install -qq bitsandbytes || echo "  (bitsandbytes unavailable; the 4-bit path is disabled)"

# TP defaults to every GPU the box reports, which is what you are paying for.
if [ -z "${TP:-}" ]; then
  TP="$(python3 -c 'import torch;print(max(1,torch.cuda.device_count()))' 2>/dev/null || echo 1)"
fi

echo "=== 4/5 plan (no GPU work, nothing spent)"
ARGS=(--kind "$KIND" --model "$MODEL" --context "$CONTEXT" --quantization "$QUANT"
      --tp "$TP" --limit "$LIMIT" --gpu-cost "$GPU_COST")
[ -n "$LANGS" ] && ARGS+=(--langs "$LANGS")
python3 vllm_gen.py "${ARGS[@]}" --plan-only

echo "=== 5/5 generate"
[ "$PUSH" = "1" ] && ARGS+=(--push)
export DATA_GEN_SHARDS="$SHARDS" DATA_GEN_SHARD_INDEX="$SHARD_INDEX" DATA_GEN_CONTEXT="$CONTEXT"
python3 vllm_gen.py "${ARGS[@]}"

echo "=== done. Shards are under data-gen/data/out/$KIND/<lang>/ ."
[ "$PUSH" = "1" ] || echo "    PUSH=1 was not set, so nothing went to the Hub. Push with:
      python3 -c \"import sys;sys.path.insert(0,'.');from pathlib import Path;from hub import push_shards;\\
      push_shards('$KIND', sorted(Path('data/out/$KIND').glob('*/shard-*.jsonl')))\""
