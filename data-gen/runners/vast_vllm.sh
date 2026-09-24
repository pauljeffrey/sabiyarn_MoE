#!/usr/bin/env bash
# Bootstrap a vast.ai box and generate with vLLM. Designed to be pasted into a fresh instance's shell.
#
#   export HF_TOKEN=...  HF_WRITE_TOKEN=...
#   bash runners/vast_vllm.sh pretrain            # or sft / rl / judge
#
# Picking an instance on vast (cheapest first; verify current prices, they move hourly):
#
#   gemma-3-27b-it, fp8      1 x 48GB (A6000/L40S)     ~$0.40-0.70/hr   cheapest viable
#   gemma-3-27b-it, bf16     1 x 80GB (A100/H100)      ~$0.80-2.00/hr
#   gpt-oss-120b (MXFP4)     1 x 80GB (A100/H100)      ~$0.80-2.00/hr   MoE, ~5B active, fast
#   either, tensor-parallel  2 x 24GB (2x RTX 4090)    ~$0.50-0.90/hr   add --tp 2; needs fp8 for 27B
#
# Ask for >= 150GB disk (the 120b weights are ~60GB and HF caches a copy) and a CUDA 12.4+ image.
# `vast` bills by the second, so the run below pushes to the Hub incrementally -- if the box dies or you
# are outbid, you lose only the shard in flight, and re-running skips everything already pushed.
set -euo pipefail

KIND="${1:-pretrain}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-1}"
LIMIT="${LIMIT:-0}"
LANGS="${LANGS:-}"
GPU_COST="${GPU_COST:-1.00}"
SHARDS="${SHARDS:-1}"
REPO="${DATA_GEN_GIT_REPO:-https://github.com/pauljeffrey/sabiyarn_MoE}"
WORK="${WORK:-/workspace/sabiyarn}"

echo "=== 1/4 system deps"
apt-get update -qq && apt-get install -y -qq git python3-pip >/dev/null

echo "=== 2/4 repo"
if [ ! -d "$WORK" ]; then git clone --depth 1 "$REPO" "$WORK"; fi
cd "$WORK/data-gen"

echo "=== 3/4 python deps (vllm is the big one; ~5 min)"
pip install -q --upgrade pip
pip install -q "vllm>=0.10" huggingface_hub pyyaml jinja2 python-dotenv pydantic
python -c "import vllm; print('vllm', vllm.__version__)"

# Keys: prefer the environment (do not bake secrets into the image or the repo).
: "${HF_TOKEN:?set HF_TOKEN before running}"
export HF_WRITE_TOKEN="${HF_WRITE_TOKEN:-$HF_TOKEN}"
export HF_HOME="${HF_HOME:-/workspace/hf}"          # big disk, not the small root volume
export DATA_GEN_OUTPUT_DIR="${DATA_GEN_OUTPUT_DIR:-/workspace/datagen}"

echo "=== 4/4 generate: kind=$KIND model=$MODEL tp=$TP shards=$SHARDS"
ARGS=(--kind "$KIND" --model "$MODEL" --tp "$TP" --gpu-cost "$GPU_COST" --push)
[ "$LIMIT" != "0" ] && ARGS+=(--limit "$LIMIT")
[ -n "$LANGS" ] && ARGS+=(--langs "$LANGS")

if [ "$SHARDS" = "1" ]; then
  # Unbuffered + tee: vast's web terminal drops scrollback, so keep a log on disk.
  DATA_GEN_SHARDS=1 python -u vllm_gen.py "${ARGS[@]}" 2>&1 | tee "/workspace/${KIND}.log"
else
  # One process per GPU, each taking a disjoint slice of the same deterministic plan.
  for i in $(seq 0 $((SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$i" DATA_GEN_SHARDS="$SHARDS" DATA_GEN_SHARD_INDEX="$i" \
      python -u vllm_gen.py "${ARGS[@]}" > "/workspace/${KIND}-w${i}.log" 2>&1 &
  done
  wait
fi

echo "done. Shards pushed to the Hub as they were written; check with:"
echo "    python hub.py --status"
