# How to run

All commands assume your shell's working directory is the repo root.

## 0. One-time setup

**Local Python env** (for anything that runs outside Modal — data prep dry runs, tests, notebooks):

```bash
pip install -r requirements.txt
```

**Secrets** — copy your real values into `.env` at the repo root (already gitignored, never commit it):

| Variable | Used by | Purpose |
|---|---|---|
| `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | `data/prepare.py`, `training/new_train.py` (via `training/s3_utils.py`) | download training bins before training, upload freshly prepared bins after `prepare.py` |
| `HF_API_KEY` | `data/prepare.py`, `training/new_train.py`, `eval/eval.py` | read HF datasets, push checkpoints, load eval models |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | `training/new_train.py` | checked in addition to `HF_API_KEY` for checkpoint push |
| `WANDB_API_KEY` | wandb SDK directly (no code change needed) | training run logging |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | `modal` CLI/SDK | alternative to `modal token set` |

`.env` is loaded automatically by `training/load_config.py` (covers `training/new_train.py` and `training/modal_train.py`), `data/prepare.py`, `data/prepare_modal.py`, and `eval/eval.py`. You don't need to `source` it yourself.

**On Modal**, the same secrets are supplied via `modal.Secret.from_name(...)` instead of `.env` (the `.env` file is explicitly excluded from everything uploaded into Modal images). Create these once in your Modal workspace before running anything that references them:

```bash
modal secret create hf-secret HF_API_KEY=<your-hf-token>
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-key>
modal secret create s3-secret S3_ACCESS_KEY_ID=<...> S3_SECRET_ACCESS_KEY=<...>
```

**S3 key rotation note**: earlier versions of `training/train_config.yaml` had real S3 keys committed in plaintext (now removed from the file, but still present in git history). If those keys are still active, rotate them with your storage provider before relying on this setup.

---

## 1. Data preparation

Tokenizes HF datasets listed in `train_config.yaml`'s `data.<mode>_datasets.<eng|african>`, dedupes globally, writes `.bin` memmaps to `data.<mode>.{eng,afr}_train_data_path` / `eval_data_path`, then pushes them to S3.

**Direct CLI** (runs locally — needs CPU/network, not GPU):

```bash
python -m data.prepare --mode pretrain --data-type eng
python -m data.prepare --mode sft --data-type african --override   # wipe + reprocess from scratch, overwrite S3
python -m data.prepare --mode pretrain --data-type african --tag "<twi>"  # also prints the tag's token-id count in the output bin
```

`--override` (default off): without it, reruns skip already-processed dataset files and never overwrite an existing S3 object; with it, the local bin + resumability ledger are wiped and the S3 copy is overwritten.

**On Modal:**

```bash
modal run data/prepare_modal.py --mode pretrain --data-type eng
modal run data/prepare_modal.py --mode sft --data-type african --override
```

**Counting a tag's occurrences in a prepared (tokenized) bin**, e.g. how much Twi ended up in the African pretrain data:

```bash
modal run data/prepare_modal.py::count_tag_main --mode pretrain --data-type african --tag "<twi>"
```

---

## 2. Training

Config-driven — everything (mode, model, optimizer, FSDP sharding strategy, freeze policy, data paths) comes from `training/train_config.yaml` (override the path via `TRAIN_CONFIG_PATH` env if needed). Uses Accelerate + FSDP; `accelerate.fsdp_sharding_strategy` (`NO_SHARD` | `SHARD_GRAD_OP` | `FULL_SHARD` | `HYBRID_SHARD`) controls sharding, applied whenever more than one process is launched (`FULL_SHARD` is the ZeRO-3-equivalent default).

A training run always loads **both** the `eng_train_data_path` and `afr_train_data_path` bins configured under `data.<mode>` and mixes them per-batch (see "Language sampling" below) — there is no `--data-type` flag for training. `--data-type` only exists on the *data prep* scripts (`data/prepare.py`, `data/prepare_modal.py`), where it genuinely selects one language's raw datasets to tokenize.

**Language sampling** (`data.sampling` in `train_config.yaml`):

```yaml
data:
  sampling:
    - use_scheduled_sampling: false
    - afr_sampling_weight: 0.5
    - eng_sampling_weight: 0.5
```

Every training batch element is drawn from either the English or African bin, chosen at the configured ratio:

- **`use_scheduled_sampling: false`** (default) — the preset `eng_sampling_weight` / `afr_sampling_weight` ratio (normalized to sum to 1) is used for every batch, for the whole run.
- **`use_scheduled_sampling: true`** — the ratio starts at the preset and cosine-anneals over the course of training (`iter_num / optimizer.max_iters`) toward the *swapped* ratio by the final iteration. E.g. a preset of `eng=0.8 / afr=0.2` ends training at roughly `eng=0.2 / afr=0.8`: early training leans English-heavy for linguistic grounding, and sampling gradually shifts weight onto African languages as training proceeds. With an even `0.5/0.5` preset the schedule has no effect (the swapped ratio is identical) — set an imbalanced preset if you want the curriculum effect.

This only kicks in when both bins are present for the active `mode`; single-language runs (e.g. only one bin configured) ignore sampling weights.

**Single GPU / CPU smoke test**: there's no CLI flag for `optimizer.max_iters` — temporarily lower it (and `training.eval_interval`) in `train_config.yaml` for a first smoke run, then restore it.

```bash
python -m training.new_train
```

**Single node, multiple GPUs:**

```bash
torchrun --standalone --nproc_per_node=<num_gpus> -m training.new_train
```

**Multi-node** (run the same command on every node, only `--node_rank` changes):

```bash
torchrun --nnodes=<N> --node_rank=<0..N-1> --nproc_per_node=<gpus_per_node> \
    --master_addr=<node0_ip> --master_port=29500 -m training.new_train
```

This is the generic path — it's what you'd run by hand on vast.ai or any other bare GPU boxes, one command per box.

**On Modal** (real multi-node cluster via `modal.experimental.clustered`, topology fixed by `modal.num_nodes` / `modal.gpus_per_node` / `modal.gpu_type` in the yaml — edit the yaml and this picks it up on the next run):

```bash
modal run training/modal_train.py --mode pretrain
modal run training/modal_train.py --mode sft --override
```

`mode`/`override` are true runtime args; GPU count/node count are not (Modal's cluster shape is fixed per Function, not something a CLI flag can resize).

Checkpoints save under `TRAIN_OUT_DIR` (set by `modal_train.py` to `/data/checkpoints`, on the same persistent Modal volume training reads its data from — survives container preemption and is what `test_generation.py` reads back), then push to `training.hf_chkpt_path` on Hugging Face Hub. Outside Modal (`python -m training.new_train` / bare `torchrun`), checkpoints save under `training.out_dir` from the yaml (default `out/`).

---

## 3. Tests

```bash
pytest tests/
```

---

## 4. Eval / inference / misc scripts

| Script | How to run | Status |
|---|---|---|
| `eval/modal_eval.py` | `modal run eval/modal_eval.py::run` | Runs `eval.run_all()` (topic classification, sentiment, NER) against `BeardedMonster/SabiYarn-125M-finetune`, logs to the `sabiyarn_v2` volume |
| `test_generation.py` | `modal run test_generation.py::main` | Loads the most recently modified `ckpt_*` dir under `/data/checkpoints/` (the same volume `modal_train.py` writes to) via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and generates from a couple of default prompts. Pass `--checkpoint-dir <path>` to target a specific checkpoint instead of "latest" |
| `inference/modal_hosting.py` | `modal deploy inference/modal_hosting.py` | Fixed for modal 1.5.1 (previously imported `Mount`/`build`/`gpu`, which no longer exist as top-level `modal.*` names); serves `BeardedMonster/SabiYarn-125M` behind a FastAPI `/predict` endpoint. Not otherwise changed/verified end-to-end — GPU-side behavior needs a real Modal deploy to confirm |
| `data/data_distribution.py` | `modal run data/data_distribution.py::run` | Dataset language/length distribution analysis + plots, writes to the `sabiyarn_data_dist` volume |
| `data/prepare_data_for_tokenizer_training.py` | — | **Currently broken** — loads `./config/mistral_config.yaml`, which doesn't exist in this repo; needs a real config path or removal, out of scope of this pass |
| `training/tokenizer_training.ipynb`, `data/tokenization (1).ipynb` | open in Jupyter, run cells top to bottom | Exploratory tokenizer-training notebooks; both now read HF tokens from env (`HF_API_KEY`/`HF_WRITE_TOKEN`) via `.env` instead of hardcoded values |

`eval/eval.py` is a library (`run_all`, `topic_classification`, `sentiment_analysis`, `NER`, ...) imported by `eval/modal_eval.py`, not directly runnable on its own.
