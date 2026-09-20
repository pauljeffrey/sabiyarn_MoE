# How to run

All commands assume your shell's working directory is the repo root.

## 0. One-time setup

**Local Python env** (data prep, tests, notebooks — *not* the training box; on vast use `requirements-train.txt`, see 2.4):

```bash
pip install -r requirements.txt
```

**Secrets** — copy your real values into `.env` at the repo root (already gitignored, never commit it):

| Variable | Used by | Purpose |
|---|---|---|
| `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | `data/prepare.py`, `training/new_train.py` (via `training/s3_utils.py`) | download training bins before training, upload freshly prepared bins after `prepare.py` |
| `HF_API_KEY` | `data/prepare.py`, `eval/eval.py`, the notebooks' `huggingface-cli login` | **read** token: download HF datasets/models |
| `HF_WRITE_TOKEN` / `HF_TOKEN` | `training/new_train.py`, `training/push_s3_checkpoint_to_hf.py`, `training/push_model_code_to_hf.py`, `training/tokenizer_training.ipynb` | **write** token: every upload to the Hub. The push paths check `HF_WRITE_TOKEN`, then `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`, then `HF_API_KEY` — a read-only token in the last slot fails the push with a 401, so keep a real write token in one of the first two |
| `WANDB_API_KEY` | wandb SDK directly (no code change needed) | training run logging |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | `modal` CLI/SDK | alternative to `modal token set` |

`.env` is loaded automatically by `training/load_config.py` (covers `training/new_train.py` and `training/modal_train.py`), `data/prepare.py`, `data/prepare_modal.py`, and `eval/eval.py`. You don't need to `source` it yourself.

**On Modal**, the same secrets are supplied via `modal.Secret.from_name(...)` instead of `.env` (the `.env` file is explicitly excluded from everything uploaded into Modal images). Create these once in your Modal workspace before running anything that references them:

```bash
modal secret create hf-secret HF_API_KEY=<your-hf-read-token> HF_WRITE_TOKEN=<your-hf-write-token>
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

Config-driven — everything (mode, model, optimizer, schedule, sharding strategy, freeze policy, data paths) comes from `training/train_config.yaml` (override the path via `TRAIN_CONFIG_PATH`). Uses Accelerate; `accelerate.fsdp_sharding_strategy` (`NO_SHARD` | `SHARD_GRAD_OP` | `FULL_SHARD` | `HYBRID_SHARD`) controls FSDP sharding whenever more than one process is launched.

A training run always loads **both** the `eng_train_data_path` and `afr_train_data_path` bins configured under `data.<mode>` and mixes them per-batch (see "Language sampling" below) — there is no `--data-type` flag for training. `--data-type` only exists on the *data prep* scripts.

### 2.1 Which launcher: DDP or FSDP

This model is ~306M parameters, so the whole training state (fp32 weights + grads + Adam moments) is ~5 GiB. **FSDP saves you ~1 sample of batch size and costs ~15% throughput** (it re-gathers weights every micro-batch). Activations dominate memory, not weights.

| | Launcher | When |
|---|---|---|
| **DDP** (recommended for 2-4 GPUs) | `torchrun --standalone --nproc_per_node=<N> -m training.new_train_ddp` · Modal: `modal run training/modal_train_ddp.py` | default; required for Muon |
| FSDP | `torchrun --standalone --nproc_per_node=<N> -m training.new_train` · Modal: `modal run training/modal_train.py` | many GPUs / bigger model; AdamW only |

`scripts/estimate_training_plan.py` prints micro-batch sizes and wall-clock estimates per GPU type (`python scripts/estimate_training_plan.py --tokens 50e9`). Its throughput numbers are assumptions until you read a real `tokens_per_sec` from the `step` log line: `hours = tokens / tokens_per_sec / 3600`.

Global batch = `train_batch_size × gradient_accumulation_steps × block_size` tokens **regardless of GPU count** (the yaml's accumulation is global and is divided by the world size). Default: `6 × 32 × 4096 = 786,432` tokens/step. If your GPUs only fit a smaller micro-batch, raise accumulation to keep `train_batch_size × gradient_accumulation_steps ≈ 192` (e.g. `3 × 64`). Set `optimizer.max_tokens` and `max_iters` is derived from the *real* tokens per step at startup (`50e9 / 786,432 ≈ 63.6K` steps) — the `max_iters_from_token_budget` log line shows it.

### 2.2 Recommended hyperparameters

These are the defaults now in `train_config.yaml`, for a ~300M model on a ~50B-token budget (~178 tokens/param, ~9× Chinchilla-optimal). They come from the recent literature, not from a sweep on your data — treat the learning rate as a starting point and confirm it with a short pilot (2.7 shows the recipe).

| Setting | Value | Why |
|---|---|---|
| Optimizer | **AdamW** | see 2.7: Muon's edge over a well-tuned AdamW shrinks at this scale and token ratio |
| Betas / weight decay / clip | `0.9, 0.95` / `0.1` / `1.0` | standard LLM pretraining setting; unchanged |
| Peak LR | **6e-4** | published optima run ≈1e-3 at ~0.13B down to ≈4e-4 at ~1B params; sweep `{4e-4, 6e-4, 1e-3}` on a ~300M-token pilot and pick by `eval/val_bpb` |
| Warmup | **1000 steps** (~0.8B tokens, ~1.6%) | typical 0.1–2% of total steps (TinyLlama used 2000); longer warmup helps if the router/aux loss is unstable early |
| Schedule | **WSD** (`scheduler: wsd`) | warmup → constant → cooldown; cooldown of 10–20% of steps with a `(1−√t)` shape matched or beat cosine ([Hägele et al. 2024](https://arxiv.org/abs/2405.18392)) and lets you cool down any stable-phase checkpoint, i.e. stop at 25B or 50B without wasting the run |
| Cooldown | `wsd_decay_frac: 0.2`, `wsd_decay_shape: sqrt` | the paper found cooldown benefits plateau around 20% for moderate runs, and even ~5% suffices for very long ones |
| Final LR | **0** (`min_lr: 0.0`) | linear-decay-to-zero beat cosine-to-10% across scales, with the benefit growing with dataset size ([Bergsma et al. 2025](https://arxiv.org/abs/2502.15938)) |
| Precision | `param_dtype: float32` + bf16 autocast (`dtype: bf16`) | fp32 master weights; pure-bf16 parameters silently drop small updates. `scheduler: linear` (also to 0) is an equally good alternative if you don't need the early-stop property |

To use linear-to-zero instead: `scheduler: "linear"`, `lr_decay_iters` = `max_iters` (≈ 63.6K for 50B tokens at 786K/step), `min_lr: 0.0`.

### 2.3 Starting from step 0 with the last checkpoint

`training.fresh_start: true` (or `FRESH_START=1`) loads the latest checkpoint's **weights** (`init_from: "resume"`) and, when the optimizer is unchanged, its **optimizer state**, but restarts everything that tracks progress: `iter_num = 0` (warmup, LR schedule and scheduled sampling begin at 0), best-val and HF-push tracking reset, and all outputs go to a **new run dir** so the old run's checkpoints and iteration numbers are untouched. It applies once: a later restart finds the new run dir (marked `fresh_started`) and resumes it normally, so you can keep the flag on.

- **Don't expect the optimizer state to matter much.** With `beta2 = 0.95`, Adam's moments re-estimate in ~20 steps, which is nothing next to a ~1000-step warmup. Set `resume_optimizer_state: false` to skip it and avoid the incompatibility risk of an old FSDP-format `resume_state` on a DDP box.
- The optimizer state is **skipped automatically** if the optimizer changed (e.g. AdamW → Muon), logged as `optimizer_state_not_restored`.
- **Check `resume_sanity_check` in the first log lines.** `current_sanity_loss` should be close to `saved_sanity_loss`; a gap of ~3-4 nats means the loaded weights are not the trained ones (see the resume-loss investigation). Do this with `TEST_RUN=1` before paying for a real run.

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

**Dry run / checkpoint sanity check** (`training.test_run: true` in `train_config.yaml`): loads the model and `model.reference_repo`, runs the weight-deviation check, then generates from 5 fixed prompts (one per pretraining language, ~20-30 words each, 150 new tokens) with **both** models under **both** `do_sample` and beam search, prints them side by side, then runs **one eval** — logging its train/val loss and the delta against the best val loss recorded in the checkpoint's `trainer_state.json` — and stops. No training step, and nothing is saved locally, uploaded to S3, or pushed to the Hub. Set it back to `false` to train. Toggle it in the yaml, or per-invocation with the `TEST_RUN` env var (`TEST_RUN=1` / `TEST_RUN=0`, checked before the yaml) so a dry run needs no file edit. The prompts live in `Trainer._STARTUP_PROMPTS` (`training/new_train.py`) and are plain untagged text — no `<yor>`/`<eng>` language tag — so they probe the model the way an untagged inference prompt would; edit them freely. The comparison itself runs at the start of *every* launch — only the stop-afterwards part is gated on `test_run`.

### 2.4 Run on vast.ai (or any bare GPU box)

1. **Rent** an instance with the GPUs you want (2× or 4× of one type; vast bills the whole box). Pick a CUDA PyTorch image and **≥150 GB disk**: the bins plus checkpoints (each fp32 checkpoint is ~1.2 GB, and `resume_state` ~4 GB, replaced in place).
2. **Set up** (once):

   ```bash
   ssh -p <vast_ssh_port> root@<vast_ip>
   git clone <your repo> sabiyarn_MoE && cd sabiyarn_MoE
   pip install -r requirements-train.txt    # NOT requirements.txt -- see the note below
   nano .env   # create it: HF_API_KEY, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, (WANDB_API_KEY)
   python -m data.prefetch_bins --mode pretrain --write-env   # downloads the bins, writes their paths into .env
   ```

   Use `requirements-train.txt` on the box: it pins `transformers==5.14.1` (what saved your checkpoints and what Modal actually ran, and what your `modeling.py` targets) and leaves torch to the image. `requirements.txt` still pins transformers 4.55 / huggingface_hub 0.34 and would give you a different, possibly broken stack. Use a CUDA PyTorch image with torch ≥ 2.8.

   Training only *checks* that the bins exist — it never downloads them — so `--write-env` (or copying the two printed lines into `.env` by hand) is required.

   **Pick the micro-batch for your GPU** (defaults are for an 80 GB card): set `TRAIN_BATCH_SIZE` and `GRAD_ACCUM_STEPS` in `.env` or the command line, keeping `TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS ≈ 192`. Rough starting points at 4096 tokens: 45/48 GB → `3` / `64`; 80 GB → `6` / `32`; 96 GB → `8` / `24` (`scripts/estimate_training_plan.py` has the reasoning). The `step` log line prints `peak_mem_gib`; if it is far below the card's memory raise the batch, and lower it on any `CUDA out of memory`.
3. **Dry run first** (loads weights, runs the deviation checks + one eval, saves nothing):

   ```bash
   TEST_RUN=1 torchrun --standalone --nproc_per_node=<N> -m training.new_train_ddp
   ```

   Look for `resume_sanity_check` (loaded weights match the checkpoint), `curated_eval`, and `eval` losses. Fix anything odd *before* the long run.
4. **Train** inside `tmux` so it survives your SSH session:

   ```bash
   tmux new -s train
   TEST_RUN=0 torchrun --standalone --nproc_per_node=<N> -m training.new_train_ddp 2>&1 | tee train.log
   # detach: Ctrl-b d      reattach: tmux attach -t train
   ```

5. **If it hangs or fails at startup**, in rough order of likelihood on rented boxes:
   - *Hangs right after `distributed_env_check`, or NCCL errors*: add `NCCL_P2P_DISABLE=1` (some hosts have broken GPU peer-to-peer) and `NCCL_DEBUG=INFO` to see why; also make sure the instance has a large `/dev/shm` (vast lets you set shared memory when renting; a tiny default causes NCCL bus errors).
   - *`CUDA out of memory` on the first steps*: lower `TRAIN_BATCH_SIZE` (raise `GRAD_ACCUM_STEPS` to match), see above.
   - *`Missing or empty training data files`*: run the `prefetch_bins --write-env` step, and start training from the repo root so `.env` is found.
   - *`wandb_skipped`* is normal without a `WANDB_API_KEY` (it no longer blocks on a login prompt); MLflow is unaffected.
   - *A `trust_remote_code` `[y/N]` prompt*: fixed in code, but if you ever see one, the tokenizer/model call is missing `trust_remote_code=True`.
   - *torch.compile errors on a single GPU*: `training.compile` is now `false` by default; leave it off unless you have tried it.
6. **If the instance dies or is reclaimed**: rent another, repeat step 2, and run the same command. Checkpoints are pushed to S3 on every save; the trainer finds the newest run there and resumes it (it is *not* another fresh start).
7. **Multi-node** (FSDP, rarely worth it at this size): same command on every node with `--nnodes=<N> --node_rank=<0..N-1> --master_addr=<node0_ip> --master_port=29500`.

Single GPU: `python -m training.new_train`. Smoke run: temporarily lower `optimizer.max_tokens` (e.g. `3e6`) and `training.eval_interval` via a copied config (`TRAIN_CONFIG_PATH=/tmp/smoke.yaml`).

### 2.5 Run on Modal

```bash
modal run training/modal_train_ddp.py --mode pretrain            # DDP, single node (recommended; needed for Muon)
modal run training/modal_train.py --mode pretrain                 # FSDP (real multi-node via modal.experimental.clustered)
modal run --detach training/modal_train_ddp.py --mode pretrain    # keep running after you close the terminal
```

GPU type and count come from `modal.gpu_type` / `modal.gpus_per_node` / `modal.num_nodes` in the yaml (Modal's cluster shape is fixed per Function, not a CLI flag; `mode`/`override` are runtime args). Data is synced from S3 onto the `sabiyarn-data` volume before training starts. Checkpoints save under `/data/<training.out_dir>` on that volume (and go to S3 / the Hub as configured); MLflow runs go to `/data/mlruns`. Secrets come from `.env` at launch time (`modal.Secret.from_dotenv`), so put `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` there too if the `modal` CLI isn't logged in (`modal token new`).

A short A/B or pilot on Modal: copy the yaml, change `optimizer.max_tokens`, and point at it with `TRAIN_CONFIG_PATH=... modal run ...` (the yaml is baked into the image at launch).

### 2.6 Viewing the MLflow run once training starts

Look for `mlflow_started` (with the `run_id`) and, on a bare box, `mlflow_ui_started` in the log. Everything below logs at every `training.log_interval` steps and every eval; experiment name `mlflow.experiment` (default `sabiyarn-ablations`).

**vast.ai / bare box** — the trainer serves the UI itself on port 5000 (`mlflow.ui.enabled: true`; `MLFLOW_UI_ENABLED=0/1`, `MLFLOW_UI_PORT` override), so you only need a tunnel from your laptop:

```bash
ssh -N -L 5000:localhost:5000 -p <vast_ssh_port> root@<vast_ip>      # then open http://localhost:5000
```

The UI stops when training ends. To browse a finished run later, on the box: `MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000`, or copy the folder home: `rsync -avz -e "ssh -p <port>" root@<ip>:~/sabiyarn_MoE/mlruns ./mlruns` and run the same `mlflow ui` locally.

**Modal** — runs are written to the volume and flushed when the job exits (or crashes), so they appear *after* it finishes:

```bash
modal volume get sabiyarn-data mlruns ./mlruns
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns        # http://localhost:5000
```

For a live view of a Modal run, host a tracking server somewhere reachable and put `MLFLOW_TRACKING_URI=http://<host>:5000` in `.env` (forwarded into the container). Modal runs skip the in-container UI by default.

**What to watch**

| Metric | Meaning |
|---|---|
| `eval/val_bpb`, `eval/val_ce` | the number to compare runs/optimizers on (bpb is tokenizer-independent); should fall steadily |
| `train/ce_loss`, `train/bpb` | pure cross-entropy (no MoE aux term); noisy — it is one micro-batch per log line |
| `train/loss` | CE + MoE aux loss, what is actually optimised |
| `curated/ce`, `curated/<lang>_ce` | hand-written probe set (`data/curated_eval.jsonl`, 5 languages × 20). Far below val ⇒ suspect the bins; far above ⇒ suspect the model |
| `train/grad_norm` | spikes / sustained growth ⇒ LR too high or instability; frequent clipping at `grad_clip` ⇒ same |
| `lr` | confirm the warmup / WSD shape you configured |
| `train/tokens_seen`, `train/lifetime_tokens_seen`, `train/tokens_seen_<bin>` | tokens consumed by optimizer steps (persisted in `trainer_state.json`, so resumes continue the count). After a fresh start `tokens_seen` restarts at 0 and `lifetime_tokens_seen` keeps the earlier runs' total; `<bin>` splits it by data file (eng / african) |
| `train/tokens_per_sec`, `train/mfu`, `train/tflops_per_gpu` | real throughput — use it to re-estimate the time and cost |
| `moe/lb_loss`, `moe/layer*_max_util`, `moe/layer*_entropy` | router health: max_util → 1 or entropy → 0 = collapse onto few experts |
| system metrics tab | GPU utilisation / memory: low utilisation means data loading or comms is the bottleneck |

Select several runs and use **Compare** to overlay curves (e.g. AdamW vs Muon). After a resume, set `MLFLOW_RUN_ID=<id>` to append to the same run instead of starting a new one. Disable with `mlflow.log: false`.

### 2.7 AdamW vs Muon (opt-in)

`optimizer.name: muon` switches the hidden matrices (attention q/k/v/out projections and every expert's `fc`/`proj` matrix) to **Muon** and keeps **AdamW** for embeddings / tied `lm_head`, position embeddings, MoE router gates and all norms/biases (`training/muon.py`). Q, K and V are orthogonalized separately and each expert as its own matrix. Update RMS is matched to AdamW's (`0.2·√max(m,n)`, [Liu et al. 2025](https://arxiv.org/abs/2502.16982)), so it reuses `learning_rate` and `weight_decay` unchanged; momentum `0.95` (Nesterov), 5 Newton–Schulz steps, per [Jordan](https://kellerjordan.github.io/posts/muon/).

- **Needs DDP or one GPU** — it errors at startup under FSDP, which shards the matrices Muon must see whole.
- **Not the default.** A careful benchmark against a tuned AdamW ([Wen et al. 2025](https://arxiv.org/abs/2509.02046)) found matrix optimizers' speedup shrinking from ≤1.4× at 0.1B to ~1.1× at 1.2B, and at 16× Chinchilla tokens/param Soap and NAdamW beat Muon at 130M/300M. You are at ~9×. Reported 2× gains mostly reflect weak AdamW baselines.
- **Untested here**: the optimizer has unit tests on CPU but has not run on a GPU or this MoE yet.
- **A/B before committing** (~300M tokens each, same seed and schedule, ~1.5–5 h on a rented box). Each arm starts from the same source checkpoint into its **own** output dir, and HF pushes are off, so the pilots can't be mistaken for your real run (a later real launch auto-resumes the newest run dir under `out_280M`, and would otherwise pick a pilot):

  ```bash
  SRC=out_280M/<the_original_run_dir>      # must exist locally (the dry run in 2.4 downloads it)
  for opt in adamw muon; do
    sed -e "s/name : \"adamw\"/name : \"$opt\"/" \
        -e "s/max_tokens : 50e9/max_tokens : 3e8/" -e "s/warmup_iters : 1000/warmup_iters : 200/" \
        -e "s/eval_interval : 150/eval_interval : 50/" -e "s/eval_iters : 200/eval_iters : 20/" \
        -e "s/test_run: true/test_run: false/" -e "s/resume_optimizer_state: true/resume_optimizer_state: false/" \
        -e "s#out_dir : \"out_280M\"#out_dir : \"out_pilot_$opt\"#" \
        -e "s#resume_run_dir: \"\"#resume_run_dir: \"$SRC\"#" \
        -e "s#hf_chkpt_path: \"[^\"]*\"#hf_chkpt_path: \"\"#" \
        -e "s/^  run_name: null/  run_name: $opt/" \
        training/train_config.yaml > /tmp/$opt.yaml
    TRAIN_CONFIG_PATH=/tmp/$opt.yaml torchrun --standalone --nproc_per_node=<N> -m training.new_train_ddp
  done
  ```

  Compare `eval/val_bpb` at equal tokens in MLflow (the two runs are named `adamw` and `muon`). Keep Muon only if it wins by enough to matter (≳5% fewer tokens to the same bpb); otherwise stay on AdamW. Delete `out_pilot_*` afterwards.

Checkpoints are compatible between the two (weights only); optimizer state is not, and is skipped automatically.

### 2.8 Where checkpoints go

Modal: `/data/<training.out_dir>` on the `sabiyarn-data` volume. Bare box: `training.out_dir` (default `out_280M/`). Both also push to S3 (replace-in-place) and to `training.hf_chkpt_path` on the Hub when the eval loss is not a regression against the last push.

### 2.9 Faster attention masking and sparse MoE (`attention_impl`, `moe_dispatch`)

Both are opt-out/opt-in switches in `train_config.yaml` and produce the **same math** as before; they only change memory and speed.

| Setting | Values | What it does |
|---|---|---|
| `model.moe_dispatch` | `sparse` (default) / `dense` | `sparse`: each expert only processes the tokens the router sent to it (~half the expert FLOPs and expert activation memory for top-2 of 4). `dense`: every expert processes every token, then the top-2 are kept (the original path). |
| `training.attention_impl` | `sdpa_mask` (default) / `flex` | how packed documents are kept from attending to each other. `sdpa_mask` builds a `(B,1,T,T)` boolean mask per batch (~4-5 GiB per sample of activations, slower masked kernel). `flex` uses a FlexAttention block mask: no T×T tensor, and blocks lying across a document boundary are skipped (needs torch ≥ 2.5 and a GPU; compiles on the first step, ~1 min). |
| `model.code` | `local` (default) / `hub` | `local` builds the network from this repo's `sabiyarn/model/modeling.py` and loads only the weights; `hub` runs whatever `modeling.py` is stored on the Hub next to the weights. These changes live in the local file, so `local` is required for them to take effect. |

**Verify before a long run** (a couple of minutes, no training): run the same `TEST_RUN=1` dry run three times, changing one setting each time, and compare `resume_sanity_check`'s `current_sanity_loss` (same weights, same batch): `moe_dispatch: dense` vs `sparse`, and `attention_impl: sdpa_mask` vs `flex`. They should agree to ~1e-3 (bf16 noise). If `flex` disagrees or fails to compile on your GPU, stay on `sdpa_mask`. Unit tests already check equality on CPU (forward and gradients for sparse MoE; forward for flex; flex gradients on GPU when one is available).

`python scripts/estimate_training_plan.py --flex` (and `--dense-moe`, `--cce`) models the batch size and time for each combination; the default assumes sparse MoE with the dense mask. `mfu.py` counts only the top-k experts when `moe_dispatch: sparse`, so the reported MFU stays comparable.

### 2.10 Evaluating a checkpoint (`eval_suite/`)

```bash
pip install -r requirements-train.txt -r requirements-eval.txt
python -m eval_suite.run --model out_280M/<run>/ckpt_best --tasks all --langs all --limit 200 --name base_10k
python -m eval_suite.run --model <sft ckpt> --style chat --tasks translation,topic,sentiment,mmlu --name sft_v1
python -m eval_suite.compare eval_results/base_10k/results.json eval_results/sft_v1/results.json
```

Writes `eval_results/<name>/results.json`, `summary.md` (language × task table) and every prediction to `predictions/<task>_<lang>.jsonl` for error analysis. `--limit 0` runs the full test sets; the default 200 per (task, language) keeps a full sweep to tens of minutes on one GPU, at the cost of ±3–7 points of sampling noise per cell (accuracy CIs are in `results.json`).

| Task | Benchmark (test split) | Your languages covered | Metric |
|---|---|---|---|
| Translation, English ↔ X | FLORES(+)-200 devtest (**gated**: accept the terms on the Hub with the account behind `HF_API_KEY`), else MAFAND-MT (news; English pairs only) | FLORES: yor hau ibo twi aka ewe fon fuv · MAFAND: yor hau ibo pcm twi | chrF++ (headline), chrF, BLEU (sacrebleu); optional AfriCOMET |
| Topic classification | SIB-200 (7 topics), MasakhaNEWS (7 topics, news) | SIB: yor hau ibo twi aka ewe fon fuv · News: yor hau ibo pcm | accuracy, macro-F1, majority baseline |
| Sentiment | AfriSenti (Twitter) | hau ibo yor pcm twi | accuracy, macro-F1 |
| NER | MasakhaNER 2.0 | yor hau ibo pcm twi ewe fon | **entity-level** F1 (exact type + span), token accuracy |
| MMLU | AfriMMLU (IrokoBench) + original English MMLU | yor hau ibo ewe twi + eng | accuracy with 95% CI, chance = 25% |

- **Efik, Urhobo and Fulah have no standard benchmark** for any of these tasks (Ewe and Fon have no MAFAND English pair either, only FLORES). The suite reports them as `n/a`. For translation you can supply your own held-out pairs: `--custom-translation my_pairs.jsonl` with lines `{"lang": "efi", "eng": "...", "xx": "..."}`.
- **Prompt styles.** `--style tag` (default) uses the multitask pretraining format (`<translate> {src} <yor>`, `<classify> {text} <topic> :`, `<NER> {tokens} <tag> :`) zero-shot, for base checkpoints. `--style chat` renders the SFT chat template with an English instruction, for instruction-tuned checkpoints (NER is tag-style only).
- **Classification and MMLU are scored by likelihood** (which label / which of A-D is most probable given the prompt), not by parsing free-form generations, so a small model isn't penalised for formatting. MMLU is 5-shot (`--few-shot`) and, at 280M parameters, **expect scores near chance (25%)** — treat it as a sanity check, not a headline.
- **Contexts mirror pretraining**: `tag` prompts are preceded by two `</s>`, like documents in the training bins.
- **Generation** is greedy by default (`--num-beams`, `--repetition-penalty` to change) and batches only prompts of identical token length, because positions are absolute and left-padding would shift them.
- **Comparing with older numbers**: `eval/eval.py` (the SabiYarn-125M evaluation) scores NER by per-sentence label *presence*, which is far more lenient than entity F1, and resolves topics with keyword rules (e.g. any "africa" → politics); the new numbers are not comparable to it, and are stricter.
- **AfriCOMET** (`--africomet <model id>`, needs `pip install unbabel-comet`; verify the exact model id on the Hub) adds a reference-based learned score. It is not run in the tests. If you later use AfriCOMET as an RL reward, keep reporting chrF++ and BLEU alongside it: a rising AfriCOMET with flat or falling chrF++ is the signature of reward hacking.

**Sources for the hyperparameter choices:** [Hägele et al. 2024 (WSD)](https://arxiv.org/abs/2405.18392) · [Bergsma et al. 2025 (linear decay to zero)](https://arxiv.org/abs/2502.15938) · [Wen et al. 2025 (optimizer benchmark)](https://arxiv.org/abs/2509.02046) · [Liu et al. 2025 (Moonlight/Muon)](https://arxiv.org/abs/2502.16982) · [Jordan 2024 (Muon)](https://kellerjordan.github.io/posts/muon/).

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
| `training/push_s3_checkpoint_to_hf.py` | `python training/push_s3_checkpoint_to_hf.py` or `modal run training/push_s3_checkpoint_to_hf.py` | Downloads the latest checkpoint's weights (`ckpt_<iter>/`) from the newest S3 run dir under `checkpoints/<training.out_dir>/` and pushes them to `training.hf_chkpt_path`. `--best` pushes `ckpt_best/` instead; `--dry-run` only prints what it would push; `--run-dir`/`--repo`/`--mode` override the defaults. Needs `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY` and `HF_TOKEN` in the env or `.env` |
| `data/data_distribution.py` | `modal run data/data_distribution.py::run` | Dataset language/length distribution analysis + plots, writes to the `sabiyarn_data_dist` volume |
| `data/prepare_data_for_tokenizer_training.py` | — | **Currently broken** — loads `./config/mistral_config.yaml`, which doesn't exist in this repo; needs a real config path or removal, out of scope of this pass |
| `training/tokenizer_training.ipynb`, `data/tokenization (1).ipynb` | open in Jupyter, run cells top to bottom | Exploratory tokenizer-training notebooks; both now read HF tokens from env (`HF_API_KEY`/`HF_WRITE_TOKEN`) via `.env` instead of hardcoded values |

`eval/eval.py` is a library (`run_all`, `topic_classification`, `sentiment_analysis`, `NER`, ...) imported by `eval/modal_eval.py`, not directly runnable on its own.
