# How to run

All commands assume your shell's working directory is the repo root.

## 0. One-time setup

**Python env** — one file for everything (training on vast.ai, Modal images, eval suite, data prep, tests). Use a Python 3.11 venv, or on a rented GPU box a CUDA PyTorch image (torch ≥ 2.8) so torch isn't re-downloaded:

```bash
pip install -r requirements.txt
```

`transformers` is pinned to `5.14.1` (the version that saved your checkpoints, and the API `sabiyarn/model/modeling.py` is written against); `huggingface_hub` is deliberately unpinned because transformers 5 needs 1.x. The Modal training images install this same file, so Modal and vast run identical versions.

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

Config-driven — everything (mode, model, optimizer, schedule, sharding strategy, freeze policy, data paths) comes from `training/train_config.yaml` (override the path via `TRAIN_CONFIG_PATH`). Uses Accelerate with plain DDP by default (`ddp.enabled: true`); set `ddp.enabled: false` to shard with FSDP, in which case `accelerate.fsdp_sharding_strategy` (`SHARD_GRAD_OP` | `FULL_SHARD` | `HYBRID_SHARD`) applies.

A training run always loads **both** the `eng_train_data_path` and `afr_train_data_path` bins configured under `data.<mode>` and mixes them per-batch (see "Language sampling" below) — there is no `--data-type` flag for training. `--data-type` only exists on the *data prep* scripts.

### 2.1 Launching: DDP is the default

`ddp.enabled` in `train_config.yaml` selects how a multi-GPU run is wrapped (env override: `DISTRIBUTED=ddp|fsdp`). **The default is `ddp.enabled: true`** — plain DistributedDataParallel, every GPU holds the whole model. For a ~306M-parameter model the whole training state (fp32 weights + grads + Adam moments) is ~5 GiB, so **FSDP would save you ~1 sample of micro-batch and cost ~15% throughput** (it re-gathers weights every micro-batch); activations dominate memory, not weights. Muon requires DDP.

| Where | Command |
|---|---|
| Any bare box / vast.ai, N GPUs | `torchrun --standalone --nproc_per_node=<N> -m training.new_train` |
| One GPU | `python -m training.new_train` |
| Modal | `modal run training/modal_train.py --mode pretrain` |
| FSDP instead | set `ddp.enabled: false` (and `accelerate.fsdp_sharding_strategy`), or `DISTRIBUTED=fsdp` on the command line |

There is one trainer (`training/new_train.py`) and one Modal launcher (`training/modal_train.py`); DDP vs FSDP is chosen by `ddp.enabled` (default `true` = DDP, bf16 autocast, fp32 master params).

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
- **`use_scheduled_sampling: true`** — the mixture changes as training progresses. It is a pure function of the step (`iter_num / max_iters`), so every rank computes the same weights, a resumed run continues on the curve, and the credit-based sampler (see 2.11) tracks whatever weights it is handed within one batch element. Two forms:
  - **cosine swap** (default when no `schedule` is given): starts at the preset and cosine-anneals toward the *swapped* ratio by the last step (`eng=0.8 / afr=0.2` ends at `0.2 / 0.8`). With an even preset there is no effect.
  - **explicit schedule** (recommended — you see exactly what happens): `schedule: [[progress, afr_weight], ...]`, linearly interpolated, flat before the first and after the last knot; English gets `1 - afr_weight`.

  ```yaml
    - use_scheduled_sampling: true
    - schedule: [[0.0, 0.45], [0.3, 0.45], [0.7, 0.65], [1.0, 0.65]]   # 45% African for the first 30%, ramp to 65% by 70%, hold
  ```

  The `step` log line prints the current `eng_sampling_weight` / `afr_sampling_weight`, and `tokens_seen_by_bin` shows what was actually consumed. The train-loss probe inside eval uses the *starting* mixture (not the scheduled one) so eval curves stay comparable across the run. The mixture cannot be edited live from a file on purpose: ranks must draw the same bins in lockstep, and a per-rank file read could desynchronise them. To change the plan mid-run, edit `schedule` and restart — it resumes.

This only kicks in when both bins are present for the active `mode`; single-language runs (e.g. only one bin configured) ignore sampling weights.

**Dry run / checkpoint sanity check** (`training.test_run: true` in `train_config.yaml`): loads the model and `model.reference_repo`, runs the weight-deviation check, then generates from 6 fixed prompts (one per pretraining language plus a short English stub, 150 new tokens) with **both** models under **both** sampling and greedy decoding, prints them side by side, then runs **one eval** — logging its train/val loss and the delta against the best val loss recorded in the checkpoint's `trainer_state.json` — and stops. No training step, and nothing is saved locally, uploaded to S3, or pushed to the Hub. Set it back to `false` to train. Toggle it in the yaml, or per-invocation with the `TEST_RUN` env var (`TEST_RUN=1` / `TEST_RUN=0`, checked before the yaml) so a dry run needs no file edit. The prompts live in `Trainer._STARTUP_PROMPTS` (`training/new_train.py`) and are plain untagged text — no `<yor>`/`<eng>` language tag — so they probe the model the way an untagged inference prompt would; edit them freely. The comparison itself runs at the start of *every* launch — only the stop-afterwards part is gated on `test_run`.

### 2.4 Run on vast.ai (or any bare GPU box)

1. **Rent** an instance with the GPUs you want (2× or 4× of one type; vast bills the whole box). Pick a CUDA PyTorch image and **≥150 GB disk**: the bins plus checkpoints (each fp32 checkpoint is ~1.2 GB, and `resume_state` ~4 GB, replaced in place).
2. **Set up** (once):

   ```bash
   ssh -p <vast_ssh_port> root@<vast_ip>
   git clone <your repo> sabiyarn_MoE && cd sabiyarn_MoE
   pip install -r requirements.txt
   nano .env   # create it: HF_API_KEY, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, (WANDB_API_KEY)
   python -m data.prefetch_bins --mode pretrain --write-env   # downloads the bins, writes their paths into .env
   ```

   `requirements.txt` pins `transformers==5.14.1` (what saved your checkpoints and what your `modeling.py` targets); use a CUDA PyTorch image with torch ≥ 2.8 so torch isn't reinstalled.

   Training only *checks* that the bins exist — it never downloads them — so `--write-env` (or copying the two printed lines into `.env` by hand) is required.

   **Pick the micro-batch for your GPU** (defaults are for an 80 GB card): set `TRAIN_BATCH_SIZE` and `GRAD_ACCUM_STEPS` in `.env` or the command line, keeping `TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS ≈ 192`. Rough starting points at 4096 tokens: 45/48 GB → `3` / `64`; 80 GB → `6` / `32`; 96 GB → `8` / `24` (`scripts/estimate_training_plan.py` has the reasoning). The `step` log line prints `peak_mem_gib`; if it is far below the card's memory raise the batch, and lower it on any `CUDA out of memory`.
3. **Dry run first** (loads weights, runs the deviation checks + one eval, saves nothing):

   ```bash
   TEST_RUN=1 torchrun --standalone --nproc_per_node=<N> -m training.new_train
   ```

   Look for `resume_sanity_check` (loaded weights match the checkpoint), `curated_eval`, and `eval` losses. Fix anything odd *before* the long run.
4. **Train** inside `tmux` so it survives your SSH session:

   ```bash
   tmux new -s train
   TEST_RUN=0 torchrun --standalone --nproc_per_node=<N> -m training.new_train 2>&1 | tee train.log
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
modal run training/modal_train.py --mode pretrain                 # DDP by default (ddp.enabled: true); 1 node, N GPUs
modal run --detach training/modal_train.py --mode pretrain        # keep running after you close the terminal
# FSDP instead: set `ddp.enabled: false` in the yaml first (multi-node runs use modal.experimental.clustered)
```

Everything in sections 2.2–2.10 works unchanged on Modal: the image installs `requirements.txt` (same versions as your vast box), the repo is uploaded to `/app` (`sabiyarn/`, `training/`, `eval_suite/`, `data/curated_eval.jsonl` included), and `model.code: local`, sparse MoE, flex attention, the epoch sampler and `fresh_start` all behave the same. The upload now skips `**/.venv`, `mlruns/`, `out_*/`, `eval_results/`, `data/bins/`, `data-gen/data/` and `*.bin` (a bare `.venv` pattern only matched the top level, so `data-gen/.venv` used to be uploaded on every launch).

GPU type and count come from `modal.gpu_type` / `modal.gpus_per_node` / `modal.num_nodes` in the yaml (Modal's cluster shape is fixed per Function, not a CLI flag; `mode`/`override` are runtime args). Data is synced from S3 onto the `sabiyarn-data` volume before training starts. Checkpoints save under `/data/<training.out_dir>` on that volume (and go to S3 / the Hub as configured); MLflow runs go to `/data/mlruns`. Secrets come from `.env` at launch time (`modal.Secret.from_dotenv`), so put `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` there too if the `modal` CLI isn't logged in (`modal token new`). The data-loader state is in `trainer_state.json`, which is pushed to S3 with every checkpoint, so a resume on a different Modal account/volume continues mid-epoch too.

**Env vars on Modal:** variables you type on your laptop (`TEST_RUN=1 modal run ...`, `DISTRIBUTED=fsdp ...`) are *not* forwarded into the container. Modal injects your `.env` file (`modal.Secret.from_dotenv`) and uploads `train_config.yaml` at launch, so control a Modal run through those two: put `TEST_RUN=1`, `TRAIN_BATCH_SIZE=…`, `FRESH_START=…` etc. in `.env`, or edit the yaml. `modal_train.py` always reads `training/train_config.yaml` (it ignores `TRAIN_CONFIG_PATH`), so a short pilot on Modal means temporarily editing that file (e.g. `optimizer.max_tokens`) rather than pointing at a copy — the env-var and copied-config recipes in 2.4/2.7 are for bare boxes.

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
    TRAIN_CONFIG_PATH=/tmp/$opt.yaml torchrun --standalone --nproc_per_node=<N> -m training.new_train
  done
  ```

  Compare `eval/val_bpb` at equal tokens in MLflow (the two runs are named `adamw` and `muon`). Keep Muon only if it wins by enough to matter (≳5% fewer tokens to the same bpb); otherwise stay on AdamW. Delete `out_pilot_*` afterwards.

Checkpoints are compatible between the two (weights only); optimizer state is not, and is skipped automatically.

### 2.8 Where checkpoints go

Modal: `/data/<training.out_dir>` on the `sabiyarn-data` volume. Bare box: `training.out_dir` (default `out_280M/`). Both also push to S3 (replace-in-place) and to `training.hf_chkpt_path` on the Hub when the eval loss is not a regression against the last push.

### 2.9 Sparse MoE and FlexAttention: what they are, and how to test them on your GPU

Both produce the **same math** as the original code; they only change memory and speed.

| Setting | Values | What it does |
|---|---|---|
| `model.moe_dispatch` | `sparse` (default) / `dense` | `sparse`: each expert only processes the tokens the router sent to it (~half the expert FLOPs and expert activation memory for top-2 of 4; ~28% fewer FLOPs per token overall). `dense`: every expert processes every token, then the top-2 are kept (the original path). |
| `training.attention_impl` | `sdpa_mask` (default) / `flex` | how packed documents are kept from attending to each other. `sdpa_mask` builds a `(B,1,T,T)` boolean mask per batch (~4-5 GiB per sample of activations, slower masked kernel). `flex` uses a FlexAttention block mask: no T×T tensor, and blocks lying across a document boundary are skipped (torch ≥ 2.5 and a GPU; compiles on the first call, ~1 min). |
| `model.code` | `local` (default) / `hub` | `local` builds the network from this repo's `sabiyarn/model/modeling.py` and loads only the weights; `hub` runs whatever `modeling.py` is stored on the Hub next to the weights. |

**Test them on real hardware — one command, real weights, real data** (a couple of minutes, no training):

```bash
python scripts/check_fast_paths.py --model out_280M/<run>/ckpt_best --batch-size 2
```

It runs the same real batch (a random window from your eval bin, containing real `</s>` document boundaries) through four configurations — dense/sparse MoE × dense-mask/flex attention — with the trainer's own precision (fp32 params + bf16 autocast), and prints loss, gradient norm, tokens/s and peak GPU memory for each, whether loss and gradients agree with the baseline within bf16 tolerance, and a recommendation. Gradient agreement is judged on the **whole flattened gradient**, not only its norm (a wrong gradient can have the right size): cosine similarity ≥ 0.99 and relative L2 error ≤ 0.15 against the baseline, plus the worst single tensor:

```
config                      loss    |grad|       tok/s  speedup  peak GiB   verdict
dense  + sdpa_mask        ...                             1.00x    ...      baseline
sparse + sdpa_mask        ...                                                PASS
dense  + flex             ...                                                PASS
                          gradient vs baseline: cosine 0.99987, rel. L2 error 0.0161, worst tensor transformer.h.3.attn.c_attn.weight @ 0.031
sparse + flex             ...                                                PASS
recommendation:  model.moe_dispatch: sparse | training.attention_impl: flex
```

Only turn `flex` on if it prints PASS on **your** GPU (it compiles a Triton kernel on first use; if compilation fails or the loss/gradients disagree, stay on `sdpa_mask` — sparse MoE alone still saves FLOPs). Raise `--batch-size` until the baseline row OOMs to see the memory headroom directly, then set `TRAIN_BATCH_SIZE`/`GRAD_ACCUM_STEPS` from the `peak_mem_gib` in the step log. `python scripts/check_fast_paths.py --tiny` runs the same comparison on a random tiny model on CPU (numerical agreement only; flex has no CPU backward, and CPU timings mean nothing). The unit tests already check sparse-vs-dense forward **and** gradients and flex-vs-dense-mask forward on CPU (`tests/test_sparse_moe_and_flex.py`), and flex gradients automatically when a GPU is present (`pytest tests/test_sparse_moe_and_flex.py -k gpu -v` on the GPU box runs just that test; it is *skipped*, not passed, without CUDA).

A second, independent check with the trainer itself: run `TEST_RUN=1 ...` three times changing one setting, and compare `resume_sanity_check`'s `current_sanity_loss` (same weights, same batch): `moe_dispatch: dense` vs `sparse`, `attention_impl: sdpa_mask` vs `flex`. They should agree to ~1e-3.

`python scripts/estimate_training_plan.py --flex` (and `--dense-moe`, `--cce`) models the batch size and time for each combination. `mfu.py` counts only the top-k experts under `moe_dispatch: sparse`, so the reported MFU stays comparable.

**Do you need to update `modeling.py` / `config.json` on the Hub for this?** No — not for training. With `model.code: local` the trainer builds the network from this repo's `modeling.py` and takes only the *weights* from `repo_name`; the Hub's `config.json` (which lacks `moe_sparse_dispatch`) just gets the class default `True`, and the yaml's `moe_dispatch` overrides it explicitly. So `init_from: hf` resumes into the sparse path with the Hub files untouched. Two things to know:

- **Anyone else loading your Hub model** with `trust_remote_code=True` still runs the Hub's old dense `modeling.py` (identical outputs, just slower) until you publish new code. That happens **automatically the next time a checkpoint is pushed**: every checkpoint the trainer saves now carries `modeling.py`, `configuration.py` and a `config.json` with `auto_map` and `moe_sparse_dispatch` (the trainer registers the local classes for saving; without that a saved checkpoint would have contained only weights, and pushing it would have overwritten the Hub's `config.json` and broken remote-code loading — there is a regression test for this). To publish the code without pushing weights: `python training/push_model_code_to_hf.py --confirm`.
- If you set `model.code: hub`, the trainer runs whatever code is on the Hub, so sparse dispatch and flex only exist there after that publish.

### 2.10 Evaluating a checkpoint (`eval_suite/`)

```bash
pip install -r requirements.txt   # includes datasets + sacrebleu for the suite
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

### 2.11 How training data is sampled (no repeats within an epoch)

`training/data_sampler.py` replaces the old "random start offset with replacement" batching (which let windows overlap and repeated some tokens before others were seen once, and — because it re-seeded on every start — replayed the same batches after each resume).

- Each bin (`eng_training.bin`, `training_cleaned.bin`) is cut into **non-overlapping windows of `block_size` tokens**; window *i* gives inputs `data[i·B : (i+1)·B]` and targets shifted by one. Every token position is therefore an input **at most once and a prediction target at most once per epoch**, independently for each bin.
- Each epoch uses a **fresh random permutation** of the windows (seeded by `(seed, bin, epoch)`, so it is reproducible); when a bin is exhausted it is reshuffled and the next epoch starts, carrying over without dropping or repeating a window.
- **Distributed**: all ranks share one global stream per bin; a step takes `batch × world` windows and rank *r* takes slice *r*, so ranks never see the same window.
- **English/African mix**: a deterministic credit scheme (identical on every rank, no RNG) picks the bin for each micro-step from `data.sampling` weights — it matches the weights exactly over time, follows scheduled sampling as the weights change, and lets each bin cycle through epochs at its own pace (the smaller bin repeats sooner; you'll see `train/epoch_<bin>` climbing at different rates).
- **State is saved** (`sampler` in `trainer_state.json`, pushed to S3 with each checkpoint) and restored on resume, so a restart continues mid-epoch. It restarts from epoch 0 on `fresh_start`, on resume-from-`best`, and for checkpoints saved before this existed. (One micro-batch — the one prefetched at save time — is skipped on each resume.)
- **Evals use a fixed set of windows** (the same ones every time, never advancing the training stream), so eval curves are comparable across steps instead of carrying fresh sampling noise; the "train" eval mixes the training bins.
- Logged per step: `epochs_by_bin` (fractional epochs, e.g. `{"eng_training": 0.37, "training_cleaned": 1.2}`); MLflow: `train/epoch_<bin>`.

### 2.12 Generating synthetic data (`data-gen/`)

Independent of the training code (own README, own venv). Besides the six tool-use/task generators it now builds three larger, **diversity-sampled** datasets for the 12 non-English languages with `gpt-4o-mini` through the OpenAI Batch API: **pretraining documents**, **Alpaca-style SFT** (`instruction`/`input`/`response` over ~33 standard task types) and **DPO pairs** (`instruction`/`input`/`chosen`/`rejected` with a sampled flaw type).

```bash
cd data-gen && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt && cp .env.example .env   # add OPENAI_API_KEY
python run.py estimate --kind all                                   # volumes, tokens, $ (no API calls)
python run.py build --kind sft --config configs/sft.yaml            # writes data/batch_input/*.jsonl (free)
python run.py submit --kind sft --confirm                           # costs money; omit --confirm for a dry run
python run.py fetch                                                 # poll / download when the batch completes
python run.py postprocess --kind sft --config configs/sft.yaml      # validate, filter, dedup -> data/processed/sft.jsonl
python run.py judge build --kind sft   # optional LLM-judge pass; then `judge apply`
```

Sample counts per language live in `data-gen/configs/{pretrain,sft,dpo}.yaml` (presets: 4,000 pretrain / 6,000 SFT / 1,500 DPO per language, ≈ $36 total (range $32–$43) at gpt-4o-mini Batch prices of $0.075 / $0.30 per 1M tokens, checked against OpenAI's pricing page on 2026-09-21; `config/settings.py` holds the constants). Coverage is driven by a sampler that cycles every (domain, sub-topic) pair before repeating and balances task, register, locale, length and difficulty; `build` prints a coverage audit. At the preset volumes **every one of the 59 domains, 636 (domain, sub-topic) pairs, 28 genres, 33 SFT task types and 12 DPO flaw types is requested in every one of the 12 languages** (`python data-gen/scripts/audit_coverage.py`, enforced by `data-gen/tests/test_full_preset_coverage.py`). **Run a small paid pilot (e.g. `--per-language 20`) and have a native speaker read a sample per language before the full run**: `gpt-4o-mini` is materially weaker than `gpt-4o` in Efik, Urhobo, Fon, Ewe and Fulah/Fulfulde, synthetic pretraining text should be a small filtered supplement to real corpora, and injected-flaw DPO pairs are off-policy. Details in `data-gen/README.md`.

**Sources for the hyperparameter choices:** [Hägele et al. 2024 (WSD)](https://arxiv.org/abs/2405.18392) · [Bergsma et al. 2025 (linear decay to zero)](https://arxiv.org/abs/2502.15938) · [Wen et al. 2025 (optimizer benchmark)](https://arxiv.org/abs/2509.02046) · [Liu et al. 2025 (Moonlight/Muon)](https://arxiv.org/abs/2502.16982) · [Jordan 2024 (Muon)](https://kellerjordan.github.io/posts/muon/).

### 2.13 The chat template (one source of truth)

`sabiyarn/chat_template.jinja` is the canonical template. `data-gen/templates/chat_template.jinja` (which renders the SFT/DPO data) must be byte-identical — `tests/test_chat_template.py` fails if they drift. Anything that builds prompts (`eval_suite --style chat`, `rl/`, `rlhf/`) calls `sabiyarn.chat.use_sabiyarn_chat_template(tokenizer)` instead of trusting the copy on the Hub tokenizer, which can be stale. The template renders exactly `<s><|system|>…</s><|user|>…</s><|assistant|>answer</s>` with **no stray whitespace**, and the generation prompt is an exact prefix of the training text (both are tested). **After changing the template, push it to the tokenizer repo** so other users get it: `tokenizer.chat_template = sabiyarn.chat.load_chat_template(); tokenizer.push_to_hub("BeardedMonster/SabiYarn-32k")`. Note the template renders tool results as `<tool_result>…</tool_result>` while the tokenizer's special tokens include `<tool_call>`; make sure the token you want for results exists before generating tool-use data.

### 2.14 Post-training: DPO, reward-based RL, SFT warm-start (`rl/`)

One config (`rl/config.yaml`), one entry point, DDP through Accelerate, fp32 master weights + bf16 autocast, MLflow metrics, `lr_schedule`, the local model class (checkpoints ship `modeling.py`/`configuration.py`/`auto_map`), the canonical chat template, optional HF push and S3 mirror — the same conventions as `training/`.

```bash
python -m rl.run                                                    # DPO on data-gen/data/processed/dpo.jsonl (rl/config.yaml)
torchrun --standalone --nproc_per_node=2 -m rl.run                  # DDP on 2 GPUs
python -m rl.run learning_rate=1e-6 max_steps=200 out_dir=outputs/dpo-test   # key=value overrides (or RL_<FIELD>=... env vars)
python -m rl.run --algo sft --config rl/config.yaml data_path=data-gen/data/processed/sft.jsonl learning_rate=1e-5
modal run rl/modal_rl.py                                            # same thing on Modal (DPO)
modal run rl/modal_rl.py --config rlhf/configs/default.yaml         # AfriCOMET RL on Modal (comet image)
modal run rl/modal_rl.py --overrides "model_path=/data/out_280M/<run>/ckpt_best max_steps=100"
```

On Modal the launcher uses the same GPU shape (`modal:` in `train_config.yaml`), volume and `.env` secrets as
training. It also **uploads a local `data_path` to the volume** first (`data-gen/data/` is excluded from the image),
points `out_dir` and MLflow at `/data`, and fails locally — before any GPU time — if `model_path` is a local
directory the container could not see. Start from a training checkpoint with
`model_path=/data/<training.out_dir>/<run>/ckpt_best` (`modal volume ls sabiyarn-data` to find it), and collect
results with `modal volume get sabiyarn-data rl/ ./rl_out`.

**DPO** (`rl/dpo.py`): loss `-log σ(β[(log π(chosen)−log ref(chosen)) − (log π(rejected)−log ref(rejected))])`; variants `dpo_loss: ipo | hinge`, `label_smoothing`, `length_normalize`, `sft_alpha` (NLL on chosen, stops likelihoods drifting down). The reference is a frozen copy of the start weights. Policy and reference are scored in eval mode (no MoE router noise), so **step 1 must log `train/loss ≈ 0.6931` and `train/init_logratio_absmax ≈ 0`**; if not, the two models differ (wrong `reference_path`, dtype, code). Watch `eval/reward_accuracy` and `train/reward_margin`; accuracy → 1.0 within a few hundred steps means the pairs are separable by surface cues (length, language, refusals) rather than quality. data-gen's rejected answers are *off-policy* (a model was asked to write a flawed answer), so keep `beta` moderate (0.1) and the learning rate small (5e-7).

**Reward-based RL** (`rl/policy_gradient.py`, `algo: rl`): for each prompt sample `group_size` completions, score them, advantage = reward − group mean (÷ std), one on-policy gradient step on `−A·log π + kl_coef·KL(π‖ref)` (k3 estimator). No critic, no clipping (each batch is used once). Log lines to watch: `train/reward`, `train/kl` (should grow slowly), `train/flat_group_frac` (groups whose completions all scored the same contribute nothing — high means raise `temperature`/`group_size`), `train/eos_rate` (falling means it learned to ramble to dodge the KL), `eval/reward`. Rewards: `chrf` (chrF++, no extra install) or `africomet`. Prompts per optimizer step per GPU = `batch_size × grad_accum_steps`; `rl_micro_batch` only chunks the update.

**AfriCOMET RL** (`rlhf/`, your translation task on top of `rl/`):

```bash
python -m venv --system-site-packages .venv-comet && .venv-comet/bin/pip install -r rlhf/requirements.txt   # see below
python rlhf/scripts/run_sft.py                                                    # optional warm-start on Aletheia-ng/tds-sft
RL_REWARD_PYTHON=.venv-comet/bin/python python rlhf/scripts/run_rlhf.py model_path=outputs/translate-sft/final
python rlhf/scripts/run_rlhf.py reward=chrf max_steps=20                          # cheap smoke test without COMET
python -m eval_suite.run --model outputs/rlhf/final --tasks translation --style chat --africomet masakhane/africomet-stl   # with AFRICOMET_PYTHON=.venv-comet/bin/python
```

Why a second environment: `unbabel-comet` pins `transformers<5`, `huggingface_hub<1`, `numpy<2`, which cannot coexist with the `transformers==5.14.1` this repo's model code needs. Its pins live in **`rlhf/requirements.txt`** — install them into their own virtualenv, never next to the root `requirements.txt`. `--system-site-packages` reuses the torch you already have (~2.5 GB saved). `rl/comet_worker.py` runs in that environment and talks to the trainer over a pipe (one worker per GPU rank); nothing else changes. On Modal, `modal run rl/modal_rl.py --config rlhf/configs/default.yaml` builds that venv into the image automatically and wires `reward_python` to it. `rlhf/configs/default.yaml` now starts from `Aletheia-ng/SabiYarn_MoE-280M` (was `google/gemma-3-270m-it`). AfriCOMET is a **learned, reference-based** metric: optimising it hard finds its blind spots. Keep `kl_coef` on, and judge progress on chrF++/BLEU from `eval_suite` (not used as a reward) and by reading samples. If the reward rises while chrF++ falls, raise `kl_coef` or stop earlier.

**Batched generation and padding.** Positions are learned and absolute, so left-padded batches need per-row positions; `generate()` now derives them from the attention mask (regression-tested against one-prompt-at-a-time generation). A 2D `attention_mask` is now combined with the causal mask — earlier versions let padded/masked forward passes (including `generate()` prefill with an all-ones mask) attend to future tokens, so **re-run evals of anything generated with a 2D mask on older code**. Training itself (no mask, or the document mask) was never affected.

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
