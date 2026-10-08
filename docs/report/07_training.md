## 7. Continued-pretraining procedure, monitoring, incidents, current state

### 7.1 Recipe **[V]** (`training/train_config.yaml`, `training/new_train.py`)

| item | value |
|---|---|
| objective | next-token cross-entropy on packed 4,096-token blocks with the document-causal mask; `loss = CE + 0.015 · L_aux`; the multi-task v1 tags are trained with the ordinary causal loss (`use_loss_mask` exists but is off for pretraining) |
| optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1, gradient clip 1.0, fp32 master weights, bf16 autocast; Muon is implemented but not used |
| peak LR | **1e-5** (see §7.4); schedule **WSD** (`scheduler: wsd`): warm-up, hold, cool-down to `min_lr` = 0 over the last 20% of `max_iters` with a (1 − √t) shape, so any stable-phase checkpoint can be cooled down on its own |
| token budget | `max_tokens` = 50e9 ⇒ `max_iters` derived at start-up from the real tokens/step (fallback 60,000) |
| batch | 786,432 tokens/step = 12 × 4,096 × 4 × 4 (§4.2) |
| data | three train bins sampled with fixed weights eng 0.15 / afr 0.55 / other-African 0.30 (§5.1, §6) |
| MoE | top-2, router noise σ = 0.1, aux-loss weight 0.015 (was 0.01 → 0.02 on 24 Jul → 0.015), per-layer utilisation logged |
| distributed | DDP over 4 × A100-80GB (Modal; also vast.ai); checkpoints pushed to S3 (replacing in place) and, behind a gate, to the Hub |
| validation | `validation_v2.bin`, 500 fixed blocks, 40% old / 60% by language; `val_loss` (best-checkpoint criterion, includes the aux term), `val_ce`, `val_loss/<lang>`, `val_bpb/<lang>` |

### 7.2 Monitoring and safety nets built around the run **[V]**
* **Resume sanity check:** before training, a fixed hash-verified batch is evaluated and compared with the loss recorded in `trainer_state.json` (`b2b3e12`, `94f6d9e`); a mismatch aborts, because a resumed run that silently loads the wrong tensors looks like normal training.
* **Reference-weight drift check** (`fb2c771`, `589a74b`): relative L2 distance (parameter-count weighted) of the live weights from the frozen `Aletheia-ng/sabiyarn-ref`; a run whose drift jumps is a divergence regardless of what the loss says.
* **Hub push gate:** a checkpoint is pushed only if its validation loss is within a band (0.4) of the best; this is what stopped every diverged checkpoint from reaching the Hub in September (§7.4).
* **Startup generation comparison:** six prompts × {sampling, greedy} × {current, reference model} are generated and logged; "number soup" in the generations was the first visible symptom of the divergence.
* **Two tracked checkpoints:** `latest` (every eval) and `best` (only on improvement); `resume_from: best` rewinds `iter_num`, the LR position and the optimiser state together.
* **Save round-trip check** (`0cc5795`): after saving, the checkpoint is re-read and compared to the live parameters, because an FSDP `state_dict` once did not capture what was live (`839fbea`, `1bdf614`).
* **Token accounting:** `tokens_seen`, `tokens_seen_lifetime` and `tokens_seen/<bin>` are logged and written to `trainer_state.json`; they are *estimates* from the number of blocks drawn (§7.5).

### 7.3 Bugs found along the way that change the interpretation of earlier numbers **[V: commit message read; rows marked † are read from the commit title only]**
| date | commit | finding | effect on results |
|---|---|---|---|
| 24 Jul | `68ed76b`, `73cf91a` | tag-pair label masking was not scoped per document: of the tokens that *were* masked, **14.2%** belonged to runs crossing a `</s>` (measured on 300 windows of `training_cleaned.bin`; 12.7% of tokens masked overall) | for the v1 multi-task runs the loss was partly masked across unrelated documents; scoped per document afterwards |
| 21 Jul † | `79a2317` | `cpu_ram_efficient_loading`/`sync_module_states` silently left `lm_head`/`wte` uninitialised on 7 of 8 ranks under FSDP | any FSDP run before the fix is suspect |
| 21–22 Jul | `4ce1ad6`, `83baebe` | `GenerationMixin.generate()` replaced by a manual greedy decode loop for the periodic samples | periodic sample generations in the logs come from this loop, not from `generate()` |
| 2 Aug † | `32f0ee6` | router noise was added during evaluation too; now gated on `torch.is_grad_enabled()` | earlier validation losses carry router-noise variance |
| 21 Sep | `b679580`, `3251821` | 2-D padding masks were not causal; left-padded batches had wrong positions | only batched evaluation/generation was affected, not training |
| 21 Sep | `1f7eb32` | `use_cce` was a silent no-op (package missing from requirements) | memory numbers before this commit are for the unfused loss |
| 22 Sep | `df1528e` | flex block mask recompiled every micro-batch (leak) | throughput before this fix is understated |
| 22 Sep | `3fe9c36` | `@torch.no_grad()` on `estimate_loss` had been dropped and evaluation OOM'd on all four ranks | – |

### 7.4 The September divergence **[V]** (commit `0288460`, `32da114`, YAML comments)
After a restart on 4 × A100 with the learning rate set to the "from scratch" value for a 300M model (**6e-4**) while resuming an already converged checkpoint (whose history was at 1e-5), validation loss moved

| iter | 17,400 | 17,700 | 18,000 | 18,150 | 18,300 | 18,450 |
|---|---|---|---|---|---|---|
| val loss | **3.3837** (best) | 3.7811 | 4.5616 | 4.4560 | 5.5819 | 7.7890 |

with step loss swinging 3.2 ↔ 7.1, reference-weight drift growing from 0.00073 to 0.36156 (largest single tensor `transformer.h.1.attn.c_attn.weight` at 1.71) and the layer-0 router collapsing onto one expert (router probabilities [1.0, 0.0], load-balance loss ≈ 2.95). Only a warm-up of 1,000 steps existed in the config, long expired at iteration 17,400.
**Recovery:** resume `ckpt_best` (iteration 17,400; the push gate had kept every diverged checkpoint off the Hub), peak LR first 5e-5 (`warmup_iters` 18,400 so the schedule sits at 94.6% of peak at the resume point — a ramp is impossible when the schedule is a pure function of the global iteration), then 1e-5 "the rate this checkpoint actually trained under", because the first step lands near the peak and the peak has to be safe.
This is a useful negative result for the paper (a learning-rate transfer failure when continuing an upcycled MoE; the layer-0 router collapse is the diagnostic) and is the reason the LR in §7.1 is 1e-5.

### 7.5 Current state **[V]** (`trainer_state.json` of the latest checkpoint)
* iteration **21,600**; best validation loss **3.136547 at iteration 19,650**; the start-up sanity loss was 2.8833 (a different, fixed batch).
* `tokens_seen_lifetime` ≈ **16.99B** tokens = 21,600 × 786,432 — an *estimate* that assumes every step consumed a full global batch; the by-bin counters read from the same file (1.30B from `training_cleaned`, 0.23B from English) are far smaller and are not the lifetime total. [A] explain which restart they count from.
* A previous run of the same checkpoint family stopped at iteration 7,560 with best validation loss 3.4248; the stratified validation (`validation_v2`) was introduced after these iterations, so the loss values are comparable only within the `old` segment.
* The new bin (`other_african_pretrain.bin`, 12.5B tokens) is **not yet in the training mix** of the logged run; the sampler weights above describe the *next* run. [A] confirm that the 3.1365 value was measured on `validation.bin` (old) and not on `validation_v2`.

### 7.6 Plan for the next stage (what is configured, what is not) 
1. Continued pretraining on the three bins with the fixed mixture above and WSD; **no model change** in this stage — vocabulary and position growth are SFT-only (§9.3) because the new language tags (`<ibb>` … `<wol>`, ids ≥ 52,050) do not occur in the continued-pretraining text.
   *Consequence to state in the paper:* the 12 new languages are learned **without** their language tags during pretraining; their tags (and `<POS>`) are first trained in SFT. [A] confirm this is intended — the alternative is to resize the embedding before pretraining so the tags can be seen.
2. A decision on the schedule: re-run `scripts/estimate_training_plan.py` with the measured MFU 0.456 (the estimator assumes 0.25).
3. Choice of the cool-down point: with WSD any stable-phase checkpoint can be annealed on the highest-quality bins; §6.5 describes the end-of-bin tail.
