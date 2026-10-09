## 4. Training-systems efficiency

The efficiency work is what turned a 306M-parameter MoE into something that trains at 786,432 tokens per optimiser step on four A100-80GB cards. Every number in this section is **[R]** (measured by the authors and recorded in `training/train_config.yaml` comments and commit bodies `bf02f35`, `a8e9088`, `77b3807`) unless marked [V].

### 4.1 Where the savings come from

| technique | what it removes | where | cost / caveat |
|---|---|---|---|
| **Top-2 routing with sparse dispatch** | Half the expert matmuls and half the stored expert activations for a 4-expert layer; 27.8% fewer training FLOPs per token overall (1.542 vs 2.137 GFLOPs, T = 4,096) | `MoE._sparse_experts` [V] | one host sync per layer; verified equal to the dense path (Δlogit 3.7e-9) |
| **Document-causal FlexAttention** | The `(B, H, T, T)` boolean mask that SDPA would materialise (≈800 MiB per layer per sample at T = 16,384); blocks entirely across a document boundary are skipped | `training/training_attention_mask.py` | block mask built uncompiled (compiling it recompiled on every micro-batch and leaked memory, fix `df1528e`); flex block size 128 |
| **Cut cross-entropy (CCE)** | The `(B, T, 52,050)` logits tensor: 2.38 GiB at batch 6 and 4.77 GiB at batch 12 (derived), plus its gradient and the fp32 upcast | `use_cce`, Apple `cut_cross_entropy` | kernels need bf16 inputs → hidden states and tied head are cast to bf16 for the loss (`77b3807`); it had been a silent no-op because the package was missing from `requirements.txt` (`1f7eb32`) |
| **fp32 master weights + bf16 autocast** | – (stability) | `param_dtype: float32`, `dtype: bf16` | double the optimizer memory of pure bf16 |
| **Activation checkpointing off on 80 GB cards** | A 40% throughput penalty (92.2K → 65.8K tokens/s) | `gradient_checkpointing: false` | on a 22 GB A10 it must be on |
| **DDP rather than FSDP** | FSDP communication and ~15% throughput; at 10.8 GiB peak there is nothing to shard | `ddp.enabled: true` (`a8e9088`) | FSDP remains available |
| **Muon** (optional) | – | `training/muon.py` | not run on a GPU or on this MoE yet [R] |
| **Packed sequences with one EOS per document** | padding | `data/prepare.py`, sampler | positions are not reset across documents |

### 4.2 Measured sweep on one A100-80GB (block 4,096, `use_cce`, 50 iterations each) **[R]**

| micro-batch | checkpointing | peak memory | tokens/s | MFU |
|---|---|---|---|---|
| 12 | on | 10.83 GiB | 65.8K | 0.325 |
| **12** | **off** (chosen) | **37.60 GiB** | **92.2K** | **0.456** |
| 16 | off | 48.43 GiB | 91.4K | 0.452 |
| 24 | off | 70.10 GiB | 93.6K | 0.463 |
| 24 | on | 17.02 GiB | 71.6K | 0.354 |
| 48 | on | 29.40 GiB | 69.6K | 0.344 |

* Throughput is flat for micro-batch 12–24 once checkpointing is off, so batch 12 (48% of the card) is chosen for headroom for evaluation and fragmentation.
* MFU is computed by `training/mfu.py` with sparse-dispatch FLOPs (1.542 GFLOP/token) against the A100 BF16 peak of 312 TFLOPS: 92.2K × 1.542 G / 312 T = 0.456, matching the logged value [V, recomputed]. It excludes recomputation FLOPs.
* On a 22 GiB A10 (checkpointing on): peak ≈ 4.6 + 0.52 × batch GiB; batch 6 = 7.73 GiB at 21.8–23.3K tokens/s (MFU 0.288); without CCE and checkpointing the same step OOM'd trying to allocate 2.38 GiB with 19.52 GiB already in use.
* **Projection for four A100s** (derived, not measured): ≈ 369K tokens/s, ≈ 38 h and ≈ 151 GPU-hours for 50B tokens. No end-to-end 4-GPU throughput is recorded in the repository (Appendix D).
* Global batch: 12 (micro-batch) × 4,096 (block) × 4 (accumulation per GPU) × 4 (GPUs) = **786,432 tokens per optimiser step**; this number is held fixed across SFT (1 × 32,768 × 24 accumulation) so that the learning-rate schedule need not be re-tuned.

### 4.3 Document-causal masking

`doc_id = cumsum(is_eos) − is_eos`, so each `</s>` stays in the document it closes; `mask_mod(q, kv) = (q ≥ kv) ∧ (doc_id[q] = doc_id[kv])`. The same function builds a dense SDPA mask on hardware where flex attention is unavailable (`attention_impl: sdpa_mask`). `scripts/check_fast_paths.py` runs one real batch through the four combinations {dense, sparse MoE} × {sdpa-mask, flex} and compares loss, gradient cosine and relative L2; HOW_TO_RUN prints an illustrative result (cosine 0.99987, relative L2 0.0161, worst tensor `transformer.h.3.attn.c_attn.weight` at 0.031) without provenance, so a fresh run on the target GPU should be archived for the paper.

### 4.4 What is *not* efficient (honest list for the limitations section)
* Absolute learned positions: 25.2M parameters and an attention cost that grows quadratically, 8.2% of the model for rows that are 87% untrained (§3.7).
* Experts are executed in a Python loop per layer; a grouped-GEMM kernel would remove the loop and the host sync.
* `torch.compile` is disabled for multi-process runs.
* The estimator in `scripts/estimate_training_plan.py` assumes MFU 0.25 on an A100, below the measured 0.456; its tables should be re-run with the measured value.
