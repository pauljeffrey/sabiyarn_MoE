## 1. Lineage: from SabiYarn-125M to SabiYarn-MoE-306M

### 1.1 What the Hub records

The checkpoints below were listed through the Hub API on 8 October 2026 (config read from each repository; sizes are `model.safetensors`). **[V]**

| Hub id | Created | Shape (from `config.json`) | Weights | What it is |
|---|---|---|---|---|
| `BeardedMonster/SabiYarn-125M` (+ `-finetune`, `-translate`, `-sentiment`, `-topic`, `-Yoruba-translate`, `-Igbo-translate`) | Jul–Sep 2024 | dense, `nanogpt-j`, 12 L, 768 d, 12 heads, block 1,024, vocab 52,050 | 553 MB fp32 (≈138M) | The first model: GPT-J-style network adapted from nanoGPT, pretrained on Nigerian-language data with a multi-task CLM objective; task-specific fine-tunes. Card: "Majorly English, Yoruba, Hausa, Igbo, Pidgin and some others: Fulah/Fulfulde, Efik, Urhobo". |
| `BeardedMonster/SabiYarn`, `original-sabiyarn-125M` | Jan 2025 / Mar 2026 | same as above | 553 MB | Re-uploads of the dense model. |
| `BeardedMonster/sabiyarn-32k` | 11 Nov 2025 | dense, `sabiyarn`, 12 L, 768 d, **block 4,096**, vocab 52,050 | 256 MB bf16 (≈128M) | Dense checkpoint in the new model class (`GPTJXMoEForCausalLM`, `use_moe=false`) with the context extended 1,024 → 4,096. The repo also holds the tokenizer used from here on. |
| `BeardedMonster/sabiyarn-32k-v2`, `sabiyarn_moe`, `sabiyarnMoE` (private), `MOE`, `Sabiyarn_MoE_` (private) | Dec 2025 – Feb 2026 | MoE, 12 L, 768 d, **7 experts, top-2**, `moe_dim` 3,072, **block 32,768** | 980 MB bf16 / 1,960 MB fp32 (≈490M) | First MoE checkpoints (uniform 7 experts per layer). |
| `BeardedMonster/Sabiyarn_MoE-750M` | 12 Apr 2026 | MoE, **20 L**, 768 d, 7 experts, top-2, block 32,768 | 3,092 MB fp32 (≈773M) | A deeper variant. [A] whether and how far it was trained. |
| `Aletheia-ng/SabiYarn_MoE_translate` | 25 Mar 2026 | MoE, 12 L, **4 experts uniform**, top-2, `moe_aux_loss_weight` 0.05, `router_z_loss_weight` 0.01 | 1,093 MB | Translation-specialised run; the only checkpoint whose config mentions a router z-loss (the current trainer has none). |
| `Aletheia-ng/SabiYarn_MoE_translate_base` | 17 Apr 2026 | MoE, 12 L, `num_experts` 4, **`expert_per_layer` = {0:2, 1:3, 2–11:4}** | 612 MB bf16 (≈306M) | First appearance of the heterogeneous 2/3/4-expert layout. |
| `Aletheia-ng/SabiYarn_MoE_base` | 4 May 2026 | same layout | 1,144 MB | Base for continued pretraining. |
| **`Aletheia-ng/SabiYarn_MoE-280M`** | 30 Jul 2026 | same layout, `moe_sparse_dispatch` true | 1,383 MB fp32 | The model the current trainer continues. The "280M" in the name is the parameter count *without* the position table (280.7M; with it 305.9M, §3.2). |
| `Aletheia-ng/sabiyarn-ref` | 1 Aug 2026 | same layout | 1,223 MB | Frozen reference used by the trainer's weight-drift sanity check (§7.4). |
| `BeardedMonster/sabiyarn-tokenizers`, `…-vocab-81920` | May / Jun 2026 | – | – | Tokenizer experiments (an 81,920-entry vocabulary; fertility compared in §2.3). |

Chronology of the training repository (the part that is under version control) starts on **20 July 2026** (`01515f5`); everything before that — the dense model, its context extension, the upcycling and the expert pruning — was done in earlier work whose code is **not** in this repository (the only upcycling and pruning *tools* in the repo are `load_dense_weights_into_moe`, `truncate_experts` and `apply_moe_layer_edits` in `sabiyarn/model/modeling.py`, and nothing in `training/` calls them).

### 1.2 The reconstructed storyline (inferred from the table; confirm in Appendix D)

1. **Dense base (2024).** A GPT-J-style transformer (§3.1) trained on a multi-task tagged corpus (§5). **[V]** Instantiated in the current model class with the 125M configuration it has 125.73M parameters (124.95M without the 1,024-row position table); the 2024 file itself (original implementation, `nanogpt-j`) is 553 MB in fp32, so its stored tensor count is not directly comparable.
2. **Context extension and re-platforming (Nov 2025).** The same network re-implemented in the `GPTJXMoEForCausalLM` class with block size 4,096 (the 32k tokenizer ships here).
3. **Sparse upcycling (Dec 2025).** Each dense MLP is cloned into E = 7 experts, routers start at zero, block size 32,768. With top-2 routing this model is *exactly* the dense model at step 0 (§3.3). Total parameters rise from 125.7M to 489.9M while active parameters per token rise from 125.7M to 206.8M.
4. **Expert pruning (Mar–Apr 2026).** Experts are removed per layer (`truncate_experts`) to a heterogeneous layout: 2 experts in layer 0, 3 in layer 1, 4 in layers 2–11, giving 45 experts and 305.9M parameters. The first/second layers keep fewer experts. [A] which experts were kept and the criterion (utilisation? router mass? magnitude?), and how many tokens were trained before and after pruning.
5. **Continued pretraining (Jul 2026 → now).** 786,432-token steps on 4 × A100-80GB; best validation loss 3.1365 at iteration 19,650 and a checkpoint at iteration 21,600 (§7.5). A divergence at LR 6e-4 (dated 23 Sep 2026 in the YAML comment; the recovery commit `0288460` is dated 22 Sep) forced a restart from the best checkpoint, first at LR 5e-5 and then at 1e-5 (§7.4).
6. **Post-training preparation (Sep–Oct 2026).** Synthetic data generation, restructuring of existing data into one grammar, RL machinery, and the data campaign for the next stage of pretraining (§6, §8, §9).

### 1.3 Why the design is what it is (rationale to state in the paper)

* **Small and fast to serve, but not too small to learn 23 languages.** 306M parameters but only 206.8M (181.6M without the position table) active per token: the expert FFNs hold 69% of the parameters, and each token uses 2 of the 2–4 experts of every layer (§3.4).
* **Upcycling instead of training an MoE from scratch.** The dense model already encodes the 12 original languages; upcycling keeps that investment (function-preserving init) and buys capacity for the languages added later.
* **Heterogeneous experts per layer.** Layer 0 has 2 experts with k = 2, so every token goes through both experts: that layer is a *dense* FFN of twice the width. Layer 1 has 3 experts. The remaining layers have 4. The auxiliary load-balance loss is constant for layer 0 (derived: with E = k the loss is identically k) and contributes no gradient there.
