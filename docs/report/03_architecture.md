## 3. Architecture and sparse upcycling

All statements in §3.1–3.5 are read from `sabiyarn/model/modeling.py` / `configuration.py` unless another source is named. **[V]**

### 3.1 The network

```
tokens ──► wte(52,050×768) + wpe(32,768×768) ──► 12 × BlockJ ──► LN_f ──► lm_head (tied to wte) ──► logits

BlockJ(x):   x_ln = LN_1(x)
             x    = x + Attn(x_ln) + LN_J(x_ln)            # GPT-J-style extra path: a second LayerNorm of the
                                                           #   already-normalised input, added to the residual
             x    = x + MoE(LN_2(x))                       # (dense MLP when use_moe = false)
```

* 12 layers, d = 768, 12 heads (head dim 64), plain multi-head attention (no GQA, no QK-norm), fused `c_attn` 768→2304, `c_proj` 768→768, **no biases** (`bias=false`), LayerNorm ε = 1e-5, no RMSNorm. `LN_J` is built with a bias (the constructor receives `bias=768`, which is truthy), the other norms do not have one.
* **Positions are learned and absolute** (`wpe`, 32,768 rows), not rotary; sequences are packed and positions run 0…T−1 across packed documents (they are not reset at `</s>`). Attention is isolated per document by a *document-causal mask* (§4.3).
* Embeddings are **tied**; the output head has no bias. Activation is exact GELU; dropout 0.0.
* Generation: standard KV cache; for left-padded batches `generate()` derives per-row `position_ids` from the attention mask (fix `3251821`), and a 2-D padding mask is combined with the causal mask (fix `b679580`, 21 Sep 2026 — before it, a 2-D mask let queries see future tokens; training was never affected, only batched evaluation/generation).

### 3.2 Parameter accounting **[V]** (instantiated on the meta device from the configs)

| model | total | without `wpe` | active / token | attention | expert FFNs | `wte` (tied) | `wpe` | routers |
|---|---|---|---|---|---|---|---|---|
| dense 12 L, block 1,024 (SabiYarn-125M shape) | 125.73M | 124.95M | 125.73M | 28.31M | – (dense MLP 56.62M) | 39.97M | 0.79M | – |
| dense 12 L, block 4,096 (`sabiyarn-32k`) | 128.09M | 124.95M | 128.09M | 28.31M | – (56.62M) | 39.97M | 3.15M | – |
| 7 experts × 12 L, block 32,768 (first MoE) | 489.92M | 464.75M | 206.80M | 28.31M | 396.36M | 39.97M | 25.17M | 0.06M |
| 4 experts × 12 L (uniform) | 320.02M | 294.85M | 206.77M | 28.31M | 226.49M | 39.97M | 25.17M | 0.04M |
| **SabiYarn-MoE: experts per layer (2, 3, 4×10) = 45** | **305.86M** | **280.69M** | **206.77M** (181.60M without `wpe`) | 28.31M | 212.34M | 39.97M | 25.17M | 0.03M |
| 7 experts × 20 L ("750M") | 773.10M | 747.93M | 301.24M | 47.19M | 660.60M | 39.97M | 25.17M | 0.11M |

* One expert is `fc_bank[e]` (768×3,072) + `proj_bank[e]` (3,072×768) = 4,718,592 parameters, i.e. exactly one dense MLP of the dense model (`moe_dim` = 4 × 768).
* **"280M" in the model name = parameters without the position table** (280.69M). With it the model has 305.86M parameters; `wpe` alone is 25.17M = 8.2% of the model, of which only the first 4,096 rows (3.15M) have ever been trained (§7.1).
* Active parameters per token count the two selected experts of each layer; the embedding matrix is counted as active (it is looked up and, being tied, used for the output projection).
* Active-parameter ratio MoE/dense = 206.8M / 125.7M = **1.64×**, total ratio = **2.43×**. In FLOPs per trained token (T = 4,096, forward + backward, `training/mfu.py`): 1.542 GFLOPs with sparse dispatch against 2.137 GFLOPs if every expert ran on every token (dense dispatch): −27.8%.
* Layer 0 has two experts and k = 2, so each token uses both: that layer behaves as a dense FFN of width 6,144. Layer 1 uses 2 of 3. Layers 2–11 use 2 of 4.

### 3.3 Sparse upcycling: dense → MoE with exact equivalence **[V]**

`GPTJXMoEForCausalLM.load_dense_weights_into_moe(dense_state_dict)`:

1. every non-MLP tensor (attention, norms, embeddings) is copied unchanged;
2. each dense `mlp.c_fc.weight` (H×D) is transposed to (D×H) and **cloned into all E experts** of `fc_bank` (E×D×H); `mlp.c_proj.weight` likewise into `proj_bank` (E×H×D);
3. every router (`gate.weight`, E×D) is initialised to **zero**, so the E routing logits are equal;
4. with top-k routing the combination weights are `softmax(top-k logits)` = 1/k each, and since all selected experts are identical, `Σ_k (1/k)·MLP(x) = MLP(x)` — the upcycled model *is* the dense model.

We checked this numerically on a 3-layer, 64-d configuration: dense logits vs upcycled logits differ by **exactly 0.0** for E = 4 and E = 7, k = 2 (the dense and sparse dispatch paths of the same MoE agree to 3.7e-9). Equivalence at step 0 holds by construction; it is not evidence about training dynamics, which needs the loss curves of §12.

Two caveats worth stating in the paper:
* **Symmetry breaking.** With zero routers all logits tie; `topk` tie-breaking is implementation-defined and in practice returns the same two experts for every token, so without noise essentially only those two would receive tokens. Training adds Gaussian noise (σ = 0.1) to the router logits (training mode only, gated on `torch.is_grad_enabled()`), which randomises the choice and lets the experts diverge; the auxiliary loss then pushes toward balance. The paper should show the expert-utilisation curves (they are logged per layer, `04d867c`).
* **Bias-free assumption.** The converter also writes `fc_bias`/`proj_bias` keys that the MoE module does not have (ignored with `strict=False`). That is harmless only because every SabiYarn configuration is bias-free; a dense model with MLP biases would silently lose them.

### 3.4 Routing, balance and the "two experts per token" design

* Router: `gate = Linear(768, E, bias=False)`, init N(0, 0.02). Logits → `topk(k = 2)` → **softmax over the selected two logits only** (not over all E): `y = Σ_{e ∈ top2} p_e · Expert_e(x)`.
* **No capacity limit, no token dropping, no shared expert, no router z-loss** (a z-loss weight 0.01 appears in the config of the earlier `SabiYarn_MoE_translate` model but the current trainer has none; the config also records `moe_aux_loss_weight` 0.05).
* Auxiliary load-balancing loss per layer (Switch/GShard form generalised to k = 2): `load_e` = mean over tokens of the number of times expert *e* is among the top-2 (so Σ load = 2), `P_e` = mean softmax probability over all E experts, `L_aux = E · Σ_e load_e · P_e`. At perfect balance `L_aux` = k = 2. The model-level loss is the mean over MoE layers; the loss added to the cross-entropy is `0.015 · L_aux` (weight history: 0.01 → 0.02 on 24 Jul → 0.015 on 19 Sep 2026). `load` carries no gradient; `P` does.
* In layer 0 (E = k = 2) `L_aux` is identically 2: no gradient, no balancing needed (every token uses both experts).
* The best-checkpoint criterion `val_loss` includes the auxiliary term; `val_ce` (cross-entropy only) is reported separately.
* Why top-2 and not top-1 (**[A]** confirm this was the intent): two experts per token keep the upcycled model exactly dense (weights 1/2 + 1/2), keep routing differentiable through the softmax of two logits, and halve the sensitivity to a single wrong routing decision, at 2× the FFN FLOPs per token of a top-1 design. A top-1 vs top-2 comparison has not been run (§12).

### 3.5 Sparse expert dispatch **[V]**
`MoE._sparse_experts` flattens the (token, slot) assignments, stable-sorts them by expert id, runs expert *e* only on its own rows (`gelu(x[rows] @ fc_bank[e]) @ proj_bank[e]`), scales by the routing probability and `index_add`s the result into the owning token. With top-2 of 4 experts it performs half the expert matmuls of the dense path and stores half the expert activations for backward. The result equals the dense path (max |Δlogit| 3.7e-9 in fp32 in our test; the repository's own check requires gradient cosine ≥ 0.99 and relative L2 ≤ 0.15 between the two paths and reports the worst tensor — `scripts/check_fast_paths.py`). One host synchronisation per layer (`bincount(...).tolist()`) is the price.

### 3.6 Expert editing tools (used before this repository's history)
* `MoE.truncate_experts(keep)` slices gate rows and expert banks in place; `apply_moe_layer_edits(expert_indices_all_layers=…, expert_keep_by_layer=…, zero_router_for_layers=…)` applies it per layer and re-synchronises `config.expert_per_layer`.
* `config.expert_per_layer` makes the layout part of the model configuration, which is how the 2/3/4 layout travels with the checkpoint.

### 3.7 Long context
* The position table has 32,768 rows. Continued pretraining uses blocks of 4,096; SFT uses 32,768 (`sft_block_size`).
* FLOP share: attention overtakes the FFN at T = 8·d = 6,144 tokens; at T = 32,768 attention costs ≈5.3× the MLP (YAML comment). Both the 8.2% parameter cost of `wpe` and this FLOP cost are the price of *absolute learned* positions; going beyond 32,768 would require RoPE/ALiBi, because `wpe` has no further rows.
* Rows ≥ 4,096 of `wpe` are still at their random initialisation after pretraining; §9.3 explains how SFT repairs them.
