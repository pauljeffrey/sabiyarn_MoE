# SabiYarn-MoE: from a 125M dense model to a 306M sparse multilingual model for 23 African languages

**Technical report, working draft v0.1 — 8 October 2026**
Repository `pauljeffrey/sabiyarn_MoE` (HEAD `df4dccb`, 183 commits, plus the uncommitted work listed in §A.6).
Models on the Hub: `Aletheia-ng/SabiYarn_MoE-280M`, `BeardedMonster/SabiYarn-125M`, tokenizer `BeardedMonster/SabiYarn-32k`.

> **How to read this document.**
> Every statement carries one of three evidence tags.
> **[V]** verified in this session by running code or reading an artifact (file/commit/Hub object is named).
> **[R]** reported by a repository document (README, HOW_TO_RUN, commit message, YAML comment); the number is quoted, not re-measured.
> **[A]** an author-supplied fact that is *not* recoverable from the repository or the Hub; these are collected in Appendix D as questions to answer before submission.
> Where the repository contradicts itself, both versions are given and the conflict is listed in Appendix E.
> Nothing here is an experimental result of the final system: no benchmark score for the 306M model is stored anywhere in the repository (Appendix D, item 1). The report documents *what was built and why*; the paper's results section still has to be produced (§12).

---

## Abstract (draft)

We describe SabiYarn-MoE, a 305.9M-parameter (206.8M active per token) sparse mixture-of-experts language model for 23 languages — English, Nigerian Pidgin and 21 African languages including the major West-African languages (Yorùbá, Hausa, Igbo, Twi, Ewe, Fon, Efik, Ibibio, Urhobo), Southern and Eastern African languages (Swahili, Zulu, Xhosa, Shona, Kinyarwanda, Oromo, Somali, Tswana, Sesotho, Chichewa, Afrikaans) and Wolof. The model began as a 125M-parameter dense GPT-J-style network, was *sparsely upcycled* into a top-2 MoE with a function-preserving initialisation (we verify that the upcycled model reproduces the dense logits exactly at step 0), and was then pruned to a heterogeneous layout of 2/3/4 experts per layer. We cover (i) the systems work that makes a 306M MoE train at 45.6% MFU on A100s — sparse expert dispatch, document-causal FlexAttention, fused cut-cross-entropy; (ii) a data campaign that grew the training mix from the original multi-task tagged corpus to a 12.5B-token, 22-language continued-pretraining bin selected with token-level UniMax sampling, prior-based noisy-text filtering, date/licence rules and a stratified validation set; (iii) a single structured *assistant-turn grammar* — input-language set, optional reasoning, task plan, target language, response, tool calls — used consistently from the chat template through SFT data to RL rewards; (iv) a conversion pipeline that restructures ~6 M heterogeneous legacy and public examples into that grammar with certainty-tiered language identification, content-hash deduplication and a QE-based audit; and (v) RL machinery for low-resource translation with a scaffold/content reward split and per-language-pair calibrated AfriCOMET-QE shaping. We state which claims are backed by measurements in the repository and which still require experiments.

## Candidate contributions and their evidence status

| # | Claim | Evidence in hand | Missing for a paper |
|---|---|---|---|
| C1 | Sparse upcycling of a dense 125M into top-2 MoE is exactly function-preserving at initialisation, and a heterogeneous 2/3/4-expert layout retains ~306M parameters for 206.8M active | [V] logit difference 0.0 for E=4 and E=7 on a tiny config (§3.3); [V] parameter accounting (§3.2) | Training curves of dense vs MoE at equal tokens/FLOPs; the pruning criterion and recipe [A] |
| C2 | Sparse dispatch + document-causal FlexAttention + CCE give 92.2K tokens/s/GPU and 45.6% MFU for a 306M MoE on A100-80GB | [R] measured sweeps (§4) | Ablation of each component on the final model; 4-GPU end-to-end throughput |
| C3 | Token-level UniMax (C = 12.5B, ≤4 epochs) with prior-based filtering is a workable recipe for 22 unequal-resource languages | [V] the realised plan (per-language epochs and discarded tokens, Table 6.4) | Downstream effect vs natural/temperature sampling |
| C4 | One structured assistant-turn grammar across chat template, SFT data and RL reward, including multi-language input tags and an explicit "unsure" form | [V] implementation + tests; [V] tool-call turn renders with the scaffold on the Hub tokenizer | Ablation: structured vs plain targets; calibration of the "unsure" tag |
| C5 | Certainty-tiered language ID + content-hash dedupe + QE audit turns ~6M heterogeneous rows into one schema | [V] build running; [V] QE audit protocol and first measurements (§8.5) | Final yield tables (generated at the end of the build) |
| C6 | Scaffold/content-split RL reward with calibrated AfriCOMET-QE shaping and reliability weighting | [R] design + local validation numbers | A trained policy and its evaluation |
| C7 | SFT-only vocabulary/position-table growth with a PCA+MLP position regressor | [V] code + tests (held-out extrapolation test, baselines) | Whether the regressor beats random-init rows in a real SFT run |

## Table of contents
1. Lineage: from SabiYarn-125M to SabiYarn-MoE-306M
2. Tokenizer
3. Architecture and sparse upcycling
4. Training-systems efficiency
5. Pretraining data, version 1 (the original multi-task corpus)
6. Continued-pretraining data campaign (selection, filtering, cleaning, new languages)
7. Training procedure, monitoring, incidents, current state
8. Post-training data: grammar, synthetic generation, restructuring of existing data
9. Post-training methods: SFT design, model growth, RL
10. Evaluation
11. Limitations, risks, licences
12. Paper plan: structure, missing experiments, figure/table list
- Appendix A: hyper-parameters and tables · B: data-source registry · C: reproduction commands · D: facts only the authors can supply · E: inconsistencies found in the repository · F: glossary of tags


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


## 2. Tokenizer

### 2.1 Construction

* **Type.** Byte-level BPE (HF `tokenizers`), **52,000 base entries (51,714 merges)** plus **117 added tokens** at ids 52,000–52,116 and 45 added at ids 0–44 (specials and structure), giving **52,117 ids** in total; 162 added-token records of which 83 are flagged `special`. `</s>` = 2 is the end-of-text/EOS id, `<s>` the BOS, `<pad>`, `<unk>`. **[V]** (`BeardedMonster/SabiYarn-32k`, `tokenizer_config.json`).
* **Training.** The notebooks `training/tokenizer_training.ipynb` and `data/tokenization (1).ipynb` retrain a tokenizer with `train_new_from_iterator` (starting from the Llama-3 tokenizer's configuration) on the pretraining corpus with a long list of task, language and NER-label special tokens (the NER labels `O B-PER … I-DATE` are in that list but are *not* in the published vocabulary); vocabulary sizes 64,000 and 128,000 appear as experiments, and a corpus token-count comparison (`tiktoken` gpt-4.1/4o vs "sabiyarn" vs "parent" tokenizer) is computed there. The published 52,000-entry vocabulary is the one in the model; **[A]** the exact training corpus, the reason for 52k and what "32k" in the repository name refers to.
* **Model `vocab_size` is 52,050** (embedding rows). Tags with ids ≥ 52,050 (the 66 ids 52,050–52,115, plus `<POS>` = 52,116 added on 2026-10-05) therefore cannot be embedded by the current checkpoint; this includes the language tags `<ibb>`, `<afr>`, `<sot>`, `<nya>`, `<som>`, `<tsn>`, `<kin>`, `<orm>`, `<zul>`, `<xho>`, `<wol>`, `<sna>` and `<|hate|>`. The SFT-time embedding growth that removes this limit is described in §9.3. **[V]**

### 2.2 The special-token vocabulary is a control language

The added tokens form a small *grammar* that the data, the chat template and the RL rewards all use (Appendix F lists them):

| family | examples | count (approx.) |
|---|---|---|
| language tags | `<eng> <pcm> <yor> <hau> <ibo> <twi> <ewe> <fon> <efi> <ibb> <urh> <swh> <zul> <xho> <sna> <afr> <sot> <tsn> <nya> <som> <orm> <kin> <wol> …` (+ ~50 tags for languages we do not train, e.g. `<lug> <lin> <nso> <ton>`) | ~80 |
| structure | `<|input_lang|> <|target_lang|> <task_plan> </task_plan> <response> <think> </think> <tool_call> </tool_call> <tool_response> </tool_response> <context> </context>` | 12 |
| chat roles | `<|system|> <|user|> <|assistant|>` | 3 |
| task verbs | `<translate> <summarize> <classify> <NER> <POS> <qa> <clean> <diacritize> <identify> <correct> <headline> <score>` and pipe verbs `<|chat|> <|generate|> <|RAG|> <|math|> <|code|> <|JSON|> <|data_extract|> <|plan|> <|analyze|> <|explain|> <|recommend|> <|debug|> <|edit|>` | ~25 |
| labels | `<sentiment> <topic> <intent> <toxic> <|hate|> <lang_ID> <answer> <reason> <title> <tag>` | ~12 |
| domain flags | `<|is_technical|> <|is_non_technical|> <|is_legal|> <|is_medical|> <|is_scientific|> <|is_financial|>` | 6 |

Repairs made along the way **[R]**: the chat template emitted `<tool_result>` which is *not* a token (cost ≈5 byte-BPE pieces per tag and +32% tokens on a five-message tool conversation: 83 vs 63); it now emits `<tool_response>` (ids 52,037/52,038). Token 52,043 was renamed in place from `|analyze|>` to `<|analyze|>`. In October 2026 four tags were renamed in place (`<zu>→<zul>`, `<xh>→<xho>`, `<wolof>→<wol>`, `<sn>→<sna>`) and `<POS>` was added (id 52,116).

### 2.3 Fertility (tokens per whitespace word)

Measured on the first 600 documents of the first shard of each language in `BeardedMonster/pretrain-new` (HPLT best-quality bins first; English from `naijaweb-edu`), truncated to 3,000 characters per document; "words" are whitespace-separated. Data: `docs/report/tokenizer_fertility.json`. **[V]**, one sample per language, so treat differences below ~0.1 as noise.

| code | language | words in sample | SabiYarn-32k tok/word | bytes/token | 81,920-vocab | GPT-2 | Llama-3 |
|---|---|---|---|---|---|---|---|
| yor | Yoruba | 318,458 | **1.31** | 4.76 | 3.09 | 4.06 | 2.75 |
| eng | English | 265,572 | **1.31** | 4.70 | 1.95 | 1.30 | 1.26 |
| hau | Hausa | 278,520 | **1.41** | 3.79 | 2.35 | 2.33 | 2.12 |
| ibo | Igbo | 298,706 | **1.47** | 4.16 | 2.81 | 3.71 | 2.55 |
| urh | Urhobo | 303,959 | **1.74** | 3.31 | 2.65 | 3.48 | 2.35 |
| pcm | Nigerian Pidgin | 147,433 | **1.83** | 2.94 | 2.18 | 1.80 | 1.69 |
| tsn | Setswana | 350,690 | **1.90** | 2.72 | 2.17 | 2.05 | 1.95 |
| afr | Afrikaans | 275,237 | **1.92** | 2.73 | 2.14 | 2.02 | 1.70 |
| sot | Sesotho | 285,866 | **1.95** | 2.62 | 2.22 | 2.15 | 1.99 |
| wol | Wolof | 327,854 | **2.06** | 2.56 | 2.46 | 2.40 | 2.15 |
| twi | Twi | 335,385 | **2.13** | 2.54 | 2.51 | 2.86 | 2.75 |
| efi | Efik | 259,663 | **2.29** | 2.55 | 2.88 | 3.00 | 2.76 |
| ibb | Ibibio | 259,057 | **2.35** | 2.76 | 3.03 | 3.60 | 2.73 |
| fon | Fon | 250,871 | **2.39** | 2.18 | 2.70 | 3.33 | 3.07 |
| swh | Swahili | 227,711 | **2.41** | 2.70 | 2.85 | 2.82 | 2.56 |
| ewe | Ewe | 350,962 | **2.47** | 2.27 | 2.70 | 3.08 | 2.82 |
| som | Somali | 235,731 | **2.60** | 2.50 | 2.97 | 2.92 | 2.68 |
| nya | Chichewa | 211,686 | **2.61** | 2.67 | 3.05 | 3.10 | 2.77 |
| kin | Kinyarwanda | 197,967 | **2.83** | 2.56 | 3.29 | 3.24 | 2.94 |
| orm | Oromo | 237,829 | **2.92** | 2.62 | 3.32 | 3.25 | 3.01 |
| xho | Xhosa | 171,004 | **3.49** | 2.49 | 3.88 | 3.98 | 3.58 |
| zul | Zulu | 161,354 | **3.57** | 2.51 | 4.01 | 4.08 | 3.65 |
| sna | Shona | 178,983 | **4.03** | 2.18 | 4.00 | 4.67 | 4.37 |

Reading the table:
* The SabiYarn tokenizer is **dramatically better than GPT-2 and Llama-3 on the languages it was trained for**: Yorùbá 1.31 vs 4.06/2.75, Igbo 1.47 vs 3.71/2.55, Hausa 1.41 vs 2.33/2.12, Urhobo 1.74 vs 3.48/2.35, Ibibio 2.35 vs 3.60/2.73. English is on par with GPT-2 (1.31 vs 1.30).
* It is **poor for the agglutinative Bantu languages added later**: Shona 4.03, Zulu 3.57, Xhosa 3.49, Kinyarwanda 2.83 — comparable to or only slightly better than Llama-3 (4.37/3.65/3.58/2.94). These are the languages with the highest cost per word in the continued-pretraining plan (a Shona word costs ≈3× a Yorùbá word in tokens, i.e. 3× fewer words per sequence and per FLOP).
* The 81,920-vocabulary variant is *worse* than the 52k one on this sample for every language (**[A]** how it was built; its fertility on English, 1.95, shows it is not simply a superset), so it was not adopted.
* The same fertility drove two decisions: Amharic was dropped from the data plan (measured 7.55 tokens/word vs Pidgin 1.24, leaving a 75-word document in a 1,024-token budget), and the data-generation budgets were re-derived from measured per-language fertility (Appendix A.4).

### 2.4 Open design question for the paper
A tokenizer extension (or a retrained larger vocabulary that includes the Bantu languages) would reduce sequence cost for roughly six of the 23 languages (the size of the gain has not been measured), but it changes `vocab_size` and invalidates all token bins; the current plan keeps the tokenizer fixed and only grows the embedding table (§9.3).


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


## 5. Pretraining data, version 1 (the corpus the current checkpoint was trained on)

### 5.1 What is on disk / on the Hub **[V]** (`v1_data_struct.json`: file listings of 27 Hub dataset repos)

The bins consumed by the current trainer (`training/train_config.yaml`) are three tokenised, uint16, one-`</s>`-per-document files plus a validation bin. Sizes are quoted from the YAML **[R]**:

| bin | size | tokens | content |
|---|---|---|---|
| `eng_training.bin` | 24.5 GB | ≈13.1B (3.21M blocks of 4,096) | English |
| `training_cleaned.bin` | 44.3 GB | ≈23.8B (5.80M blocks) | the "afr" bin: Nigerian / West-African languages (+ English-side of translation pairs); the name `afr` means *African*, not Afrikaans |
| `other_african_pretrain.bin` | 25.0 GB | 12.5B | the 22-language UniMax bin of §6 (third train bin) |
| `validation.bin` → `validation_v2.bin` | 3.3 GB → 3.62 GB | ≈1.8B (+19.7M new) | stratified validation (§6.7) |

Mixture currently configured: eng 0.15 / afr 0.55 / other African 0.30, fixed (`use_scheduled_sampling: false`); the older curve (afr weight 0.45 → 0.95) is kept inert.

The repositories that fed version 1 (names only; contents were reconstructed from file names, not re-read) **[V]**:

* **Monolingual web text:** `GlotCC-V1`, `DCAD-2000`, `Glot500` (ewe, fon, fuv, hau, ibo, … shards) in `pretrain_data3/6/7`; `BeardedMonster/NLP2` and `naija` (Hausa, Igbo, Pidgin, Efik, Fulfulde, Urhobo monolingual parquet shards).
* **Parallel text:** `mt560` (English–Yorùbá), `Pontoon-Translations`, `IGBONLP`, `sentence-pairs` for Akan/Ewe/Fon (`low_resource_languages_pretrain`, `…_data4`), Aya translation shards (`naija-translation`, `trs`: 1,316 files each).
* **Task data in the old tagged format** (§8.2): `naija`, `naija2`, `naija3` (≈250–370 files each; AfroLingu-MT, Global-MMLU, MIRACL, InjongoIntent, Business-Knowledge …), `naija_sft`, `eng_sft` (alpaca-style jsonl), `rag_synthetic_data`.
* **English quality text:** `cosmopedia`, a Bloomberg-news pretraining set.

### 5.2 The multi-task tagged format of version 1 **[R]/[V]**
The 2024 SabiYarn recipe (the reason the repositories above contain *task* data inside the pretraining repos) formats every training example, translations included, as a single string with language/task tags and trains it with the ordinary causal-LM loss over the whole text; tasks are therefore learned during pretraining, not in a separate fine-tuning phase. The v1 grammar uses *language names* and two tags per example, pads plans with newlines and has no target-language tag; §8.2 describes how it is parsed and converted to the current grammar.

### 5.3 What we cannot reconstruct **[A]**
* the exact per-source token counts of `training_cleaned.bin` (the trainer's own accounting counts 1.30B tokens for it and 0.23B for English because of how `tokens_seen` is estimated — see §7.5; the YAML sizes imply ≈23.8B);
* which of the sources above were deduplicated against each other, and with what method;
* the filters that produced "cleaned" (the YAML says only that the bin is `training_cleaned`).
These need author input (Appendix D, items 5–6).


## 6. Continued-pretraining data campaign (selection, filtering, cleaning, new languages)

### 6.1 Goal and constraints
The version-1 corpus covered 11 named languages plus Fulfulde (eng, pcm, hau, ibo, yor, efi, ibb, urh, twi, ewe, fon, plus Fulfulde). The campaign of October 2026 added **12 languages** — Kinyarwanda, Shona, Zulu, Swahili, Oromo, Wolof, Xhosa, Tswana, Chichewa, Afrikaans, Somali, Sesotho — and refreshed the original ones. Amharic was evaluated and **dropped** after the tokenizer-efficiency review (§2.3); "for" in the request meant Fon, not Fulfulde. The constraints below were set by the authors and are the *selection criteria* of the campaign **[V]** (memory log, code):

| rule | detail |
|---|---|
| **Recency** | original languages: data created ≥ 2025-08-01 (the model was already trained on older public data; the aim is *new* text); the 12 new languages: ≥ 2021-05-01 (little data exists at all); datasets named explicitly by the authors are exempt; Wikipedia monthly dumps allowed for the new languages only |
| **Licence** | no non-commercial licences (Sagalee, LDS talks, CC-BY-NC-SA sets dropped); Watchtower/JW text explicitly allowed |
| **LLM-written data** | admitted only if a quality column can gate it (only `HuggingFaceFW/finetranslations` has one), except sets the authors named |
| **Hygiene** | no benchmark/test splits (FLORES, AfroLingu test, uhura-eval excluded); no speech datasets except for the lowest-resource languages |
| **Compute etiquette** | one niced process, stream-and-delete, disk guard ≤ 35% of free space, worker count kept low (the machine is also the authors' workstation) |
| **Scale** | `pretrain-new` on the Hub: columns `text`, `lang`, `score`; 100k–5M documents per language (per-language cap 8M, longest documents first when a source exceeds it); languages under 100k documents are spared near-duplicate removal (exact dedupe only) |

### 6.2 Cleaning and filtering **[V]** (`data/prepare_unimax.py::rule_check`, `prior_select`, `final_split`)
1. **Rule filters (Gopher / C4 style)**, applied per document, first failing rule wins: fewer than 20 or more than 200,000 words; mean word length outside [2.2, 14]; (#, "...", "…") per word > 0.10; alphabetic-word share < 0.75; more than 30% of lines ending in an ellipsis; more than 90% bullet lines; duplicate lines covering > 30% of characters; most-frequent 2/3/4-gram covering > 20/18/16% of characters; duplicated 5/8/10-gram covering > 15/12/10% (first 1,000 words only, for cost). A `relax` factor loosens the repetition thresholds for formulaic text such as scripture in the smallest languages.
2. **Prior-based noisy-text filtering** (Seo et al., arXiv 2509.18577), applied **only to languages for which UniMax allots less than one epoch** (`U_l < count_l`: Afrikaans, Swahili, Somali). A token prior p(x) is estimated from a 10% sample of the training documents (+0.5 smoothing); each document gets μ_d = mean log-prior and σ_d = std of the prior averaged over 512-token blocks; documents farthest from the language's median μ and median σ are discarded (k grown until the union of the k farthest by each criterion holds the required number of tokens). Rationale: where more data exists than the budget can use, spend the budget on typical text; where every token will be used at least once, filtering only discards scarce signal.
3. Exactly one `</s>` (id 2) per document; attention never crosses documents (document-causal mask, §4.3).
4. **Validation split:** a document hash marks validation *candidates*; candidates are kept in random order until 1M tokens per language (`MAX_VAL_TOKENS`) and the surplus returns to training, so no language contributes more than ≈ 1M validation tokens. Ibibio, Urhobo, Yorùbá, Twi, Igbo and Fon have fewer because their corpora are small.
5. **Epoch construction:** for a language with U_l larger than its kept data, the sequence is whole epochs (fresh shuffle each) plus a random part; the whole bin is then globally shuffled at document level. Sources flagged *tail* (§6.5) are appended after the shuffle.

### 6.3 Token-level UniMax sampling **[V]** (`unimax()` implements Algorithm 1 of Chung et al., 2023)
Languages are sorted by available tokens ascending; the i-th language receives `min(count × max_repeat, remaining_budget / (L − i))`. With budget C = 12.5B tokens, max_repeat = 4 and L = 22 the realised plan is (per-language columns from `plan.json`):

| lang | unique train tokens available | removed by prior filter | unique tokens kept | tokens written (target U_l) | passes over kept data | sampling share | natural share | validation tokens |
|---|---|---|---|---|---|---|---|---|
| afr | 4,089.8M | 3,029.5M | 1,060.3M | 1,060.3M | 1.00 | 8.48% | 35.74% | 1.00M |
| swh | 2,374.8M | 1,314.5M | 1,060.3M | 1,060.3M | 1.00 | 8.48% | 20.76% | 1.00M |
| som | 1,708.9M | 648.6M | 1,060.3M | 1,060.3M | 1.00 | 8.48% | 14.94% | 1.00M |
| zul | 672.7M | 0.0M | 672.7M | 1,060.3M | 1.58 | 8.48% | 5.88% | 1.00M |
| xho | 538.4M | 0.0M | 538.4M | 1,060.3M | 1.97 | 8.48% | 4.71% | 1.00M |
| kin | 462.9M | 0.0M | 462.9M | 1,060.3M | 2.29 | 8.48% | 4.05% | 1.00M |
| sna | 310.3M | 0.0M | 310.3M | 1,060.3M | 3.42 | 8.48% | 2.71% | 1.00M |
| nya | 279.4M | 0.0M | 279.4M | 1,060.3M | 3.79 | 8.48% | 2.44% | 1.00M |
| hau | 229.2M | 0.0M | 229.2M | 916.7M | 4.00 | 7.33% | 2.00% | 1.00M |
| sot | 217.0M | 0.0M | 217.0M | 868.2M | 4.00 | 6.95% | 1.90% | 1.00M |
| orm | 144.2M | 0.0M | 144.2M | 576.9M | 4.00 | 4.62% | 1.26% | 1.00M |
| ibo | 89.0M | 0.0M | 89.0M | 356.1M | 4.00 | 2.85% | 0.78% | 0.91M |
| yor | 84.0M | 0.0M | 84.0M | 336.1M | 4.00 | 2.69% | 0.73% | 0.78M |
| twi | 78.4M | 0.0M | 78.4M | 313.6M | 4.00 | 2.51% | 0.69% | 0.74M |
| ewe | 62.8M | 0.0M | 62.8M | 251.3M | 4.00 | 2.01% | 0.55% | 1.00M |
| tsn | 47.3M | 0.0M | 47.3M | 189.1M | 4.00 | 1.51% | 0.41% | 1.00M |
| pcm | 17.3M | 0.0M | 17.3M | 69.3M | 4.00 | 0.55% | 0.15% | 1.00M |
| wol | 13.8M | 0.0M | 13.8M | 55.1M | 4.00 | 0.44% | 0.12% | 1.00M |
| efi | 10.6M | 0.0M | 10.6M | 42.6M | 4.00 | 0.34% | 0.09% | 1.00M |
| fon | 8.1M | 0.0M | 8.1M | 32.5M | 4.00 | 0.26% | 0.07% | 0.95M |
| ibb | 1.4M | 0.0M | 1.4M | 5.6M | 4.00 | 0.04% | 0.01% | 0.17M |
| urh | 1.2M | 0.0M | 1.2M | 4.7M | 4.00 | 0.04% | 0.01% | 0.13M |

*Table 6.4.* Totals: 11.44B unique tokens available; **12,500,012,710 tokens written in 12,658,634 documents** (12,658,769 `</s>` tokens; the file ends with `</s>`); 19,696,583 validation tokens.
Reading the table:
* **Capped from above (< 1 pass):** afr (0.26 of the data used), swh (0.45), som (0.62). These are the three languages on which the prior-based filter removed 3.03B, 1.31B and 0.65B tokens.
* **Equal-share plateau, repeated 1.6–3.8×:** zul, xho, kin, sna, nya each receive 1.06B tokens (8.48% of the bin) although only 0.28–0.67B unique tokens exist, so they are repeated.
* **Exhausted at the 4-pass limit:** from Hausa (7.33%) down, every language is shown 4 times (hau, sot, orm, ibo, yor, twi, ewe, tsn, pcm, wol, efi, fon, ibb, urh). These are the languages most exposed to memorisation and where the validation curve should be watched for divergence between train and validation loss. Four passes follow Muennighoff et al. (2023), who observed little loss increase up to 4 epochs of repeated data.
* The headline effect of UniMax is the contrast with natural sampling: Afrikaans would take 35.7% of the bin and Ibibio 0.01%; here Afrikaans takes 8.5%, Hausa 7.3% and Yorùbá 2.7%.
* Two points a reviewer will ask: (i) the bin is sampled by *block* with fixed mixture weights eng 0.15 / afr 0.55 / other 0.30, so the passes over Table 6.4 depend on how many steps are run — at 786,432 tokens per step the 12.5B-token bin is one full pass after ≈15.9k steps if it were the only bin, and ≈53k steps at its 0.30 weight (derived); (ii) UniMax treats languages as interchangeable units while tokenizer fertility differs (§2.3), so a per-language *byte* budget would be a different plan.

### 6.5 Placement of web-scale English-language Nigerian text ("tail")
The three NaijaWeb sets (`naijaweb`, `naijaweb-edu`, `naijaweb-edu2`) are Nigerian English from the web with an educational-quality score. They are written to a separate `eng-tail` folder and **appended at the end** of `other_african_pretrain.bin` (`TAIL_DIRS`), so that at the end of an epoch-ordered pass the model sees the highest-quality, on-domain English last (the same end-loaded curriculum used for annealing in other recipes). Because the bin is sampled by random block, the effect of "last" is realised only if the bin is consumed sequentially; whether it is, is a decision for the continued-pretraining launch (Appendix D, item 9).

### 6.6 New pretraining sources queued but not yet merged **[V]** (`sft_new/pretrain_add.py`)
`BeardedMonster/lrln`, `BeardedMonster/monolingual`, `Aletheia-ng/pidgin-corpus-synth` (title + content merged, long documents chunked), `Aletheia-ng/yoruba-corpus-synth`, `yo_wiki_processed`, Glot500-Swahili, polynews-Swahili, the NaijaWeb trio (→ tail), `text_ratings` (text column only), `datagen_pretrain`, and the **monolingual leftovers of the legacy sets** (67 `*_monolingual` families in `naija`/`naija2`, ≈25 GB, "late" folders, used last). Each source goes through the same hash-dedupe table (`Seen`) as the SFT build, so a document that appears in an existing bin is dropped.

### 6.7 Evaluation set for the new bin **[V]** (`training/eval_plan.py`)
`validation_v2.bin` = the old validation bin (1.8B tokens, segment `old`) + ≤ 1M tokens per new language (segments `afr`, `swh`, …) described by a manifest JSON. A fixed set of 500 blocks of 4,096 tokens (≈ 2.05M tokens) is evaluated: 40% from `old`, 60% shared equally by the language segments, equally spaced inside each segment, never crossing a segment boundary, deterministic from the seed. Logged: `val_loss` (token-weighted, used for best-checkpoint selection), `val_loss_old`, `val_loss_lang_macro`, per-language `val_loss/<lang>` and **bits-per-byte** `val_bpb/<lang>` (token losses are not comparable across languages because fertility differs, §2.3).

### 6.8 What the paper should add to this section
* the fraction of each source removed by each rule filter (counts are not stored; a re-run of `prepare.py` with `--stats` would produce them);
* a noisy-text-filter ablation on Swahili (train 1.06B filtered vs unfiltered tokens, compare `val_bpb/swh`);
* the contamination check between `validation_v2` and the SFT sets (the hash set exists; the cross-check does not).


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


## 8. Post-training data: one grammar, synthetic generation, restructuring of existing data

### 8.1 The assistant-turn grammar (single source of truth) **[V]**
`data-gen/assemble.py::assistant_content`, `data-gen/sft_new/grammar.py` (the parser used by the tests), `rl/structure.py` (the parser used by the reward) and `sabiyarn/chat_template.jinja` (the renderer) implement one grammar:

```
<|input_lang|> L1 [L2 [L3]]            every language of the user turn, in reading order, once each (max 3)
   or  <|input_lang|> unsure, but most likely L1 [L2 …]        (the hedged form)
[<think> … </think>]                    optional reasoning, starts at the decision (not at the greeting)
<task_plan> [domain flag] <verb> [<label> | <tag>] … </task_plan>
<|target_lang|> Lt                      (hedged form allowed)
<response> … </s>
```
where `Lk = <hau>`, `<ibo>`, …, e.g. `<|input_lang|><ibo><efi><task_plan><translate></task_plan><|target_lang|><eng><response>`.
Design points, each a deliberate decision recorded in the work log:
* **All input languages, not only the "text" language.** A user turn such as an Igbo instruction around an Efik passage contains two languages; the early RL data tagged only one (`<efi>`), which the authors spotted and had corrected (`<ibo><efi>`). The correction is implemented end to end: assembler, SFT composer (`turn_languages`), RL structure scoring (0.5 × Jaccard partial credit when the set is right but the order is wrong), RL bin metadata (`ins`, `pos`: instruction language and position) and the chat template. The RL v4 data on the Hub still carry the single-tag field and are regenerated in RL v5.
* **Hedged form.** When language identification is not certain (§8.4) the data say so (`unsure, but most likely <yor>`) instead of asserting a label, which trains the model to express uncertainty about the *language of the input*, one of the most common failure modes on low-resource text.
* **Label tokens inside the plan.** Classification labels and NER/POS tags live inside `<task_plan>` (`<classify><intent>`, `<NER><tag>`), domain flags come first (`<|is_*|>`, §8.3). `<POS>` was added to the tokenizer (id 52,116) for this.
* **Tool-call turns follow the same scaffold.** The chat template previously dropped the `<|input_lang|>…<task_plan>` prefix of RAG tool-call turns; it now emits the turn's content before `<tool_call>`, so a function-calling turn is `scaffold + <tool_call>…</tool_call>`. The template was pushed to the Hub tokenizer `BeardedMonster/SabiYarn-32k` (commit `b0b1122`); the `Aletheia-ng` copy of the tokenizer is untouched.
* `<|input_lang|>` etc. are single special tokens; the tags for 14 of the 23 languages (ids ≥ 52,050) do not yet have embedding rows — the SFT-only vocabulary growth of §9.3 supplies them.

### 8.2 Three families of SFT data
| family | origin | how it enters |
|---|---|---|
| **A. Synthetic, generated** (13 → 12 non-English languages) | LLM generation pipeline in `data-gen/` (§8.3) | JSON → deterministic assembly → chat template; already in the grammar |
| **B. Existing tagged data** (`naija`, `naija2`, `tds`, `african_sft`, `hate`, `intent`, `topic`, `pos`, `summarization`, `eng_sft`, `naija_sft`, `rag_synthetic_data`, …) | the authors' older repositories in the **v1 multi-task format** (language *names*, two tags, no target tag, newline-padded plans) | parsed (`old_format.py`, `tagged_parse.py`), validated, re-expressed in the grammar |
| **C. Public datasets in other formats** (WebScraped Igbo–English, IgboSenti-BBC, hausa_voa_topics, aya_collection, aya-translate, English–Efik Bible, TinyStories-Igbo/Yoruba, ApolloMoE, gpt-teacher-yo, nvidia_steer_yo, Roleplay-Hausa, MMMLU-Yoruba, wiki-yo, coqa-stories, AI_Storyteller, gsm8k variants, multilingual MMLU train/dev, low_resource_multilingual*, …) | Hub | one converter per dataset (`converters.py`), each producing records of the same schema |

### 8.3 Synthetic generation (family A) **[R]** from `data-gen/README.md` and commit messages
* **Principle:** the generator never writes special tokens. It returns typed JSON validated against a schema; Python assembles the conversation deterministically and renders it through the tokenizer's real `chat_template.jinja`. Special-token syntax therefore cannot be wrong, and a template change re-renders every past batch with zero regeneration. RAG "retrieval" is always grounded in real chunk text injected after generation.
* **Task types:** RAG (tool call naming a chunk id), summarisation (document and conversation), edge-device action triggering (tools, reasoning, plan, tool call, tool result, answer), structured output (10 JSON domains), math/statistics (with and without a `calculate` tool), translation phrased naturally. Later work added chunked retrieval with three real use cases, native-language source documents (Wikipedia pages in the target language, 925-document pool; MasakhaNEWS was measured *worse* than the wikis on both axes and dropped), two-turn pairing, and a language-direction dimension. Behaviour tasks move to RL, world knowledge to SFT (`c557202`).
* **Generator choice was measured, not assumed:** `gemma-3-27b` gave 89% clean low-resource pretraining text where `gemma-4-31b` gave 6% (`c23b985`); Llama 3.x and Gemini Flash were compared and lost (`23d05ef`); the free tier produced 0 of 24 samples in 163 minutes (`c2e74b9`) and fails for two independent reasons (`7339c1f`).
* **Yields and cost (all from commit bodies):** RL path from 4% to 44% yield (`0617a2e`), then 69% (`2fd63d9`); a contract A/B raised Doubleword yield 35% → 45% (`8efe5db`); provider-side JSON-schema decoding made the data *worse* in two ways (`8c04019`); 4,736 SFT conversations across 13 languages cost $9.67 (`48b9670`); the sourced-document pool's cost fell from $5.58 to $2.21 (`79a7b86`).
* **Gates:** n-gram looping checks, recycled-vocabulary rejection (`7320e59`), invented-pipe-token rejection, distractor-tool injection after generation, judge/discriminator scores replacing a forced judge (`090e25a`, `57df113`), "quality from scores" (`a495b24`).
* **Limitation:** the pipeline still writes single `<|input_lang|>` tags and must be updated to the multi-language form before its output is merged; the combined build includes only the already-published `datagen_sft` set (4,042 rows kept after the language checks; the build log shows 568 + 157 + 73 rows dropped for unsupported language or low confidence).

### 8.4 Restructuring existing data (families B and C) **[V]** — `data-gen/sft_new/`
**Goal:** every record becomes `Rec(task, dataset, src, tgt, input, instruction, position, plan, label, think, response, unsure flags, …)`, is validated against the grammar, rendered once, hashed and counted by (language, task).

1. **Parsing the old format** (`old_format.py`, `tagged_parse.py`): first/second-tag regexes for the 2024 grammar (including the text/title/summarise/headline variants), task families, label recovery (UPOS/NER inventories checked token by token; records with invalid labels are dropped — e.g. 97,427 POS records), plan normalisation (the padded `<chunk>` plan token is rejected — 20,744 records).
2. **Language identification with evidence tiers** (`langdetect.py`): GlotLID top-3 (mapped to our 23 languages) combined with function-word evidence mined from `pretrain-new` (`stopwords.json`; English function words removed from the other lists). Outcome per segment: *certain* (GlotLID agrees and stopword evidence), *likely* (GlotLID plus stopword score ≥ 0.02), or *none*. File-name languages are never trusted: `sib200_fon` is Afrikaans, for example. A **family prior** (the language distribution of the source family) resolves very short texts. Foreign-language segments inside a text are detected and the record is either tagged with the extra language (up to 3) or dropped.
3. **Rules (all set by the authors):** a record needs at least two special tags to be accepted; unsupported or undeterminable languages are dropped (or, for monolingual leftovers, routed to pretraining "at the very end", `late/<lang>`); every translation dataset is used in both directions (`pair_stream`, `AYA_SKIP` avoids the eng↔hau pairs already covered); uncertain *tasks* are routed to pretraining; the benchmarks FLORES, AfroLingu test splits, Math-Shepherd and uhura-eval are excluded; MMMLU-Yoruba is used as *train*.
4. **Instruction templates:** large pools per task in English, Pidgin, Yorùbá, Hausa, Igbo, Twi and Ewe (summarise, headline, sentiment, topic, hate, intent, story, story prompt, synopsis, answer-in-language, wiki ask, MCQ, math, QA-with-context, NER, POS) so the model sees varied instruction language and *position* (before or after the text) — the user's requirement that "the pool of multilingual instruction templates must be large".
5. **Reframing as RAG:** everything that can be a context-based QA becomes `rag_qa` (CONTEXT_QUESTION), wiki-yo becomes a real tool-call RAG conversation with a `SEARCH_TOOLS` pool, `rag_synthetic_data` is converted to the same form.
6. **Deduplication:** a content-hash of (task, src, tgt, input — or the first user message when there is no separate input — and response), stored in an on-disk SQLite table (`Seen`) with a per-step tag so that a failed step can be forgotten and re-run. *Bug found and fixed:* the first version ignored the prompt for classification rows, so 99.9% of hate/intent rows were declared duplicates; the key now includes the first user message and a test pins it.
7. **Licences:** `naija`/`naija2` carry XL-Sum rows whose licence is CC-BY-NC-SA; the authors asked for those sets not to be skipped, so the rows are kept and flagged (`metadata.license_flag`) — a flag, not an exclusion, to be resolved before any public release (§11).
8. **Reproducible orchestration** (`combine.py`): one step per converter, naija/naija2 split into one step per source family (≈95 steps); each step is resumable (`stats/<tag>.json`), retried three times, guarded by a disk monitor (≤ 35% of free space), pushes shards to the private repo `BeardedMonster/sft-all`, and deletes local shards only after a successful push. Per step it keeps up to 40 short translation rows per (source, language pair) for the quality audit (§8.5).
9. **Caps:** translation is capped at 3M rows per language (only Afrikaans was affected); the Afrikaans rows beyond 1M that had been deleted from `sft-new` are rebuilt from `multi-open` and `finetranslations`.

### 8.5 Quality audit of translation data with AfriCOMET-QE **[V]**
Reference-free quality estimation with `masakhane/africomet-qe-stl-1.1` (run in a separate Python 3.11 virtualenv; ≈ 1–5 pairs/s on the authors' CPU). Protocol (`quality_check.py`, `audit_translation.py`):
* score a uniform sample of up to 30 short rows (≤ 500 characters both sides) per (source dataset, language pair), because QE falls with length (the same known-good Igbo set scores 0.63 on sentences and 0.46 on paragraphs);
* skip pairs where QE is unreliable: its own validity run on FLORES+ (42 pairs × 100 sentences) gave AUC 0.90 against mismatched eng-centric pairs and 0.79 for inter-language pairs, and four pairs below 0.7 (tsn>sot, sot>tsn, fon>ewe, ewe>twi) are excluded;
* **failure line 0.46** (a fluent but wrong sentence scores ≈ 0.44–0.46, half-omitted ≈ 0.42; the authors' hand check agrees that <0.46 is gibberish);
* a group is put on the **drop list** when ≥ 20 rows were scored, ≥ 80% of them are below 0.46 and the mean is below 0.38; the list is applied when the data are re-downloaded for tokenisation; nothing on the Hub is deleted.
**Measured before the full audit** (80 pairs each): controls `english_to_igbo` 0.63 and WebScraped 0.67; `aya-translate` eng→yor 0.52 (7% below the line), eng→ibo 0.43 (68% below), eng→hau 0.33 (95% below); `aya_interlanguage` yor→ibo 0.47, hau→yor 0.55; in `aya_collection` 94% of the QA targets carry machine-translation numbered-list artefacts (removed by `AYA_LIST_MARKERS`). The first judgement that these sets were "low quality" had **not** been measured; this audit replaced it with numbers, and the full-build results are in §8.6.

### 8.6 Result of the full build (generated)
The tables in `docs/report/generated/` (`sft_all_stats.md`, `sft_all_audit.md`) are written by `docs/report/make_tables.py` after the build and the audit finish and are concatenated in §A.7.


## 9. Post-training methods: SFT design, model growth, RL

### 9.1 SFT configuration **[V]** (`training/train_config.yaml`, SFT-only keys)
* **Context:** `sft_block_size` = 32,768 (the model's own limit: the position table has 32,768 rows). Long-document summarisation samples carry 4k–16k tokens of document plus the summary; at 4,096 the *summary* — the only supervised part of that sample — would be the part truncated, so the model would learn to emit documents and stop. Generated documents run 1,492–12,481 words (median 4,344).
* **Batch:** 1 × 32,768 × 24 accumulation = 786,432 tokens per step, i.e. the *same tokens per optimiser step* as pretraining (192 × 4,096), so the schedule need not be re-tuned. Flex attention is required at this length (an SDPA boolean mask would be ≈ 800 MiB per layer per sample at 16k).
* **Loss masking (in the trainer, never in the bin)** — `training/label_masking.py`, `process_sft_labels_turns(tokens, assistant_token, eos_token, mask, mask_spans)`: the loss covers **every assistant turn** — scaffold (`<|input_lang|>…<|target_lang|>`), reasoning, plan, tool calls, final answers and the closing `</s>`. It excludes system and user turns and every `<tool_response>…</tool_response>` and `<context>…</context>` span wherever it appears (anything that imitates an input to the model). Reason for doing it in the trainer: the bin stays a faithful token stream usable with any masking policy; the same bin serves ablations. The earlier pretraining-time tag-pair masking is a different mechanism (§7.3).
* **Data:** `sft-all` (§8) + `datagen_sft`, deduplicated, language × task balanced by the combined build's caps; the QE drop list applied when re-downloading.

### 9.2 Evidence that the chosen scaffold is learnable **[A]**
No SFT run with the new grammar exists yet. The first experiments to run are listed in §12 (structured vs plain targets; effect of the hedged form; masked vs unmasked tool responses).

### 9.3 SFT-only growth of the vocabulary and the position table **[V]** (`sabiyarn/model/resize.py`, `position_regressor.py`; tests `tests/test_resize.py`)
Why SFT-only: the tokenizer was extended (language tags for 14 languages from `<ibb>` at id 52,050 upward, and `<POS>` at 52,116), but the checkpoint has 52,050 embedding rows, so every id ≥ 52,050 is out of range. The authors decided the architecture must not change during continued pretraining (the published `config.json` stays 52,050 × 32,768), so the change happens when SFT loads the model:

1. **Vocabulary:** 52,050 → **52,160** rows (52,117 ids rounded up to a multiple of 64 for tensor-core efficiency): +110k parameters. New rows = mean of the old rows + small noise (the standard "mean embedding" initialisation, which keeps the new tokens' initial logits near the average rather than extreme); the tied `lm_head` grows with it.
2. **Position table:** optional growth, **default none** (`sft_resize_block_size: 0`). Growing to 65,536 rows would add 25.2M parameters (+8.2% of the model); the authors rejected this because the goal is to keep the model small and no SFT sample exceeds 32,768 tokens. The module docstring in `resize.py` still describes the rejected 64k plan (Appendix E).
3. **Filling the untrained rows.** Pretraining only trained `wpe[:4096]`; rows 4,096…32,767 still hold their random initialisation (so any context beyond 4k starts SFT from noise). A **PCA + MLP regressor** treats the trained rows as a multivariate series indexed by position: PCA (≥ 99.5% variance, ≤ 128 components) → an MLP from position (scalar plus multi-scale sin/cos features) to the PCA coordinates → inverse PCA. `fit` always holds out the tail of the trained range, predicts it from the head and compares with two baselines (the mean row; the last trained row); `fill_untrained_positions` writes the guesses **only if the regressor beats both** — otherwise the rows are left alone. `estimate_trained_rows` estimates how many rows were really trained, so the 4,096 assumption can be checked (`sft_fill_positions_from`).
4. *Scientific caveat for the paper:* extrapolating a learned absolute table by regression is a heuristic. The honest comparison is three SFT starts on long-context data — random rows, regressor rows, mean-plus-noise rows — evaluated on long-document tasks; until that exists the regressor is an engineering hypothesis (claim C7).

### 9.4 Reinforcement learning machinery **[V: code and tests; R: validation numbers in commit bodies]**
`rl/` contains four algorithms behind one config (`algo: sft | dpo | rl | rs`): SFT warm-start, DPO (sigmoid/IPO/hinge losses, ranked-pair records produced by data-gen), a GRPO-family policy gradient (REINFORCE with a group-mean baseline A = r − mean_group(r), optionally ÷ std; per-token k3 KL penalty to a frozen reference; no value function — a ~300M model with a sentence-level reward does not need a critic), and rejection sampling (filter an existing SFT set by QE; or best-of-N from the current policy → new SFT records).

**Reward for translation (`StructuredTranslationReward`, `rl/structure.py`, `rl/shaping.py`).** The reward separates *scaffold* from *content*:

R = w_f · scaffold + (1 − w_f) · gate · content, with w_f = 0.2.

* **Scaffold** (`format_score`): field-wise match of the generated scaffold against the sample's expected one: input-language set (0.5 × Jaccard partial credit for the right set in the wrong order), plan landmarks in order (plan internals order-agnostic F1), target language, presence of `<response>`. The expected structure is computed from the prompt metadata (`src`, `tgt`, instruction language, position). The completion is decoded *with* special tokens (`wants_raw`).
* **Content**: reference-free AfriCOMET-QE (`reward_use_reference: false`), shaped by a per-language-pair piecewise-linear curve with four regions (failure −2…0 below t_fail; 0…+1 to t_mid; +1…+2 to t_great; a tiny slope beyond, so a score above what *human* translations obtain earns almost nothing: that region is where a learned metric is gamed). The anchors are **measured**, not guessed: t_fail/t_mid/t_great = P10/P50/P90 of QE scores of *human* reference translations on the benchmark (`calibrate_shaping.py`); t_fail is never below 0.46. An `adaptive` variant lets t_mid/t_great follow rolling quantiles of the policy's own scores while t_fail stays anchored.
* **Gate:** content counts only when the target-language tag is right and the response is non-empty and clean — otherwise a policy could earn content reward in the wrong language; the additive format term keeps a gradient on the scaffold while the gate is closed. A length-ratio guard halves the content score for implausibly short/long output and a copy of the source scores 0 (the classic way to fool a QE metric).
* The ranking among shaping variants proposed to the authors (best → worst): calibrated > adaptive > threshold > none. Affine maps of the reward are irrelevant for group-normalised advantages; the shaping matters only through its *nonlinearity* (it spends resolution where the policy lives and punishes failures).
* QE validity (FLORES+ devtest, 42 pairs × 100): AUC 0.90 vs mismatched eng-centric, 0.79 inter-language; human P10/P50/P90 mean .51/.65/.76 (eng-centric) and .46/.59/.71 (inter-language). Ibibio and Urhobo are outside AfriCOMET's languages and are not in the RL set.

**Behaviour tasks** (tool use, grounding) are scored by programmatic verifiers (`rl/verifiers.py`: did the model call a tool before claiming to know; is the quoted figure in the document; was the stated constraint obeyed) — no learned metric in the loop. A blended mode `verifier+africomet` (`reward_blend` 0.5) mixes both.

**RL data.** A prompt bin in compressed-sparse-row layout (`rl/bin_data.py`; metadata `src, tgt, ins, pos`) built from translation windows (v3 non-NC windows + 12 new-language windows from `pretrain-new`; v4 on the Hub). **RL v5** (pending) adds the multi-language input field and larger per-language quotas drawn from monolingual text, plus chunks of `pidgin-corpus-synth` and `coqa-stories` (`text_pcm`).

### 9.5 Order of operations recommended to the authors
continued pretraining → SFT (grammar learnt, tags embedded, masked loss) → rejection-sampling / DPO on QE-ranked pairs → GRPO-style RL with the scaffold/content reward. "SFT first, then RL" because RL reward on the scaffold is only meaningful once the model can produce it.


## 10. Evaluation

### 10.1 What exists **[V]**
* **Validation loss on fixed, stratified blocks** (§6.7) — the only signal that has been run on the 306M model.
* **`eval_suite/`** (`python -m eval_suite.run`): translation English↔X (FLORES-200 devtest, else MAFAND-MT; chrF++ headline, chrF, BLEU; optional AfriCOMET), topic classification (SIB-200, MasakhaNEWS), sentiment (AfriSenti), NER (MasakhaNER 2.0, entity-level F1), MMLU (AfriMMLU + English; 5-shot; likelihood-scored). Languages with a standard benchmark: yor, hau, ibo, twi, ewe, fon, pcm (+ English); Efik, Urhobo, Fulah have none and report `n/a`. Prompts exist in two styles: `tag` (the v1 multitask format, for base checkpoints) and `chat` (the SFT template).
* **`eval/eval.py`** — the SabiYarn-125M evaluation; **not comparable** with the suite (NER by per-sentence label presence; topics by keyword rules).
* **AfriCOMET-QE** — reference-free, used for data audit and RL reward (§8.5, §9.4).
* **Start-up generation probes** and the hand-written curated probe set (`eval` loss on a probe file; the YAML warns "if it is far below train/val loss, suspect the bins, not the model").

### 10.2 What is missing for a paper **[A]**
1. No benchmark number for the 306M model is stored in the repository (Appendix D, item 1).
2. The eval suite covers the original languages; it has no task data for kin, sna, zul, swh, orm, wol, xho, tsn, nya, afr, som, sot. Candidate benchmarks: FLORES+ (translation, all of them), SIB-200 (topic, all of them), AfriMMLU / Global-MMLU where available, MasakhaNEWS, MasakhaNER 2.0 (kin, sna, zul, swh, xho, wol, nya …), AfriSenti (swh, orm, kin …), Belebele (reading comprehension), AfriXNLI. The MMLU "train" splits were used as SFT data (MMMLU-Yoruba and multilingual MMLU dev) — **dev/test separation must be kept** when these benchmarks are reported.
3. Decontamination between the bins/SFT sets and every evaluation set (hash n-gram overlap against FLORES+, SIB-200 and the MMLU test splits).
4. Comparable baselines at the same budget: the dense 125M, a 306M dense with the same active parameters, and public multilingual models of similar size (e.g. mT5-small/base, AfroXLMR, Gemma-3 270M / SmolLM2 360M where their language coverage permits), using the same prompts.
5. Metrics for the assistant-turn grammar itself: exact-match of the scaffold (input-language set, plan verbs, target language), calibration of the hedged `unsure` form (accuracy when unsure vs when certain), and tool-call validity.
6. Efficiency measurements in the paper's format: tokens/s and MFU at the *final* configuration on 4 × A100, and active-vs-total FLOP comparison against the dense baseline at equal quality.


## 11. Limitations, risks, licences

**Scientific**
1. *No downstream result yet.* Everything about quality of the 306M model rests on validation loss (§7.5, §10.2).
2. *Upcycling + pruning provenance is missing* (Appendix D): the experiments that would show the MoE beats the dense model at equal training FLOPs are not in the repository.
3. *Absolute learned positions* cap the context at 32,768, cost 8.2% of the parameters, and 87% of the table (rows ≥ 4,096) has never been trained (§3.7, §9.3). A comparison with RoPE is outside the current plan.
4. *The 12 new languages' tags are first embedded in SFT* (§7.6): during continued pretraining they are learned from text only.
5. *Repetition:* languages from Hausa down are seen four times (§6.4); memorisation and a rising train/validation gap are expected in the smallest languages.
6. *Synthetic and machine-translated data:* LLM-written data are admitted only with a gating score; Aya-type collections carry machine-translation artefacts (94% of `aya_collection` QA targets); the QE audit is a sample (≈30 rows per group), has a failure line calibrated on one hand-checked set, and is blind to adequacy errors QE does not catch; QE itself is unreliable for four language pairs and absent for Ibibio/Urhobo.
7. *Language identification* is the weakest link of the restructuring: GlotLID + stop-words are reliable for long texts and unreliable for short or code-switched ones; the hedged `unsure` form and the family prior mitigate but do not remove label noise in `<|input_lang|>`.
8. *Reward hacking* of a learned metric (AfriCOMET-QE) is a known risk; the design (calibrated anchors, flat top, chrF++ anchor option, copy guard, scaffold gate) reduces it but there is no human evaluation yet.

**Data licences and provenance (must be resolved before releasing weights or data)**
* The authors' rule was *no non-commercial licence*; one deliberate exception was made on request: `xlsum` rows inside `naija`/`naija2` (CC-BY-NC-SA) are kept and flagged `license_flag`. Decide: exclude, or release the model as research-only, or ask for permission.
* Watchtower/JW text is explicitly admitted by the authors; its redistribution terms differ from its reading terms. Keep it in training data only, do not republish it.
* Aya collection and public benchmark-derived sets keep their own licences (several Apache/CC-BY, some CC-BY-SA); the combined `sft-all` repository is **private** and should stay so until a per-source licence table exists (`docs/report/generated/sft_all_stats.md` lists source names).
* LLM-generated data carries the provider's terms (the generator model names are in the data-gen manifests).
* Personal data: web-scraped text has not been through a PII scrub; none is documented in the repository.

**Safety.** No safety/harm evaluation exists. The SFT sets contain a hate-speech classification set (`hate`) whose *inputs* are offensive by construction; they are classification inputs, but a safety review of the data and the model is needed before deployment.

**Compute & environmental note.** Throughput numbers are in §4; a total GPU-hour count for the project is not recorded (Appendix D, item 14).


## 12. Paper plan: structure, missing experiments, figures/tables

### 12.1 Suggested framing
A systems-and-data paper with one clean empirical claim per section, rather than a "new model" paper:
**"Building a small multilingual MoE for 23 African languages under a tiny budget: function-preserving upcycling, token-level UniMax with prior filtering, and a single structured output grammar from pretraining data to RL reward."**
(Alternative short framing: an *experience/resource* paper for a workshop such as AfricaNLP, MRL or a COLM/EMNLP findings submission.)

### 12.2 Outline → source sections
| paper § | content | report § | status |
|---|---|---|---|
| 1 Intro | motivation, 23 languages, constraints | 0, 1 | text exists |
| 2 Related work | upcycling (Komatsuzaki 2022), Switch/GShard/Mixtral/Qwen-MoE, MoE pruning, UniMax, Seo et al., African LMs (AfroXLMR, Serengeti, Lugha-Llama, SabiYarn-125M) | – | to write |
| 3 Model | block, upcycling proof, heterogeneous experts, sparse dispatch | 3, 4 | ready |
| 4 Tokenizer | fertility table, tag families | 2 | ready (add fertility for new languages) |
| 5 Pretraining data | v1, campaign, UniMax table, validation | 5, 6 | ready, ablations missing |
| 6 Training | recipe, divergence case study, efficiency | 4, 7 | ready |
| 7 Post-training data | grammar, restructuring, audit | 8 | ready, results table generated by `make_tables.py` |
| 8 SFT/RL | masking, growth, reward | 9 | method ready, **no results** |
| 9 Evaluation | benchmarks | 10 | **missing** |
| 10 Limitations / Ethics | licences, safety | 11 | ready |

### 12.3 Experiments that would turn claims into results (ordered by value per GPU-hour)
1. **Evaluate the current checkpoint** (per-language val loss/bpb, FLORES+/SIB-200/Belebele zero/few-shot) — a baseline for every later table (cheap, no training).
2. **Upcycling ablation at small scale:** dense 125M continued vs 4-expert upcycled vs 4-expert from-scratch at equal tokens (C1); report the router-collapse diagnostic of §7.4.
3. **Pruning ablation:** which experts were removed and what it cost (needs the original 7-expert checkpoint; the recipe is [A]).
4. **UniMax vs natural vs temperature sampling** at 1–2B-token scale with the same bin builder (C3).
5. **Prior-based filter ablation** on Swahili or Somali (train with/without filtering at equal tokens).
6. **Grammar ablation:** SFT with structured scaffold vs plain chat, on translation and classification; effect of multi-language input tags and of the hedged form (C4).
7. **Position-regressor ablation** (C7): random vs mean-noise vs regressor rows on 16k–32k tasks.
8. **Masking ablation:** loss on tool responses/contexts vs masked.
9. **RL:** scaffold-only vs content-only vs combined reward; calibrated vs threshold shaping; human preference on 100 pairs per language.
10. **Efficiency table:** the sweep of §4.2 with each component switched off on the final model (sparse dispatch, flex mask, CCE).

### 12.4 Figures and tables already producible from files in this repository
Table: parameter accounting (§3.2). Table: tokenizer fertility (`tokenizer_fertility.json`). Table: UniMax plan (§6.4) + bar chart of epochs vs language. Figure: validation loss before/after the divergence (§7.4 numbers). Table: MFU sweep (§4.2). Figure: language × task heat-map of `sft-all` (generated). Table: QE audit per source (generated). Figure: scaffold grammar diagram (§8.1).

### 12.5 Reproducibility package
Commit hash `df4dccb` + the uncommitted tree (commit it before submission); `HOW_TO_RUN.md`; Hub repos (models, `pretrain-new`, `sft-new`, `sft-all`, `africomet-translation`); the tests (553 passed / 2 skipped in `data-gen`, repository tests green as of 7 Oct 2026).


## Appendix A. Hyper-parameters and tables

### A.1 Model (`sabiyarn/model/configuration.py`, SabiYarn_MoE-280M)
12 layers · d 768 · 12 heads · MLP width 3,072 · experts per layer {0:2, 1:3, 2–11:4} · top-2 · router noise σ 0.1 · aux-loss weight 0.015 · vocab 52,050 (SFT: 52,160) · block 32,768 · LayerNorm ε 1e-5 · no biases · GELU · dropout 0 · tied embeddings · sparse dispatch on.

### A.2 Pretraining
AdamW β (0.9, 0.95), wd 0.1, clip 1.0, peak LR 1e-5, WSD (20% sqrt cool-down to 0), `warmup_iters` 18,400, `max_tokens` 50e9 (fallback `max_iters` 60,000), block 4,096, micro-batch 12, accumulation 16 global (4 per GPU), 4 × A100-80GB, DDP, bf16 autocast with fp32 master weights, CCE on, activation checkpointing off, eval every 150 iterations (500 stratified blocks), log every 100, mixture eng/afr/other = 0.15/0.55/0.30.

### A.3 SFT-only
Block 32,768 · accumulation 24 · loss masked to assistant turns excluding `<tool_response>` and `<context>` spans · vocab → 52,160 · position growth 0 · position fill from row 4,096 (PCA ≥ 99.5% variance, ≤ 128 comps + MLP, written only when it beats the mean-row and last-row baselines).

### A.4 RL / post-training
`algo` ∈ {sft, dpo, rl, rs}; reward `structured_africomet`, w_f 0.2, QE only (no reference), failure line 0.46, shaping calibrated/adaptive/threshold/none on −2…+2; `max_prompt_len` 28,672; response cap 2,048 for RL candidates (an earlier mismatch 4,096 vs 2,048 was fixed in `5443a99`).

### A.5 Token statistics
Tokenizer: 52,117 ids. Version-1 bins ≈ 13.1B (eng) + ≈ 23.8B (afr) tokens by file size; new bin 12,500,012,710 tokens / 12,658,634 documents; validation 1.812B + 19.7M tokens. Trainer lifetime estimate 16.99B tokens at iteration 21,600.

### A.6 Uncommitted work at the time of writing (HEAD `df4dccb`)
39 modified and 33 untracked paths, among them: the `sft_new` package (conversion, language detection, combine/dedupe/audit/pretrain-add), `data/prepare_unimax.py`, `rl/{bin_data,rejection_sampling,shaping}.py`, `rlhf/scripts/{build_rl_bin,calibrate_shaping}.py`, `sabiyarn/model/{resize,position_regressor}.py`, `training/{eval_plan,log_format,token_accounting}.py`, new tests, the dataset cards (`data-gen/cards/`), and this report (`docs/`). **Commit before submission so that the paper cites a hash.**

### A.7 Generated tables
`docs/report/generated/*.md` are produced by `python docs/report/make_tables.py BUILD_DIR` and appended by `docs/report/assemble.sh`.

## Appendix B. Source registry (restructured SFT data)
Converter names in `data-gen/sft_new/converters.py` (≈ 95 build steps). Grouped:

| group | converters / sources |
|---|---|
| translation | WebScraped Igbo–English, English–Igbo, English–Efik Bible, aya-translate, aya_interlanguage_translate, aya_collection (language split), Apollo-MoE, low_resource_multilingual(_short, _short2), xP3x-Kongo (dropped: target always Kikongo), multi-open, finetranslations, parallel-sentences-opus-100, swim-ir |
| classification | hate, intent, topic, hausa_voa_topics (labels inferred: 0 Africa, 1 Health, 2 Nigeria, 3 Politics, 4 World), IgboSenti-BBC, SIB-200 (mapping inferred from 3 sentences) |
| sequence labelling | pos (UPOS validated), ner |
| generation | TinyStories-Igbo / -Yoruba, AI_Storyteller, coqa-stories, Roleplay-Hausa, gpt-teacher-yo, nvidia_steer_yo, summarisation sets |
| QA / RAG | wiki-yo (tool-call RAG), rag_synthetic_data, CoQA, qa-with-context rows reframed as RAG |
| reasoning / knowledge | gsm8k variants, MMMLU-Yoruba (train), Global-MMLU dev, MMLU-ProX validation, multilingual MMLU train |
| legacy tagged | naija, naija2 (one step per source family; train + validation splits only), tds, african_sft, eng_sft, naija_sft, `sft_new_existing` |
| synthetic | `datagen_sft` |
| dropped | sumnews (non-target languages), uhura-eval (excluded by the authors), Math-Shepherd, uhura-instruct, `tasks/data/all` (duplicate of per-task files), 35k blank rows at the start of `tasks` |

## Appendix C. Reproduction commands
```bash
# environment
uv sync                                  # repo; QE needs .venv-comet (py3.11): see HOW_TO_RUN §2.14
# tokenise the new pretraining bin (budget 12.5B, repeat 4)
python data/prepare_unimax.py --budget 12.5e9 --max-repeat 4
# build the combined SFT dataset (resumable; low-priority single process)
cd data-gen && python -m sft_new.combine --all --out SCRATCH/sft-all --push
python -m sft_new.audit_translation SCRATCH/sft-all          # AfriCOMET-QE audit -> audit.json, drop_list.json
python -m sft_new.combine --stats SCRATCH/sft-all             # language × task tables
# train (4 GPUs)
torchrun --nproc_per_node 4 -m training.new_train
# evaluate
python -m eval_suite.run --model out_280M/<run>/ckpt_best --tasks all --langs all --limit 200 --name base
# report tables
python docs/report/make_tables.py SCRATCH/sft-all && bash docs/report/assemble.sh
```
(Exact flags: `HOW_TO_RUN.md`.)

## Appendix D. Facts only the authors can supply
1. Any benchmark or probe result for the 306M model (none stored).
2. The **pruning** recipe: which experts were kept in each layer and why (utilisation? router mass? weight norm?), tokens trained before and after pruning, and the loss change from pruning.
3. The dense → 4,096-context → 32,768-context steps: tokens used, LR, whether the position table was extended by copy or by new random rows; what "32k" meant at the time.
4. Token counts and loss at each earlier phase (dense 125M, 7-expert, 4-expert, 2/3/4 layout).
5. Per-source token counts of `training_cleaned.bin` and `eng_training.bin`.
6. How "cleaned" was produced (filters, dedupe method, dedupe across bins).
7. Status of `Sabiyarn_MoE-750M` (trained? how far?).
8. Whether the 12 new language tags were intentionally left un-embedded during continued pretraining (§7.6), and whether any other mechanism teaches them.
9. Whether `other_african_pretrain.bin` is consumed sequentially (so the NaijaWeb tail is really last) or sampled randomly by block.
10. Which validation set the 3.1365 figure refers to (old vs `validation_v2`).
11. How `tokens_seen/<bin>` counters relate to the 16.99B lifetime estimate (restart offsets).
12. Intended licence of weights and data; decision on the `xlsum` rows.
13. Intent behind top-2 routing (vs top-1) and behind two experts in layer 0.
14. Total GPU-hours and cost to date (pretraining, generation, QE).
15. Names and affiliations for the author list; which results the authors want to claim.

## Appendix E. Inconsistencies found in the repository
1. `README.md` describes RoPE / MLA / LoRA components that are not in the model code (the model uses learned absolute positions, plain MHA, full fine-tuning).
2. `HOW_TO_RUN.md` says `model.code` is local while the YAML has a different default for it in places; the `tool_result` vs `tool_response` spelling differs between `data-gen/README.md` ("tool_result") and the current template/tokenizer (`<tool_response>`).
3. `data-gen/assemble.py` still carries a comment that `<POS>` is not a tokenizer token; it is (id 52,116).
4. `sabiyarn/model/resize.py` docstring describes the abandoned 64k-context plan (65,536 positions, `SFT_BLOCK_SIZE = 65536`); the trainer default is no position growth.
5. `scripts/estimate_training_plan.py` assumes MFU 0.25 on A100; measured MFU is 0.456.
6. Divergence date: the YAML comment says 23 Sep 2026, the recovery commit is stamped 22 Sep 22:25 (UTC−6); the YAML also says "6e-4 for a 300M model".
7. `LN_J` is created with `bias=768` (truthy) so it has a bias while the other norms do not (§3.1) — probably unintended.
8. `load_dense_weights_into_moe` writes `fc_bias`/`proj_bias` keys that do not exist in the MoE module (§3.3).
9. `Aletheia-ng/SabiYarn_MoE-280M` tokenizer config still carries an older template (`<tool_result>`); `BeardedMonster/SabiYarn-32k` has the current one.
10. `data-gen` generation pipeline still emits single `<|input_lang|>` tags (§8.3).
11. The `afr` bin name means *African*, not Afrikaans, and is confusable with the new `afr` language code.
12. The HOW_TO_RUN fast-path result (cosine 0.99987, relative L2 0.0161) is printed without date/GPU.

## Appendix F. Glossary of tags
| tag | meaning |
|---|---|
| `<|input_lang|> <xx>…` | languages of the full user input, reading order, ≤ 3; or `unsure, but most likely <xx>` |
| `<|target_lang|> <xx>` | language of the answer |
| `<think>…</think>` | optional reasoning that starts at the decision |
| `<task_plan>…</task_plan>` | the plan: optional domain flag(s) `<|is_*|>`, verb (`<translate>`, `<classify>`, `<summarize>`, `<NER>`, `<POS>`, …), optional label token or `<tag>` |
| `<response>` | start of the answer |
| `<tool_call>…</tool_call>` / `<tool_response>…</tool_response>` | function call by the model / result returned by the tool (masked in SFT) |
| `<context>…</context>` | retrieved context (masked in SFT) |
| `</s>` (id 2) | end of document / end of assistant turn |
| language codes | eng, pcm, hau, ibo, yor, efi, ibb, urh, twi, ewe, fon, kin, sna, zul, swh, orm, wol, xho, tsn, nya, afr, som, sot (ISO 639-3 style; `efi` Efik, `ibb` Ibibio, `urh` Urhobo, `pcm` Nigerian Pidgin) |




## A.7 Result of the `sft-all` build (7 steps finished)

**Rows kept: 6,025,319**; rows routed to late pretraining folders: 0; total build time of the finished steps: 0.4 h.

### By converter

| converter | kept | late rows | seconds |
|---|---|---|---|
| sft_new_existing | 5,627,582 | 0 | 531 |
| old_tasks | 245,460 | 0 | 613 |
| old_intent | 56,434 | 0 | 80 |
| old_hate | 45,821 | 0 | 75 |
| old_pos | 27,212 | 0 | 56 |
| old_topic | 18,768 | 0 | 115 |
| datagen_sft | 4,042 | 0 | 12 |

### By task

| task | rows | share |
|---|---|---|
| unknown | 5,171,887 | 85.8% |
| sentiment | 236,595 | 3.9% |
| intent | 111,857 | 1.9% |
| hate | 80,631 | 1.3% |
| clean | 66,000 | 1.1% |
| identify | 65,922 | 1.1% |
| ner | 63,653 | 1.1% |
| topic | 47,848 | 0.8% |
| structured_output | 34,164 | 0.6% |
| translation_score | 32,399 | 0.5% |
| correct | 29,999 | 0.5% |
| diacritize | 22,560 | 0.4% |
| summarization | 22,388 | 0.4% |
| pos | 15,676 | 0.3% |
| toxic | 6,192 | 0.1% |
| chat | 4,838 | 0.1% |
| qa | 4,774 | 0.1% |
| summarize | 2,046 | 0.0% |
| headline | 2,046 | 0.0% |
| insufficient-context | 509 | 0.0% |
| extractive-qa | 492 | 0.0% |
| knowledge-boundary | 469 | 0.0% |
| content-writing | 408 | 0.0% |
| financial-analysis | 333 | 0.0% |
| health-education | 320 | 0.0% |
| health-advice | 297 | 0.0% |
| multi-turn-chat | 296 | 0.0% |
| health-triaging | 257 | 0.0% |
| structured-output | 138 | 0.0% |
| intent-detection | 114 | 0.0% |
| language-identification | 86 | 0.0% |
| pos-tagging | 42 | 0.0% |
| sentiment-analysis | 34 | 0.0% |
| tool-calling | 20 | 0.0% |
| toxicity-spam-detection | 13 | 0.0% |
| topic-classification | 12 | 0.0% |
| translation | 4 | 0.0% |

### By language (as source / as target)

| language | as source | as target |
|---|---|---|
| eng | 2,549,809 | 3,115,608 |
| swh | 682,950 | 460,105 |
| yor | 585,805 | 477,447 |
| afr | 540,518 | 512,086 |
| hau | 257,411 | 241,493 |
| xho | 238,653 | 220,569 |
| ibo | 176,716 | 164,852 |
| zul | 126,646 | 107,284 |
| som | 117,016 | 109,793 |
| kin | 108,762 | 87,929 |
| sot | 96,663 | 85,087 |
| sna | 94,587 | 81,930 |
| nya | 83,303 | 79,521 |
| twi | 67,712 | 56,777 |
| wol | 59,512 | 39,295 |
| fon | 51,474 | 47,208 |
| orm | 55,335 | 35,406 |
| tsn | 48,014 | 38,980 |
| pcm | 34,741 | 30,650 |
| ewe | 28,384 | 21,025 |
| efi | 9,138 | 6,113 |
| urh | 6,177 | 3,161 |
| ibb | 5,993 | 3,000 |

### Languages per input (`<|input_lang|>` size)

| languages in input | rows |
|---|---|
| 1 | 5,700,926 |
| 2 | 324,393 |

### Top 40 language pairs

| src>tgt | rows |
|---|---|
| swh>eng | 679,616 |
| afr>eng | 527,703 |
| eng>afr | 499,271 |
| eng>swh | 456,771 |
| yor>eng | 389,613 |
| eng>yor | 281,252 |
| xho>eng | 229,256 |
| hau>eng | 217,100 |
| eng>xho | 211,172 |
| eng>hau | 201,186 |
| yor>yor | 196,131 |
| ibo>eng | 138,182 |
| eng>ibo | 126,304 |
| zul>eng | 111,308 |
| som>eng | 111,158 |
| kin>eng | 104,434 |
| eng>som | 103,935 |
| sot>eng | 93,663 |
| eng>zul | 91,946 |
| sna>eng | 87,704 |
| eng>kin | 83,601 |
| eng>sot | 82,087 |
| nya>eng | 79,641 |
| eng>nya | 75,859 |
| eng>sna | 75,047 |
| eng>eng | 67,302 |
| twi>eng | 60,560 |
| orm>eng | 52,335 |
| eng>twi | 49,684 |
| wol>eng | 48,729 |
| fon>eng | 44,424 |
| hau>hau | 40,247 |
| eng>fon | 40,145 |
| ibo>ibo | 38,487 |
| tsn>eng | 37,694 |
| eng>orm | 32,406 |
| pcm>pcm | 29,987 |
| eng>tsn | 28,660 |
| eng>wol | 28,512 |
| ewe>eng | 21,391 |

### Most frequent drop reasons

| reason | rows |
|---|---|
| config_skipped | 654,303 |
| pos_labels_invalid | 97,427 |
| blank_row | 35,379 |
| ner_labels_invalid | 31,545 |
| duplicate | 21,391 |
| plan_token:<chunk> | 20,744 |
| empty_response | 5,734 |
| unsupported_language_config | 2,400 |
| language_not_supported | 568 |
| assistant_language_not_supported | 157 |
| low_confidence_or_markers | 73 |
