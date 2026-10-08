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
