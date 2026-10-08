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
