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
