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
