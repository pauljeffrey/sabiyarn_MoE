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
