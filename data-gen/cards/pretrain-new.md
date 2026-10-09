---
license: other
license_name: mixed-per-source-see-card
language:
- af
- efi
- ee
- fon
- ha
- ibb
- ig
- rw
- ny
- om
- pcm
- sn
- so
- st
- sw
- tn
- tw
- urh
- wo
- xh
- yo
- zu
pretty_name: SabiYarn pretrain-new
size_categories:
- 10M<n<100M
task_categories:
- text-generation
tags:
- african-languages
- pretraining
- low-resource
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---

# SabiYarn pretrain-new

12,442,028 cleaned, deduplicated documents in 22 African languages. This is the text behind , the third training bin of the SabiYarn MoE (≈306M parameters) pretraining mix; it is also the monolingual source of the RL translation prompts ([](https://huggingface.co/datasets/BeardedMonster/africomet-translation)).

## Fields

| field | meaning |
|---|---|
|  | one document |
|  | language code (), also the folder name under  |
|  | quality score on a 0-10 scale from the source, where the source has one (HPLT's document score); otherwise a fixed default. Use it to rank within a source, not across sources |

## Documents per language

| code | language | documents |
|---|---|---|
| afr | Afrikaans | 3,882,735 |
| swh | Swahili | 2,874,225 |
| som | Somali | 2,312,797 |
| zul | Zulu | 678,090 |
| hau | Hausa | 508,484 |
| xho | Xhosa | 450,318 |
| kin | Kinyarwanda | 407,246 |
| sot | Sesotho | 249,178 |
| sna | Shona | 240,333 |
| nya | Chichewa | 206,468 |
| ibo | Igbo | 127,490 |
| orm | Oromo | 125,546 |
| yor | Yoruba | 117,298 |
| twi | Twi | 85,759 |
| pcm | Nigerian Pidgin | 53,114 |
| tsn | Setswana | 49,001 |
| ewe | Ewe | 40,376 |
| wol | Wolof | 17,392 |
| efi | Efik | 8,125 |
| fon | Fon | 5,693 |
| ibb | Ibibio | 1,183 |
| urh | Urhobo | 1,177 |

Documents are 100k-8M per language. A language with more than 8M candidate documents keeps the longest ones. Languages under 100k documents were only de-duplicated exactly (near-duplicate removal would have discarded too much of a small corpus).

## How it was built

* **Sources** (streamed, never stored in full): HPLT 3.0 and FinePDFs (best quality bins first), AfriSpeech  and  (scripture and jw.org text), Wikipedia and Wikipedia-monthly (only for the twelve languages added later: ), MADLAD-400 / WURA / GlotCC / Masakhane news for the legacy and news passes, speech-corpus *transcripts* for the lowest-resource languages (never audio), and a handful of language-specific corpora found by a wide Hub sweep.
* **Date rule.** Only data created on or after **2025-08-01** for the original languages and **2021-05-01** for the twelve added languages, apart from datasets the owner named explicitly (, , , ).
* **No non-commercial licences.** Sources licensed CC-BY-NC / NC-SA were removed. The jw.org text is included at the owner's explicit instruction; its copyright is not cleared.
* **No language-model-written text** unless a quality column can measure it (the only such source is , which is used in the SFT data, not here).
* **Cleaning.** Unicode NFC, control characters, URLs, e-mail addresses and phone numbers removed; boilerplate lines (cookie banners, share buttons, "read more") dropped; per-language minimum length (150-250 characters, 25-40 words); language checked by GlotLID on sampled lines; then exact and near-duplicate (shared paragraph) removal against everything already kept for the language.
* **Not stored:** per-document provenance. The licences of the sources differ, so the dataset as a whole is only as permissive as its most restrictive source.

## Planned additions

Documents from , , , , , , ,  and  are being added under ; the data-gen pretraining shards too.

## Intended use and limits

Pretraining a small multilingual model. Web text carries web biases (religion, politics, sport and news are over-represented); the Bible-derived languages (Efik, Ibibio, Urhobo, Fon) are dominated by scripture and by jw.org. No personal-data scrub beyond URL / e-mail / phone removal was done.
