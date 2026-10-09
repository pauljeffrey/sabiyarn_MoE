"""Write the dataset cards (README.md) for the pretrain and SFT repos from live row counts.

    python cards/make_cards.py COUNTS.json OUT_DIR           # COUNTS.json: {"repo": {"counts": {"data/<dir>": rows}, "schema": [...]}}
    python cards/make_cards.py COUNTS.json OUT_DIR --push    # also upload each README.md to its repo

The counts come from the parquet footers (no data is downloaded): see data-gen/sft_new/combine.py --stats for the combined SFT repo.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NAMES = {"afr": "Afrikaans", "efi": "Efik", "ewe": "Ewe", "fon": "Fon", "hau": "Hausa", "ibb": "Ibibio", "ibo": "Igbo", "kin": "Kinyarwanda",
         "nya": "Chichewa", "orm": "Oromo", "pcm": "Nigerian Pidgin", "sna": "Shona", "som": "Somali", "sot": "Sesotho", "swh": "Swahili",
         "tsn": "Setswana", "twi": "Twi", "urh": "Urhobo", "wol": "Wolof", "xho": "Xhosa", "yor": "Yoruba", "zul": "Zulu", "eng": "English"}
ISO2 = {"afr": "af", "efi": "efi", "ewe": "ee", "fon": "fon", "hau": "ha", "ibb": "ibb", "ibo": "ig", "kin": "rw", "nya": "ny", "orm": "om", "pcm": "pcm",
        "sna": "sn", "som": "so", "sot": "st", "swh": "sw", "tsn": "tn", "twi": "tw", "urh": "urh", "wol": "wo", "xho": "xh", "yor": "yo", "zul": "zu", "eng": "en"}


def pretrain_card(counts: dict) -> str:
    pre = {k.split("/")[-1]: v for k, v in counts.items()}
    rows = "\n".join(f"| {k} | {NAMES[k]} | {v:,} |" for k, v in sorted(pre.items(), key=lambda x: -x[1]))
    langs = "\n".join(f"- {ISO2[k]}" for k in sorted(pre))
    return f"""---
license: other
license_name: mixed-per-source-see-card
language:
{langs}
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

{sum(pre.values()):,} cleaned, de-duplicated documents in {len(pre)} African languages. This is the text behind `other_african_pretrain.bin`, the third
training bin of the SabiYarn MoE (about 306M parameters) pretraining mix. It is also the monolingual source of the RL translation
prompts in [`BeardedMonster/africomet-translation`](https://huggingface.co/datasets/BeardedMonster/africomet-translation).

## Fields

| field | meaning |
|---|---|
| `text` | one document |
| `lang` | language code, also the folder under `data/` |
| `score` | quality score on a 0-10 scale from the source where the source has one (HPLT's document score); otherwise a fixed default. Rank within a source, not across sources |

## Documents per language

| code | language | documents |
|---|---|---|
{rows}

A language with more than 8M candidate documents keeps the longest ones. Languages under 100k documents were only de-duplicated exactly:
near-duplicate removal would have discarded too much of a small corpus.

## How it was built

* **Sources** (streamed, never stored in full): HPLT 3.0 and FinePDFs (best quality bins first); AfriSpeech `africa-corpus` and
  `african-corpus-jw` (scripture and jw.org text); Wikipedia and Wikipedia-monthly (only for the twelve languages added later:
  `afr kin nya orm sna som sot swh tsn wol xho zul`); MADLAD-400, WURA, GlotCC and Masakhane news for the legacy and news passes;
  speech-corpus *transcripts* (never audio) for the lowest-resource languages; and a few language-specific corpora found by a wide Hub sweep.
* **Date rule.** Data created on or after 2025-08-01 for the original languages and on or after 2021-05-01 for the twelve added languages,
  apart from datasets the owner named explicitly (`African-Languages-Lab/multi-open`, `AfriSpeech/africa-corpus`,
  `nthakur/swim-ir-cross-lingual`, `sentence-transformers/parallel-sentences-opus-100`).
* **No non-commercial licences.** CC-BY-NC and CC-BY-NC-SA sources were removed. The jw.org text is included at the owner's explicit
  instruction; its copyright is not cleared.
* **No model-written text** unless a quality column can measure it.
* **Cleaning.** NFC normalisation; control characters, URLs, e-mail addresses and phone numbers removed; boilerplate lines (cookie
  banners, share buttons, "read more") dropped; per-language minimum length (150-250 characters, 25-40 words); language checked by
  GlotLID on sampled lines; exact and near-duplicate (shared paragraph) removal against everything already kept for the language.
* **Not stored:** per-document provenance. Source licences differ, so the dataset as a whole is only as permissive as its most
  restrictive source.

## Being added

`BeardedMonster/lrln`, `monolingual`, `Aletheia-ng/pidgin-corpus-synth`, `yoruba-corpus-synth`, `yo_wiki_processed`, `Glot500-swahili`,
`polynews-swahili`, `naijaweb`, `naijaweb-edu`, `naijaweb-edu2`, `text_ratings` and the pretraining shards of `BeardedMonster/data-gen`
(`data/<lang>/add-*.parquet`). Some of these are model-written (the `*-synth` sets); they were named explicitly by the owner.

## Use and limits

Pretraining a small multilingual model. Web text carries web biases (religion, politics, sport and news are over-represented); the
Bible-derived languages (Efik, Ibibio, Urhobo, Fon) are dominated by scripture and by jw.org. Personal data is only scrubbed for URLs,
e-mail addresses and phone numbers.
"""


def sft_card(counts: dict) -> str:
    by = defaultdict(int)
    for k, v in counts.items():
        name = k.split("/")[-1]
        kind = re.split(r"-(?=[a-z]{3}$)", name)[0] if re.search(r"-[a-z]{3}$", name) else name
        by[kind] += v
    lines = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(by.items(), key=lambda x: -x[1]))
    return f"""---
license: other
license_name: mixed-per-source-see-card
language:
- en
- yo
- ha
- ig
- pcm
- ee
- fon
- tw
- efi
- af
- zu
- xh
- sn
- ny
- st
- tn
- rw
- om
- wo
- so
- sw
pretty_name: SabiYarn SFT (first build)
task_categories:
- translation
- text-generation
- text-classification
tags:
- african-languages
- sft
- translation
- instruction-tuning
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---

# SabiYarn SFT, first build

{sum(by.values()):,} rows in the SabiYarn chat structure. Superseded by `BeardedMonster/sft-all`, which contains these rows and everything built since.

## The structure of a row

`messages` is a chat ready for `apply_chat_template`; `text` is that chat rendered with the SabiYarn template (`<s><|user|>...</s><|assistant|>...</s>`),
which is the string that gets tokenized. The assistant turn always has the same shape:

```
<|input_lang|><src>[<think>..</think>]<task_plan>[domain flag]<verb>[label or <tag>]</task_plan><|target_lang|><tgt><response>CONTENT
```

* `<src>` is the language of the *input text*, `<tgt>` the language of the *answer* (labels are English).
* Label tokens (`<intent> <sentiment> <topic> <|hate|> <toxic> <headline> <lang_ID> <score> <correct>`) and the `<tag>` marker of tagging tasks sit inside
  `<task_plan>` next to the verb. `<summary>` is never used: a summary is `<task_plan><summarize></task_plan>`.
* `output_structure` is the assistant turn up to and including `<response>`: what the model has to produce before the content (the RL reward scores it).

| column | meaning |
|---|---|
| `instruction`, `input`, `response` | the request, the text it applies to, the answer |
| `language` | language of the input text |
| `task` | `translation`, `intent`, `hate`, `ner`, `pos`, `identify`, `clean`, `diacritize`, ... |
| `messages`, `text`, `output_structure` | as above |
| `source_dataset` | provenance of the row |

## Rows by source group

| group | rows |
|---|---|
{lines}

## How it was built

* **Translation** (both directions): `African-Languages-Lab/multi-open` rows with a judge score of 6 or more (scores 1-5 failed a GlotLID
  check 15-80% of the time), `HuggingFaceFW/finetranslations` (model-written English side, gated on its `edu_score >= 2`,
  `og_language_score >= 0.95` and `early_stop == False`, at most 100k per language) and OPUS-100. Both sides are language-identified and
  pairs with a wild length ratio are dropped. The instruction is drawn from a bank of hand-written templates in seven languages
  (English, Pidgin, Yoruba, Hausa, Igbo, Twi, Ewe) and placed before or after the text at random.
* **Instruction following**: `nthakur/swim-ir-cross-lingual` (Swahili, Yoruba); `CohereLabs/aya_dataset` (human-written; new-language rows only).
* **Supervised tasks** from human-annotated, commercially usable sets: NCHLT (NER), Universal Dependencies (POS), AfriHate, Injongo and MASSIVE
  (intent), WolBanking77, news topic sets, `bltlab/lr-sum` (summaries).
* **Built from our own text**: language identification (the `<think>` cites cue words that really are distinctive for the language),
  diacritic restoration, text cleaning.
* Translation is capped per language (1,000,000 rows; only Afrikaans was above it). `sft-all` raises the cap to 3,000,000.

## Limits

Machine-written English sides (FineTranslations) are filtered, not verified. Licences differ per source. Not for evaluation: some task
sets are derived from public benchmarks' training splits only, but the sources were not checked against every benchmark's test set.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("counts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    counts = json.loads(a.counts.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    cards = {"BeardedMonster/pretrain-new": pretrain_card(counts["BeardedMonster/pretrain-new"]["counts"]),
             "BeardedMonster/sft-new": sft_card(counts["BeardedMonster/sft-new"]["counts"])}
    for repo, text in cards.items():
        (a.out / (repo.split("/")[1] + ".md")).write_text(text, encoding="utf-8")
        print(repo, len(text), "chars")
    if a.push:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from africomet_translation.build import hf_token
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token())
        for repo, text in cards.items():
            api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type="dataset", commit_message="dataset card")
            print("pushed README to", repo)


if __name__ == "__main__":
    main()
