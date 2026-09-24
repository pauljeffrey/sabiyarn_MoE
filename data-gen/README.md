# data-gen

Synthetic SFT data generation for the SabiYarn 280M MoE model, via the
OpenAI Batch API. This folder is fully independent of the rest of the
repo (no imports from `training/`, `sabiyarn/`, etc.) — it only *reads*
the tokenizer's chat template as reference data (see below).

Six task types, each balanced equally across 13 languages: English,
Yoruba, Hausa, Igbo, Efik, Urhobo, Twi, Fon, Nigerian Pidgin, Ewe, Akan,
Fulah, Fulfulde.

- **RAG** — multi-turn conversation grounded in a document, via a
  `search_document` tool call that names a chunk id (the real chunk text
  is injected deterministically at postprocess time, never written by the
  generation model).
- **Summarization** — either "summarize this document" (doc given in the
  system prompt) or "summarize our conversation so far" (a fabricated
  chat history).
- **Edge-device action triggering** — available tools, user instruction,
  short reasoning + task plan, a tool call, a tool result, final response.
- **Structured output** — extract information from an invented source
  text into a specific JSON schema (10 domains: contact card, event
  invite, invoice line items, school result slip, etc.).
- **Math/stats** — everyday word problems, sometimes solved directly,
  sometimes via a `calculate` tool call (teaching *when* to reach for a
  tool, not just how).
- **Translation** — translate between language pairs, phrased as a
  natural request rather than "translate X to Y".

Also available: three larger, diversity-sampled dataset kinds (pretraining
documents, Alpaca-style SFT, DPO preference pairs) for the 12 non-English
languages with gpt-4o-mini — see [Corpus kinds](#corpus-kinds-pretraining-sft-alpaca-and-dpo-data--12-languages-gpt-4o-mini).

## Why it's built this way

The generation model (GPT-4o via Structured Outputs) never writes the
final `<|system|>...<tool_call>...` text directly — it only returns typed
JSON (validated by the API itself via a strict JSON schema). Python then
deterministically assembles that JSON into `schemas.messages.Conversation`
objects and renders them through **the tokenizer's actual
`chat_template.jinja`** (`templates/chat_template.jinja` — a byte-identical copy of the repo's canonical `sabiyarn/chat_template.jinja`, kept in step by `tests/test_chat_template.py` at the repo root; edit one, copy it over the other, and push it to the tokenizer repo;
loaded with the same `trim_blocks=True, lstrip_blocks=True` Jinja2 settings
`transformers` itself uses). This means:

- The model can never get the special-token syntax wrong — it only has to
  produce valid JSON.
- If the chat template ever changes, update it as described above and every past batch of
  generated JSON re-renders correctly with zero regeneration.
- RAG "retrieval" is always grounded in your real document text, never a
  paraphrase invented by the generation model.

## ⚠️ Two things worth your attention

1. **Tokenizer/template mismatch.** `chat_template.jinja` emits literal
   `<tool_result>...</tool_result>` text for tool messages, but the
   tokenizer's `special_tokens_map.json` registers `<tool_call>`,
   `</tool_call>`, `<tool_response>`, `</tool_response>` — **not**
   `<tool_result>`/`</tool_result>`. So today, tool-result text gets split
   into ordinary subwords instead of using a dedicated learned token. This
   is a mismatch in the tokenizer/template config in the parent repo, out
   of scope for this folder to fix — worth deciding whether to add
   `<tool_result>`/`</tool_result>` as special tokens (and re-train/extend
   the embedding table) or change the template to emit `<tool_response>`
   instead. Either way, `templates/chat_template.jinja` here should be
   re-synced afterward.
2. **Edge-action reasoning/plan convention.** The tokenizer already has
   `<think>`/`</think>`, `<reason>`, `<task_plan>`/`</task_plan>`
   registered as special tokens, which is a strong signal for how
   "short reasoning, task plan" should be encoded — so `generators/edge_action.py`
   emits `<reason>...</reason><task_plan>...</task_plan>` as one assistant
   text turn, followed by a separate assistant tool-call turn (the chat
   template doesn't allow content + tool_calls in the same message). If
   you intended a different convention, `render_edge_action` in
   `pipeline/postprocess.py` is the one place to change it.

## Setup

```bash
cd data-gen
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

## Running it

Every step below can also be run through `python run.py <subcommand>`.

### 1. Build batch input files (free, local only)

```bash
python pipeline/build_batch.py                      # all 6 tasks
python pipeline/build_batch.py --tasks rag,translation
```

Writes `data/batch_input/<task>.jsonl` (the actual OpenAI Batch API
payload) and `data/batch_input/<task>.manifest.jsonl` (side-car context
needed later — e.g. which document a RAG example drew from — that must
NOT be sent to the API). **Inspect these before spending money.**

Tune volume via env vars before this step:

```bash
DATA_GEN_PER_CELL=40 python pipeline/build_batch.py   # examples per (task, language) cell; default 40
DATA_GEN_MODEL=gpt-4o-2024-08-06 python pipeline/build_batch.py
```

At the default of 40/cell × 6 tasks × 13 languages that's **3,120
requests** total (~520 per task). Scale `DATA_GEN_PER_CELL` up or down
freely — cost and volume both scale linearly with it.

### 2. Dry-run for free before spending anything

```bash
python scripts/mock_generate.py     # fabricates structurally-valid (linguistically fake) responses
python pipeline/postprocess.py      # runs the full validate/render/dedup pipeline against them
```

This exercises every code path (parsing, tool-arg validation, dedup,
chat-template rendering) without calling the API, so you can confirm the
pipeline mechanics work before paying for real generation. Mock output is
never linguistically real — don't mistake `data/processed/*.jsonl` from
this step for actual training data. Re-run `pipeline/build_batch.py`
afterward to get a clean `data/batch_input/` before submitting for real.

### 3. Submit to the Batch API (this spends money)

```bash
python pipeline/submit_batch.py                        # dry run: prints file/request counts only
python pipeline/submit_batch.py --tasks rag --confirm   # actually uploads + creates the batch job
```

Batch API pricing is typically ~50% of standard sync pricing; jobs
complete within the 24h completion window, not immediately.

### 4. Fetch results

```bash
python pipeline/fetch_results.py               # polls status, downloads completed batches
python pipeline/fetch_results.py --check-only   # just prints status
```

### 5. Postprocess into final SFT data

```bash
python pipeline/postprocess.py
```

Writes `data/processed/<task>.jsonl` and `data/processed/all.jsonl`, each
record shaped as:

```json
{"id": "...", "task": "...", "language": "yor", "messages": [...template-shaped dicts...], "text": "<s>...rendered training text...", "meta": {...}}
```

`messages` is there so you can re-render later (e.g. after a template
fix) without regenerating; `text` is the ready-to-tokenize string.

Also writes `data/reports/summary.json` — per-(task, language) counts of
generated / parse-failed / validation-failed / dedup-dropped / kept, plus
every quality warning raised (missing-diacritic language mismatches,
near-duplicate drops, etc.) for spot-checking. **Low-resource languages
(Efik, Urhobo, Fon, Ewe, Fulah, Fulfulde — see `resource_tier` in
`config/languages.py`) deserve extra manual spot-checking**; GPT-4o's
fluency there is materially weaker than for the higher-resource
languages, and the automated checks here are heuristics, not a substitute
for a native speaker review pass.

## Adding things

- **A language**: append one `Language(...)` entry to `config/languages.py`.
  Everything else (generators, distribution balancing) picks it up
  automatically.
- **Your own documents** (for RAG/summarization): drop `.txt`/`.md` files
  into `documents/corpus/`. Two sample documents already live in
  `documents/sample/` for testing.
- **A structured-output domain**: append a `StructuredSchemaSpec` to
  `schemas/structured_output_schemas.py`.
- **An edge-device tool**: append a tool def (standard OpenAI
  function-calling shape) to `EDGE_DEVICE_TOOLS` in `schemas/tools.py`.
- **A new task type**: add a `generators/<task>.py` with a
  `build_requests() -> list[BatchRequestSpec]`, a `render_<task>` function
  in `pipeline/postprocess.py`, and register both in
  `pipeline/build_batch.py::GENERATOR_MODULES` and
  `pipeline/postprocess.py::RENDERERS`.

## Diversity/quality mechanisms already in place

- Every example gets an independently-sampled persona (`generators/personas.py`:
  age/occupation, tone, writing quirk) injected into the meta-prompt, so
  conversations don't all sound like the same person.
- Per-example generation is seeded deterministically
  (`generators/base.py::rng_for`, keyed on task/language/index/`DATA_GEN_SEED`)
  so a single missing/failed example can be regenerated later with the
  same sampling choices, without needing to re-run the whole cell.
- `HARD_EXAMPLE_FRACTION` (default 20%, in `config/settings.py`) injects
  adversarial variants — unanswerable RAG questions, edge-device requests
  no available tool can satisfy — so the model learns to say "I don't
  know" / ask for clarification instead of pattern-matching a tool call
  every time.
- `quality/dedup.py` does exact + near-duplicate (5-word shingle Jaccard)
  filtering within each (task, language) cell after generation.
- `quality/validators.py` structurally validates model-invented tool
  arguments against the real tool schema, and flags (soft warning, not a
  hard drop) suspiciously low usage of a language's distinctive
  diacritics as a possible language-mismatch signal for spot-checking.

## Corpus kinds: pretraining, SFT (Alpaca) and DPO data — 12 languages, gpt-4o-mini

Three additional dataset kinds, generated with the OpenAI Batch API and
Structured Outputs, for the **12 non-English languages only** (`yor hau ibo
efi urh twi fon pcm ewe aka ful fuv`; English is never a target language):

| kind | record | preset per language | preset total |
|---|---|---|---|
| `pretrain` | plain-text document: `title` + `text` | 4,000 | 48,000 |
| `sft` | Alpaca: `instruction`, `input` (may be empty), `response` | 6,000 | 72,000 |
| `dpo` | `instruction`, `input`, `chosen`, `rejected` (+ flaw metadata) | 1,500 | 18,000 |

Default model is **gpt-4o-mini** (`model:` in each yaml, or `--model`). The
emphasis is quality and diversity: a real coverage sampler (below) decides
*what* every request is about; the model only writes the text.

### Commands

```bash
cd data-gen

# 0. Free: volume, tokens, dollars (no API calls). --per-language / --languages to try a pilot size.
python run.py estimate                      # all three kinds, preset counts
python run.py estimate --kind sft --per-language 200

# 1. Free: write batch input + manifest (split into __partN files above 50,000 requests / ~190 MB)
python run.py build --kind pretrain --config configs/pretrain.yaml
python run.py build --kind sft      --config configs/sft.yaml
python run.py build --kind dpo      --config configs/dpo.yaml
python run.py build --kind sft --per-language 20 --languages yor,efi   # tiny pilot; --dry-run = estimate only

# 2. Free: offline dress rehearsal (fake but structurally valid outputs; never real data)
python run.py mock
python run.py postprocess --kind all

# 3. COSTS MONEY (dry run unless --confirm). Pilot first: build with --per-language 20 and read the outputs.
python run.py submit --kind sft                # lists files/requests only
python run.py submit --kind sft --confirm      # uploads + creates Batch jobs (needs OPENAI_API_KEY)
python run.py fetch                            # polls; downloads completed batches to data/batch_output/

# 4. Free: parse, validate, filter, dedup, write data/processed/{pretrain,sft,dpo}.jsonl + report
python run.py postprocess --kind sft           # or --kind all

# 5. OPTIONAL LLM judge (costs money to submit): score a sample (or all), then filter by score
python run.py judge build --kind sft --fraction 0.2
python run.py submit --tasks judge_sft --confirm
python run.py fetch
python run.py judge apply --kind sft --min language_correctness=4 --min fluency=3
```

Every script also runs directly (`python pipeline/build_corpus.py ...`,
`pipeline/postprocess_corpus.py`, `pipeline/estimate.py`, `pipeline/judge.py`)
and takes `--data-dir` to write somewhere other than `data/`. `--kind` on
`build`/`postprocess` selects this corpus pipeline; the original six-task
scripts are untouched. (Two small fixes to the shared scripts were needed for
split files: `submit_batch.py` no longer mistakes `<kind>__partN.manifest.jsonl`
for a batch input, and `fetch_results.py` names outputs after the source file
so `sft__part0` / `sft__part1` do not overwrite each other. Behaviour for the
original six tasks is identical.)

**Batch queue limits.** OpenAI limits enqueued tokens per model by usage tier,
and the preset volumes (54M / 81M / 23M input tokens) exceed what low tiers may
queue at once. Use `--max-requests-per-file 5000` (or whatever fits your tier)
and submit/fetch files one after another. Verify current limits in your account.

### How the diversity sampler works (`sampling/`)

LLMs asked for "a diverse example" collapse onto a handful of familiar topics,
and independent random draws are lumpy (some sub-topics get 0 uses, others 15).
So diversity is imposed from outside by a deterministic `CoverageSampler`
(one instance per *(kind, language)*, seeded from `(seed, kind, language)`
with the same scheme for every language, so every language gets identical
coverage structure):

* **Taxonomy** (`sampling/taxonomy.py`, hand-written): 59 domains
  (agriculture, health, public health, law, finance & mobile money, energy,
  Nollywood, proverbs, chieftaincy, ...) with 10-12 concrete sub-topics each,
  **636 (domain, sub-topic) pairs**; 28 text genres, 7 registers, 14
  audiences, 3 length buckets, 5 perspectives, 4 eras, 3 reading levels; for
  SFT/DPO, 33 task types and 12 rejection (flaw) types.
* **Primary key = (domain, sub-topic)**: cycle through a *shuffled list of all
  pairs* before repeating any, reshuffle each cycle. After N draws every pair
  was used `floor(N/636)` or `ceil(N/636)` times. With 4,000 documents per
  language, each pair appears 6-7 times.
* **Secondary attributes** (genre or task, register, audience, length bucket,
  perspective, era, difficulty, instruction style, rejection type, locale):
  **least-used-first** against target weights — pick the compatible value
  minimising `(count+1)/weight`, ties broken by the seeded RNG. Counts track
  the targets to within a few draws; a value starved by compatibility rules
  accumulates a deficit and wins the next time it is legal.
* **Compatibility rules** are data (domain tags vs. genre/audience/era/task
  requirements): no product description of a historical empire, no "news
  report" on mathematics, sermons only for faith topics, `poor_translation`
  only for translation tasks, `wrong_label` only for classification, etc.
* **Locale** (`sampling/locales.py`): each language has 13-20 real places and
  a country (currency, everyday realities) plus local given names for
  fictional characters — Yoruba: Lagos, Ibadan, Ogun, Osun, Benin Republic;
  Hausa: Kano, Kaduna, Sokoto, Niger Republic; Twi/Akan: Kumasi, Accra, Ashanti,
  Akuapem; Ewe: Volta Region, Togo; Fon: Cotonou, Abomey, Ouidah; Efik: Calabar,
  Cross River; Urhobo: Warri, Effurun, Delta State; Fulfulde: Adamawa, Sokoto,
  Cameroon; Fulah: Senegal, Guinea, Mali; Pidgin: Nigeria broadly.
* **Auditable**: the sampled tuple is stored in every manifest line
  (`context.attributes`); `build` prints a requested-coverage audit table and
  `postprocess` reports coverage of the *kept* records per language (dropping
  is not uniform, low-resource languages lose more). `sampling.sampler.coverage_report`
  gives counts per value, normalised entropy and max/min ratio.

### Editing the yaml presets (`configs/{pretrain,sft,dpo}.yaml`)

* `samples_per_language`: `default` plus all 12 languages listed explicitly
  (equal by default). Change any number; the loader rejects `eng`/unknown codes.
* `attribute_weights`: target shares per genre/task, register, audience, length
  bucket, ... `0` disables a value; unknown names are rejected. Defaults live
  in the taxonomy; the yaml shows them so you can edit in place.
* `domain_group_weights`: 1.0 = every pair equally often, 2.0 = that group's
  pairs twice per cycle, 0 = exclude the group.
* `quality`, `dedup`, `judge`: thresholds (see below). Only listed keys change.
* Volume presets and their estimated Batch cost are documented next to the
  numbers in each file. Prices live in **one** place, `config/settings.py`
  (`BATCH_PRICING_USD_PER_M_TOKENS`, marked *verify current pricing*).

### Output formats (`data/processed/`)

```jsonc
// pretrain.jsonl
{"id": "pretrain__yor__00042__base", "language": "yor", "text": "...", "title": "...", "domain": "agriculture_crops",
 "subtopic": "yam mound farming and storing yams", "genre": "how_to_guide", "register": "semi_formal", "audience": "farmers_rural",
 "locale": "Abeokuta, Ogun State", "length_bucket": "medium", "perspective": "second_person", "era": "contemporary",
 "difficulty": "basic", "n_words": 311}

// sft.jsonl  (messages/text rendered through the real chat template; user = instruction [+ "\n\n" + input])
{"id": "sft__hau__00007__base", "language": "hau", "task": "summarization", "domain": "...", "subtopic": "...",
 "instruction": "...", "input": "...", "response": "...", "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
 "text": "<s><|system|>...", "confidence": "high", "register": "...", "instruction_style": "...", "response_length": "...", "difficulty": "...", "locale": "..."}

// dpo.jsonl
{"id": "dpo__ibo__00003__base", "language": "ibo", "task": "...", "domain": "...", "instruction": "...", "input": "",
 "prompt_messages": [{"role": "system", ...}, {"role": "user", ...}], "chosen": "...", "rejected": "...",
 "rejection_type": "ignores_constraint", "chosen_confidence": "high", ...}
```

`processed/<kind>.judged.jsonl` (judge stage) adds `judge_scores`.
`data/reports/<kind>_summary.json` has, per language: generated/kept, drop
reasons (first reason per dropped record plus all reason hits), soft-warning
counts, per-domain and per-task/genre counts, and coverage entropy / max-min
ratio of the kept records, plus a few example ids per drop reason to inspect.

### Quality filters (all local, no API)

Applied in `quality/corpus_filters.py`; each drop is counted by reason.

* **Parse/schema**: API error, refusal object, invalid JSON, schema mismatch, missing output.
* **Length**: min/max words (pretrain, relative to the requested bucket) or chars (SFT/DPO fields).
* **Repetition**: repeated word 4-gram ratio, repeated line/sentence ratio, single-token domination.
* **English leakage**: share of English function words, with a higher tolerance for `pcm` (which shares vocabulary; a stricter word list is used) and none for the English side of translation tasks.
* **Meta text / placeholders / refusals**: "Here is...", "As an AI...", `[...]`, `[name]`, `{{x}}`, code fences, English refusals (allowed only for the `safe_decline` task and for `unhelpful_refusal` rejected answers); markdown headings in pretraining text.
* **Script sanity**: mostly non-Latin letters or U+FFFD = drop; missing distinctive diacritics/letters = *soft warning* counted in the report (`no_distinctive_chars:<field>`), never a drop.
* **Confidence**: `confidence == low` / `chosen_confidence == low` and pretrain `language_self_check == false` dropped.
* **DPO**: chosen != rejected (after normalisation), both non-empty, `rejection_type` echo matches the sampled one, rejected in the same language (except `wrong_language`), length ratio bounds (skipped for length-related flaws), refusal only tolerated where it is the flaw.
* **Dedup** (`quality/dedup.py::find_near_duplicates`, sketch + inverted index, sub-quadratic): exact + near-duplicate (word-shingle Jaccard) within each language, then across the 12 languages of the same kind; earliest index wins.

### Cost estimate (Batch API, gpt-4o-mini)

From `python run.py estimate` (prompt sizes measured on real requests; output sizes from the sampled length
buckets). Tokens per word were **measured with tiktoken `o200k_base`** (the gpt-4o-mini tokenizer) on the real
sentences in `data/curated_eval.jsonl`: eng 1.11, hau 1.61, ibo 1.83, pcm 1.13, yor 2.31. The estimator uses those
values rounded up (hau 1.8, ibo 2.0, yor 2.5, pcm 1.35); the languages with no measured sample (twi/aka 2.4, ful/fuv
2.4, efi/urh 2.7, ewe 2.9, fon 3.0) are extrapolated and deliberately higher. Real spend is expected at or a little
below the estimate.

| kind | requests | input tokens | output tokens | est. USD |
|---|---|---|---|---|
| pretrain (4,000 x 12) | 48,000 | ~54.0M | ~40.2M | ~$16.1 |
| sft (6,000 x 12) | 72,000 | ~80.5M | ~30.3M | ~$15.1 |
| dpo (1,500 x 12) | 18,000 | ~23.1M | ~11.3M | ~$5.1 |
| **all** | **138,000** | | | **~$36.4** |

Range: ≈ $32 if the unmeasured languages tokenize like the measured ones, ≈ $43 with the old, more pessimistic
multipliers (3.0-3.4 tokens/word). Input tokens dominate the uncertainty for SFT/DPO: they are measured, not
estimated. Add ~5-10% for requests that fail validation but are still billed. Pricing: $0.075 / $0.30 per 1M
input/output tokens (Batch; standard is $0.15 / $0.60), verified against OpenAI's pricing page on 2026-09-21 —
re-check before a big run and edit `config/settings.py`. Optional judging adds roughly one short request per judged
record.

### Limitations (please read)

* **gpt-4o-mini is materially weaker than gpt-4o in low-resource languages.**
  Expect noticeably worse fluency, orthography, invented words and factual
  slips in **Efik, Urhobo, Fon, Ewe and Fulah/Fulfulde** (and Nigerian
  Fulfulde), and mediocre-but-usable output in Yoruba/Igbo/Hausa/Twi/Akan.
  Local filters catch gross failures only; they cannot judge grammar. If budget
  allows, generate the low-resource languages with a stronger model
  (`--model gpt-4o-2024-08-06`, edit `model:`; the price table already has it).
* **A native-speaker spot check of a sample per language is essential** before
  training — e.g. 50-100 records per language and kind, focusing on the
  low-resource ones, using the per-language reports and judge pass rates to
  decide how much to trust each language. Drop or down-weight languages that fail.
* **The LLM judge is a triage signal, not ground truth**: the same weak model
  scoring text it cannot write well is lenient on exactly the languages that
  need scrutiny. Use a stronger judge model for a sample and compare pass rates.
* **Synthetic pretraining text should be a small, filtered supplement mixed
  with real corpora, not a replacement.** Model-written text is more uniform
  and cleaner than real text, can carry the generator's subtle errors, and
  training on it at scale risks reinforcing them (model-collapse-style
  effects). Mix it with real web/book/news data in these languages and keep its
  share modest.
* **Synthetic DPO with injected flaws is off-policy**: the rejected answers are
  what a strong model *imagines* a weak one says, not what your SFT model
  actually generates, so gains can be narrow (language fidelity, constraint
  following) and may not transfer. For on-policy preference data later, sample
  several answers from your SFT model per prompt and rank them with a judge
  (or, for translation, a metric such as AfriCOMET), keeping confident pairs.
* Factual content is only as good as the model's knowledge; prompts push
  hedged, general statements and forbid invented statistics/quotes, but some
  errors will remain. The `safe_decline` and proverb tasks are especially
  worth reviewing by native speakers.

### Tests

```bash
data-gen/.venv/bin/python -m pytest data-gen/tests -q      # no network, no API key needed
```

Covers taxonomy integrity, sampler balance/determinism/compatibility, yaml
presets, request building (strict schemas, unique ids, English never a
target), the quality filters on hand-made bad records, offline build -> mock
-> postprocess -> stats loops for each kind (plus file splitting, judge
build/apply, CLI wiring and the original six-task smoke flow), and the cost
estimator.

## Layout

```
config/            language roster + global settings (volume, model, pricing), corpus_config.py (yaml loader)
configs/           pretrain.yaml / sft.yaml / dpo.yaml presets (counts, weights, quality thresholds)
sampling/          taxonomy.py (domains, genres, ...), locales.py, sampler.py (CoverageSampler)
schemas/           Message/Conversation models, tool defs, strict-schema helpers
templates/          chat_template.jinja (copy of sabiyarn/chat_template.jinja; drift-tested)
rendering/          deterministic Conversation -> training text
documents/          chunker + sample docs + your own corpus/
generators/         one module per task, each building BatchRequestSpec lists; pretrain.py, sft_tasks.py, dpo.py for the corpus kinds
pipeline/           build_batch / submit_batch / fetch_results / postprocess; build_corpus / estimate / postprocess_corpus / judge for the corpus kinds
quality/            validators + dedup + corpus_filters
stats/              run summary reporting (report.py, corpus_report.py)
scripts/            mock_generate.py (free pipeline test, no API calls)
tests/              pytest suite (offline)
run.py              convenience dispatcher for all of the above
```

---

# Seed-driven generation (Together AI / OpenRouter / Modal / RunPod / vast)

The pipeline above targets the OpenAI Batch API. This newer one is seed-driven, multi-provider and
multi-platform, and pushes straight to `BeardedMonster/data-gen` on the Hub.

## The idea

The target model is ~306M parameters. It cannot hold the world's facts. So the corpus teaches **general
understanding plus two reflexes**: reason inside `<think>...</think>`, then either call a tool or say
plainly that it does not know. Asked "what is AWS?", the right behaviour is not a memorised definition --
it is to notice it has not heard of it, call `search_internet`, and answer from what comes back. With no
such tool, the right answer is "I don't know, and I can't look it up." ~24% of the SFT task mix is exactly
this: looking things up, admitting it cannot, and noticing that a retrieval did not answer the question.

## Files

| | |
|---|---|
| `schemas/seed.py` | the seed schema: tasks, tools, tags, volumes, validation |
| `seeds/build_seeds.py` | **edit this**, then re-run it; it writes the JSON |
| `seeds/{pretrain,sft,rl}.json` | generated -- the brief every provider and platform reads |
| `prompts.py` | seed + plan row -> meta-prompt (deterministic) |
| `generate.py` | the driver: plan, call, validate, shard, push |
| `postprocess_gen.py` | response -> validated record (both `messages` and rendered `text`) |
| `providers/{together,openrouter}.py` | the two providers |
| `hub.py` | incremental push to the Hub |
| `runners/{modal_gen,runpod_gen}.py` | Modal / RunPod / vast |

## Run it

```bash
python seeds/build_seeds.py --print                        # see the plan, write nothing
python generate.py --kind sft --provider openrouter --limit 5 --dry-run    # see the prompts, spend nothing
python -m providers.openrouter --check                     # 1 live request, confirms the key works

python generate.py --kind sft --provider openrouter --limit 500 --push
python generate.py --kind pretrain --provider together --batch             # ~50% price, 24h window
python generate.py --kind pretrain --provider together --fetch <batch_id> --push

modal run runners/modal_gen.py --kind sft --limit 20000 --shards 8
python runners/runpod_gen.py --kind sft --limit 20000 --shards 4           # on a RunPod/vast box
python hub.py --push-seeds && python hub.py --status
```

## Why it is safe to interrupt

Every planned sample has a deterministic `custom_id` derived from (kind, language, task, index). Completed
ids are read back from the shards on disk and skipped, so re-running after a crash, a rate-limit wall or a
spot eviction costs nothing. `--shards N` gives worker *i* every *N*th row of the same fixed plan, so Modal,
RunPod and a vast box can all generate into one corpus without overlapping or coordinating.

Coverage is by construction, not by sampling: the row index walks the (domain, sub-topic) list with a
per-language stride, so every pair is used once before any repeats -- with no shared sampler state.

## Volumes

| kind | total | notes |
|---|---|---|
| pretrain | 435,000 docs | pcm 60k; yor/hau/ibo 40k; urh/efi + 6 others 30k; eng 15k |
| sft | 79,000 conversations | 6-10 messages, 2-4 tasks each, ends on the assistant |
| rl | 19,400 prompts | x3 ranked candidate replies |

## Tokenizer: two fixed, one outstanding

FIXED and pushed to both `BeardedMonster/SabiYarn-32k` and `Aletheia-ng/SabiYarn_MoE-280M`:

1. The chat template now emits `<tool_response>`/`</tool_response>` (single tokens 52037/52038) instead of
   `<tool_result>`, which was not a special token and cost ~5 byte-BPE tokens per tag.
2. Token 52043 was `|analyze|>`, missing its leading `<`. Renamed in place to `<|analyze|>` -- same id, same
   vocab size, so no embedding resize and no retraining needed.

STILL OUTSTANDING (does not block generation): tokenizer ids **52050-52115** are at or above the model's
`vocab_size` (52050), so those 66 tokens can never be embedded. That range holds `<|hate|>` and ~65
other-language tags. Nothing in this pipeline uses them -- the toxicity task uses `<toxic>` (52008), which is
in range -- but resize the embedding before you rely on any of them.

## Judging RL candidates

`judge_gen.py` is a second batch pass that reduces the 3 generated candidates to `response_1` (chosen) /
`response_2` (rejected). Candidates are relabelled A/B/C in a deterministic shuffle with the generator's own
quality labels withheld, judged against ranked criteria (honesty at the knowledge boundary first, style
last), and the verdict records whether the judge agreed with the generator. A near-100% agreement rate means
the judge is rubber-stamping; near 33% means the generator's own labels are noise. Use a **different model**
for the judge than for generation.

## Cost (Together AI: $0.15/$0.60 per 1M standard, $0.075/$0.300 batch)

`python estimate_gen.py` measures the real prompt sizes rather than guessing. At the mid output estimate:

| phase | requests | input tokens | batch | standard |
|---|---|---|---|---|
| pretrain | 543k | 592M | $240 | $480 |
| sft | 225k | 592M | $180 | $359 |
| rl | 79k | 248M | $76 | $151 |
| judge | 65k | 143M | $16 | $31 |
| **total** | | | **~$511** | **~$1,022** |

Request counts are yield-adjusted, so failed generations are already priced in.

## Self-hosting with vLLM (usually cheaper than the API here)

`vllm_gen.py` runs the same seeds, the same `prompts.build_request`, the same validation and the same shard
files on your own GPU instead of a per-token API.

```bash
python vllm_gen.py --kind sft --plan-only --gpu-cost 2.0        # cost model + break-even; no GPU needed
pip install vllm                                               # on the GPU box
python vllm_gen.py --kind sft --model openai/gpt-oss-120b --limit 200   # measure real throughput first
python vllm_gen.py --kind pretrain --model google/gemma-3-27b-it --tp 2 --push
python vllm_gen.py --kind judge --model google/gemma-3-27b-it --push    # judge with a DIFFERENT model
```

**Why it wins on this specific workload.** 45-78% of every prompt is a byte-identical prefix -- the seed
brief, the special-token rules, the tool catalogue -- shared by every row of the same (kind, language):

| kind | shared prefix | of input tokens |
|---|---|---|
| pretrain | 78% | 1,076/request |
| sft | 57% | 2,517/request |
| rl | 45% | 2,608/request |

An API bills that prefix on every request. vLLM computes it once and reuses the KV cache, so the script turns
on `enable_prefix_caching` **and sorts the work by (language, task)** so identical prefixes arrive
consecutively rather than being evicted between rows. The system prompt is deliberately a pure function of
(kind, language) -- anything row-specific lives in the user message -- and a test enforces that, because one
varying character in the prefix costs a full recompute per row.

The second win is `--guided` (on by default): the output JSON schema (`schemas/output.py`) is compiled into a
decoding constraint, so malformed JSON becomes structurally impossible. `json_invalid` is the biggest drop
reason on the API path; here it is zero, which raises effective yield.

**Break-even.** At $2/hr for the box, self-hosting beats Together's batch price above roughly:

| kind | break-even (output tok/s) |
|---|---|
| pretrain | ~1,600/s |
| sft | ~1,400/s |
| rl | ~1,400/s |

Below that, the API is cheaper. Break-even rises with output length, because the API's per-request input cost
amortises over more output -- so self-hosting wins most clearly on short outputs and prompt-heavy kinds. The
script prints measured tok/s and $/1M as it runs, so decide from a `--limit 200` pilot rather than from this
table.

Both models fit one 80GB card: `gpt-oss-120b` is MoE with ~5B active params in MXFP4 (~60GB, needs vLLM
>=0.10), and `gemma-3-27b-it` is ~54GB in bf16. Use `--tp N` for multiple GPUs and `DATA_GEN_SHARDS` to split
one plan across several boxes.

## vLLM vs llama.cpp for this job

**vLLM, clearly** -- for this workload, not in general.

| | vLLM | llama.cpp |
|---|---|---|
| batching 543k independent prompts | continuous batching: hundreds of sequences in flight, scheduler refills slots as they finish | parallel slots (`-np`), far lower throughput at high concurrency |
| the 45-78% shared prefix | automatic prefix caching across concurrent requests (a radix tree of KV blocks) -- the entire cost argument here | prompt cache is single-sequence oriented, not shared across a live batch |
| constrained JSON | xgrammar / outlines | GBNF grammars (also good) |
| gpt-oss-120b MXFP4 | native | needs a GGUF conversion first |
| memory | PagedAttention -> bigger batches on the same card | simpler allocator, smaller batches |

Break-even against the API is ~1,400-1,600 output tok/s. llama.cpp at high batch sizes is typically several
times slower than vLLM on the same GPU, and being slow here does not just cost time -- it flips the economics
back toward just paying Together.

llama.cpp is the better tool when you are on CPU, on a Mac, or on a GPU too small to hold the model without
aggressive 4-bit quantization. None of those apply to a rented 80GB card doing bulk offline generation.

**Yes, vLLM batches.** `engine.chat(list_of_conversations, params)` takes the whole list and vLLM's scheduler
decides the running batch size dynamically from free KV-cache blocks. `--chunk` is NOT the batch size -- it
only controls how often shards flush and progress prints; leave it large.

## Running on vast.ai

```bash
export HF_TOKEN=...  HF_WRITE_TOKEN=...
bash runners/vast_vllm.sh pretrain
MODEL=google/gemma-3-27b-it TP=2 SHARDS=2 bash runners/vast_vllm.sh pretrain
```

Instance sizing (verify prices, they move hourly):

| setup | GPU | rough $/hr |
|---|---|---|
| gemma-3-27b fp8 | 1 x 48GB (A6000/L40S) | $0.40-0.70 |
| gemma-3-27b bf16 / gpt-oss-120b MXFP4 | 1 x 80GB (A100/H100) | $0.80-2.00 |
| either, tensor-parallel | 2 x 24GB (2x RTX 4090, needs fp8 for 27B) | $0.50-0.90 |

Ask for **>= 150GB disk** (the 120b weights are ~60GB and HF caches a copy) and a CUDA 12.4+ image. The script
sets `HF_HOME` to the big volume, pushes shards to the Hub as they are written, and logs to
`/workspace/<kind>.log` because vast's web terminal loses scrollback. Being outbid costs you only the shard in
flight -- re-running skips everything already on the Hub.
