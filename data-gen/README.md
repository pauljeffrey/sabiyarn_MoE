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

## Why it's built this way

The generation model (GPT-4o via Structured Outputs) never writes the
final `<|system|>...<tool_call>...` text directly — it only returns typed
JSON (validated by the API itself via a strict JSON schema). Python then
deterministically assembles that JSON into `schemas.messages.Conversation`
objects and renders them through **the tokenizer's actual
`chat_template.jinja`** (copied verbatim into `templates/chat_template.jinja`,
loaded with the same `trim_blocks=True, lstrip_blocks=True` Jinja2 settings
`transformers` itself uses). This means:

- The model can never get the special-token syntax wrong — it only has to
  produce valid JSON.
- If the tokenizer's chat template ever changes, re-copy the new
  `chat_template.jinja` from the tokenizer repo and every past batch of
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

## Layout

```
config/            language roster + global settings (volume, model, weights)
schemas/           Message/Conversation models, tool defs, strict-schema helpers
templates/          chat_template.jinja (verbatim copy from the tokenizer)
rendering/          deterministic Conversation -> training text
documents/          chunker + sample docs + your own corpus/
generators/         one module per task, each building BatchRequestSpec lists
pipeline/           build_batch / submit_batch / fetch_results / postprocess
quality/            validators + dedup
stats/              run summary reporting
scripts/            mock_generate.py (free pipeline test, no API calls)
run.py              convenience dispatcher for all of the above
```
