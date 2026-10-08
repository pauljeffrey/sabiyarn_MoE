# Tool-call format (chat template v2) — contract for data, rewards and serving

Source of truth: `sabiyarn/chat_template.jinja` (byte-identical copy in `data-gen/templates/`). Everything below is what that template renders; if this note and the
template disagree, the template wins. Status: implemented in the repo and **pushed to both tokenizer repos** (`BeardedMonster/SabiYarn-32k` commit e2cfaf1, `Aletheia-ng/SabiYarn_MoE-280M` commit 3030760; both carry identical `tokenizer.json`, `tokenizer_config.json` (with the `chat_template` key), `special_tokens_map.json`, `chat_template.jinja`; `generation_config.json` has `eos_token_id: [2, 52030]`). See §7 for the tool vocabulary (`<tools>`, `</tools>` and 16 quoted JSON keys).

## 1. What a tool-using turn looks like

Messages (OpenAI-shaped, what `messages` columns hold):
```json
[{"role": "system",    "content": "You are a helpful assistant. When you need facts you do not have, call one of the functions listed below."},
 {"role": "user",      "content": "Tani Biyi Bandele?"},
 {"role": "assistant", "content": "<|input_lang|><yor><think>Mo nilo lati wa alaye.</think><task_plan><|RAG|></task_plan>",
                       "tool_calls": [{"function": {"name": "search_wikipedia", "arguments": {"query": "Biyi Bandele", "language": "yo"}}}]},
 {"role": "tool",      "content": "Biyi Bandele jẹ onkọwe ọmọ Naijiria."},
 {"role": "assistant", "content": "<|input_lang|><yor><task_plan><|RAG|><|explain|></task_plan><|target_lang|><yor><response>Biyi Bandele jẹ onkọwe ọmọ Naijiria.</response>"}]
```
with `tools=[{"type": "function", "function": {"name": "search_wikipedia", "description": "...", "parameters": {JSON schema}}}]`.

Rendered text (one line in reality; line breaks added here):
```
<s><|system|>You are a helpful assistant. ...

<tools>
{"type": "function", "function": {"name": "search_wikipedia", "description": "...", "parameters": {...}}}
</tools></s>
<|user|>Tani Biyi Bandele?</s>
<|assistant|><|input_lang|><yor><think>Mo nilo lati wa alaye.</think><task_plan><|RAG|></task_plan>
<tool_call>{"name": "search_wikipedia", "arguments": {"query": "Biyi Bandele", "language": "yo"}}</tool_call>
<tool_response>Biyi Bandele jẹ onkọwe ọmọ Naijiria.</tool_response>
<|input_lang|><yor><task_plan><|RAG|><|explain|></task_plan><|target_lang|><yor><response>Biyi Bandele jẹ onkọwe ọmọ Naijiria.</response></s>
```

## 2. Rules (what changed from the old template)
| | old | now |
|---|---|---|
| call body | `<tool_call>name\n{args}</tool_call>` | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` — ONE parseable JSON object (Hermes/Qwen shape). `arguments` may also be a JSON string; it is embedded as is |
| end of the call | `</s>` after every call | **no `</s>`**. `</tool_call>` is the last thing the model generates in that step |
| tool result | `<tool_response>NAME text</tool_response></s>` | `<tool_response>text</tool_response>` — **no function name, no `</s>`**; several results sit side by side, in call order |
| turn structure | each assistant message started with `<|assistant|>` and ended with `</s>` | an assistant message that follows a `tool` message continues the same turn: **no new `<|assistant|>`**; the turn has exactly one `</s>`, after the final answer |
| scaffold | missing on call turns (committed template) | content (scaffold: `<|input_lang|>`, optional `<think>`, `<task_plan>`) is written before the call |
| second scaffold | – | after `</tool_response>` the model writes a full scaffold again (`<|input_lang|>…<|target_lang|>…<response>`) before the answer; the scaffold parser/reward should read it from the text after the LAST `</tool_response>` |
| function definitions | free text in the system prompt | `tools` list rendered as `<tools>\n{json}\n{json}\n</tools>` (compact JSON, one function per line) appended to the system turn (a tools-only system turn is created if there is no system message) |
| JSON rendering | jinja `tojson` (sorted keys, `ẹ` escapes, `'`) in our data build | same as `transformers`: `json.dumps(ensure_ascii=False)`, key order preserved. Training text and `apply_chat_template` are now byte-identical |

## 3. Training (SFT)
* Loss covers everything in the assistant turn: scaffold, `<think>`, the call **including `</tool_call>`**, the post-result scaffold, the answer and the final `</s>`.
* Masked: system, user, and every `<tool_response>…</tool_response>` span (tags included) and `<context>…</context>` span. The model is never asked to predict what follows `</tool_call>`.
* Rows carry `messages` (JSON), `tools` (JSON list or ""), `text` (the rendered string that is tokenised), `output_structure`.
* Every tool-calling row must list the function it calls in `tools` (and the system turn must tell the model it may call them). `wiki_yo` and the data-gen conversations do; rows without `tools` are not tool-calling data.

## 4. Serving / rollout contract
1. Prompt = `apply_chat_template(messages, tools=tools, add_generation_prompt=True)`. After a `tool` message the prompt ends in `</tool_response>` (no `<|assistant|>`): the model continues its turn.
2. Generate with stop ids `[</s>, </tool_call>]` (`</tool_call>` = 52030). If the last token is `</tool_call>`: parse the JSON object, run the tool, append `{"role":"assistant","content": text so far, "tool_calls":[…]}` and `{"role":"tool","content": result}`, build the next prompt, continue. If it is `</s>`: the turn is done.
3. Sequential single calls are the supported pattern. The template can render several calls in one message (`<tool_call>…</tool_call><tool_call>…</tool_call>`), but since `</tool_call>` is a stop token a rollout halts after the first, and the model is never trained to emit anything after the last `</tool_call>`. No current data has several calls in one message. If parallel calls are wanted later, add a dedicated end-of-calls token.
4. A candidate that ends at `</tool_call>` is a complete **step**; step-level pairs for DPO/RL share the exact rendered prefix and differ only in the text between the prefix and the stop.

## 5. Changes already made in the repo (for whoever works on rewards/DPO)
* `rl/verifiers.py` reads the JSON call (and still the legacy `name\n{args}`); `think_precedes_tool_call` etc. unchanged.
* `data-gen/postprocess_gen.py` renders candidate tool calls in the JSON shape; `data-gen/seeds/build_seeds.py::ASSISTANT_FORMAT["tool_call"]` updated; `schemas/tools.py::format_tools_for_system_prompt` now returns the `<tools>` block.
* `data-gen/assemble.py::output_structure` still reduces a call to `<tool_call>NAME</tool_call>` (name only) for the *expected structure* column; it reads the name out of the JSON object.
* `data-gen/agents/loop.py` and the `agent/` framework were not touched here: they must (a) stop on `</tool_call>`, (b) keep the whole turn as one assistant text, (c) pass `tools=` to the template, (d) not append function names to tool results.

## 6. Done / still open
* Done: template, tokens and `generation_config.json` (`eos_token_id: [2, 52030]` = `</s>` and `</tool_call>`, so `generate()` stops after a call) are on both repos; the Aletheia tokenizer was brought level with the reference one (it also gained `<POS>` 52116 and the renamed `<zul> <xho> <wol> <sna>`). `apply_chat_template(tools=...)` from the Hub equals the data build's rendering byte for byte.
* The model weights still have 52,050 embedding rows; every id >= 52,050 (new language tags, `<POS>`, `<tools>`, `</tools>`, the 16 JSON keys) only works after the SFT-time growth to 52,160 (`sft_resize_vocab_size`). Do not use those tokens against the current checkpoint.
* `sft.bin` can be tokenised now with the current tokenizer. Continued-pretraining text must NOT be (see §7, "pretraining pin").
* Still open: parallel calls in one message (needs a dedicated end-of-calls token), and the Hub copies of `tokenizer_config.json` carry `transformers_version` metadata from older writes (harmless).

## 7. The tool vocabulary
| tokens | ids | kind | why |
|---|---|---|---|
| `<tools>`, `</tools>` | 52117, 52118 | **special** | delimit the function block in the system turn |
| `"name"` `"arguments"` `"description"` `"parameters"` `"properties"` `"required"` `"type"` `"function"` `"string"` `"object"` `"integer"` `"number"` `"boolean"` `"array"` `"enum"` `"query"` | 52119 … 52134 (in that order) | **ordinary added tokens, quotes included** | these keys were 37% of the tokens of a typical `<tools>` block (`"arguments"` = 5 pieces); each is now one token. On a real sample (search tool + 3 device tools + one call) the whole text goes from 637 to 534 tokens |
* They are ordinary, not special, on purpose: `decode(..., skip_special_tokens=True)` drops special tokens, which would delete the keys from a decoded tool call and break `json.loads`. An ordinary added token decodes to exactly its text (checked for all 16, with and without `skip_special_tokens`).
* The match is on the quoted string, so `"type"` becomes the token wherever that exact string appears (also in ordinary JSON in user text). Argument names that differ per tool (`city`, `expression`, ...) stay plain text.
* **Pretraining pin.** Ordinary web/JSON text contains these strings, so text tokenised with the current tokenizer gets ids up to 52,134, which the 52,050-row checkpoint cannot embed. `data/prepare.py` (pretrain mode) and `data/prepare_unimax.py` therefore load the tokenizer at revision `0c711cb2f7af669b2782c90c1d0d4d097df21422` (override: env `PRETRAIN_TOKENIZER_REVISION`). SFT and RL tokenisation use the current tokenizer (`mode` sft/rl). The bins already built are unaffected.
* **Initialisation at SFT.** `sabiyarn/model/resize.py` starts each of the 16 keys (and `<tools>`/`</tools>`) as the mean of the embedding rows of the pieces the old tokenizer splits it into (`"arguments"` ≈ mean of `"`, `ar`, `g`, `uments`, `"`); the language tags and `<POS>` keep mean + noise. `Trainer._resize_for_sft` does this on a fresh SFT start (falls back to mean + noise, with a warning, if the Hub is unreachable).
* Adding more tokens: `scripts/add_tool_tokens.py` (edit `KEY_NAMES`, dry run, then `--push-to` both repos). 25 of the 52,160 SFT rows are still free (ids 52,135–52,159).
