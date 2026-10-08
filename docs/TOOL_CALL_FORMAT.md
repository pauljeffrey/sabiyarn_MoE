# Tool-call format (chat template v2) — contract for data, rewards and serving

Source of truth: `sabiyarn/chat_template.jinja` (byte-identical copy in `data-gen/templates/`). Everything below is what that template renders; if this note and the
template disagree, the template wins. Status: implemented in the repo, **not yet pushed to the Hub tokenizer** (`BeardedMonster/SabiYarn-32k` still carries the previous
version), and `<tools>`/`</tools>` are **not yet tokens** (see "Pending").

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

## 6. Pending (needs the owner's decision)
* Push the template to the Hub tokenizer; the `Aletheia-ng/SabiYarn_MoE-280M` tokenizer still has an older template (`<tool_result>`) and must be synced before it is used for serving or rollouts.
* Add `<tools>` (52117) and `</tools>` (52118) as special tokens: `python scripts/add_tool_tokens.py --out DIR` (dry run, verified) then `--push`. This must happen **before** `sft.bin` is tokenised; the vocabulary growth to 52,160 rows already covers them. `data-gen/tests/test_chat_template.py::test_every_marker_the_template_emits_is_a_real_token` fails until then, on purpose.
* Add `</tool_call>` to `generation_config.eos_token_id` of the served model.
