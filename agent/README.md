# agent — a small agentic runtime for SabiYarn

Tools, memory, streaming and structured output around a function-calling model.
The core is **standard library only**: copy this directory into any codebase and
it works. `torch`/`transformers` are needed only for `HFBackend`, `pydantic`
only if you want model classes.

- [60-second start](#60-second-start)
- [How the loop works](#how-the-loop-works)
- [The format contract](#the-format-contract) ← read this before changing anything
- [Tools](#tools)
- [Memory](#memory)
- [Structured output](#structured-output)
- [Backends](#backends)
- [Cookbook](#cookbook)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Porting it elsewhere](#porting-it-elsewhere)

```
agent/
  protocol.py    tags, renderer, parser -- the only file that knows the format
  tools.py       @tool, schema inference, validation, confirm, timeout
  memory.py      window / summarise-by-chunk / vector recall
  llm.py         LLM protocol + HFBackend + ScriptedLLM
  structured.py  pydantic or JSON Schema -> validated output
  events.py      the event types run() yields
  agent.py       the loop
  example.py     runnable demos (`python -m agent.example --dry-run`)
```

## 60-second start

```python
from agent import Agent, HFBackend, tool

@tool
def get_weather(city: str) -> dict:
    """Current weather for a city."""          # <- becomes the tool description
    return {"city": city, "temp": 31}

llm = HFBackend(model, tokenizer)              # any .stream()/.complete() works
bot = Agent(llm, tools=[get_weather])

bot.chat("What's the weather in Lagos?")       # -> 'It is 31C in Lagos.'
bot.chat("And in Kano?")                       # history is kept automatically
```

No GPU to hand? Drive the whole loop with a scripted model:

```python
from agent import Agent, ScriptedLLM, tool
from agent.protocol import EOS

llm = ScriptedLLM(replies=[
    '<think>need live data</think><task_plan><|RAG|></task_plan>'
    '<tool_call>get_weather\n{"city": "Lagos"}</tool_call>' + EOS,
    '<response>It is 31C in Lagos.' + EOS,
])
Agent(llm, tools=[get_weather]).chat("weather in Lagos?")
```

## How the loop works

```
user message
   -> render history + tool catalogue to training-format text
   -> generate (stops at </s>, </tool_call>, or a role marker)
   -> no tool call?  -> Final
   -> tool call?     -> validate args -> confirm if required -> run
                     -> append <tool_response> -> generate again
   -> after max_rounds without an answer -> Failed
```

`chat()` returns the answer text. `run()` yields events as they happen:

```python
from agent import Delta, Final, Think, Plan, ToolCalled, ToolReturned, ConfirmNeeded, Failed

for ev in bot.run("And in Kano?"):
    match ev:
        case Delta(text):                        print(text, end="", flush=True)
        case Think(text):                        log.debug("scratchpad: %s", text)
        case Plan(verbs, invalid):               log.debug("plan: %s", verbs)
        case ToolCalled(name, args, _):          log.info("calling %s%s", name, args)
        case ToolReturned(name, value, err, _):  log.info("%s -> %s", name, err or value)
        case Final(text, turn, rounds, data):    log.info("done in %d round(s)", rounds)
        case Failed(reason, text, turn):         log.warning("gave up: %s", reason)
```

`Delta` carries **only** text inside `<response>`. Scaffolding arrives as
`Think`/`Plan`/`ToolCalled` so a UI shows or hides it deliberately rather than
having `<think>` leak into an answer. `stream=False` collapses the answer into a
single `Delta`; the event sequence is otherwise identical, so callers never
branch on mode.

## The format contract

Matches the training corpus, and is pinned byte-for-byte against
`sabiyarn/chat_template.jinja` by `tests/test_agent_render_parity.py`.

| | |
|---|---|
| roles | `<\|system\|>` `<\|user\|>` `<\|assistant\|>`, every turn closed by `</s>` |
| tool call | `<tool_call>name\n{"arg": "value"}</tool_call>` (JSON, `{"city": "Lagos"}` spacing) |
| tool result | `<tool_response>name payload</tool_response>` |
| reasoning | `<think>…</think>` (English by convention, not for the user) |
| plan | `<task_plan><\|RAG\|><\|explain\|></task_plan>` — verbs from `VALID_VERBS` |
| answer | `<response>…` — **no closing tag**, it runs to `</s>` |
| tool schemas | OpenAI function JSON, compact, as plain text in the system turn |

`<tool_response>` and `<task_plan>` are single tokens in the tokenizer
(52037/52040). Do not invent tags: `<tool_result>` is *not* in the vocab and
costs ~5 byte-BPE tokens per tag.

### Why tool-call turns are recorded content-only

Two facts about this stack, both of which the loop is built around:

1. **The model emits tool calls inline**, inside one assistant turn:
   `<think>…</think><task_plan>…</task_plan><tool_call>name\n{json}</tool_call>`.
   So the loop parses *text*; it never reads a structured `tool_calls` field
   back from the model.
2. **The chat template drops `message['content']` when `tool_calls` is present.**
   So a tool-calling turn goes back into history as content-only
   (`protocol.record_tool_call_turn`). Passing `tool_calls` instead would delete
   the `<think>`/`<task_plan>` prefix from the next prompt, showing the model a
   shape it never saw. `data-gen/postprocess_gen.py` re-appends the call to
   content for the same reason.

> **Known pipeline mismatch (not this package's doing).** SFT trains on the
> rendered `text` column (`data/prepare.py:188` → `render_messages`), so for a
> tool-calling turn the model is trained on `<|assistant|><tool_call>…</tool_call></s>`
> with **no** think and **no** task_plan — while `build_seeds.py`'s
> `ASSISTANT_FORMAT["tool_call"]` specifies both and
> `rl/verifiers.py::think_precedes_tool_call` rewards them. If you fix the
> template to emit content *and* the calls, re-rendering the corpus is free
> (`text` is derived from `messages`), and this package's recording becomes
> byte-identical to the training text. Until then it parses both shapes, so
> nothing here breaks either way.

## Tools

```python
from typing import Literal
from agent import tool

@tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Search the local store. Returns passages, not an answer."""
    return retriever.search(query, k=top_k)

@tool(requires_confirmation=True, timeout=10)
def send_message(to: str, body: str, channel: Literal["sms", "whatsapp"] = "sms") -> dict:
    """Send a message to a contact."""
    return gateway.send(to, body, channel)
```

**Schemas** come from type hints plus the docstring's first line. Understood:
`str/int/float/bool`, `list[...]`, `dict`, `X | None`, `Literal[...]`, `Enum`,
and pydantic models. Override with `parameters={...}` for hand-written JSON
Schema.

**Validation runs before your function does** — missing, unknown, mistyped and
out-of-enum arguments are refused, and the reason goes back to the model, which
usually corrects its own call. A hallucinated argument never reaches real code.
Only registered tools can be invoked; the registry *is* the allowlist.

**Confirmation.** `requires_confirmation=True` means the agent asks first:

```python
bot = Agent(llm, tools=[send_message], confirm=lambda call: ui.ask(call.name, call.arguments))
```

Without a `confirm` callback the loop yields `ConfirmNeeded`; call
`.approve()`/`.decline()` on the event. **Unanswered means declined** — a tool
that spends money does not run because nobody was listening.

**Timeouts** abandon the wait, they do not kill the thread (Python cannot do
that safely). Treat `timeout=` as keeping the agent responsive, not as a
resource guarantee.

**Errors are information.** A tool that raises produces
`{"error": "RuntimeError: ..."}` in the model's context instead of crashing the
loop, so the model can apologise, retry, or try another tool.

## Memory

Default is `SummarizingMemory` sized for a 4096-token context, with the agent's
own model as the summariser.

| class | behaviour |
|---|---|
| `WindowMemory` | keep the newest turns that fit; archive the rest |
| `SummarizingMemory` | **default** — fold evicted turns into a running digest placed right after the system turn |
| `RecallMemory` | adds embedding retrieval over archived turns (bring your own `embed`) |

```python
from agent import Agent, SummarizingMemory

bot = Agent(llm, tools=[...], memory=SummarizingMemory(
    max_tokens=3072,                  # budget for system + history
    reserve_tokens=512,               # headroom left for the answer
    count_tokens=llm.count_tokens,    # real tokenizer; falls back to ~4 chars/token
    chunk_messages=6,                 # summarise in chunks of this many
))
```

Invariants worth knowing: the system turn (carrying the tool catalogue) is
always pinned; a tool call is never separated from its `<tool_response>`,
because half a pair is a shape the model never saw; and a failing summariser
degrades to plain eviction rather than breaking the turn.

## Structured output

```python
from pydantic import BaseModel

class Contact(BaseModel):
    name: str
    phone: str

contact = bot.structured("Extract the contact: Ada, 0801 234 5678", schema=Contact)
contact.name          # 'Ada'
```

A plain JSON Schema dict works identically, so `pydantic` stays optional:

```python
bot.structured("Summarise as JSON.", schema={
    "type": "object",
    "properties": {"city": {"type": "string"}, "high_c": {"type": "number"}},
    "required": ["city", "high_c"],
    "additionalProperties": False,
})
```

The schema is tightened before it enters the prompt (every field required, no
extra properties, `$ref`s inlined — a small model will not resolve a `$ref`).
Output JSON is extracted tolerantly: markdown fences and surrounding prose are
handled. A validation failure is fed back once with the error text, which is far
cheaper than expecting a 280M model to nest JSON correctly first time; raise
`retries=` for more attempts.

## Backends

```python
from agent import HFBackend, GenerationSettings

llm = HFBackend(model, tokenizer, settings=GenerationSettings(
    max_new_tokens=512, temperature=0.7, top_k=40, top_p=0.95, repetition_penalty=1.15,
))
```

`HFBackend` goes through `model.generate()` rather than a hand-rolled decode
loop, because this model's positions are **learned and absolute**: its own
`prepare_inputs_for_generation` recomputes `position_ids` from the attention
mask, and reimplementing the KV-cache plumbing would mean duplicating that
correctly. Streaming uses `TextIteratorStreamer` on a worker thread.
`skip_special_tokens` is always `False` — the tags *are* the protocol. A prompt
over the context limit raises rather than truncating, because truncation would
cut `<s>` and the system turn with the tool catalogue.

Writing your own is one class — implement `stream`, and `complete`/`count_tokens`:

```python
class MyBackend:
    def stream(self, prompt, *, stop=(), max_new_tokens=None, **kw):
        for chunk in my_runtime.generate(prompt, stop=list(stop), max_tokens=max_new_tokens):
            yield chunk

    def complete(self, prompt, **kw):
        return "".join(self.stream(prompt, **kw))

    def count_tokens(self, text):
        return len(my_tokenizer.encode(text))
```

Stop strings must be honoured and **included** in the output: the loop needs to
see a complete `</tool_call>` to tell a finished call from one the token budget
cut off.

## Cookbook

**RAG.** A retriever is just a tool. Say so in the docstring — "returns
passages, not an answer" is what teaches the model to read results before
replying:

```python
@tool
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search the knowledge base. Returns a list of passages, not an answer."""
    return [{"title": d.title, "text": d.text} for d in store.search(query, top_k)]
```

**A streaming web endpoint.**

```python
def sse(prompt: str):
    for ev in bot.run(prompt):
        if isinstance(ev, Delta):
            yield f"data: {json.dumps({'text': ev.text})}\n\n"
        elif isinstance(ev, ToolCalled):
            yield f"data: {json.dumps({'tool': ev.name})}\n\n"
        elif isinstance(ev, (Final, Failed)):
            yield "data: [DONE]\n\n"
```

**Human-in-the-loop confirmation** over a queue, where the UI answers later:

```python
for ev in bot.run(prompt):
    if isinstance(ev, ConfirmNeeded):
        ev.approved = await ui.confirm(ev.call.name, ev.call.arguments)
```

**Tools added at runtime** (the system prompt is rebuilt so the model sees them):

```python
bot.add_tool(get_exchange_rate)
```

**Inspect exactly what the model saw** — the single most useful debugging move:

```python
llm = ScriptedLLM(replies=[...])
bot = Agent(llm, tools=[get_weather])
bot.chat("weather?")
print(llm.prompts[-1])      # the full rendered prompt, tags and all
```

**Start a fresh conversation** while keeping tools: `bot.reset()`.

## API reference

**`Agent(llm, tools=(), memory=None, identity=…, system_extra="", max_rounds=6,
confirm=None, max_new_tokens=512, stop=STOP_STRINGS, repair_bad_arguments=True)`**

| method | returns |
|---|---|
| `chat(message, **kw)` | answer text (runs to completion) |
| `run(message=None, *, stream=True, system_extra="", max_new_tokens=None, **gen_kw)` | iterator of events |
| `structured(message, schema, *, retries=1, **kw)` | validated object |
| `add_tool(fn_or_tool)` | the `Tool`; refreshes the system prompt |
| `resume(**kw)` | continue with no new user message |
| `reset()` | clear history, keep tools |

**Events**: `RoundStarted(round, prompt_tokens)`, `Delta(text)`, `Think(text)`,
`Plan(verbs, invalid)`, `ToolCalled(name, arguments, raw_arguments)`,
`ToolReturned(name, value, error, declined)`, `ConfirmNeeded(call, approved)`,
`Final(text, turn, rounds, data)`, `Failed(reason, text, turn)`.

**`protocol`**: `render(messages, add_generation_prompt=True)`,
`parse_assistant(text) -> ParsedTurn`, `format_tools(schemas, identity)`,
`record_tool_call_turn(raw)`, `tool_message(name, payload)`,
`tool_response_content(payload)`, `strip_tags(text)`, `VALID_VERBS`,
`STOP_STRINGS`.

**`ParsedTurn`**: `.raw`, `.think`, `.task_plan`, `.tool_calls`, `.response`,
`.input_lang`, `.target_lang`, `.truncated_tool_call`, `.invalid_verbs`,
`.wants_tool`, `.text` (user-visible answer).

**`structured`**: `to_strict_json_schema(schema)`, `schema_instruction(schema)`,
`extract_json(text)`, `validate_payload(payload, schema)`,
`StructuredOutputError`.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| answer contains `<think>` or `</s>` | you printed `raw`, not `Final.text` / `Delta.text` |
| `Failed: tool call was cut off` | the call hit the token budget mid-JSON — raise `max_new_tokens` |
| every tool argument rejected as "must be string" | schema inference could not resolve an annotation; import the type at module level, or pass `parameters=`/`model=` |
| `ToolReturned.error` says "unknown tool" | the model invented a name; it is told so and usually recovers next round |
| tool never runs, `declined=True` | it is `requires_confirmation` and nothing approved it — pass `confirm=` or handle `ConfirmNeeded` |
| `Failed: stopped after N rounds` | tool loop; check the tool returns something the model can finish with, or raise `max_rounds` |
| `ValueError: prompt is N tokens, over the limit` | lower `Memory.max_tokens`, or summarise more aggressively |
| model answers without using tools | tools reach it as text in the system turn; check `llm.prompts[-1]` contains the catalogue |

## Porting it elsewhere

Copy `agent/`. The core needs nothing but Python ≥3.10.

- **Another runtime** — implement the three methods in [Backends](#backends).
- **Another model family** — edit the tags and `render()`/`parse_assistant()` in
  `protocol.py`. Nothing else knows the format.
- **No pydantic** — schemas accept plain dicts everywhere; the pydantic paths
  are lazily imported.
- **Testing without a GPU** — `ScriptedLLM` returns canned turns and records
  every prompt, which is how this package's own tests assert that history and
  tool results render the way training saw them.

## Tests

```bash
pytest tests/test_agent_framework.py tests/test_agent_render_parity.py -q   # 58 tests
python -m agent.example --dry-run                                           # end-to-end, no GPU
```

The framework suite is stdlib-only. The parity suite needs `jinja2` and skips
without it; the pydantic tests skip without `pydantic`.
