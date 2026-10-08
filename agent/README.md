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
    # step 1 ends at </tool_call>, with no </s>: the turn continues
    '<think>need live data</think><task_plan><|RAG|></task_plan>'
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Lagos"}}</tool_call>',
    # step 2 continues the same turn and closes it
    '<response>It is 31C in Lagos.</response>' + EOS,
])
Agent(llm, tools=[get_weather]).chat("weather in Lagos?")
```

## How the loop works

```
user message
   -> render history, with tool schemas passed as tools= (a <tools> block)
   -> generate (stops at </tool_call> or </s>)
   -> ended at </s>?         -> the turn is done -> Final
   -> ended at </tool_call>? -> validate args -> confirm if required -> run
                             -> append <tool_response>RESULT</tool_response>
                             -> re-render (NO new <|assistant|>) and generate
                                again: the model continues the SAME turn
   -> after max_rounds without an answer -> Failed
```

Each round is one step of a single assistant turn, so `Final.turn` is parsed
from the whole turn's text, not just the last step.

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

Chat template **v2**. Source of truth: `sabiyarn/chat_template.jinja`, with the
full contract in [docs/TOOL_CALL_FORMAT.md](../docs/TOOL_CALL_FORMAT.md). This
renderer is pinned against that template byte-for-byte by
`tests/test_agent_render_parity.py` (32 cases).

**One assistant turn runs from `<|assistant|>` to `</s>`, and tool use happens
inside it:**

```
<|assistant|>SCAFFOLD<tool_call>{"name": "f", "arguments": {...}}</tool_call><tool_response>RESULT</tool_response>SCAFFOLD<response>ANSWER</response></s>
```

| | |
|---|---|
| roles | `<\|system\|>` `<\|user\|>` `<\|assistant\|>` |
| tool call | `<tool_call>{"name": "f", "arguments": {...}}</tool_call>` — one JSON object (Hermes/Qwen shape); `arguments` may be a JSON string |
| end of a call | **no `</s>`** — `</tool_call>` is the last thing the model generates in that step |
| tool result | `<tool_response>text</tool_response>` — **no function name, no `</s>`** |
| continuation | the text after a result has **no new `<\|assistant\|>`**; the turn has exactly one `</s>`, after the answer |
| scaffold | `<\|input_lang\|><yor>`, optional `<think>…</think>`, `<task_plan><\|RAG\|></task_plan>` — written before the call, and **again** after the result |
| answer | `<response>…</response>` (closing tag optional in pre-v2 data) |
| tool schemas | passed as `tools=` and rendered once as `<tools>\n{json}\n{json}\n</tools>` in the system turn |

Four consequences the framework is built around:

1. **`</tool_call>` is the stop.** The loop stops there, runs the tool, appends
   the result, and lets the model continue the same turn.
2. **One call per step.** Because `</tool_call>` stops generation, a rollout
   halts after the first call, and nothing is trained to follow the last one.
   Sequential calls are the supported pattern — the loop does several rounds,
   all inside one turn.
3. **Tool results carry no name** and are masked from the loss; results answer
   the calls above them, in order.
4. **A tool-call turn is recorded as scaffold + structured call**, via
   `record_tool_call_turn()` — `content` keeps the scaffold, `tool_calls` the
   call. Recording the raw text as content-only would make the template close
   the turn with `</s>` between the call and its result, a shape the model never
   saw. (This is the opposite of what v1 required.)

JSON is rendered with `json.dumps(ensure_ascii=False)` and key order preserved,
matching `transformers`' `tojson` exactly. Sorted keys or `\uXXXX` escapes would
make training text and inference text differ on precisely the Yorùbá queries.

> **Two pending items** from `docs/TOOL_CALL_FORMAT.md` §6 that affect serving,
> neither of which this package depends on: `<tools>`/`</tools>` are not yet
> tokens (52117/52118 — they render and parse fine meanwhile, just at more
> tokens), and `</tool_call>` is not yet in the served model's
> `generation_config.eos_token_id`. `HFBackend` passes it as a `stop_strings`
> argument on every call, so stopping works regardless of that config.

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
