# agent — a small agentic runtime for SabiYarn

Tools, memory, streaming and structured output around a function-calling model.
The core is **standard library only**, so this directory can be copied into
another codebase as-is.

```
agent/
  protocol.py    the wire format: tags, renderer, parser          (stdlib)
  tools.py       @tool, schemas, validation, confirm, timeout      (stdlib)
  memory.py      window / summarise-by-chunk / vector recall       (stdlib)
  llm.py         LLM protocol + HFBackend + ScriptedLLM            (torch+transformers for HFBackend only)
  structured.py  pydantic or JSON Schema -> validated output       (pydantic optional)
  events.py      the event types run() yields                      (stdlib)
  agent.py       the loop                                           (stdlib)
```

## Quick start

```python
from agent import Agent, HFBackend, tool

@tool
def get_weather(city: str) -> dict:
    """Current weather for a city."""
    return {"city": city, "temp": 31}

llm = HFBackend(model, tokenizer)          # or anything with .stream()/.complete()
bot = Agent(llm, tools=[get_weather])

bot.chat("What's the weather in Lagos?")   # -> 'It is 31C in Lagos.'
```

Streaming, with everything the loop does exposed as events:

```python
from agent import Delta, Final, Think, ToolCalled, ToolReturned

for ev in bot.run("And in Kano?"):
    match ev:
        case Delta(text):              print(text, end="", flush=True)   # answer only
        case Think(text):              log.debug("scratchpad: %s", text)
        case ToolCalled(name, args, _): log.info("calling %s%s", name, args)
        case ToolReturned(name, value, error, _): log.info("%s -> %s", name, error or value)
        case Final(text, turn, rounds, _): log.info("done in %d round(s)", rounds)
```

`Delta` carries **only** text inside `<response>`. `<think>`, `<task_plan>` and
tool calls arrive as their own events, so a UI shows or hides them deliberately
instead of leaking scaffolding into the answer.

## The format it speaks

Matches the training corpus exactly — see `protocol.py`, pinned by
`tests/test_agent_render_parity.py` against `sabiyarn/chat_template.jinja`.

| | |
|---|---|
| roles | `<|system|>` `<|user|>` `<|assistant|>`, every turn closed by `</s>` |
| tool call | `<tool_call>name\n{"arg": "value"}</tool_call>` |
| tool result | `<tool_response>name payload</tool_response>` |
| reasoning | `<think>…</think>` (English, not for the user), `<task_plan><|RAG|><|explain|></task_plan>` |
| answer | `<response>…` — no closing tag, it runs to `</s>` |
| tool schemas | OpenAI function JSON, compact, as text in the system turn |

Two traps this package exists to get right:

1. **The model emits tool calls inline**, inside one assistant turn, so the loop
   parses text rather than reading a structured field.
2. **The chat template drops `content` when `tool_calls` is present.** A
   tool-calling turn therefore goes back into history as content-only
   (`record_tool_call_turn`), or the `<think>`/`<task_plan>` prefix vanishes from
   the next prompt and the model sees a shape it never saw in training.

## Tools

Schemas come from type hints and the docstring's first line:

```python
@tool(requires_confirmation=True, timeout=10)
def send_money(to: str, amount: float) -> str:
    """Sends money to a contact."""
    ...
```

- `requires_confirmation=True` → the agent asks before running. Pass
  `Agent(confirm=fn)` to answer programmatically, or handle the `ConfirmNeeded`
  event and call `.approve()`. **Unanswered means declined.**
- `timeout=` abandons the wait. The thread is not killed — Python cannot do that
  safely — so treat it as responsiveness, not a resource guarantee.
- Arguments are validated against the schema **before** the function runs:
  missing, unknown, mistyped and out-of-enum values are refused and the reason
  is handed back to the model, which usually fixes its own call.
- Only registered tools can be invoked; the allowlist is the registry.
- `Literal[...]`, `Enum`, `list[...]`, `X | None` and pydantic models are all
  understood. `parameters=` overrides inference entirely.

## Memory

```python
from agent import SummarizingMemory

bot = Agent(llm, tools=[...], memory=SummarizingMemory(
    max_tokens=3072,
    count_tokens=llm.count_tokens,
    summarize=bot_llm_complete,     # Agent wires its own model in by default
    chunk_messages=6,
))
```

| | |
|---|---|
| `WindowMemory` | keep the newest turns that fit, archive the rest |
| `SummarizingMemory` | **default** — fold evicted turns into a running digest placed right after the system turn |
| `RecallMemory` | adds embedding retrieval over archived turns (bring your own `embed`) |

The system turn (with the tool catalogue) is always pinned, and a tool call is
never separated from its result — half a pair is a shape the model never saw.
A failing summariser degrades to plain eviction rather than breaking the turn.

## Structured output

```python
from pydantic import BaseModel

class Contact(BaseModel):
    name: str
    phone: str

contact = bot.structured("Extract the contact: Ada, 0801...", schema=Contact)
```

Works with a plain JSON Schema dict too, so pydantic stays optional. The schema
is tightened (every field required, no extra properties, `$ref`s inlined) before
it goes in the prompt, output JSON is extracted tolerantly (markdown fences and
surrounding prose are handled), and a validation failure is fed back once with
the error text — far cheaper than expecting a 300M model to nest JSON correctly
first time.

## Stopping

Generation stops at `</s>` (the tokenizer's own `eos_token_id`), at
`</tool_call>` so the tool can actually run, and at a role marker if the model
tries to continue the conversation itself. Anything generated past a stop is
discarded, so an injected `<|user|>` turn cannot leak into the answer. A call cut
off by the token budget is reported as `Failed` rather than shown as text.

## Porting it

Copy `agent/` and you are done. The core needs nothing but Python ≥3.10.

- **Another runtime?** Implement `stream(prompt, *, stop, max_new_tokens, **kw)
  -> Iterator[str]` plus `complete` and `count_tokens` (see `llm.LLM`). vLLM, an
  HTTP endpoint and a stub are all a single class.
- **Another model family?** Edit the tags and the renderer in `protocol.py`.
  Nothing else knows the format.
- **Testing without a GPU:** `ScriptedLLM(replies=[...])` returns canned turns
  and records every prompt it saw, which is how the tests assert that history
  and tool results are rendered the way training saw them.

## Tests

```bash
pytest tests/test_agent_framework.py tests/test_agent_render_parity.py -q
```

58 tests. The framework suite is stdlib-only; the parity suite needs `jinja2`
and skips without it, and the pydantic tests skip without `pydantic`.
