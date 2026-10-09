"""agent/protocol.py's renderer must match sabiyarn/chat_template.jinja exactly.

agent/ carries its own pure-Python renderer so the package is standalone (and
because jinja2 is not a root dependency, which makes
tokenizer.apply_chat_template unavailable there). That is only safe if the two
stay byte-identical: one missing space, one stray </s> between a call and its
result, and training text and inference text diverge.

Skipped when jinja2 is absent -- tests/test_agent_framework.py still pins the
renderer against literal expected strings in that case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2", reason="jinja2 is not a root dependency")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.protocol import render  # noqa: E402

TEMPLATE_PATH = ROOT / "sabiyarn" / "chat_template.jinja"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Search Wikipedia.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}


def _jinja_render(messages, tools=None, **kw) -> str:
    """Render through the real template with transformers' filter semantics."""
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    template = env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.render(
        messages=messages, tools=tools, bos_token="<s>", eos_token="</s>", **kw
    )


SCAFFOLD = "<|input_lang|><yor><think>Mo nilo lati wa alaye.</think><task_plan><|RAG|></task_plan>"
ANSWER = "<|input_lang|><yor><task_plan><|RAG|><|explain|></task_plan><|target_lang|><yor><response>Onkqwe.</response>"

CASES: list[tuple[list[dict], list[dict] | None]] = [
    # plain chat, no tools
    ([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ], None),
    # tools appended to an existing system turn
    ([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "weather?"},
    ], [WEATHER_TOOL]),
    # tools with NO system message: they get a system turn of their own
    ([{"role": "user", "content": "weather?"}], [WEATHER_TOOL]),
    # tools with an EMPTY system message (no blank-line separator)
    ([{"role": "system", "content": ""}, {"role": "user", "content": "hi"}], [WEATHER_TOOL]),
    # two tools: one JSON object per line
    ([{"role": "user", "content": "hi"}], [WEATHER_TOOL, SEARCH_TOOL]),
    # the full v2 exchange: scaffold + call, result, continuation, one </s>
    ([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tani Biyi Bandele?"},
        {"role": "assistant", "content": SCAFFOLD,
         "tool_calls": [{"function": {"name": "search_wikipedia",
                                      "arguments": {"query": "Biyi Bandele", "language": "yo"}}}]},
        {"role": "tool", "content": "Biyi Bandele jẹ onkọwe."},
        {"role": "assistant", "content": ANSWER},
    ], [SEARCH_TOOL]),
    # mid-exchange: last message is a tool result, so NO <|assistant|> is added
    ([
        {"role": "user", "content": "Tani?"},
        {"role": "assistant", "content": SCAFFOLD,
         "tool_calls": [{"function": {"name": "search_wikipedia", "arguments": {"query": "x"}}}]},
        {"role": "tool", "content": "result text"},
    ], [SEARCH_TOOL]),
    # call with no scaffold content at all
    ([
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Lagos"}}}]},
    ], [WEATHER_TOOL]),
    # arguments already serialized as a JSON string: embedded as-is
    ([
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Kano"}'}}]},
    ], None),
    # empty arguments
    ([
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "ping", "arguments": {}}}]},
    ], None),
    # two calls in one message (renderable, though a rollout stops after the first)
    ([
        {"role": "user", "content": "both"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "a", "arguments": {"x": 1}}},
            {"function": {"name": "b", "arguments": {"y": "2"}}},
        ]},
    ], None),
    # two sequential calls, each with its own result, inside one turn
    ([
        {"role": "user", "content": "chain"},
        {"role": "assistant", "content": SCAFFOLD,
         "tool_calls": [{"function": {"name": "a", "arguments": {"x": 1}}}]},
        {"role": "tool", "content": "first"},
        {"role": "assistant", "content": "<think>again</think>",
         "tool_calls": [{"function": {"name": "b", "arguments": {"x": 2}}}]},
        {"role": "tool", "content": "second"},
        {"role": "assistant", "content": ANSWER},
    ], None),
    # non-ASCII in arguments must survive unescaped (ensure_ascii=False)
    ([
        {"role": "user", "content": "Ìjọba ìpínlẹ̀ Èkó"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": {"q": "Èkó", "apos": "it's"}}}]},
    ], None),
    # empty tool_calls list: falsy, so the turn ends with eos
    ([{"role": "assistant", "content": "text", "tool_calls": []}], None),
    # empty content
    ([{"role": "user", "content": ""}, {"role": "assistant", "content": ""}], None),
]


@pytest.mark.parametrize("messages,tools", CASES, ids=range(len(CASES)))
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_matches_jinja_template(messages, tools, add_generation_prompt):
    assert render(messages, tools=tools, add_generation_prompt=add_generation_prompt) == _jinja_render(
        messages, tools=tools, add_generation_prompt=add_generation_prompt
    )


def test_worked_example_from_the_format_doc():
    """The exact shape docs/TOOL_CALL_FORMAT.md section 1 documents."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tani Biyi Bandele?"},
        {"role": "assistant", "content": SCAFFOLD,
         "tool_calls": [{"function": {"name": "search_wikipedia",
                                      "arguments": {"query": "Biyi Bandele", "language": "yo"}}}]},
        {"role": "tool", "content": "Biyi Bandele jẹ onkọwe ọmọ Naijiria."},
        {"role": "assistant", "content": ANSWER},
    ]
    out = render(messages, tools=[SEARCH_TOOL], add_generation_prompt=False)

    assert out.startswith("<s><|system|>You are a helpful assistant.\n\n<tools>\n")
    assert out.count("<|assistant|>") == 1, "the whole exchange is ONE assistant turn"
    assert out.count("</s>") == 3, "system, user, and one closing the assistant turn"
    assert '<tool_call>{"name": "search_wikipedia", "arguments": {"query": "Biyi Bandele", "language": "yo"}}</tool_call>' in out
    assert "<tool_response>Biyi Bandele jẹ onkọwe ọmọ Naijiria.</tool_response>" in out
    # no </s> between the call and the result, and none after the result
    assert "</tool_call></s>" not in out and "</tool_response></s>" not in out
    assert out.endswith("<response>Onkqwe.</response></s>")


def test_generation_prompt_is_omitted_after_a_tool_result():
    mid = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": {}}}]},
        {"role": "tool", "content": "r"},
    ]
    prompt = render(mid, add_generation_prompt=True)
    assert prompt.endswith("</tool_response>"), "the model continues its own turn"
    assert render([{"role": "user", "content": "q"}]).endswith("<|assistant|>")
