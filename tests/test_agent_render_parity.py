"""agent/protocol.py's pure-Python renderer must match sabiyarn/chat_template.jinja exactly.

agent/ carries its own renderer so the package is standalone (and because
jinja2 is not in the root environment, which makes
tokenizer.apply_chat_template unavailable there). That freedom is only safe if
the two stay byte-identical: a single space or a dropped eos_token would train
one format and serve another.

This file is skipped when jinja2 is absent -- tests/test_agent_framework.py
still pins the renderer against the literal expected strings in that case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2", reason="jinja2 is not a root dependency")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.protocol import render  # noqa: E402

TEMPLATE_PATH = ROOT / "sabiyarn" / "chat_template.jinja"


def _jinja_render(messages, **kw) -> str:
    """Render through the real template, with transformers' filter semantics."""
    import json

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    template = env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.render(messages=messages, bos_token="<s>", eos_token="</s>", **kw)


CASES: list[list[dict]] = [
    # plain chat
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ],
    # no system turn
    [{"role": "user", "content": "just a question"}],
    # tool call via the structured field, then its result
    [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Lagos"}}}]},
        {"role": "tool", "name": "get_weather", "content": "31C"},
        {"role": "assistant", "content": "<response>It is 31C."},
    ],
    # two calls in one turn
    [
        {"role": "user", "content": "both"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "a", "arguments": {"x": 1}}},
                {"function": {"name": "b", "arguments": {}}},
            ],
        },
    ],
    # tool result with no name
    [{"role": "tool", "content": "nameless payload"}],
    # the inline form the policy actually emits at inference: tags inside content
    [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Kini AWS?"},
        {
            "role": "assistant",
            "content": '<|input_lang|><yor><think>unknown</think><task_plan><|RAG|></task_plan>'
                       '<tool_call>search_internet\n{"query": "what is AWS"}</tool_call>',
        },
        {"role": "tool", "name": "search_internet", "content": '{"result":"cloud"}'},
    ],
    # non-ASCII must survive unescaped (ensure_ascii=False)
    [
        {"role": "user", "content": "Ìjọba ìpínlẹ̀ Èkó"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": {"q": "Èkó"}}}]},
    ],
    # empty content
    [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
]


@pytest.mark.parametrize("messages", CASES, ids=range(len(CASES)))
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_matches_jinja_template(messages, add_generation_prompt):
    assert render(messages, add_generation_prompt=add_generation_prompt) == _jinja_render(
        messages, add_generation_prompt=add_generation_prompt
    )


def test_empty_tool_calls_list_matches_jinja():
    # The template branches on `'tool_calls' not in message`, so an empty list
    # renders no content and no eos. Worth pinning: the intuitive reading of
    # that branch is the opposite.
    messages = [{"role": "assistant", "content": "dropped", "tool_calls": []}]
    assert render(messages, add_generation_prompt=False) == _jinja_render(
        messages, add_generation_prompt=False
    )
