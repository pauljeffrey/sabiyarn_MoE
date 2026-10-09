"""Render conversations to the string that is actually tokenized: the canonical SabiYarn chat template
(sabiyarn/chat_template.jinja) applied with <s> / </s>. Used for the `text` column of every SFT / RL dataset."""

from __future__ import annotations

import json
from pathlib import Path

import jinja2

_TEMPLATE = None
BOS, EOS = "<s>", "</s>"


def _tpl():
    global _TEMPLATE
    if _TEMPLATE is None:
        src = (Path(__file__).resolve().parents[2] / "sabiyarn" / "chat_template.jinja").read_text(encoding="utf-8")
        # trim_blocks / lstrip_blocks are the flags transformers uses when it applies a chat template
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        # transformers replaces jinja's own `tojson` (which sorts keys and writes <, >, &, ' and every non-ASCII letter as \uXXXX) with a plain
        # json.dumps(ensure_ascii=False). Training text must be byte-identical to what apply_chat_template renders at inference, so do the same.
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        _TEMPLATE = env.from_string(src)
    return _TEMPLATE


def render(messages: list[dict], add_generation_prompt: bool = False, tools: list | None = None) -> str:
    return _tpl().render(messages=messages, tools=tools, bos_token=BOS, eos_token=EOS, add_generation_prompt=add_generation_prompt)


def render_json(messages_json: str, add_generation_prompt: bool = False, tools_json: str | None = None) -> str:
    return render(json.loads(messages_json), add_generation_prompt, json.loads(tools_json) if tools_json else None)
