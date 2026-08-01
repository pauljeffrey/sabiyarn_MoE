"""Deterministic rendering of a Conversation into final training text.

Loads the model's *actual* chat_template.jinja (copied verbatim from the
tokenizer at templates/chat_template.jinja -- see README for how to refresh
it if the tokenizer's template ever changes) and renders through Jinja2
exactly as `tokenizer.apply_chat_template` would. This is the only place
that knows about `<|system|>`, `<tool_call>`, etc. -- generators and
postprocessing never hardcode special tokens.
"""

from __future__ import annotations

from functools import lru_cache

from jinja2 import Environment
from jinja2.sandbox import ImmutableSandboxedEnvironment

from config.settings import CHAT_TEMPLATE_PATH
from schemas.messages import Conversation

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


@lru_cache(maxsize=1)
def _env() -> Environment:
    # Must match transformers' own Jinja2 environment settings
    # (ImmutableSandboxedEnvironment with trim_blocks=True, lstrip_blocks=True)
    # or rendered whitespace will diverge from what apply_chat_template
    # produces at training/inference time.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise

    def tojson(value, indent=None):
        import json

        return json.dumps(value, ensure_ascii=False, indent=indent)

    env.filters["tojson"] = tojson
    return env


def _raise(msg: str):  # pragma: no cover - template safety hook
    raise ValueError(msg)


@lru_cache(maxsize=1)
def _template():
    template_str = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _env().from_string(template_str)


def render_messages(
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
    bos_token: str = BOS_TOKEN,
    eos_token: str = EOS_TOKEN,
) -> str:
    """Render a raw list of template-shaped message dicts to text."""
    return _template().render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=bos_token,
        eos_token=eos_token,
    )


def render_conversation(conversation: Conversation, *, add_generation_prompt: bool = False) -> str:
    return render_messages(
        conversation.to_template_messages(),
        add_generation_prompt=add_generation_prompt,
    )
