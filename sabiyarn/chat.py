"""The SabiYarn chat template: one canonical copy (`sabiyarn/chat_template.jinja`).

The data generator (data-gen/templates/chat_template.jinja) renders SFT/DPO data with this template, so the
tokenizer used for evaluation, RL and inference must render with it too -- otherwise the model is prompted
in a format it was never trained on. The copy on the Hub tokenizer repo can lag behind, so anything that
builds prompts calls `use_sabiyarn_chat_template(tokenizer)` instead of trusting `tokenizer.chat_template`.
tests/test_chat_template.py fails if the data-gen copy drifts from this one.
"""

from __future__ import annotations

from pathlib import Path

CHAT_TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")


def load_chat_template() -> str:
    return CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")


def use_sabiyarn_chat_template(tokenizer):
    """Set the canonical template on `tokenizer` (in place) and return it."""
    tokenizer.chat_template = load_chat_template()
    return tokenizer
