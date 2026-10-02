"""The chat template must only emit markers that EXIST in the tokenizer.

This exists because the published template did not. `tokenizer_config.json` on the Hub renders a tool result
as <tool_result>...</tool_result>, and neither of those is in the vocabulary -- while <tool_response> and
</tool_response> are real single tokens (52037, 52038). Following the published template literally would
shred every tool result into sub-word pieces and teach the model a marker it can never emit cleanly.

The local copy in templates/chat_template.jinja is therefore the CORRECT one, and this test is what keeps it
that way: refreshing it from the Hub without checking would silently reintroduce the bug.
"""
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config.settings import CHAT_TEMPLATE_PATH  # noqa: E402

TOKENIZER_DIRS = sorted(
    Path.home().glob(".cache/huggingface/hub/models--BeardedMonster--SabiYarn-32k/snapshots/*"))


def _vocab() -> set[str]:
    if not TOKENIZER_DIRS:
        pytest.skip("tokenizer not in the local HF cache")
    tj = TOKENIZER_DIRS[-1] / "tokenizer.json"
    if not tj.exists():
        pytest.skip("tokenizer.json not cached")
    d = json.loads(tj.read_text(encoding="utf-8"))
    return set(d["model"]["vocab"]) | {a["content"] for a in d.get("added_tokens", [])}


def test_every_marker_the_template_emits_is_a_real_token():
    tmpl = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    vocab = _vocab()
    # Every literal the template emits, e.g. "<|user|>" or "<tool_response>".
    literals = set(re.findall(r'"(<[^"]+>)"', tmpl))
    assert literals, "no markers found in the template -- the parse is wrong, not the template"
    missing = sorted(m for m in literals if m not in vocab)
    assert not missing, (
        f"the template emits markers that are not in the tokenizer: {missing}. "
        f"They would be split into sub-words. Add the tokens or fix the template.")


def test_the_tool_marker_is_tool_response_not_tool_result():
    """Named explicitly, because this is the specific regression to guard."""
    tmpl = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "<tool_response>" in tmpl
    assert "<tool_result>" not in tmpl, (
        "<tool_result> is NOT in this tokenizer; <tool_response> (52037) is. The Hub's published template "
        "has this wrong -- do not copy it over without checking.")


def test_local_template_differs_from_the_hub_only_where_we_intend():
    """If the Hub template is ever fixed, this test says so, so the local copy can be retired."""
    if not TOKENIZER_DIRS:
        pytest.skip("tokenizer not in the local HF cache")
    cfg = TOKENIZER_DIRS[-1] / "tokenizer_config.json"
    if not cfg.exists():
        pytest.skip("tokenizer_config.json not cached")
    hub = json.loads(cfg.read_text(encoding="utf-8")).get("chat_template") or ""
    if not hub:
        pytest.skip("no chat_template on the Hub copy")
    if "<tool_result>" not in hub:
        pytest.skip("the Hub template no longer uses <tool_result> -- it may have been fixed; re-check "
                    "whether templates/chat_template.jinja can now be refreshed from it")
    assert "<tool_result>" in hub, "sanity: this test only applies while the Hub copy is still wrong"
