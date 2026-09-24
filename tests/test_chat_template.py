"""One chat template for the whole repo: sabiyarn/chat_template.jinja == data-gen's copy."""
from pathlib import Path

import jinja2

from sabiyarn.chat import CHAT_TEMPLATE_PATH, load_chat_template, use_sabiyarn_chat_template

ROOT = Path(__file__).resolve().parents[1]


def _render(template: str, messages, **kw):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v: __import__("json").dumps(v, ensure_ascii=False)
    return env.from_string(template).render(messages=messages, bos_token="<s>", eos_token="</s>", **kw)


def test_data_gen_copy_has_not_drifted():
    gen = ROOT / "data-gen" / "templates" / "chat_template.jinja"
    assert gen.read_text(encoding="utf-8") == CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"), (
        "data-gen/templates/chat_template.jinja and sabiyarn/chat_template.jinja differ -- copy one over the other "
        "and re-push the template to the tokenizer repo"
    )


def test_rendering_matches_training_format():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert _render(load_chat_template(), msgs) == "<s><|system|>S</s><|user|>hi</s><|assistant|>yo</s>"
    assert _render(load_chat_template(), msgs[:2], add_generation_prompt=True).endswith("</s><|assistant|>")


def test_tool_call_rendering():
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Lagos"}}}]},
        {"role": "tool", "name": "get_weather", "content": "31C"},
    ]
    out = _render(load_chat_template(), msgs)
    assert '<tool_call>get_weather\n{"city": "Lagos"}</tool_call></s>' in out
    # <tool_response> is a real single token in the tokenizer (52037/52038); <tool_result> was not and cost
    # ~5 byte-BPE tokens per tag.
    assert "<tool_response>get_weather 31C</tool_response></s>" in out


def test_use_sabiyarn_chat_template_overrides_stale_template():
    class Tok:
        chat_template = "stale"

    assert use_sabiyarn_chat_template(Tok).chat_template == load_chat_template()


def test_generation_prompt_is_an_exact_prefix_of_the_training_text():
    """No stray whitespace: what the model is prompted with must equal the start of what it was trained on."""
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    full = _render(load_chat_template(), msgs + [{"role": "assistant", "content": "yo"}])
    prompt = _render(load_chat_template(), msgs, add_generation_prompt=True)
    assert full.startswith(prompt) and full[len(prompt):] == "yo</s>"
