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
    """A call is one parseable JSON object; the turn is NOT closed after it; the result carries no function name."""
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Lagos"}}}]},
        {"role": "tool", "name": "get_weather", "content": "31C"},
    ]
    out = _render(load_chat_template(), msgs)
    assert out == ('<s><|user|>weather?</s><|assistant|><tool_call>{"name": "get_weather", "arguments": {"city": "Lagos"}}</tool_call>'
                   "<tool_response>31C</tool_response>")
    call = out.split("<tool_call>")[1].split("</tool_call>")[0]
    assert __import__("json").loads(call) == {"name": "get_weather", "arguments": {"city": "Lagos"}}
    # arguments that arrive as a JSON string (the OpenAI wire shape) are embedded as they are
    s = [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}]
    assert _render(load_chat_template(), s) == '<s><|assistant|><tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'


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


def test_a_tool_using_turn_is_one_turn_with_one_closing_eos():
    scaffold = "<|input_lang|><yor><task_plan><|RAG|></task_plan>"
    msgs = [
        {"role": "user", "content": "Tell me about Biyi Bandele"},
        {"role": "assistant", "content": scaffold, "tool_calls": [{"function": {"name": "search_wikipedia", "arguments": {"query": "Biyi Bandele"}}}]},
        {"role": "tool", "name": "search_wikipedia", "content": "Biyi Bandele (1967-2022) was a writer."},
        {"role": "assistant", "content": "<|input_lang|><yor><task_plan><|RAG|><|explain|></task_plan><|target_lang|><yor><response>Biyi Bandele jẹ onkọwe."},
    ]
    out = _render(load_chat_template(), msgs)
    assert out == ("<s><|user|>Tell me about Biyi Bandele</s><|assistant|>" + scaffold +
                   '<tool_call>{"name": "search_wikipedia", "arguments": {"query": "Biyi Bandele"}}</tool_call>'
                   "<tool_response>Biyi Bandele (1967-2022) was a writer.</tool_response>"
                   "<|input_lang|><yor><task_plan><|RAG|><|explain|></task_plan><|target_lang|><yor><response>Biyi Bandele jẹ onkọwe.</s>")
    assert out.count("<|assistant|>") == 1 and out.count("</s>") == 2          # the user's </s> and the turn's single closing </s>
    # prompting after a tool result continues the same turn: no new <|assistant|>
    assert _render(load_chat_template(), msgs[:3], add_generation_prompt=True) == out.split("<|input_lang|><yor><task_plan><|RAG|><|explain|>")[0]
    # two calls in one message are adjacent, and still no </s> after them
    two = [{"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {}}}]}]
    assert _render(load_chat_template(), two) == ('<s><|assistant|><tool_call>{"name": "a", "arguments": {}}</tool_call>'
                                                   '<tool_call>{"name": "b", "arguments": {}}</tool_call>')


def test_functions_are_rendered_once_in_the_system_turn():
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    block = "<tools>\n" + __import__("json").dumps(tools[0], ensure_ascii=False) + "\n</tools>"
    with_sys = _render(load_chat_template(), [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}], tools=tools)
    assert with_sys == f"<s><|system|>S\n\n{block}</s><|user|>hi</s>"
    no_sys = _render(load_chat_template(), [{"role": "user", "content": "hi"}], tools=tools)
    assert no_sys == f"<s><|system|>{block}</s><|user|>hi</s>"
    assert _render(load_chat_template(), [{"role": "user", "content": "hi"}], tools=None) == "<s><|user|>hi</s>"
