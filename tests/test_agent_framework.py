"""Tests for the agent/ framework. Stdlib only: no torch, pydantic or jinja2.

The format assertions duplicate the ground truth in tests/test_chat_template.py
on purpose -- agent/protocol.py has its own pure-Python renderer (jinja2 is not
installed in the root env, so tokenizer.apply_chat_template cannot be relied
on), and these are what pin it to the training format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (  # noqa: E402
    Agent, ConfirmNeeded, Delta, Failed, Final, ScriptedLLM, Think, ToolCalled, ToolReturned,
    Toolbox, WindowMemory, parse_assistant, render, tool,
)
from agent.memory import SummarizingMemory, approx_tokens  # noqa: E402
from agent.protocol import EOS, format_tools, record_tool_call_turn, tool_message  # noqa: E402
from agent.structured import (  # noqa: E402
    StructuredOutputError, extract_json, to_strict_json_schema, validate_payload,
)
from agent.tools import ToolError  # noqa: E402


# --------------------------------------------------------------------------
# wire format
# --------------------------------------------------------------------------
def test_render_matches_training_format():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert render(msgs, add_generation_prompt=False) == "<s><|system|>S</s><|user|>hi</s><|assistant|>yo</s>"
    assert render(msgs[:2]).endswith("</s><|assistant|>")


def test_render_tool_call_and_response():
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Lagos"}}}]},
        {"role": "tool", "name": "get_weather", "content": "31C"},
    ]
    out = render(msgs, add_generation_prompt=False)
    assert '<tool_call>get_weather\n{"city": "Lagos"}</tool_call></s>' in out
    assert "<tool_response>get_weather 31C</tool_response></s>" in out


def test_render_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown message role"):
        render([{"role": "assistnat", "content": "typo"}])


def test_assistant_with_empty_tool_calls_emits_nothing():
    # The template branches on key presence, not truthiness.
    out = render([{"role": "assistant", "content": "ignored", "tool_calls": []}], add_generation_prompt=False)
    assert out == "<s><|assistant|>"


def test_parse_inline_tool_call():
    raw = (
        "<|input_lang|><yor><think>I do not know this.</think>"
        "<task_plan><|RAG|><|explain|></task_plan>"
        '<tool_call>search_internet\n{"query": "what is AWS"}</tool_call>'
    )
    turn = parse_assistant(raw)
    assert turn.wants_tool
    assert turn.think == "I do not know this."
    assert turn.task_plan == ("<|RAG|>", "<|explain|>")
    assert turn.invalid_verbs == ()
    assert turn.input_lang == "yor"
    call = turn.tool_calls[0]
    assert (call.name, call.arguments) == ("search_internet", {"query": "what is AWS"})


def test_parse_final_response_and_invalid_verb():
    turn = parse_assistant("<task_plan><|nope|></task_plan><response>Mo wa daadaa.</s>")
    assert not turn.wants_tool
    assert turn.invalid_verbs == ("<|nope|>",)
    assert turn.text == "Mo wa daadaa."


def test_parse_reports_bad_json_without_raising():
    turn = parse_assistant('<tool_call>calc\n{"a": }</tool_call>')
    call = turn.tool_calls[0]
    assert call.arguments is None and not call.ok and "invalid JSON" in call.error


def test_parse_detects_truncated_tool_call():
    turn = parse_assistant('<think>x</think><tool_call>calc\n{"a": 1')
    assert turn.truncated_tool_call and not turn.wants_tool


def test_tool_catalogue_is_compact_json_in_system_turn():
    schemas = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]
    text = format_tools(schemas, "You are X.")
    assert text.startswith("You are X. You have these tools:\n")
    assert ", " not in text.split("\n", 1)[1]  # compact separators, no wasted tokens
    assert json.loads(text.split("\n", 1)[1]) == schemas
    assert format_tools([], "You are X.") == "You are X."


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Current weather for a city."""
    return {"city": city, "temp": 31, "units": units}


def test_schema_from_signature_and_docstring():
    schema = get_weather.schema()["function"]
    assert schema["name"] == "get_weather"
    assert schema["description"] == "Current weather for a city."
    assert schema["parameters"]["required"] == ["city"]
    assert schema["parameters"]["properties"]["units"]["type"] == "string"
    assert schema["parameters"]["additionalProperties"] is False


def test_validation_rejects_missing_unknown_and_mistyped_arguments():
    box = Toolbox([get_weather])
    assert "missing required argument" in box.call("get_weather", {}).error
    assert "unknown argument" in box.call("get_weather", {"city": "Lagos", "temp": 1}).error
    assert "must be string" in box.call("get_weather", {"city": 7}).error
    assert box.call("nope", {}).error.startswith("unknown tool")


def test_tool_runs_and_captures_its_own_failure():
    @tool
    def boom() -> None:
        """Always fails."""
        raise RuntimeError("kaboom")

    box = Toolbox([get_weather, boom])
    ok = box.call("get_weather", {"city": "Lagos"})
    assert ok.ok and ok.value["temp"] == 31
    bad = box.call("boom", {})
    assert not bad.ok and "RuntimeError: kaboom" in bad.error
    assert bad.payload() == {"error": "RuntimeError: kaboom"}


def test_enum_and_literal_schemas():
    from typing import Literal

    @tool
    def convert(value: float, to: Literal["kg", "lb"]) -> float:
        """Convert a mass."""
        return value

    props = convert.schema()["function"]["parameters"]["properties"]
    assert props["value"]["type"] == "number"
    assert props["to"] == {"type": "string", "enum": ["kg", "lb"]}
    assert "must be one of" in Toolbox([convert]).call("convert", {"value": 1, "to": "st"}).error


def test_timeout_surfaces_as_tool_error():
    import time

    @tool(timeout=0.05)
    def slow() -> str:
        """Sleeps."""
        time.sleep(1.0)
        return "done"

    result = Toolbox([slow]).call("slow", {})
    assert not result.ok and "timed out" in result.error


# --------------------------------------------------------------------------
# agent loop
# --------------------------------------------------------------------------
def _tool_then_answer() -> ScriptedLLM:
    return ScriptedLLM(
        replies=[
            "<|input_lang|><eng><think>Need live data.</think><task_plan><|RAG|></task_plan>"
            '<tool_call>get_weather\n{"city": "Lagos"}</tool_call>' + EOS,
            "<task_plan><|explain|></task_plan><response>It is 31C in Lagos." + EOS,
        ]
    )


def test_loop_calls_tool_then_answers():
    llm = _tool_then_answer()
    bot = Agent(llm, tools=[get_weather], memory=WindowMemory(count_tokens=approx_tokens))
    events = list(bot.run("Weather in Lagos?"))
    kinds = [type(e).__name__ for e in events]
    assert "ToolCalled" in kinds and "ToolReturned" in kinds and kinds[-1] == "Final"

    called = next(e for e in events if isinstance(e, ToolCalled))
    returned = next(e for e in events if isinstance(e, ToolReturned))
    final = events[-1]
    assert called.arguments == {"city": "Lagos"}
    assert returned.ok and returned.value["temp"] == 31
    assert final.text == "It is 31C in Lagos." and final.rounds == 2
    assert any(isinstance(e, Think) for e in events)


def test_second_prompt_contains_tool_result_in_training_shape():
    llm = _tool_then_answer()
    Agent(llm, tools=[get_weather], memory=WindowMemory(count_tokens=approx_tokens)).chat("Weather in Lagos?")
    second = llm.prompts[1]
    # The call must survive as inline content (the template would drop it if it
    # were passed as tool_calls), and the result must come back as <tool_response>.
    assert '<tool_call>get_weather\n{"city": "Lagos"}</tool_call>' in second
    assert '<tool_response>get_weather {"city":"Lagos","temp":31,"units":"celsius"}</tool_response>' in second
    assert second.endswith("<|assistant|>")


def test_chat_returns_text_only():
    bot = Agent(_tool_then_answer(), tools=[get_weather], memory=WindowMemory(count_tokens=approx_tokens))
    assert bot.chat("Weather in Lagos?") == "It is 31C in Lagos."


def test_streaming_hides_scaffolding_and_yields_response_only():
    llm = ScriptedLLM(
        replies=["<think>secret</think><task_plan><|chat|></task_plan><response>Hello there, friend." + EOS],
        chunk_size=3,
    )
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    deltas = [e.text for e in bot.run("hi") if isinstance(e, Delta)]
    assert "".join(deltas) == "Hello there, friend."
    assert "secret" not in "".join(deltas) and "<think>" not in "".join(deltas)


def test_non_streaming_mode_gives_one_delta():
    llm = ScriptedLLM(replies=["<response>Short answer." + EOS])
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    deltas = [e for e in bot.run("hi", stream=False) if isinstance(e, Delta)]
    assert [d.text for d in deltas] == ["Short answer."]


def test_generation_stops_at_eos_and_drops_anything_after():
    llm = ScriptedLLM(replies=["<response>Done." + EOS + "<|user|>injected next turn"])
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    final = [e for e in bot.run("hi") if isinstance(e, Final)][0]
    assert final.text == "Done." and "injected" not in final.text


def test_unknown_tool_is_reported_back_to_the_model():
    llm = ScriptedLLM(
        replies=[
            '<tool_call>get_moon_phase\n{"city": "Lagos"}</tool_call>' + EOS,
            "<response>Sorry, I cannot do that." + EOS,
        ]
    )
    bot = Agent(llm, tools=[get_weather], memory=WindowMemory(count_tokens=approx_tokens))
    events = list(bot.run("moon?"))
    returned = next(e for e in events if isinstance(e, ToolReturned))
    assert not returned.ok and "unknown tool" in returned.error
    assert "unknown tool" in llm.prompts[1]  # the model gets a chance to recover


def test_bad_json_arguments_are_fed_back_not_executed():
    calls: list[dict] = []

    @tool
    def record(city: str) -> str:
        """Records a city."""
        calls.append({"city": city})
        return "ok"

    llm = ScriptedLLM(
        replies=['<tool_call>record\n{"city": }</tool_call>' + EOS, "<response>Fixed." + EOS]
    )
    bot = Agent(llm, tools=[record], memory=WindowMemory(count_tokens=approx_tokens))
    events = list(bot.run("x"))
    assert calls == []  # never executed
    returned = next(e for e in events if isinstance(e, ToolReturned))
    assert "invalid JSON" in returned.error and "valid JSON" in returned.error


def test_truncated_tool_call_fails_loudly():
    llm = ScriptedLLM(replies=['<think>x</think><tool_call>get_weather\n{"city": "Lag'])
    bot = Agent(llm, tools=[get_weather], memory=WindowMemory(count_tokens=approx_tokens))
    failed = [e for e in bot.run("x") if isinstance(e, Failed)]
    assert failed and "cut off" in failed[0].reason


def test_round_limit_ends_with_failure():
    llm = ScriptedLLM(replies=['<tool_call>get_weather\n{"city": "Lagos"}</tool_call>' + EOS] * 10)
    bot = Agent(llm, tools=[get_weather], max_rounds=3, memory=WindowMemory(count_tokens=approx_tokens))
    events = list(bot.run("loop"))
    assert isinstance(events[-1], Failed) and "3 rounds" in events[-1].reason


# --------------------------------------------------------------------------
# confirmation
# --------------------------------------------------------------------------
def _send_money_agent(confirm=None):
    sent: list[dict] = []

    @tool(requires_confirmation=True)
    def send_money(to: str, amount: float) -> str:
        """Sends money."""
        sent.append({"to": to, "amount": amount})
        return "sent"

    llm = ScriptedLLM(
        replies=[
            '<tool_call>send_money\n{"to": "Ada", "amount": 50.0}</tool_call>' + EOS,
            "<response>Handled." + EOS,
        ]
    )
    bot = Agent(llm, tools=[send_money], confirm=confirm, memory=WindowMemory(count_tokens=approx_tokens))
    return bot, sent


def test_confirm_callback_can_approve():
    bot, sent = _send_money_agent(confirm=lambda call: True)
    list(bot.run("send Ada 50"))
    assert sent == [{"to": "Ada", "amount": 50.0}]


def test_confirm_callback_can_decline_and_tool_never_runs():
    bot, sent = _send_money_agent(confirm=lambda call: False)
    events = list(bot.run("send Ada 50"))
    returned = next(e for e in events if isinstance(e, ToolReturned))
    assert sent == [] and returned.declined
    assert returned.name == "send_money"


def test_without_callback_the_loop_yields_confirmneeded_and_defaults_to_declining():
    bot, sent = _send_money_agent(confirm=None)
    events = list(bot.run("send Ada 50"))
    asks = [e for e in events if isinstance(e, ConfirmNeeded)]
    assert asks and asks[0].call.name == "send_money"
    assert sent == []  # unanswered means not approved


def test_confirmneeded_can_be_approved_in_band():
    bot, sent = _send_money_agent(confirm=None)
    # Approving the event object as it is yielded is what a UI would do.
    for event in bot.run("send Ada 50"):
        if isinstance(event, ConfirmNeeded):
            event.approve()
    assert sent == [{"to": "Ada", "amount": 50.0}]


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------
def test_window_memory_pins_system_and_drops_oldest():
    mem = WindowMemory(max_tokens=120, reserve_tokens=0, count_tokens=approx_tokens)
    mem.add({"role": "system", "content": "SYS"})
    for i in range(12):
        mem.add({"role": "user", "content": f"message number {i} with some padding text"})
        mem.add({"role": "assistant", "content": f"reply number {i} with some padding text"})
    out = mem.messages()
    assert out[0]["content"] == "SYS"
    assert len(out) < 25 and mem.archived
    assert "11" in out[-1]["content"]  # newest kept


def test_window_memory_keeps_tool_call_with_its_result():
    mem = WindowMemory(max_tokens=90, reserve_tokens=0, count_tokens=approx_tokens)
    mem.add({"role": "system", "content": "SYS"})
    for i in range(6):
        mem.add(record_tool_call_turn(f'<tool_call>get_weather\n{{"city": "city{i}"}}</tool_call>'))
        mem.add(tool_message("get_weather", {"temp": i}))
    roles = [m["role"] for m in mem.messages() if m["role"] in ("assistant", "tool")]
    # Never a dangling tool result without the call that produced it.
    assert roles[0] == "assistant"
    assert all(roles[i] == "tool" for i in range(1, len(roles), 2))


def test_summarizing_memory_folds_evicted_turns_into_a_digest():
    summaries: list[str] = []

    def fake_summarize(prompt: str) -> str:
        summaries.append(prompt)
        return "User asked about six cities; assistant answered each."

    mem = SummarizingMemory(
        max_tokens=110,
        reserve_tokens=0,
        count_tokens=approx_tokens,
        summarize=fake_summarize,
        chunk_messages=2,
    )
    mem.add({"role": "system", "content": "SYS"})
    for i in range(10):
        mem.add({"role": "user", "content": f"tell me about city number {i} please"})
        mem.add({"role": "assistant", "content": f"city {i} is a place with some detail"})

    out = mem.messages()
    assert summaries, "older turns should have been summarised"
    digest = out[1]
    assert digest["role"] == "system" and "Earlier in this conversation" in digest["content"]
    assert "six cities" in digest["content"]
    assert out[0]["content"] == "SYS"  # tool catalogue still first


def test_summarizer_failure_does_not_break_the_turn():
    def broken(_: str) -> str:
        raise RuntimeError("summariser down")

    mem = SummarizingMemory(
        max_tokens=80, reserve_tokens=0, count_tokens=approx_tokens, summarize=broken, chunk_messages=2
    )
    mem.add({"role": "system", "content": "SYS"})
    for i in range(8):
        mem.add({"role": "user", "content": f"a reasonably long message number {i}"})
    assert mem.messages()[0]["content"] == "SYS"
    assert mem.digest == ""


def test_agent_memory_carries_context_across_turns():
    llm = ScriptedLLM(replies=["<response>First." + EOS, "<response>Second." + EOS])
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    bot.chat("my name is Ada")
    bot.chat("what is my name?")
    assert "my name is Ada" in llm.prompts[1]  # earlier turn still in the prompt


# --------------------------------------------------------------------------
# structured output
# --------------------------------------------------------------------------
CONTACT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "phone": {"type": "string"}},
    "required": ["name", "phone"],
    "additionalProperties": False,
}


def test_extract_json_tolerates_prose_and_fences():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('sure! {"a": {"b": [1,2]}} hope that helps') == {"a": {"b": [1, 2]}}
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


def test_structured_with_dict_schema():
    llm = ScriptedLLM(replies=['<response>{"name": "Ada", "phone": "0801"}' + EOS])
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    out = bot.structured("extract", CONTACT_SCHEMA)
    assert out == {"name": "Ada", "phone": "0801"}
    assert "JSON schema:" in llm.prompts[0]


def test_structured_retries_once_with_the_validation_error():
    llm = ScriptedLLM(
        replies=[
            '<response>{"name": "Ada"}' + EOS,  # missing phone
            '<response>{"name": "Ada", "phone": "0801"}' + EOS,
        ]
    )
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    assert bot.structured("extract", CONTACT_SCHEMA)["phone"] == "0801"
    assert "rejected" in llm.prompts[1]


def test_structured_gives_up_with_a_clear_error():
    llm = ScriptedLLM(replies=["<response>not json" + EOS] * 4)
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    with pytest.raises(StructuredOutputError, match="schema not satisfied"):
        bot.structured("extract", CONTACT_SCHEMA, retries=1)


def test_strict_schema_tightening():
    strict = to_strict_json_schema(
        {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    )
    assert strict["required"] == ["a", "b"] and strict["additionalProperties"] is False


# pydantic is an optional extra, so the model lives at module scope guarded by
# an import check: declared inside a test function, postponed annotations would
# leave only the name "Contact" behind and it could not be resolved.
try:  # pragma: no cover - availability varies by environment
    from pydantic import BaseModel

    class Contact(BaseModel):
        name: str
        phone: str

    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAVE_PYDANTIC = False

needs_pydantic = pytest.mark.skipif(not HAVE_PYDANTIC, reason="pydantic is an optional extra")


@needs_pydantic
def test_pydantic_model_as_tool_argument_and_schema():
    @tool
    def save_contact(contact: Contact) -> str:
        """Saves a contact."""
        return f"saved {contact.name}"

    props = save_contact.schema()["function"]["parameters"]["properties"]
    assert props["name"]["type"] == "string" and props["phone"]["type"] == "string"
    # Arguments arrive as loose JSON and are passed to the function as the model.
    assert Toolbox([save_contact]).call("save_contact", {"name": "Ada", "phone": "1"}).value == "saved Ada"
    assert "missing required argument" in Toolbox([save_contact]).call("save_contact", {"name": "Ada"}).error


@needs_pydantic
def test_structured_output_returns_a_model_instance():
    llm = ScriptedLLM(replies=['<response>{"name": "Ada", "phone": "0801"}' + EOS])
    bot = Agent(llm, memory=WindowMemory(count_tokens=approx_tokens))
    out = bot.structured("extract", Contact)
    assert isinstance(out, Contact) and out.name == "Ada"
    assert isinstance(validate_payload({"name": "A", "phone": "1"}, Contact), Contact)
    with pytest.raises(StructuredOutputError):
        validate_payload({"name": "A"}, Contact)


@needs_pydantic
def test_locally_declared_model_works_via_explicit_model_arg():
    # The escape hatch for a model the annotation cannot name.
    class Local(BaseModel):
        city: str

    @tool(model=Local)
    def lookup(arg) -> str:
        """Looks a city up."""
        return f"found {arg.city}"

    assert lookup.schema()["function"]["parameters"]["properties"]["city"]["type"] == "string"
    assert Toolbox([lookup]).call("lookup", {"city": "Kano"}).value == "found Kano"
