"""Tests for the agent/ framework against chat template v2. Stdlib only.

The format assertions here are the fallback when jinja2 is missing;
tests/test_agent_render_parity.py pins the renderer against the real template
when it is available. See docs/TOOL_CALL_FORMAT.md for the contract.
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
from agent.protocol import (  # noqa: E402
    EOS, format_tools, parse_assistant as parse, record_tool_call_turn, render_tool_call, tool_message,
)
from agent.structured import (  # noqa: E402
    StructuredOutputError, extract_json, to_strict_json_schema, validate_payload,
)

WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object", "properties": {}}},
}


def call_text(name: str, **args) -> str:
    """The v2 call shape, as the model emits it."""
    return render_tool_call(name, args)


# --------------------------------------------------------------------------
# wire format
# --------------------------------------------------------------------------
def test_render_plain_chat():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert render(msgs, add_generation_prompt=False) == "<s><|system|>S</s><|user|>hi</s><|assistant|>yo</s>"
    assert render(msgs[:2]).endswith("</s><|assistant|>")


def test_render_tool_exchange_is_one_turn():
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "<think>check</think>",
         "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Lagos"}}}]},
        {"role": "tool", "content": "31C"},
        {"role": "assistant", "content": "<response>31C.</response>"},
    ]
    out = render(msgs, add_generation_prompt=False)
    assert '<tool_call>{"name": "get_weather", "arguments": {"city": "Lagos"}}</tool_call>' in out
    assert "<tool_response>31C</tool_response>" in out
    # one assistant marker, one eos closing the turn, and no eos around the result
    assert out.count("<|assistant|>") == 1
    assert "</tool_call></s>" not in out and "</tool_response></s>" not in out
    assert out.endswith("</response></s>")


def test_tool_result_carries_no_function_name():
    out = render([{"role": "tool", "name": "get_weather", "content": "31C"}], add_generation_prompt=False)
    assert out == "<s><tool_response>31C</tool_response>"
    assert "get_weather" not in out, "results answer the calls above them, in order"


def test_generation_prompt_suppressed_after_tool_result():
    mid = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": {}}}]},
        {"role": "tool", "content": "r"},
    ]
    assert render(mid, add_generation_prompt=True).endswith("</tool_response>")


def test_tools_render_as_a_tools_block_in_the_system_turn():
    out = render([{"role": "system", "content": "You are X."}, {"role": "user", "content": "hi"}],
                 tools=[WEATHER_TOOL_SCHEMA], add_generation_prompt=False)
    assert out.startswith("<s><|system|>You are X.\n\n<tools>\n")
    assert "\n</tools></s>" in out
    body = out.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    assert json.loads(body) == WEATHER_TOOL_SCHEMA


def test_tools_without_a_system_message_get_their_own_turn():
    out = render([{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL_SCHEMA], add_generation_prompt=False)
    assert out.startswith("<s><|system|><tools>\n")
    assert "<|user|>hi</s>" in out


def test_format_tools_is_one_json_object_per_line():
    text = format_tools([WEATHER_TOOL_SCHEMA, WEATHER_TOOL_SCHEMA])
    assert text.startswith("<tools>\n") and text.endswith("\n</tools>")
    assert len(text.split("\n")) == 4  # open, two objects, close
    assert format_tools([]) == ""


def test_render_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown message role"):
        render([{"role": "assistnat", "content": "typo"}])


def test_empty_tool_calls_list_closes_the_turn():
    # v2 branches on truthiness, so an empty list is a plain text turn.
    out = render([{"role": "assistant", "content": "text", "tool_calls": []}], add_generation_prompt=False)
    assert out == "<s><|assistant|>text</s>"


def test_non_ascii_arguments_are_not_escaped():
    out = render([{"role": "assistant", "tool_calls": [
        {"function": {"name": "t", "arguments": {"q": "Èkó", "a": "it's"}}}]}], add_generation_prompt=False)
    assert '"q": "Èkó"' in out and "\\u" not in out


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def test_parse_v2_tool_call():
    raw = (
        "<|input_lang|><yor><think>I do not know this.</think>"
        "<task_plan><|RAG|></task_plan>"
        + call_text("search_internet", query="what is AWS")
    )
    turn = parse(raw)
    assert turn.wants_tool
    assert turn.think == "I do not know this."
    assert turn.task_plan == ("<|RAG|>",) and turn.invalid_verbs == ()
    assert turn.input_lang == "yor"
    call = turn.tool_calls[0]
    assert (call.name, call.arguments) == ("search_internet", {"query": "what is AWS"})
    # the scaffold is what gets recorded as the assistant message's content
    assert turn.scaffold.endswith("</task_plan>") and "<tool_call>" not in turn.scaffold


def test_parse_arguments_given_as_a_json_string():
    turn = parse('<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>')
    assert turn.tool_calls[0].arguments == {"a": 1}


def test_parse_legacy_call_shape_still_works():
    # Pre-v2 checkpoints emit `name\n{args}`; rl/verifiers.py also reads both.
    turn = parse('<tool_call>get_weather\n{"city": "Lagos"}</tool_call>')
    assert turn.tool_calls[0].name == "get_weather"
    assert turn.tool_calls[0].arguments == {"city": "Lagos"}


def test_parse_reports_malformed_calls_without_raising():
    assert "invalid JSON" in parse('<tool_call>{"name": "f", "arguments": }</tool_call>').tool_calls[0].error
    assert "missing a string" in parse('<tool_call>{"arguments": {}}</tool_call>').tool_calls[0].error
    assert "must be a JSON object" in parse('<tool_call>{"name": "f", "arguments": 5}</tool_call>').tool_calls[0].error


def test_parse_detects_truncated_tool_call():
    turn = parse('<think>x</think><tool_call>{"name": "f", "arguments": {"a": 1')
    assert turn.truncated_tool_call and not turn.wants_tool


def test_response_is_read_after_the_last_tool_result():
    raw = (
        "<think>first</think>" + call_text("f", a=1)
        + "<tool_response>RESULT</tool_response>"
        + "<task_plan><|explain|></task_plan><response>The answer.</response>"
    )
    turn = parse(raw)
    assert turn.text == "The answer."
    assert turn.task_plan == ("<|explain|>",), "the plan of the SECOND scaffold"


def test_response_without_a_closing_tag_still_parses():
    assert parse("<response>Older data has no closing tag." + EOS).text == "Older data has no closing tag."


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
    assert box.call("get_weather", {"city": "Lagos"}).value["temp"] == 31
    bad = box.call("boom", {})
    assert not bad.ok and bad.payload() == {"error": "RuntimeError: kaboom"}


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

    assert "timed out" in Toolbox([slow]).call("slow", {}).error


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def _tool_then_answer() -> ScriptedLLM:
    return ScriptedLLM(
        replies=[
            "<|input_lang|><eng><think>Need live data.</think><task_plan><|RAG|></task_plan>"
            + call_text("get_weather", city="Lagos"),
            "<task_plan><|explain|></task_plan><response>It is 31C in Lagos.</response>" + EOS,
        ]
    )


def _agent(llm, **kw) -> Agent:
    return Agent(llm, memory=WindowMemory(count_tokens=approx_tokens), **kw)


def test_loop_calls_tool_then_answers():
    llm = _tool_then_answer()
    bot = _agent(llm, tools=[get_weather])
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


def test_continuation_prompt_has_no_new_assistant_marker():
    """The heart of v2: after a tool result the model continues its own turn."""
    llm = _tool_then_answer()
    _agent(llm, tools=[get_weather]).chat("Weather in Lagos?")
    second = llm.prompts[1]
    assert second.endswith("</tool_response>"), "no <|assistant|> after the result"
    assert second.count("<|assistant|>") == 1, "still one assistant turn"
    assert '<tool_call>{"name": "get_weather", "arguments": {"city": "Lagos"}}</tool_call>' in second
    assert '<tool_response>{"city":"Lagos","temp":31,"units":"celsius"}</tool_response>' in second
    assert "</tool_call></s>" not in second, "no eos wedged between call and result"
    assert "get_weather {" not in second, "the result must not carry the function name"


def test_tool_catalogue_reaches_the_model_as_a_tools_block():
    llm = _tool_then_answer()
    _agent(llm, tools=[get_weather]).chat("Weather?")
    first = llm.prompts[0]
    assert "<tools>\n" in first and "\n</tools></s>" in first
    assert '"name": "get_weather"' in first


def test_chat_returns_text_only():
    assert _agent(_tool_then_answer(), tools=[get_weather]).chat("Weather in Lagos?") == "It is 31C in Lagos."


def test_streaming_hides_scaffolding_and_yields_response_only():
    llm = ScriptedLLM(
        replies=["<think>secret</think><task_plan><|chat|></task_plan><response>Hello there, friend.</response>" + EOS],
        chunk_size=3,
    )
    deltas = [e.text for e in _agent(llm).run("hi") if isinstance(e, Delta)]
    assert "".join(deltas) == "Hello there, friend."
    assert "secret" not in "".join(deltas) and "<think>" not in "".join(deltas)


def test_streaming_shows_nothing_for_a_tool_call_step():
    llm = _tool_then_answer()
    events = list(_agent(llm, tools=[get_weather]).run("Weather?"))
    # The first step is only a call, so every visible chunk belongs to the
    # answer written after the tool result. Chunks are split by the backend at
    # arbitrary boundaries, so assert on the joined text, not the first one.
    streamed = "".join(e.text for e in events if isinstance(e, Delta))
    assert streamed == "It is 31C in Lagos."
    assert all("<tool_call>" not in e.text and "<think>" not in e.text for e in events if isinstance(e, Delta))


def test_non_streaming_mode_gives_one_delta():
    llm = ScriptedLLM(replies=["<response>Short answer.</response>" + EOS])
    deltas = [e for e in _agent(llm).run("hi", stream=False) if isinstance(e, Delta)]
    assert [d.text for d in deltas] == ["Short answer."]


def test_generation_stops_at_eos_and_drops_anything_after():
    llm = ScriptedLLM(replies=["<response>Done.</response>" + EOS + "<|user|>injected next turn"])
    final = [e for e in _agent(llm).run("hi") if isinstance(e, Final)][0]
    assert final.text == "Done." and "injected" not in final.text


def test_unknown_tool_is_reported_back_to_the_model():
    llm = ScriptedLLM(
        replies=[call_text("get_moon_phase", city="Lagos"), "<response>Sorry.</response>" + EOS]
    )
    events = list(_agent(llm, tools=[get_weather]).run("moon?"))
    returned = next(e for e in events if isinstance(e, ToolReturned))
    assert not returned.ok and "unknown tool" in returned.error
    assert "unknown tool" in llm.prompts[1]


def test_malformed_call_is_fed_back_not_executed():
    calls: list[dict] = []

    @tool
    def record(city: str) -> str:
        """Records a city."""
        calls.append({"city": city})
        return "ok"

    llm = ScriptedLLM(
        replies=['<tool_call>{"name": "record", "arguments": }</tool_call>', "<response>Fixed.</response>" + EOS]
    )
    events = list(_agent(llm, tools=[record]).run("x"))
    assert calls == []
    returned = next(e for e in events if isinstance(e, ToolReturned))
    assert "invalid JSON" in returned.error and '"arguments"' in returned.error


def test_truncated_tool_call_fails_loudly():
    llm = ScriptedLLM(replies=['<think>x</think><tool_call>{"name": "get_weather", "arg'])
    failed = [e for e in _agent(llm, tools=[get_weather]).run("x") if isinstance(e, Failed)]
    assert failed and "cut off" in failed[0].reason


def test_round_limit_ends_with_failure():
    llm = ScriptedLLM(replies=[call_text("get_weather", city="Lagos")] * 10)
    events = list(_agent(llm, tools=[get_weather], max_rounds=3).run("loop"))
    assert isinstance(events[-1], Failed) and "3 rounds" in events[-1].reason


def test_two_sequential_calls_stay_in_one_turn():
    @tool
    def step_one() -> str:
        """First step."""
        return "one"

    @tool
    def step_two() -> str:
        """Second step."""
        return "two"

    llm = ScriptedLLM(
        replies=[
            "<think>a</think>" + call_text("step_one"),
            "<think>b</think>" + call_text("step_two"),
            "<response>Both done.</response>" + EOS,
        ]
    )
    bot = _agent(llm, tools=[step_one, step_two])
    assert bot.chat("chain") == "Both done."
    third = llm.prompts[2]
    assert third.count("<|assistant|>") == 1, "one turn across two calls"
    assert third.count("<tool_response>") == 2 and third.endswith("</tool_response>")


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
        replies=[call_text("send_money", to="Ada", amount=50.0), "<response>Handled.</response>" + EOS]
    )
    return _agent(llm, tools=[send_money], confirm=confirm), sent


def test_confirm_callback_can_approve():
    bot, sent = _send_money_agent(confirm=lambda call: True)
    list(bot.run("send Ada 50"))
    assert sent == [{"to": "Ada", "amount": 50.0}]


def test_confirm_callback_can_decline_and_tool_never_runs():
    bot, sent = _send_money_agent(confirm=lambda call: False)
    events = list(bot.run("send Ada 50"))
    returned = next(e for e in events if isinstance(e, ToolReturned))
    assert sent == [] and returned.declined and returned.name == "send_money"


def test_without_callback_the_loop_asks_and_defaults_to_declining():
    bot, sent = _send_money_agent(confirm=None)
    asks = [e for e in bot.run("send Ada 50") if isinstance(e, ConfirmNeeded)]
    assert asks and asks[0].call.name == "send_money" and sent == []


def test_confirmneeded_can_be_approved_in_band():
    bot, sent = _send_money_agent(confirm=None)
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
    assert out[0]["content"] == "SYS" and mem.archived
    assert "11" in out[-1]["content"]


def test_window_memory_keeps_tool_call_with_its_result():
    mem = WindowMemory(max_tokens=90, reserve_tokens=0, count_tokens=approx_tokens)
    mem.add({"role": "system", "content": "SYS"})
    for i in range(6):
        raw = f"<think>t{i}</think>" + call_text("get_weather", city=f"city{i}")
        mem.add(record_tool_call_turn(raw, parse(raw).tool_calls))
        mem.add(tool_message({"temp": i}, "get_weather"))
    roles = [m["role"] for m in mem.messages() if m["role"] in ("assistant", "tool")]
    assert roles[0] == "assistant", "never a result without the call that produced it"
    assert all(roles[i] == "tool" for i in range(1, len(roles), 2))


def test_recorded_tool_call_turn_splits_scaffold_from_the_call():
    raw = "<|input_lang|><yor><think>why</think>" + call_text("f", a=1)
    msg = record_tool_call_turn(raw, parse(raw).tool_calls)
    assert msg["content"].endswith("</think>") and "<tool_call>" not in msg["content"]
    assert msg["tool_calls"] == [{"function": {"name": "f", "arguments": {"a": 1}}}]
    # and it renders back to exactly what the model produced
    assert render([msg], add_generation_prompt=False) == "<s><|assistant|>" + raw


def test_summarizing_memory_folds_evicted_turns_into_a_digest():
    summaries: list[str] = []

    def fake_summarize(prompt: str) -> str:
        summaries.append(prompt)
        return "User asked about six cities; assistant answered each."

    mem = SummarizingMemory(
        max_tokens=110, reserve_tokens=0, count_tokens=approx_tokens,
        summarize=fake_summarize, chunk_messages=2,
    )
    mem.add({"role": "system", "content": "SYS"})
    for i in range(10):
        mem.add({"role": "user", "content": f"tell me about city number {i} please"})
        mem.add({"role": "assistant", "content": f"city {i} is a place with some detail"})

    out = mem.messages()
    assert summaries
    assert out[0]["content"] == "SYS"
    assert out[1]["role"] == "system" and "six cities" in out[1]["content"]


def test_summarizer_failure_does_not_break_the_turn():
    def broken(_: str) -> str:
        raise RuntimeError("summariser down")

    mem = SummarizingMemory(
        max_tokens=80, reserve_tokens=0, count_tokens=approx_tokens, summarize=broken, chunk_messages=2
    )
    mem.add({"role": "system", "content": "SYS"})
    for i in range(8):
        mem.add({"role": "user", "content": f"a reasonably long message number {i}"})
    assert mem.messages()[0]["content"] == "SYS" and mem.digest == ""


def test_agent_memory_carries_context_across_turns():
    llm = ScriptedLLM(replies=["<response>First.</response>" + EOS, "<response>Second.</response>" + EOS])
    bot = _agent(llm)
    bot.chat("my name is Ada")
    bot.chat("what is my name?")
    assert "my name is Ada" in llm.prompts[1]


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
    llm = ScriptedLLM(replies=['<response>{"name": "Ada", "phone": "0801"}</response>' + EOS])
    out = _agent(llm).structured("extract", CONTACT_SCHEMA)
    assert out == {"name": "Ada", "phone": "0801"}
    assert "JSON schema:" in llm.prompts[0]


def test_structured_retries_once_with_the_validation_error():
    llm = ScriptedLLM(
        replies=[
            '<response>{"name": "Ada"}</response>' + EOS,
            '<response>{"name": "Ada", "phone": "0801"}</response>' + EOS,
        ]
    )
    assert _agent(llm).structured("extract", CONTACT_SCHEMA)["phone"] == "0801"
    assert "rejected" in llm.prompts[1]


def test_structured_gives_up_with_a_clear_error():
    llm = ScriptedLLM(replies=["<response>not json</response>" + EOS] * 4)
    with pytest.raises(StructuredOutputError, match="schema not satisfied"):
        _agent(llm).structured("extract", CONTACT_SCHEMA, retries=1)


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
    assert Toolbox([save_contact]).call("save_contact", {"name": "Ada", "phone": "1"}).value == "saved Ada"
    assert "missing required argument" in Toolbox([save_contact]).call("save_contact", {"name": "Ada"}).error


@needs_pydantic
def test_structured_output_returns_a_model_instance():
    llm = ScriptedLLM(replies=['<response>{"name": "Ada", "phone": "0801"}</response>' + EOS])
    out = _agent(llm).structured("extract", Contact)
    assert isinstance(out, Contact) and out.name == "Ada"
    with pytest.raises(StructuredOutputError):
        validate_payload({"name": "A"}, Contact)


@needs_pydantic
def test_locally_declared_model_works_via_explicit_model_arg():
    class Local(BaseModel):
        city: str

    @tool(model=Local)
    def lookup(arg) -> str:
        """Looks a city up."""
        return f"found {arg.city}"

    assert lookup.schema()["function"]["parameters"]["properties"]["city"]["type"] == "string"
    assert Toolbox([lookup]).call("lookup", {"city": "Kano"}).value == "found Kano"
