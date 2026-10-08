"""A small agentic runtime for SabiYarn: tools, memory, streaming, structured output.

Standalone by design -- the core imports only the standard library, so this
directory can be copied into another codebase as-is. Optional extras:
    transformers + torch   only for HFBackend (agent.llm)
    pydantic               only if you want model classes for tools/schemas

    from agent import Agent, HFBackend, tool

    @tool
    def get_weather(city: str) -> dict:
        "Current weather for a city."
        return {"city": city, "temp": 31}

    llm = HFBackend(model, tokenizer)          # or any .stream()/.complete()
    bot = Agent(llm, tools=[get_weather])
    print(bot.chat("What is the weather in Lagos?"))

The wire format lives in agent/protocol.py and matches the training corpus
(chat template v2, see docs/TOOL_CALL_FORMAT.md): one assistant turn runs from
<|assistant|> to </s> with tool use INSIDE it --
<tool_call>{"name": ..., "arguments": {...}}</tool_call> then
<tool_response>RESULT</tool_response>, no new <|assistant|> and no </s> in
between. Function schemas are passed as tools= and render as a <tools> block.
"""

from __future__ import annotations

from .agent import Agent
from .events import (
    ConfirmNeeded, Delta, Event, Failed, Final, Plan, RoundStarted, Think, ToolCalled, ToolReturned,
)
from .llm import GenerationSettings, HFBackend, LLM, ScriptedLLM
from .memory import Memory, RecallMemory, SummarizingMemory, WindowMemory, approx_tokens
from .protocol import (
    ASSISTANT, EOS, RESPONSE_CLOSE, RESPONSE_OPEN, STOP_STRINGS, SYSTEM, TOOLS_CLOSE, TOOLS_OPEN,
    TOOL_CALL_CLOSE, TOOL_CALL_OPEN, USER, VALID_VERBS, ParsedCall, ParsedTurn, format_tools,
    parse_assistant, record_tool_call_turn, render, render_tool_call, strip_tags, system_message,
    tool_message, tool_response_content, user_message,
)
from .structured import (
    StructuredOutputError, extract_json, schema_instruction, to_strict_json_schema, validate_payload,
)
from .tools import Tool, Toolbox, ToolError, ToolResult, tool

__all__ = [
    "Agent",
    # events
    "Event", "RoundStarted", "Delta", "Think", "Plan", "ToolCalled", "ToolReturned",
    "ConfirmNeeded", "Final", "Failed",
    # backends
    "LLM", "HFBackend", "ScriptedLLM", "GenerationSettings",
    # memory
    "Memory", "WindowMemory", "SummarizingMemory", "RecallMemory", "approx_tokens",
    # tools
    "tool", "Tool", "Toolbox", "ToolError", "ToolResult",
    # protocol
    "render", "render_tool_call", "parse_assistant", "format_tools", "strip_tags",
    "tool_response_content", "record_tool_call_turn", "tool_message", "user_message",
    "system_message", "ParsedTurn", "ParsedCall", "VALID_VERBS", "STOP_STRINGS",
    "EOS", "SYSTEM", "USER", "ASSISTANT", "TOOLS_OPEN", "TOOLS_CLOSE",
    "TOOL_CALL_OPEN", "TOOL_CALL_CLOSE", "RESPONSE_OPEN", "RESPONSE_CLOSE",
    # structured output
    "to_strict_json_schema", "schema_instruction", "extract_json", "validate_payload",
    "StructuredOutputError",
]

__version__ = "0.1.0"
