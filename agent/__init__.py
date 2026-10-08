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

The wire format lives in agent/protocol.py and matches the training corpus:
roles <|system|>/<|user|>/<|assistant|>, calls <tool_call>name\\n{json}</tool_call>,
results <tool_response>name payload</tool_response>, every turn closed by </s>.
"""

from __future__ import annotations

from .agent import Agent
from .events import (
    ConfirmNeeded, Delta, Event, Failed, Final, Plan, RoundStarted, Think, ToolCalled, ToolReturned,
)
from .llm import GenerationSettings, HFBackend, LLM, ScriptedLLM
from .memory import Memory, RecallMemory, SummarizingMemory, WindowMemory, approx_tokens
from .protocol import (
    ASSISTANT, EOS, SYSTEM, USER, VALID_VERBS, ParsedCall, ParsedTurn,
    format_tools, parse_assistant, render, strip_tags, tool_response_content,
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
    "render", "parse_assistant", "format_tools", "strip_tags", "tool_response_content",
    "ParsedTurn", "ParsedCall", "VALID_VERBS", "EOS", "SYSTEM", "USER", "ASSISTANT",
    # structured output
    "to_strict_json_schema", "schema_instruction", "extract_json", "validate_payload",
    "StructuredOutputError",
]

__version__ = "0.1.0"
