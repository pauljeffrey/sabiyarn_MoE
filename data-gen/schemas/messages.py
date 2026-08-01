"""Canonical conversation representation.

This is the single source of truth for "what a training example looks
like" between generation and rendering. Generators never emit the final
`<|system|>...` string directly -- they emit these Pydantic objects (via
structured-output JSON from the model), which are then deterministically
rendered against the model's real `chat_template.jinja` in
`rendering/render.py`. That split means an LLM never has to get the exact
special-token syntax right; it only has to produce valid structured JSON,
and Python guarantees the final text matches training format exactly.

Design points that mirror templates/chat_template.jinja precisely:
  - An assistant message is EITHER plain text OR one-or-more tool calls,
    never both (the template branches on `'tool_calls' not in message`,
    it does not look at content when tool_calls is present).
  - A tool-result message may carry an optional `name` (the function that
    produced it); the template prefixes it into <tool_result> when present.
  - `to_template_dict()` must OMIT absent keys entirely (not set them to
    None) because the template tests key presence, not truthiness.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Role = Literal["system", "user", "tool", "assistant"]


class ToolFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    function: ToolFunctionCall

    def to_template_dict(self) -> dict[str, Any]:
        return {
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            }
        }


class Message(BaseModel):
    role: Role
    content: Optional[str] = None
    name: Optional[str] = None  # only meaningful for role == "tool"
    tool_calls: Optional[list[ToolCall]] = None  # only meaningful for role == "assistant"

    @model_validator(mode="after")
    def _check_shape(self) -> "Message":
        if self.role in ("system", "user"):
            if not self.content:
                raise ValueError(f"role={self.role!r} requires non-empty content")
            if self.tool_calls or self.name:
                raise ValueError(f"role={self.role!r} must not set tool_calls/name")
        elif self.role == "tool":
            if not self.content:
                raise ValueError("role='tool' requires non-empty content")
            if self.tool_calls:
                raise ValueError("role='tool' must not set tool_calls")
        elif self.role == "assistant":
            has_text = bool(self.content)
            has_calls = bool(self.tool_calls)
            if has_text == has_calls:
                raise ValueError(
                    "role='assistant' must set exactly one of content or "
                    "tool_calls, never both/neither "
                    f"(content={self.content!r}, tool_calls={self.tool_calls!r})"
                )
            if self.name:
                raise ValueError("role='assistant' must not set name")
        return self

    def to_template_dict(self) -> dict[str, Any]:
        """Render to the exact dict shape chat_template.jinja expects.

        Keys that the template checks for *presence* (not just truthiness)
        -- namely 'tool_calls' on assistant messages -- are omitted
        entirely when not applicable, matching Jinja's `in` test.
        """
        d: dict[str, Any] = {"role": self.role}
        if self.role == "assistant" and self.tool_calls:
            d["tool_calls"] = [tc.to_template_dict() for tc in self.tool_calls]
        else:
            d["content"] = self.content or ""
        if self.role == "tool" and self.name:
            d["name"] = self.name
        return d


class Conversation(BaseModel):
    """A single training example: metadata + the message list to render."""

    example_id: str
    task: str
    language: str
    messages: list[Message]
    # Free-form provenance for debugging/stats, never rendered into text.
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_template_messages(self) -> list[dict[str, Any]]:
        return [m.to_template_dict() for m in self.messages]

    def to_jsonl_record(self, rendered_text: str) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "task": self.task,
            "language": self.language,
            "messages": self.to_template_messages(),
            "text": rendered_text,
            "meta": self.meta,
        }


def tool_result_content(payload: dict[str, Any]) -> str:
    """Canonical JSON string used as the `content` of role='tool' messages."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
