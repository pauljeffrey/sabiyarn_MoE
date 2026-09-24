"""JSON schemas the generating model's output must satisfy.

Used two ways:
  * API path -- as `response_format={"type": "json_schema", ...}` where the provider supports it.
  * vLLM path -- as `guided_json`, which CONSTRAINS DECODING: the sampler can only emit tokens that keep the
    output a valid instance of the schema. Malformed JSON becomes structurally impossible rather than merely
    discouraged, which removes the single largest drop reason (`json_invalid`) outright.

Kept deliberately shallow. Grammar-constrained decoding compiles the schema into a state machine, and deeply
nested `oneOf`/`allOf` blows up compile time for no benefit here. `arguments` is an open object on purpose --
each tool has its own parameter schema, and enforcing 20 different ones inside one grammar is not worth the
compile cost when postprocess_gen.py validates arguments against the real tool schema anyway.
"""

from __future__ import annotations

from typing import Any

_CONF = {"type": "number", "minimum": 0, "maximum": 1,
         "description": "Honest estimate that this sample is correct and fluent in the target language."}

_MESSAGE = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
        "content": {"type": "string"},
        "name": {"type": "string", "description": "Tool name; role='tool' only."},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                        "required": ["name", "arguments"],
                    }
                },
                "required": ["function"],
            },
        },
    },
    "required": ["role"],
}

PRETRAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "text": {"type": "string"},
        "language_self_check": {"type": "boolean"},
        "confidence": _CONF,
    },
    "required": ["title", "text", "language_self_check", "confidence"],
}

SFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "messages": {"type": "array", "items": _MESSAGE, "minItems": 4},
        "tasks": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": _CONF,
    },
    "required": ["messages", "tasks", "tags", "confidence"],
}

RL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt_messages": {"type": "array", "items": _MESSAGE, "minItems": 3},
        "responses": {
            "type": "array", "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "quality": {"type": "string", "enum": ["best", "partial", "worst"]},
                    "why": {"type": "string"},
                },
                "required": ["content", "quality", "why"],
            },
        },
        "tasks": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": _CONF,
    },
    "required": ["prompt_messages", "responses", "tasks", "tags", "confidence"],
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "loser": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "deciding_criterion": {"type": "integer", "minimum": 1, "maximum": 6},
        "why": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "both_bad": {"type": "boolean"},
    },
    "required": ["winner", "loser", "deciding_criterion", "why", "confidence", "both_bad"],
}

SCHEMAS = {"pretrain": PRETRAIN_SCHEMA, "sft": SFT_SCHEMA, "rl": RL_SCHEMA, "judge": JUDGE_SCHEMA}


def schema_for(kind: str) -> dict[str, Any]:
    if kind not in SCHEMAS:
        raise KeyError(f"no output schema for {kind!r}; have {sorted(SCHEMAS)}")
    return SCHEMAS[kind]
