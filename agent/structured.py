"""Structured output: pydantic models or raw JSON Schema, no hard dependency.

pydantic is not in the root environment, so it is imported lazily and a plain
JSON-Schema dict works everywhere a model class does.

    class Contact(BaseModel):
        name: str
        phone: str

    agent.structured("Extract the contact: Ada, 0801...", schema=Contact)  # -> Contact
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "json_schema_of", "to_strict_json_schema", "validate_payload",
    "extract_json", "schema_instruction", "StructuredOutputError",
]


class StructuredOutputError(ValueError):
    """The model's JSON was missing, unparseable, or failed validation."""


def _is_pydantic(schema: Any) -> bool:
    return callable(getattr(schema, "model_json_schema", None)) and callable(
        getattr(schema, "model_validate", None)
    )


def json_schema_of(schema: Any) -> dict[str, Any]:
    """Accept a pydantic model class or a JSON-Schema dict; return the dict."""
    if isinstance(schema, dict):
        return schema
    if _is_pydantic(schema):
        return schema.model_json_schema()
    raise TypeError(f"schema must be a pydantic model or a JSON Schema dict, got {type(schema).__name__}")


def to_strict_json_schema(schema: Any, *, name: str | None = None) -> dict[str, Any]:
    """Tighten a schema the way strict structured output expects: every property
    required, no additional properties, $defs inlined.

    Inlining matters because the model sees this as text in a prompt -- a $ref
    pointing elsewhere in the document is one more thing for it to resolve, and
    a 300M model will not.
    """
    raw = json_schema_of(schema)
    defs = {**raw.get("$defs", {}), **raw.get("definitions", {})}
    out = _tighten(_inline(raw, defs, depth=0), top=True)
    out.pop("$defs", None)
    out.pop("definitions", None)
    if name or _is_pydantic(schema):
        out["title"] = name or getattr(schema, "__name__", "Response")
    return out


def _inline(node: Any, defs: dict[str, Any], depth: int) -> Any:
    """Replace $ref with its definition. Depth-capped: a self-referential model
    would otherwise recurse forever, and a truncated schema beats a hang."""
    if depth > 8:
        return {"type": "object"}
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            key = ref.rsplit("/", 1)[-1]
            target = defs.get(key)
            if target is not None:
                merged = {**_inline(target, defs, depth + 1)}
                merged.update({k: v for k, v in node.items() if k != "$ref"})
                return merged
        return {k: _inline(v, defs, depth + 1) for k, v in node.items() if k not in ("$defs", "definitions")}
    if isinstance(node, list):
        return [_inline(v, defs, depth + 1) for v in node]
    return node


def _tighten(node: Any, top: bool = False) -> Any:
    if isinstance(node, dict):
        out = {k: _tighten(v) for k, v in node.items()}
        if out.get("type") == "object" or "properties" in out:
            props = out.get("properties") or {}
            out.setdefault("type", "object")
            out["additionalProperties"] = False
            out["required"] = list(props)
        return out
    if isinstance(node, list):
        return [_tighten(v) for v in node]
    return node


SCHEMA_INSTRUCTION = (
    "Respond with ONLY a single JSON object that exactly matches the schema below. "
    "No prose, no markdown fence, no extra keys.\nJSON schema:\n{schema}"
)


def schema_instruction(schema: Any) -> str:
    """The system-prompt text that asks for schema-conforming JSON.

    Mirrors the structured-output task the model was trained on, which presents
    the schema as compact JSON after that wording.
    """
    strict = to_strict_json_schema(schema)
    return SCHEMA_INSTRUCTION.format(schema=json.dumps(strict, ensure_ascii=False, separators=(",", ":")))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Pull one JSON value out of model output.

    Tolerant by design: strips a markdown fence if present, else scans for the
    first balanced {...} or [...]. A 300M model will occasionally wrap JSON in a
    sentence, and failing the whole turn over a stray prefix wastes a round.
    """
    candidate = text.strip()
    fence = _FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth, in_str, escape = 0, False, False
        for i, ch in enumerate(candidate[start:], start):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise StructuredOutputError(f"no JSON object found in output: {text[:200]!r}")


def validate_payload(payload: Any, schema: Any) -> Any:
    """Validate parsed JSON against the schema.

    With a pydantic model, returns a model instance (full validation). With a
    dict schema, returns the payload after the same shallow checks the tool
    layer applies, so behaviour is predictable without pydantic installed.
    """
    if _is_pydantic(schema):
        try:
            return schema.model_validate(payload)
        except Exception as exc:
            raise StructuredOutputError(str(exc)) from exc
    from .tools import ToolError, _validate  # shallow checker, shared on purpose

    try:
        return _validate(json_schema_of(schema), payload, "response")
    except ToolError as exc:
        raise StructuredOutputError(str(exc)) from exc
