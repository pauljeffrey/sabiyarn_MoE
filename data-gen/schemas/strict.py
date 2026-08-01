"""Convert a Pydantic model into an OpenAI Structured Outputs -compatible
strict JSON schema.

OpenAI's `strict: true` json_schema mode requires, recursively, on every
object: `additionalProperties: false` and `required` listing EVERY key
(optional fields are expressed as nullable via `type: [T, "null"]` rather
than being absent from `required`). Pydantic's default
`model_json_schema()` doesn't produce that shape (optional fields are just
missing from `required`, and $defs aren't inlined the way the API wants
for nested models), so we post-process it here rather than hand-writing
each schema twice.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def _tighten(node: dict[str, Any], defs: dict[str, Any]) -> None:
    # Resolve $ref by inlining (strict mode wants self-contained schemas).
    if "$ref" in node:
        ref_name = node["$ref"].rsplit("/", 1)[-1]
        resolved = copy.deepcopy(defs[ref_name])
        node.clear()
        node.update(resolved)

    for key in ("$defs", "definitions", "default"):
        node.pop(key, None)

    if node.get("type") == "object" or "properties" in node:
        props: dict[str, Any] = node.get("properties", {})
        for v in props.values():
            _tighten(v, defs)
        node["properties"] = props
        node["required"] = list(props.keys())
        node["additionalProperties"] = False

    if node.get("type") == "array" and "items" in node:
        _tighten(node["items"], defs)

    for combo_key in ("anyOf", "oneOf", "allOf"):
        if combo_key in node:
            for sub in node[combo_key]:
                _tighten(sub, defs)


def tighten_raw_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Same tightening as `to_strict_json_schema`, but for a hand-written
    plain JSON-Schema dict (no $ref/$defs expected) rather than a Pydantic
    model. Used to embed domain-specific extraction schemas (see
    schemas/structured_output_schemas.py) directly inside a larger strict
    response_format.
    """
    tightened = copy.deepcopy(schema)
    _tighten(tightened, {})
    return tightened


def object_response_format(properties: dict[str, Any], *, name: str) -> dict[str, Any]:
    """Build a strict response_format from a flat {field_name: schema} map,
    for callers assembling a response shape from pieces (e.g. combining a
    fixed envelope with a dynamically-selected nested schema) rather than a
    Pydantic model.
    """
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def to_strict_json_schema(model: type[BaseModel], *, name: str) -> dict[str, Any]:
    """Return a `response_format` value for the OpenAI Chat Completions API."""
    raw = model.model_json_schema()
    defs = raw.get("$defs", raw.get("definitions", {}))
    schema = copy.deepcopy(raw)
    _tighten(schema, defs)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }
