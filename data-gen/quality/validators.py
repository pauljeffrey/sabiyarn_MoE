"""Validation helpers used during postprocessing.

Kept dependency-free (no `jsonschema` package) since our tool schemas are
shallow enough that a small recursive checker is simpler than pulling in a
full JSON Schema implementation. This validates the model-invented tool
call arguments and tool results for the edge_action task -- the one place
generation produces JSON we didn't get to constrain via Structured Outputs
(the shape varies per selected tool, see generators/edge_action.py).
"""

from __future__ import annotations

import re
from typing import Any

# Diacritics/letters that are distinctive enough in these languages that
# their near-total absence over a longish text is a useful (soft) signal
# something may have been generated in the wrong language. This is a cheap
# heuristic, not a language identifier -- false negatives (text that
# happens not to need these letters) are expected and fine; it's meant to
# catch gross mismatches (e.g. plain English text) for manual spot-check,
# not to hard-reject examples.
DISTINCTIVE_CHARS: dict[str, str] = {
    "yor": "ẹọṣàáèéìíòóùúẹ́ẹ̀ọ́ọ̀",
    "ibo": "ịọụ",
    "hau": "ɓɗƙ",
    "ewe": "ɖɣʋŋɔɛ",
    "twi": "ɔɛ",
    "aka": "ɔɛ",
    "fon": "ɖɛɔ",
}

MIN_CHECK_LEN = 40  # don't bother flagging very short strings


def check_json_schema(value: Any, schema: dict[str, Any], *, path: str = "$") -> list[str]:
    """Minimal structural check against a plain JSON-Schema dict.

    Supports the subset our tool schemas actually use: type, enum,
    properties/required (one or two levels deep), items. Returns a list of
    human-readable error strings; empty list means it validated.
    """
    errors: list[str] = []
    expected_type = schema.get("type")

    type_ok = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
    }
    if expected_type and expected_type in type_ok and not type_ok[expected_type](value):
        errors.append(f"{path}: expected type {expected_type!r}, got {type(value).__name__}")
        return errors  # further checks would be meaningless

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")

    if expected_type == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        for req_key in schema.get("required", []):
            if req_key not in value:
                errors.append(f"{path}: missing required key {req_key!r}")
        for key, sub_schema in props.items():
            if key in value:
                errors.extend(check_json_schema(value[key], sub_schema, path=f"{path}.{key}"))

    if expected_type == "array" and isinstance(value, list) and "items" in schema:
        for idx, item in enumerate(value):
            errors.extend(check_json_schema(item, schema["items"], path=f"{path}[{idx}]"))

    return errors


def check_tool_arguments(tool_def: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    params_schema = tool_def["function"]["parameters"]
    if not isinstance(arguments, dict):
        return [f"arguments must be a JSON object, got {type(arguments).__name__}"]
    return check_json_schema(arguments, params_schema, path=f"$({tool_def['function']['name']})")


_WHITESPACE_RE = re.compile(r"\s+")


def language_sanity_warnings(text: str, language_code: str) -> list[str]:
    warnings: list[str] = []
    stripped = _WHITESPACE_RE.sub(" ", text).strip()
    if not stripped:
        warnings.append("empty or whitespace-only text")
        return warnings

    chars = DISTINCTIVE_CHARS.get(language_code)
    if chars and len(stripped) >= MIN_CHECK_LEN:
        if not any(ch in stripped.lower() for ch in chars):
            warnings.append(
                f"no distinctive {language_code} characters ({chars}) found in "
                f"a {len(stripped)}-char text -- possible language mismatch, "
                f"recommend spot-check"
            )
    return warnings


def min_length_warnings(text: str, *, min_chars: int = 2, field: str = "text") -> list[str]:
    if len(text.strip()) < min_chars:
        return [f"{field} shorter than {min_chars} chars"]
    return []
