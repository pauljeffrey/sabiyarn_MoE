"""Tool (function) schema definitions, in standard OpenAI function-calling
JSON-schema shape: {"type": "function", "function": {name, description,
parameters}}.

These serve two purposes:
  1. They are embedded verbatim into the `system` message content of
     training examples (the chat template has no dedicated "tools" slot,
     so the convention here is a plain-text JSON block -- see
     `format_tools_for_system_prompt`), so the model learns to read a
     tool list from its system prompt the same way at train and inference
     time.
  2. Generators pass them to the meta-model (GPT) as the actual
     `tools=` argument during data generation, so the meta-model's own
     native function-calling produces realistic, schema-valid call
     arguments that we then transcribe into our structured output.

Keep descriptions short and unambiguous -- the 280M target model has to
learn tool selection from relatively few examples, so vague/overlapping
descriptions directly hurt it.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

SEARCH_DOCUMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_document",
        "description": (
            "Search the document provided in the conversation for passages "
            "relevant to a query. Use this before answering any question "
            "whose answer might be in the document, rather than guessing. "
            "Returns the most relevant chunk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A short, specific search query describing the "
                        "information needed, in the same language as the "
                        "conversation or in English."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Math / statistics
# ---------------------------------------------------------------------------

CALCULATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a numeric arithmetic expression and return the exact "
            "result. Use this for any non-trivial arithmetic instead of "
            "computing it yourself, since manual arithmetic is error-prone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A mathematical expression using digits and the "
                        "operators + - * / ^ ( ), e.g. '(120000 - 45000) / 3'."
                    ),
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Edge-device action tools
# ---------------------------------------------------------------------------
# A curated set of realistic on-device assistant actions. Generators sample
# a random subset (the correct tool + distractors) as the "available tools"
# for a given example, to teach tool *selection* as well as tool *use*.

EDGE_DEVICE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set a device alarm for a specific time, optionally with a label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "24-hour time, e.g. '06:30'."},
                    "label": {"type": "string", "description": "Optional short label for the alarm."},
                },
                "required": ["time", "label"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Start a countdown timer for a duration, optionally with a label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {"type": "integer", "description": "Timer length in seconds."},
                    "label": {"type": "string", "description": "Optional short label for the timer."},
                },
                "required": ["duration_seconds", "label"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Create a reminder for a future date/time with a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime": {"type": "string", "description": "ISO-like date/time, e.g. '2026-08-02 09:00'."},
                    "message": {"type": "string", "description": "What to be reminded about."},
                },
                "required": ["datetime", "message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send a text message to a contact by name or phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact": {"type": "string", "description": "Contact name or phone number."},
                    "message": {"type": "string", "description": "Message body to send."},
                },
                "required": ["contact", "message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_call",
            "description": "Place a phone call to a contact by name or phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact": {"type": "string", "description": "Contact name or phone number."},
                },
                "required": ["contact"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_setting",
            "description": "Turn a device setting on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "setting_name": {
                        "type": "string",
                        "enum": ["wifi", "bluetooth", "airplane_mode", "flashlight", "silent_mode", "data_saver", "location"],
                        "description": "Which setting to change.",
                    },
                    "state": {"type": "boolean", "description": "True to turn on, false to turn off."},
                },
                "required": ["setting_name", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the device volume level for a given audio stream.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Volume level from 0 (mute) to 100 (max)."},
                    "stream": {
                        "type": "string",
                        "enum": ["media", "ringtone", "alarm", "call"],
                        "description": "Which volume stream to change.",
                    },
                },
                "required": ["level", "stream"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_brightness",
            "description": "Set the screen brightness level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Brightness level from 0 to 100."},
                },
                "required": ["level"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application on the device by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the app to open, e.g. 'Camera', 'WhatsApp'."},
                },
                "required": ["app_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_battery_status",
            "description": "Get the current battery percentage and charging state.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_time",
            "description": "Get the device's current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_note",
            "description": "Save a short text note on the device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note content to save."},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_device",
            "description": "Lock the device screen immediately.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_photo",
            "description": "Open the camera and capture a photo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "camera": {
                        "type": "string",
                        "enum": ["front", "back"],
                        "description": "Which camera to use.",
                    },
                },
                "required": ["camera"],
                "additionalProperties": False,
            },
        },
    },
]

EDGE_DEVICE_TOOLS_BY_NAME: dict[str, dict[str, Any]] = {
    t["function"]["name"]: t for t in EDGE_DEVICE_TOOLS
}


def format_tools_for_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Text block describing available tools, embedded in the system prompt.

    Convention: a labeled JSON array. Keep this EXACT function as the only
    place that decides the textual convention, so every generator and the
    eventual inference-side system prompt builder stay in sync.
    """
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"You have access to the following tools:\n{tools_json}"
