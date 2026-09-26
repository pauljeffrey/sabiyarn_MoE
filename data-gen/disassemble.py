"""Marker string -> structured fields. The inverse of assemble.py.

assemble.py is the ONLY place marker strings are built; this is the only place they are taken apart. Three
callers need it, which is why it is a module rather than a helper inside a script:

  * scripts/repair_published.py -- 108 already-published records written in the old pre-assembled shape;
  * prompts._fewshot_block      -- the few-shot exemplars are finished training records, and rendering them
                                   verbatim into the prompt TAUGHT THE GENERATOR THE OLD SHAPE. Measured: every
                                   captured `non-tool turn must carry a non-empty response` drop was a
                                   low-resource language -- the only tier that gets exemplars -- returning
                                   "<|input_lang|><ful><think>...</think>..." in `content` instead of fields;
  * postprocess_gen             -- recovery for a generator that emits the old shape anyway.

Every recovery here is DETERMINED, never guessed. A record whose markers contradict each other (a language-pair
prefix asserting one target while <|target_lang|> says another) is dropped, because there is no way to know
which the generator meant.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional

from assemble import LABEL_TOKENS, VALID_LANGS, VALID_VERBS

# Counts what was repaired and what was dropped. repair_published.py reports it.
STATS: Counter = Counter()

_LANG_AFTER_MARKER = re.compile(r"<\|(?:input_lang|target_lang)\|>\s*<([a-z]{2,4})>")
_PSEUDO_LANG = re.compile(r"<\|([a-z]{2,4})\|?>")
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_PLAN = re.compile(r"<task_plan>(.*?)</task_plan>", re.S)
_RESPONSE = re.compile(r"<response>(.*)", re.S)
_VERB = re.compile(r"<\|?[A-Za-z_][A-Za-z0-9_|]*\|?>")
_LABEL_AT_START = re.compile(r"^\s*(<[a-z_A-Z]{2,12}>)")
# "pcm->igbo", "pcm -> igbo", "<lang>pcm<target_lang>igbo"
_LANGPAIR_JUNK = re.compile(
    r"^\s*(?:<lang>)?\s*([a-z]{2,4})\s*(?:->|<target_lang>)\s*([a-zA-Z]{2,10})\s*")


def _lang_code(name: str) -> Optional[str]:
    """'igbo' -> 'ibo', 'yoruba' -> 'yor'. Only names we can map unambiguously."""
    n = name.strip().lower()
    if n in VALID_LANGS:
        return n
    return {"igbo": "ibo", "yoruba": "yor", "hausa": "hau", "english": "eng", "pidgin": "pcm",
            "efik": "efi", "urhobo": "urh", "twi": "twi", "akan": "aka", "fon": "fon", "ewe": "ewe",
            "fulah": "ful", "fulfulde": "fuv"}.get(n)


def _langs_from_content(c: str, fallback: str) -> tuple[str, Optional[str]]:
    """(input_lang, target_lang) recovered from however the generator wrote them."""
    proper = _LANG_AFTER_MARKER.findall(c)
    pseudo = [g for g in _PSEUDO_LANG.findall(c) if g in VALID_LANGS]
    found = [g for g in (proper + pseudo) if g in VALID_LANGS]
    if not found:
        return fallback, fallback
    src = found[0]
    tgt = found[1] if len(found) > 1 else None
    return src, tgt


def _verbs(c: str) -> list[str]:
    m = _PLAN.search(c)
    if not m:
        return []
    return [v for v in _VERB.findall(m.group(1)) if v in VALID_VERBS]


def _plan_for_tool(name: str) -> list[str]:
    """A tool-calling turn that never carried a plan: infer one from the tool it called."""
    by_tool = {
        "search_documents": ["<|RAG|>"], "search_internet": ["<|explain|>"],
        "lookup_health_guidance": ["<|recommend|>"], "find_health_facility": ["<|recommend|>"],
        "calculate": ["<|math|>"], "run_statistics": ["<|math|>"], "convert_units": ["<|math|>"],
        "get_exchange_rate": ["<|math|>"], "get_market_prices": ["<|analyze|>"],
        "search_db": ["<|data_extract|>"], "db_get_record": ["<|data_extract|>"],
        "db_insert_record": ["<|plan|>"], "send_message": ["<|plan|>"], "set_reminder": ["<|plan|>"],
        "translate_text": ["<translate>"], "get_weather_forecast": ["<|explain|>"],
        "get_transport_route": ["<|plan|>"], "lookup_crop_guidance": ["<|recommend|>"],
        "lookup_legal_info": ["<|explain|>"], "get_current_datetime": ["<|chat|>"],
    }
    return by_tool.get(name, ["<|plan|>"])


SEQUENCE_TASKS = {"ner", "pos_tagging", "token_labelling"}


def _turn_from_assistant(m: dict, lang: str, sequence: bool = False) -> Optional[dict]:
    c = m.get("content") or ""
    calls = m.get("tool_calls") or []
    src, tgt = _langs_from_content(c, lang)
    think = None
    tm = _THINK.search(c)
    if tm and tm.group(1).strip():
        think = tm.group(1).strip()
    verbs = _verbs(c)

    if calls:
        fn = (calls[0] or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            STATS["drop_tool_call_without_name"] += 1
            return None
        if not verbs:
            verbs = _plan_for_tool(name)
            STATS["fixed_inferred_plan_for_tool_turn"] += 1
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        return {"role": "assistant", "input_lang": src, "think": think, "task_plan": verbs,
                "tool_call": {"name": name, "arguments": args or {}}}

    rm = _RESPONSE.search(c)
    if not rm:
        STATS["drop_no_response_text"] += 1
        return None
    body = rm.group(1)

    # A language-pair prefix ASSERTS a target language. If it contradicts what the markers said, the record
    # is unrecoverable: we cannot know which the generator meant.
    jm = _LANGPAIR_JUNK.match(body)
    label = None
    if jm:
        asserted = _lang_code(jm.group(2))
        if asserted and tgt and asserted != tgt:
            STATS["drop_contradictory_target_language"] += 1
            return None
        if asserted:
            tgt = asserted
            STATS["fixed_target_from_langpair_prefix"] += 1
        body = body[jm.end():]
    # A label token sitting after <response> is a real label in the wrong place.
    lm = _LABEL_AT_START.match(body)
    if lm:
        tok = lm.group(1)
        key = next((k for k, v in LABEL_TOKENS.items() if v == tok), None)
        if key:
            label = key
            body = body[lm.end():]
            STATS["fixed_label_token_moved"] += 1
        else:
            body = body[lm.end():]
            STATS["fixed_stray_tag_removed"] += 1
    body = body.strip()
    if not body:
        STATS["drop_empty_response_after_cleanup"] += 1
        return None
    if not verbs:
        verbs = ["<|chat|>"]
        STATS["fixed_default_plan_for_answer_turn"] += 1
    return {"role": "assistant", "input_lang": src, "target_lang": tgt or src, "think": think,
            "task_plan": verbs, "response": body, "label_token": label,
            "sequence_labels": sequence}



def turn_from_assistant(message: dict, lang: str, sequence: bool = False) -> Optional[dict]:
    """Public name. `message` is a template-shaped assistant message whose `content` carries markers."""
    return _turn_from_assistant(message, lang, sequence)


def looks_pre_assembled(content: object) -> bool:
    """Is this the old finished-string shape rather than plain text?"""
    return isinstance(content, str) and ("<|input_lang|>" in content or "<response>" in content)
