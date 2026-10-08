"""Canonical shape of an assistant turn, and the `output_structure` column.

    <|input_lang|><src>[<think>..</think>]<task_plan>[flag]<verb>[<label or <tag>]</task_plan><|target_lang|><tgt><response>CONTENT

The label token that names the kind of answer (<intent>, <|hate|>, <toxic>, <sentiment>, <topic>, <headline>, <lang_ID>,
<score>, <correct>; NOT <summary>) and the <tag> marker of tagging tasks live INSIDE <task_plan> next to the verb, so the plan alone says what
kind of answer follows. The tokens inside the plan are meant to be read as a set (the RL reward in rl/structure.py ignores their
order); the order written here is only the canonical one.

`output_structure` is the part of the turn the model must produce before the content: everything up to and including <response>.
"""

from __future__ import annotations

import json
import re

LABELS = ("<intent>", "<|hate|>", "<toxic>", "<sentiment>", "<topic>", "<headline>", "<lang_ID>", "<score>", "<correct>",
          "<reason>", "<answer>", "<title>")
# <summary> is dropped altogether: a summary is just <task_plan><summarize></task_plan> + the summary (a headline keeps <headline>
# because it is a different answer to the same verb)
DROPPED = ("<summary>",)
_TURN = re.compile(r"^(<\|input_lang\|>(?: unsure, but most likely )?(?:<[a-z]{2,4}>)+)(<think>.*?</think>)?<task_plan>(.*?)</task_plan>(<\|target_lang\|>(?: unsure, but most likely )?<[^<>]+>)<response>(.*)$", re.S)
_TOK = re.compile(r"<[^<>]+>")


def normalize_turn(content: str, task: str | None = None) -> str:
    """Move a leading label token (and <tag> for tagging tasks) from the response into the plan. Idempotent."""
    m = _TURN.match(content)
    if not m:
        return content
    head, think, plan_s, tail, rest = m.groups()
    plan = [t for t in _TOK.findall(plan_s) if t not in DROPPED]
    for d in DROPPED:
        if rest.startswith(d):
            rest = rest[len(d):]
    for lab in LABELS:
        if rest.startswith(lab):
            rest = rest[len(lab):]
            if lab not in plan:
                plan.append(lab)
            break
    if task == "pos":
        plan = ["<POS>" if t == "<NER>" else t for t in plan]
    if re.search(r"\S <tag> \S", rest) and "<tag>" not in plan:
        plan.append("<tag>")
    return f"{head}{think or ''}<task_plan>{''.join(plan)}</task_plan>{tail}<response>{rest}"


def output_structure(content: str) -> str:
    i = content.find("<response>")
    return content[: i + len("<response>")] if i >= 0 else ""


def normalize_messages_json(messages_json: str, task: str | None = None) -> tuple[str, str]:
    """-> (normalised messages json, output_structure of the last assistant turn)."""
    msgs = json.loads(messages_json)
    struct = ""
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m["content"] = normalize_turn(m["content"], task)
            struct = output_structure(m["content"])
    return json.dumps(msgs, ensure_ascii=False), struct
