"""One intermediate record -> one finished row of the combined SFT table.

A converter yields `Rec`s (what the task is, which languages, the pieces of text). `finish` turns a Rec into the row that is
published: it composes the user prompt, assembles the assistant turn with data-gen/assemble.py (so a malformed turn raises
instead of being written), renders the chat template, and derives `output_structure` and the dedupe hash.

Columns of the combined table (BeardedMonster/sft-all):
  id, task, source_dataset, source_lang, targ_lang, language (= source_lang, kept for the older repos), instruction, input,
  response, messages (json), text (chat-template string, what is tokenized), output_structure, metadata (json)
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Optional

from africomet_translation.instructions import Instruction, compose_prompt
from sft_new import structured as S
from sft_new.grammar import normalize_turn, output_structure
from sft_new.render import render
from sft_new.textutil import blank

A = S.A
A.set_response_budget(24_000)                       # SFT now runs at 64k: a full story or article fits one response
A.LABEL_TOKENS.update({"hate": "<|hate|>", "toxic": "<toxic>", "sentiment": "<sentiment>", "topic": "<topic>", "intent": "<intent>",
                       "tag": "<tag>"})
A.VALID_VERBS |= {"<POS>"}                           # token 52116 (chunking has no token of its own: it is written <classify><tag>)

COLUMNS = ["id", "task", "source_dataset", "source_lang", "targ_lang", "language", "instruction", "input", "response", "messages",
           "text", "output_structure", "metadata"]


@dataclass
class Rec:
    task: str
    dataset: str
    src: str                                   # language of the input text (a code from langdetect.SUPPORTED)
    tgt: str                                   # language the response is written in
    response: str
    input: str = ""
    instruction: str = ""                      # natural-language request; "" when the source already carries the whole prompt in `input`
    position: str = "prepend"                  # where the instruction goes relative to the input
    plan: list = field(default_factory=lambda: ["<translate>"])
    label: Optional[str] = None                # label token key for A.assistant_content (sentiment, topic, intent, ...)
    think: Optional[str] = None
    unsure_src: bool = False
    unsure_tgt: bool = False
    sequence_labels: bool = False
    system: Optional[str] = None
    input_langs: Optional[list] = None         # every language of the user turn in reading order; derived when None
    lang_hints: Optional[list] = None          # languages the source states for a prompt_override turn (instruction language, text language)
    tools: Optional[list] = None               # function definitions ({"type": "function", "function": {...}}); rendered by the template inside the system turn
    convo: Optional[list] = None               # a ready multi-message conversation (tool calls); then plan/label/response below are not used to build turns
    prompt_override: Optional[str] = None      # the user turn exactly as the source had it (instruction + text already combined)
    meta: dict = field(default_factory=dict)   # goes to the `metadata` column (instruction_language, direction, quality, ...)


@dataclass
class Late:
    """Plain text that could not become an SFT row (task not certain) but is worth having for pretraining. Written to pretrain-new/late/<lang>/
    so the main bin is not touched until the owner decides to include it."""
    text: str
    lang: str
    source: str
    score: float = 5.0


class Skip(Exception):
    """A record that must not be written (the reason is counted in the run's stats)."""


def prompt_of(rec: Rec, rng: random.Random) -> str:
    if rec.prompt_override:
        return rec.prompt_override
    if rec.instruction and rec.input:
        return compose_prompt(Instruction(rec.instruction, rec.position, rec.meta.get("instruction_language", "eng"), "n/a"), rec.input, rng)
    return rec.instruction or rec.input


def dedupe_key(rec: Rec, msgs: list) -> str:
    """What makes two rows 'the same example' for exact deduplication: task, languages, the text the request is about (or, when the source gave
    the whole prompt, the prompt itself) and the answer. The instruction wording is NOT part of it for templated rows, so one pair found in two
    sources collapses even though each got a different random template; it IS part of it for prompt-only rows, where it cannot be separated."""
    first_user = next((blank(m.get("content")) for m in msgs if m["role"] == "user"), "")
    about = blank(rec.input) or first_user
    return "\x1f".join((rec.task, rec.src, rec.tgt, about, blank(rec.response)))


def _row(rec: Rec, msgs: list, content: str) -> dict:
    h = hashlib.blake2b(dedupe_key(rec, msgs).encode(), digest_size=8).hexdigest()
    return {"id": h, "task": rec.task, "source_dataset": rec.dataset, "source_lang": rec.src, "targ_lang": rec.tgt, "language": rec.src,
            "instruction": rec.instruction, "input": rec.input, "response": rec.response.strip(),
            "messages": json.dumps(msgs, ensure_ascii=False), "tools": json.dumps(rec.tools, ensure_ascii=False) if rec.tools else "",
            "text": render(msgs, tools=rec.tools or None), "output_structure": output_structure(content),
            "metadata": json.dumps({**rec.meta, "unsure_src": rec.unsure_src, "unsure_tgt": rec.unsure_tgt}, ensure_ascii=False)}


def turn_languages(rec: Rec, prompt: str) -> list:
    """The <|input_lang|> field: every language present in the user's turn, in reading order, once each (never more than MAX_INPUT_LANGS)."""
    if rec.input_langs:
        langs = list(rec.input_langs)
    elif rec.prompt_override:
        from sft_new import langdetect as L
        langs = L.prompt_languages(prompt, rec.lang_hints or [rec.src], max_langs=A.MAX_INPUT_LANGS)
    else:
        il = rec.meta.get("instruction_language")
        if rec.instruction and rec.input and il and il != rec.src:
            langs = [il, rec.src] if rec.position != "append" else [rec.src, il]
        else:
            langs = [il if (il and not rec.input) else rec.src]
    if rec.src not in langs:
        langs.append(rec.src)
    return list(dict.fromkeys(langs))[:A.MAX_INPUT_LANGS]


def finish(rec: Rec, rng: random.Random) -> dict:
    # Many sources have an instruction and an EMPTY or null input (the request carries its own text), or the reverse. Normalise once, here, so every
    # converter gets it: an empty field is "", never the string 'None', and the prompt is then just the field that exists (see prompt_of).
    if not rec.convo:
        rec.input, rec.instruction = blank(rec.input), blank(rec.instruction)
        if blank(rec.response) == "":
            raise Skip("empty_response")
    if rec.convo:
        rec.input, rec.instruction, rec.response = blank(rec.input), blank(rec.instruction), blank(rec.response)
        for m in rec.convo:                                              # a null content is an empty string (legal on a tool-call turn), never None
            if m.get("content") is None:
                m["content"] = ""
        last = next((m["content"] for m in reversed(rec.convo) if m["role"] == "assistant" and "tool_calls" not in m), "")
        if not blank(last):
            raise Skip("empty_response")
        if not any(m["role"] == "user" and blank(m["content"]) for m in rec.convo):
            raise Skip("empty_prompt")
        return _row(rec, rec.convo, last)
    if not rec.response or not rec.response.strip():
        raise Skip("empty_response")
    prompt = prompt_of(rec, rng)
    if not prompt.strip():
        raise Skip("empty_prompt")
    try:
        langs = turn_languages(rec, prompt)
        content = A.assistant_content(input_lang=langs, target_lang=rec.tgt, think=rec.think, task_plan=list(rec.plan),
                                      response=rec.response.strip(), label_token=rec.label, sequence_labels=rec.sequence_labels,
                                      unsure_input=rec.unsure_src, unsure_target=rec.unsure_tgt)
    except A.AssemblyError as e:
        raise Skip("assembly:" + str(e)[:40])
    content = normalize_turn(content, rec.task)
    msgs = ([{"role": "system", "content": rec.system}] if rec.system else []) + [{"role": "user", "content": prompt},
                                                                                   {"role": "assistant", "content": content}]
    rec.meta = {**rec.meta, "input_langs": langs}
    return _row(rec, msgs, content)
