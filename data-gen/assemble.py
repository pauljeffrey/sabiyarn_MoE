"""Assemble assistant-turn content from structured fields. The ONLY place marker strings are built.

The generator used to be asked for the finished string --
`<|input_lang|><yor><think>..</think><task_plan>..</task_plan><|target_lang|><yor><response>..` -- and got it
wrong constantly. Measured over 108 published records: 88 turns missing `<|input_lang|>`, 67 missing
`<|target_lang|>`, ~150 pseudo-markers (`<|pcm>`, `<|eng|>`, `<|ful>`), 78 junk tokens after `<response>`
(`<tag>`, `<sentiment>`, `<lang>pcm<target_lang>igbo`), and 114 tool-call turns with no `<task_plan>` at all.

So it is no longer asked. It returns FIELDS:

    {"think": "...English or null...", "task_plan": ["<|explain|>"], "input_lang": "eng",
     "target_lang": "pcm", "response": "the answer", "tool_call": {"name": ..., "arguments": {...}}}

and this module builds the string. Every one of those failure classes becomes structurally impossible rather
than merely discouraged, because the model never touches a marker. Field values are still validated -- an
unknown task_plan verb or language code is rejected -- but the ORDER and SPELLING of the scaffolding is ours.

Canonical order, which is also the order a turn actually happens in:

    <|input_lang|><src>  [<think>…</think>]  <task_plan>…</task_plan>  then either
        <|target_lang|><tgt><response>TEXT        (a turn that answers)
        (nothing more)                            (a turn that calls a tool)

`<think>` is optional per turn: a simple translation or a label does not need one, and emitting it everywhere
wasted tokens in 80 of 108 records. `<task_plan>` comes BEFORE a tool call, because the plan is what decides
to call it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# Verbs that exist in the tokenizer. A task_plan entry outside this set is a hard error.
VALID_VERBS = {
    "<|chat|>", "<|generate|>", "<|edit|>", "<|data_extract|>", "<|math|>", "<|JSON|>", "<|code|>",
    "<|plan|>", "<|analyze|>", "<|recommend|>", "<|explain|>", "<|debug|>", "<|RAG|>",
    "<summarize>", "<translate>", "<classify>", "<NER>", "<identify>", "<qa>", "<diacritize>", "<clean>",
}
# Language tags that exist AND are below the model's vocab_size (52050); everything from <ibb> (52050) up
# can never be embedded.
VALID_LANGS = {"eng", "yor", "hau", "ibo", "efi", "urh", "twi", "fon", "pcm", "ewe", "aka", "ful", "fuv"}
# Beyond West Africa. Their tags exist in the tokenizer but ALL sit above vocab_size 52050 and so cannot be
# embedded yet (<zu> 52059, <sn> 52066, <som> 52075, <amh> 52079, <kin> 52080, <fra> 52089, <orm> 52112), and
# Swahili has no tag at all. The generated TEXT is unaffected -- only the marker is -- so the corpus can be
# built now and becomes trainable once the embedding is resized.
PENDING_VOCAB_LANGS = {"swa", "fra", "zul", "amh", "som", "orm", "kin", "sna"}
VALID_LANGS |= PENDING_VOCAB_LANGS
# Label tokens a classification answer may legitimately open with.
LABEL_TOKENS = {"sentiment": "<sentiment>", "topic": "<topic>", "intent": "<intent>", "toxic": "<toxic>",
                "lang_id": "<lang_ID>", "ner": "<tag>", "answer": "<answer>", "summary": "<summary>",
                "title": "<title>", "headline": "<headline>"}

# Exactly the tokens that exist, so `a < b` in ordinary prose is never mistaken for markup.
_KNOWN_TOKENS = (
    sorted(VALID_VERBS)
    + [f"<{c}>" for c in sorted(VALID_LANGS)]
    + sorted(LABEL_TOKENS.values())
    + ["<|input_lang|>", "<|target_lang|>", "<response>", "<think>", "</think>",
       "<task_plan>", "</task_plan>", "<tool_call>", "</tool_call>",
       "<tool_response>", "</tool_response>", "<context>", "</context>",
       "<|system|>", "<|user|>", "<|assistant|>"]
)
# Anything TAG-SHAPED in a plain-text field: `<` then a word then `>`, no spaces. This catches the real
# tokens, the malformed ones the generator invents (<|pcm>, <|eng|>, <|ful> -- the unterminated variant is why
# an earlier validator missed 35 occurrences, its regex demanded the closing "|>"), and the ones that are not
# tokens at all but still corrupt the text (<lang>, <target_lang>, <person>, <location>).
# It deliberately does NOT match ordinary prose or arithmetic: "a < b", "x <= y" and "3 < 4 and 5 > 2" all
# have a space or a non-word character after the "<".
_TAG_SHAPED = re.compile(r"<\|?/?[A-Za-z_][A-Za-z0-9_.|-]{0,22}\|?>")


# A single assistant response is capped at MAX_RESPONSE_TOKENS. Enforced in CHARACTERS, because counting real
# tokens here would pull the tokenizer into post-processing, and the character budget differs sharply by
# language: measured with tiktoken o200k on real text, Pidgin runs ~1.35 tokens/word while Yoruba runs ~2.5,
# so the same token budget is roughly 20k characters of Pidgin but only 11k of Yoruba. Budget =
# MAX_RESPONSE_TOKENS x (CHARS_PER_WORD / tokens_per_word).
#
# PHASE-DEPENDENT, not a constant: budgets.py derives it from the target model's context (4,096 for SFT at a
# 32,768 context, 2,048 for RL where two candidates share the window). The drivers call set_response_budget()
# once at startup; the default here is the largest case so an importer that forgets cannot accidentally
# tighten it.
from budgets import budget_for  # noqa: E402

MAX_RESPONSE_TOKENS = budget_for("sft").max_response_tokens


def set_response_budget(tokens: int) -> None:
    """Set the per-response ceiling for this process. Called by generate.py / vllm_gen.py from the phase."""
    global MAX_RESPONSE_TOKENS
    MAX_RESPONSE_TOKENS = int(tokens)


_CHARS_PER_WORD = 5.5
# MEASURED with the TARGET tokenizer (BeardedMonster/SabiYarn-32k) over ~73,000 words of real generated text
# from this pipeline -- SFT responses on native-target turns plus every pretraining document.
#
# The previous table was calibrated against a generic tokenizer and was wrong by up to 2x, because SabiYarn-32k
# is trained ON these languages and is far more efficient on them than a general-purpose vocabulary. Yoruba was
# assumed at 2.50 and measures 1.31; Igbo 2.00 -> 1.32; Hausa 1.80 -> 1.30; Urhobo 2.70 -> 1.53. The visible
# symptom was documents landing at half their requested token size: asked for ~1,000 tokens, a pilot produced a
# median of 470.
#
# Sample sizes: pcm 23.8k words, yor 11.4k, hau 9.1k, ibo 8.3k, twi 6.4k, aka 4.7k, eng 2.5k, fon 2.2k,
# ewe 1.6k, ful 1.3k, fuv 0.9k, efi 0.7k, urh 0.7k. The last four are thin -- re-measure them once there is
# more text, with scripts/measure_tokens_per_word.py.
_TOKENS_PER_WORD = {"pcm": 1.24, "eng": 1.28, "hau": 1.30, "ibo": 1.32, "yor": 1.31, "twi": 2.11, "aka": 2.06,
                    "efi": 2.05, "urh": 1.53, "ewe": 2.50, "fon": 2.63, "ful": 1.99, "fuv": 1.99,
                    # MEASURED on real generated text (73 pretraining documents from gemma-3-27b), not on a
                    # single sentence -- the first estimates were within ~20% but these are what the budgets
                    # spend. All are worse than the West African languages because the tokenizer was not trained
                    # on them, and Amharic is in a different category: its Ethiopic script falls back to bytes,
                    # so one Amharic word costs 6.1 Pidgin words. At a 1,024-token budget that leaves an Amharic
                    # document ~75 words long, which is not worth generating.
                    "fra": 2.09, "som": 2.39, "swa": 2.45, "orm": 2.69, "kin": 2.90, "sna": 3.12,
                    "zul": 3.55, "amh": 7.55}


def max_response_chars(lang: str) -> int:
    return int(MAX_RESPONSE_TOKENS * _CHARS_PER_WORD / _TOKENS_PER_WORD.get(lang, 2.5))


class AssemblyError(ValueError):
    """The fields could not be turned into a valid turn. The sample is dropped, never patched."""


# Token-sequence tasks. NER and POS answers are legitimately a stream of `word <tag> LABEL` pairs, so <tag>
# is real content there rather than a leaked marker. Nothing else is exempt.
SEQUENCE_TOKENS = {"<tag>"}


def _check_plain(text: str, field: str, allow: Optional[set[str]] = None) -> str:
    """Text fields are PLAIN TEXT by contract. A special token inside one means the generator ignored the
    contract, so the sample is REJECTED rather than repaired.

    Stripping was tried and is worse: `<lang>pcm<target_lang>yorubaOwo tan.` strips down to
    `pcmyorubaOwo tan.`, which is silently corrupted text that then looks valid. Dropping the sample is the
    honest outcome, and with field-based generation the model has no reason to emit a marker here at all.
    """
    t = (text or "").strip()
    if not t:
        return t
    allow = allow or set()
    for m in _TAG_SHAPED.finditer(t):
        tok = m.group(0)
        if tok in allow:
            continue
        kind = "special token" if tok in _KNOWN_TOKENS else "stray tag"
        raise AssemblyError(f"{field} contains the {kind} {tok!r}; text fields must be plain text")
    return t


# <|summarize|> vs <summarize>: the generator picks the wrong bracket form for a verb that does exist. The
# intent is unambiguous, so the form is normalised rather than the sample dropped.
_VERB_BY_BARE = {v.strip("<>|"): v for v in VALID_VERBS}


def normalise_verb(v: str) -> str:
    bare = (v or "").strip().strip("<>").strip("|")
    return _VERB_BY_BARE.get(bare, v)


# Several verbs concatenated into ONE list element: task_plan ["<|RAG|><|chat|>"] instead of
# ["<|RAG|>", "<|chat|>"]. The intent is unambiguous when the tokens tile the string exactly with nothing
# between them, so it is split rather than the sample dropped. Anything with prose mixed in is left alone and
# fails validation, because there the intent is NOT clear.
_VERB_TOKEN = re.compile(r"<\|[^|<>]+\|>|<[A-Za-z_][A-Za-z0-9_]*>")


def split_verbs(items: list[str]) -> list[str]:
    out: list[str] = []
    for raw in items or []:
        s = (raw or "").strip()
        if not s:
            continue
        found = _VERB_TOKEN.findall(s)
        out.extend(found) if len(found) > 1 and "".join(found) == s else out.append(s)
    return out


def assistant_content(
    *,
    input_lang: str,
    target_lang: Optional[str] = None,
    think: Optional[str] = None,
    task_plan: Optional[list[str]] = None,
    response: Optional[str] = None,
    label_token: Optional[str] = None,
    is_tool_call: bool = False,
    sequence_labels: bool = False,
) -> str:
    """Build one assistant turn's content. Raises AssemblyError on anything it cannot build correctly."""
    if input_lang not in VALID_LANGS:
        raise AssemblyError(f"input_lang {input_lang!r} is not a supported language tag")
    parts = [f"<|input_lang|><{input_lang}>"]

    if think:
        t = _check_plain(think, "think")
        if t:
            parts.append(f"<think>{t}</think>")

    verbs = [normalise_verb(v) for v in split_verbs(task_plan or []) if v]
    bad = [v for v in verbs if v not in VALID_VERBS]
    if bad:
        raise AssemblyError(f"task_plan verb(s) {bad} not in the tokenizer")
    if not verbs:
        raise AssemblyError("task_plan is required: it is the plan the turn carries out")
    parts.append("<task_plan>" + "".join(verbs) + "</task_plan>")

    if is_tool_call:
        # A tool-calling turn ends here: the plan decided to call, the call itself rides in `tool_calls`,
        # and there is no <response> because the assistant has not answered yet.
        if response:
            raise AssemblyError("a tool-calling turn must not also carry a response")
        return "".join(parts)

    # NER/POS answers are a `word <tag> LABEL` stream, so <tag> is content there, not a leaked marker.
    body = _check_plain(response or "", "response",
                        allow=SEQUENCE_TOKENS if sequence_labels else None)
    if not body:
        raise AssemblyError("a non-tool turn must carry a non-empty response")
    if target_lang not in VALID_LANGS:
        raise AssemblyError(f"target_lang {target_lang!r} is not a supported language tag")
    budget = max_response_chars(target_lang)
    if len(body) > budget:
        raise AssemblyError(
            f"response is {len(body)} chars, over the ~{MAX_RESPONSE_TOKENS}-token budget for "
            f"{target_lang} ({budget} chars). Each response must be complete and self-contained within it, "
            f"not truncated.")
    parts.append(f"<|target_lang|><{target_lang}><response>")
    if label_token:
        tok = LABEL_TOKENS.get(label_token, label_token)
        if tok not in LABEL_TOKENS.values():
            raise AssemblyError(f"label token {label_token!r} is not in the tokenizer")
        parts.append(tok)
    parts.append(body)
    return "".join(parts)


def assistant_message(turn: dict[str, Any]) -> dict[str, Any]:
    """One structured assistant turn -> a template-shaped message dict."""
    call = turn.get("tool_call") or None
    if call and not isinstance(call, dict):
        raise AssemblyError("tool_call must be an object")
    content = assistant_content(
        input_lang=turn.get("input_lang", ""),
        target_lang=turn.get("target_lang"),
        think=turn.get("think"),
        task_plan=turn.get("task_plan"),
        response=turn.get("response"),
        label_token=turn.get("label_token"),
        is_tool_call=bool(call),
        sequence_labels=bool(turn.get("sequence_labels")),
    )
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if call:
        name, args = call.get("name"), call.get("arguments")
        if not name:
            raise AssemblyError("tool_call needs a name")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                raise AssemblyError("tool_call arguments are not valid JSON") from None
        msg["tool_calls"] = [{"function": {"name": name, "arguments": args or {}}}]
    return msg


def build_messages(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A list of structured turns -> template-shaped messages, with the scaffolding built here.

    Non-assistant turns pass through with their text cleaned: a user message or tool result has no business
    carrying special tokens either, and the generator sometimes put them there.
    """
    if not isinstance(turns, list) or not turns:
        raise AssemblyError("turns must be a non-empty list")
    out: list[dict[str, Any]] = []
    for t in turns:
        if not isinstance(t, dict):
            raise AssemblyError("each turn must be an object")
        # Generators emit stray quotes and newlines around the role -- measured live as
        # `unknown role "user'\n"`. The intent is not in doubt, so it is normalised.
        role = t.get("role")
        if isinstance(role, str):
            role = role.strip().strip("\"'").strip().lower()
        if role == "assistant":
            out.append(assistant_message(t))
        elif role in ("user", "system"):
            text = _check_plain(t.get("content") or "", f"{role} content")
            if not text:
                raise AssemblyError(f"{role} turn is empty")
            out.append({"role": role, "content": text})
        elif role == "tool":
            # Tool results are verbatim payloads -- HTML, snippets, JSON -- so they are NOT tag-stripped.
            text = (t.get("content") or "").strip()
            if not text:
                raise AssemblyError("tool result is empty")
            out.append({"role": "tool", "name": t.get("name") or "", "content": text})
        else:
            raise AssemblyError(f"unknown role {role!r}")
    return out
