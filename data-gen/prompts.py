"""Seed + plan row -> the meta-prompt sent to the generating model.

Everything here is DETERMINISTIC in `custom_id`. The same plan row produces the same prompt on any machine,
in any provider, on any platform -- which is what makes a run resumable and a corpus reproducible.

Coverage is by construction rather than by sampling: the row's index walks the (domain, sub-topic) list with
a per-language stride, so every pair is used once before any is used twice. No shared sampler state, nothing
to persist, and a job split across Modal + RunPod + a vast box still covers the taxonomy exactly once.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Optional

from providers.base import Request
from sampling.taxonomy import DOMAINS, GENRES, all_pairs
from schemas.output import schema_for
from schemas.seed import Seed

_PAIRS = all_pairs()
_GENRES = sorted(GENRES)
_REGISTERS = ["plain everyday", "formal", "conversational", "explanatory/teacherly", "journalistic", "storytelling"]
# 300-500 words, in four buckets so length still varies within the band.
_LENGTHS = [(300, 350), (350, 400), (400, 450), (450, 500)]


def _rng(custom_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(custom_id.encode()).hexdigest()[:16], 16))


def _lang_offset(lang: str) -> int:
    return int(hashlib.sha256(lang.encode()).hexdigest()[:8], 16)


def _lang_spec(seed: Seed, code: str):
    return next(l for l in seed.languages if l.code == code)


def _coverage_pick(lang: str, index: int) -> tuple[str, str, str]:
    """(domain, subtopic, genre), walking the FULL CROSS PRODUCT of pairs x genres once before repeating.

    The obvious version -- pair = (i + off) % n_pairs, genre = (i * 7 + off) % n_genres -- is broken: with
    636 pairs and 28 genres, 636 * 7 is a multiple of 28, so for any fixed pair the genre never advances.
    Every document about one sub-topic came out in the same genre, collapsing 17,808 combinations to 636.

    Treating (pair, genre) as one odometer fixes it: the low digit cycles pairs, and each time it laps, the
    high digit moves to the next genre. Still stateless and still deterministic in `index`, so it partitions
    cleanly across workers and resumes exactly.
    """
    n_pairs, n_genres = len(_PAIRS), len(_GENRES)
    combo = (index + _lang_offset(lang)) % (n_pairs * n_genres)
    domain, subtopic = _PAIRS[combo % n_pairs]
    return domain, subtopic, _GENRES[(combo // n_pairs) % n_genres]


def _tool_defs(seed: Seed, names: list[str]) -> list[dict]:
    """OpenAI-shaped definitions, carried in metadata so a batch job can attach them verbatim."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                             "parameters": t.parameters}}
            for t in seed.tools if t.name in names]


def _tool_block(seed: Seed, names: list[str]) -> str:
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                               "parameters": t.parameters}}
             for t in seed.tools if t.name in names]
    return json.dumps(tools, ensure_ascii=False, indent=2)


def _tool_behaviour(seed: Seed, names: list[str]) -> str:
    out = []
    for t in seed.tools:
        if t.name in names:
            out.append(f"- {t.name}: returns {t.returns} Realistic failures: {'; '.join(t.failure_modes)}.")
    return "\n".join(out)


# --------------------------------------------------------------------------- pretrain


def _pretrain_request(seed: Seed, row: dict) -> Request:
    lang = _lang_spec(seed, row["lang"])
    domain, subtopic, genre = _coverage_pick(lang.code, row["index"])
    rng = _rng(row["custom_id"])
    register = rng.choice(_REGISTERS)
    lo, hi = rng.choice(_LENGTHS)
    d = DOMAINS[domain]
    g = GENRES[genre]

    system = (
        f"{seed.details}\n\n"
        f"You are writing PRETRAINING TEXT in {lang.name} ({lang.code}). "
        f"Language guidance: {lang.guidance}\n"
        "Return strict JSON only, no commentary."
    )
    user = f"""Write one document in {lang.name}.

Domain: {d.name} -- {d.description}
Sub-topic: {subtopic}
Genre: {getattr(g, 'text', genre)}
Register: {register}
Length: {lo}-{hi} words.

Requirements:
- Entirely in {lang.name}. No English except words the language genuinely borrows.
- Explain HOW and WHY things work, not just what they are called. Mechanism over name-dropping.
- Locally grounded: real West African settings, foods, prices, institutions, seasons.
- No invented statistics, no fake citations, no made-up named people presented as real.
- Plain continuous prose. No markdown, no headings, no bullet lists, no chat markup.

Return JSON: {{"title": "<short natural title in {lang.name}>", "text": "<the document>", "language_self_check": <true only if the whole text is fluent {lang.name}>, "confidence": <float 0-1: your honest estimate that this text is accurate AND fluent {lang.name}>}}"""

    return Request(
        custom_id=row["custom_id"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        # 4 tokens/word is generous for English but tight for diacritic-heavy languages, where 500
        # words can exceed 1,200 tokens; the floor stops truncation mid-document.
        max_tokens=max(1800, min(4096, int(hi * 4))), temperature=0.95,
        response_format={"type": "json_object"},
        metadata={"kind": "pretrain", "lang": lang.code, "domain": domain, "subtopic": subtopic,
                  "genre": genre, "register": register, "target_words": [lo, hi]},
    )


# --------------------------------------------------------------------------- sft / rl


# Literal braces below are part of the target model's format, so this uses __PLACEHOLDER__ substitution
# rather than str.format().
_FORMAT_BRIEF = """\
SPECIAL TOKEN FORMAT (the target model's own vocabulary -- use these EXACTLY):
- An assistant turn that answers normally has content:
    <|input_lang|><__LANG__><task_plan>{verbs}</task_plan><|target_lang|><__LANG__><response>{the answer}
- An assistant turn that reasons first inserts <think>...</think> right after <|input_lang|><__LANG__>:
    <|input_lang|><__LANG__><think>{short reasoning}</think><task_plan>{verbs}</task_plan><|target_lang|><__LANG__><response>{the answer}
- An assistant turn that CALLS A TOOL emits NO <response>. Put the call in `tool_calls` (JSON), and its
  content is:  <|input_lang|><__LANG__><think>{why this tool, and this query}</think>
- A tool result is a separate message with role "tool" and the tool's `name` set.
- <think> IS ALWAYS IN ENGLISH, never in __LANGUAGE_NAME__, whatever language the conversation is in. It is
  the model's private scratchpad: 1-3 short English sentences reasoning about what is being asked, whether it
  already knows, and what it would need. Never a restatement of the question.
- The <response> text is ALWAYS in __LANGUAGE_NAME__.
- <task_plan> is a PLAN, not a label. For a multi-step turn, list the verbs in the order they will be carried
  out, e.g. <task_plan><|RAG|><|analyze|><|explain|></task_plan>. Verbs must come from: __VERBS_ALLOWED__
- CLOSING TAGS: only these five have one -- </think>, </task_plan>, </tool_call>, </tool_response>, </context>.
  Every other marker is an OPENER ONLY and must never be closed. Write <response>the answer with nothing after
  it, and <sentiment>positive with no closing tag. Inventing </response> or </sentiment> is wrong: they are not
  in the model's vocabulary and cost 3-4 junk sub-word tokens each.
- Label tasks put the label straight after its marker, e.g. <|target_lang|><__LANG__><response><sentiment>mixed
"""


def _format_brief(lang_code: str, language_name: str, verbs_allowed: str) -> str:
    """A pure function of (language). Anything row-specific belongs in the user message instead: the system
    prompt is the shared prefix that vLLM's prefix cache reuses across a whole (kind, lang) group, and one
    varying character in it costs a full recompute per row."""
    return (_FORMAT_BRIEF.replace("__LANG__", lang_code).replace("__VERBS_ALLOWED__", verbs_allowed)
            .replace("__LANGUAGE_NAME__", language_name))


def _sft_like_request(seed: Seed, row: dict, *, rl: bool) -> Request:
    lang = _lang_spec(seed, row["lang"])
    rng = _rng(row["custom_id"])
    task = next(t for t in seed.tasks if t.name == row["task"])
    # 1-3 extra tasks so a conversation genuinely mixes themes, as the seed's conversation policy requires.
    others = [t for t in seed.tasks if t.name != task.name and (not t.languages or lang.code in t.languages)]
    extra = rng.sample(others, k=min(len(others), rng.randint(1, 3)))
    tasks = [task] + extra
    domain, subtopic, _ = _coverage_pick(lang.code, row["index"])

    needs_tools = any(t.uses_tools for t in tasks)
    if needs_tools:
        # Every task that needs a specific tool must have it in scope, or the brief contradicts itself
        # ("use get_exchange_rate" while the tool list omits it).
        required: list[str] = []
        for t in tasks:
            for hint, name in (("rag", "search_documents"), ("extractive-qa", "search_documents"),
                               ("knowledge-boundary", "search_internet"),
                               ("insufficient-context", "search_internet"),
                               ("health-advice", "lookup_health_guidance"),
                               ("health-triaging", "lookup_health_guidance"),
                               ("financial-analysis", "get_exchange_rate")):
                if hint in t.tags and name not in required:
                    required.append(name)
        if any("financial-analysis" in t.tags for t in tasks) and "calculate" not in required:
            required.append("calculate")
        # 2-3 deliberately IRRELEVANT tools are always in scope. Tool selection is only a real skill if
        # there is something wrong to select, and a model trained only on catalogues where every tool is
        # applicable learns to call whatever it is given.
        filler = [t.name for t in seed.tools if t.name not in required]
        rng.shuffle(filler)
        n_distract = rng.randint(2, 3)
        distractors = filler[:n_distract]
        tool_names = required + distractors
        rng.shuffle(tool_names)
    else:
        tool_names, distractors = [], []

    n_user = rng.randint(int(seed.conversation.get("min_user_turns", 3)),
                         int(seed.conversation.get("max_user_turns", 5)))
    verbs = sorted({v for t in tasks for v in t.task_plan}) or ["<|chat|>"]
    fmt = _format_brief(lang.code, lang.name,
                        " ".join(seed.format["special_tokens"]["task_plan_verbs"]))

    task_lines = "\n".join(f"  * {t.name} [{', '.join(t.tags)}]: {t.description}" for t in tasks)
    verbs_line = ("task_plan verbs for this conversation (use them in the order each turn carries them out): "
                  + "".join(verbs))
    system = (f"{seed.details}\n\n{fmt}\n"
              f"You are generating training conversations in {lang.name} ({lang.code}). "
              f"Language guidance: {lang.guidance}\nReturn strict JSON only, no commentary.")

    tools_section = ""
    if tool_names:
        tools_section = (
            f"\nTOOLS available in this conversation (put this exact JSON in the system message):\n"
            f"{_tool_block(seed, tool_names)}\n\nHow these tools behave when called:\n"
            f"{_tool_behaviour(seed, tool_names)}\n"
            f"\nIRRELEVANT TOOLS: {', '.join(distractors)} are in the catalogue but are NOT useful for this "
            f"conversation. The assistant must never call them. Their presence is deliberate -- the model has "
            f"to learn to pick the right tool, not just any tool.\n")
    else:
        tools_section = ("\nThis conversation has NO tools and NO system message. If the user asks something "
                         "the assistant does not know, the correct behaviour is to say so plainly.\n")

    lang_name = lang.name
    if not rl:
        shape = f"""Produce ONE conversation with EXACTLY {n_user} messages of role "user" -- no more, no
fewer. Each is answered by the assistant, and the conversation ends on an assistant message. The assistant may
add tool-call turns and their role="tool" results in between; those are not user messages and do not count.
Count your user messages before you finish: there must be exactly {n_user}.

Return JSON:
{{"messages": [{{"role": "system"|"user"|"assistant"|"tool", "content": "...", "name": "<tool name, tool role only>", "tool_calls": [{{"function": {{"name": "...", "arguments": {{...}}}}}}]}}],
  "tasks": ["<task names used>"], "tags": ["<tags from the closed list>"],
  "confidence": <float 0-1: your honest estimate that this conversation is correct AND fluent {lang_name}>}}"""
    else:
        shape = f"""Produce ONE conversation PREFIX with EXACTLY {n_user} messages of role "user", the LAST
of which ends the prefix (tool turns are extra and do not count), then
{seed.conversation['responses_per_prompt']} alternative assistant replies to that final user message.

The replies must be genuinely rankable, not paraphrases:
  - exactly one is best: honest, correctly grounded, uses the right tool or correctly admits it cannot;
  - exactly one is worst: confidently wrong -- invents a fact, answers from an irrelevant tool result, or
    claims to recognise something it was never told;
  - the rest are partial: right but uselessly hedged, right answer via the wrong tool, or refusing when the
    answer WAS actually available.

Return JSON:
{{"prompt_messages": [...same message shape as above...],
  "responses": [{{"content": "<full assistant turn incl. special tokens>", "quality": "best"|"partial"|"worst", "why": "<one line, English>"}}],
  "tasks": [...], "tags": [...],
  "confidence": <float 0-1: your honest estimate that the ranking is right and the text is fluent {lang_name}>}}"""

    user = f"""Conversation language: {lang.name}. Everything the user and assistant say is in {lang.name}.

Tasks this conversation must cover (mix them, change subject at least once):
{task_lines}

{verbs_line}

Ground the content in: {DOMAINS[domain].name} / {subtopic}.
{tools_section}
{shape}

Hard requirements:
- The assistant NEVER invents facts. When it does not know and cannot look it up, it says so.
- When a tool result does not answer the question, the assistant says that instead of forcing an answer.
- Tool arguments must be valid against the tool's JSON schema.
- Tool results must be realistic: concrete snippets, sometimes partial, sometimes unhelpful.
- No sycophancy ("Great question!"), no padding, no restating the question before answering."""

    return Request(
        custom_id=row["custom_id"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        # 5 user turns + tool calls + tool results + think blocks overruns 4096 and the JSON is truncated
        # mid-string, which showed up as json_invalid on ~17% of a pilot. Headroom is far cheaper than a retry.
        max_tokens=6144, temperature=0.9,
        # Deliberately json_object, NOT json_schema. A pilot with the schema attached produced *more*
        # rejects (messages_malformed 11/24 vs 1/24): providers coerce the output to satisfy the schema
        # literally -- emitting every optional key, including empty tool_calls on plain turns -- which is
        # valid JSON but not a well-formed conversation. The same schema IS worth using in vllm_gen.py, where
        # it constrains decoding directly rather than being reinterpreted by a provider.
        response_format={"type": "json_object"},
        # tools= is attached natively so the generating model produces schema-valid call arguments with its
        # own function-calling machinery instead of inventing the JSON as prose.
        tools=_tool_defs(seed, tool_names) or None,
        metadata={"kind": seed.kind, "lang": lang.code, "task": task.name,
                  "tasks": [t.name for t in tasks], "tags": sorted({g for t in tasks for g in t.tags}),
                  "tools": tool_names, "distractor_tools": distractors,
                  "tool_definitions": _tool_defs(seed, tool_names),
                  "domain": domain, "subtopic": subtopic, "n_user_turns": n_user},
    )


def build_request(seed: Seed, row: dict) -> Request:
    if seed.kind == "pretrain":
        return _pretrain_request(seed, row)
    return _sft_like_request(seed, row, rl=(seed.kind == "rl"))
