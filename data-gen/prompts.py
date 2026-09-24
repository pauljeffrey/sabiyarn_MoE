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
from schemas.seed import Seed

_PAIRS = all_pairs()
_GENRES = sorted(GENRES)
_REGISTERS = ["plain everyday", "formal", "conversational", "explanatory/teacherly", "journalistic", "storytelling"]
_LENGTHS = [(120, 200), (200, 320), (320, 480), (480, 700)]


def _rng(custom_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(custom_id.encode()).hexdigest()[:16], 16))


def _lang_offset(lang: str) -> int:
    return int(hashlib.sha256(lang.encode()).hexdigest()[:8], 16)


def _lang_spec(seed: Seed, code: str):
    return next(l for l in seed.languages if l.code == code)


def _coverage_pick(lang: str, index: int) -> tuple[str, str, str]:
    """(domain, subtopic, genre) -- walks every pair once before repeating, offset per language."""
    off = _lang_offset(lang)
    domain, subtopic = _PAIRS[(index + off) % len(_PAIRS)]
    # A stride coprime-ish with the genre count so genre does not lock in step with domain.
    genre = _GENRES[(index * 7 + off) % len(_GENRES)]
    return domain, subtopic, genre


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

Return JSON: {{"title": "<short natural title in {lang.name}>", "text": "<the document>", "language_self_check": <true only if the whole text is fluent {lang.name}>}}"""

    return Request(
        custom_id=row["custom_id"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=min(4096, int(hi * 4)), temperature=0.95,
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
    <|input_lang|><__LANG__><task_plan>__VERBS__</task_plan><|target_lang|><__LANG__><response>{the answer}
- An assistant turn that reasons first inserts <think>...</think> right after <|input_lang|><__LANG__>:
    <|input_lang|><__LANG__><think>{short reasoning}</think><task_plan>__VERBS__</task_plan><|target_lang|><__LANG__><response>{the answer}
- An assistant turn that CALLS A TOOL emits NO <response>. Put the call in `tool_calls` (JSON), and its
  content is:  <|input_lang|><__LANG__><think>{why this tool, and this query}</think>
- A tool result is a separate message with role "tool" and the tool's `name` set.
- <think> is written in __LANGUAGE_NAME__, 1-3 sentences, and never just restates the question.
- task_plan verbs must come from: __VERBS_ALLOWED__
"""


def _format_brief(lang_code: str, language_name: str, verbs: str, verbs_allowed: str) -> str:
    return (_FORMAT_BRIEF.replace("__LANG__", lang_code).replace("__VERBS_ALLOWED__", verbs_allowed)
            .replace("__LANGUAGE_NAME__", language_name).replace("__VERBS__", verbs))


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
        filler = [t.name for t in seed.tools if t.name not in required]
        rng.shuffle(filler)
        # 3-6 tools total: enough distractors that tool SELECTION is a real choice, few enough that a
        # 306M model can learn the catalogue.
        n = max(len(required), min(6, rng.randint(3, 6)))
        tool_names = required + filler[:max(0, n - len(required))]
    else:
        tool_names = []

    n_msgs = rng.randint(seed.conversation["min_messages"], seed.conversation["max_messages"])
    verbs = sorted({v for t in tasks for v in t.task_plan}) or ["<|chat|>"]
    fmt = _format_brief(lang.code, lang.name, "".join(verbs),
                        " ".join(seed.format["special_tokens"]["task_plan_verbs"]))

    task_lines = "\n".join(f"  * {t.name} [{', '.join(t.tags)}]: {t.description}" for t in tasks)
    system = (f"{seed.details}\n\n{fmt}\n"
              f"You are generating training conversations in {lang.name} ({lang.code}). "
              f"Language guidance: {lang.guidance}\nReturn strict JSON only, no commentary.")

    tools_section = ""
    if tool_names:
        tools_section = (
            f"\nTOOLS available in this conversation (put this exact JSON in the system message):\n"
            f"{_tool_block(seed, tool_names)}\n\nHow these tools behave when called:\n"
            f"{_tool_behaviour(seed, tool_names)}\n")
    else:
        tools_section = ("\nThis conversation has NO tools and NO system message. If the user asks something "
                         "the assistant does not know, the correct behaviour is to say so plainly.\n")

    if not rl:
        shape = f"""Produce ONE conversation of exactly {n_msgs} messages, ending with an assistant message.

Return JSON:
{{"messages": [{{"role": "system"|"user"|"assistant"|"tool", "content": "...", "name": "<tool name, tool role only>", "tool_calls": [{{"function": {{"name": "...", "arguments": {{...}}}}}}]}}],
  "tasks": ["<task names used>"], "tags": ["<tags from the closed list>"]}}"""
    else:
        shape = f"""Produce ONE conversation PREFIX of {n_msgs - 1} messages ending with a USER message, then
{seed.conversation['responses_per_prompt']} alternative assistant replies to that final user message.

The replies must be genuinely rankable, not paraphrases:
  - exactly one is best: honest, correctly grounded, uses the right tool or correctly admits it cannot;
  - exactly one is worst: confidently wrong -- invents a fact, answers from an irrelevant tool result, or
    claims to recognise something it was never told;
  - the rest are partial: right but uselessly hedged, right answer via the wrong tool, or refusing when the
    answer WAS actually available.

Return JSON:
{{"prompt_messages": [...same message shape as above...],
  "responses": [{{"content": "<full assistant turn incl. special tokens>", "quality": "best"|"partial"|"worst", "why": "<one line>"}}],
  "tasks": [...], "tags": [...]}}"""

    user = f"""Conversation language: {lang.name}. Everything the user and assistant say is in {lang.name}.

Tasks this conversation must cover (mix them, change subject at least once):
{task_lines}

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
        max_tokens=4096, temperature=0.9,
        response_format={"type": "json_object"},
        metadata={"kind": seed.kind, "lang": lang.code, "task": task.name,
                  "tasks": [t.name for t in tasks], "tags": sorted({g for t in tasks for g in t.tags}),
                  "tools": tool_names, "domain": domain, "subtopic": subtopic, "n_messages": n_msgs},
    )


def build_request(seed: Seed, row: dict) -> Request:
    if seed.kind == "pretrain":
        return _pretrain_request(seed, row)
    return _sft_like_request(seed, row, rl=(seed.kind == "rl"))
