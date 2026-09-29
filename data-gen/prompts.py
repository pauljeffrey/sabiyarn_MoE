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
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import os

from assemble import MAX_RESPONSE_TOKENS, VALID_LANGS, VALID_VERBS
from longdocs import DOCUMENT_TASKS, document_brief
from providers.base import Request
from sampling.taxonomy import DOMAINS, GENRES, all_pairs
from schemas.output import schema_for
from schemas.seed import Seed

_FEWSHOT_PATH = Path(__file__).resolve().parent / "seeds" / "fewshot.json"


@lru_cache(maxsize=1)
def _fewshot_block() -> str:
    """Structurally perfect exemplars, rendered into the system prompt AS FIELDS.

    Only the STRUCTURE transfers: these are real conversations in whatever language they were generated in,
    and the prompt says so. Structure is what collapses in low-resource languages, so that is what the
    examples are for.

    THEY MUST BE RENDERED IN THE FIELD SHAPE, and were not. `seeds/fewshot.json` holds finished training
    records, whose assistant turns are pre-assembled marker strings -- the exact shape the field contract
    replaced. Printing them verbatim told the generator, immediately after the contract forbade it, to do it
    anyway, and the generator believed the example: every captured `non-tool turn must carry a non-empty
    response` drop was a low-resource language (the only tier that receives exemplars) returning
    "<|input_lang|><ful><think>...</think>...<response>..." in `content`. 31 drops in 225 requests on one live
    run, and it was the single largest failure class.

    So the records are taken APART here (disassemble.py, the same parser that repaired the published corpus)
    and shown as the fields the generator is actually asked for.
    """
    if not _FEWSHOT_PATH.exists():
        return ""
    from assemble import AssemblyError, build_messages
    from disassemble import looks_pre_assembled, turn_from_assistant

    data = json.loads(_FEWSHOT_PATH.read_text(encoding="utf-8"))
    parts = []
    for i, ex in enumerate(data.get("exemplars", []), start=1):
        lines, turns = [], []
        for m in ex["messages"]:
            role = m["role"]
            if role == "assistant":
                turn = (turn_from_assistant(m, ex.get("lang", "eng"))
                        if looks_pre_assembled(m.get("content")) or m.get("tool_calls") else None)
                if not turn:
                    continue
                # Key order matches the contract's, so the example reads as the contract does.
                ordered = {k: v for k in ("role", "think", "task_plan", "input_lang", "target_lang",
                                          "response", "label_token", "tool_call")
                           if (v := turn.get(k)) not in (None, "", [], False)}
                turns.append(turn)
                lines.append("  " + json.dumps(ordered, ensure_ascii=False))
            elif role == "tool":
                msg = {"role": "tool", "name": m.get("name", ""), "content": m.get("content") or ""}
                turns.append(msg)
                lines.append("  " + json.dumps(msg, ensure_ascii=False))
            elif role == "user":
                msg = {"role": "user", "content": m.get("content") or ""}
                turns.append(msg)
                lines.append("  " + json.dumps(msg, ensure_ascii=False))
            # the system message is rebuilt from the tool catalogue after generation, so it is not an example
        # An exemplar must itself SURVIVE the assembler. One did not: it answered NER as
        # "<NER>Etidiong<tag>PERSON", which the assembler rejects because <NER> is a task_plan verb and not
        # legal inside a response -- and that exemplar was the source of the <NER> drops in live generation.
        # An example of something we discard is worse than no example.
        try:
            build_messages(turns)
        except (AssemblyError, Exception):  # noqa: BLE001
            continue
        if len(lines) >= 4:
            parts.append(f"--- EXEMPLAR {i} ({ex['lang']}, io_direction={ex.get('io_direction')}, "
                         f"tools={'yes' if ex.get('uses_tools') else 'no'}) ---\n"
                         + "[\n" + ",\n".join(lines) + "\n]")
    if not parts:
        return ""
    return (
        "\n\nWORKED EXAMPLES OF THE REQUIRED SHAPE\n"
        "Real, verified-correct conversations, shown as the FIELDS you must return. Copy the SHAPE exactly: "
        "which keys each turn carries, how a tool-calling turn has a tool_call and no response, how the tool "
        "result comes back as its own role='tool' message, and how the assistant then takes another turn with "
        "a fresh `think` judging what came back.\n"
        "Note that NO turn contains <|input_lang|>, <think>, <task_plan> or <response> -- those are built from "
        "your fields afterwards. Never write them yourself.\n"
        "Do NOT copy the language, topic, names or wording -- these happen to be in the languages they were "
        "generated in, and yours must be in the language this task specifies.\n\n"
        + "\n\n".join(parts) + "\n"
    )


_PAIRS = all_pairs()
_GENRES = sorted(GENRES)
# Implied current date/time, cycled so time-dependent behaviour is not learned against one anchor.
_WHENS = [
    "a Monday morning in January", "a Wednesday afternoon in March", "a Friday evening in May",
    "a Saturday morning in July", "a Sunday afternoon in August", "a Tuesday night in October",
    "a Thursday midday in November", "a Saturday evening in December",
    "early morning during Ramadan", "the week before Christmas", "the middle of the rainy season",
    "the height of the harmattan",
]
_REGISTERS = ["plain everyday", "formal", "conversational", "explanatory/teacherly", "journalistic", "storytelling"]


def _rng(custom_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(custom_id.encode()).hexdigest()[:16], 16))


def _lang_offset(lang: str) -> int:
    return int(hashlib.sha256(lang.encode()).hexdigest()[:8], 16)


def _io_direction(seed: Seed, lang: str, index: int, rng: random.Random) -> dict:
    """Pick the input/output language pattern on its own odometer.

    `index` advances by 1 per row for a fixed (lang, task), so stepping through a 5-element list gives an
    EXACTLY even distribution rather than an approximately even one. It also stays independent of the
    domain/genre odometer: that one laps every 17,808 rows and 17,808 mod 5 = 3, which is coprime with 5, so
    a given (domain, genre) still sees all five directions.
    """
    dirs = seed.conversation.get("io_directions") or [{"key": "native_native", "brief": "Everything in {lang}."}]
    d = dict(dirs[(index + _lang_offset(lang)) % len(dirs)])
    others = [l for l in seed.languages if l.code not in (lang, "eng")]
    other = rng.choice(others) if others else None
    me = _lang_spec(seed, lang)
    d["brief"] = (d["brief"].replace("{lang}", me.name).replace("{code}", me.code)
                  .replace("{other_name}", other.name if other else "Hausa")
                  .replace("{other_code}", other.code if other else "hau"))
    d["other"] = other.code if other else None
    # the language tags the assistant's markers must carry, so compliance can be measured
    expect = {"native_native": (me.code, me.code), "english_english": ("eng", "eng"),
              "english_to_native": ("eng", me.code), "native_to_english": (me.code, "eng"),
              "crosslingual": (d["other"] or "hau", me.code)}
    d["expect_markers"] = list(expect.get(d["key"], (me.code, me.code)))
    return d


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


# Public alias: generate.py needs the same (domain, subtopic) a row will get, so a document is about the same
# thing the conversation is about.
def coverage_pick(lang: str, index: int) -> tuple[str, str, str]:
    return _coverage_pick(lang, index)


def _tool_defs(seed: Seed, names: list[str]) -> list[dict]:
    """OpenAI-shaped definitions, carried in metadata so a batch job can attach them verbatim."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                             "parameters": t.parameters}}
            for t in seed.tools if t.name in names]


def _tool_block(seed: Seed, names: list[str]) -> str:
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                               "parameters": t.parameters}}
             for t in seed.tools if t.name in names]
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def _tool_behaviour(seed: Seed, names: list[str]) -> str:
    out = []
    for t in seed.tools:
        if t.name in names:
            out.append(f"- {t.name}: returns {t.returns} Realistic failures: {'; '.join(t.failure_modes)}.")
    return "\n".join(out)


# --------------------------------------------------------------------------- pretrain


# Words per pretraining document, from the phase's TOKEN target and the language's cost per word: Yoruba runs
# ~2.5 tokens/word against English's ~1.15, so one word figure would either overshoot the budget in Yoruba or
# waste half of it in English. Four buckets so length still varies within the band.
_PRETRAIN_SPREAD = ((0.70, 0.82), (0.80, 0.92), (0.88, 1.00), (0.94, 1.06))
# Low-resource languages get a fraction of it. This is a correctness fix, not a preference: asked for 400 words
# of continuous Fon the generator ran out of vocabulary and cycled, and on a pilot BOTH Fon documents were one
# clause repeated for ~400 words while every Yoruba, Hausa and Twi document was clean.
_LOW_TIER_SCALE = 0.55


def _pretrain_length(seed: Seed, lang, rng: random.Random) -> tuple[int, int]:
    from assemble import _TOKENS_PER_WORD
    from budgets import budget_for

    target = budget_for("pretrain").pretrain_tokens
    words = target / _TOKENS_PER_WORD.get(lang.code, 2.5)
    if lang.tier == "low":
        words *= _LOW_TIER_SCALE
    lo_f, hi_f = rng.choice(_PRETRAIN_SPREAD)
    return max(60, int(words * lo_f)), max(90, int(words * hi_f))


def _pretrain_request(seed: Seed, row: dict) -> Request:
    lang = _lang_spec(seed, row["lang"])
    domain, subtopic, genre = _coverage_pick(lang.code, row["index"])
    rng = _rng(row["custom_id"])
    register = rng.choice(_REGISTERS)
    lo, hi = _pretrain_length(seed, lang, rng)
    d = DOMAINS[domain]
    g = GENRES[genre]

    system = (
        f"{seed.details}\n\n"
        f"You are writing PRETRAINING TEXT in {lang.name} ({lang.code}). "
        f"Language guidance: {lang.guidance}\n"
        f"{_simplicity(lang)}"
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
- NEVER REPEAT YOURSELF. Every sentence must say something the previous ones did not, and no sentence may
  appear twice. Padding with a repeated clause makes the whole document worthless and it will be discarded.
- If you cannot fill {lo} words without repeating, write about a DIFFERENT, more concrete aspect of the
  sub-topic -- one person's day, one tool, one season, one price -- rather than restating. {lo} words is a
  HARD FLOOR: a document shorter than that is discarded too, so neither padding nor stopping early works.
  The way through is more specifics.

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
OUTPUT CONTRACT -- you emit FIELDS, never marker strings.
You never write <|input_lang|>, <|target_lang|>, <think>, <task_plan> or <response> yourself. Those are built
from your fields afterwards, so getting them right is not your problem. Every text field is PLAIN TEXT: a
sample containing any <tag> inside a text field is discarded.

A conversation is a list of `turns`. Each turn is one of:

  {"role": "user", "content": "what the user says, plain text"}

  {"role": "assistant",
   "think":       "OPTIONAL. 1-3 short sentences of reasoning, ALWAYS IN ENGLISH whatever the conversation
                   language. Include it when the turn involves a tool, a judgement about whether you know
                   something, triage, or a hard translation. OMIT IT for simple turns -- a short translation,
                   a label, a greeting. Do not think on every turn.",
   "task_plan":   ["REQUIRED. The verbs this turn carries out, in order, e.g. ["<|RAG|>","<|explain|>"]."],
   "input_lang":  "REQUIRED. Language code of what came IN, e.g. "eng".",
   "target_lang": "Language code of the answer, e.g. "pcm". Required unless this turn calls a tool.",
   "response":    "The answer, PLAIN TEXT, in target_lang. Omit if this turn calls a tool. It must be
                   COMPLETE and SELF-CONTAINED -- finish the thought, never trail off or promise to continue,
                   and do not depend on a later turn to make sense. Hard ceiling __MAX_TOK__ tokens; over that
                   the sample is discarded, so if the answer would run longer, answer the part asked for
                   properly rather than starting something you cannot finish.",
   "label_token": "OPTIONAL, classification only: one of sentiment|topic|intent|toxic|lang_id|ner.",
   "tool_call":   {"name": "tool_name", "arguments": {...}}}   // omit unless calling a tool

  {"role": "tool", "name": "tool_name", "content": "what the tool returned, verbatim"}

RULES THAT FOLLOW FROM THIS
- A turn either ANSWERS (response + target_lang, no tool_call) or CALLS A TOOL (tool_call, no response).
  After a tool result the assistant takes ANOTHER turn, usually with a fresh `think` judging whether the
  result actually answers the question.
- task_plan is a PLAN and every assistant turn has one, including a tool-calling turn -- the plan is what
  decided to call the tool. Plans are REVISED as work proceeds, exactly as a person would: think -> plan
  (search the document) -> call -> read the result -> think again -> revised plan (that was not it; search for
  X instead, or now analyse and explain) -> act. Do not reuse the same plan every turn.
- task_plan verbs must come from: __VERBS_ALLOWED__
- Language codes must come from: __LANGS_ALLOWED__
- For a translation, input_lang and target_lang DIFFER and must match the real direction. Put only the
  translated text in `response` -- no "pcm->igbo" prefix, no language names.
"""


def _format_brief(lang_code: str, language_name: str, verbs_allowed: str) -> str:
    """A pure function of (language): anything row-specific belongs in the user message, because this is the
    shared prefix that prefix caching reuses across a whole (kind, lang) group."""
    return (_FORMAT_BRIEF.replace("__MAX_TOK__", str(MAX_RESPONSE_TOKENS))
            .replace("__VERBS_ALLOWED__", verbs_allowed)
            .replace("__LANGS_ALLOWED__", " ".join(sorted(VALID_LANGS)))
            .replace("__LANGUAGE_NAME__", language_name).replace("__LANG__", lang_code))


# --------------------------------------------------------------------------- complexity by tier
#
# A generator that writes good Yoruba writes bad Fon, and it fails in a specific way: it attempts the same
# ambitious sentence it would write in Yoruba, runs out of vocabulary, and drifts -- into French for Fon, into
# Ibibio for Efik, or into a repeating loop (the `degenerate_user_message` drop). The fix is to ask for less.
# Short concrete sentences in a language the generator half-knows are usable data; long subordinate clauses in
# the same language are not.
#
# This is a CEILING on ambition, not on quality: the structure, the honesty behaviour and the tool use are
# identical across tiers. Only the linguistic load changes.

_SIMPLICITY = {
    "low": """\
SIMPLICITY REQUIREMENT -- __LANGUAGE_NAME__ is a very low-resource language and FLUENCY BEATS AMBITION here.
- Short, plain sentences. One idea per sentence. No stacked subordinate clauses, no long noun phrases.
- Everyday concrete vocabulary: things you can point at, actions people do, numbers, places. Avoid abstract
  nouns and technical registers unless the task is about them.
- Keep each assistant response SHORT: 2-5 sentences for an explanation, one line for a label or translation.
- ONE subject for the whole conversation. Do not change topic; the theme-drift requirement is waived here.
- If you cannot say something naturally in __LANGUAGE_NAME__, say a SIMPLER thing that is true instead of
  reaching for a construction you are unsure of. A correct plain sentence is worth more than an elegant
  guess.
- Never pad to length, and never repeat a phrase to fill space -- a looping sentence makes the sample
  worthless and it will be discarded.""",
    "medium": """\
Keep the __LANGUAGE_NAME__ natural over clever: normal sentence length, everyday word choice, and the
diacritics correct throughout. Length is never a goal in itself.""",
    "high": "",
}

# Ceiling on user turns and on how many tasks a conversation mixes, by tier. A low-resource conversation that
# tries six turns across four tasks produces six turns of mediocre language; two good turns are worth more.
_TIER_TURNS = {"low": 3, "medium": 5, "high": 6}
_TIER_EXTRA_TASKS = {"low": 1, "medium": 3, "high": 3}


def _simplicity(lang) -> str:
    block = _SIMPLICITY.get(lang.tier, "")
    return ("\n" + block.replace("__LANGUAGE_NAME__", lang.name) + "\n") if block else ""


# --------------------------------------------------------------------------- sft / rl


# Literal braces below are part of the target model's format, so this uses __PLACEHOLDER__ substitution
# rather than str.format().
def _sft_like_request(seed: Seed, row: dict, *, rl: bool, document: Optional[dict] = None) -> Request:
    lang = _lang_spec(seed, row["lang"])
    rng = _rng(row["custom_id"])
    task = next(t for t in seed.tasks if t.name == row["task"])
    # 1-3 extra tasks so a conversation genuinely mixes themes, as the seed's conversation policy requires --
    # but at most 1 for a low-resource language, and none at all alongside a long document, which is already
    # the whole conversation.
    others = [t for t in seed.tasks if t.name != task.name and (not t.languages or lang.code in t.languages)]
    # A summarisation document IS the conversation, so it covers one task. A RAG document is retrieved
    # mid-conversation and mixes normally.
    max_extra = 0 if (document and not document.get("rag")) else _TIER_EXTRA_TASKS.get(lang.tier, 3)
    extra = (rng.sample(others, k=min(len(others), rng.randint(1, max_extra))) if max_extra else [])
    tasks = [task] + extra
    domain, subtopic, _ = _coverage_pick(lang.code, row["index"])
    io = _io_direction(seed, lang.code, row["index"], rng)
    # A RAG document reaches the assistant through a TOOL, so it does not change what came IN: the user still
    # writes in their own language and io_direction is already correct. Only a pasted summarisation document
    # changes the input language.
    if document and not document.get("rag"):
        # What came IN is the document, whatever io_direction would otherwise have said. The markers have to
        # match reality: an English report summarised into Yoruba is <|input_lang|><eng>, not <yor>. The brief
        # is rewritten too, or it would tell the generator "everything is in Yoruba" over an English document.
        src = "eng" if document.get("doc_lang_mode") == "mixed" else (document.get("doc_lang") or "eng")
        if io["expect_markers"][1] == src and src != lang.code:
            io["expect_markers"][1] = lang.code      # do not degenerate to an English-in/English-out summary
        io["expect_markers"][0] = src
        tgt = io["expect_markers"][1]
        src_name = _lang_spec(seed, src).name
        tgt_name = _lang_spec(seed, tgt).name
        io["brief"] = (
            f"The document the user pastes is in {src_name}"
            + (" (mixed with English in places)" if document.get("doc_lang_mode") == "mixed" else "")
            + f", and the assistant answers in {tgt_name}. What the USER writes around the document -- the "
            f"request itself -- is in {tgt_name} too, because that is the language they are speaking."
            + ("" if src == tgt else
               f" Reading {src_name} and answering in {tgt_name} is the point of this sample.")
            # Genuinely ambiguous otherwise, and the generator resolved it the other way three times out of
            # three: the user's own words are Yoruba but the document is thousands of words of English.
            # input_lang follows the DOCUMENT, because that is almost all of what came in.
            + f" input_lang is the language of the DOCUMENT ({src}), not of the user's own sentence around it,"
              f" because the document is nearly all of what came in.")
    identities = seed.conversation.get("identities") or []
    identity = identities[(row["index"] + _lang_offset(lang.code)) % len(identities)] if identities else ""
    # Spread the implied "now" across the year, days of the week and hours so nothing is anchored to one
    # date: anything time-dependent (reminders, forecasts, rates, "today") must work at any point.
    # Stride must be COPRIME with len(_WHENS) or most values are unreachable: *3 mod 12 reached only 4 of 12,
    # the same arithmetic trap that once locked every sub-topic to one genre. 5 is coprime with 12.
    when = _WHENS[(row["index"] * 5 + _lang_offset(lang.code)) % len(_WHENS)]

    needs_tools = any(t.uses_tools for t in tasks)
    if needs_tools:
        # Which tools a task genuinely needs. Tag hints alone left generic tool-calling tasks
        # (action_tool_use) with an EMPTY set once distractors stopped padding the list -- a tool-calling
        # task with no tools. Task names are explicit here; tags stay as a fallback for new tasks.
        BY_TASK = {
            "action_tool_use": ["send_message", "set_reminder", "get_transport_route",
                                "get_weather_forecast", "get_current_datetime"],
            "tool_compute": ["run_statistics", "calculate", "convert_units"],
            "tool_database": ["search_db", "db_get_record", "db_insert_record"],
            "rag_document_qa": ["search_documents", "search_internet"],
            # search_internet returns LINKS, so fetch_page must be in scope or the second stage of the
            # search can never happen -- measured: fetch_page was called 0 times without this.
            "tool_search_answer": ["search_internet", "fetch_page"],
            "retrieval_insufficient": ["search_internet", "fetch_page", "search_documents"],
            "tool_chain_insufficient": ["search_internet", "fetch_page", "search_documents"],
            "health_advice": ["lookup_health_guidance", "find_health_facility"],
            "health_triage": ["lookup_health_guidance", "find_health_facility"],
            "financial_analysis": ["get_exchange_rate", "calculate", "get_market_prices", "run_statistics"],
        }
        BY_TAG = {
            "rag": "search_documents", "extractive-qa": "search_documents",
            "knowledge-boundary": "search_internet", "insufficient-context": "search_internet",
            "health-advice": "lookup_health_guidance", "health-triaging": "lookup_health_guidance",
            "financial-analysis": "get_exchange_rate",
        }
        required: list[str] = []
        for t in tasks:
            for name in BY_TASK.get(t.name, []):
                if name not in required:
                    required.append(name)
            for tag, name in BY_TAG.items():
                if tag in t.tags and name not in required:
                    required.append(name)
        if not required:
            # A task marked uses_tools with nothing mapped: give it a small plausible set rather than none.
            required = rng.sample([t.name for t in seed.tools], k=3)

        # The generator sees ONLY these. The 2-3 irrelevant "distractor" tools that must appear in the
        # finished sample are chosen here but INJECTED AT POST-PROCESSING (see
        # postprocess_gen._rebuild_system_message). Two reasons, both measured:
        #   * the generator called them 12 times in 35 calls once few-shot exemplars were added, so a third
        #     of samples had to be discarded for teaching the exact behaviour distractors exist to prevent;
        #   * their definitions cost 300-1,200 input tokens per request for no generative benefit.
        # The training signal is identical: the target model learns from the finished catalogue and cannot
        # tell whether a definition was in the generator's prompt or added afterwards.
        # A tool named in any of this conversation's task descriptions ("calls search_documents", "uses
        # calculate / get_exchange_rate") must never be a distractor: the brief would be telling the
        # generator to use a tool the catalogue withholds, which is the same self-contradiction that once
        # had tasks referencing out-of-scope tools.
        # seed.details counts too: the SFT brief names search_documents when it explains that RAG is
        # modelled as a tool call, and details goes into the system prompt.
        # Tool DESCRIPTIONS name other tools too -- search_internet's says "you must then call fetch_page" --
        # and those descriptions are rendered into the prompt, so a tool named there cannot be a distractor.
        req_desc = [t.description for t in seed.tools if t.name in required]
        mentioned = " ".join([t.description for t in tasks] + req_desc + [seed.details])
        plausible = [t.name for t in seed.tools
                     if t.name not in required and t.name not in mentioned]
        rng.shuffle(plausible)
        distractors = plausible[:rng.randint(2, 3)]
        tool_names = list(required)
        rng.shuffle(tool_names)
    else:
        tool_names, distractors = [], []

    # tool conversations get an extra exchange of headroom (see the seed's conversation policy)
    _hi = int(seed.conversation.get("max_user_turns_with_tools", 6) if needs_tools
              else seed.conversation.get("max_user_turns", 5))
    _lo = int(seed.conversation.get("min_user_turns", 3))
    # A low-resource language is capped lower: six turns of shaky Fon is six turns of unusable data, and the
    # generator spends its fluency budget on quantity instead of correctness.
    _hi = min(_hi, _TIER_TURNS.get(lang.tier, 6))
    if document and not document.get("rag"):
        # The sample's purpose is the summary: ask for it, then optionally one or two follow-up questions
        # answered from the same document, which is the part that teaches grounding over a long context.
        # More than that and the document stops being the subject of the conversation.
        _lo, _hi = 1, 3
    n_user = rng.randint(min(_lo, _hi), _hi)
    verbs = sorted({v for t in tasks for v in t.task_plan}) or ["<|chat|>"]
    fmt = _format_brief(lang.code, lang.name, " ".join(sorted(VALID_VERBS)))

    task_lines = "\n".join(f"  * {t.name} [{', '.join(t.tags)}]: {t.description}" for t in tasks)
    lo_m, hi_m = seed.conversation.get("target_messages_with_tools", [12, 16])
    length_hint = (f"Because this conversation uses tools, expect {lo_m}-{hi_m} messages in total once the "
                   f"tool-call turns and their results are included. That is correct -- do not drop tool use "
                   f"to hit a smaller number."
                   if needs_tools else
                   "This conversation uses no tools, so it should stay compact: one assistant reply per user "
                   "message and nothing else.")
    verbs_line = ("task_plan verbs for this conversation (use them in the order each turn carries them out): "
                  + "".join(verbs))
    # Exemplars are attached for the low-resource tier, where structure collapses, and can be forced
    # everywhere with DATA_GEN_FEWSHOT=1. They live in the SYSTEM prompt, so they stay part of the shared
    # prefix: constant per (kind, language) and free under vLLM prefix caching.
    want_fewshot = lang.tier == "low" or os.environ.get("DATA_GEN_FEWSHOT") == "1"
    shots = _fewshot_block() if want_fewshot else ""
    # Part of the shared prefix, so the tier's simplicity ceiling is free under prefix caching.
    system = (f"{seed.details}\n\n{fmt}{shots}{_simplicity(lang)}\n"
              f"You are generating training conversations in {lang.name} ({lang.code}). "
              f"Language guidance: {lang.guidance}\nReturn strict JSON only, no commentary.")

    tools_section = ""
    if tool_names:
        tools_section = (
            f"\nTOOLS available in this conversation (put this exact JSON in the system message):\n"
            f"{_tool_block(seed, tool_names)}\n\nHow these tools behave when called:\n"
            f"{_tool_behaviour(seed, tool_names)}\n"
            f"\nEvery tool listed above is applicable to this conversation. Use the ones the tasks call for.\n")
    else:
        tools_section = ("\nThis conversation has NO tools and NO system message. If the user asks something "
                         "the assistant does not know, the correct behaviour is to say so plainly.\n")

    lang_name = lang.name
    if not rl:
        _msg = "message" if n_user == 1 else "messages"
        shape = f"""Produce ONE conversation with EXACTLY {n_user} {_msg} of role "user" -- no more, no
fewer. Each is answered by the assistant, and the conversation ends on an assistant message. The assistant may
add tool-call turns and their role="tool" results in between; those are not user messages and do not count.
Count your user messages before you finish: there must be exactly {n_user}.
{length_hint}

Return JSON:
{{"turns": [ ...turn objects exactly as the OUTPUT CONTRACT describes... ],
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
{{"prompt_turns": [ ...turn objects exactly as the OUTPUT CONTRACT describes... ],
  "responses": [{{"content": "<full assistant turn incl. special tokens>", "quality": "best"|"partial"|"worst", "why": "<one line, English>"}}],
  "tasks": [...], "tags": [...],
  "confidence": <float 0-1: your honest estimate that the ranking is right and the text is fluent {lang_name}>}}"""

    # A long-document sample covers ONE task, so "change subject at least once" would be self-contradictory.
    mix_line = ("Task for this conversation:" if len(tasks) == 1 else
                "Tasks this conversation must cover (mix them, change subject at least once):")
    doc_section = ("\n" + document_brief(document) + "\n") if document else ""

    user = f"""LANGUAGE DIRECTION for this conversation -- {io['key']}:
{io['brief']}
Set <|input_lang|> to <{io['expect_markers'][0]}> and <|target_lang|> to <{io['expect_markers'][1]}> on every
assistant turn that produces a <response>. Where the two differ, that difference is the point of the sample:
the user asks in one language for an answer in another, and the assistant complies.

{mix_line}
{task_lines}
{doc_section}

{verbs_line}

Ground the content in: {DOMAINS[domain].name} / {subtopic}.
It is {when}; anything time-dependent (a reminder, a forecast, a rate, "today") must fit that.
The assistant's identity in the system message will be: "{identity}" -- keep its self-references consistent
with that, and do not assume any other name.
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
        # A long-document sample needs LESS, not more: the document arrives as input and is referred to by a
        # placeholder, so this completion holds only the summary and its follow-ups.
        max_tokens=4096 if document else 6144,
        temperature=0.9,
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
                  # injected after generation, so the finished sample still teaches tool SELECTION
                  "distractor_definitions": _tool_defs(seed, distractors),
                  "domain": domain, "subtopic": subtopic, "n_user_turns": n_user,
                  "io_direction": io["key"], "io_other_lang": io["other"],
                  "identity": identity, "when": when,
                  # The turn bounds this row was actually asked for, so the validator checks the request that
                  # was made rather than the seed-wide floor -- a long-document sample legitimately has one
                  # user turn, which the global min_user_turns=3 would reject.
                  "min_user_turns": _lo, "max_user_turns": _hi,
                  "document": document, "expect_markers": io["expect_markers"]},
    )


def build_request(seed: Seed, row: dict, document: Optional[dict] = None) -> Request:
    """`document` is a finished stage-1 document (see longdocs.py) for the long-document tasks. Without one,
    a long-document row still builds -- the generator is then asked to write the document itself, which is
    the behaviour that produced 635-word "long" documents, so generate.py always supplies one."""
    if seed.kind == "pretrain":
        return _pretrain_request(seed, row)
    return _sft_like_request(seed, row, rl=(seed.kind == "rl"), document=document)


# ---------------------------------------------------------------------------------------- packing
#
# One request, several conversations. The system prompt -- seed brief, format rules, tool catalogue -- is
# ~1,200 tokens and identical for every row of the same (kind, language). Asking for N conversations in one
# call pays that once instead of N times, which is what turns a 1,000-request/day free tier into 1,000 x N
# samples/day. The ceiling is output length, not context: gemma-4 has 262k of context but a single completion
# has to hold all N conversations, so N trades throughput against truncation. Sweep it (scripts/sweep_pack.py)
# rather than guessing.


# Tasks whose sample carries a long document. They are never packed: each already carries 4,000-16,000 tokens
# of document as input, and they are generated in stages (longdocs.py) rather than in one request.
UNPACKABLE_TASKS = DOCUMENT_TASKS


def _packed_pretrain_request(seed: Seed, rows: list[dict]) -> Request:
    """N pretraining documents in one request.

    Worth doing wherever the limit is REQUESTS rather than tokens -- an API's free tier, or a daily quota. The
    system prompt is the seed brief plus the language's guidance, ~850 of a single request's ~1,150 input
    tokens and byte-identical for every row of the same language, so N documents in one call pay it once and
    turn a 1,000-request/day allowance into 1,000 x N documents.

    NOT used on the vLLM path, deliberately: there `enable_prefix_caching` already computes that shared prefix
    once per group, achieving the same saving without the failure mode packing brings -- one malformed reply
    costing N samples instead of one.
    """
    singles = [_pretrain_request(seed, r) for r in rows]
    system = singles[0].messages[0]["content"]      # identical by construction; that is the point
    lang = _lang_spec(seed, rows[0]["lang"])

    specs = []
    for i, req in enumerate(singles, start=1):
        md = req.metadata
        d = DOMAINS[md["domain"]]
        g = GENRES[md["genre"]]
        lo, hi = md["target_words"]
        specs.append(
            f"--- DOCUMENT {i} of {len(rows)} ---\n"
            f"Domain: {d.name} -- {d.description}\n"
            f"Sub-topic: {md['subtopic']}\n"
            f"Genre: {getattr(g, 'text', md['genre'])}\n"
            f"Register: {md['register']}\n"
            f"Length: {lo}-{hi} words.")

    user = (
        f"Write {len(rows)} SEPARATE documents in {lang.name}, one for each specification below.\n\n"
        f"They share nothing. Different sub-topics, different genres, different registers. Do not let them echo "
        f"one another: no repeated openings, no reused names, numbers, villages or examples across documents. A "
        f"reader must not be able to tell they were written together.\n\n"
        + "\n\n".join(specs)
        + f"""

Every document must meet the same bar:
- Entirely in {lang.name}. No English except words the language genuinely borrows.
- Explain HOW and WHY things work, not just what they are called. Mechanism over name-dropping.
- Locally grounded: real West African settings, foods, prices, institutions, seasons.
- No invented statistics, no fake citations, no made-up named people presented as real.
- Plain continuous prose. No markdown, no headings, no bullet lists, no chat markup.
- NEVER REPEAT YOURSELF, within a document or across them. No sentence may appear twice anywhere in this
  reply. A repeated clause makes that document worthless and it will be discarded.
- If you cannot fill a document's word range without repeating, write about a more concrete aspect of its
  sub-topic -- one person's day, one tool, one season, one price. The lower bound is a HARD FLOOR: a shorter
  document is discarded too, so neither padding nor stopping early works.

Return strict JSON: {{"documents": [ ... {len(rows)} objects, in the order given above ... ]}}
Each object: {{"title": "<short natural title in {lang.name}>", "text": "<the document>",
"language_self_check": <true only if the whole text is fluent {lang.name}>,
"confidence": <float 0-1: your honest estimate that this text is accurate AND fluent {lang.name}>}}
If you cannot complete one properly, still emit an object for it with an empty text rather than shifting the
others out of order."""
    )
    return Request(
        custom_id="pack__" + "|".join(r["custom_id"] for r in rows),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=min(32_000, sum(r.max_tokens for r in singles) + 512),
        temperature=0.95, top_p=0.95,
        response_format={"type": "json_object"},
        metadata={"packed": True, "kind": "pretrain", "lang": rows[0]["lang"],
                  "members": [r.metadata for r in singles],
                  "custom_ids": [r["custom_id"] for r in rows]},
    )


def build_packed_request(seed: Seed, rows: list[dict]) -> Request:
    """One Request carrying `len(rows)` independent specs. Rows must share a language."""
    if not rows:
        raise ValueError("no rows")
    langs = {r["lang"] for r in rows}
    if len(langs) != 1:
        raise ValueError(f"a packed request must be single-language, got {sorted(langs)}")
    if seed.kind == "pretrain":
        return _packed_pretrain_request(seed, rows)
    singles = [_sft_like_request(seed, r, rl=(seed.kind == "rl")) for r in rows]
    system = singles[0].messages[0]["content"]          # identical by construction; this is the whole point

    specs = []
    for i, (row, req) in enumerate(zip(rows, singles), start=1):
        # Keep each spec's own body, minus the generic shape instructions that now apply to all of them.
        body = req.messages[1]["content"]
        specs.append(f"=================== SAMPLE {i} of {len(rows)} ===================\n{body}")

    kind_key = "samples"
    user = (
        f"You will produce {len(rows)} INDEPENDENT samples in one reply. They share nothing: different tasks, "
        f"different domains, different language directions, different tool sets. Do not let them echo each "
        f"other -- no repeated openings, no reused names, numbers or examples across samples. A reader must not "
        f"be able to tell they were written together.\n\n"
        + "\n\n".join(specs)
        + f"\n\n=================== OUTPUT ===================\n"
        f'Return strict JSON: {{"{kind_key}": [ ... {len(rows)} objects, in the order given above ... ]}}\n'
        f"Each object has exactly the shape described in its own SAMPLE block. If you cannot complete a "
        f"sample properly, still emit an object for it with an empty messages list rather than shifting the "
        f"others out of order."
    )
    return Request(
        custom_id="pack__" + "|".join(r["custom_id"] for r in rows),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        # Output is the binding constraint: each conversation runs 700-2,500 tokens.
        max_tokens=min(60000, 3000 * len(rows) + 1000),
        temperature=0.95, top_p=0.95,
        response_format={"type": "json_object"},
        tools=singles[0].tools,
        metadata={"packed": True, "kind": seed.kind, "lang": rows[0]["lang"],
                  "members": [r.metadata for r in singles],
                  "custom_ids": [r["custom_id"] for r in rows]},
    )
