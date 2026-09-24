#!/usr/bin/env python3
"""Build seeds/{pretrain,sft,rl}.json -- the single source of truth for every generation run.

    python seeds/build_seeds.py            # write all three
    python seeds/build_seeds.py --kind sft --print

Edit THIS file, not the JSON: the JSON is generated and overwritten. Volumes, task mix, the tool catalogue
and the `details` brief all live here so a change is one reviewable diff that every provider and platform
picks up at once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.seed import LanguageSpec, Seed, TaskSpec, ToolSpec

# ---------------------------------------------------------------------------
# What the data is FOR. Injected into every meta-prompt so the generator knows
# it is writing for a very small model, not for a frontier one.
# ---------------------------------------------------------------------------

PHILOSOPHY = """\
The target model is SabiYarn, a ~306M-parameter mixture-of-experts causal LM for 12 West African languages
plus English. It is SMALL. It cannot and must not be expected to memorise the world's facts.

The goal is a model with solid general understanding of the world -- how physical things work, how people
and institutions behave, basic health, basic technology -- plus two learned reflexes:

  1. THINK FIRST. When a question arrives, reason briefly inside <think>...</think> about what is being
     asked, whether it already knows, and what it would need in order to answer.
  2. USE TOOLS, OR SAY YOU CANNOT. If it lacks the knowledge and a suitable tool exists, call the tool,
     read the result, and answer from it. If no suitable tool exists, say plainly that it does not know
     and is not connected to a knowledge source. If a tool ran but the result does not actually answer
     the question, say THAT, rather than inventing an answer from the irrelevant result.

The canonical example. Asked "what is AWS?", a well-trained SabiYarn should NOT recite a memorised
definition. It should think along the lines of "I have not come across this before; it looks like a
technical product name; I should look it up", then call search_internet with a query like "what is AWS",
read the returned snippet, and answer from it in the user's language. With no search tool available, the
correct behaviour is to say it does not know and cannot look it up right now.

Aim for the register of a bright, curious 16-year-old: confident about everyday things, honest about what
it has not learned, and quick to look something up rather than bluff. Never sycophantic, never padded.
Confabulation is the single worst failure mode; admitting ignorance is always preferred to a plausible
invention, and the training data must make that trade-off visible over and over."""

SPECIAL_TOKENS = {
    "bos": "<s>", "eos": "</s>",
    "roles": {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"},
    "think": ["<think>", "</think>"],
    "task_plan": ["<task_plan>", "</task_plan>"],
    "tool_call": ["<tool_call>", "</tool_call>"],
    "tool_result": ["<tool_response>", "</tool_response>"],
    "lang_markers": {"input": "<|input_lang|>", "target": "<|target_lang|>"},
    "response": "<response>",
    "context": ["<context>", "</context>"],
    "task_plan_verbs": [
        "<|chat|>", "<|generate|>", "<|edit|>", "<|data_extract|>", "<|math|>", "<|JSON|>", "<|code|>",
        "<|plan|>", "<|analyze|>", "<|recommend|>", "<|explain|>", "<|debug|>", "<|RAG|>",
        "<summarize>", "<translate>", "<classify>", "<NER>", "<identify>", "<qa>",
    ],
    "domain_flags": [
        "<|is_technical|>", "<|is_legal|>", "<|is_medical|>", "<|is_scientific|>",
        "<|is_non_technical|>", "<|is_financial|>",
    ],
    "label_tokens": {
        "sentiment": "<sentiment>", "topic": "<topic>", "intent": "<intent>", "toxic": "<toxic>",
        "hate": "<|hate|>", "lang_id": "<lang_ID>", "ner_tag": "<tag>", "answer": "<answer>",
        "title": "<title>", "headline": "<headline>", "summary": "<summary>", "question": "<question>",
    },
    "_warnings": [
        "Tokenizer ids 52050-52115 are ABOVE the model's vocab_size (52050), so they can never be embedded. "
        "That range holds <|hate|> and ~65 other-language tags. Do not use them until the embedding is "
        "resized; <toxic> (52008) is in range and is what the toxicity task uses.",
    ],
}

# The serialized shape of one assistant turn. Kept here so generator, post-processor and the eventual
# training code all agree; tests assert a round trip through this against the real chat template.
ASSISTANT_FORMAT = {
    "plain": "<|input_lang|>{src}<task_plan>{verbs}</task_plan><|target_lang|>{tgt}<response>{text}",
    "with_think": ("<|input_lang|>{src}<think>{reasoning}</think>"
                   "<task_plan>{verbs}</task_plan><|target_lang|>{tgt}<response>{text}"),
    "tool_call": "<|input_lang|>{src}<think>{reasoning}</think><tool_call>{name}\n{arguments_json}</tool_call>",
    "notes": (
        "src/tgt are language tags like <yor>; for translation they differ and must match the real "
        "direction. A turn that calls a tool emits ONLY the tool_call form (no <response>); the tool result "
        "comes back as a separate role='tool' message and the assistant then speaks again. "
        "<think> IS ALWAYS IN ENGLISH, regardless of the conversation language -- it is the model's internal "
        "scratchpad, English keeps it short and consistent across all 13 languages, and it is never shown to "
        "the user. The <response> is always in the target language. "
        "<task_plan> is a PLAN, not a label: for a multi-step turn it lists the verbs in the order they will "
        "be carried out (e.g. <|RAG|><|analyze|><|explain|>), so the model learns to commit to a sequence "
        "before executing it."
    ),
    "confidence": (
        "Every generated sample carries `confidence`, a float in [0, 1]: the generating model's own estimate "
        "that the sample is correct AND fluent in the target language. Low-confidence samples are kept but "
        "flagged, so a filter threshold can be chosen after inspection rather than guessed up front."
    ),
}

# ---------------------------------------------------------------------------
# Languages. `samples` = documents (pretrain) or conversations (sft/rl).
# Volumes follow the owner's floors (>=60k pcm, >=30k urh/efi and the rest) and
# then scale by how much the model benefits: pcm is English-lexified so it
# transfers fastest; yor/hau/ibo have the most real downstream use; eng is
# present only to keep the model's English from decaying.
# ---------------------------------------------------------------------------

VOLUMES_DIR = Path(__file__).resolve().parent / "volumes"

# Language metadata that is NOT a volume (tier + orthography traps). Counts live in volumes/<kind>.yaml so
# they can be retuned without touching code.
_LANG_META = {
    "pcm": ("Nigerian Pidgin", "high", "English-lexified creole. Keep it genuinely Pidgin, not English with dropped copulas."),
    "yor": ("Yoruba", "medium", "Tone marks and under-dots are mandatory and meaning-bearing: ọ ẹ ṣ plus diacritics."),
    "hau": ("Hausa", "medium", "Use hooked letters ɓ ɗ ƙ and the apostrophe in 'y correctly."),
    "ibo": ("Igbo", "medium", "Dotted vowels ị ọ ụ and the nasal ṅ are required."),
    "twi": ("Twi (Akan)", "medium", "Use ɛ and ɔ. Keep Asante Twi consistent within a sample."),
    "aka": ("Akan", "medium", "Closely related to Twi; keep them distinguishable."),
    "efi": ("Efik", "low", "Very low-resource. Watch for drift into Ibibio. Prefer short concrete sentences."),
    "urh": ("Urhobo", "low", "Very low-resource. Prefer short, concrete sentences over ambitious prose."),
    "fon": ("Fon", "low", "Uses ɖ ɛ ɔ and tone marks. Do not drift into French."),
    "ewe": ("Ewe", "low", "Uses ɖ ƒ ŋ ɣ ʋ and tone marks."),
    "ful": ("Fulah", "low", "Uses ɓ ɗ ƴ ŋ. Latin script only -- not Adlam."),
    "fuv": ("Nigerian Fulfulde", "low", "Nigerian variety specifically, distinct from Pular/Fuuta."),
    "eng": ("English", "high", "Plain, concrete English. No flowery register."),
}


def _load_volumes(kind: str) -> tuple[dict[str, int], dict[str, float]]:
    import yaml
    raw = yaml.safe_load((VOLUMES_DIR / f"{kind}.yaml").read_text(encoding="utf-8")) or {}
    counts = raw.get("samples_per_language") or {}
    unknown = set(counts) - set(_LANG_META)
    if unknown:
        raise SystemExit(f"volumes/{kind}.yaml: unknown language(s) {sorted(unknown)}")
    return counts, (raw.get("yield_by_tier") or {})


def _languages(kind: str) -> tuple[list[LanguageSpec], dict[str, float]]:
    counts, yields = _load_volumes(kind)
    langs = [LanguageSpec(code=c, name=_LANG_META[c][0], tier=_LANG_META[c][1],
                          samples=n, guidance=_LANG_META[c][2])
             for c, n in counts.items()]
    return langs, yields


# ---------------------------------------------------------------------------
# Tool catalogue. Deliberately small: a 306M model has to learn tool SELECTION,
# and overlapping descriptions are what make that fail. Every tool declares its
# failure modes, because conversations where a tool returns nothing useful are
# as important to train on as the happy path.
# ---------------------------------------------------------------------------

def _t(name, description, props, required, returns, failures, domains):
    return ToolSpec(name=name, description=description,
                    parameters={"type": "object", "properties": props, "required": required},
                    returns=returns, failure_modes=failures)


_STR = {"type": "string"}
_INT = {"type": "integer"}

# Tool catalogue. Deliberately broad in KIND (retrieval, computation, storage, domain lookup, actions) and
# narrow in overlap: a 306M model has to learn tool SELECTION, and two tools that sound alike make that
# impossible. Each conversation sees only 4-7 of these, of which 2-3 are deliberately irrelevant.
TOOLS = [
    # -- retrieval
    _t("search_internet", "Search the public internet for information the assistant does not already know. "
       "Use for named products, companies, people, places, events, or any technical term it has not met.",
       {"query": {**_STR, "description": "Short search query in English, 2-8 words."}}, ["query"],
       "2-4 snippets of 1-3 sentences, sometimes with a source name. Often partial, sometimes stale.",
       ["no results", "results about a different sense of the word", "mentions the term but never defines it"],
       ["all"]),
    _t("search_documents", "Search the documents supplied in this conversation for relevant passages. Use "
       "before answering any question about a supplied document. Does NOT search the internet.",
       {"query": {**_STR, "description": "What to look for."},
        "top_k": {**_INT, "description": "1-5, default 3."}}, ["query"],
       "Verbatim passages from the supplied document with a rough location.",
       ["the document does not discuss it", "passages are topically near but do not answer"], ["all"]),
    # -- databases
    _t("search_db", "Search a structured table by free text or field filters. Use when the user asks about "
       "stored records: customers, stock, patients, students, transactions.",
       {"table": {**_STR, "description": "e.g. 'inventory', 'patients', 'transactions'."},
        "query": {**_STR, "description": "Free text or 'field=value' filters."},
        "limit": _INT}, ["table", "query"],
       "Matching rows as JSON objects, newest first, with a total match count.",
       ["table does not exist", "zero matching rows", "more matches than the limit returned"],
       ["economy", "livelihoods", "health", "society"]),
    _t("db_get_record", "Fetch one record from a table by its id.",
       {"table": _STR, "record_id": _STR}, ["table", "record_id"],
       "The full record as JSON, or a not-found error.",
       ["id does not exist", "id belongs to a different table"], ["economy", "health", "society"]),
    _t("db_insert_record", "Save a new record to a table. Only call this after the user has confirmed the "
       "values, and echo back what was saved.",
       {"table": _STR, "record": {"type": "object", "description": "Field/value pairs to store."}},
       ["table", "record"],
       "The stored record with its new id and a timestamp.",
       ["required field missing", "duplicate of an existing record", "write rejected: read-only table"],
       ["economy", "livelihoods", "health"]),
    # -- computation
    _t("calculate", "Evaluate an arithmetic expression. Use for any calculation instead of doing it mentally.",
       {"expression": {**_STR, "description": "e.g. '4500 * 12 - 3000'"}}, ["expression"],
       "The numeric result.", ["malformed expression", "division by zero"], ["all"]),
    _t("run_statistics", "Compute a summary statistic over a list of numbers: mean, median, mode, min, max, "
       "sum, standard deviation, percentage change, or correlation between two lists.",
       {"operation": {**_STR, "description": "mean|median|mode|min|max|sum|stdev|pct_change|correlation"},
        "data": {"type": "array", "items": {"type": "number"}},
        "data2": {"type": "array", "items": {"type": "number"},
                  "description": "Second series, for correlation only."}}, ["operation", "data"],
       "The statistic, plus n and the unit where obvious.",
       ["fewer than 2 values for stdev/correlation", "series of unequal length", "non-numeric value in data"],
       ["knowledge", "economy", "health", "environment"]),
    _t("convert_units", "Convert between units of length, mass, volume, area, temperature or currency-free "
       "quantities like bags and tonnes.",
       {"value": {"type": "number"}, "from_unit": _STR, "to_unit": _STR}, ["value", "from_unit", "to_unit"],
       "The converted value with both units named.",
       ["unknown unit", "incompatible dimensions (mass to length)"], ["livelihoods", "economy", "knowledge"]),
    _t("get_exchange_rate", "Get the current exchange rate between two currencies by ISO code.",
       {"base": {**_STR, "description": "e.g. 'NGN'"}, "quote": {**_STR, "description": "e.g. 'USD'"}},
       ["base", "quote"], "A rate and the timestamp it was quoted at.",
       ["unknown currency code", "rate unavailable for that pair"], ["economy"]),
    # -- domain lookups
    _t("lookup_health_guidance", "Look up plain-language public-health guidance on a symptom, condition, "
       "medicine or prevention topic. Use for any health question before answering. Not a diagnosis.",
       {"topic": {**_STR, "description": "Symptom, condition or medicine, in English."}}, ["topic"],
       "What it is, warning signs, home care, and when to go to a clinic.",
       ["topic too vague to match", "guidance is for adults but the question was about an infant"], ["health"]),
    _t("find_health_facility", "Find clinics, hospitals or pharmacies near a place, optionally filtered by "
       "the service needed.",
       {"location": _STR, "service": {**_STR, "description": "e.g. 'antenatal', 'dialysis', 'pharmacy'."}},
       ["location"], "Facility names with distance, opening hours and whether the service is offered.",
       ["no facilities found for that service", "location too vague"], ["health"]),
    _t("lookup_crop_guidance", "Look up agricultural guidance for a crop or livestock problem: planting "
       "times, pests, diseases, storage.",
       {"crop": _STR, "issue": {**_STR, "description": "e.g. 'leaf spots', 'when to plant', 'storage rot'."}},
       ["crop"], "Practical steps, with local seasons where relevant.",
       ["crop not covered", "guidance assumes irrigation the user does not have"],
       ["livelihoods", "environment"]),
    _t("get_market_prices", "Get recent prices for a commodity at a named market.",
       {"commodity": {**_STR, "description": "e.g. 'maize', 'tomatoes', 'cement'."}, "market": _STR},
       ["commodity"], "Recent price per unit with the date and market name.",
       ["no data for that market", "prices are weeks stale", "unit differs from what the user assumed"],
       ["economy", "livelihoods"]),
    _t("get_weather_forecast", "Get the weather forecast for a location.",
       {"location": _STR, "days": {**_INT, "description": "1-7, default 3."}}, ["location"],
       "Daily forecast: rain chance, temperature range, wind.",
       ["location not recognised", "forecast only available for 2 of the requested days"],
       ["environment", "livelihoods"]),
    _t("get_transport_route", "Find how to travel between two places: route, rough fare, duration.",
       {"origin": _STR, "destination": _STR}, ["origin", "destination"],
       "Route options with rough fare and travel time.",
       ["no route found", "fare is out of date"], ["society", "economy"]),
    _t("lookup_legal_info", "Look up plain-language information about a law, right or official procedure.",
       {"topic": {**_STR, "description": "e.g. 'tenancy notice period', 'registering a business'."},
        "country": _STR}, ["topic"],
       "Plain-language summary with the relevant authority named. Not legal advice.",
       ["topic not covered for that country", "summary is generic where the user needed specifics"],
       ["society"]),
    _t("get_current_datetime", "Get the current date, time and timezone. Use whenever the answer depends on "
       "today's date.", {"timezone": {**_STR, "description": "IANA zone, e.g. 'Africa/Lagos'."}}, [],
       "ISO timestamp and timezone name.", ["unknown timezone name"], ["all"]),
    # -- actions
    _t("send_message", "Send a text message on the user's behalf. Only after the user has clearly asked, and "
       "confirm recipient and wording first if either is unclear.",
       {"recipient": {**_STR, "description": "Name or phone number."},
        "body": {**_STR, "description": "The message text, in the user's language."}}, ["recipient", "body"],
       "Delivery confirmation, or an error.",
       ["recipient not found", "network failure", "two contacts match the name"], ["all"]),
    _t("set_reminder", "Create a reminder for the user at a specific time.",
       {"when": {**_STR, "description": "ISO datetime or a phrase like 'tomorrow 07:00'."}, "text": _STR},
       ["when", "text"], "Confirmation with the resolved absolute time.",
       ["time in the past", "unparseable time phrase"], ["all"]),
    _t("translate_text", "Translate text between the languages this assistant supports. Use only when the "
       "assistant is not confident producing the translation itself.",
       {"text": _STR, "target_language": {**_STR, "description": "Language code, e.g. 'yor'."}},
       ["text", "target_language"], "The translated text.",
       ["unsupported language code", "input too long"], ["all"]),
]

# ---------------------------------------------------------------------------
# Task mix. `share` is the target share of EXCHANGES (a conversation of 8
# messages holds ~4 exchanges and normally mixes 2-4 different tasks).
#
# The three knowledge-boundary behaviours together take ~24% -- by far the
# largest block -- because they are the entire point of the project and are the
# hardest thing to teach. Everything else is the general competence that makes
# the model worth talking to in the first place.
# ---------------------------------------------------------------------------

def _task(name, tags, share, description, plan=(), flags=(), tools=False, think=False, langs=None, notes=""):
    return TaskSpec(name, list(tags), description, share=share, task_plan=list(plan),
                    domain_flags=list(flags), uses_tools=tools, requires_think=think,
                    languages=langs, notes=notes)


# Every tag in schemas.seed.TAGS gets its own task, so each is separately countable, auditable and
# holdable-out. The three knowledge-boundary behaviours total ~22% -- the largest block -- because they are
# the point of the project and the hardest thing to teach.
SFT_TASKS = [
    # ---- knowledge boundary: the reason this corpus exists
    _task("tool_search_answer", ["tool-calling", "knowledge-boundary", "qa"], 9.0,
          "User asks about something the assistant plainly has not learned (a product, company, acronym, "
          "recent event, technical term). It thinks -- noting it does not recognise the term -- calls "
          "search_internet with a tight query, reads the snippets, and answers in the user's language, "
          "attributing what it found. It must NOT pretend prior familiarity.",
          plan=["<|explain|>"], tools=True, think=True),
    _task("no_tool_admit_unknown", ["knowledge-boundary", "qa"], 6.5,
          "Same, but NO tool in scope could help (or there are no tools at all). The assistant thinks, "
          "concludes it neither knows nor can look it up, and says so plainly and briefly. It may offer what "
          "general reasoning it can (what kind of thing the name looks like) without inventing facts, and "
          "does not apologise at length.",
          plan=["<|chat|>"], think=True),
    _task("retrieval_insufficient", ["insufficient-context", "rag", "tool-calling"], 6.5,
          "A tool is called and the result does NOT answer the question -- empty, a different sense of the "
          "term, or topically adjacent. The assistant must notice this in <think>, say that what it found "
          "does not answer the question, and stop. Optionally one better re-query, then admit failure. "
          "Inventing an answer from an irrelevant snippet is the exact failure this prevents.",
          plan=["<|explain|>"], tools=True, think=True),
    # ---- retrieval / grounding
    _task("rag_document_qa", ["rag", "extractive-qa", "tool-calling"], 8.0,
          "A document is supplied; the user asks about it. The assistant calls search_documents with a query "
          "it composes itself, grounds its answer in the returned passages, and says where it came from. "
          "Asked something the document does not cover, it says so rather than using general knowledge.",
          plan=["<|RAG|>"], tools=True, think=True),
    # ---- tool families beyond retrieval
    _task("tool_compute", ["tool-calling", "structured-output"], 4.0,
          "Quantitative questions answered with run_statistics / calculate / convert_units rather than mental "
          "arithmetic: average rainfall over a season, price change between two months, bags to tonnes, "
          "spread of test scores. The assistant shows the numbers it passed in and interprets the result in "
          "one sentence.",
          plan=["<|math|>", "<|analyze|>"], tools=True, think=True),
    _task("tool_database", ["tool-calling", "structured-output"], 4.0,
          "Record keeping with search_db / db_get_record / db_insert_record: look up a customer's balance, "
          "find stock below a threshold, save a new patient visit or sale. Before writing, the assistant "
          "confirms the values in words; after writing, it echoes what was stored. Includes zero-result "
          "searches and rejected writes.",
          plan=["<|data_extract|>", "<|plan|>"], tools=True, think=True),
    _task("action_tool_use", ["tool-calling"], 4.0,
          "The user asks for an action: send a message, set a reminder, check a route or a forecast. The "
          "assistant confirms any ambiguous parameter FIRST, then emits the call, then reports the result -- "
          "including cases where the tool fails and it explains the failure plainly.",
          plan=["<|plan|>"], tools=True, think=True),
    # ---- general understanding: the base the whole design rests on
    _task("world_knowledge_qa", ["qa", "multi-turn-chat"], 11.0,
          "Everyday how-and-why questions a curious 16-year-old could answer unaided: why bread rises, how a "
          "generator makes electricity, why the harmattan is dusty, how loan interest works, why boiling "
          "water makes it safe. Concrete, mechanism-first, no tools, no hedging.",
          plan=["<|explain|>"]),
    # ---- health
    _task("health_education", ["health-education"], 4.0,
          "Explaining a condition, medicine or prevention measure in plain language: what it is, how it "
          "spreads or develops, how it is prevented. Accurate, non-alarmist, locally grounded (malaria, "
          "typhoid, sickle cell, hypertension, immunisation).",
          plan=["<|explain|>"], flags=["<|is_medical|>"]),
    _task("health_advice", ["health-advice", "tool-calling"], 3.5,
          "A non-urgent symptom, and what to do about it. Calls lookup_health_guidance, then gives practical "
          "home care AND explicit signs that mean going to a clinic. Never diagnoses, never names a "
          "prescription dose, never discourages seeking care.",
          plan=["<|recommend|>"], flags=["<|is_medical|>"], tools=True, think=True),
    _task("health_triage", ["health-triaging"], 3.0,
          "Symptoms that may be serious. The job is sorting urgency: go now, go today, watch at home. Danger "
          "signs stated concretely (a child too weak to drink, bleeding in pregnancy, chest pain with "
          "breathlessness, a convulsion). Always errs toward seeking care.",
          plan=["<|recommend|>"], flags=["<|is_medical|>"], think=True),
    # ---- money
    _task("financial_analysis", ["financial-analysis", "tool-calling"], 3.5,
          "Small-trader and household money questions: margin on a sack of rice, whether a loan is worth it, "
          "daily contribution savings, currency conversion, market price trends. Uses calculate / "
          "get_exchange_rate / get_market_prices / run_statistics rather than mental arithmetic, and shows "
          "the working.",
          plan=["<|math|>", "<|analyze|>"], flags=["<|is_financial|>"], tools=True, think=True),
    # ---- structure and format
    _task("structured_output", ["structured-output"], 4.0,
          "Return strictly-valid JSON matching a schema stated in the prompt -- extracting fields from a "
          "passage, or formatting an answer. Emits the JSON and nothing around it.",
          plan=["<|JSON|>", "<|data_extract|>"]),
    # ---- translation: BOTH directions to English, and between the African languages
    _task("translation_english", ["translation"], 4.0,
          "Translate between this language and English, in BOTH directions (X->English and English->X). "
          "Preserve register, names and numbers. The <|input_lang|> and <|target_lang|> markers must differ "
          "and must match the actual direction.",
          plan=["<translate>"]),
    _task("translation_interlanguage", ["translation"], 3.0,
          "Translate between two of the supported African languages WITHOUT going through English in the "
          "output (e.g. Yoruba->Hausa, Twi->Ewe, Pidgin->Igbo). State both languages in the markers. This is "
          "the hardest translation direction and the one no public corpus covers.",
          plan=["<translate>"], think=True),
    _task("summarization", ["summarization"], 4.0,
          "Condense a passage to its substance. Both same-language and cross-language (summarise this English "
          "text in Hausa).",
          plan=["<summarize>"]),
    # ---- labelling: one task per tag so each is separately countable
    _task("topic_classification", ["topic-classification"], 2.0,
          "Assign a topic label to a short text, using the <topic> token. The answer is the label plus at "
          "most one clause of justification -- never an essay.",
          plan=["<classify>"]),
    _task("sentiment_analysis", ["sentiment-analysis"], 2.0,
          "Label sentiment (positive / negative / neutral, or mixed) with the <sentiment> token, on real-"
          "sounding text: market complaints, radio comments, product feedback.",
          plan=["<classify>"]),
    _task("intent_detection", ["intent-detection"], 1.5,
          "Identify what the user is actually trying to do (book, complain, ask a price, cancel, greet) with "
          "the <intent> token. Includes utterances whose surface form hides the intent.",
          plan=["<classify>", "<identify>"]),
    _task("toxicity_detection", ["toxicity-spam-detection"], 1.5,
          "Flag abusive, hateful or spam text with the <toxic> token, and say briefly which it is. Includes "
          "hard negatives: blunt or angry but not abusive, and local slang that only looks offensive.",
          plan=["<classify>"]),
    _task("language_identification", ["language-identification"], 1.5,
          "Identify which language a snippet is in, using <lang_ID>. Must include the genuinely hard cases: "
          "Twi vs Akan, Fulah vs Nigerian Fulfulde, Efik vs Ibibio, and Pidgin vs English.",
          plan=["<identify>"], think=True),
    _task("ner", ["ner"], 1.5,
          "Named-entity recognition over a sentence, returned as aligned token/tag pairs using <NER> and "
          "<tag>. Local person, place and organisation names specifically.",
          plan=["<NER>"]),
    _task("pos_tagging", ["pos-tagging"], 1.5,
          "Part-of-speech tagging over a sentence in the target language, as aligned token/tag pairs.",
          plan=["<NER>"]),
    # ---- generation
    _task("rephrasing_and_writing", ["text-rephrasing", "content-writing"], 4.0,
          "Rewrite for register, length or clarity; or compose something short and real -- a market notice, a "
          "condolence message, a school announcement, a radio advert.",
          plan=["<|edit|>", "<|generate|>"]),
    _task("general_chat", ["multi-turn-chat"], 2.5,
          "Ordinary conversational glue: greetings, follow-ups, clarifying questions, changing the subject. "
          "Keeps the model from sounding like a task-executor only.",
          plan=["<|chat|>"]),
]

# RL reuses the same task vocabulary but concentrates on what a judge can actually separate: honesty under
# uncertainty, correct tool choice, and grounding.
_RL_WEIGHTS = {
    "tool_search_answer": 15.0, "no_tool_admit_unknown": 12.0, "retrieval_insufficient": 14.0,
    "rag_document_qa": 13.0, "tool_compute": 6.0, "tool_database": 5.0, "action_tool_use": 6.0,
    "world_knowledge_qa": 9.0, "health_advice": 6.0, "health_triage": 5.0, "financial_analysis": 4.0,
    "structured_output": 2.0, "translation_english": 2.0, "translation_interlanguage": 1.0,
}
RL_TASKS = [
    TaskSpec(t.name, t.tags, t.description, share=_RL_WEIGHTS[t.name], task_plan=t.task_plan,
             domain_flags=t.domain_flags, uses_tools=t.uses_tools, requires_think=t.requires_think,
             languages=t.languages,
             notes="Candidates must differ in a way a judge can rank: one grounded and honest, one "
                   "confidently wrong or invented, one partially right (correct but uselessly hedged, right "
                   "answer via the wrong tool, or refusing when the answer WAS available). Never three "
                   "paraphrases of the same answer.")
    for t in SFT_TASKS if t.name in _RL_WEIGHTS
]

CONVERSATION = {
    "min_messages": 6, "max_messages": 10, "ends_with": "assistant",
    "counting": "A tool_call and its tool result count as messages. 6-10 covers roughly 3-5 user turns.",
    "task_mix_per_conversation": [2, 4],
    "theme_drift": "A conversation should change subject at least once -- that is what teaches the model to "
                   "track context rather than answer in isolation.",
    "system_prompt": "Present in ~70% of conversations; carries the tool catalogue as a JSON block. The "
                     "remaining 30% have no system prompt and no tools, which is where no_tool_admit_unknown "
                     "lives.",
    "think_policy": "Open with <think>...</think> whenever the turn involves a tool, a judgement about whether "
                    "the assistant knows something, or triage. Skip it for simple chat and labelling -- a model "
                    "that thinks about everything is as badly calibrated as one that never does.",
    "language_policy": "The whole conversation is in one language unless the task is translation or "
                       "cross-lingual summarisation. The assistant always replies in the language the user "
                       "wrote in.",
}

PRETRAIN_DETAILS = f"""{PHILOSOPHY}

THIS PHASE: pretraining documents. Plain continuous prose -- no chat markup, no special tokens, no
instructions, no question-and-answer shape. Each document is a self-contained piece of natural writing of
the requested genre, register and length, entirely in the target language.

What this corpus must give the model is WORLD MODEL, not facts to recite: how things work, what objects
and materials do, how institutions and markets and families behave, what health means day to day. Prefer
explanation and mechanism over lists of names and dates. A document about a generator should explain why
fuel makes a coil turn and why the room must be ventilated -- not list generator brands.

Health is a first-class domain here (6 domains, ~60 sub-topics: clinical, public health, maternal and
child, nutrition, mental wellbeing, disability). The model should come out of pretraining able to reason
about illness and care in plain terms, so that at SFT time it can triage and advise with tool support.

Coverage is driven by the sampler: every (domain, sub-topic) pair, every genre, every register is used
before any repeats. Do not let the generator drift toward the same five topics.

Quality bar: natural, locally grounded, factually careful, no invented statistics, no English code-switching
except where a language genuinely borrows the word."""

SFT_DETAILS = f"""{PHILOSOPHY}

THIS PHASE: supervised fine-tuning conversations. Each sample is a multi-turn conversation of 6-10 messages
ending in an assistant message, mixing 2-4 different tasks and changing subject at least once.

Conversations are serialized twice and BOTH are kept: (a) `messages`, a list of role/content dicts, and
(b) `text`, the exact string the SabiYarn chat template renders from it. The dict form is the source of
truth; the text form is what training consumes, and keeping both means a template change is a re-render
rather than a regeneration.

Tool use is the centrepiece. RAG is NOT a context-stuffing exercise -- it is modelled as a tool call: the
assistant emits a search_documents call with a query it composes itself, receives passages back as a tool
result message, and answers from those. The model must learn the whole loop, including the loop failing.

Every sample is tagged with `lang` and one or more task tags, so the mix can be audited and slices held out.

The hardest and most important quality bar: the assistant must be visibly honest about the edge of its
knowledge. Roughly a quarter of all exchanges are about exactly that -- looking something up, admitting it
cannot, or noticing that what came back does not answer the question."""

RL_DETAILS = f"""{PHILOSOPHY}

THIS PHASE: preference data. Each sample is a conversation prefix ending with a user message, plus 2-3
candidate assistant responses to that final turn, to be ranked.

Only the FINAL response is graded. The prefix is shared and fixed.

Candidates must differ along an axis a judge can actually rank, and the intended ranking is recorded:
  - one grounded, honest, correctly-tool-using response (best);
  - one that is confidently wrong -- invents a fact, answers from an irrelevant retrieval, or claims
    familiarity with something it was never told (worst; this is the behaviour we are training against);
  - optionally one that is partially right: correct but hedged into uselessness, or right answer with the
    wrong tool, or over-refusal where the answer was actually available.
Three paraphrases of the same answer are worthless here and must not be generated.

Over-refusal is a real failure too. If the answer IS available -- in the document, from the tool result, or
from ordinary world knowledge -- then refusing is the worse response, and some samples must teach that
direction so the model does not collapse into refusing everything."""


def build(kind: str) -> Seed:
    languages, yields = _languages(kind)
    target = {"name": "SabiYarn", "params": "306M", "architecture": "MoE, 12 layers, top-2 of up to 4 experts",
              "tokenizer": "BeardedMonster/SabiYarn-32k", "context": 4096,
              "philosophy": PHILOSOPHY}
    fmt = {"special_tokens": SPECIAL_TOKENS}
    if kind == "pretrain":
        return Seed(kind="pretrain", details=PRETRAIN_DETAILS, languages=languages,
                    target_model=target, yield_by_tier=yields,
                    format={**fmt, "output": "plain prose, no markup",
                            "columns": ["id", "lang", "domain", "subtopic", "genre", "title", "text"]}).validate()
    tasks = SFT_TASKS if kind == "sft" else RL_TASKS
    conv = dict(CONVERSATION)
    columns = ["id", "lang", "tags", "tasks", "messages", "text", "instruction", "input", "context", "response"]
    if kind == "rl":
        conv["responses_per_prompt"] = 3
        columns = ["id", "lang", "tags", "tasks", "prompt_messages", "prompt_text",
                   "response_1", "response_2", "response_3", "ranking", "rationale",
                   "instruction", "input", "context"]
    return Seed(kind=kind, details=SFT_DETAILS if kind == "sft" else RL_DETAILS,
                languages=languages, tasks=tasks, tools=TOOLS, conversation=conv, target_model=target,
                yield_by_tier=yields,
                format={**fmt, "assistant_turn": ASSISTANT_FORMAT, "columns": columns,
                        "column_notes": (
                            "messages/prompt_messages are the source of truth (role/content dicts, with "
                            "tool_calls on assistant turns and role='tool' results). text/prompt_text is the "
                            "chat-template rendering. instruction/input/context/response are flattened "
                            "conveniences derived from the LAST exchange, for tooling that expects the "
                            "Alpaca shape; they are lossy for multi-turn samples and must not be trained on "
                            "instead of messages.")}).validate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="all", choices=["all", "pretrain", "sft", "rl"])
    ap.add_argument("--print", action="store_true", help="print the plan instead of writing")
    args = ap.parse_args()
    kinds = ["pretrain", "sft", "rl"] if args.kind == "all" else [args.kind]
    for kind in kinds:
        seed = build(kind)
        if args.print:
            print(f"\n=== {kind}: target {seed.total_samples():,} samples "
                  f"({seed.total_requests():,} requests) across {len(seed.languages)} languages")
            if not seed.tasks:  # pretrain: the domain/genre sampler decides the mix, not a task list
                for l in seed.languages:
                    print(f"  {l.code:5s} {l.samples:>7,}  tier={l.tier}")
                continue
            for lang, tasks in seed.plan().items():
                top = sorted(tasks.items(), key=lambda kv: -kv[1])[:4]
                print(f"  {lang:5s} {sum(tasks.values()):>7,}  " + "  ".join(f"{k}={v:,}" for k, v in top))
        else:
            print(f"wrote {seed.save()}  target {seed.total_samples():,} samples "
                  f"-> {seed.total_requests():,} requests at the budgeted yield")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
