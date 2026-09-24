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
    "tool_result": ["<tool_result>", "</tool_result>"],
    "lang_markers": {"input": "<|input_lang|>", "target": "<|target_lang|>"},
    "response": "<response>",
    "context": ["<context>", "</context>"],
    "task_plan_verbs": [
        "<|chat|>", "<|generate|>", "<|edit|>", "<|data_extract|>", "<|math|>", "<|JSON|>", "<|code|>",
        "<|plan|>", "<|recommend|>", "<|explain|>", "<|debug|>", "<|RAG|>",
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
        "The chat template emits <tool_result>...</tool_result>, which is NOT in the tokenizer's special "
        "vocabulary and therefore costs ~5 byte-BPE tokens per tag. The tokenizer DOES have <tool_response> "
        "(52037) and </tool_response> (52038) as single tokens. Either change the template to emit "
        "<tool_response>, or add <tool_result> to the tokenizer -- but decide BEFORE generating, because "
        "the choice is baked into every generated sample.",
        "Token 52043 is '|analyze|>' -- it is missing its leading '<'. Do not use it in task plans; use "
        "<|explain|> or <|plan|> instead until the tokenizer is fixed.",
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
        "src/tgt are language tags like <yor>. verbs are one or more task_plan verbs. A turn that calls a "
        "tool emits ONLY the tool_call form (no <response>); the tool result comes back as a separate "
        "role='tool' message and the assistant then speaks again. Reasoning inside <think> is written in "
        "the TARGET language, short (1-3 sentences), and never restates the question."
    ),
}

# ---------------------------------------------------------------------------
# Languages. `samples` = documents (pretrain) or conversations (sft/rl).
# Volumes follow the owner's floors (>=60k pcm, >=30k urh/efi and the rest) and
# then scale by how much the model benefits: pcm is English-lexified so it
# transfers fastest; yor/hau/ibo have the most real downstream use; eng is
# present only to keep the model's English from decaying.
# ---------------------------------------------------------------------------

_LANGS = [
    # code   name                tier      pretrain   sft     rl    guidance
    ("pcm", "Nigerian Pidgin",   "high",     60000, 12000, 3000, "English-lexified creole. Keep it genuinely Pidgin, not English with dropped copulas."),
    ("yor", "Yoruba",            "medium",   40000,  8000, 2000, "Tone marks and under-dots are mandatory and meaning-bearing: ọ ẹ ṣ and the diacritics."),
    ("hau", "Hausa",             "medium",   40000,  8000, 2000, "Use hooked letters ɓ ɗ ƙ and the apostrophe in 'y correctly."),
    ("ibo", "Igbo",              "medium",   40000,  8000, 2000, "Dotted vowels ị ọ ụ and the nasal ṅ are required."),
    ("urh", "Urhobo",            "low",      30000,  5000, 1200, "Very low-resource. Prefer short, concrete sentences over ambitious prose."),
    ("efi", "Efik",              "low",      30000,  5000, 1200, "Very low-resource. Watch for drift into Ibibio."),
    ("twi", "Twi (Akan)",        "medium",   30000,  5000, 1200, "Use ɛ and ɔ. Keep Asante Twi consistent within a document."),
    ("ewe", "Ewe",               "low",      30000,  5000, 1200, "Uses ɖ ƒ ŋ ɣ ʋ and tone marks."),
    ("fon", "Fon",               "low",      30000,  5000, 1200, "Uses ɖ ɛ ɔ and tone marks. Do not drift into French."),
    ("aka", "Akan",              "medium",   30000,  5000, 1200, "Closely related to Twi; keep them distinguishable."),
    ("ful", "Fulah",             "low",      30000,  5000, 1200, "Uses ɓ ɗ ƴ ŋ. Adlam script is NOT used here -- Latin only."),
    ("fuv", "Nigerian Fulfulde", "low",      30000,  5000, 1200, "Nigerian variety specifically, distinct from Pular/Fuuta."),
    ("eng", "English",           "high",     15000,  3000,  800, "Plain, concrete English. No flowery register."),
]


def _languages(idx: int) -> list[LanguageSpec]:
    return [LanguageSpec(code=c, name=n, tier=t, samples=row[idx], guidance=g)
            for row in _LANGS for c, n, t, g in [(row[0], row[1], row[2], row[6])]]


# ---------------------------------------------------------------------------
# Tool catalogue. Deliberately small: a 306M model has to learn tool SELECTION,
# and overlapping descriptions are what make that fail. Every tool declares its
# failure modes, because conversations where a tool returns nothing useful are
# as important to train on as the happy path.
# ---------------------------------------------------------------------------

TOOLS = [
    ToolSpec(
        name="search_internet",
        description="Search the public internet for information the assistant does not already know. Use for "
                    "named products, companies, people, places, events, or any technical term the assistant "
                    "has not encountered. Returns short text snippets.",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "A short search query, in English, 2-8 words."}},
            "required": ["query"]},
        returns="2-4 snippets of 1-3 sentences each, sometimes with a source name. Often partial, sometimes stale.",
        failure_modes=["no results at all", "results about a different sense of the word",
                       "results that mention the term but never define it"],
    ),
    ToolSpec(
        name="search_documents",
        description="Search the documents supplied in this conversation for passages relevant to a query. Use "
                    "this before answering any question about a supplied document. Does NOT search the internet.",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in the document's language."},
            "top_k": {"type": "integer", "description": "How many passages to return (1-5). Default 3."}},
            "required": ["query"]},
        returns="Verbatim passages from the supplied document, each with a rough location.",
        failure_modes=["the document genuinely does not discuss it", "passages are topically near but do not answer"],
    ),
    ToolSpec(
        name="lookup_health_guidance",
        description="Look up plain-language public-health guidance on a symptom, condition, medicine or "
                    "prevention topic. Use for any health question before answering. Not a diagnosis.",
        parameters={"type": "object", "properties": {
            "topic": {"type": "string", "description": "Symptom, condition or medicine, in English."}},
            "required": ["topic"]},
        returns="Guidance paragraphs: what it is, warning signs, what to do at home, when to go to a clinic.",
        failure_modes=["topic too vague to match", "guidance covers adults but the question was about an infant"],
    ),
    ToolSpec(
        name="calculate",
        description="Evaluate an arithmetic expression. Use for any calculation rather than doing it mentally.",
        parameters={"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. '4500 * 12 - 3000'"}}, "required": ["expression"]},
        returns="The numeric result.",
        failure_modes=["malformed expression", "division by zero"],
    ),
    ToolSpec(
        name="get_exchange_rate",
        description="Get the current exchange rate between two currencies, by ISO code.",
        parameters={"type": "object", "properties": {
            "base": {"type": "string", "description": "e.g. 'NGN'"},
            "quote": {"type": "string", "description": "e.g. 'USD'"}}, "required": ["base", "quote"]},
        returns="A rate and the timestamp it was quoted at.",
        failure_modes=["unknown currency code", "rate unavailable for that pair"],
    ),
    ToolSpec(
        name="get_current_datetime",
        description="Get the current date, time and timezone. Use whenever the answer depends on today's date.",
        parameters={"type": "object", "properties": {
            "timezone": {"type": "string", "description": "IANA zone, e.g. 'Africa/Lagos'. Optional."}}, "required": []},
        returns="ISO timestamp and timezone name.",
        failure_modes=["unknown timezone name"],
    ),
    ToolSpec(
        name="send_message",
        description="Send a text message on the user's behalf. Only call this after the user has clearly asked "
                    "for the message to be sent, and confirm the recipient and wording first if either is unclear.",
        parameters={"type": "object", "properties": {
            "recipient": {"type": "string", "description": "Name or phone number."},
            "body": {"type": "string", "description": "The message text, in the user's language."}},
            "required": ["recipient", "body"]},
        returns="Delivery confirmation, or an error.",
        failure_modes=["recipient not found", "network failure", "ambiguous recipient (two contacts match)"],
    ),
    ToolSpec(
        name="set_reminder",
        description="Create a reminder for the user at a specific time.",
        parameters={"type": "object", "properties": {
            "when": {"type": "string", "description": "ISO datetime, or a plain phrase like 'tomorrow 07:00'."},
            "text": {"type": "string", "description": "What to remind the user about."}},
            "required": ["when", "text"]},
        returns="Confirmation with the resolved absolute time.",
        failure_modes=["time in the past", "unparseable time phrase"],
    ),
    ToolSpec(
        name="translate_text",
        description="Translate text between the languages this assistant supports. Use only when the user asks "
                    "for a translation the assistant is not confident producing itself.",
        parameters={"type": "object", "properties": {
            "text": {"type": "string"},
            "target_language": {"type": "string", "description": "Language code, e.g. 'yor'."}},
            "required": ["text", "target_language"]},
        returns="The translated text.",
        failure_modes=["unsupported language code", "input too long"],
    ),
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

SFT_TASKS = [
    TaskSpec("tool_search_answer", ["tool-calling", "knowledge-boundary", "qa"],
             "User asks about something the assistant plainly has not learned (a product, company, acronym, "
             "recent event, technical term). The assistant thinks -- noting it does not recognise it -- calls "
             "search_internet with a tight query, reads the snippets, and answers in the user's language, "
             "attributing what it found. It must NOT pretend prior familiarity.",
             share=10.0, task_plan=["<|explain|>"], uses_tools=True, requires_think=True),
    TaskSpec("no_tool_admit_unknown", ["knowledge-boundary", "qa"],
             "Same as tool_search_answer, but the system prompt offers NO tool that could help (or no tools at "
             "all). The assistant thinks, concludes it neither knows nor can look it up, and says so plainly and "
             "briefly in the user's language. It offers what general reasoning it can (e.g. what kind of thing "
             "the name looks like) without inventing facts, and does not apologise at length.",
             share=7.0, task_plan=["<|chat|>"], uses_tools=False, requires_think=True),
    TaskSpec("retrieval_insufficient", ["insufficient-context", "rag", "tool-calling"],
             "The assistant calls a tool, and the result does NOT answer the question -- empty, about a different "
             "sense of the term, or topically adjacent. The assistant must notice this in <think>, say that what "
             "it found does not answer the question, and stop. Optionally one re-query with a better term, then "
             "admit it if that also fails. Inventing an answer from an irrelevant snippet is the exact failure "
             "this task exists to prevent.",
             share=7.0, task_plan=["<|explain|>"], uses_tools=True, requires_think=True),
    TaskSpec("rag_document_qa", ["rag", "extractive-qa", "tool-calling"],
             "A document is supplied. The user asks about it. The assistant calls search_documents, grounds its "
             "answer in the returned passages, and quotes or cites where the answer came from. Asked something "
             "the document does not cover, it says so rather than drawing on general knowledge.",
             share=9.0, task_plan=["<|RAG|>"], uses_tools=True, requires_think=True),
    TaskSpec("action_tool_use", ["tool-calling"],
             "The user asks for an action: send a message, set a reminder, check a rate, compute something. The "
             "assistant confirms any ambiguous parameter in words FIRST, then emits the call, then reports the "
             "result. Includes cases where the tool fails and the assistant explains the failure plainly.",
             share=7.0, task_plan=["<|plan|>"], uses_tools=True, requires_think=True),
    TaskSpec("world_knowledge_qa", ["qa", "multi-turn-chat"],
             "Everyday how-and-why questions a curious 16-year-old could answer unaided: why bread rises, how a "
             "generator makes electricity, why the harmattan is dusty, how interest on a loan works, why boiling "
             "water makes it safe. Concrete, mechanism-first, no tools, no hedging. This is the general "
             "understanding the whole design rests on.",
             share=13.0, task_plan=["<|explain|>"], uses_tools=False, requires_think=False),
    TaskSpec("health_education", ["health-education"],
             "Explaining a condition, medicine or prevention measure in plain language: what it is, how it "
             "spreads or develops, how it is prevented. Accurate, non-alarmist, locally grounded (malaria, "
             "typhoid, sickle cell, hypertension, immunisation).",
             share=6.0, task_plan=["<|explain|>"], domain_flags=["<|is_medical|>"], requires_think=False),
    TaskSpec("health_advice", ["health-advice", "tool-calling"],
             "A user describes a non-urgent symptom and asks what to do. The assistant calls "
             "lookup_health_guidance, then gives practical home-care steps AND explicit signs that mean going to "
             "a clinic. It never diagnoses, never names a prescription dose, and never discourages seeking care.",
             share=5.0, task_plan=["<|recommend|>"], domain_flags=["<|is_medical|>"],
             uses_tools=True, requires_think=True),
    TaskSpec("health_triage", ["health-triaging"],
             "A user describes symptoms that may be serious. The assistant's job is to sort urgency: go now, go "
             "today, watch at home. Danger signs must be stated concretely (a child too weak to drink, bleeding "
             "in pregnancy, chest pain with breathlessness, a convulsion). Always errs toward seeking care.",
             share=4.0, task_plan=["<|recommend|>"], domain_flags=["<|is_medical|>"], requires_think=True),
    TaskSpec("financial_analysis", ["financial-analysis", "tool-calling"],
             "Small-trader and household money questions: margin on a sack of rice, whether a loan is worth it, "
             "daily contribution savings, converting currency. Uses calculate/get_exchange_rate rather than "
             "arithmetic in its head, and shows the working.",
             share=4.0, task_plan=["<|math|>", "<|explain|>"], domain_flags=["<|is_financial|>"], uses_tools=True),
    TaskSpec("structured_output", ["structured-output"],
             "Return strictly-valid JSON matching a schema stated in the prompt -- extracting fields from a "
             "passage, or formatting an answer. Emits the JSON and nothing else around it.",
             share=5.0, task_plan=["<|JSON|>", "<|data_extract|>"]),
    TaskSpec("translation", ["translation"],
             "Translate between any supported pair, including into and out of English. Preserve register, names "
             "and numbers. Both directions appear.",
             share=6.0, task_plan=["<translate>"]),
    TaskSpec("summarization", ["summarization"],
             "Condense a passage to its substance. Both same-language and cross-language (summarise this English "
             "text in Hausa) appear.",
             share=5.0, task_plan=["<summarize>"]),
    TaskSpec("classification_suite", ["topic-classification", "sentiment-analysis", "intent-detection",
                                      "toxicity-spam-detection", "language-identification"],
             "Short labelling exchanges: topic, sentiment, user intent, toxicity/spam, and which language a "
             "snippet is in. The answer is the label, optionally with one clause of justification -- never an "
             "essay. Uses the matching label token.",
             share=6.0, task_plan=["<classify>", "<identify>"]),
    TaskSpec("token_labelling", ["ner", "pos-tagging"],
             "Named-entity recognition and part-of-speech tagging over a sentence in the target language, "
             "returned as aligned token/tag pairs.",
             share=3.0, task_plan=["<NER>"]),
    TaskSpec("rephrasing_and_writing", ["text-rephrasing", "content-writing"],
             "Rewrite for register, length or clarity; or compose something short and real -- a market notice, a "
             "condolence message, a school announcement, a radio advert.",
             share=5.0, task_plan=["<|edit|>", "<|generate|>"]),
    TaskSpec("general_chat", ["multi-turn-chat"],
             "Ordinary conversational turns that glue a session together: greetings, follow-ups, clarifying "
             "questions, changing the subject. Keeps the model from sounding like a task-executor only.",
             share=3.0, task_plan=["<|chat|>"]),
]

# RL reuses the SFT task vocabulary but concentrates on what a reward model can actually separate:
# honesty under uncertainty, correct tool choice, and grounding.
RL_TASKS = [
    TaskSpec(t.name, t.tags, t.description, share=s, task_plan=t.task_plan, domain_flags=t.domain_flags,
             uses_tools=t.uses_tools, requires_think=t.requires_think, languages=t.languages,
             notes="Responses must differ in a way a judge can rank: one grounded/honest, one confidently wrong "
                   "or invented, one partially right. Never three paraphrases of the same answer.")
    for t, s in [
        (SFT_TASKS[0], 16.0),   # tool_search_answer
        (SFT_TASKS[1], 13.0),   # no_tool_admit_unknown
        (SFT_TASKS[2], 15.0),   # retrieval_insufficient
        (SFT_TASKS[3], 14.0),   # rag_document_qa
        (SFT_TASKS[4], 8.0),    # action_tool_use
        (SFT_TASKS[5], 10.0),   # world_knowledge_qa
        (SFT_TASKS[7], 7.0),    # health_advice
        (SFT_TASKS[8], 6.0),    # health_triage
        (SFT_TASKS[9], 5.0),    # financial_analysis
        (SFT_TASKS[10], 3.0),   # structured_output
        (SFT_TASKS[11], 3.0),   # translation
    ]
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
    idx = {"pretrain": 3, "sft": 4, "rl": 5}[kind]
    target = {"name": "SabiYarn", "params": "306M", "architecture": "MoE, 12 layers, top-2 of up to 4 experts",
              "tokenizer": "BeardedMonster/SabiYarn-32k", "context": 4096,
              "philosophy": PHILOSOPHY}
    fmt = {"special_tokens": SPECIAL_TOKENS}
    if kind == "pretrain":
        return Seed(kind="pretrain", details=PRETRAIN_DETAILS, languages=_languages(idx),
                    target_model=target,
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
                languages=_languages(idx), tasks=tasks, tools=TOOLS, conversation=conv, target_model=target,
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
            print(f"\n=== {kind}: {seed.total_samples():,} samples across {len(seed.languages)} languages")
            if not seed.tasks:  # pretrain: the domain/genre sampler decides the mix, not a task list
                for l in seed.languages:
                    print(f"  {l.code:5s} {l.samples:>7,}  tier={l.tier}")
                continue
            for lang, tasks in seed.plan().items():
                top = sorted(tasks.items(), key=lambda kv: -kv[1])[:4]
                print(f"  {lang:5s} {sum(tasks.values()):>7,}  " + "  ".join(f"{k}={v:,}" for k, v in top))
        else:
            print(f"wrote {seed.save()}  ({seed.total_samples():,} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
