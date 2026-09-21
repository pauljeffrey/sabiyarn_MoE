"""SFT (Alpaca-style) task catalogue + generator.

Record shape: `instruction`, `input` (may be empty), `response`. The
catalogue below lists the standard NLP/instruction task types the small
assistant should handle. Every request is grounded in a sampled
(domain, sub-topic, locale) so that tasks are spread over all fields instead
of collapsing onto a few familiar topics; the task type itself is drawn
least-used-first against the weights in `configs/sft.yaml`.

The DPO generator (`generators/dpo.py`) reuses this catalogue and the prompt
pieces here, so both kinds cover exactly the same task space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Optional

from config.corpus_config import CorpusConfig
from generators.base import BatchRequestSpec, language_style_block
from generators.corpus_common import (
    attribute_spec,
    corpus_custom_id,
    domain_block,
    language_names,
    locale_spec,
    locale_text,
    resolve_languages,
    sample_names,
)
from sampling.sampler import AttributeSpec, CoverageSampler, pair_weights_from_groups
from sampling.taxonomy import (
    DIFFICULTIES,
    DOMAINS,
    INSTRUCTION_STYLES,
    RESPONSE_LENGTHS,
    USER_REGISTERS,
    all_pairs,
    option_weights,
)
from schemas.corpus import SFT_FORMAT_NAME, SFTExample, strict_response_format

KIND = "sft"

InputMode = Literal["required", "empty", "optional"]

_ALL_LENGTHS = frozenset(RESPONSE_LENGTHS)


def _fs(s: Optional[str]) -> Optional[frozenset[str]]:
    return frozenset(s.split()) if s else None


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    weight: float
    input_mode: InputMode
    tags: frozenset[str]
    what: str  # what the example should contain
    input_note: str  # what goes in `input`
    response_note: str  # what the ideal response looks like
    english_fields: tuple[str, ...] = ()  # fields written in English ("input" and/or "response")
    domain_tags_any: Optional[frozenset[str]] = None  # restricts which domains fit this task
    lengths: frozenset[str] = _ALL_LENGTHS  # allowed response-length buckets

    def domain_ok(self, tags: frozenset[str]) -> bool:
        return self.domain_tags_any is None or bool(self.domain_tags_any & tags)


def _t(key: str, label: str, weight: float, mode: InputMode, tags: str, what: str, input_note: str, response_note: str,
       *, english: tuple[str, ...] = (), domains: Optional[str] = None, lengths: str = "brief short medium long") -> TaskSpec:
    return TaskSpec(key, label, weight, mode, frozenset(tags.split()), what, input_note, response_note,
                    english_fields=english, domain_tags_any=_fs(domains), lengths=frozenset(lengths.split()))


_TASK_LIST: list[TaskSpec] = [
    _t("open_qa", "Open-ended question answering", 1.3, "empty", "qa",
       "An open-ended question a real person in this setting might ask about the topic, and a helpful, culturally grounded answer.",
       'Leave empty ("").', "Answer directly with practical, locally relevant detail; hedge honestly where facts vary by place. No invented statistics.",
       lengths="short medium long"),
    _t("factual_qa", "Closed-book factual question", 1.0, "empty", "qa factual",
       "A factual question about GENERAL, well-established knowledge related to the topic, with a short verifiable answer. No obscure statistics, recent events or named living people.",
       'Leave empty ("").', "A short, correct answer, optionally with one explanatory sentence.",
       domains="academic historical culture civic technical commerce", lengths="brief short"),
    _t("reading_comprehension", "Reading-comprehension question", 1.0, "required", "comprehension qa",
       "A question about a passage. Write an original 80-150 word passage on the topic and ask a question that the passage answers.",
       "The passage (original, 80-150 words).", "An answer based ONLY on the passage; do not add outside facts.",
       lengths="brief short medium"),
    _t("summarization", "Summarisation", 1.0, "required", "summarization",
       "Ask for a summary of a text. Write an original 120-220 word text on the topic as the input.",
       "The text to summarise (original, 120-220 words).", "A faithful summary clearly shorter than the input, adding no new facts; obeys any length constraint in the instruction.",
       lengths="brief short medium"),
    _t("title_generation", "Title / headline generation", 0.8, "required", "generation",
       "Ask for a suitable title or headline for a short text.",
       "A short paragraph or news snippet (50-120 words).", "One good title or headline (or the exact number of options requested), in the right register.",
       lengths="brief"),
    _t("keyword_extraction", "Keyword extraction", 0.8, "required", "extraction list_output format",
       "Ask for the key words or key phrases of a text.",
       "A passage (60-140 words).", "A list of 4-8 keywords or key phrases actually present in (or directly drawn from) the passage, one per line or comma-separated as requested.",
       lengths="brief short"),
    _t("translation_en_to_lang", "Translation, English to the target language", 1.2, "required", "translation",
       "Ask (in the target language) for an English text to be translated into the target language. The English source text is about the topic.",
       "Plain, everyday English source text (1-4 sentences).", "An accurate, natural (not word-for-word) translation into the target language that keeps meaning, tone and names.",
       english=("input",), lengths="brief short medium"),
    _t("translation_lang_to_en", "Translation, target language to English", 1.2, "required", "translation",
       "Ask (in the target language) for a target-language text to be translated into English. The source text is about the topic.",
       "A natural target-language source text (1-4 sentences).", "An accurate, natural English translation that keeps meaning, tone and names.",
       english=("response",), lengths="brief short medium"),
    _t("paraphrase", "Paraphrasing", 0.9, "required", "rewrite",
       "Ask for a paraphrase of a text in different words while keeping the meaning.",
       "A text of 1-4 sentences.", "A paraphrase with clearly different wording and structure but identical meaning.",
       lengths="brief short medium"),
    _t("formality_rewrite", "Formal <-> informal rewriting", 0.9, "required", "rewrite tone",
       "Ask to rewrite a text in a more formal (or more informal/friendly) register for a stated situation.",
       "A message of 1-4 sentences in the opposite register.", "The rewritten text in the requested register with the same content.",
       lengths="brief short medium"),
    _t("grammar_correction", "Grammar / spelling correction", 0.8, "required", "rewrite",
       "Ask to correct a text. Write a short text containing 2-4 realistic spelling, diacritic, agreement or punctuation mistakes.",
       "The flawed text (1-4 sentences); the mistakes must be ones a native speaker would recognise as mistakes.", "The corrected text (optionally followed by one short line saying what was fixed).",
       lengths="brief short medium"),
    _t("text_classification", "Text classification (topic / sentiment / intent)", 1.0, "required", "classification format",
       "Ask to classify a short text into one of a stated set of labels (topic, sentiment or intent). State the label set in the instruction.",
       "A short text (1-3 sentences) whose label is unambiguous.", "Exactly one label from the stated set, optionally with a one-sentence reason.",
       lengths="brief"),
    _t("ner_extraction", "Named-entity extraction (JSON)", 0.8, "required", "extraction format",
       "Ask to extract people, places, organisations and dates from a text as JSON. Use only FICTIONAL people and generic organisations.",
       "A text of 2-4 sentences mentioning a few such entities.", 'Valid JSON only: {"persons": [], "locations": [], "organizations": [], "dates": []} with values copied verbatim from the input (empty list if none).',
       lengths="brief short medium"),
    _t("info_extraction_json", "Information extraction to JSON", 0.9, "required", "extraction format",
       "Ask to extract specific fields from a short notice, message or advert into JSON. The instruction names the keys.",
       "A short realistic notice/message/advert with the needed details (fictional).", "Valid JSON only, using exactly the requested keys (English keys), values taken from the input; null for missing values.",
       lengths="brief short medium"),
    _t("sentence_completion", "Sentence / passage completion", 0.7, "required", "generation",
       "Ask to continue or complete an unfinished sentence or short passage naturally.",
       "The beginning of a sentence or paragraph, cut off mid-way.", "A natural, coherent continuation in the same voice.",
       lengths="short medium"),
    _t("creative_story", "Creative writing: short story", 0.9, "optional", "creative",
       "Ask for a short original story. The input may hold optional seed details (characters, setting, moral); otherwise leave it empty.",
       'Optional story seed, or "".', "An original story with characters, a problem and a resolution, locally grounded.",
       lengths="medium long"),
    _t("creative_poem", "Creative writing: poem or song", 0.7, "optional", "creative",
       "Ask for an original poem or song verse. The input may hold optional constraints (theme, rhyme, number of lines).",
       'Optional constraints, or "".', "An original poem with imagery from local life; obeys line/rhyme constraints.",
       lengths="short medium"),
    _t("letter_email_drafting", "Letter / email / message drafting", 1.0, "optional", "letter tone",
       "Ask to draft a letter, email, SMS or WhatsApp message for a concrete situation. The input may list key details to include.",
       'Optional key details as a short list, or "".', "A ready-to-send message with greeting, body and closing appropriate to the relationship and register.",
       domains="commerce civic personal academic culture", lengths="short medium long"),
    _t("explanation_howto", "Explanation / how-to", 1.1, "empty", "explanation",
       "Ask for an explanation of how something works or a step-by-step guide to doing something, related to the topic.",
       'Leave empty ("").', "A clear, accurate explanation or ordered steps a layperson can follow.",
       lengths="short medium long"),
    _t("brainstorming_list", "Brainstorming / lists", 0.8, "empty", "brainstorm list_output format",
       "Ask for a list of ideas, options or examples related to the topic (with a stated number if you like).",
       'Leave empty ("").', "A tidy numbered or dashed list of distinct, practical, locally relevant items.",
       lengths="short medium"),
    _t("math_word_problem", "Math word problem", 1.0, "empty", "math",
       "An everyday arithmetic word problem using local prices, units and currency related to the topic. Keep the numbers modest.",
       'Leave empty ("").', "Short step-by-step working then a clear final answer. Every calculation MUST be exactly correct -- recompute before answering.",
       domains="numeric", lengths="short medium"),
    _t("statistics_simple", "Simple statistics", 0.7, "required", "math",
       "Ask for a simple statistic (mean, median, mode, range, percentage) of a small data set related to the topic.",
       "A small data set (5-10 numbers or a tiny table) in plain text.", "Short working and a clear final answer. Every calculation MUST be exactly correct -- recompute before answering.",
       domains="numeric", lengths="short medium"),
    _t("commonsense_reasoning", "Commonsense reasoning", 0.8, "empty", "reasoning",
       "A question that needs everyday commonsense to answer (cause and effect, what to do next, what is likely).",
       'Leave empty ("").', "The sensible answer with one or two sentences of reasoning.",
       lengths="brief short"),
    _t("logical_reasoning", "Logical reasoning puzzle", 0.6, "optional", "reasoning",
       "A small logic puzzle (3-4 clear premises) about people, places or objects from the setting. The premises may be in the instruction or in the input.",
       'Optional premises, or "".', "The deduction spelled out in a few clear steps, ending with the conclusion. The puzzle must be consistent and solvable.",
       lengths="short medium"),
    _t("dialogue_continuation", "Dialogue continuation", 0.8, "required", "dialogue tone",
       "Ask to continue a conversation with the next few natural replies.",
       "A short dialogue (4-6 lines, 'Name: line' format) between fictional local characters.", "The next 2-4 lines in the same format, consistent in tone and characters.",
       lengths="short medium"),
    _t("proverb_explanation", "Proverb / idiom explanation", 0.7, "required", "culture explanation",
       "Ask for the meaning and use of a proverb or idiom. Use ONLY a proverb or saying you are genuinely confident exists and is widely known in this language; if you are not, use a very common everyday idiom or lower your confidence.",
       "The proverb or idiom, in the target language.", "The meaning in plain words, a situation where it is used, and (optionally) the value it teaches.",
       domains="culture", lengths="short medium"),
    _t("data_to_text", "Data-to-text description", 0.7, "required", "data_to_text",
       "Ask to describe a small table or list in a short paragraph.",
       "A small table or list in plain text (3-6 rows, e.g. market prices, class scores, rainfall) with fictional values.", "A fluent paragraph that reports the data accurately, adds no facts and invents no numbers.",
       domains="numeric commerce", lengths="short medium"),
    _t("format_constrained", "Format-constrained generation", 1.0, "optional", "format constraint",
       "Ask for content related to the topic under strict, checkable format constraints (exactly N bullet points, a maximum word count, one sentence, JSON with given keys, ...).",
       'Optional material to use, or "".', "Output that obeys EVERY constraint exactly -- count the items and words before answering.",
       lengths="brief short medium"),
    _t("safe_decline", "Safe helpfulness: polite decline of a harmful request", 0.6, "empty", "safety tone",
       "A request touching the topic that a responsible assistant should NOT fulfil (e.g. help with a scam, harming someone, dangerous misinformation, cheating on an exam, medical advice that could endanger someone). The request itself must NOT contain operational harmful details.",
       'Leave empty ("").', "A brief, polite, non-judgemental decline in the target language that says why in one sentence and offers a safe, genuinely useful alternative or a suitable place to get help.",
       domains="risk", lengths="short medium"),
    _t("advice", "Practical / personal advice", 1.0, "empty", "advice tone",
       "A person describes a realistic personal or practical dilemma related to the topic and asks what to do.",
       'Leave empty ("").', "Warm, practical, culturally aware advice with concrete next steps; not preachy; recommends professional help only where truly needed.",
       lengths="short medium long"),
    _t("text_simplification", "Text simplification", 0.7, "required", "rewrite",
       "Ask to rewrite a complex passage in simpler words for children or beginners.",
       "A somewhat complex passage of 60-140 words.", "A simpler version with shorter sentences and easier words that keeps every key point.",
       lengths="short medium"),
    _t("question_generation", "Question generation", 0.6, "required", "comprehension list_output",
       "Ask for questions that a passage can answer (e.g. for a quiz).",
       "A passage of 80-150 words (original).", "3-5 clear questions, each answerable from the passage alone.",
       lengths="short medium"),
    _t("compare_contrast", "Compare and contrast", 0.7, "empty", "explanation",
       "Ask to compare two related things from the topic (two methods, places, options) and say when each is better.",
       'Leave empty ("").', "A balanced comparison with concrete similarities and differences.",
       lengths="medium long"),
]

TASKS: dict[str, TaskSpec] = {t.key: t for t in _TASK_LIST}
TASK_KEYS: list[str] = [t.key for t in _TASK_LIST]


def task_weights() -> dict[str, float]:
    return {t.key: t.weight for t in _TASK_LIST}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SFT_SYSTEM_PROMPT = """\
You create ONE high-quality supervised fine-tuning example (instruction, optional input, ideal response) in Alpaca style for a small multilingual assistant serving speakers of West African languages.

Mandatory rules:
1. LANGUAGE: Write each field in the language the request specifies. The text must sound native -- natural word order and idiom, not translated English. Use the correct alphabet, special letters and diacritics/tone marks of the language, consistently. No stray English.
2. INSTRUCTION: Realistic and self-contained: something a real person in the setting would type. Vary the phrasing; avoid stock openings such as "Translate the following text" or "Write a ...".
3. INPUT: The material the instruction operates on (a passage, a text to rewrite, data...). It must be original and written by you. If the task has no input, set it to "". Never put the instruction inside `input`, and never refer to "the text above/below" when `input` is empty.
4. RESPONSE: Correct, complete and directly on task; natural in the target language. Obey every constraint in the instruction exactly (counts, word limits, format). No preambles ("Sure!", "Here is..."), no meta-comments, no code fences, no emojis; plain text unless the instruction asks for a list or JSON.
5. FACTS AND MATH: Be accurate. Recompute any arithmetic. Never invent statistics, quotations, laws, or claims about real people or organisations; if unsure of a fact, choose a different question or answer in general, hedged terms.
6. LOCAL GROUNDING: Use the requested setting's names, places, currency, units and customs, not Western defaults.
7. HONESTY: If you cannot write correct, natural text in this language for this request, still do your best but set confidence to "low"; use "medium" if unsure of a few words. Prefer short, grammatically safe sentences to ambitious ones you are unsure of.
Return only the JSON object required by the schema."""


def language_directive(language: str, english_fields: tuple[str, ...], *, dpo: bool = False) -> str:
    lang = language_names(language)
    if dpo:
        eng_map = {"input": ("input",), "response": ("chosen", "rejected")}
        english = [f for e in english_fields for f in eng_map[e]]
        target = [f for f in ("instruction", "input", "chosen", "rejected") if f not in english]
    else:
        english = list(english_fields)
        target = [f for f in ("instruction", "input", "response") if f not in english]
    text = f"Write {', '.join(target)} in {lang}."
    if english:
        text += f" Write {', '.join(english)} in plain, natural English."
    return text


def task_block(task: TaskSpec, language: str, *, dpo: bool = False) -> str:
    resp = "`chosen` answer" if dpo else "`response`"
    return (
        f"Task type: {task.label}\n"
        f"{task.what}\n"
        f"Input field: {task.input_note}\n"
        f"Ideal {resp}: {task.response_note}\n"
        f"Languages: {language_directive(language, task.english_fields, dpo=dpo)}"
    )


def shared_prompt_lines(language: str, attrs: dict[str, str], index: int, kind: str) -> str:
    names = ", ".join(sample_names(language, index, kind))
    return (
        f"Setting: {locale_text(language, attrs['locale'])}\n"
        f"{domain_block(attrs['domain'], attrs['subtopic'])}\n"
        f"Register of the user's request: {USER_REGISTERS[attrs['register']].text}.\n"
        f"Form of the instruction: {INSTRUCTION_STYLES[attrs['instruction_style']].text}.\n"
        f"Task difficulty: {DIFFICULTIES[attrs['difficulty']].text}.\n"
        f"Target answer length: {RESPONSE_LENGTHS[attrs['response_length']].text}.\n"
        f"Local given names you may use for fictional people: {names}."
    )


def build_user_prompt(language: str, attrs: dict[str, str], index: int) -> str:
    task = TASKS[attrs["task"]]
    return (
        f"{language_style_block(language)}\n\n"
        f"{task_block(task, language)}\n\n"
        f"{shared_prompt_lines(language, attrs, index, KIND)}\n\n"
        "Write the instruction, input and response now. The topic is background grounding -- keep the task type as specified."
    )


# ---------------------------------------------------------------------------
# Sampler + request building
# ---------------------------------------------------------------------------


def sft_compat(chosen: dict, attr: str, value: str) -> bool:
    tags = DOMAINS[chosen["domain"]].tags
    if attr == "task":
        return TASKS[value].domain_ok(tags)
    if attr == "difficulty":
        return DIFFICULTIES[value].domain_ok(tags)
    task = TASKS[chosen["task"]]
    if attr == "response_length":
        return value in task.lengths
    return True


def common_sft_attributes(cfg: CorpusConfig, language: str) -> list[AttributeSpec]:
    """Attributes shared by SFT and DPO, in draw order (task first)."""
    tw = task_weights()
    return [
        attribute_spec("task", tw, cfg),
        attribute_spec("register", option_weights(USER_REGISTERS), cfg),
        attribute_spec("instruction_style", option_weights(INSTRUCTION_STYLES), cfg),
        attribute_spec("response_length", option_weights(RESPONSE_LENGTHS), cfg),
        attribute_spec("difficulty", option_weights(DIFFICULTIES), cfg),
        locale_spec(language, cfg),
    ]


def build_sft_sampler(cfg: CorpusConfig, language: str) -> CoverageSampler:
    pairs = all_pairs()
    group_of = {k: d.group for k, d in DOMAINS.items()}
    return CoverageSampler(
        kind=KIND, language=language, seed=cfg.seed, pairs=pairs,
        attributes=common_sft_attributes(cfg, language),
        compat=sft_compat,
        pair_weights=pair_weights_from_groups(pairs, group_of, cfg.domain_group_weights),
    )


def task_context(attrs: dict[str, str]) -> dict:
    task = TASKS[attrs["task"]]
    return {"input_mode": task.input_mode, "english_fields": list(task.english_fields), "task_tags": sorted(task.tags)}


def iter_requests(cfg: CorpusConfig, languages: Optional[list[str]] = None) -> Iterator[BatchRequestSpec]:
    response_format = strict_response_format(SFTExample, SFT_FORMAT_NAME)
    for language in resolve_languages(cfg, languages):
        sampler = build_sft_sampler(cfg, language)
        for i in range(cfg.samples_per_language[language]):
            attrs = sampler.draw()
            yield BatchRequestSpec(
                custom_id=corpus_custom_id(KIND, language, i),
                task=KIND,
                language=language,
                system_prompt=SFT_SYSTEM_PROMPT,
                user_prompt=build_user_prompt(language, attrs, i),
                response_format=response_format,
                context={"kind": KIND, "index": i, "attributes": attrs, **task_context(attrs)},
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )


def build_requests(cfg: CorpusConfig, languages: Optional[list[str]] = None) -> list[BatchRequestSpec]:
    return list(iter_requests(cfg, languages))
