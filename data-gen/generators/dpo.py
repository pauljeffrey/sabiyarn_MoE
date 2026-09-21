"""DPO preference-pair generator.

Same task/domain sampling as SFT, plus a `rejection_type` drawn by the
sampler (least-used-first, restricted to types that make sense for the task).
The model writes `chosen` (clearly better, fully correct) and `rejected` (a
PLAUSIBLE answer with exactly the requested flaw).

Honest caveat (also in the README): rejected answers produced by injecting a
named flaw are OFF-POLICY -- they do not look like what your SFT model
would actually generate. That is useful for a first preference-tuning pass
and for teaching specific behaviours (language fidelity, constraint
following), but for on-policy preferences later, sample several answers from
the SFT model and rank them with a judge (or, for translation, a metric such
as AfriCOMET).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, get_args

from config.corpus_config import CorpusConfig
from generators.base import BatchRequestSpec, language_style_block
from generators.corpus_common import corpus_custom_id, resolve_languages
from generators.sft_tasks import (
    TASKS,
    common_sft_attributes,
    sft_compat,
    shared_prompt_lines,
    task_block,
)
from sampling.sampler import AttributeSpec, CoverageSampler, merge_weights, pair_weights_from_groups
from sampling.taxonomy import DOMAINS, all_pairs
from schemas.corpus import DPO_FORMAT_NAME, DPOPair, RejectionType, strict_response_format

KIND = "dpo"


@dataclass(frozen=True)
class RejectionSpec:
    key: str
    weight: float
    flaw: str  # what the rejected answer must do wrong
    tags_any: Optional[frozenset[str]] = None  # task tags this flaw applies to
    tags_none: frozenset[str] = frozenset()
    length_related: bool = False  # exempt from the chosen/rejected length-ratio filter

    def task_ok(self, task_tags: frozenset[str]) -> bool:
        if self.tags_any is not None and not (self.tags_any & task_tags):
            return False
        return not (self.tags_none & task_tags)


def _r(key: str, weight: float, flaw: str, any_: Optional[str] = None, none: str = "", length: bool = False) -> RejectionSpec:
    return RejectionSpec(key, weight, flaw, frozenset(any_.split()) if any_ else None, frozenset(none.split()), length)


_REJECTION_LIST: list[RejectionSpec] = [
    _r("factual_error", 1.0,
       "The rejected answer sounds confident and well-formed but contains one or two clear errors (a wrong fact, wrong number, wrong step of reasoning or wrong conclusion). Everything else is fine.",
       any_="qa comprehension explanation math reasoning culture summarization data_to_text extraction"),
    _r("hallucinated_details", 1.0,
       "The rejected answer adds specific-sounding details (names, numbers, dates, places, quotations) that the input does not support or that cannot be verified, while sounding authoritative.",
       any_="qa explanation summarization comprehension data_to_text advice culture letter"),
    _r("ignores_constraint", 1.0,
       "The instruction contains an explicit checkable constraint (word or sentence limit, number of items, required or forbidden content, required format). The rejected answer breaks it (e.g. too many words, wrong number of items) although the content is otherwise reasonable.",
       none="safety", length=True),
    _r("wrong_language", 1.0,
       "The rejected answer is written in English, or mixes a lot of English into the target language, or drifts into another language, although the instruction is in the target language.",
       none="translation"),
    _r("incomplete", 1.0,
       "The rejected answer stops early or covers only part of what was asked, or omits required items, without saying so.",
       length=True),
    _r("rambling_verbose", 0.9,
       "The rejected answer is padded, repetitive and roughly 2-3 times longer than needed, with tangents; the useful content is present but buried.",
       none="classification extraction format", length=True),
    _r("unhelpful_refusal", 0.7,
       "The rejected answer declines or deflects a perfectly reasonable request (e.g. 'I cannot help with that' or 'ask a professional') in the target language and gives nothing useful.",
       none="safety", length=True),
    _r("format_violation", 1.0,
       "The rejected answer gets the required output format wrong (invalid JSON, commentary around the JSON, prose instead of a list, wrong number of items, missing fields).",
       any_="format extraction classification list_output"),
    _r("tone_mismatch", 0.9,
       "The rejected answer uses the wrong tone or register for the situation (too casual for a formal message, rude, patronising, preachy, or stiff and bureaucratic for a friendly chat) while the content is acceptable.",
       any_="tone advice letter dialogue rewrite safety"),
    _r("poor_translation", 0.5,
       "The rejected answer is a flawed translation: mistranslated key words, added or omitted content, unnatural word-for-word structure (translationese) or the wrong register. It is still recognisably a translation of the source.",
       any_="translation"),
    _r("wrong_label", 0.2,
       "The rejected answer gives the wrong class label (plausible but incorrect) with a plausible-sounding reason.",
       any_="classification"),
    _r("off_topic", 0.8,
       "The rejected answer is fluent but addresses a different, related question or drifts to another topic, so it does not answer what was asked."),
]

REJECTIONS: dict[str, RejectionSpec] = {r.key: r for r in _REJECTION_LIST}
REJECTION_KEYS: list[str] = [r.key for r in _REJECTION_LIST]
assert set(REJECTION_KEYS) == set(get_args(RejectionType)), "schemas.corpus.RejectionType and dpo REJECTIONS diverged"

# Flaw types whose rejected answer legitimately differs in length or language
# from `chosen`; postprocess relaxes the matching checks for these.
LENGTH_RELATED = frozenset(r.key for r in _REJECTION_LIST if r.length_related)


def rejection_weights() -> dict[str, float]:
    return {r.key: r.weight for r in _REJECTION_LIST}


DPO_SYSTEM_PROMPT = """\
You create ONE preference pair (instruction, optional input, a better answer `chosen` and a worse answer `rejected`) for preference-tuning a small multilingual assistant serving speakers of West African languages.

Mandatory rules:
1. LANGUAGE: Write each field in the language the request specifies, in natural, native-sounding text with correct alphabet, special letters and diacritics/tone marks. No stray English except where the request says English.
2. INSTRUCTION: Realistic and self-contained; vary phrasing; avoid stock openings. Put the material the instruction operates on in `input` (original, written by you) or "" if none.
3. CHOSEN: clearly the better answer -- fully correct, helpful, complete, natural, and obeying every constraint exactly. Recompute any arithmetic. Never invent statistics, quotations or claims about real people or organisations.
4. REJECTED: a PLAUSIBLE but flawed answer showing exactly the requested flaw type -- the kind of mistake a weaker model really makes. It must not be gibberish, obviously absurd, or insulting. Same language as `chosen` (unless the flaw is about language) and comparable length (unless the flaw is about length). It must be worse than `chosen` ONLY because of the named flaw, not for extra unrelated reasons.
5. `chosen` and `rejected` must differ substantially and both must be self-contained answers to the same instruction. No preambles ("Sure!", "Here is..."), no meta-comments about the flaw, no code fences.
6. LOCAL GROUNDING: use the requested setting's names, places, currency and units.
7. HONESTY: if you cannot write a correct, natural `chosen` in this language for this request, still do your best but set chosen_confidence to "low"; use "medium" if unsure of a few words. Prefer short, grammatically safe sentences.
Set rejection_type to the requested flaw type exactly. Return only the JSON object required by the schema."""


def dpo_compat(chosen: dict, attr: str, value: str) -> bool:
    if attr == "rejection_type":
        return REJECTIONS[value].task_ok(TASKS[chosen["task"]].tags)
    if attr == "instruction_style":
        # A constraint-violation pair needs an instruction that states a constraint.
        if chosen.get("rejection_type") == "ignores_constraint":
            return value == "constraint_included"
        return True
    return sft_compat(chosen, attr, value)


def build_dpo_sampler(cfg: CorpusConfig, language: str) -> CoverageSampler:
    attributes = common_sft_attributes(cfg, language)
    rejection = AttributeSpec("rejection_type", merge_weights(rejection_weights(), cfg.attribute_weights.get("rejection_type"), name="rejection_type"))
    attributes.insert(1, rejection)  # right after `task`, before anything that depends on it
    pairs = all_pairs()
    group_of = {k: d.group for k, d in DOMAINS.items()}
    return CoverageSampler(
        kind=KIND, language=language, seed=cfg.seed, pairs=pairs,
        attributes=attributes,
        compat=dpo_compat,
        pair_weights=pair_weights_from_groups(pairs, group_of, cfg.domain_group_weights),
    )


def build_user_prompt(language: str, attrs: dict[str, str], index: int) -> str:
    task = TASKS[attrs["task"]]
    rej = REJECTIONS[attrs["rejection_type"]]
    return (
        f"{language_style_block(language)}\n\n"
        f"{task_block(task, language, dpo=True)}\n\n"
        f"{shared_prompt_lines(language, attrs, index, KIND)}\n\n"
        f"Requested flaw type for `rejected`: {rej.key}\n{rej.flaw}\n\n"
        "Write the instruction, input, chosen and rejected now. The topic is background grounding -- keep the task type as specified."
    )


def iter_requests(cfg: CorpusConfig, languages: Optional[list[str]] = None) -> Iterator[BatchRequestSpec]:
    response_format = strict_response_format(DPOPair, DPO_FORMAT_NAME)
    for language in resolve_languages(cfg, languages):
        sampler = build_dpo_sampler(cfg, language)
        for i in range(cfg.samples_per_language[language]):
            attrs = sampler.draw()
            task = TASKS[attrs["task"]]
            yield BatchRequestSpec(
                custom_id=corpus_custom_id(KIND, language, i),
                task=KIND,
                language=language,
                system_prompt=DPO_SYSTEM_PROMPT,
                user_prompt=build_user_prompt(language, attrs, i),
                response_format=response_format,
                context={
                    "kind": KIND,
                    "index": i,
                    "attributes": attrs,
                    "input_mode": task.input_mode,
                    "english_fields": list(task.english_fields),
                    "task_tags": sorted(task.tags),
                    "rejection_type": attrs["rejection_type"],
                    "length_related": attrs["rejection_type"] in LENGTH_RELATED,
                },
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )


def build_requests(cfg: CorpusConfig, languages: Optional[list[str]] = None) -> list[BatchRequestSpec]:
    return list(iter_requests(cfg, languages))
