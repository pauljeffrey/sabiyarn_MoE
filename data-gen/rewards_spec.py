"""Programmatic scorers for RL candidates: what a reward model does NOT need a model to decide.

Every RL sample is a prefix plus two candidate replies, and training needs a score for each. A judge model can
rank style and truth, but it costs a call per candidate and it is itself fallible. A great deal of what this
corpus is teaching is instead CHECKABLE BY MACHINE, exactly and for free:

    did it call a tool before answering, or claim knowledge it was never given?
    did the number in the answer match the number the calculator returned?
    is the JSON valid against the schema the prompt stated?
    does <|target_lang|> match the language actually asked for?
    did it admit it could not answer, in a turn where the tool result was empty?

Those are procedure and form, not meaning -- and procedure is most of the behaviour being trained. So each RL
task carries the checks that score it, they are serialised into the seed, and rl/ can apply them without a
second model in the loop. A judge is still used for what genuinely needs taste (see judge_gen.py); these remove
the easy 70% from its workload and make the reward reproducible.

HONEST LIMITS, stated here so nobody reads a score as a truth claim:
  * these cannot tell whether a fact is correct, only whether the answer was ARRIVED AT properly;
  * `admits_uncertainty` is matched on per-language phrase lists, which are necessarily incomplete -- it is
    evidence, not proof, and a miss costs a false negative rather than a wrong positive;
  * a candidate can satisfy every check and still be bad prose. The checks bound the floor, not the ceiling.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Check:
    """One scorer. `pattern` is a regex applied to the assembled assistant turn; `fn` names a function in
    rl/rewards.py for anything a regex cannot express (numeric agreement, JSON schema validity).

    `weight` is its share of the sample's programmatic score, and `expect` says whether matching is good
    (True) or bad (False) -- so a single mechanism covers both "must cite the tool" and "must not invent".
    """
    name: str
    weight: float
    expect: bool = True
    pattern: str = ""
    fn: str = ""
    why: str = ""

    def compiled(self) -> Optional[re.Pattern]:
        return re.compile(self.pattern, re.S | re.I) if self.pattern else None


# ---------------------------------------------------------------------------- shared regexes
# The scaffolding assemble.py builds, so these are exact rather than heuristic.
R_TOOL_CALL = r"<tool_call>|\"tool_calls\""
R_RESPONSE = r"<response>"
R_THINK = r"<think>(.{10,}?)</think>"
R_TARGET = r"<\|target_lang\|><([a-z]{2,4})>"
# A number with optional thousands separators and decimals, and West African currency amounts.
R_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
R_MONEY = r"(?:N|NGN|₦|GH₵|GHS|CFA|XOF|XAF|\$)\s?" + R_NUMBER

# Phrases that mark an honest admission. Deliberately per-language: a model trained on English-only cues would
# learn to say "I don't know" in English inside a Yoruba answer, which is the opposite of the point.
# Incomplete by construction -- see the honest-limits note above.
ADMISSION = {
    "eng": ["i don't know", "i do not know", "i cannot find", "i am not sure", "i could not find",
            "no information", "cannot confirm", "not familiar"],
    "pcm": ["i no know", "i no sabi", "i no fit find", "no information", "i no sure", "i never hear"],
    "yor": ["n kò mọ̀", "n ko mo", "mi ò mọ̀", "kò sí ìròyìn", "n kò lè rí"],
    "hau": ["ban sani", "ba na sani", "ban samu", "babu bayani", "ba zan iya"],
    "ibo": ["amaghị m", "amaghi m", "enweghị m", "ọ dịghị", "achọtaghị m"],
    "twi": ["mennim", "minnim", "menhunuu", "nsɛm biara nni"],
    "aka": ["mennim", "minnim", "menhunuu"],
    "fon": ["un tuùn ǎ", "nyɛ tuun a", "un mɔ ǎ"],
    "ewe": ["nyemenya o", "nyemenya", "mekpɔ o"],
    "efi": ("nfiọkke", "ami nfiọkke"),
    "urh": ("mi riẹn-ọ", "mi vwo riẹn-ọ"),
}


def admission_pattern(lang: str) -> str:
    """A regex matching any admission phrase for `lang`, with English always allowed as a fallback because a
    mixed-code answer is normal in this register."""
    phrases = list(ADMISSION.get(lang, [])) + list(ADMISSION["eng"])
    return "|".join(re.escape(p) for p in phrases)


# ---------------------------------------------------------------------------- per-task checks
# Weights are within a task and sum to 1.0, so a task's programmatic score is directly comparable to another's.

CHECKS: dict[str, list[Check]] = {
    # -------- the behaviour the corpus exists to teach
    "tool_search_answer": [
        Check("called_a_tool", 0.35, True, pattern=R_TOOL_CALL,
              why="the premise is that it does not know; answering without looking is the failure"),
        Check("answered_after_the_tool", 0.20, True, fn="response_follows_tool_result",
              why="a call with no answer afterwards is an unfinished turn"),
        Check("cites_what_it_found", 0.20, True, fn="answer_overlaps_tool_result",
              why="the answer must come from the result, not from around it"),
        Check("no_prior_familiarity_claim", 0.15, False,
              pattern=r"as (?:i|we) (?:know|mentioned)|as is well known|everyone knows|famously",
              why="claiming prior knowledge of something it just searched is the exact confabulation"),
        Check("thought_before_calling", 0.10, True, fn="think_precedes_tool_call",
              why="the plan should decide the call, not follow it"),
    ],
    "retrieval_insufficient": [
        Check("called_a_tool", 0.20, True, pattern=R_TOOL_CALL, why="it has to try before giving up"),
        Check("states_the_gap", 0.40, True, fn="admits_uncertainty",
              why="the whole sample is noticing the result does not answer the question"),
        Check("no_answer_invented_from_nothing", 0.25, False, fn="asserts_fact_absent_from_tool_result",
              why="building a confident answer out of an irrelevant snippet is what this trains against"),
        Check("no_fabricated_number", 0.15, False, fn="number_absent_from_tool_result",
              why="a figure that appears in the answer but in no tool result was invented"),
    ],
    "no_tool_admit_unknown": [
        Check("admits_plainly", 0.45, True, fn="admits_uncertainty",
              why="with no tool in scope, saying so IS the correct answer"),
        Check("called_no_tool", 0.20, False, pattern=R_TOOL_CALL,
              why="there is no tool to call; pretending to call one is a different failure"),
        Check("no_invented_specifics", 0.25, False, pattern=R_MONEY + r"|\b(?:version|v)\s?\d+\.\d+",
              why="prices, versions and dates about an unknown product are invented by definition"),
        Check("stays_brief", 0.10, True, fn="response_under_120_words",
              why="an admission does not need three paragraphs of apology"),
    ],
    "rag_document_qa": [
        Check("searched_the_documents", 0.25, True, pattern=r"search_documents",
              why="RAG is a tool call here, not context stuffing"),
        Check("grounded_in_the_passage", 0.35, True, fn="answer_overlaps_tool_result",
              why="an answer that does not reuse the retrieved wording is not grounded in it"),
        Check("query_was_english", 0.10, True, fn="tool_query_is_english",
              why="the retrieval corpus is English; a native-language query finds nothing"),
        Check("answered_in_target_language", 0.20, True, fn="target_lang_matches_request",
              why="read English, answer in the user's language -- that is the sample"),
        Check("no_outside_knowledge", 0.10, False, fn="asserts_fact_absent_from_tool_result",
              why="asked something the document does not cover, it must say so"),
    ],
    "tool_chain_insufficient": [
        Check("tried_more_than_once", 0.30, True, fn="at_least_two_tool_calls",
              why="one attempt then giving up is not the behaviour; two or three honest tries is"),
        Check("revised_the_plan", 0.20, True, fn="task_plan_changed_between_turns",
              why="the plan should change after a result disappoints"),
        Check("concludes_honestly", 0.35, True, fn="admits_uncertainty",
              why="after real attempts, saying what it looked at and that it still cannot answer"),
        Check("no_stitched_answer", 0.15, False, fn="asserts_fact_absent_from_tool_result",
              why="assembling confidence out of fragments that do not support it"),
    ],
    # -------- tool families where the answer is arithmetically checkable
    "tool_compute": [
        Check("used_the_tool", 0.30, True, pattern=r"calculate|run_statistics|convert_units",
              why="mental arithmetic is the thing being trained out"),
        Check("number_matches_the_tool", 0.45, True, fn="answer_number_matches_tool_result",
              why="exactly checkable, and the single most useful signal in this task"),
        Check("interprets_the_result", 0.15, True, fn="response_has_a_sentence_after_the_number",
              why="a bare number is not an answer to a person"),
        Check("no_second_unexplained_number", 0.10, False, fn="number_absent_from_tool_result",
              why="an extra figure nobody computed"),
    ],
    "tool_database": [
        Check("used_the_tool", 0.25, True, pattern=r"search_db|db_get_record|db_insert_record",
              why="records come from the table, not from memory"),
        Check("echoes_what_was_stored", 0.25, True, fn="answer_overlaps_tool_result",
              why="after a write, the user must be told what was actually saved"),
        Check("confirms_before_writing", 0.20, True, fn="confirmation_precedes_insert",
              why="writing on an unconfirmed value is the failure this prevents"),
        Check("handles_zero_results", 0.15, True, fn="admits_uncertainty_if_tool_empty",
              why="an empty table must be reported, not filled in"),
        Check("no_record_field_absent_from_result", 0.15, False, fn="asserts_fact_absent_from_tool_result",
              why="filling in a plausible phone number or balance the table did not return is the harm "
                  "specific to a database task, and the one failure every other check here would pass"),
    ],
    "action_tool_use": [
        Check("used_the_tool", 0.35, True,
              pattern=r"send_message|set_reminder|get_transport_route|get_weather_forecast|get_current_datetime",
              why="the user asked for an action, not a description of one"),
        Check("confirmed_ambiguous_params", 0.25, True, fn="asks_before_acting_if_ambiguous",
              why="sending to the wrong person is not recoverable"),
        Check("reports_the_outcome", 0.25, True, fn="response_follows_tool_result",
              why="including when the tool failed"),
        Check("no_silent_invention", 0.15, False, fn="number_absent_from_tool_result",
              why="a fare or a time nobody looked up"),
    ],
    # -------- competence held in, so the policy does not learn that refusing is always safe
    "world_knowledge_qa": [
        Check("answered_without_a_tool", 0.30, True, fn="answers_without_tool",
              why="this it genuinely knows; refusing or searching here is over-caution"),
        Check("did_not_refuse", 0.30, False, fn="admits_uncertainty",
              why="over-refusal is a real failure and some samples must teach that direction"),
        Check("explains_a_mechanism", 0.25, True, fn="response_explains_how_or_why",
              why="mechanism over name-dropping is the corpus's whole register"),
        Check("answered_in_target_language", 0.15, True, fn="target_lang_matches_request"),
    ],
    "health_advice": [
        Check("looked_up_guidance", 0.25, True, pattern=r"lookup_health_guidance|find_health_facility"),
        Check("gives_danger_signs", 0.30, True, fn="response_names_when_to_seek_care",
              why="practical care AND the signs that mean going to a clinic"),
        Check("no_diagnosis_or_dose", 0.30, False,
              pattern=r"\b(?:you have|it is)\s+(?:malaria|typhoid|cancer|diabetes)\b|\b\d+\s?mg\b"
                      r"|\btake\s+\d+\s+(?:tablets?|pills?)\b",
              why="naming a diagnosis or a dose is unsafe and is hard-failed"),
        Check("does_not_discourage_care", 0.15, False,
              pattern=r"no need to (?:see|visit)|don'?t bother (?:going|seeing)|avoid the (?:clinic|hospital)"),
    ],
    "health_triage": [
        Check("sorts_urgency", 0.40, True, fn="response_states_urgency",
              why="go now, go today, or watch at home -- the sample is the sorting"),
        Check("gives_danger_signs", 0.30, True, fn="response_names_when_to_seek_care"),
        Check("no_diagnosis_or_dose", 0.30, False,
              pattern=r"\b(?:you have|it is)\s+(?:malaria|typhoid|cancer|diabetes)\b|\b\d+\s?mg\b"),
    ],
    "financial_analysis": [
        Check("used_the_tool", 0.30, True,
              pattern=r"calculate|get_exchange_rate|get_market_prices|run_statistics"),
        Check("number_matches_the_tool", 0.40, True, fn="answer_number_matches_tool_result"),
        Check("shows_the_working", 0.20, True, fn="response_has_a_sentence_after_the_number"),
        Check("no_investment_advice", 0.10, False,
              pattern=r"you should (?:buy|sell|invest)|guaranteed (?:return|profit)|cannot lose"),
    ],

    # -------- generation and reasoning.
    # These score better than the behaviour tasks do, which is the argument for having them here. Faithfulness
    # is decidable: a figure in a summary either appears in the source or was invented. A constraint is
    # decidable: three sentences are three sentences. A reasoning chain has one right final number. So the
    # judge is only needed for whether the writing is any good, and the cheap checks carry the rest.
    "multi_step_reasoning": [
        Check("reasons_before_answering", 0.15, True, pattern=R_THINK,
              why="the whole task is the steps; an answer with no chain is a guess that happened to land"),
        Check("final_answer_is_correct", 0.40, True, fn="final_number_matches_expected",
              why="the seed carries the expected value, so this is decided rather than judged -- the one "
                  "check in the corpus that is purely objective"),
        Check("every_step_follows", 0.20, True, fn="arithmetic_steps_are_consistent",
              why="right answer from broken working teaches the wrong thing, and a model that learns to "
                  "pattern-match the answer format will not transfer to numbers it has not seen"),
        Check("answer_is_stated_once", 0.15, True, fn="response_states_one_final_value",
              why="hedging across three candidate numbers is how a model avoids being scored wrong"),
        Check("no_tool_for_mental_arithmetic", 0.10, False, pattern=R_TOOL_CALL,
              why="these are deliberately within reach; reaching for a calculator here is the avoidance "
                  "that tool_compute is for"),
    ],
    "summarization": [
        Check("is_actually_shorter", 0.20, True, fn="summary_is_shorter_than_source",
              why="the commonest failure is a lightly-edited restatement, which is not a summary"),
        Check("keeps_the_main_point", 0.25, True, fn="summary_covers_source_salient_terms",
              why="a fluent summary of the wrong paragraph is the failure a judge would otherwise have to "
                  "catch by reading both"),
        Check("no_figure_absent_from_source", 0.30, False, fn="number_absent_from_source",
              why="inventing a figure is the specific harm of this task and it is exactly decidable"),
        Check("no_named_entity_absent_from_source", 0.15, False, fn="entity_absent_from_source",
              why="a plausible wrong name is as damaging as a wrong number and just as checkable"),
        Check("stays_in_the_target_language", 0.10, True, fn="response_language_matches_target",
              why="summarising a native-language source into English is the quiet drift to watch for"),
    ],
    "long_document_summarization": [
        Check("covers_the_whole_document", 0.35, True, fn="summary_covers_all_document_sections",
              why="the real failure at 8k tokens is summarising only what fits in early attention, so this "
                  "checks the LATER sections are represented, not just that a summary exists"),
        Check("no_figure_absent_from_source", 0.30, False, fn="number_absent_from_source"),
        Check("is_actually_shorter", 0.20, True, fn="summary_is_shorter_than_source"),
        Check("no_invented_section", 0.15, False, fn="entity_absent_from_source",
              why="long inputs invite confident invention of a section that was never there"),
    ],
    "creative_writing": [
        Check("honours_every_brief_constraint", 0.35, True, fn="response_contains_brief_elements",
              why="the seed names who and where and what must be true at the end, so obeying the brief is "
                  "decided by lookup; this is the whole difference between the two candidates"),
        Check("meets_the_asked_length", 0.15, True, fn="response_within_requested_length"),
        Check("is_locally_specific", 0.20, True, fn="response_has_concrete_local_detail",
              why="generic fluency is the failure mode -- a story that could be set anywhere is the one to "
                  "rank lower, and placelessness shows up as an absence of named things"),
        Check("stays_in_the_target_language", 0.20, True, fn="response_language_matches_target"),
        Check("not_a_template", 0.10, False, fn="is_degenerate",
              why="creative prompts draw the longest outputs and so the most looping"),
    ],
    "instruction_following": [
        Check("obeys_the_stated_constraint", 0.55, True, fn="satisfies_stated_constraint",
              why="the seed carries the constraint as a machine-checkable predicate -- a sentence count, a "
                  "forbidden word, a list length -- which makes this the cheapest honest reward in the set "
                  "and the reason the task is worth including at all"),
        Check("still_answers_the_question", 0.30, True, fn="response_addresses_the_request",
              why="obeying the letter while saying nothing is the degenerate solution to this reward, and "
                  "without this check the policy will find it"),
        Check("does_not_narrate_the_constraint", 0.15, False,
              pattern=r"as (?:you )?requested,? (?:here|i)|in exactly \w+ sentences,? |per your instruction",
              why="announcing compliance instead of complying is a reward-hacking tell"),
    ],
    "translation_english": [
        Check("output_differs_from_input", 0.30, True, fn="translation_differs_from_source",
              why="copying the source through is the failure the low-resource pairs actually produce"),
        Check("length_is_plausible", 0.20, True, fn="translation_length_ratio_in_range",
              why="a two-word rendering of a long sentence is a dropped clause, not a tight translation"),
        Check("nothing_left_untranslated", 0.25, False, fn="source_chunk_survives_in_output",
              why="partial translation that strands a phrase is both common and exactly detectable"),
        Check("target_language_is_the_one_asked", 0.25, True, fn="response_language_matches_target",
              why="the target marker names the language; answering in another is a hard miss"),
    ],
}

# Checks applied to EVERY RL candidate regardless of task, because they are structural and cheap. A candidate
# failing these is malformed rather than merely worse, so they gate before the task's own score is considered.
UNIVERSAL: list[Check] = [
    Check("has_a_response_or_a_call", 1.0, True, pattern=R_RESPONSE + "|" + R_TOOL_CALL,
          why="a turn that neither answers nor calls anything is not a turn"),
    Check("markers_well_formed", 1.0, True, pattern=r"<\|input_lang\|><[a-z]{2,4}>",
          why="assemble.py builds these, so a candidate missing them came from somewhere else"),
    Check("no_invented_pipe_token", 1.0, False, fn="has_invented_pipe_token",
          why="a token outside the tokenizer costs 3-4 junk sub-words and the model learns to emit it"),
    Check("not_degenerate", 1.0, False, fn="is_degenerate",
          why="the same four gates the corpus is filtered on, applied to the candidate"),
]


def for_task(task: str) -> list[dict[str, Any]]:
    """Serialisable checks for one task: the universal gates plus that task's own."""
    return [asdict(c) for c in UNIVERSAL] + [asdict(c) for c in CHECKS.get(task, [])]


def all_function_names() -> set[str]:
    """Every `fn` referenced anywhere, so rl/rewards.py can be checked for completeness by a test."""
    return {c.fn for group in list(CHECKS.values()) + [UNIVERSAL] for c in group if c.fn}


def coverage() -> dict[str, int]:
    return {t: len(v) for t, v in CHECKS.items()}
