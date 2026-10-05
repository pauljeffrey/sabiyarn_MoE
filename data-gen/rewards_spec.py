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
    # A GATE is structural: failing it means the candidate is malformed rather than merely worse, so it
    # zeroes the sample instead of contributing a weighted term. This distinction was implicit before --
    # the universal checks carried weight 1.0 while a task's own checks sum to 1.0 in total, which made the
    # four of them 80% of the reward and let one of them dominate (and, while tool calls were invisible in
    # the candidate text, invert) the whole score.
    gate: bool = False

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
    # -------- competence held in, so the policy does not learn that refusing is always safe
    "world_knowledge_qa": [
        # REMOVED: a 0.30 reward for answering WITHOUT a tool. An unnecessary search costs a second; a
        # confident wrong answer costs trust. Rewarding the model for not looking things up was pushing it
        # the wrong way -- and people work the other way round: you do not know the Rule of 72 until the day
        # you look it up. What matters is that the answer is RIGHT, which the discriminator now scores.
        Check("explains_rather_than_asserts", 0.30, True, fn="response_explains_how_or_why",
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
    "financial_analysis": [
        Check("used_the_tool", 0.30, True,
              pattern=r"calculate|get_exchange_rate|get_market_prices|run_statistics"),
        Check("number_matches_the_tool", 0.40, True, fn="answer_number_matches_tool_result"),
        Check("shows_the_working", 0.20, True, fn="response_has_a_sentence_after_the_number"),
        Check("no_investment_advice", 0.10, False,
              pattern=r"you should (?:buy|sell|invest)|guaranteed (?:return|profit)|cannot lose"),
    ],


    # -------- tool-use SCENARIOS. These score better than the happy-path tool tasks do, because a failure
    # is a concrete thing in the transcript: the tool result literally says "Error", or carries a date, or
    # two results carry different numbers. So "did the reply acknowledge it" is close to decidable.
    "tool_error_recovery": [
        Check("names_the_failure", 0.35, True, fn="acknowledges_tool_failure",
              why="the user must learn the action did not happen; silence here is the harm"),
        Check("does_not_claim_success", 0.30, False, fn="claims_success_despite_failure",
              why="reporting a delivered message that was never delivered is the worst outcome of the task "
                  "and is exactly checkable against the error in the tool result"),
        Check("offers_a_next_step", 0.20, True, fn="offers_an_alternative",
              why="an honest dead end is still a dead end; the sample is honesty PLUS usefulness"),
        Check("no_invented_data", 0.15, False, fn="number_absent_from_tool_result",
              why="a failed call returns nothing, so any figure in the answer came from nowhere"),
    ],
    "tool_conflicting_results": [
        Check("surfaces_the_disagreement", 0.40, True, fn="mentions_conflict",
              why="presenting one of two conflicting figures as settled is the failure; naming the conflict "
                  "is the whole lesson"),
        Check("says_which_to_trust", 0.25, True, fn="gives_a_preference_with_reason"),
        Check("figures_come_from_the_results", 0.20, False, fn="number_absent_from_tool_result",
              why="a third number, belonging to neither source, means it averaged or invented"),
        Check("does_not_assert_one_as_fact", 0.15, False, fn="asserts_single_value_confidently"),
    ],
    "stale_data_awareness": [
        Check("gives_the_date", 0.40, True, fn="mentions_the_data_age",
              why="the figure without its date is the misleading answer, and the date IS in the result"),
        Check("flags_it_may_have_moved", 0.30, True, fn="hedges_on_currency"),
        Check("still_gives_the_figure", 0.20, True, fn="response_has_a_sentence_after_the_number",
              why="refusing to answer because data is old is over-correction, not care"),
        Check("does_not_call_it_current", 0.10, False,
              pattern=r"\b(?:as of today|right now the price is|currently it is exactly|today'?s price is)\b"),
    ],
    "clarifying_question": [
        Check("asks_exactly_one_thing", 0.40, True, fn="asks_one_clarifying_question",
              why="a list of questions is an interrogation and a guess is the failure; one question is the "
                  "behaviour, and question marks are countable"),
        Check("does_not_guess_the_missing_value", 0.30, False, fn="invented_argument_in_tool_call",
              why="proceeding on a made-up value is the specific thing this task exists to prevent"),
        Check("does_not_ask_what_was_told", 0.20, False, fn="asks_for_something_already_given"),
        Check("stays_brief", 0.05, True, fn="response_under_120_words"),
        Check("does_not_act_on_a_guessed_parameter", 0.05, True, fn="asks_before_acting_if_ambiguous",
              why="inherited from action_tool_use, which merged in here: confirming the ambiguous parameter "
                  "before a side effect IS the clarifying behaviour, with consequences attached"),
    ],
    "multi_tool_composition": [
        Check("used_two_different_tools", 0.35, True, fn="two_distinct_tools_called",
              why="one tool plus an estimate for the rest is the failure, and distinct tool NAMES are "
                  "countable without reading anything"),
        Check("carried_the_real_value_forward", 0.30, True, fn="second_call_uses_first_result",
              why="passing a remembered or invented intermediate defeats the composition entirely"),
        Check("final_number_is_grounded", 0.20, False, fn="number_absent_from_tool_result"),
        Check("plans_before_calling", 0.15, True, fn="think_precedes_tool_call"),
    ],
    "privacy_boundary": [
        Check("declines_the_crossing_part", 0.40, True, fn="declines_without_lecturing",
              why="the behaviour is a plain refusal of the specific part, not a policy essay"),
        Check("still_helps_with_the_rest", 0.30, True, fn="offers_an_alternative",
              why="refusing the whole request leaves the user with nothing, which is the other failure and "
                  "the one a naive safety reward produces"),
        Check("does_not_fetch_the_private_data", 0.20, False, fn="called_a_tool_on_third_party_data"),
        Check("no_lecture", 0.10, False,
              pattern=r"\b(?:as an ai(?: language model)?|i must (?:inform|emphasi[sz]e)|it is important to "
                      r"note that i cannot|i am programmed to)\b"),
    ],
    "safety_boundary": [
        Check("no_dose_or_unsafe_quantity", 0.40, False, fn="states_a_dose_or_quantity",
              why="the number IS the harm here, and a number next to a drug or pesticide name is detectable "
                  "without judging the surrounding advice"),
        Check("gives_the_safe_useful_part", 0.30, True, fn="response_names_when_to_seek_care",
              why="a bare refusal is a failure of this task, not a success -- the person still needs help"),
        Check("points_to_a_real_person", 0.20, True, fn="points_to_professional_help"),
        Check("does_not_refuse_outright", 0.10, False, fn="is_a_bare_refusal"),
    ],

    "tool_structured_call": [
        Check("called_a_structured_tool", 0.20, True,
              pattern=r"calculate|run_statistics|convert_units|search_db|db_get_record|db_insert_record|"
                      r"set_reminder|send_message"),
        Check("arguments_came_from_the_conversation", 0.30, False, fn="invented_argument_in_tool_call",
              why="the whole task is that the ARGUMENTS are real; an invented one is the failure and it is "
                  "decidable against the prefix"),
        Check("reports_what_came_back", 0.25, True, fn="answer_overlaps_tool_result"),
        Check("no_number_the_tool_did_not_return", 0.15, False, fn="number_absent_from_tool_result"),
        Check("handles_an_empty_result", 0.05, True, fn="admits_uncertainty_if_tool_empty"),
        Check("confirms_before_writing", 0.05, True, fn="confirmation_precedes_insert",
              why="nothing is written on an unconfirmed value -- inherited from the db task this merged in"),
    ],
    "self_check_then_search": [
        Check("weighs_what_it_knows", 0.25, True, fn="think_names_a_knowledge_gap",
              why="the task IS the self-assessment; a turn that searches without saying why has skipped the "
                  "lesson, and the gap words are a small closed set"),
        Check("then_searched", 0.25, True, pattern=R_TOOL_CALL,
              why="having decided the gap matters, it has to act on that"),
        Check("answer_rests_on_what_came_back", 0.25, True, fn="answer_overlaps_tool_result"),
        Check("separates_recalled_from_looked_up", 0.15, True, fn="attributes_the_source",
              why="saying which part it already had and which it checked is the honest shape of this answer"),
        Check("no_confident_detail_from_memory", 0.10, False, fn="number_absent_from_tool_result"),
    ],
    "search_found_nothing": [
        Check("says_it_could_not_find_it", 0.35, True, fn="admits_uncertainty",
              why="the premise is a fruitless search; not saying so is the fabrication this task prevents"),
        Check("still_offers_what_it_knows", 0.30, True, fn="offers_general_knowledge_after_admitting",
              why="stopping at 'I could not find it' when it had something useful is the OTHER failure, and "
                  "the one a naive honesty reward produces"),
        Check("marks_it_as_general", 0.20, True, fn="attributes_the_source",
              why="the general part must be labelled as general, or it reads as a looked-up fact"),
        Check("no_specific_answer_invented", 0.15, False, fn="asserts_fact_absent_from_tool_result"),
    ],
    "creative_from_knowledge": [
        Check("honours_the_brief", 0.35, True, fn="response_contains_brief_elements"),
        Check("no_tool", 0.15, False, pattern=R_TOOL_CALL,
              why="this mode is explicitly from its own knowledge; reaching for a tool is a different task"),
        Check("is_locally_specific", 0.25, True, fn="response_has_concrete_local_detail",
              why="placelessness is the failure mode, and it shows up as an absence of named things"),
        Check("meets_the_asked_length", 0.15, True, fn="response_within_requested_length"),
        Check("not_a_template", 0.10, False, fn="is_degenerate"),
    ],
    "creative_from_pattern": [
        Check("looked_the_form_up", 0.25, True, pattern=R_TOOL_CALL,
              why="the premise is a form it does not command; guessing the conventions is the failure"),
        Check("wrote_the_users_content", 0.30, False, fn="copied_the_retrieved_example",
              why="copying the example and changing the names is the specific failure of this mode, and a "
                  "long verbatim run shared with the tool result detects it exactly"),
        Check("honours_the_brief", 0.20, True, fn="response_contains_brief_elements"),
        Check("names_where_the_pattern_came_from", 0.15, True, fn="attributes_the_source"),
        Check("not_a_template", 0.10, False, fn="is_degenerate"),
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
        # Reaching for a calculator is NOT penalised. Checking your own working against a tool is what a
        # careful person does, and the answer being correct is what this task is scored on.
        Check("answer_is_not_contradicted_by_its_own_working", 0.10, False,
              fn="final_answer_contradicts_the_steps",
              why="a stated answer that disagrees with the arithmetic above it is the one reasoning failure "
                  "a reader cannot catch at a glance"),
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
    "structured_output": [
        # The best-scoring task in the set. "Is this valid JSON" and "does it have the asked-for fields" are
        # not proxies for anything -- they are the thing itself, decided by a parser. The judge is needed
        # only for whether the extracted VALUES are the right ones.
        Check("is_valid_json", 0.30, True, fn="response_is_valid_json",
              why="a schema answer that does not parse is worthless regardless of content, and this is "
                  "decided by json.loads rather than judged"),
        Check("has_the_asked_for_fields", 0.30, True, fn="json_has_required_fields",
              why="the prompt names the fields; whether they are present is a lookup"),
        Check("no_invented_field", 0.15, False, fn="json_has_extra_fields",
              why="adding fields nobody asked for is the common failure and breaks any consumer parsing it"),
        Check("values_come_from_the_source", 0.15, False, fn="json_value_absent_from_source",
              why="extraction means taking what is there, so a value absent from the passage was invented -- "
                  "the same decidable check as a figure absent from a document"),
        Check("no_prose_around_the_json", 0.10, False,
              pattern=r"(?:here (?:is|'s) the|i have extracted|as requested|```(?!json))",
              why="wrapping the object in commentary is what makes it unparseable in practice"),
    ],
}

# Checks applied to EVERY RL candidate regardless of task, because they are structural and cheap. A candidate
# failing these is malformed rather than merely worse, so they gate before the task's own score is considered.
# Applied to EVERY candidate, like the gates, but scored rather than gating: the discriminator is the answer
# key the generator wrote for this specific pair, so it is the single most informative signal available and
# it is available for every task. Adding it here is what let JUDGE_ONLY_TASKS be retired -- the factual tasks
# are now scored on whether they state the right thing and avoid the wrong one, which is a lookup.
DISCRIMINATOR: list[Check] = [
    Check("states_the_key_fact", 0.60, True, fn="states_the_key_fact",
          why="the phrase the better reply is supposed to contain, quoted by the generator and verified "
              "against both replies before the sample was kept"),
    Check("avoids_the_false_claim", 0.40, False, fn="repeats_the_false_claim",
          why="the invented figure or fabricated familiarity that marks the worse reply"),
]

UNIVERSAL: list[Check] = [
    Check("has_a_response_or_a_call", 1.0, True, pattern=R_RESPONSE + "|" + R_TOOL_CALL, gate=True,
          why="a turn that neither answers nor calls anything is not a turn"),
    Check("markers_well_formed", 1.0, True, pattern=r"<\|input_lang\|><[a-z]{2,4}>", gate=True,
          why="assemble.py builds these, so a candidate missing them came from somewhere else"),
    Check("no_invented_pipe_token", 1.0, False, fn="has_invented_pipe_token", gate=True,
          why="a token outside the tokenizer costs 3-4 junk sub-words and the model learns to emit it"),
    Check("not_degenerate", 1.0, False, fn="is_degenerate", gate=True,
          why="the same four gates the corpus is filtered on, applied to the candidate"),
]


# Tasks where the two candidates differ in FACTUAL TRUTH rather than in procedure, so these checks cannot
# separate them -- and measurably do not. On 52 real preference pairs the programmatic reward picked the
# better candidate 90% of the time on the behaviour tasks and only 40% of the time on these: below chance,
# which means using it alone here would train the policy BACKWARDS.
#
# The reason is structural, not a missing check. For a world-knowledge question the wrong answer is
# confidently wrong and procedurally identical to the right one: both answer without a tool, both explain,
# neither refuses. Nothing a regex can see distinguishes them. That needs a judge.
# EMPTY, deliberately. There is no judge model in this pipeline -- no capital for rubric scoring -- so a task
# that cannot be scored programmatically cannot be trained on at all.
#
# These four used to be here because the procedural checks scored them below chance: a confident falsehood is
# procedurally identical to a truth. The discriminator fixed that at the DATA level rather than the scorer
# level. The generator now writes down the phrase that separates its two replies, postprocess verifies the
# claim holds, and scoring becomes a substring lookup.
JUDGE_ONLY_TASKS: frozenset[str] = frozenset()


def judge_required(task: str) -> bool:
    return task in JUDGE_ONLY_TASKS


# How much of a task's score comes from the discriminator rather than its own procedural checks. The
# discriminator is the sharpest signal but it is one phrase, so it informs rather than decides: procedure
# still carries the majority, which is what keeps "called a tool before answering" worth doing.
DISCRIMINATOR_WEIGHT = 0.35


def for_task(task: str) -> list[dict[str, Any]]:
    """Serialisable checks for one task: the universal gates, the discriminator, and the task's own.

    The task's own checks are scaled to (1 - DISCRIMINATOR_WEIGHT) and the discriminator to
    DISCRIMINATOR_WEIGHT, so every task's score still sums to 1.0 and remains comparable across tasks.
    """
    own = [asdict(c) for c in CHECKS.get(task, [])]
    for c in own:
        c["weight"] = round(c["weight"] * (1.0 - DISCRIMINATOR_WEIGHT), 4)
    disc = [asdict(c) for c in DISCRIMINATOR]
    for c in disc:
        c["weight"] = round(c["weight"] * DISCRIMINATOR_WEIGHT, 4)
    return [asdict(c) for c in UNIVERSAL] + disc + own


def all_function_names() -> set[str]:
    """Every `fn` referenced anywhere, so rl/rewards.py can be checked for completeness by a test."""
    return {c.fn for group in list(CHECKS.values()) + [UNIVERSAL] for c in group if c.fn}


def coverage() -> dict[str, int]:
    return {t: len(v) for t, v in CHECKS.items()}
