"""Pydantic models = the strict Structured Outputs contracts for the corpus kinds.

Each model is converted to an OpenAI strict json_schema by
`schemas/strict.py`. Field descriptions double as per-field generation
instructions (the API shows them to the model), so they are written as
guidance rather than documentation. The same models validate the responses
again at postprocess time -- the API's strictness is a guarantee about shape,
not about content, so we never trust it for content.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.strict import to_strict_json_schema

Confidence = Literal["high", "medium", "low"]

# Source of truth for DPO flaw types (generators/dpo.py attaches prompts and
# compatibility rules to exactly these keys and asserts they match).
RejectionType = Literal[
    "factual_error",
    "hallucinated_details",
    "ignores_constraint",
    "wrong_language",
    "incomplete",
    "rambling_verbose",
    "unhelpful_refusal",
    "format_violation",
    "tone_mismatch",
    "poor_translation",
    "wrong_label",
    "off_topic",
]

Score = Literal[1, 2, 3, 4, 5]


class PretrainDoc(BaseModel):
    title: str = Field(description="A short natural title for the document, in the target language (no quotes, no English).")
    text: str = Field(description="The full document in the target language, matching the requested genre, register and length. Plain text only.")
    language_self_check: bool = Field(description="true only if you are confident the whole text is in the target language, natural-sounding, and free of invented facts; otherwise false.")


class SFTExample(BaseModel):
    instruction: str = Field(description="The user's instruction/request, written the way a real speaker would write it, in the language required by the task.")
    input: str = Field(description="The material the instruction operates on (passage, text to rewrite, data...). Empty string \"\" if the task needs no separate input.")
    response: str = Field(description="The ideal assistant response: correct, helpful, natural, complete, and in the language required by the task.")
    confidence: Confidence = Field(description="Your honest confidence that instruction, input and response are all correct and natural in the target language: high, medium or low.")


class DPOPair(BaseModel):
    instruction: str = Field(description="The user's instruction/request, in the language required by the task.")
    input: str = Field(description="The material the instruction operates on, or \"\" if none.")
    chosen: str = Field(description="The clearly better answer: fully correct, helpful, follows every constraint, natural in the target language.")
    rejected: str = Field(description="A plausible but flawed answer exhibiting exactly the requested flaw type. Same language, similar length unless the flaw is about length. Not gibberish.")
    rejection_type: RejectionType = Field(description="Echo the requested flaw type exactly.")
    chosen_confidence: Confidence = Field(description="Your honest confidence that `chosen` is correct and natural in the target language: high, medium or low.")


class JudgeScores(BaseModel):
    language_correctness: Score = Field(description="1-5: text is in the target language, with correct orthography/diacritics and grammar (5 = native-quality).")
    fluency: Score = Field(description="1-5: reads naturally to a native speaker, not translationese (5 = fully natural).")
    factuality: Score = Field(description="1-5: free of factual errors and invented details (5 = nothing doubtful).")
    instruction_following: Score = Field(description="1-5: does what was asked / matches the requested topic, genre and constraints (5 = fully).")
    usefulness: Score = Field(description="1-5: would be genuinely valuable as training data (5 = very).")
    issues: str = Field(description="One short English sentence naming the main problem, or \"none\".")


class DPOJudgeScores(JudgeScores):
    chosen_better_than_rejected: bool = Field(description="true if `chosen` is clearly better than `rejected` on correctness and helpfulness.")


def strict_response_format(model: type[BaseModel], name: str) -> dict[str, Any]:
    """`to_strict_json_schema` minus pydantic's per-field "title" noise.

    Every request carries the schema as input tokens; across 100k+ requests
    the redundant titles are pure cost.
    """
    fmt = copy.deepcopy(to_strict_json_schema(model, name=name))

    def strip(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for v in props.values():
                    if isinstance(v, dict):
                        v.pop("title", None)
                        strip(v)
            node.pop("title", None) if "properties" in node else None
            for k in ("items", "anyOf", "oneOf", "allOf"):
                if k in node:
                    strip(node[k])
        elif isinstance(node, list):
            for x in node:
                strip(x)

    strip(fmt["json_schema"]["schema"])
    return fmt


# Response-format names double as the routing key for scripts/mock_generate.py.
PRETRAIN_FORMAT_NAME = "pretrain_doc"
SFT_FORMAT_NAME = "sft_example"
DPO_FORMAT_NAME = "dpo_pair"
JUDGE_FORMAT_NAME = "judge_scores"
DPO_JUDGE_FORMAT_NAME = "dpo_judge_scores"
