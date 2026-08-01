"""Mathematical and statistical operations task generator.

Word problems grounded in everyday West African contexts (market prices,
farm yields, school scores, transport fares, savings/contributions) rather
than abstract "compute 34 * 12" prompts. About half the examples use the
`calculate` tool (for arithmetic non-trivial enough that mental math is
error-prone); the rest are simple enough that the assistant answers
directly with brief shown work -- teaching the model to be selective about
when a tool is actually warranted rather than always/never reaching for it.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from config.languages import LANGUAGES
from config.settings import EXAMPLES_PER_LANGUAGE_PER_TASK
from generators.base import (
    GENERAL_DATA_QUALITY_RULES,
    BatchRequestSpec,
    language_style_block,
    make_custom_id,
    rng_for,
)
from generators.personas import sample_persona
from schemas.strict import to_strict_json_schema
from schemas.tools import CALCULATE_TOOL, format_tools_for_system_prompt

TASK_NAME = "math_stats"

FINAL_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful multilingual assistant for West African language speakers, \
able to help with everyday arithmetic and simple statistics (totals, \
averages, percentages, ratios, simple probability). For arithmetic \
involved enough that mental math would be error-prone, use the calculate \
tool rather than computing it yourself; for simple arithmetic, just answer \
directly with brief working shown.

{tools_block}"""

TOPIC_BANK = [
    "splitting a market/shop total among a group of contributors (ajo/esusu-style savings contribution)",
    "calculating the average of a student's test scores across several subjects",
    "working out the percentage discount or markup on a market price",
    "converting a farm harvest (bags/baskets) into total revenue at a given price per unit",
    "working out fuel or transport cost for a trip given fare per distance",
    "calculating simple interest or a loan repayment amount over months",
    "figuring out the price per unit when buying in bulk versus retail",
    "working out how many days/weeks something will last given a daily usage rate",
    "calculating the mean, median, or mode of a small set of numbers (prices, scores, ages)",
    "working out a fair split of a bill or profit between business partners in different ratios",
]


class MathStatsSpec(BaseModel):
    problem_statement: str = Field(description="A word problem in the target language, grounded in an everyday West African context, with concrete numbers.")
    uses_calculator: bool = Field(description="True if the arithmetic is involved enough to warrant using the calculate tool rather than mental math.")
    calculator_expression: Optional[str] = Field(description="The arithmetic expression to evaluate (digits and + - * / ^ ( ) only), or null if uses_calculator is false.")
    calculator_result: Optional[str] = Field(description="The exact correct numeric result of calculator_expression, or null if uses_calculator is false.")
    reasoning: str = Field(description="Brief reasoning/working, in the target language. If uses_calculator is true, keep this to a short lead-in before the tool call (e.g. noting what needs to be computed); if false, show the brief mental-math working.")
    final_answer: str = Field(description="The final answer in the target language, clearly stated with appropriate units/currency, consistent with calculator_result when a calculator was used.")


def _build_user_prompt(language_code: str, persona: str, topic: str) -> str:
    return f"""\
{language_style_block(language_code)}

{persona}

Write a math/statistics word problem about: {topic}. Use concrete, \
realistic numbers (e.g. Naira, Cedis, or another locally-plausible \
currency/unit where relevant) rather than round toy numbers every time. \
Decide honestly whether the arithmetic is simple enough to do mentally or \
involved enough (e.g. multi-step, large numbers, decimals) to warrant the \
calculate tool, and set uses_calculator accordingly."""


def build_requests() -> list[BatchRequestSpec]:
    tools_block = format_tools_for_system_prompt([CALCULATE_TOOL])
    final_system_prompt = FINAL_SYSTEM_PROMPT_TEMPLATE.format(tools_block=tools_block)
    response_format = to_strict_json_schema(MathStatsSpec, name="math_stats")

    specs: list[BatchRequestSpec] = []
    for lang in LANGUAGES:
        for i in range(EXAMPLES_PER_LANGUAGE_PER_TASK):
            rng = rng_for(TASK_NAME, lang.code, i)
            persona = sample_persona(rng)
            topic = rng.choice(TOPIC_BANK)

            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(TASK_NAME, lang.code, i),
                    task=TASK_NAME,
                    language=lang.code,
                    system_prompt=GENERAL_DATA_QUALITY_RULES,
                    user_prompt=_build_user_prompt(lang.code, persona, topic),
                    response_format=response_format,
                    context={"final_system_prompt": final_system_prompt},
                )
            )
    return specs
