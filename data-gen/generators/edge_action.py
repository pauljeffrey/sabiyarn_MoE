"""Edge-device action-triggering task generator.

Per-example flow: available tools (a random subset, including the target
tool plus a few distractors) -> user instruction -> short reasoning +
task plan -> a tool call -> a (plausible, model-invented) tool result ->
final assistant response confirming/using the result.

Rendering note: the tokenizer already has `<think>`/`</think>`, `<reason>`,
`<task_plan>`/`</task_plan>` registered as dedicated special tokens (see
training/constant_tokens.py in the parent repo), which is a strong signal
this is the intended way to encode a reasoning-then-plan step in assistant
content. We follow that convention here: the reasoning/plan is emitted as
ONE assistant text turn (`<reason>...</reason><task_plan>...</task_plan>`),
followed by a SEPARATE assistant turn that is a pure tool call (the chat
template does not allow content and tool_calls in the same message). If
you intend a different convention, this is the one place to change it --
see `pipeline/postprocess.py::render_edge_action`.

A fraction of examples are "hard": the available tools genuinely cannot
satisfy the request, so the correct behavior is to explain that / ask a
clarifying question rather than force a tool call.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from config.languages import LANGUAGES
from config.settings import EXAMPLES_PER_LANGUAGE_PER_TASK, HARD_EXAMPLE_FRACTION
from generators.base import (
    GENERAL_DATA_QUALITY_RULES,
    BatchRequestSpec,
    language_style_block,
    make_custom_id,
    rng_for,
)
from generators.personas import sample_persona
from schemas.strict import to_strict_json_schema
from schemas.tools import EDGE_DEVICE_TOOLS, format_tools_for_system_prompt

TASK_NAME = "edge_action"

FINAL_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful multilingual assistant running on a mobile device for \
West African language speakers. You can trigger on-device actions using \
the tools below. Think briefly about what the user wants and how to do it \
before acting. If none of the available tools can satisfy the request, say \
so or ask a clarifying question instead of guessing or calling the wrong \
tool.

{tools_block}"""


class EdgeActionSpec(BaseModel):
    user_instruction: str = Field(description="The user's instruction/request, in the target language, naturally implying a device action.")
    reasoning: str = Field(description="A short (1-2 sentence) reasoning about what the user wants and how to accomplish it, in the target language.")
    task_plan: str = Field(description="A brief step-by-step plan (1-3 short steps) for accomplishing the request, in the target language.")
    needs_tool_call: bool = Field(description="True if one of the available tools can satisfy the request.")
    tool_name: Optional[str] = Field(description="Exact name of the tool to call from the available tools list, or null if needs_tool_call is false.")
    tool_arguments_json: Optional[str] = Field(description="A JSON object (as a string) with the exact arguments for that tool call, matching its parameter schema exactly, or null if needs_tool_call is false.")
    tool_result_json: Optional[str] = Field(description="A JSON object (as a string) with a plausible result the tool would return (e.g. {\"status\": \"success\"} for an action, or realistic invented values for a query like battery status/time), or null if needs_tool_call is false.")
    final_response: str = Field(description="The assistant's final reply to the user in the target language: confirm the action (referencing tool_result_json values if relevant) if a tool was called, or explain/ask for clarification if not.")


def _sample_tool_subset(rng, hard: bool) -> tuple[list[dict], Optional[dict]]:
    all_tools = list(EDGE_DEVICE_TOOLS)
    rng.shuffle(all_tools)
    if hard:
        # Leave out enough tools that at least one plausible-but-unsupported
        # request category has no matching tool available.
        subset = all_tools[: rng.randint(2, 4)]
        return subset, None
    target = rng.choice(all_tools)
    others = [t for t in all_tools if t is not target]
    n_distractors = rng.randint(2, 5)
    subset = [target] + others[:n_distractors]
    rng.shuffle(subset)
    return subset, target


def _build_user_prompt(language_code: str, persona: str, tools_subset: list[dict], target: Optional[dict], hard: bool) -> str:
    tools_block = format_tools_for_system_prompt(tools_subset)
    if hard:
        task_line = (
            "The user's request should sound like a plausible device request, "
            "but must NOT be satisfiable by any of the available tools above "
            "(ask for something none of them cover, or something requiring "
            "information/action outside this tool set). Set needs_tool_call "
            "to false, leave tool_name/tool_arguments_json/tool_result_json "
            "null, and write a helpful final_response that explains the "
            "limitation or asks a clarifying question -- in the target "
            "language."
        )
    else:
        task_line = (
            f"The user's request should naturally require calling the "
            f"'{target['function']['name']}' tool specifically (not one of "
            f"the other available tools, which are distractors). Fill in "
            f"realistic arguments and a realistic result for it."
        )

    return f"""\
{language_style_block(language_code)}

{persona}

Available tools for this conversation:
{tools_block}

{task_line}"""


def build_requests() -> list[BatchRequestSpec]:
    response_format = to_strict_json_schema(EdgeActionSpec, name="edge_action")
    specs: list[BatchRequestSpec] = []

    for lang in LANGUAGES:
        for i in range(EXAMPLES_PER_LANGUAGE_PER_TASK):
            rng = rng_for(TASK_NAME, lang.code, i)
            persona = sample_persona(rng)
            is_hard = rng.random() < HARD_EXAMPLE_FRACTION
            tools_subset, target = _sample_tool_subset(rng, is_hard)

            user_prompt = _build_user_prompt(lang.code, persona, tools_subset, target, is_hard)
            final_system_prompt = FINAL_SYSTEM_PROMPT_TEMPLATE.format(
                tools_block=format_tools_for_system_prompt(tools_subset)
            )

            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(TASK_NAME, lang.code, i, variant="hard" if is_hard else "base"),
                    task=TASK_NAME,
                    language=lang.code,
                    system_prompt=GENERAL_DATA_QUALITY_RULES,
                    user_prompt=user_prompt,
                    response_format=response_format,
                    context={
                        "variant": "hard" if is_hard else "base",
                        "available_tools": tools_subset,
                        "final_system_prompt": final_system_prompt,
                    },
                )
            )
    return specs
