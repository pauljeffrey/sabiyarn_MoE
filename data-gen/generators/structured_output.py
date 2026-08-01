"""Structured-output (JSON extraction) task generator.

For each example we pick one of the domain schemas in
schemas/structured_output_schemas.py, ask the model to invent a short
English source text matching that domain, a natural user instruction in
the target language asking for the info in structured form, and the
extracted value itself -- validated structurally at generation time by
embedding the domain's own JSON schema directly into the Structured
Outputs response_format (not just prompted informally), so a malformed
extraction cannot come back from the API in the first place.

Descriptive/free-text field values are written in the target language;
proper nouns, numbers, dates and currency amounts are preserved as they'd
naturally appear regardless of language.
"""

from __future__ import annotations

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
from schemas.strict import object_response_format, tighten_raw_schema
from schemas.structured_output_schemas import STRUCTURED_OUTPUT_SCHEMAS

TASK_NAME = "structured_output"

FINAL_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful multilingual assistant for West African language speakers. \
When asked to extract structured information from a document, respond with \
ONLY a single JSON object that exactly matches the schema below -- no \
extra commentary, no markdown code fences, just the JSON.

JSON schema:
{schema_json}

Source document:
__SOURCE_TEXT_PLACEHOLDER__"""


class EnvelopeMeta(BaseModel):
    """Fixed part of the structured-output generation schema; the
    'extracted' field is added dynamically per domain in build_requests().
    """

    source_text: str = Field(description="Short invented English source text containing the information to extract (per the domain hint given).")
    user_instruction: str = Field(description="A natural request from the user, in the target language, asking to pull this information into a structured/organized format. Vary phrasing -- don't always say the equivalent of 'extract as JSON'; a real user just asks in plain language.")


def _build_user_prompt(language_code: str, persona: str, domain_name: str, source_hint: str, schema_json: str) -> str:
    return f"""\
{language_style_block(language_code)}

{persona}

Domain: {domain_name}
Invent {source_hint}.

The target JSON schema you must extract into (field values that are \
free/descriptive text should be written in the target language; proper \
nouns, numbers, dates, and currency amounts should be written as they \
would naturally appear regardless of language):
{schema_json}

Write: (1) the invented English source_text, (2) a natural user_instruction \
in the target language asking the assistant to organize/extract this info, \
and (3) the correctly extracted value for every field in the schema."""


def build_requests() -> list[BatchRequestSpec]:
    import json

    specs: list[BatchRequestSpec] = []

    for lang in LANGUAGES:
        for i in range(EXAMPLES_PER_LANGUAGE_PER_TASK):
            rng = rng_for(TASK_NAME, lang.code, i)
            persona = sample_persona(rng)
            domain = STRUCTURED_OUTPUT_SCHEMAS[rng.randrange(len(STRUCTURED_OUTPUT_SCHEMAS))]

            envelope_props = EnvelopeMeta.model_json_schema()["properties"]
            # Pydantic emits plain {"type": "string", "description": ...};
            # no $ref to resolve here since EnvelopeMeta has no nested models.
            tightened_domain_schema = tighten_raw_schema(domain["schema"])
            properties = {
                **envelope_props,
                "extracted": tightened_domain_schema,
            }
            response_format = object_response_format(properties, name=f"structured_output_{domain['name']}")

            schema_json_str = json.dumps(domain["schema"], ensure_ascii=False, indent=2)
            user_prompt = _build_user_prompt(lang.code, persona, domain["name"], domain["source_hint"], schema_json_str)
            # NOTE: source text isn't known until the model responds, so we
            # fill schema_json now and leave __SOURCE_TEXT_PLACEHOLDER__ for
            # postprocess.py to str.replace() in later -- NOT another
            # .format() call, since schema_json_str itself contains braces
            # that would confuse str.format().
            final_system_prompt = FINAL_SYSTEM_PROMPT_TEMPLATE.format(schema_json=schema_json_str)

            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(TASK_NAME, lang.code, i),
                    task=TASK_NAME,
                    language=lang.code,
                    system_prompt=GENERAL_DATA_QUALITY_RULES,
                    user_prompt=user_prompt,
                    response_format=response_format,
                    context={
                        "domain": domain["name"],
                        "final_system_prompt_template": final_system_prompt,
                    },
                )
            )
    return specs
