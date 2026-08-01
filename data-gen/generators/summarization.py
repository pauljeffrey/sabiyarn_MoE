"""Summarization task generator.

Two variants, split across the per-language budget:
  - "document": the source document is given directly in the system prompt
    (plain text, not chunk-numbered -- unlike RAG, the assistant has the
    whole thing in context and there's no retrieval step). The user asks
    for a summary in their language, optionally with a follow-up request
    to shorten/expand/change format.
  - "chat_history": there is no document; instead the model fabricates a
    short prior conversation on an everyday topic, then the user asks the
    assistant to summarize "what we discussed so far", in their language.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from config.languages import LANGUAGES
from config.settings import EXAMPLES_PER_LANGUAGE_PER_TASK
from documents.loader import load_default_documents
from generators.base import (
    GENERAL_DATA_QUALITY_RULES,
    BatchRequestSpec,
    language_style_block,
    make_custom_id,
    rng_for,
)
from generators.personas import sample_persona
from schemas.strict import to_strict_json_schema

TASK_NAME = "summarization"

FINAL_SYSTEM_PROMPT_DOC = """\
You are a helpful multilingual assistant for West African language speakers. \
Summarize the document below when asked, in whatever language the user \
writes in. Be accurate and do not add information the document does not \
contain.

Document:
{document_text}"""

FINAL_SYSTEM_PROMPT_CHAT = """\
You are a helpful multilingual assistant for West African language speakers. \
If the user asks you to summarize the conversation so far, give a concise, \
accurate summary in whatever language they ask in."""

SUMMARY_STYLE_BANK = [
    "a single-sentence summary",
    "a short summary of 2-3 sentences",
    "3 short bullet points",
    "a brief summary suitable for someone in a hurry",
    "a slightly more detailed summary covering the main points",
]

TOPIC_BANK = [
    "a delayed bus/keke trip and what to do next",
    "a child's school fees and payment plan",
    "a minor health concern and whether to see a doctor",
    "planning a small family event (naming ceremony, wedding, funeral rites)",
    "a mobile money/bank transfer that didn't go through",
    "negotiating the price of goods at a market",
    "a landlord/tenant disagreement about rent or repairs",
    "advice about a phone that stopped charging properly",
    "planning what crops to plant this season",
    "a disagreement about splitting a bill among friends",
]


class DocSummarizationSpec(BaseModel):
    user_instruction: str = Field(description="The user's request to summarize the document, in the target language. Vary phrasing and desired style/length naturally.")
    summary: str = Field(description="The assistant's summary in the target language, matching the requested style/length.")
    has_followup: bool = Field(description="Whether there is a natural follow-up turn (e.g. asking to shorten, expand, or reformat the summary).")
    followup_user_instruction: Optional[str] = Field(description="The follow-up request in the target language, or null if has_followup is false.")
    followup_summary: Optional[str] = Field(description="The assistant's revised summary in the target language, or null if has_followup is false.")


class ChatTurnSpec(BaseModel):
    user_message: str = Field(description="A user message in the target language, part of an everyday prior conversation (not about summarization).")
    assistant_message: str = Field(description="The assistant's reply in the target language.")


class ChatHistorySummarizationSpec(BaseModel):
    prior_turns: list[ChatTurnSpec] = Field(min_length=2, max_length=5, description="A short, realistic prior conversation on the given everyday topic.")
    summarize_request: str = Field(description="The user's final message asking the assistant to summarize the conversation so far, in the target language.")
    summary: str = Field(description="The assistant's summary of the prior conversation, in the target language.")


def _doc_variant_prompt(document_text: str, language_code: str, persona: str, style: str) -> str:
    return f"""\
{language_style_block(language_code)}

{persona}

Source document (English -- the conversation itself should be entirely in \
the target language):

{document_text}

Write a summarization request-and-response example where the user asks for \
{style}. Decide naturally whether a realistic follow-up turn happens \
(e.g. asking to shorten/expand/reformat) and set has_followup accordingly; \
if not, leave the followup fields null."""


def _chat_history_variant_prompt(language_code: str, persona: str, topic: str, n_turns: int) -> str:
    return f"""\
{language_style_block(language_code)}

{persona}

Invent a short, realistic {n_turns}-turn prior conversation between this \
user and the assistant about: {topic}. Entirely in the target language. \
Then write the user's final message asking the assistant to summarize what \
they discussed so far (phrase this request naturally, in the target \
language, not as a literal translation of "summarize our conversation"), \
and the assistant's accurate summary of the prior_turns."""


def build_requests() -> list[BatchRequestSpec]:
    documents = load_default_documents()
    doc_response_format = to_strict_json_schema(DocSummarizationSpec, name="doc_summarization")
    chat_response_format = to_strict_json_schema(ChatHistorySummarizationSpec, name="chat_history_summarization")

    specs: list[BatchRequestSpec] = []
    for lang in LANGUAGES:
        n_doc = EXAMPLES_PER_LANGUAGE_PER_TASK // 2
        n_chat = EXAMPLES_PER_LANGUAGE_PER_TASK - n_doc

        for i in range(n_doc):
            rng = rng_for(TASK_NAME, lang.code, "doc", i)
            doc = documents[rng.randrange(len(documents))]
            persona = sample_persona(rng)
            style = rng.choice(SUMMARY_STYLE_BANK)
            document_text = "\n\n".join(c.text for c in doc.chunks)

            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(TASK_NAME, lang.code, i, variant="doc"),
                    task=TASK_NAME,
                    language=lang.code,
                    system_prompt=GENERAL_DATA_QUALITY_RULES,
                    user_prompt=_doc_variant_prompt(document_text, lang.code, persona, style),
                    response_format=doc_response_format,
                    context={
                        "variant": "doc",
                        "doc_id": doc.doc_id,
                        "final_system_prompt": FINAL_SYSTEM_PROMPT_DOC.format(document_text=document_text),
                    },
                )
            )

        for i in range(n_chat):
            rng = rng_for(TASK_NAME, lang.code, "chat", i)
            persona = sample_persona(rng)
            topic = rng.choice(TOPIC_BANK)
            n_turns = rng.choice([2, 3, 3, 4, 5])

            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(TASK_NAME, lang.code, i, variant="chat"),
                    task=TASK_NAME,
                    language=lang.code,
                    system_prompt=GENERAL_DATA_QUALITY_RULES,
                    user_prompt=_chat_history_variant_prompt(lang.code, persona, topic, n_turns),
                    response_format=chat_response_format,
                    context={
                        "variant": "chat",
                        "final_system_prompt": FINAL_SYSTEM_PROMPT_CHAT,
                    },
                )
            )
    return specs
