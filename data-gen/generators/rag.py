"""RAG task generator.

For each example we pick a source document, ask the generation model to
write a multi-turn conversation in a target language where the user asks
questions about the document's content and the assistant answers by
"searching" it. The generation model only ever tells us WHICH chunk is
relevant (`relevant_chunk_id`) -- it never writes the tool's returned text
itself. That keeps retrieval grounded in the real document rather than a
model's paraphrase of it: `pipeline/postprocess.py` looks up the actual
chunk text from the Document object and injects it as the tool result when
rendering the final training example.

A fraction of examples are "hard": the user asks something the document
does not actually answer, so the correct behavior is still to search, but
then to say (in the target language) that the information isn't available
-- this teaches the model not to hallucinate when retrieval comes up empty.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from config.languages import LANGUAGES
from config.settings import EXAMPLES_PER_LANGUAGE_PER_TASK, HARD_EXAMPLE_FRACTION
from documents.loader import Document, load_default_documents
from generators.base import (
    GENERAL_DATA_QUALITY_RULES,
    BatchRequestSpec,
    language_style_block,
    make_custom_id,
    rng_for,
)
from generators.personas import sample_persona
from schemas.strict import to_strict_json_schema
from schemas.tools import SEARCH_DOCUMENT_TOOL, format_tools_for_system_prompt

TASK_NAME = "rag"

FINAL_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful multilingual assistant for West African language speakers. \
You do not have a document loaded in your context. When the user asks \
something that might be answered by the knowledge base, call the \
search_document tool to look it up before answering. Never claim \
information you did not retrieve. If the search does not return anything \
relevant, say so honestly instead of guessing.

{tools_block}"""


class RagTurnSpec(BaseModel):
    user_message: str = Field(description="The user's question, in the target language.")
    search_query: str = Field(description="Search query the assistant would use to look this up (English or target language).")
    answerable: bool = Field(description="True if the provided document chunks actually contain the answer.")
    relevant_chunk_id: Optional[int] = Field(description="The chunk number that answers the question, or null if answerable is false.")
    assistant_response: str = Field(description="The assistant's final answer to the user, in the target language, grounded ONLY in the retrieved chunk if answerable, or a polite 'I don't have that information' style response if not.")


class RagConversationSpec(BaseModel):
    turns: list[RagTurnSpec] = Field(min_length=2, max_length=4, description="2 to 4 sequential turns; each later turn may naturally follow up on the previous one (same topic thread), simulating a real multi-turn chat.")


def _build_user_prompt(doc: Document, language_code: str, persona: str, n_turns: int, n_hard: int) -> str:
    return f"""\
{language_style_block(language_code)}

{persona}

Source document (English, already split into numbered chunks -- the user \
and assistant conversation should be entirely in the target language, but \
you use this English document as your ground truth for facts):

{doc.as_numbered_block()}

Write a {n_turns}-turn conversation between this user and an assistant, \
about topics covered in the document above, where each turn follows the \
schema (user_message, search_query, answerable, relevant_chunk_id, \
assistant_response). Later turns should feel like a real follow-up \
conversation on the same general topic (not unrelated random questions each \
time) -- e.g. asking for more detail, a related aspect, or a clarification \
about something mentioned in a previous answer.

Exactly {n_hard} of the {n_turns} turn(s) must be genuinely NOT answerable \
from the document (a related-sounding but out-of-scope question) -- set \
answerable=false and relevant_chunk_id=null for those, and write a natural, \
honest 'I don't have that information' style assistant_response for them, \
still in the target language. For the rest, ground the answer strictly in \
the correct chunk's content."""


def build_requests() -> list[BatchRequestSpec]:
    documents = load_default_documents()
    tools_block = format_tools_for_system_prompt([SEARCH_DOCUMENT_TOOL])
    final_system_prompt = FINAL_SYSTEM_PROMPT_TEMPLATE.format(tools_block=tools_block)
    response_format = to_strict_json_schema(RagConversationSpec, name="rag_conversation")

    specs: list[BatchRequestSpec] = []
    for lang in LANGUAGES:
        for i in range(EXAMPLES_PER_LANGUAGE_PER_TASK):
            rng = rng_for(TASK_NAME, lang.code, i)
            doc = documents[rng.randrange(len(documents))]
            persona = sample_persona(rng)
            n_turns = rng.choice([2, 2, 3, 3, 4])
            is_hard = rng.random() < HARD_EXAMPLE_FRACTION
            n_hard = 1 if is_hard else 0
            n_hard = min(n_hard, n_turns - 1)  # never make every turn unanswerable

            user_prompt = _build_user_prompt(doc, lang.code, persona, n_turns, n_hard)

            specs.append(
                BatchRequestSpec(
                    custom_id=make_custom_id(TASK_NAME, lang.code, i),
                    task=TASK_NAME,
                    language=lang.code,
                    system_prompt=GENERAL_DATA_QUALITY_RULES,
                    user_prompt=user_prompt,
                    response_format=response_format,
                    context={
                        "doc_id": doc.doc_id,
                        "final_system_prompt": final_system_prompt,
                        "variant": "hard" if is_hard else "base",
                    },
                )
            )
    return specs
