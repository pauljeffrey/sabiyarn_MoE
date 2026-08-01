"""Shared infrastructure for all task generators.

Each generator (generators/rag.py, generators/summarization.py, ...)
builds a list of `BatchRequestSpec` objects -- one per synthetic example
to generate -- and hands them to `write_batch_and_manifest`, which writes:

  1. data/batch_input/<task>.jsonl   -- the actual OpenAI Batch API input
     file (one JSON object per line, exactly the shape the Batch API
     expects: custom_id/method/url/body).
  2. data/batch_input/<task>.manifest.jsonl -- a side-car file, one line
     per custom_id, carrying whatever extra context postprocessing needs
     later (e.g. which document a RAG example drew from) that must NOT be
     sent to the API as part of the request body.

Keeping (1) and (2) separate means the batch input file is always a
minimal, valid OpenAI Batch API payload -- nothing generator-specific
leaks into the API request itself.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.languages import get_language
from config.settings import BATCH_INPUT_DIR, DEFAULT_MAX_OUTPUT_TOKENS, GENERATION_MODEL, RANDOM_SEED

# ---------------------------------------------------------------------------
# Deterministic per-example RNG
# ---------------------------------------------------------------------------


def rng_for(*parts: Any) -> random.Random:
    """A reproducible RNG seeded from (global seed, task, language, index, ...).

    Using a fresh Random per example (rather than one shared stream) means
    re-running generation for a single missing/failed example later
    produces the same persona/sampling choices it would have gotten in the
    original run.
    """
    key = "|".join(str(p) for p in (RANDOM_SEED, *parts))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def make_custom_id(task: str, language: str, index: int, variant: str = "base") -> str:
    return f"{task}__{language}__{index:05d}__{variant}"


# ---------------------------------------------------------------------------
# Meta-prompt scaffolding
# ---------------------------------------------------------------------------

GENERAL_DATA_QUALITY_RULES = """\
You are generating a single high-quality synthetic training example for \
fine-tuning a small (280M parameter) multilingual assistant model that \
serves speakers of West African languages. The example must read like a \
real conversation between a real person and an assistant -- not like a \
textbook exercise or a translated English template.

Hard requirements:
- Write ALL user-facing text (user turns and assistant turns) in the \
target language specified below. Do not default to English or mix in \
English except for isolated proper nouns, brand names, or numbers where a \
real speaker would naturally do so.
- Do not produce a stiff, literal translation of an English sentence. \
Phrase things the way a native speaker actually would.
- Vary sentence length, opening phrasing, and structure. Do not reuse the \
same opening words across turns.
- Stay strictly factually consistent with any provided document/context; \
never invent facts that contradict it, and never state something the \
document does not support as if it were confirmed fact.
- Follow the exact output JSON schema given via structured outputs. Do not \
add commentary outside the schema.
"""


def language_style_block(language_code: str) -> str:
    lang = get_language(language_code)
    lines = [f"Target language: {lang.english_name} ({lang.endonym}), code '{lang.code}'."]
    if lang.style_notes:
        lines.append(f"Language-specific style guidance: {lang.style_notes}")
    if lang.resource_tier.value == "low":
        lines.append(
            "This is a lower-resource language for you -- prioritize simple, "
            "grammatically safe sentences over ambitious ones you are not "
            "confident about."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch request spec
# ---------------------------------------------------------------------------


@dataclass
class BatchRequestSpec:
    custom_id: str
    task: str
    language: str
    system_prompt: str  # meta-prompt system instructions (to the generation model)
    user_prompt: str  # meta-prompt user content (the concrete instance)
    response_format: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)
    model: str = GENERATION_MODEL
    temperature: float = 1.0
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def to_batch_line(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt},
                ],
                "response_format": self.response_format,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }

    def to_manifest_line(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "task": self.task,
            "language": self.language,
            "context": self.context,
        }


def write_batch_and_manifest(task_name: str, specs: list[BatchRequestSpec]) -> tuple[Path, Path, int]:
    batch_path = BATCH_INPUT_DIR / f"{task_name}.jsonl"
    manifest_path = BATCH_INPUT_DIR / f"{task_name}.manifest.jsonl"

    seen_ids: set[str] = set()
    with batch_path.open("w", encoding="utf-8") as bf, manifest_path.open("w", encoding="utf-8") as mf:
        for spec in specs:
            if spec.custom_id in seen_ids:
                raise ValueError(f"Duplicate custom_id: {spec.custom_id}")
            seen_ids.add(spec.custom_id)
            bf.write(json.dumps(spec.to_batch_line(), ensure_ascii=False) + "\n")
            mf.write(json.dumps(spec.to_manifest_line(), ensure_ascii=False) + "\n")

    return batch_path, manifest_path, len(specs)
