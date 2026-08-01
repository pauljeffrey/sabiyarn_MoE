"""Global generation settings.

Everything that controls cost, volume, and model choice lives here so a run
can be tuned without touching generator code. All paths are relative to the
data-gen package root unless overridden via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT_DIR / "templates"
CHAT_TEMPLATE_PATH = TEMPLATES_DIR / "chat_template.jinja"
DOCUMENTS_DIR = ROOT_DIR / "documents"
SAMPLE_DOCUMENTS_DIR = DOCUMENTS_DIR / "sample"

DATA_DIR = Path(os.environ.get("DATA_GEN_OUTPUT_DIR", ROOT_DIR / "data"))
BATCH_INPUT_DIR = DATA_DIR / "batch_input"
BATCH_OUTPUT_DIR = DATA_DIR / "batch_output"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "batch_manifest.jsonl"
REPORTS_DIR = DATA_DIR / "reports"

for d in (BATCH_INPUT_DIR, BATCH_OUTPUT_DIR, PROCESSED_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model / API
# ---------------------------------------------------------------------------

# Model used for generation via the Batch API. Must support Structured
# Outputs (json_schema with strict:true) in chat.completions.
GENERATION_MODEL = os.environ.get("DATA_GEN_MODEL", "gpt-4o-2024-08-06")

# Optional stronger model for low-resource languages / a verification pass.
# Leave equal to GENERATION_MODEL to disable tiering.
HIGH_EFFORT_MODEL = os.environ.get("DATA_GEN_HIGH_EFFORT_MODEL", "gpt-4o-2024-08-06")

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 3000

# Batch API completion window (OpenAI currently only offers "24h").
BATCH_COMPLETION_WINDOW = "24h"

# ---------------------------------------------------------------------------
# Volume / distribution targets
# ---------------------------------------------------------------------------

# Target number of *conversations* generated per (task, language) cell.
# Keep this equal across languages by default -- diversity comes from
# varying personas/domains/turn-count within a cell, not from skewing counts.
EXAMPLES_PER_LANGUAGE_PER_TASK = int(os.environ.get("DATA_GEN_PER_CELL", "40"))

# Relative weight of each task when building a combined run. These do not
# have to sum to 1; they're normalized. Adjust to shift overall dataset
# composition without touching per-cell counts above.
TASK_WEIGHTS: dict[str, float] = {
    "rag": 1.0,
    "summarization": 1.0,
    "edge_action": 1.0,
    "structured_output": 1.0,
    "math_stats": 1.0,
    "translation": 1.0,
}

TASK_NAMES: list[str] = list(TASK_WEIGHTS.keys())

# A small fraction of examples per cell get a *distractor-heavy* or
# *adversarial* variant (irrelevant tools available, ambiguous query, no
# answer in the document, etc.) to teach robustness rather than pure
# pattern-matching. Expressed as a fraction of EXAMPLES_PER_LANGUAGE_PER_TASK.
HARD_EXAMPLE_FRACTION = 0.2

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = int(os.environ.get("DATA_GEN_SEED", "42"))
