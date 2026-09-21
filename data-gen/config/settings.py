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

# ---------------------------------------------------------------------------
# Corpus kinds (pretrain / sft / dpo) -- see configs/*.yaml for per-kind knobs
# ---------------------------------------------------------------------------

CORPUS_KINDS: list[str] = ["pretrain", "sft", "dpo"]
CONFIGS_DIR = ROOT_DIR / "configs"

# Default generation model for the corpus kinds. gpt-4o-mini is cheap enough
# to generate at scale but is materially weaker than gpt-4o in low-resource
# languages (Efik, Urhobo, Fon, Ewe, Fulah/Fulfulde) -- see README.
CORPUS_GENERATION_MODEL = os.environ.get("DATA_GEN_CORPUS_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Pricing constants (USD per 1M tokens, *Batch API* rates)
#
# !!! VERIFY CURRENT PRICING before trusting any dollar figure !!!
# These are the single source of truth for every cost estimate the tooling
# prints (pipeline/estimate.py). Verified 2026-09-21 against OpenAI's published pricing
# page (gpt-4o-mini Batch $0.075 in / $0.30 out per 1M tokens = 50% of standard) -- re-check before a big run.
# Check https://openai.com/api/pricing/ and edit here -- nowhere else.
# ---------------------------------------------------------------------------

BATCH_PRICING_USD_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "gpt-4o-mini": (0.075, 0.30),  # sync price is $0.15 / $0.60
    "gpt-4o": (1.25, 5.00),  # sync price is $2.50 / $10.00
    "gpt-4o-2024-08-06": (1.25, 5.00),
}
# Used when a model name is not in the table above (prints a warning).
FALLBACK_BATCH_PRICING_USD_PER_M_TOKENS: tuple[float, float] = (1.25, 5.00)

# OpenAI tokenizers were trained mostly on English. Text in these languages
# (heavy diacritics, hooked letters, rare morphology) splits into many more
# tokens per *word*. English is ~1.3 tokens/word; the numbers below are
# deliberately conservative (over-estimate cost rather than under-estimate).
ENGLISH_TOKENS_PER_WORD = 1.35
# Measured with tiktoken o200k_base (the gpt-4o-mini tokenizer) on the 20 real sentences per language in
# data/curated_eval.jsonl: eng 1.11, hau 1.61, ibo 1.83, pcm 1.13, yor 2.31 tokens/word. Those four are
# rounded UP a little (short samples, and generated text is longer/more formal); the languages with no
# measured sample (efi urh fon ewe aka twi ful fuv) are extrapolated from the closest measured script
# behaviour and left deliberately higher. Real cost is expected at or a little below the estimate.
AFRICAN_LANG_TOKENS_PER_WORD_DEFAULT = 2.6
TOKENS_PER_WORD_BY_LANGUAGE: dict[str, float] = {
    "pcm": 1.35,  # measured 1.13 (kept >= the English constant)
    "hau": 1.8,  # measured 1.61
    "ibo": 2.0,  # measured 1.83
    "yor": 2.5,  # measured 2.31 (tone marks + underdots)
    "twi": 2.4,
    "aka": 2.4,
    "efi": 2.7,
    "urh": 2.7,
    "ewe": 2.9,
    "fon": 3.0,
    "ful": 2.4,
    "fuv": 2.4,
}
# Rough English prompt tokenisation (meta-prompts are written in English).
CHARS_PER_TOKEN_ENGLISH = 4.0

# Batch API hard limit on the size of one input file (bytes). Kept with the
# pricing constants because, like them, it is an external API fact to verify.
MAX_BATCH_FILE_BYTES = 190_000_000  # documented limit is 200 MB; keep headroom



class DataPaths:
    """Where a run reads/writes its files.

    Corpus tooling takes an explicit `data_dir` (defaulting to `DATA_DIR`)
    instead of reading module-level constants, so tests and side-by-side
    pilot runs can point at a scratch directory without env-var tricks.
    Directories are created lazily on first access.
    """

    def __init__(self, root: "Path | str | None" = None) -> None:
        self.root = Path(root) if root is not None else DATA_DIR

    def _ensure(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def batch_input(self) -> Path:
        return self._ensure("batch_input")

    @property
    def batch_output(self) -> Path:
        return self._ensure("batch_output")

    @property
    def processed(self) -> Path:
        return self._ensure("processed")

    @property
    def reports(self) -> Path:
        return self._ensure("reports")

    @property
    def manifest(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / "batch_manifest.jsonl"
