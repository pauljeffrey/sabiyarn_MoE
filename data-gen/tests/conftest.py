"""Test bootstrap.

* Puts `data-gen/` on sys.path (the scripts import each other that way).
* Points DATA_GEN_OUTPUT_DIR at a scratch dir BEFORE any project module is
  imported, because `config/settings.py` creates directories at import time
  and we never want tests writing into `data-gen/data`.
* No test in this suite touches the network or the OpenAI API.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRATCH = tempfile.mkdtemp(prefix="data_gen_tests_")
os.environ["DATA_GEN_OUTPUT_DIR"] = _SCRATCH
os.environ.pop("OPENAI_API_KEY", None)  # make accidental API use impossible

import pytest  # noqa: E402

from config.corpus_config import CorpusConfig, load_config  # noqa: E402


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture(scope="session")
def presets() -> dict[str, CorpusConfig]:
    return {kind: load_config(kind) for kind in ("pretrain", "sft", "dpo")}


def tiny(cfg: CorpusConfig, per_language: int = 3, languages: list[str] | None = None) -> CorpusConfig:
    return cfg.with_overrides(per_language=per_language, languages=languages)
