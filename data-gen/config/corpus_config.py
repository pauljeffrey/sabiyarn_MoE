"""Loader/validator for `configs/{pretrain,sft,dpo}.yaml`.

WHY a dedicated loader: the yaml files are the user-facing control surface
(volumes, weights, quality thresholds), so typos must fail loudly and early --
a misspelled language code or weight key should not silently produce an
unbalanced or wrongly-filtered dataset after a paid Batch run.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from config.languages import LANGUAGE_CODES
from config.settings import CONFIGS_DIR, CORPUS_GENERATION_MODEL, CORPUS_KINDS

# Only the 12 non-English languages are generated for the corpus kinds.
TARGET_LANGUAGE_CODES: list[str] = [c for c in LANGUAGE_CODES if c != "eng"]

# Per-kind defaults for the quality/dedup/judge sections. A yaml file only
# needs to list what it overrides; unknown keys are rejected.
_COMMON_QUALITY: dict[str, Any] = {
    "max_repetition_ratio": 0.25,  # share of repeated word 4-grams
    "max_repeated_line_ratio": 0.35,  # share of duplicated non-empty lines/paragraphs
    "max_top_token_share": 0.30,  # one token making up >30% of a 30+ word text = degenerate
    "max_english_ratio": 0.15,  # share of tokens that are common English stopwords
    "max_english_ratio_by_language": {"pcm": 0.30},  # Pidgin legitimately shares vocabulary
    "min_english_check_tokens": 8,  # shorter texts are too noisy for the ratio
    "max_non_latin_ratio": 0.20,  # clear script failure (e.g. Cyrillic/CJK/Arabic output)
    "drop_low_confidence": True,
    "drop_meta_text": True,
}

DEFAULT_QUALITY: dict[str, dict[str, Any]] = {
    "pretrain": {
        **_COMMON_QUALITY,
        "min_words": 60,
        "max_words": 1100,
        "bucket_min_ratio": 0.5,  # need >= 50% of the requested bucket's lower bound
        "bucket_max_ratio": 2.0,  # and <= 200% of its upper bound
        "min_title_chars": 3,
        "max_title_chars": 200,
        "drop_on_self_check_false": True,
        "drop_markdown_headings": True,
    },
    "sft": {
        **_COMMON_QUALITY,
        "min_instruction_chars": 6,
        "max_instruction_chars": 1500,
        "max_input_chars": 5000,
        "min_response_chars": 1,
        "max_response_chars": 6000,
    },
    "dpo": {
        **_COMMON_QUALITY,
        "min_instruction_chars": 6,
        "max_instruction_chars": 1500,
        "max_input_chars": 5000,
        "min_response_chars": 2,
        "max_response_chars": 6000,
        "min_length_ratio": 0.3,  # len(rejected)/len(chosen) lower bound (unless length-related flaw)
        "max_length_ratio": 3.0,
    },
}

DEFAULT_DEDUP: dict[str, Any] = {
    "near_dup_threshold": 0.8,  # Jaccard over word shingles
    "shingle_size": 5,
    "within_language": True,
    "cross_language": True,
}

DEFAULT_JUDGE: dict[str, Any] = {
    "model": CORPUS_GENERATION_MODEL,
    "temperature": 0.0,
    "max_tokens": 300,
    "sample_fraction": 0.2,
    "min_scores": {
        "language_correctness": 4,
        "fluency": 3,
        "factuality": 3,
        "instruction_following": 3,
        "usefulness": 3,
    },
    "drop_unjudged": False,
}

_ALLOWED_TOP_KEYS = {
    "kind", "model", "temperature", "max_tokens", "seed", "samples_per_language",
    "attribute_weights", "domain_group_weights", "quality", "dedup", "judge", "system_prompt",
}


@dataclass
class CorpusConfig:
    kind: str
    model: str
    temperature: float
    max_tokens: int
    seed: int
    samples_per_language: dict[str, int]
    attribute_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    domain_group_weights: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    dedup: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    path: Optional[Path] = None

    @property
    def languages(self) -> list[str]:
        return list(self.samples_per_language)

    @property
    def total_requests(self) -> int:
        return sum(self.samples_per_language.values())

    def with_overrides(self, *, per_language: Optional[int] = None, languages: Optional[list[str]] = None) -> "CorpusConfig":
        """Copy with a uniform per-language count and/or a language subset (for pilots/tests)."""
        cfg = copy.deepcopy(self)
        if languages is not None:
            bad = [c for c in languages if c not in TARGET_LANGUAGE_CODES]
            if bad:
                raise ValueError(f"Unsupported language(s) {bad}; supported: {TARGET_LANGUAGE_CODES}")
            cfg.samples_per_language = {c: cfg.samples_per_language[c] for c in languages}
        if per_language is not None:
            cfg.samples_per_language = {c: int(per_language) for c in cfg.samples_per_language}
        return cfg


def _merge_section(defaults: dict[str, Any], given: Optional[dict[str, Any]], section: str) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for k, v in (given or {}).items():
        if k not in merged:
            raise ValueError(f"Unknown key {k!r} in `{section}` section. Allowed: {sorted(merged)}")
        merged[k] = v
    return merged


def parse_config(raw: dict[str, Any], *, path: Optional[Path] = None) -> CorpusConfig:
    unknown = set(raw) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level key(s) in config: {sorted(unknown)}")
    kind = raw.get("kind")
    if kind not in CORPUS_KINDS:
        raise ValueError(f"`kind` must be one of {CORPUS_KINDS}, got {kind!r}")

    spl = raw.get("samples_per_language")
    if not isinstance(spl, dict) or "default" not in spl:
        raise ValueError("`samples_per_language` must be a map with a `default` entry")
    bad = [k for k in spl if k != "default" and k not in TARGET_LANGUAGE_CODES]
    if bad:
        raise ValueError(f"Unsupported language code(s) in samples_per_language: {bad}. Supported: {TARGET_LANGUAGE_CODES} (English is excluded)")
    resolved: dict[str, int] = {}
    for code in TARGET_LANGUAGE_CODES:
        n = spl.get(code, spl["default"])
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"samples_per_language.{code} must be a non-negative integer, got {n!r}")
        resolved[code] = n

    quality = _merge_section(DEFAULT_QUALITY[kind], raw.get("quality"), "quality")
    dedup = _merge_section(DEFAULT_DEDUP, raw.get("dedup"), "dedup")
    judge = _merge_section(DEFAULT_JUDGE, raw.get("judge"), "judge")
    if raw.get("judge") and "min_scores" in raw["judge"]:
        judge["min_scores"] = {**DEFAULT_JUDGE["min_scores"], **raw["judge"]["min_scores"]}

    return CorpusConfig(
        kind=kind,
        model=str(raw.get("model", CORPUS_GENERATION_MODEL)),
        temperature=float(raw.get("temperature", 1.0)),
        max_tokens=int(raw.get("max_tokens", 2000)),
        seed=int(raw.get("seed", 42)),
        samples_per_language=resolved,
        attribute_weights=copy.deepcopy(raw.get("attribute_weights") or {}),
        domain_group_weights={k: float(v) for k, v in (raw.get("domain_group_weights") or {}).items()},
        quality=quality,
        dedup=dedup,
        judge=judge,
        system_prompt=str(raw.get("system_prompt", "")),
        path=path,
    )


def load_config(path_or_kind: "str | Path") -> CorpusConfig:
    """Load `configs/<kind>.yaml` from a path, or from a bare kind name."""
    p = Path(path_or_kind)
    if not p.suffix and str(path_or_kind) in CORPUS_KINDS:
        p = CONFIGS_DIR / f"{path_or_kind}.yaml"
    elif not p.is_absolute() and not p.exists():
        # allow `configs/sft.yaml` relative to the data-gen root regardless of cwd
        candidate = CONFIGS_DIR.parent / p
        if candidate.exists():
            p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return parse_config(raw, path=p)
