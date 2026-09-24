"""Schema for the seed files that drive every generation run.

A *seed file* is the single source of truth for one corpus kind (pretrain / sft / rl). It is a plain JSON
document so it can be read by any provider, on any platform (Together batch, OpenRouter, Modal, RunPod,
vast), and diffed in review. Nothing downstream invents task names, tags, tools or volumes -- they all come
from here, which is what stops two providers generating subtly different corpora.

Every seed file carries a `details` field: prose describing what this generation phase must produce and why.
It is injected verbatim into the meta-prompt, so editing it changes what the generator is asked for.

Layout:

    {
      "kind": "sft",
      "version": 1,
      "details": "...what this phase must produce...",
      "target_model": {...},          # what the data is FOR (SabiYarn: size, tokenizer, special tokens)
      "languages": {"yor": {...}},    # per-language volume + tier + guidance
      "tasks": [ {...TaskSpec...} ],  # what to generate, with per-task share of the total
      "tools": [ {...ToolSpec...} ],  # the tool catalogue conversations may call
      "conversation": {...},          # turn counts, who speaks last, think-tag policy
      "tags": [...],                  # the closed metadata vocabulary
      "format": {...}                 # special tokens + how a sample is serialized
    }

The schema is intentionally permissive about *content* and strict about *keys*: an unknown key is an error,
because a typo'd task name that silently generates nothing is the expensive failure mode here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

SEEDS_DIR = Path(__file__).resolve().parent.parent / "seeds"
KINDS = ("pretrain", "sft", "rl")

# ---------------------------------------------------------------------------
# Metadata tag vocabulary (closed set).
#
# Every sft/rl sample carries `lang` plus >= 1 of these. Normalised from the
# list given by the project owner: "structured-output" and "sructured output"
# were the same tag twice (one a typo), and "lang" is a field, not a tag.
# ---------------------------------------------------------------------------
TAGS: tuple[str, ...] = (
    "tool-calling",
    "rag",
    "structured-output",
    "qa",
    "extractive-qa",
    "health-education",
    "health-advice",
    "health-triaging",
    "financial-analysis",
    "language-identification",
    "ner",
    "pos-tagging",
    "intent-detection",
    "toxicity-spam-detection",
    "text-rephrasing",
    "topic-classification",
    "summarization",
    "sentiment-analysis",
    "content-writing",
    "translation",
    # Added: the behaviours the whole project exists to teach. Without their own
    # tags they cannot be counted, filtered or held out at eval time.
    "knowledge-boundary",   # "I have not heard of this" -> reason -> tool, or admit ignorance
    "insufficient-context",  # tool ran, result does not answer the question, say so
    "multi-turn-chat",
)


@dataclass
class ToolSpec:
    """A tool the assistant may call. Serialised into the system prompt as OpenAI-style JSON."""
    name: str
    description: str
    parameters: dict[str, Any]
    returns: str                       # prose: what a realistic result looks like
    failure_modes: list[str] = field(default_factory=list)  # empty results, irrelevant hits, errors


@dataclass
class TaskSpec:
    """One kind of exchange inside a conversation."""
    name: str
    tags: list[str]                    # from TAGS
    description: str                   # what the exchange must contain
    share: float                       # target share of all *exchanges* (normalised across tasks)
    task_plan: list[str] = field(default_factory=list)   # <task_plan> verbs, e.g. ["<|chat|>"]
    domain_flags: list[str] = field(default_factory=list)  # e.g. ["<|is_medical|>"]
    uses_tools: bool = False
    requires_think: bool = False       # must open with <think>...</think>
    languages: Optional[list[str]] = None   # None = all; else restrict (e.g. translation pairs)
    notes: str = ""


@dataclass
class LanguageSpec:
    code: str
    name: str
    tier: str                          # high | medium | low  (generation reliability, not speakers)
    samples: int                       # documents (pretrain) or conversations (sft/rl)
    guidance: str = ""                 # orthography traps to put in the meta-prompt


def _validate_tags(tags: list[str], where: str) -> None:
    bad = [t for t in tags if t not in TAGS]
    if bad:
        raise ValueError(f"{where}: unknown tag(s) {bad}. Allowed: {list(TAGS)}")


@dataclass
class Seed:
    kind: str
    details: str
    languages: list[LanguageSpec]
    tasks: list[TaskSpec] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    conversation: dict[str, Any] = field(default_factory=dict)
    target_model: dict[str, Any] = field(default_factory=dict)
    format: dict[str, Any] = field(default_factory=dict)
    # Expected share of requests that survive validation, per resource tier. 6-10 turn tool-calling
    # conversations in Fon or Efik genuinely fail more often than in Pidgin, so the planner over-requests
    # by 1/yield and the TARGET count is what lands rather than what was asked for.
    yield_by_tier: dict[str, float] = field(default_factory=lambda: {"high": 0.92, "medium": 0.85, "low": 0.72})
    version: int = 1

    # -- validation ---------------------------------------------------------
    def validate(self) -> "Seed":
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not self.details.strip():
            raise ValueError("details must not be empty: it is the meta-prompt's brief")
        if not self.languages:
            raise ValueError("at least one language is required")
        codes = [l.code for l in self.languages]
        if len(codes) != len(set(codes)):
            raise ValueError(f"duplicate language codes: {codes}")
        for l in self.languages:
            if l.samples < 0:
                raise ValueError(f"{l.code}: samples must be >= 0")
            if l.tier not in ("high", "medium", "low"):
                raise ValueError(f"{l.code}: tier must be high|medium|low")
        names = [t.name for t in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate task names: {names}")
        for t in self.tasks:
            _validate_tags(t.tags, f"task {t.name}")
            if t.share <= 0:
                raise ValueError(f"task {t.name}: share must be > 0")
            if t.languages:
                bad = [c for c in t.languages if c not in codes]
                if bad:
                    raise ValueError(f"task {t.name}: unknown language(s) {bad}")
            if t.uses_tools and not self.tools:
                raise ValueError(f"task {t.name} uses tools but the seed defines none")
        tool_names = [t.name for t in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"duplicate tool names: {tool_names}")
        if self.kind in ("sft", "rl"):
            c = self.conversation
            lo, hi = c.get("min_messages"), c.get("max_messages")
            if not isinstance(lo, int) or not isinstance(hi, int) or not 0 < lo <= hi:
                raise ValueError("conversation.min_messages/max_messages must be ints with 0 < min <= max")
            if c.get("ends_with") != "assistant":
                raise ValueError("conversation.ends_with must be 'assistant' (the model is graded on its reply)")
            if self.kind == "rl" and int(c.get("responses_per_prompt", 0)) < 2:
                raise ValueError("rl: conversation.responses_per_prompt must be >= 2 to form preference pairs")
        return self

    # -- volumes ------------------------------------------------------------
    def total_samples(self) -> int:
        """Target samples -- what should end up in the corpus after validation drops."""
        return sum(l.samples for l in self.languages)

    def yield_for(self, tier: str) -> float:
        y = float(self.yield_by_tier.get(tier, 0.8))
        if not 0 < y <= 1:
            raise ValueError(f"yield for tier {tier} must be in (0, 1], got {y}")
        return y

    def requests_for(self, lang: LanguageSpec) -> int:
        """How many to REQUEST so that `lang.samples` survive."""
        return int(round(lang.samples / self.yield_for(lang.tier)))

    def total_requests(self) -> int:
        return sum(self.requests_for(l) for l in self.languages)

    def task_shares(self) -> dict[str, float]:
        total = sum(t.share for t in self.tasks) or 1.0
        return {t.name: t.share / total for t in self.tasks}

    def plan(self) -> dict[str, dict[str, int]]:
        """language -> task -> number of samples. The generator's work list."""
        shares = self.task_shares()
        out: dict[str, dict[str, int]] = {}
        for lang in self.languages:
            n_requests = self.requests_for(lang)
            allowed = [t for t in self.tasks if not t.languages or lang.code in t.languages]
            denom = sum(shares[t.name] for t in allowed) or 1.0
            counts, run = {}, 0
            for i, t in enumerate(allowed):
                # last task absorbs the rounding remainder so the per-language total is exact
                n = n_requests - run if i == len(allowed) - 1 else round(n_requests * shares[t.name] / denom)
                counts[t.name] = max(0, n)
                run += counts[t.name]
            out[lang.code] = counts
        return out

    # -- io -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "version": self.version, "details": self.details,
            "target_model": self.target_model,
            "languages": {l.code: asdict(l) for l in self.languages},
            "tasks": [asdict(t) for t in self.tasks],
            "tools": [asdict(t) for t in self.tools],
            "conversation": self.conversation, "tags": list(TAGS), "format": self.format,
            "yield_by_tier": self.yield_by_tier,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or SEEDS_DIR / f"{self.kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, kind_or_path: str) -> "Seed":
        path = Path(kind_or_path)
        if not path.exists():
            path = SEEDS_DIR / f"{kind_or_path}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {"kind", "version", "details", "target_model", "languages", "tasks", "tools",
                 "conversation", "tags", "format", "yield_by_tier"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}")
        return cls(
            kind=raw["kind"], version=raw.get("version", 1), details=raw["details"],
            target_model=raw.get("target_model", {}),
            languages=[LanguageSpec(**v) for v in raw["languages"].values()],
            tasks=[TaskSpec(**t) for t in raw.get("tasks", [])],
            tools=[ToolSpec(**t) for t in raw.get("tools", [])],
            conversation=raw.get("conversation", {}), format=raw.get("format", {}),
            yield_by_tier=raw.get("yield_by_tier") or {"high": 0.92, "medium": 0.85, "low": 0.72},
        ).validate()
