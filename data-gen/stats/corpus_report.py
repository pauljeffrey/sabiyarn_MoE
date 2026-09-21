"""Run statistics for the corpus kinds (pretrain / sft / dpo).

Answers the questions you need before trusting a dataset: how many records
per language survived, why the rest were dropped, and whether the *kept*
data is still balanced across domains, sub-topics, tasks and genres (dropping
is not uniform -- low-resource languages lose more -- so coverage after
filtering can differ from coverage as requested).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sampling.sampler import coverage_report
from sampling.taxonomy import DOMAIN_KEYS, all_pairs

# Attributes whose per-language distribution we report for each kind.
COVERAGE_ATTRIBUTES: dict[str, list[str]] = {
    "pretrain": ["domain", "genre", "register", "audience", "length_bucket", "perspective", "era", "difficulty", "locale"],
    "sft": ["domain", "task", "register", "instruction_style", "response_length", "difficulty", "locale"],
    "dpo": ["domain", "task", "rejection_type", "register", "instruction_style", "response_length", "difficulty", "locale"],
}


def _vocabularies(kind: str) -> dict[str, list[str]]:
    from sampling import taxonomy as t

    vocab: dict[str, list[str]] = {
        "domain": list(DOMAIN_KEYS),
        "pair": [f"{d}::{s}" for d, s in all_pairs()],
    }
    if kind == "pretrain":
        vocab.update(genre=list(t.GENRES), register=list(t.REGISTERS), audience=list(t.AUDIENCES), length_bucket=list(t.LENGTH_BUCKETS),
                     perspective=list(t.PERSPECTIVES), era=list(t.ERAS), difficulty=list(t.DIFFICULTIES))
    else:
        from generators.sft_tasks import TASK_KEYS

        vocab.update(task=list(TASK_KEYS), register=list(t.USER_REGISTERS), instruction_style=list(t.INSTRUCTION_STYLES),
                     response_length=list(t.RESPONSE_LENGTHS), difficulty=list(t.DIFFICULTIES))
        if kind == "dpo":
            from generators.dpo import REJECTION_KEYS

            vocab["rejection_type"] = list(REJECTION_KEYS)
    return vocab


class CorpusStats:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.generated: Counter = Counter()  # language -> requests with an output line
        self.kept: Counter = Counter()
        self.drop_first_reason: dict[str, Counter] = defaultdict(Counter)  # language -> reason -> n (one per dropped record)
        self.reason_hits: dict[str, Counter] = defaultdict(Counter)  # language -> reason -> n (every reason of every dropped record)
        self.warning_hits: dict[str, Counter] = defaultdict(Counter)
        self._kept_attrs: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)  # reason -> a few custom_ids

    # -- recording -----------------------------------------------------

    def record_generated(self, language: str) -> None:
        self.generated[language] += 1

    def record_drop(self, language: str, reasons: list[str], custom_id: str = "") -> None:
        self.drop_first_reason[language][reasons[0]] += 1
        for r in reasons:
            self.reason_hits[language][r] += 1
        key = reasons[0].split(":")[0]
        if len(self.examples[key]) < 5:
            self.examples[key].append({"custom_id": custom_id, "language": language, "reasons": reasons})

    def record_warnings(self, language: str, warnings: list[str]) -> None:
        for w in warnings:
            self.warning_hits[language][w] += 1

    def record_kept(self, language: str, attrs: dict[str, str]) -> None:
        self.kept[language] += 1
        self._kept_attrs[language].append(attrs)

    # -- aggregation ---------------------------------------------------

    def languages(self) -> list[str]:
        return sorted(set(self.generated) | set(self.kept))

    def dropped(self, language: str) -> int:
        return sum(self.drop_first_reason[language].values())

    def totals(self) -> dict[str, int]:
        return {
            "generated": sum(self.generated.values()),
            "kept": sum(self.kept.values()),
            "dropped": sum(self.dropped(lang) for lang in self.languages()),
        }

    def summary(self) -> dict[str, Any]:
        vocab = _vocabularies(self.kind)
        attr_names = COVERAGE_ATTRIBUTES[self.kind]
        per_language: dict[str, Any] = {}
        for lang in self.languages():
            attrs = self._kept_attrs.get(lang, [])
            cov = coverage_report(attrs, vocabularies={k: v for k, v in vocab.items() if k in attr_names or k == "pair"}, include_counts=False)
            gen = self.generated[lang]
            per_language[lang] = {
                "generated": gen,
                "kept": self.kept[lang],
                "kept_rate": round(self.kept[lang] / gen, 4) if gen else None,
                "dropped": self.dropped(lang),
                "drop_reasons": dict(self.drop_first_reason[lang].most_common()),
                "all_reason_hits": dict(self.reason_hits[lang].most_common()),
                "warnings": dict(self.warning_hits[lang].most_common()),
                "domain_counts": dict(Counter(a["domain"] for a in attrs).most_common()),
                ("task_counts" if self.kind != "pretrain" else "genre_counts"): dict(
                    Counter(a["task" if self.kind != "pretrain" else "genre"] for a in attrs).most_common()
                ),
                "coverage": {
                    name: {k: v for k, v in cov["attributes"].get(name, {}).items() if k in ("distinct", "vocabulary", "entropy_norm", "max_min_ratio")}
                    for name in ["pair", *attr_names]
                    if name in cov["attributes"]
                },
            }
        return {
            "kind": self.kind,
            "totals": self.totals(),
            "languages": per_language,
            "drop_reason_examples": self.examples,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")

    def as_table(self) -> str:
        head = f"{'lang':<6}{'generated':>10}{'kept':>8}{'kept%':>8}{'dropped':>9}  top drop reasons"
        lines = [head]
        for lang in self.languages():
            gen = self.generated[lang]
            pct = f"{100 * self.kept[lang] / gen:.0f}%" if gen else "-"
            top = ", ".join(f"{r}={n}" for r, n in self.drop_first_reason[lang].most_common(3))
            lines.append(f"{lang:<6}{gen:>10}{self.kept[lang]:>8}{pct:>8}{self.dropped(lang):>9}  {top}")
        t = self.totals()
        lines.append(f"{'TOTAL':<6}{t['generated']:>10}{t['kept']:>8}{'':>8}{t['dropped']:>9}")
        return "\n".join(lines)
