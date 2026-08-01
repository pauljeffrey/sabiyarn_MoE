"""Summary reporting for a postprocess run: counts per (task, language) cell
plus drop reasons, so distribution balance and failure modes are visible
at a glance instead of buried in logs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class RunStats:
    def __init__(self) -> None:
        # (task, language) -> counts
        self.generated: dict[tuple[str, str], int] = defaultdict(int)
        self.parse_failed: dict[tuple[str, str], int] = defaultdict(int)
        self.validation_failed: dict[tuple[str, str], int] = defaultdict(int)
        self.dedup_dropped: dict[tuple[str, str], int] = defaultdict(int)
        self.kept: dict[tuple[str, str], int] = defaultdict(int)
        self.warnings: list[dict[str, Any]] = []

    def record_warning(self, task: str, language: str, custom_id: str, message: str) -> None:
        self.warnings.append({"task": task, "language": language, "custom_id": custom_id, "message": message})

    def as_table(self) -> str:
        cells = sorted(set(self.generated) | set(self.kept))
        lines = [f"{'task':<20}{'lang':<6}{'generated':>10}{'parse_fail':>11}{'valid_fail':>11}{'dedup_drop':>11}{'kept':>7}"]
        for cell in cells:
            task, lang = cell
            lines.append(
                f"{task:<20}{lang:<6}{self.generated[cell]:>10}{self.parse_failed[cell]:>11}"
                f"{self.validation_failed[cell]:>11}{self.dedup_dropped[cell]:>11}{self.kept[cell]:>7}"
            )
        return "\n".join(lines)

    def totals(self) -> dict[str, int]:
        return {
            "generated": sum(self.generated.values()),
            "parse_failed": sum(self.parse_failed.values()),
            "validation_failed": sum(self.validation_failed.values()),
            "dedup_dropped": sum(self.dedup_dropped.values()),
            "kept": sum(self.kept.values()),
            "warnings": len(self.warnings),
        }

    def write(self, path: Path) -> None:
        data = {
            "totals": self.totals(),
            "cells": [
                {
                    "task": task,
                    "language": lang,
                    "generated": self.generated[(task, lang)],
                    "parse_failed": self.parse_failed[(task, lang)],
                    "validation_failed": self.validation_failed[(task, lang)],
                    "dedup_dropped": self.dedup_dropped[(task, lang)],
                    "kept": self.kept[(task, lang)],
                }
                for task, lang in sorted(set(self.generated) | set(self.kept))
            ],
            "warnings": self.warnings,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
