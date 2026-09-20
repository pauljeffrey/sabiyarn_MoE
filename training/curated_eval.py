"""Hand-curated multilingual probe set for sanity-checking the model against
the train/val bins.

`data/curated_eval.jsonl` holds one JSON object per line, `{"lang": "<code>", "text": "<passage>"}`.
It is independent of train.bin/validation.bin, so comparing its loss with the
train/val loss separates "the model is broken" from "the bins are broken":

  curated loss ~= train/val loss  -> bins are consistent with the model (a model problem, or a genuinely hard corpus)
  curated loss << train/val loss  -> the model is fine on clean text; look at how the bins were built/tokenized
  curated loss >> train/val loss  -> the model itself is bad on clean text

Kept torch-free so it can be unit-tested anywhere; the forward passes live in
Trainer._curated_eval (training/new_train.py).
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

LN2 = math.log(2.0)


def load_curated_samples(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("text") or not row.get("lang"):
                raise ValueError(f"{path}:{line_no}: each line needs non-empty 'lang' and 'text'")
            rows.append({"lang": str(row["lang"]), "text": str(row["text"])})
    return rows


def resolve_path(path: str, repo_root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def build_sequence(ids: list[int], eos: int, bos: Optional[int] = None) -> list[int]:
    """Mirror how data/prepare.py lays a document into the bin: leading BOS
    stripped, the text ids, then TWO eos tokens. The sequence is also preceded
    by the previous document's two eos, so the first real token is predicted
    from exactly the context it has in the bin."""
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]
    return [eos, eos, *ids, eos, eos]


class CuratedTotals:
    """Accumulates CE nats / target tokens / decoded bytes per language."""

    def __init__(self) -> None:
        self.data: dict[str, list[float]] = {}

    def add(self, lang: str, ce_mean: float, n_tokens: float, n_bytes: float) -> None:
        cell = self.data.setdefault(lang, [0.0, 0.0, 0.0, 0.0])
        cell[0] += ce_mean * n_tokens
        cell[1] += n_tokens
        cell[2] += n_bytes
        cell[3] += 1

    def summary(self) -> dict:
        def cell_stats(nats: float, toks: float, byts: float, n: float) -> dict:
            return {
                "n": int(n),
                "ce": nats / toks if toks else None,
                "bpb": nats / (LN2 * byts) if byts else None,
            }

        by_lang = {lang: cell_stats(*c) for lang, c in sorted(self.data.items())}
        total = [sum(c[i] for c in self.data.values()) for i in range(4)]
        return {"overall": cell_stats(*total), "by_language": by_lang}
