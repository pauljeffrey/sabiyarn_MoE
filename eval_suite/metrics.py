"""Metrics, all dependency-light. sacrebleu is only needed for translation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def chrf_bleu(hyps: Sequence[str], refs: Sequence[str]) -> dict:
    """Corpus chrF++ (word_order=2), chrF and BLEU (sacrebleu, 13a). chrF++ is the headline number:
    BLEU is well known to be unreliable for morphologically rich / tonal languages, so it is
    reported for comparability with older papers, not as the primary metric."""
    import sacrebleu

    refs_l = [list(refs)]
    return {
        "chrf++": sacrebleu.corpus_chrf(list(hyps), refs_l, word_order=2).score,
        "chrf": sacrebleu.corpus_chrf(list(hyps), refs_l).score,
        "bleu": sacrebleu.corpus_bleu(list(hyps), refs_l).score,
    }


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def classification_metrics(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict:
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    majority = Counter(gold).most_common(1)[0][1] / n if n else 0.0
    lo, hi = wilson_interval(correct, n)
    return {
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "majority_baseline": majority,
        "acc_ci95": [lo, hi],
        "n": n,
    }


def bio_entities(tags: Sequence[str]) -> set[tuple[str, int, int]]:
    """(type, start, end) spans from a BIO tag sequence; a stray I-X starts a span (seqeval-style)."""
    spans, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, etype = tag.partition("-")
        continues = prefix == "I" and kind == etype
        if start is not None and not continues:
            spans.add((kind, start, i))
            start, kind = None, None
        if prefix in ("B", "I") and start is None:
            start, kind = i, etype
    return spans


def entity_f1(gold_seqs: Sequence[Sequence[str]], pred_seqs: Sequence[Sequence[str]]) -> dict:
    """Micro-averaged ENTITY-level precision/recall/F1 (a span counts only if type and boundaries
    both match), plus token accuracy. This is what NER papers report; per-sentence label-presence
    F1 (what eval/eval.py computed) is far more lenient and not comparable."""
    tp = fp = fn = tok_ok = tok_n = 0
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = bio_entities(g), bio_entities(p)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        tok_ok += sum(a == b for a, b in zip(g, p))
        tok_n += len(g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "f1": f1, "precision": prec, "recall": rec,
        "token_accuracy": tok_ok / tok_n if tok_n else 0.0,
        "gold_entities": tp + fn, "n": len(gold_seqs),
    }
