"""Benchmark loaders -> small normalised dicts. All splits are the benchmarks' held-out test sets
(FLORES devtest); none of them are used for training here.

Some of these Hub datasets are still script-based, which `datasets` >= 4 refuses to load. For
those we fall back to the Hub's auto-converted parquet files (`refs/convert/parquet`).
"""

from __future__ import annotations

import ast
import json
import random
from typing import Optional

from eval_suite.langs import Lang


class DataUnavailable(RuntimeError):
    """The benchmark does not cover this language, or could not be downloaded."""


SIB_LABELS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
NEWS_LABELS = ["business", "entertainment", "health", "politics", "religion", "sports", "technology"]
SENTI_LABELS = ["positive", "neutral", "negative"]
NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]


def _load(repo: str, name: Optional[str], split: str, token: Optional[str] = None):
    from datasets import load_dataset

    try:
        return load_dataset(repo, name, split=split, token=token) if name else load_dataset(repo, split=split, token=token)
    except RuntimeError as exc:
        if "script" not in str(exc).lower():
            raise
    url = f"hf://datasets/{repo}@refs%2Fconvert%2Fparquet/{name}/{split}/0000.parquet"
    return load_dataset("parquet", data_files={split: url}, split=split)


def subsample(items: list, limit: Optional[int], seed: int = 0) -> list:
    """Deterministic, order-preserving subsample (test sets are often ordered by domain)."""
    if not limit or len(items) <= limit:
        return items
    keep = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in keep]


# ------------------------------------------------------------------ translation

def _flores(code: str, split: str, token: Optional[str]) -> list[str]:
    errors = []
    for repo, column in (("openlanguagedata/flores_plus", "text"), ("facebook/flores", "sentence")):
        try:
            ds = _load(repo, code, split, token)
            return list(ds[column])
        except Exception as exc:  # gated, missing config, offline ...
            errors.append(f"{repo}: {type(exc).__name__}: {str(exc)[:100]}")
    raise DataUnavailable(
        "FLORES is gated: accept the terms on https://huggingface.co/datasets/openlanguagedata/flores_plus "
        "with the account behind HF_API_KEY/HF_TOKEN. " + " | ".join(errors)
    )


def load_translation(lang: Lang, limit: Optional[int], source: str = "auto", token: Optional[str] = None,
                     custom_path: Optional[str] = None) -> tuple[list[dict], str]:
    """Returns ([{"eng", "xx"}...], source name). English-pivot pairs only."""
    if custom_path:
        rows = [json.loads(l) for l in open(custom_path, encoding="utf-8") if l.strip()]
        rows = [{"eng": r["eng"], "xx": r["xx"]} for r in rows if r.get("lang") == lang.code]
        if rows:
            return subsample(rows, limit), "custom"
    flores_error = None
    if source in ("auto", "flores") and lang.flores:
        try:
            eng, xx = _flores("eng_Latn", "devtest", token), _flores(lang.flores, "devtest", token)
            return subsample([{"eng": e, "xx": x} for e, x in zip(eng, xx)], limit), "flores"
        except DataUnavailable as exc:
            if source == "flores":
                raise
            flores_error = exc
    if source in ("auto", "mafand") and lang.mafand:
        ds = _load("masakhane/mafand", f"en-{lang.mafand}", "test", token)
        rows = []
        for r in ds:
            tr = r["translation"]
            other = next(k for k in tr if k != "en")
            rows.append({"eng": tr["en"], "xx": tr[other]})
        return subsample(rows, limit), "mafand"
    if flores_error is not None:
        raise DataUnavailable(f"{lang.name} is only in FLORES, which could not be loaded: {flores_error}")
    raise DataUnavailable(f"no translation benchmark for {lang.name} (source={source}); use --custom-translation")


# ------------------------------------------------------------------ classification

def load_topic(lang: Lang, source: str, limit: Optional[int], token=None) -> tuple[list[dict], list[str]]:
    if source == "sib200":
        if not lang.sib:
            raise DataUnavailable(f"SIB-200 has no {lang.name}")
        ds = _load("Davlan/sib200", lang.sib, "test", token)
        rows = [{"text": r["text"], "label": r["category"]} for r in ds]
        return subsample(rows, limit), SIB_LABELS
    if source == "masakhanews":
        if not lang.news:
            raise DataUnavailable(f"MasakhaNEWS has no {lang.name}")
        ds = _load("masakhane/masakhanews", lang.news, "test", token)
        rows = [{"text": " ".join((r["headline"] + " . " + r["text"]).split()[:128]), "label": r["category"]} for r in ds]
        return subsample(rows, limit), NEWS_LABELS
    raise ValueError(f"unknown topic source {source!r}")


def load_sentiment(lang: Lang, limit: Optional[int], token=None) -> tuple[list[dict], list[str]]:
    if not lang.senti:
        raise DataUnavailable(f"AfriSenti has no {lang.name}")
    ds = _load("shmuhammad/AfriSenti-twitter-sentiment", lang.senti, "test", token)
    names = getattr(ds.features["label"], "names", None) or SENTI_LABELS
    rows = [{"text": r["tweet"], "label": names[r["label"]] if isinstance(r["label"], int) else str(r["label"])} for r in ds]
    return subsample(rows, limit), SENTI_LABELS


def load_ner(lang: Lang, limit: Optional[int], token=None) -> tuple[list[dict], list[str]]:
    if not lang.ner:
        raise DataUnavailable(f"MasakhaNER 2.0 has no {lang.name}")
    ds = _load("masakhane/masakhaner2", lang.ner, "test", token)
    names = getattr(ds.features["ner_tags"].feature, "names", None) or NER_TAGS
    rows = [{"tokens": list(r["tokens"]), "tags": [names[t] for t in r["ner_tags"]]} for r in ds]
    return subsample(rows, limit), list(names)


# ------------------------------------------------------------------ MMLU

def _mc_row(r: dict) -> dict:
    choices = r["choices"]
    if isinstance(choices, str):  # AfriMMLU stores the list as its repr
        choices = ast.literal_eval(choices)
    ans = r["answer"]
    answer = "ABCD".index(ans) if isinstance(ans, str) else int(ans)
    return {"subject": r["subject"], "question": r["question"], "choices": list(choices), "answer": answer}


def load_mmlu(lang: Lang, limit: Optional[int], token=None) -> tuple[list[dict], dict[str, list[dict]], str]:
    """(test rows, few-shot pool by subject, benchmark name). English uses the original MMLU."""
    if lang.code == "eng":
        test = [_mc_row(r) for r in _load("cais/mmlu", "all", "test", token)]
        dev = [_mc_row(r) for r in _load("cais/mmlu", "all", "dev", token)]
        name = "mmlu"
    else:
        if not lang.mmlu:
            raise DataUnavailable(f"AfriMMLU has no {lang.name}")
        test = [_mc_row(r) for r in _load("masakhane/afrimmlu", lang.mmlu, "test", token)]
        dev = [_mc_row(r) for r in _load("masakhane/afrimmlu", lang.mmlu, "dev", token)]
        name = "afrimmlu"
    pool: dict[str, list[dict]] = {}
    for r in dev:
        pool.setdefault(r["subject"], []).append(r)
    return subsample(test, limit), pool, name
