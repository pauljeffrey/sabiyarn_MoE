"""Sentence splitting, sentence quality filters and window partitioning for the AfriCOMET translation prompts."""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Iterable, Iterator, Sequence

MAX_SENTENCES_PER_WINDOW = 5

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "inc", "ltd", "co", "corp", "no", "nos",
    "fig", "figs", "vol", "ca", "approx", "dept", "est", "gen", "gov", "sen", "rep", "rev", "col", "lt", "sgt",
    "capt", "cmdr", "mme", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "e.g", "i.e", "cf", "al", "ed", "eds", "pp", "p", "ch", "sec", "u.s", "u.k", "a.m", "p.m",
}
_CLOSERS = "\"'”’)]»"
_URLISH = re.compile(r"https?://|www\.|@\w+\.\w+|\.com\b|\.org\b", re.IGNORECASE)
_MARKUP = re.compile(r"[<>{}|\[\]\\]|={2,}|\bedit\b\]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏  ﻿�]")
_SPACES = re.compile(r"[ \t\r\f\v ]+")


def normalize(text: str) -> str:
    """NFC, control characters removed, runs of whitespace collapsed to one space."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    return _SPACES.sub(" ", text.replace("\n", " ")).strip()


def dedupe_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", text).casefold())


def _is_abbreviation(token: str) -> bool:
    core = token.rstrip(".!?" + _CLOSERS).lstrip("(\"'“‘[")
    low = core.lower().rstrip(".")
    if low in _ABBREVIATIONS:
        return True
    if re.fullmatch(r"[A-Z]", core):                       # an initial: "J."
        return True
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}[A-Za-z]?", token.rstrip(_CLOSERS)):   # "U.S." "e.g."
        return True
    if re.fullmatch(r"\d{1,3}", core) and token.endswith("."):   # "3." in a numbered reference
        return True
    return False


def _starts_sentence(token: str) -> bool:
    for ch in token:
        if ch in "\"'“‘([«¿¡":
            continue
        return ch.isupper() or ch.isdigit()
    return False


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by an upper-case (or digit) start, skipping abbreviations and initials.

    Deliberately simple and language-agnostic: every language in this set is Latin-script and capitalises
    sentence starts, and no sentence tokeniser model exists for most of them.
    """
    tokens = normalize(text).split(" ")
    sentences: list[str] = []
    current: list[str] = []
    for i, tok in enumerate(tokens):
        current.append(tok)
        stripped = tok.rstrip(_CLOSERS)
        ends = stripped.endswith((".", "!", "?", "…"))
        if not ends or i + 1 >= len(tokens):
            continue
        if stripped.endswith(".") and _is_abbreviation(tok):
            continue
        if _starts_sentence(tokens[i + 1]):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return [s for s in sentences if s]


def good_sentence(
    s: str, *, min_words: int = 3, max_words: int = 90, require_terminal: bool = False,
) -> bool:
    """Cheap hygiene filter: drops markup remnants, URLs, headings, tables, mostly-numeric and all-caps lines."""
    words = s.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if _URLISH.search(s) or _MARKUP.search(s):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 8:
        return False
    if sum(c.isdigit() for c in s) / max(1, len(s)) > 0.25:
        return False
    if sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False
    if require_terminal and not s.rstrip(_CLOSERS).endswith((".", "!", "?", "…")):
        return False
    if not _starts_sentence(s) and require_terminal:
        return False
    return True


def partition_windows(sentences: Sequence[str], rng: random.Random) -> list[list[str]]:
    """Cut one document into consecutive, non-overlapping windows of 1..5 sentences, each size uniform."""
    out, i, n = [], 0, len(sentences)
    while i < n:
        k = min(rng.randint(1, MAX_SENTENCES_PER_WINDOW), n - i)
        out.append(list(sentences[i:i + k]))
        i += k
    return out


def interleave_windows(per_doc: Sequence[Sequence[list[str]]], rng: random.Random) -> Iterator[list[str]]:
    """Round-robin over documents (shuffled): every document yields one window before any yields a second,
    so a small quota is spread across as many documents as possible instead of exhausting a few."""
    order = list(range(len(per_doc)))
    rng.shuffle(order)
    depth = 0
    while True:
        emitted = False
        for d in order:
            if depth < len(per_doc[d]):
                emitted = True
                yield per_doc[d][depth]
        if not emitted:
            return
        depth += 1


def allocate(total: int, weights: dict[str, float], caps: dict[str, int]) -> dict[str, int]:
    """Weighted split of `total` across sources, capped by what each has; surplus from a capped source is
    redistributed over the rest (water-filling). Returns {source: count}; sum < total only if every source is capped."""
    alloc = {k: 0 for k in weights}
    active = {k for k in weights if caps.get(k, 0) > 0}
    remaining = total
    while remaining > 0 and active:
        wsum = sum(weights[k] for k in active)
        capped = [k for k in active if caps[k] - alloc[k] <= remaining * weights[k] / wsum]
        if not capped:
            shares = {k: remaining * weights[k] / wsum for k in active}
            base = {k: int(shares[k]) for k in active}
            leftover = remaining - sum(base.values())
            for k in sorted(active, key=lambda k: shares[k] - base[k], reverse=True)[:leftover]:
                base[k] += 1
            for k in active:
                alloc[k] += base[k]
            return alloc
        for k in capped:
            take = caps[k] - alloc[k]
            alloc[k] += take
            remaining -= take
            active.discard(k)
    return alloc


def balance_by_k(windows: Sequence, n: int) -> list:
    """Pick `n` windows so the 1..5 sentence-count distribution is as close to uniform as the pool allows
    (short-article sources simply have few long windows; the rest is filled from the sizes that exist)."""
    buckets: dict[int, list] = {}
    for w in windows:
        buckets.setdefault(w.n_sentences, []).append(w)
    ks = sorted(buckets)
    got = allocate(n, {k: 1.0 for k in ks}, {k: len(buckets[k]) for k in ks})
    return [w for k in ks for w in buckets[k][:got[k]]]


def flatten(docs: Iterable[Sequence[str]]) -> list[str]:
    return [s for d in docs for s in d]
