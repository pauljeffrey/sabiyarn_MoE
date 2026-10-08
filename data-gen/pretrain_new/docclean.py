"""Document-level cleaning, quality gating and low-memory near-duplicate removal for monolingual pretrain text."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁠-⁯﻿�]")
_SPACES = re.compile(r"[ \t\r\f\v ]+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,}\d(?!\d)")
_BOILER = re.compile(
    r"(cookie|all rights reserved|©|copyright|privacy policy|terms (and|&) conditions|sign (in|up)|log ?in|subscribe|"
    r"share (this|on)|follow us|read more|click here|leave a (reply|comment)|related (posts|articles)|"
    r"powered by|skip to (main )?content|back to top|newsletter|whatsapp|facebook|twitter|instagram|youtube)", re.I)
_SENT_END = re.compile(r"[.!?…:;»”\"')]\s*$")

# per-language minimums: the big languages can afford strictness, the small ones cannot
SMALL = {"twi", "ewe", "fon", "wol", "tsn", "efi", "ibb", "urh", "pcm", "sot", "nya", "orm"}


def min_sizes(lang: str) -> tuple[int, int]:
    """(min_chars, min_words) of a kept document."""
    return (150, 25) if lang in SMALL else (250, 40)


def _letters(s: str) -> int:
    return sum(1 for c in s if c.isalpha() or unicodedata.category(c) == "Mn")


def clean_document(text: str, lang: str) -> str | None:
    """Return the cleaned text, or None when the document is not worth keeping."""
    if not text:
        return None
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    seen: set[str] = set()
    keep: list[str] = []
    for raw in text.split("\n"):
        line = _SPACES.sub(" ", raw).strip()
        if not line:
            if keep and keep[-1] != "":
                keep.append("")                                   # one blank line between paragraphs
            continue
        line = _URL.sub("", line).strip()
        if _EMAIL.search(line) or _PHONE.search(line):
            continue
        words = line.split()
        if len(words) < 3 and not _SENT_END.search(line):
            continue                                                # menu / heading debris
        if _BOILER.search(line) and len(words) < 14:
            continue
        key = line.lower()
        if len(words) >= 4 and key in seen:
            continue
        seen.add(key)
        keep.append(line)
    while keep and keep[-1] == "":
        keep.pop()
    out = "\n".join(keep).strip()
    min_chars, min_words = min_sizes(lang)
    if len(out) < min_chars:
        return None
    words = out.split()
    if len(words) < min_words:
        return None
    nonspace = len(out) - out.count(" ") - out.count("\n")
    if nonspace <= 0 or _letters(out) / nonspace < 0.6:
        return None
    if sum(c.isdigit() for c in out) / nonspace > 0.25:
        return None
    lw = [w.lower() for w in words]
    if len(set(lw)) / len(lw) < 0.12 and len(lw) > 60:
        return None
    top = max((lw.count(w) for w in set(lw[:400])), default=0) / min(len(lw), 400)
    if top > 0.3:
        return None
    return out


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


class SeenTable:
    """Fixed-memory 'have I seen this 64-bit hash' table: direct-mapped, so a collision can only cause a missed
    duplicate, never a wrongly dropped document. 2**25 slots = 256 MB."""

    def __init__(self, bits: int = 25):
        self.mask = (1 << bits) - 1
        self.tab = np.zeros(1 << bits, dtype=np.uint64)

    def check_and_add(self, hashes: np.ndarray) -> np.ndarray:
        slots = (hashes & np.uint64(self.mask)).astype(np.int64)
        seen = self.tab[slots] == hashes
        self.tab[slots] = hashes
        return seen


def paragraph_hashes(text: str, limit: int = 12) -> list[int]:
    ps = [re.sub(r"\W+", " ", p.lower()).strip() for p in text.split("\n")]
    ps = [p for p in ps if len(p) >= 50]
    if len(ps) > limit:                       # an even spread keeps the signature stable for very long documents
        step = len(ps) / limit
        ps = [ps[int(i * step)] for i in range(limit)]
    return [_h64(p) | 1 for p in ps]


class Deduper:
    """Exact-document and near-duplicate (shared paragraphs) removal, constant memory."""

    def __init__(self, bits: int = 25, near_share: float = 0.5):
        self.docs = SeenTable(bits)
        self.paras = SeenTable(bits)
        self.near_share = near_share
        self.dropped_exact = self.dropped_near = 0

    def save(self, path) -> None:
        np.savez(path, docs=self.docs.tab, paras=self.paras.tab,
                 counts=np.array([self.dropped_exact, self.dropped_near], dtype=np.int64))

    @classmethod
    def load(cls, path, bits: int = 25, near_share: float = 0.5) -> "Deduper":
        d = cls(bits, near_share)
        z = np.load(path)
        d.docs.tab, d.paras.tab = z["docs"], z["paras"]
        d.dropped_exact, d.dropped_near = (int(x) for x in z["counts"])
        return d

    def is_duplicate(self, text: str) -> bool:
        d = np.array([_h64(re.sub(r"\W+", " ", text.lower()).strip()) | 1], dtype=np.uint64)
        if self.docs.check_and_add(d)[0]:
            self.dropped_exact += 1
            return True
        ph = paragraph_hashes(text)
        if not ph:
            return False
        seen = self.paras.check_and_add(np.array(ph, dtype=np.uint64))
        if len(ph) >= 2 and seen.mean() >= self.near_share:
            self.dropped_near += 1
            return True
        return False
