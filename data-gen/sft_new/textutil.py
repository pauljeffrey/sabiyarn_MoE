"""Small text hygiene helpers shared by every converter."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"[ \t ​]+")
_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CALC = re.compile(r"<{1,2}[^<>a-zA-Z]*=[^<>]*>{1,2}'?")          # GSM8K calculator annotations: <<48/2=24>> and their broken variants (<48/2=24>>, <<48/2=24>)
_PLACEHOLDER = {"none", "nan", "null", "#value!", "n/a", ""}


def clean(s) -> str:
    """NFC, no control characters, single spaces, at most one blank line, stripped. Non-strings become ''."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", _CTRL.sub("", s))
    s = _WS.sub(" ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    return _NL.sub("\n\n", s).strip()


def blank(s) -> str:
    """None, NaN, non-strings, '' and export placeholders ('None', 'null', 'nan', 'N/A', '#VALUE!') all mean "this field is empty" -> ''.
    Real text is returned with only its outer whitespace stripped (inner whitespace matters in code and poetry). Use it for TEXT fields only: a
    label column may legitimately hold the class 'None'."""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    return "" if t.lower() in _PLACEHOLDER else t


def num(x, default: float = 0.0) -> float:
    """float(x) for scores/ratings that may be None, '', 'nan' or a real NaN -> default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def real(s) -> str:
    """clean(), but sentinel values from spreadsheet exports ('None', '#VALUE!') count as empty."""
    c = clean(s)
    return "" if c.lower() in _PLACEHOLDER else c


def strip_calc(s: str) -> str:
    return _CALC.sub("", s)


def rep_ratio(text: str, n: int = 4) -> float:
    """Share of word n-grams that repeat an earlier one (0 = no repetition). High values mean a degenerate machine translation."""
    w = text.lower().split()
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def has_loop(text: str, min_run: int = 4) -> bool:
    """True when some 1-4 word phrase repeats `min_run`+ times in a row ("(a.s) (a.s) (a.s) (a.s)", "jami'an jami'an jami'an ...")."""
    w = text.split()
    n = len(w)
    for k in (1, 2, 3, 4):
        run = 1
        for i in range(k, n):
            if w[i] == w[i - k]:
                run += 1
                if run >= min_run * k and run // k >= min_run:
                    return True
            else:
                run = 1
    return False


def degenerate(text: str) -> bool:
    return rep_ratio(text) > 0.4 or has_loop(text)


_CAP = re.compile(r"\b[A-Z][\w'’-]{2,}\b")
_NUM = re.compile(r"\d[\d.,]*")


def anchors(text: str) -> set[str]:
    """Numbers and capitalised words that are NOT the first word of a sentence: what survives translation (names, figures)."""
    out = {m.group(0).lower().strip(".,") for m in _NUM.finditer(text)}
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        for m in _CAP.finditer(sent):
            if m.start() > 0:
                out.add(m.group(0).lower().replace("’", "'"))
    return out


def aligned(a: str, b: str, min_anchors: int = 3, min_overlap: float = 0.25) -> bool:
    """Do two texts that claim to be translations share their names and numbers? Texts with fewer than `min_anchors` anchors pass."""
    x, y = anchors(a), anchors(b)
    if len(x) < min_anchors or len(y) < min_anchors:
        return True
    near = sum(1 for t in x if t in y or any(len(t) > 4 and (t[:5] in u) for u in y))
    return near / min(len(x), len(y)) >= min_overlap


def recase(text: str) -> str:
    """Capitalise sentence starts in text that was lower-cased by a pipeline (only when most sentences start lower-case)."""
    sents = re.split(r"(?<=[.!?])(\s+)", text)
    parts = [s for s in sents if s.strip()]
    starts = [p.lstrip()[:1] for p in parts]
    if not starts or sum(1 for c in starts if c.islower()) / len(starts) < 0.8:
        return text
    return "".join(s[:1].upper() + s[1:] if s.strip() else s for s in sents)


def words(s: str) -> int:
    return len(s.split())


def length_ratio_ok(a: str, b: str, lo: float = 0.3, hi: float = 3.5) -> bool:
    return lo <= (len(b) + 1) / (len(a) + 1) <= hi


def first_sentence(s: str, limit: int = 220) -> str:
    m = re.search(r"(.+?[.!?])(\s|$)", s.strip(), re.S)
    out = (m.group(1) if m else s.strip())
    return out[:limit].strip()


def chunks(text: str, max_chars: int = 900, min_chars: int = 120) -> list[str]:
    """Cut a long document into pieces of whole sentences/paragraphs of roughly max_chars (for RL prompts and long translation)."""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        for sent in re.split(r"(?<=[.!?])\s+", p):
            if cur and len(cur) + len(sent) + 1 > max_chars:
                out.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    if len(out) > 1 and len(out[-1]) < min_chars:
        out[-2] = f"{out[-2]} {out[-1]}"
        out.pop()
    return out
