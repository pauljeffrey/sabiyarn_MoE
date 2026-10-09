"""Which of OUR 23 languages is this text in -- and how sure are we?

Two independent signals, combined:
  * GlotLID top-3 (fastText; labels only, no probabilities in this build)
  * stopword score: share of the text's words that are function words specific to a language (stopwords.json, mined from our
    own corpus by build_stopwords.py -- a word counts only if it is >= 6x more frequent in that language than in the others)

Levels
  certain   GlotLID's top label is a supported language AND the stopwords agree (or the text is long and does not disagree)
  likely    only one signal points at a supported language (e.g. GlotLID says yor but the text is too short for stopwords)
  none      neither; or the text is clearly some other language (Arabic, Amharic, Luganda, ...)

A text written in a language we do not support is `none` even when a supported language is the nearest label: GlotLID always
answers with *something*, so "top-1 is supported" alone is not accepted for short strings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = ("eng", "pcm", "hau", "ibo", "yor", "efi", "ibb", "urh", "twi", "ewe", "fon", "kin", "sna", "zul", "swh", "orm", "wol",
             "xho", "tsn", "nya", "afr", "som", "sot")
SUPPORTED_SET = frozenset(SUPPORTED)

NAME_TO_CODE = {
    "english": "eng", "pidgin": "pcm", "nigerian pidgin": "pcm", "naija": "pcm", "hausa": "hau", "igbo": "ibo", "ibo": "ibo",
    "yoruba": "yor", "efik": "efi", "ibibio": "ibb", "urhobo": "urh", "twi": "twi", "akan": "twi", "asante twi": "twi", "ewe": "ewe",
    "fon": "fon", "kinyarwanda": "kin", "shona": "sna", "zulu": "zul", "isizulu": "zul", "swahili": "swh", "kiswahili": "swh",
    "oromo": "orm", "afaan oromo": "orm", "oromifa": "orm", "wolof": "wol", "xhosa": "xho", "isixhosa": "xho", "tswana": "tsn",
    "setswana": "tsn", "chichewa": "nya", "chewa": "nya", "nyanja": "nya", "afrikaans": "afr", "somali": "som", "sesotho": "sot",
    "sotho": "sot", "southern sotho": "sot",
}
ISO_TO_CODE = {
    "en": "eng", "eng": "eng", "pcm": "pcm", "ha": "hau", "hau": "hau", "ig": "ibo", "ibo": "ibo", "yo": "yor", "yor": "yor",
    "efi": "efi", "ibb": "ibb", "urh": "urh", "tw": "twi", "twi": "twi", "ak": "twi", "aka": "twi", "ee": "ewe", "ewe": "ewe",
    "fon": "fon", "rw": "kin", "kin": "kin", "sn": "sna", "sna": "sna", "zu": "zul", "zul": "zul", "sw": "swh", "swa": "swh",
    "swh": "swh", "om": "orm", "orm": "orm", "gaz": "orm", "wo": "wol", "wol": "wol", "xh": "xho", "xho": "xho", "tn": "tsn",
    "tsn": "tsn", "ny": "nya", "nya": "nya", "af": "afr", "afr": "afr", "so": "som", "som": "som", "st": "sot", "sot": "sot",
}
GLOT_TO_CODE = {"eng_Latn": "eng", "pcm_Latn": "pcm", "wes_Latn": "pcm", "hau_Latn": "hau", "ibo_Latn": "ibo", "yor_Latn": "yor",
                "efi_Latn": "efi", "ibb_Latn": "ibb", "urh_Latn": "urh", "twi_Latn": "twi", "aka_Latn": "twi", "ewe_Latn": "ewe",
                "fon_Latn": "fon", "kin_Latn": "kin", "sna_Latn": "sna", "zul_Latn": "zul", "swh_Latn": "swh", "gaz_Latn": "orm",
                "orm_Latn": "orm", "wol_Latn": "wol", "xho_Latn": "xho", "tsn_Latn": "tsn", "nya_Latn": "nya", "afr_Latn": "afr",
                "som_Latn": "som", "sot_Latn": "sot"}
# near-identical pairs where the wrong sibling is not a real error for our purposes
SIBLINGS = ({"efi", "ibb"}, {"zul", "xho"}, {"tsn", "sot"}, {"swh"}, {"twi"})

_TOKEN = re.compile(r"[^\W\d_]+", re.U)
_STOP: dict[str, frozenset] | None = None
# English function words are removed from every other language's list: the lists were mined against the other African languages
# only, so ordinary English words that sit inside Pidgin, Nigerian news or code-switched text looked "specific" to them.
_EN = frozenset(("the of and to in is that it for was on are as with his they at be this from have or by one had not but what all were when we "
                 "there can an your which their said if do will each about how up out them then she many some so these would other into has more "
                 "her two like him see time could no make than first been its who now people my made over did down only way find use may water long "
                 "little very after words called just where most know you he me us our i am been being does done get got going come came went say says "
                 "tell told give gave take took good new old great high small large next early young important few public bad same able police "
                 "government president state country school over under again further once here why both such own too can should now also because while "
                 "before between through during above below off any another every much more less than").split())


def stopwords() -> dict[str, frozenset]:
    global _STOP
    if _STOP is None:
        p = Path(__file__).with_name("stopwords.json")
        raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        _STOP = {k: (frozenset(v) if k == "eng" else frozenset(w for w in v if len(w) >= 2) - _EN) for k, v in raw.items()}
        _STOP["eng"] = _STOP.get("eng", frozenset()) | _EN
    return _STOP


def code_of(name_or_iso: str | None) -> str | None:
    """'Hausa' / 'ha' / 'hau' / '<hau>' -> 'hau'; unsupported or unknown -> None."""
    if not name_or_iso:
        return None
    s = name_or_iso.strip().strip("<>|").strip().lower()
    s = re.sub(r"\(.*?\)", "", s).strip()
    return NAME_TO_CODE.get(s) or ISO_TO_CODE.get(s)


def stop_scores(text: str) -> tuple[dict[str, float], int]:
    ws = [w.lower() for w in _TOKEN.findall(text[:3000])]
    n = len(ws)
    if not n:
        return {}, 0
    sw = stopwords()
    return {l: sum(1 for w in ws if w in s) / n for l, s in sw.items()}, n


@dataclass
class Detection:
    code: str | None          # best supported language, or None
    level: str                # certain | likely | none
    glot: tuple               # supported-or-None codes of the GlotLID top-3
    stop_best: str | None
    n_words: int
    foreign: bool = False     # a long text that is clearly SOME OTHER language (GlotLID's top label is not one of ours and no stopword evidence for ours)


def _glot_codes(labels: list[str]) -> tuple:
    return tuple(GLOT_TO_CODE.get(l) for l in labels)


def detect_many(texts: list[str]) -> list[Detection]:
    from africomet_translation import lid
    if not texts:
        return []
    raw = lid._model().f.multilinePredict([t.replace("\n", " ")[:2000] for t in texts], 3, 0.0, "replace")
    out = []
    for t, labs in zip(texts, raw):
        g = _glot_codes([l.replace("__label__", "") for l in labs])
        sc, n = stop_scores(t)
        best = max(sc, key=sc.get) if sc else None
        top = sc.get(best, 0.0) if best else 0.0
        second = sorted(sc.values())[-2] if len(sc) > 1 else 0.0
        code, level = None, "none"
        pidgin = sc.get("pcm", 0.0) >= 0.07 and sc.get("pcm", 0.0) >= sc.get("eng", 0.0) * 0.6 and g[0] in ("pcm", "eng")
        if pidgin:
            code, level = "pcm", ("certain" if g[0] == "pcm" or n >= 8 else "likely")
        elif g[0] in SUPPORTED_SET:
            c = g[0]
            agree = best == c and top >= 0.05
            sib = best is not None and any({best, c} <= s for s in SIBLINGS)
            if agree or (n >= 30 and sc.get(c, 0.0) >= 0.03 and best in (c, None)) or (sib and top >= 0.05 and n >= 6):
                code, level = (best if sib and best != c and n >= 12 else c), "certain"
            elif n >= 3 and sc.get(c, 0.0) >= 0.02 and (best in (None, c) or top < 0.12 or sc.get(c, 0.0) > 0):                    # some stopword evidence is required: GlotLID alone also answers 'somali' for Afar
                code, level = c, "likely"
            elif best and best in g and top >= 0.10:
                code, level = best, "likely"
        elif best in SUPPORTED_SET and best in g and top >= 0.12 and n >= 6:
            code, level = best, "likely"
        foreign = g[0] is None and n >= 10 and top < 0.08 and code is None
        out.append(Detection(code, level, g, best, n, foreign))
    return out


def detect(text: str) -> Detection:
    return detect_many([text])[0]


_SENT_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")


def segments(prompt: str, max_segments: int = 24) -> list[str]:
    """A user turn cut into line/sentence pieces, tiny pieces glued to their neighbour so that every piece has enough words to identify."""
    out: list[str] = []
    for piece in _SENT_SPLIT.split(prompt):
        piece = piece.strip()
        if not piece:
            continue
        if out and (len(piece.split()) < 3 or len(out[-1].split()) < 3):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out[:max_segments]


def prompt_languages(prompt: str, hints: tuple | list = (), max_langs: int = 3) -> list[str]:
    """Every supported language in a whole user turn, in the order they first appear, once each.

    With `hints` (languages the source states: instruction language, text language) the answer is exactly those languages, ordered by where
    each is first detected; a hint the detector could not find keeps its place relative to the other one (first hint first). Detected languages
    that are not hints are ignored then: a single mis-labelled sentence must not invent a third language. Without hints only `certain`
    detections count."""
    hs = [h for h in dict.fromkeys(hints) if h in SUPPORTED_SET]
    segs = segments(prompt)
    dets = detect_many([x[:800] for x in segs]) if segs else []
    first: dict[str, int] = {}
    for i, d in enumerate(dets):
        if d.code and d.level != "none" and (d.level == "certain" or d.code in hs):
            first.setdefault(d.code, i)
    if not hs:
        return sorted(first, key=first.get)[:max_langs]
    found = [h for h in hs if h in first]
    missing = [h for h in hs if h not in first]
    order = sorted(found, key=first.get)
    for h in missing:                                         # keep a missing hint on its original side of the ones that were found
        order.insert(0, h) if hs.index(h) < min((hs.index(f) for f in order), default=len(hs)) else order.append(h)
    return order[:max_langs]


def has_foreign_segment(prompt: str) -> bool:
    """True when some stretch of a user turn is clearly a language we do not support (an xP3-style prompt wrapping Basque or French text in
    English template words is identified as English as a whole; its segments are not)."""
    segs = [x for x in segments(prompt) if len(x.split()) >= 10]
    return any(d.foreign for d in detect_many([x[:800] for x in segs])) if segs else False
