"""Local (no-API) quality filters for pretrain / SFT / DPO records.

Every check is a cheap heuristic. They exist to remove the *gross* failure
modes of LLM synthetic data -- degenerate repetition, English fallback,
assistant boilerplate, placeholders, refusals, wrong script -- not to judge
linguistic quality; only a native speaker (or a strong judge model, see
`pipeline/judge.py`) can do that.

Each validator returns `(reasons, warnings)`:
  reasons   -> hard failures; the record is dropped and the reason counted.
  warnings  -> soft signals (e.g. no distinctive diacritics found); counted
               in the report, never a reason to drop by themselves.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Mapping, Optional

from quality.validators import language_sanity_warnings

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Combining marks (tone/underdot) must stay inside the word: Yoruba "ẹ̀" is
# decomposed into base + combining characters that `\w` alone would split on.
_TOKEN_RE = re.compile(r"[\w\u0300-\u036f]+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text).lower())


# ---------------------------------------------------------------------------
# English-leak detection
# ---------------------------------------------------------------------------

# Function words distinctive enough of English to count as "leak" evidence.
# Words of <= 2 letters are deliberately excluded (they collide with real
# words in Twi, Hausa, Yoruba, Igbo, ... e.g. "no", "me", "a", "in", "to").
ENGLISH_STOPWORDS: frozenset[str] = frozenset("""
the and are was were that this with for from have has had you your they their them which would could should there what when will been
not but all our his her she how who why its also than then into about because these those some more other just very only where while being
such any may might must each most over after before between through during without within can does did doing our out
""".split())

# Stricter list for Nigerian Pidgin, which legitimately shares words like
# "for", "you", "this", "about", "but" with English. These are function words
# Pidgin replaces with its own ("dey", "na", "wey", "don", "dem").
ENGLISH_STRICT_STOPWORDS: frozenset[str] = frozenset("""
the of and is are was were that which have has been would could their there these those with from because
""".split())

STRICT_ENGLISH_LANGUAGES: frozenset[str] = frozenset({"pcm"})


def english_leak_ratio(text: str, language: str = "", *, min_tokens: int = 8) -> Optional[float]:
    """Share of tokens that are English function words; None if too short to judge."""
    toks = tokens(text)
    if len(toks) < min_tokens:
        return None
    stop = ENGLISH_STRICT_STOPWORDS if language in STRICT_ENGLISH_LANGUAGES else ENGLISH_STOPWORDS
    return sum(1 for t in toks if t in stop) / len(toks)


def english_threshold(q: Mapping[str, Any], language: str) -> float:
    return float(q.get("max_english_ratio_by_language", {}).get(language, q["max_english_ratio"]))


# ---------------------------------------------------------------------------
# Script sanity
# ---------------------------------------------------------------------------


def _is_latin_like(ch: str) -> bool:
    o = ord(ch)
    return (
        o <= 0x24F  # Basic Latin .. Latin Extended-B (covers ɓ ƙ ɗ ŋ etc.)
        or 0x250 <= o <= 0x2AF  # IPA extensions (ɔ ɛ ɖ ɣ ʋ)
        or 0x1E00 <= o <= 0x1EFF  # Latin Extended Additional (ẹ ọ ị ụ ṣ)
        or 0x2C60 <= o <= 0x2C7F
        or 0xA720 <= o <= 0xA7FF
        or 0x300 <= o <= 0x36F  # combining marks
    )


def non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if not _is_latin_like(c)) / len(letters)


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------


def repetition_metrics(text: str, *, ngram: int = 4) -> dict[str, float]:
    toks = tokens(text)
    n = len(toks)
    out = {"repeated_ngram_ratio": 0.0, "repeated_line_ratio": 0.0, "top_token_share": 0.0}
    if n >= ngram + 8:
        grams = [tuple(toks[i : i + ngram]) for i in range(n - ngram + 1)]
        out["repeated_ngram_ratio"] = 1.0 - len(set(grams)) / len(grams)
    if n >= 30:
        out["top_token_share"] = Counter(toks).most_common(1)[0][1] / n

    lines = [normalize(x).lower() for x in text.split("\n") if x.strip()]
    sents = [normalize(x).lower() for x in _SENT_SPLIT_RE.split(text) if len(x.strip()) > 12]
    ratios = []
    for units in (lines, sents):
        if len(units) >= 3:
            ratios.append(1.0 - len(set(units)) / len(units))
    out["repeated_line_ratio"] = max(ratios) if ratios else 0.0
    return out


# ---------------------------------------------------------------------------
# Meta text / placeholders / refusals
# ---------------------------------------------------------------------------

_PLACEHOLDER_RES = [
    re.compile(r"\[\s*\.\.\.\s*\]"),
    re.compile(r"\[\s*…\s*\]"),
    re.compile(r"\[[A-Za-z][^\]\n]{0,40}\]"),  # [name], [insert town], [Your Name]
    re.compile(r"\{\{[^}\n]*\}\}"),
    re.compile(r"<\s*(?:placeholder|insert|name|your)[^>\n]*>", re.I),
    re.compile(r"lorem ipsum", re.I),
    re.compile(r"\b(?:TODO|TBD|XXX+)\b"),
    re.compile(r"\binsert [a-z ]{1,25} here\b", re.I),
]
_FENCE_RE = re.compile(r"```")
_META_LEAD_RE = re.compile(
    r"^\s*(?:here is|here's|here are|sure[,!.]|certainly[,!.]|of course[,!.]|okay[,!.]|translation\s*:|translated text\s*:|"
    r"the translation is|response\s*:|output\s*:|below is|i hope this helps)",
    re.I,
)
_AI_DISCLAIMER_RE = re.compile(
    r"\bas an ai\b|\bas a (?:large )?language model\b|\blanguage model\b|\bopenai\b|\bchatgpt\b|\bi(?: am|'m) an? (?:ai|artificial)\b|\bgpt-?4",
    re.I,
)
_REFUSAL_RE = re.compile(
    r"\bi (?:cannot|can't|can’t|am unable to|'m unable to|am not able to) (?:assist|help|fulfil|fulfill|comply|provide|do that|answer)|"
    r"\bi(?:'m| am) sorry,? but\b|\bi apologi[sz]e,? but\b|\bi won't be able to\b",
    re.I,
)
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")
_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")


def meta_text_issues(text: str, *, allow_refusal: bool = False) -> list[str]:
    issues: list[str] = []
    if any(r.search(text) for r in _PLACEHOLDER_RES):
        issues.append("placeholder")
    if _FENCE_RE.search(text):
        issues.append("markdown_fence")
    if _META_LEAD_RE.search(text) or _AI_DISCLAIMER_RE.search(text):
        issues.append("meta_text")
    if not allow_refusal and _REFUSAL_RE.search(text):
        issues.append("refusal")
    return issues


def has_markdown_heading(text: str) -> bool:
    return bool(_HEADING_RE.search(text))


# ---------------------------------------------------------------------------
# Per-field checks
# ---------------------------------------------------------------------------


def check_text_field(
    text: str,
    language: str,
    field: str,
    q: Mapping[str, Any],
    *,
    is_english: bool = False,
    allow_refusal: bool = False,
    check_leak: bool = True,
) -> tuple[list[str], list[str]]:
    """Content checks shared by every text field of every kind."""
    reasons: list[str] = []
    warnings: list[str] = []

    if "\ufffd" in text:
        reasons.append(f"replacement_char:{field}")
    if non_latin_ratio(text) > q["max_non_latin_ratio"]:
        reasons.append(f"wrong_script:{field}")

    if q.get("drop_meta_text", True):
        for issue in meta_text_issues(text, allow_refusal=allow_refusal):
            reasons.append(f"{issue}:{field}")

    rep = repetition_metrics(text)
    if rep["repeated_ngram_ratio"] > q["max_repetition_ratio"]:
        reasons.append(f"repetition_ngram:{field}")
    if rep["repeated_line_ratio"] > q["max_repeated_line_ratio"]:
        reasons.append(f"repetition_lines:{field}")
    if rep["top_token_share"] > q["max_top_token_share"]:
        reasons.append(f"degenerate_token:{field}")

    if check_leak and not is_english:
        ratio = english_leak_ratio(text, language, min_tokens=int(q["min_english_check_tokens"]))
        if ratio is not None and ratio > english_threshold(q, language):
            reasons.append(f"english_leak:{field}")

    if not is_english:
        for w in language_sanity_warnings(text, language):
            if w.startswith("no distinctive"):
                warnings.append(f"no_distinctive_chars:{field}")
            elif w.startswith("empty"):
                reasons.append(f"empty_field:{field}")
    if _BOLD_RE.search(text):
        warnings.append(f"markdown_bold:{field}")
    return reasons, warnings


def _len_bounds(reasons: list[str], text: str, field: str, lo: int, hi: int) -> None:
    n = len(text.strip())
    if n < lo:
        reasons.append(f"too_short:{field}")
    elif n > hi:
        reasons.append(f"too_long:{field}")


# ---------------------------------------------------------------------------
# Kind validators
# ---------------------------------------------------------------------------


def validate_pretrain(doc: Mapping[str, Any], context: Mapping[str, Any], language: str, q: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    reasons: list[str] = []
    warnings: list[str] = []

    if not text:
        return ["empty_field:text"], warnings
    n_words = len(tokens(text))
    lo, hi = context.get("length_words", [0, 10**9])
    min_words = max(int(q["min_words"]), int(lo * float(q["bucket_min_ratio"])))
    max_words = min(int(q["max_words"]), int(hi * float(q["bucket_max_ratio"]))) if hi else int(q["max_words"])
    if n_words < min_words:
        reasons.append("too_short:text")
    elif n_words > max(max_words, min_words):
        reasons.append("too_long:text")

    if not (int(q["min_title_chars"]) <= len(title) <= int(q["max_title_chars"])):
        reasons.append("bad_title")
    if q.get("drop_on_self_check_false", True) and not doc.get("language_self_check", True):
        reasons.append("self_check_failed")
    if q.get("drop_markdown_headings", True) and has_markdown_heading(text):
        reasons.append("markdown_heading:text")

    r, w = check_text_field(text, language, "text", q)
    reasons += r
    warnings += w
    tr, tw = check_text_field(title, language, "title", q, check_leak=False)
    reasons += [x for x in tr if not x.startswith(("repetition", "degenerate"))]
    return reasons, warnings


def _shared_sft_fields(rec: Mapping[str, Any], context: Mapping[str, Any], language: str, q: Mapping[str, Any]) -> tuple[list[str], list[str], set[str], bool]:
    """Checks on `instruction` and `input`; also returns (english_fields, allow_refusal) for the caller."""
    reasons: list[str] = []
    warnings: list[str] = []
    english = set(context.get("english_fields", []))
    task = context.get("attributes", {}).get("task", "")
    allow_refusal = task == "safe_decline"

    instruction = str(rec.get("instruction", "")).strip()
    inp = str(rec.get("input", "")).strip()

    if not instruction:
        reasons.append("empty_field:instruction")
    else:
        _len_bounds(reasons, instruction, "instruction", int(q["min_instruction_chars"]), int(q["max_instruction_chars"]))
        r, w = check_text_field(instruction, language, "instruction", q, allow_refusal=True)
        reasons += r
        warnings += w

    if context.get("input_mode") == "required" and not inp:
        reasons.append("missing_input")
    if inp:
        if len(inp) > int(q["max_input_chars"]):
            reasons.append("too_long:input")
        r, w = check_text_field(inp, language, "input", q, is_english="input" in english, allow_refusal=True)
        reasons += r
        warnings += w
    return reasons, warnings, english, allow_refusal


def validate_sft(rec: Mapping[str, Any], context: Mapping[str, Any], language: str, q: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    reasons, warnings, english, allow_refusal = _shared_sft_fields(rec, context, language, q)
    response = str(rec.get("response", "")).strip()
    if not response:
        reasons.append("empty_field:response")
    else:
        _len_bounds(reasons, response, "response", int(q["min_response_chars"]), int(q["max_response_chars"]))
        r, w = check_text_field(response, language, "response", q, is_english="response" in english, allow_refusal=allow_refusal)
        reasons += r
        warnings += w
    if q.get("drop_low_confidence", True) and rec.get("confidence") == "low":
        reasons.append("low_confidence")
    return reasons, warnings


def _canon(s: str) -> str:
    return " ".join(tokens(s))


def validate_dpo(rec: Mapping[str, Any], context: Mapping[str, Any], language: str, q: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    reasons, warnings, english, allow_refusal = _shared_sft_fields(rec, context, language, q)
    chosen = str(rec.get("chosen", "")).strip()
    rejected = str(rec.get("rejected", "")).strip()
    rtype = context.get("rejection_type", rec.get("rejection_type"))
    length_related = bool(context.get("length_related", False))
    resp_is_english = "response" in english

    if not chosen:
        reasons.append("empty_field:chosen")
    if not rejected:
        reasons.append("empty_field:rejected")
    if chosen and rejected:
        if _canon(chosen) == _canon(rejected):
            reasons.append("chosen_eq_rejected")
        if not length_related:
            ratio = len(rejected) / max(len(chosen), 1)
            if ratio < float(q["min_length_ratio"]) or ratio > float(q["max_length_ratio"]):
                reasons.append("length_ratio")

    if chosen:
        _len_bounds(reasons, chosen, "chosen", int(q["min_response_chars"]), int(q["max_response_chars"]))
        r, w = check_text_field(chosen, language, "chosen", q, is_english=resp_is_english, allow_refusal=allow_refusal)
        reasons += r
        warnings += w
    if rejected:
        # a truncated / rambling rejected answer is short (or long) by design: no floor for length-related flaws
        _len_bounds(reasons, rejected, "rejected", 1 if length_related else int(q["min_response_chars"]),
                    int(q["max_response_chars"]))
        # The rejected answer is *expected* to contain flaws: skip the refusal
        # check for unhelpful_refusal, and the English-leak check for
        # wrong_language (its whole point). Everything else must still pass,
        # in particular "same language as chosen".
        r, w = check_text_field(
            rejected, language, "rejected", q,
            is_english=resp_is_english,
            allow_refusal=(allow_refusal or rtype == "unhelpful_refusal"),
            check_leak=(rtype != "wrong_language"),
        )
        # A flawed answer may legitimately be repetitive/verbose only for the
        # length-related types; keep repetition failures for the rest.
        if rtype == "rambling_verbose":
            r = [x for x in r if not x.startswith(("repetition", "degenerate"))]
        r = [x.replace("english_leak:rejected", "rejected_wrong_language") for x in r]
        reasons += r
        warnings += w

    if q.get("drop_low_confidence", True) and rec.get("chosen_confidence") == "low":
        reasons.append("low_confidence")
    if rec.get("rejection_type") != rtype:
        reasons.append("rejection_type_mismatch")
    return reasons, warnings
