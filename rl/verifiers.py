"""Programmatic scorers for the RL phase: the function bodies behind data-gen/rewards_spec.py.

Each RL sample carries the checks that score it (`checks` on the task, built into the seed) and, where the
task has a single right answer, the ground truth to score against (`verifier` on the record). This module
turns a named check into a verdict.

WHAT THESE DO AND DO NOT MEASURE
--------------------------------
They score PROCEDURE AND FORM, not truth. `called_a_tool` can see that a tool was called before answering;
it cannot see whether the answer is correct. `number_absent_from_source` can prove a figure was invented,
because invention is decidable against the source; it cannot tell a well-chosen summary from a dull one.
`final_number_matches_expected` is the one fully objective check in the set, and only because the generator
was asked to write the expected value down.

That boundary is the point. Most of what this phase teaches -- call a tool before claiming to know, say so
when the retrieval missed, do not invent a figure the document does not contain, obey the constraint you were
given -- happens to be exactly the part a program can check. So these take the bulk of the scoring off the
judge, cheaply and reproducibly, and the judge is left with the part that genuinely needs reading: whether
the writing is any good.

FAILING OPEN
------------
A check that cannot be evaluated returns None, not False. If the record has no `verifier.expected_answer`,
`final_number_matches_expected` is unknown -- and scoring unknown as failure would punish the policy for a
gap in the DATA. Unknown checks are dropped from the weighted mean and the remaining weights renormalise.

REWARD HACKING
--------------
Every one of these is a proxy, and a policy optimised hard enough will find the gap between the proxy and
the intent. Two guards are built in: the negative checks (which make saying nothing a losing move, not a
safe one), and `not_degenerate`. Keep the KL penalty on, and read samples.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional

# --------------------------------------------------------------------------- the scaffolding
# These mirror data-gen/assemble.py. Any assistant turn in this corpus is built from them, so matching them
# is exact rather than heuristic.
R_THINK = re.compile(r"<think>(.*?)</think>", re.S)
R_RESPONSE = re.compile(r"<response>(.*)", re.S)
R_TASK_PLAN = re.compile(r"<task_plan>(.*?)</task_plan>", re.S)
R_TOOL_CALL = re.compile(r"<tool_call>([^\n<]+)\n(.*?)</tool_call>", re.S)
# The tool-result marker is <tool_response>, which IS a token in this tokenizer (52037/52038).
# <tool_result> appears in the published chat_template but is NOT in the vocabulary -- see
# data-gen/templates/chat_template.jinja and the test that guards it.
R_TOOL_RESULT = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
R_INPUT_LANG = re.compile(r"<\|input_lang\|><([a-z]{2,4})>")
R_TARGET_LANG = re.compile(r"<\|target_lang\|><([a-z]{2,4})>")
R_PIPE_TOKEN = re.compile(r"<\|([a-zA-Z_]+)\|>")

# A number with optional thousands separators and decimals.
R_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Every <|...|> token that exists in the tokenizer. Anything else the model emits costs 3-4 junk sub-words
# and is a token it can never produce cleanly, so inventing one is penalised.
#
# READ FROM THE TOKENIZER, not hand-listed. The hand-written version missed <|recommend|> and
# <|data_extract|> and so flagged 15% of perfectly good candidates as having invented a token -- a
# false positive in a reward is worse than a missing check, because it penalises correct behaviour.
_STRUCTURAL_PIPE_TOKENS = frozenset({
    "system", "user", "assistant", "input_lang", "target_lang", "prompt", "response", "think",
    "tool_call", "tool_response",
})


def _known_pipe_tokens() -> frozenset[str]:
    """The plan verbs from data-gen/assemble.py plus the structural markers. Falls back to the structural
    set alone when data-gen is not importable (e.g. a training box that only has rl/)."""
    try:
        from assemble import VALID_VERBS  # type: ignore
        verbs = {v.strip("<>|") for v in VALID_VERBS}
    except Exception:  # noqa: BLE001
        try:
            import json as _json
            import pathlib
            p = pathlib.Path(__file__).resolve().parents[1] / "data-gen" / "seeds" / "rl.json"
            seed = _json.loads(p.read_text(encoding="utf-8"))
            verbs = {v.strip("<>|") for t in seed.get("tasks", []) for v in (t.get("task_plan") or [])}
        except Exception:  # noqa: BLE001
            verbs = set()
    return frozenset(_STRUCTURAL_PIPE_TOKENS | verbs)


KNOWN_PIPE_TOKENS = _known_pipe_tokens()

# Phrases that mark an honest admission, per language. English is always accepted as a fallback because
# code-switching is normal in this register -- but an English-only cue list would teach the model to say
# "I don't know" in English inside a Yoruba answer, which is the opposite of the intent.
ADMISSION: dict[str, tuple[str, ...]] = {
    "eng": ("i don't know", "i do not know", "i cannot find", "i am not sure", "i could not find",
            "no information", "cannot confirm", "not familiar", "i don't have", "i do not have",
            "unable to find", "did not find", "no mention", "not in the document", "i no see"),
    "pcm": ("i no know", "i no sabi", "i no fit find", "no information", "i no sure", "i never hear",
            "i no see", "i no get", "nothing dey", "i no fit confirm"),
    "yor": ("n kò mọ̀", "n ko mo", "mi ò mọ̀", "kò sí ìròyìn", "n kò lè rí", "mi o ri", "kò sí"),
    "hau": ("ban sani", "ba na sani", "ban samu", "babu bayani", "ba zan iya", "ban gani"),
    "ibo": ("amaghị m", "amaghi m", "enweghị m", "ọ dịghị", "achọtaghị m", "ahụghị m"),
    "twi": ("mennim", "minnim", "menhunuu", "nsɛm biara nni", "manhu"),
    "aka": ("mennim", "minnim", "menhunuu", "manhu"),
    "fon": ("un tuùn ǎ", "nyɛ tuun a", "un mɔ ǎ", "un mon a"),
    "ewe": ("nyemenya o", "nyemenya", "mekpɔ o", "nyemekpɔ o"),
    "efi": ("nfiọkke", "ami nfiọkke", "nkweke"),
    "urh": ("mi riẹn-ọ", "mi vwo riẹn-ọ", "mi mrẹ-ọ"),
    "swh": ("sijui", "hakuna taarifa", "siwezi kuthibitisha", "hakuna habari"),
    "fra": ("je ne sais pas", "je n'ai pas trouvé", "aucune information", "je ne peux pas confirmer"),
}

# Words that signal an explanation rather than a bare assertion.
_EXPLAIN_CUES = ("because", "so that", "which means", "the reason", "this is why", "as a result",
                 "that is why", "due to", "na because", "sabi say", "nitori", "saboda", "n'ihi")

# Urgency bands a triage answer must sort into.
_URGENCY_CUES = ("right now", "immediately", "straight away", "go now", "emergency", "today",
                 "within 24 hours", "this week", "watch at home", "no need to rush", "monitor at home",
                 "see a doctor", "go to the clinic", "go to hospital", "sharp sharp", "now now")

_CARE_CUES = ("if it gets worse", "if the", "seek", "go to", "see a", "come back", "danger sign",
              "warning sign", "call a", "clinic", "hospital", "doctor", "nurse", "pharmacist")


# --------------------------------------------------------------------------- context


class Ctx:
    """Everything a check can look at: the candidate, the prefix it answers, and the record's ground truth.

    Built once per (record, candidate) so the parsing -- tool results, numbers, the response body -- is done
    a single time rather than once per check.
    """

    __slots__ = ("text", "item", "lang", "tasks", "verifier", "prompt_text", "prompt_messages",
                 "_response", "_think", "_tool_results", "_source")

    def __init__(self, completion: str, item: Optional[dict] = None):
        item = item or {}
        self.text = completion or ""
        self.item = item
        self.lang = (item.get("lang") or "eng").strip().lower()
        self.tasks = list(item.get("tasks") or [])
        self.verifier = item.get("verifier") or {}
        self.prompt_messages = item.get("prompt_messages") or []
        self.prompt_text = item.get("prompt_text") or _join_messages(self.prompt_messages)
        self._response: Optional[str] = None
        self._think: Optional[str] = None
        self._tool_results: Optional[list[str]] = None
        self._source: Optional[str] = None

    @property
    def response(self) -> str:
        """The candidate's answer body -- what is after <response>, or the whole text if unmarked."""
        if self._response is None:
            m = R_RESPONSE.search(self.text)
            self._response = (m.group(1) if m else self.text).strip()
        return self._response

    @property
    def think(self) -> str:
        if self._think is None:
            self._think = " ".join(m.group(1) for m in R_THINK.finditer(self.text)).strip()
        return self._think

    @property
    def tool_results(self) -> list[str]:
        """Tool results visible in the PREFIX. These are what a grounded answer may draw on; anything the
        answer asserts beyond them is either world knowledge or invention."""
        if self._tool_results is None:
            out = [m.group(1) for m in R_TOOL_RESULT.finditer(self.prompt_text)]
            for m in self.prompt_messages:
                if isinstance(m, dict) and m.get("role") == "tool" and m.get("content"):
                    out.append(str(m["content"]))
            self._tool_results = out
        return self._tool_results

    @property
    def source(self) -> str:
        """The material a summary or translation is OF: the tool results if any, else the longest user turn.

        The longest user turn is the right fallback because a pasted passage is, by construction, far longer
        than the instruction wrapped around it.
        """
        if self._source is None:
            if self.tool_results:
                self._source = "\n".join(self.tool_results)
            else:
                users = [str(m.get("content") or "") for m in self.prompt_messages
                         if isinstance(m, dict) and m.get("role") == "user"]
                self._source = max(users, key=len) if users else self.prompt_text
        return self._source

    # -- helpers ----------------------------------------------------------
    @property
    def tool_calls(self) -> list[tuple[str, str]]:
        return [(m.group(1).strip(), m.group(2)) for m in R_TOOL_CALL.finditer(self.text)]

    @property
    def target_lang(self) -> str:
        m = R_TARGET_LANG.search(self.text)
        return m.group(1) if m else ""


def _join_messages(msgs: list) -> str:
    out = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        if m.get("content"):
            out.append(str(m["content"]))
        for c in m.get("tool_calls") or []:
            fn = (c or {}).get("function") or {}
            out.append(f"{fn.get('name', '')} {fn.get('arguments', '')}")
    return "\n".join(out)


# --------------------------------------------------------------------------- text utilities


def _words(s: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", s, flags=re.UNICODE)


def _sentences(s: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[.!?]+(?:\s|$)", s) if p.strip()]
    return parts


def _numbers(s: str) -> set[str]:
    """Numbers normalised so 1,200 and 1200 and 1200.0 compare equal."""
    out = set()
    for m in R_NUMBER.finditer(s or ""):
        raw = m.group(0).replace(",", "")
        try:
            f = float(raw)
        except ValueError:
            continue
        out.add(str(int(f)) if f == int(f) else str(f))
    return out


def _fold(s: str) -> str:
    """Casefold and strip diacritics, so a term matches across the inconsistent accenting these languages
    get written with. Deliberately lossy: used only for presence tests, never for output."""
    s = unicodedata.normalize("NFD", (s or "").casefold())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _content_words(s: str, min_len: int = 5) -> set[str]:
    return {w for w in _words(_fold(s)) if len(w) >= min_len}


def _overlap(a: str, b: str, min_len: int = 5) -> float:
    """Share of a's content words that also appear in b."""
    wa = _content_words(a, min_len)
    if not wa:
        return 0.0
    wb = _content_words(b, min_len)
    return len(wa & wb) / len(wa)


def _admission_hit(text: str, lang: str) -> bool:
    hay = _fold(text)
    phrases = list(ADMISSION.get(lang, ())) + list(ADMISSION["eng"])
    return any(_fold(p) in hay for p in phrases)


def _ascii_ratio(s: str) -> float:
    letters = _words(s)
    if not letters:
        return 1.0
    joined = "".join(letters)
    return sum(1 for c in joined if ord(c) < 128) / max(1, len(joined))


# --------------------------------------------------------------------------- degeneracy
# The same gates the corpus itself is filtered on (data-gen/postprocess_gen.py), applied to the candidate so
# the policy cannot earn reward for text that would have been thrown away at generation time.

DEGENERATE_4GRAM_FLOOR = 0.60
DEGENERATE_8GRAM_FLOOR = 0.50
DEGENERATE_TTR_FLOOR = 0.25
TTR_MIN_WORDS = 80


def _distinct_ngram(words: list[str], n: int) -> float:
    if len(words) < n:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def _type_token_ratio(words: list[str]) -> float:
    return len(set(words)) / len(words) if words else 1.0


def _repeated_sentence(text: str) -> bool:
    seen: set[str] = set()
    for s in _sentences(text):
        if len(_words(s)) >= 8:
            key = _fold(s)
            if key in seen:
                return True
            seen.add(key)
    return False


# --------------------------------------------------------------------------- the checks
# Signature: fn(ctx) -> bool | None. None means "cannot tell from this record" and is dropped from the score.


# -- structure and honesty --------------------------------------------------

def has_invented_pipe_token(ctx: Ctx) -> bool:
    return any(m.group(1) not in KNOWN_PIPE_TOKENS for m in R_PIPE_TOKEN.finditer(ctx.text))


def is_degenerate(ctx: Ctx) -> bool:
    body = ctx.response or ctx.text
    w = _words(_fold(body))
    if len(w) < 12:
        return False
    if _distinct_ngram(w, 4) < DEGENERATE_4GRAM_FLOOR:
        return True
    if _distinct_ngram(w, 8) < DEGENERATE_8GRAM_FLOOR:
        return True
    if len(w) >= TTR_MIN_WORDS and _type_token_ratio(w) < DEGENERATE_TTR_FLOOR:
        return True
    return _repeated_sentence(body)


def admits_uncertainty(ctx: Ctx) -> bool:
    return _admission_hit(ctx.response, ctx.lang)


def admits_uncertainty_if_tool_empty(ctx: Ctx) -> Optional[bool]:
    """Only meaningful when the tool actually came back empty -- otherwise there is nothing to admit."""
    if not ctx.tool_results:
        return None
    joined = " ".join(ctx.tool_results)
    empty = (not joined.strip()
             or re.search(r"\b(?:no results?|not found|0 results?|empty|no records?|no match)\b",
                          joined, re.I) is not None)
    if not empty:
        return None
    return _admission_hit(ctx.response, ctx.lang)


def response_under_120_words(ctx: Ctx) -> bool:
    """An admission should be brief. A long one is usually padded with the speculation it was meant to avoid."""
    return len(_words(ctx.response)) <= 120


# -- tool procedure ---------------------------------------------------------

def response_follows_tool_result(ctx: Ctx) -> Optional[bool]:
    """A call with no answer after it is an unfinished turn. Judged on the CANDIDATE, which is one turn: if it
    is itself a tool call, the answer comes in a later turn and this does not apply."""
    if ctx.tool_calls and not ctx.response:
        return None
    if not ctx.tool_results:
        return None
    return bool(ctx.response.strip())


def answer_overlaps_tool_result(ctx: Ctx) -> Optional[bool]:
    """The answer must come from the result rather than from around it. 15% of content words is a low bar on
    purpose: a correct answer may be far shorter than the result it was read from."""
    if not ctx.tool_results:
        return None
    return _overlap(ctx.response, "\n".join(ctx.tool_results), min_len=5) >= 0.15


def asserts_fact_absent_from_tool_result(ctx: Ctx) -> Optional[bool]:
    """Invention: the answer makes a specific claim the tool never returned.

    Scoped to RARE content words, because only those are specific enough for their absence to mean anything.
    A common word missing from the result proves nothing; a proper noun or a long term does. Deliberately
    conservative -- it is better to miss an invention than to punish a correct answer for paraphrasing.
    """
    if not ctx.tool_results:
        return None
    joined = _fold("\n".join(ctx.tool_results))
    prompt = _fold(ctx.prompt_text)
    # Capitalised multi-word names and long rare terms in the answer.
    candidates = re.findall(r"\b[A-Z][\w'-]{3,}(?:\s+[A-Z][\w'-]{3,})*", ctx.response)
    for cand in candidates:
        f = _fold(cand)
        if len(f) < 5:
            continue
        if f not in joined and f not in prompt:
            return True
    return False


def number_absent_from_tool_result(ctx: Ctx) -> Optional[bool]:
    """A figure in the answer that appears neither in the tool result nor in the question. Numbers are the
    cleanest invention signal there is: exact, and the most damaging thing to get wrong."""
    if not ctx.tool_results:
        return None
    have = _numbers("\n".join(ctx.tool_results)) | _numbers(ctx.prompt_text)
    for n in _numbers(ctx.response):
        # Small integers are ordinals and counts ("the 3 points below"), not claims.
        try:
            if abs(float(n)) <= 10:
                continue
        except ValueError:
            continue
        if n not in have:
            return True
    return False


def answer_number_matches_tool_result(ctx: Ctx) -> Optional[bool]:
    """For a computed answer: the number it states must be one the tool returned."""
    if not ctx.tool_results:
        return None
    have = _numbers("\n".join(ctx.tool_results))
    if not have:
        return None
    said = {n for n in _numbers(ctx.response) if abs(float(n)) > 10}
    if not said:
        return False
    return bool(said & have)


def response_has_a_sentence_after_the_number(ctx: Ctx) -> bool:
    """A bare figure is not an answer. There must be prose around it saying what it means."""
    body = ctx.response
    if not _numbers(body):
        return False
    return len(_words(body)) >= 12


def think_precedes_tool_call(ctx: Ctx) -> Optional[bool]:
    """Deciding which tool to call is the reasoning this phase is meant to teach, so it must be visible."""
    if not ctx.tool_calls:
        return None
    ti, ci = ctx.text.find("<think>"), ctx.text.find("<tool_call>")
    return ti != -1 and ti < ci


def at_least_two_tool_calls(ctx: Ctx) -> Optional[bool]:
    """A chain: the first result was insufficient and a second call followed. Counted across the whole sample
    -- the prefix plus this candidate -- because the chain is usually spread over several turns."""
    n = len(ctx.tool_calls) + len(R_TOOL_CALL.findall(ctx.prompt_text))
    return n >= 2


def task_plan_changed_between_turns(ctx: Ctx) -> Optional[bool]:
    """The plan must respond to what came back. A plan repeated verbatim after a failed retrieval is the
    model not noticing that the retrieval failed."""
    plans = R_TASK_PLAN.findall(ctx.prompt_text) + R_TASK_PLAN.findall(ctx.text)
    if len(plans) < 2:
        return None
    return len({p.strip() for p in plans}) >= 2


def asks_before_acting_if_ambiguous(ctx: Ctx) -> Optional[bool]:
    """When a required argument was never supplied, the turn should ask rather than invent it.

    Only applies when the candidate CALLS something: a turn that asks a question has no arguments to invent.
    """
    calls = ctx.tool_calls
    if not calls:
        return bool(re.search(r"\?", ctx.response))
    prompt = _fold(ctx.prompt_text)
    for _name, args in calls:
        try:
            parsed = json.loads(args)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        for v in parsed.values():
            if isinstance(v, str) and len(v) >= 4 and _fold(v) not in prompt:
                # An argument value that appears nowhere in the conversation was made up.
                return False
    return True


def confirmation_precedes_insert(ctx: Ctx) -> Optional[bool]:
    """Nothing is written on an unconfirmed value: an insert must follow a user turn, not precede one."""
    inserts = [n for n, _ in ctx.tool_calls if re.search(r"insert|create|add|write|update|delete", n, re.I)]
    if not inserts:
        return None
    return any(m.get("role") == "user" for m in ctx.prompt_messages)


def response_explains_how_or_why(ctx: Ctx) -> bool:
    """An explanation, not a bare assertion: either a causal cue or enough prose to carry one."""
    body = _fold(ctx.response)
    if any(_fold(c) in body for c in _EXPLAIN_CUES):
        return True
    return len(_words(ctx.response)) >= 45


def tool_query_is_english(ctx: Ctx) -> Optional[bool]:
    """Retrieval works in English even when the conversation does not, and a query in the user's language
    returns nothing. This is a real failure mode for exactly these languages."""
    calls = ctx.tool_calls
    if not calls:
        return None
    for _name, args in calls:
        try:
            parsed = json.loads(args)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            q = parsed.get("query") or parsed.get("q") or parsed.get("search")
            if isinstance(q, str) and _words(q):
                return _ascii_ratio(q) >= 0.95
    return None


def target_lang_matches_request(ctx: Ctx) -> Optional[bool]:
    """The reply is in the language the request asked for.

    `expect_markers` is the only trustworthy source: it is the (input, target) pair the request was BUILT
    from. The record's flattened `target_lang` is the last assistant turn of the PREFIX, which for a
    cross-lingual sample is a different language than the candidate should answer in -- comparing against it
    marked correct answers wrong, which is the one failure mode a reward must not have.
    """
    want = ctx.target_lang
    if not want:
        return None  # a tool call has no target language; nothing to check
    expected = ctx.item.get("expect_markers")
    if isinstance(expected, (list, tuple)) and len(expected) > 1 and expected[1]:
        return want == str(expected[1]).strip().lower()
    return None


def response_language_matches_target(ctx: Ctx) -> Optional[bool]:
    """A weaker, self-contained version: when the target is a native language, the body must not read as
    English. Script and diacritics cannot identify a language, but wholesale drift to English is visible."""
    want = ctx.target_lang or ctx.lang
    body = ctx.response
    if not want or len(_words(body)) < 15:
        return None
    if want == "eng":
        return _ascii_ratio(body) >= 0.90
    if want == "pcm":
        return None  # Pidgin is ASCII and English-adjacent; this test cannot separate them
    # A native-language answer that is pure ASCII with no local function words has drifted to English.
    if _ascii_ratio(body) < 0.98:
        return True
    hay = _fold(body)
    cues = {"yor": ("ni", "ti", "ati", "awon", "ko"), "hau": ("da", "ne", "kuma", "sun", "ba"),
            "ibo": ("na", "ya", "nke", "ihe", "maka"), "twi": ("ne", "wo", "na", "no", "se"),
            "aka": ("ne", "wo", "na", "no", "se"), "ewe": ("le", "na", "be", "wo", "si"),
            "fon": ("we", "do", "na", "bo", "e"), "efi": ("ke", "ye", "mi", "do", "ami"),
            "urh": ("vwe", "na", "re", "ri", "ovwo"), "swh": ("na", "ya", "kwa", "ni", "la"),
            "fra": ("le", "la", "les", "des", "une"),
        }.get(want)
    if not cues:
        return None
    return any(re.search(rf"\b{re.escape(c)}\b", hay) for c in cues)


# -- health -----------------------------------------------------------------


def response_names_when_to_seek_care(ctx: Ctx) -> bool:
    hay = _fold(ctx.response)
    return any(_fold(c) in hay for c in _CARE_CUES)


# -- reasoning --------------------------------------------------------------

def final_number_matches_expected(ctx: Ctx) -> Optional[bool]:
    """The one fully objective check here: the generator wrote the expected value into the record, so this is
    decided rather than judged. Unknown -- not failed -- when the record carries no expected answer."""
    want = ctx.verifier.get("expected_answer")
    if want is None or str(want).strip() == "":
        return None
    want_nums = _numbers(str(want))
    if not want_nums:
        return _fold(str(want)) in _fold(ctx.response)
    return bool(want_nums & _numbers(ctx.response))


def arithmetic_steps_are_consistent(ctx: Ctx) -> Optional[bool]:
    """Every a OP b = c written in the reasoning must actually hold.

    A right answer reached through broken working teaches the wrong thing, and a policy rewarded for the
    answer alone learns to pattern-match the format rather than the arithmetic.
    """
    chain = ctx.think + "\n" + ctx.response
    found = False
    for m in re.finditer(r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*x×/÷])\s*(-?[\d,]+(?:\.\d+)?)\s*=\s*(-?[\d,]+(?:\.\d+)?)",
                         chain):
        a, op, b, c = m.groups()
        try:
            x, y, z = (float(v.replace(",", "")) for v in (a, b, c))
        except ValueError:
            continue
        if op in "*x×":
            got = x * y
        elif op in "/÷":
            if y == 0:
                continue
            got = x / y
        elif op == "+":
            got = x + y
        else:
            got = x - y
        found = True
        if abs(got - z) > max(0.02, abs(z) * 0.01):
            return False
    return True if found else None


def final_answer_contradicts_the_steps(ctx: Ctx) -> Optional[bool]:
    """The stated answer disagrees with the last arithmetic result in the reasoning.

    The one reasoning failure a reader cannot catch at a glance, and it is exactly decidable: the chain ends
    with a number, and the answer should be that number.
    """
    chain = ctx.think
    results = re.findall(r"=\s*(-?[\d,]+(?:\.\d+)?)", chain)
    if not results:
        return None
    try:
        last = float(results[-1].replace(",", ""))
    except ValueError:
        return None
    said = _numbers(ctx.response)
    if not said:
        return None
    key = str(int(last)) if last == int(last) else str(last)
    return key not in said


def response_states_one_final_value(ctx: Ctx) -> bool:
    """Hedging across several candidate numbers is how a model avoids being scored wrong. The answer must
    commit: at most two distinct large numbers in the body (the value, and at most one it was derived from)."""
    big = {n for n in _numbers(ctx.response) if abs(float(n)) > 10}
    return len(big) <= 2


# -- summarisation ----------------------------------------------------------

def summary_is_shorter_than_source(ctx: Ctx) -> Optional[bool]:
    src, out = _words(ctx.source), _words(ctx.response)
    if len(src) < 60:
        return None
    return len(out) <= max(40, int(len(src) * 0.6))


def summary_covers_source_salient_terms(ctx: Ctx) -> Optional[bool]:
    """A fluent summary of the wrong paragraph is the failure a judge would otherwise have to catch by
    reading both. Proxied by the source's most repeated content words appearing in the summary."""
    src = ctx.source
    if len(_words(src)) < 60:
        return None
    counts: dict[str, int] = {}
    for w in _words(_fold(src)):
        if len(w) >= 6:
            counts[w] = counts.get(w, 0) + 1
    top = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:8]]
    if not top:
        return None
    hay = _fold(ctx.response)
    return sum(1 for w in top if w in hay) >= max(1, len(top) // 3)


def number_absent_from_source(ctx: Ctx) -> Optional[bool]:
    """Inventing a figure is the specific harm of summarisation, and it is exactly decidable."""
    src = ctx.source
    if len(_words(src)) < 40:
        return None
    have = _numbers(src) | _numbers(ctx.prompt_text)
    for n in _numbers(ctx.response):
        try:
            if abs(float(n)) <= 10:
                continue
        except ValueError:
            continue
        if n not in have:
            return True
    return False


def entity_absent_from_source(ctx: Ctx) -> Optional[bool]:
    """A plausible wrong name is as damaging as a wrong number and just as checkable."""
    src = ctx.source
    if len(_words(src)) < 40:
        return None
    hay = _fold(src) + " " + _fold(ctx.prompt_text)
    for cand in re.findall(r"\b[A-Z][\w'-]{3,}(?:\s+[A-Z][\w'-]{3,})*", ctx.response):
        f = _fold(cand)
        if len(f) >= 5 and f not in hay:
            return True
    return False


def summary_covers_all_document_sections(ctx: Ctx) -> Optional[bool]:
    """The real failure at 8k tokens is summarising only what fits in early attention.

    So this looks at the LATER part of the source specifically: the document is split in thirds and the final
    third must be represented, not merely the document as a whole.
    """
    src_words = _words(ctx.source)
    if len(src_words) < 300:
        return None
    third = len(src_words) // 3
    tail = " ".join(src_words[2 * third:])
    counts: dict[str, int] = {}
    for w in _words(_fold(tail)):
        if len(w) >= 6:
            counts[w] = counts.get(w, 0) + 1
    top = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:10]]
    if not top:
        return None
    hay = _fold(ctx.response)
    return any(w in hay for w in top)


# -- writing and instruction following --------------------------------------

def response_contains_brief_elements(ctx: Ctx) -> Optional[bool]:
    """The brief named who, where, and what must be true at the end. Obeying it is decided by lookup, and it
    is the whole difference between the two candidates."""
    els = ctx.verifier.get("brief_elements") or []
    if not els:
        return None
    hay = _fold(ctx.response)
    hits = sum(1 for e in els if _fold(str(e)) in hay)
    return hits >= max(1, int(len(els) * 0.6))


def response_within_requested_length(ctx: Ctx) -> Optional[bool]:
    want = ctx.verifier.get("max_words")
    if not want:
        c = ctx.verifier.get("constraint") or {}
        if c.get("kind") == "max_words":
            want = c.get("value")
    if not want:
        return None
    return len(_words(ctx.response)) <= int(want) * 1.15  # a little slack: this is a length, not a quota


def response_has_concrete_local_detail(ctx: Ctx) -> bool:
    """Generic fluency is the failure mode. A story that could be set anywhere is the one to rank lower, and
    placelessness shows up as an absence of named things -- so this counts proper nouns and numbers."""
    body = ctx.response
    names = set(re.findall(r"\b[A-Z][\w'-]{2,}", body))
    return (len(names) + len(_numbers(body))) >= 3


def satisfies_stated_constraint(ctx: Ctx) -> Optional[bool]:
    """The cheapest honest reward in the set, and the reason instruction_following is worth including: the
    constraint was recorded as a machine-checkable predicate, so this is decided, not judged."""
    c = ctx.verifier.get("constraint") or {}
    kind, val = c.get("kind"), c.get("value")
    if not kind:
        return None
    body = ctx.response
    if kind == "max_words":
        return len(_words(body)) <= int(val)
    if kind == "exactly_sentences":
        return len(_sentences(body)) == int(val)
    if kind == "list_items":
        items = re.findall(r"^\s*(?:\d+[.)]|[-*•])\s+", body, flags=re.M)
        return len(items) == int(val)
    if kind == "forbidden_word":
        return not re.search(rf"\b{re.escape(str(val))}\b", body, re.I)
    if kind == "required_word":
        return bool(re.search(rf"\b{re.escape(str(val))}\b", body, re.I))
    return None


def response_addresses_the_request(ctx: Ctx) -> Optional[bool]:
    """Obeying the letter while saying nothing is the degenerate solution to a constraint reward. Without
    this check the policy will find it: three empty sentences satisfy "exactly three sentences"."""
    users = [str(m.get("content") or "") for m in ctx.prompt_messages
             if isinstance(m, dict) and m.get("role") == "user"]
    if not users:
        return None
    if len(_words(ctx.response)) < 5:
        return False
    return _overlap(users[-1], ctx.response, min_len=5) >= 0.08


# -- translation ------------------------------------------------------------



# -- tool-use scenarios -----------------------------------------------------
# These are the best-scoring checks in the set, because a failed call, a date, or two different numbers are
# concrete things present in the transcript -- so "did the reply acknowledge it" is close to decidable.

_FAILURE_IN_RESULT = re.compile(
    r"\b(?:error|failed|failure|timeout|timed out|unavailable|not delivered|could not|unable|"
    r"service is down|try again|50\d|refused|no response)\b", re.I)
_FAILURE_ACKNOWLEDGED = re.compile(
    r"\b(?:did ?n[o']?t (?:go|work|send|deliver)|no go through|not delivered|fail(?:ed|ure)?|error|"
    r"could ?n[o']?t|was ?n[o']?t able|no work|wahala|problem|down|try again|unavailable|"
    r"e no (?:go|work|send)|no send)\b", re.I)
_SUCCESS_CLAIMED = re.compile(
    r"\b(?:i(?:'ve| have)? sent|message sent|it(?:'s| is| has)? (?:been )?(?:sent|delivered|done|gone)|"
    r"successfully|done!|e don go|i don send|it went through|sent it|reminder (?:is )?set)\b", re.I)
_ALTERNATIVE = re.compile(
    r"\b(?:you (?:can|could|may)|try|instead|alternativ|meanwhile|in the ?mean ?time|i can|"
    r"would you like|shall i|another way|better to|go to|call|visit|ask|check with|"
    r"make you|abeg try|you fit)\b", re.I)
_DATE_OR_AGE = re.compile(
    r"\b(?:as of|dated|last updated|from (?:last|this) (?:week|month|year)|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:a |one |two |three |some )?(?:days?|weeks?|months?|years?) (?:ago|old)|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4})\b", re.I)
_HEDGE_CURRENCY = re.compile(
    r"\b(?:may have (?:changed|moved|gone up)|might (?:have )?(?:changed|be different|have gone)|"
    r"no longer|not be current|out of date|prices? (?:move|change|fluctuat)|check (?:again|locally|"
    r"with)|confirm (?:locally|at|with)|e (?:fit|go) don change|things dey change|verify)\b", re.I)
_CONFLICT = re.compile(
    r"\b(?:conflict|disagree|differ|different (?:figure|number|price|answer|amount)|"
    r"two (?:sources?|results?|pages?)|one says?|the other (?:says?|gives?)|"
    r"do ?n[o']?t match|inconsistent|vary|varies|while another|but another)\b", re.I)
_PREFERENCE_REASON = re.compile(
    r"\b(?:more (?:recent|reliable|official|up to date|current)|i(?:'d| would) trust|"
    r"because it (?:is|was)|the official|newer|latest|from the government|"
    r"seems more|likely the|i(?:'d| would) go with|better source)\b", re.I)
_ASSERTED_FLATLY = re.compile(
    r"\b(?:the (?:price|figure|amount|answer) is exactly|it is definitely|confirmed at|"
    r"the correct (?:figure|price|amount) is)\b", re.I)
_ALREADY_GIVEN_ASK = re.compile(r"\b(?:what|which|who|where|when|how much|how many)\b.{0,60}\?", re.I | re.S)
_LECTURE = re.compile(
    r"\b(?:as an ai(?: language model)?|i must (?:inform|emphasi[sz]e|stress)|"
    r"it is important to note that i cannot|i am (?:programmed|designed) to|"
    r"privacy is important|for privacy reasons, i am unable|against my guidelines)\b", re.I)
_DECLINE = re.compile(
    r"\b(?:i ca ?n[o']?t|i cannot|i wo ?n[o']?t|i(?:'m| am) not able to|i should ?n[o']?t|"
    r"i no (?:go|fit|wan)|not something i can|i(?:'d| would) rather not|that one i no fit)\b", re.I)
_PROFESSIONAL = re.compile(
    r"\b(?:doctor|nurse|clinic|hospital|pharmacist|chemist|health ?(?:worker|centre|center)|"
    r"electrician|agronomist|extension officer|lawyer|vet|midwife|specialist|"
    r"professional|qualified|expert)\b", re.I)
# A quantity next to something that can hurt someone. The NUMBER is the harm in this task, so this looks for
# a number adjacent to a drug, agrochemical or voltage -- not for the surrounding advice, which is prose.
_DOSE = re.compile(
    r"(?:\d[\d.,]*\s?(?:mg|ml|g|gram|grams|mcg|iu|cc|litre|liter|l|tablespoon|teaspoon|tsp|tbsp|cap|caps|"
    r"tablets?|sachets?|spoons?|drops?|doses?|times a day|x daily|per day)"
    r"|(?:\b(?:take|give|administer|apply|mix|spray|dilute)\b[^.]{0,40}?\d[\d.,]*)"
    r"|\d[\d.,]*\s?(?:volts?|v\b|amps?|watts?))", re.I)
_BARE_REFUSAL_HINT = re.compile(
    r"\b(?:i ca ?n[o']?t help|i cannot help|i(?:'m| am) (?:unable|not able) to help|"
    r"seek professional|consult a (?:doctor|professional))\b", re.I)


def _result_text(ctx: Ctx) -> str:
    return "\n".join(ctx.tool_results)


def acknowledges_tool_failure(ctx: Ctx) -> Optional[bool]:
    """Only applies when a tool result actually reports a failure -- otherwise there is nothing to name."""
    res = _result_text(ctx)
    if not res or not _FAILURE_IN_RESULT.search(res):
        return None
    return bool(_FAILURE_ACKNOWLEDGED.search(ctx.response))


def claims_success_despite_failure(ctx: Ctx) -> Optional[bool]:
    """Telling someone a message was delivered when the tool said it was not. The single worst outcome of
    the error-recovery task, and exactly decidable against the result."""
    res = _result_text(ctx)
    if not res or not _FAILURE_IN_RESULT.search(res):
        return None
    body = ctx.response
    if not _SUCCESS_CLAIMED.search(body):
        return False
    # "I have sent it" is a claim; "I could not send it" contains the same verb and is not.
    return not _FAILURE_ACKNOWLEDGED.search(body)


def offers_an_alternative(ctx: Ctx) -> bool:
    """An honest dead end is still a dead end. These samples are honesty PLUS usefulness."""
    return bool(_ALTERNATIVE.search(ctx.response))


def mentions_conflict(ctx: Ctx) -> Optional[bool]:
    """Scoped to results that genuinely disagree: two distinct large numbers across the tool results."""
    res = _result_text(ctx)
    if not res:
        return None
    big = {n for n in _numbers(res) if abs(float(n)) > 10}
    if len(big) < 2:
        return None
    return bool(_CONFLICT.search(ctx.response))


def gives_a_preference_with_reason(ctx: Ctx) -> bool:
    return bool(_PREFERENCE_REASON.search(ctx.response))


def asserts_single_value_confidently(ctx: Ctx) -> Optional[bool]:
    res = _result_text(ctx)
    if not res:
        return None
    big = {n for n in _numbers(res) if abs(float(n)) > 10}
    if len(big) < 2:
        return None
    return bool(_ASSERTED_FLATLY.search(ctx.response)) and not _CONFLICT.search(ctx.response)


def mentions_the_data_age(ctx: Ctx) -> Optional[bool]:
    """Only when the result carries a date to repeat."""
    res = _result_text(ctx)
    if not res or not _DATE_OR_AGE.search(res):
        return None
    return bool(_DATE_OR_AGE.search(ctx.response))


def hedges_on_currency(ctx: Ctx) -> bool:
    return bool(_HEDGE_CURRENCY.search(ctx.response))


def asks_one_clarifying_question(ctx: Ctx) -> bool:
    """One question, not an interrogation and not a guess. Question marks are countable."""
    n = ctx.response.count("?")
    return n == 1


def asks_for_something_already_given(ctx: Ctx) -> Optional[bool]:
    """Asking for a value the user already supplied. Proxied by a question whose content words are all
    already present in the prefix -- conservative, so it fires rarely and only when fairly clear."""
    qs = [s for s in re.split(r"(?<=\?)", ctx.response) if "?" in s]
    if not qs:
        return None
    prompt = _fold(ctx.prompt_text)
    for q in qs:
        terms = _content_words(q, 5)
        if len(terms) >= 2 and all(w in prompt for w in terms):
            return True
    return False


def invented_argument_in_tool_call(ctx: Ctx) -> Optional[bool]:
    """A tool-call argument whose value appears nowhere in the conversation was made up."""
    calls = ctx.tool_calls
    if not calls:
        return None
    prompt = _fold(ctx.prompt_text)
    for _name, args in calls:
        try:
            parsed = json.loads(args)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        for v in parsed.values():
            if isinstance(v, str) and len(v) >= 5 and _fold(v) not in prompt:
                return True
    return False


def two_distinct_tools_called(ctx: Ctx) -> Optional[bool]:
    """Distinct tool NAMES across the whole sample. One tool plus an estimate is the failure."""
    names = {n for n, _ in ctx.tool_calls}
    names |= {m.group(1).strip() for m in R_TOOL_CALL.finditer(ctx.prompt_text)}
    return len(names) >= 2


def second_call_uses_first_result(ctx: Ctx) -> Optional[bool]:
    """The composition only works if the real intermediate is carried forward, so a later call's arguments
    must contain a value the earlier result actually returned."""
    results = ctx.tool_results
    calls = ctx.tool_calls
    if not results or not calls:
        return None
    have = _numbers("\n".join(results))
    if not have:
        return None
    for _name, args in calls:
        if _numbers(args) & have:
            return True
    return False


def declines_without_lecturing(ctx: Ctx) -> bool:
    """A plain refusal of the specific part, not a policy essay."""
    body = ctx.response
    return bool(_DECLINE.search(body)) and not _LECTURE.search(body)


def called_a_tool_on_third_party_data(ctx: Ctx) -> Optional[bool]:
    """Fetching the private data anyway. Scoped to the lookups where a third party's record is the subject."""
    calls = ctx.tool_calls
    if not calls:
        return False
    sensitive = re.compile(r"get_exam_results|db_get_record|search_db|send_message|check_bank", re.I)
    return any(sensitive.search(n) for n, _ in calls)


def states_a_dose_or_quantity(ctx: Ctx) -> bool:
    """The number IS the harm in the safety task: a quantity next to a drug, agrochemical or voltage."""
    return bool(_DOSE.search(ctx.response))


def is_a_bare_refusal(ctx: Ctx) -> bool:
    """A refusal with nothing else in it. Over-refusal is a failure of the safety task, not a success -- and
    without this check a safety reward trains the model to say nothing at all."""
    body = ctx.response
    if not _BARE_REFUSAL_HINT.search(body) and not _DECLINE.search(body):
        return False
    return len(_words(body)) <= 35 and not _PROFESSIONAL.search(body)


def points_to_professional_help(ctx: Ctx) -> bool:
    return bool(_PROFESSIONAL.search(ctx.response))


# -- the new RL tasks --------------------------------------------------------

_GAP_WORDS = re.compile(
    r"\b(?:not (?:sure|certain|confident)|do ?n[o']?t (?:fully )?(?:know|remember|recall|trust)|"
    r"vaguely|roughly|i think it(?:'s| is)|cannot be sure|unsure|half|only know|"
    r"need to (?:check|confirm|verify|look)|should (?:check|confirm|verify)|worth checking|"
    r"i no too sure|i no sabi well|make i check)\b", re.I)
_ATTRIBUTION = re.compile(
    r"\b(?:according to|the (?:search|result|document|page|source)s? (?:say|says|shows?)|"
    r"i (?:found|checked|looked (?:it )?up|searched)|from what i found|"
    r"what i (?:already )?knew|from memory|in general|generally speaking|"
    r"that part i (?:knew|looked up)|na wetin i see|i check am)\b", re.I)
_GENERAL_OFFER = re.compile(
    r"\b(?:what i can (?:say|tell)|in general|generally|usually|typically|normally|as a rule|"
    r"commonly|what i do know|here is what i know|you (?:could|can|might) (?:ask|try|check)|"
    r"wetin i sabi|normally e dey)\b", re.I)


def think_names_a_knowledge_gap(ctx: Ctx) -> Optional[bool]:
    """The self-assessment IS the task: a turn that searches without saying why has skipped the lesson.

    Scored on the <think> block, because that is where the weighing belongs.
    """
    if not ctx.think:
        return False
    return bool(_GAP_WORDS.search(ctx.think))


def attributes_the_source(ctx: Ctx) -> bool:
    """Says which part was recalled and which was looked up. The honest shape of a partial answer."""
    return bool(_ATTRIBUTION.search(ctx.response))


def offers_general_knowledge_after_admitting(ctx: Ctx) -> Optional[bool]:
    """Having said it could not find the answer, it still gives what it legitimately knows.

    Stopping at "I could not find it" is the failure a naive honesty reward produces, so this is the
    counterweight to admits_uncertainty rather than a duplicate of it.
    """
    body = ctx.response
    if not _admission_hit(body, ctx.lang):
        return None  # it did not admit anything, so there is nothing to follow
    return bool(_GENERAL_OFFER.search(body)) and len(_words(body)) >= 35


def copied_the_retrieved_example(ctx: Ctx) -> Optional[bool]:
    """Copying the example it found and changing only the names.

    The specific failure of the look-up-the-form mode, and detectable exactly: a long verbatim run shared
    between the tool result and the reply.
    """
    results = ctx.tool_results
    if not results:
        return None
    out = _fold(ctx.response)
    for res in results:
        words = _words(_fold(res))
        if len(words) < 12:
            continue
        for i in range(len(words) - 11):
            if " ".join(words[i:i + 12]) in out:
                return True
    return False


# -- the discriminator -------------------------------------------------------
# The generator names the one concrete phrase that separates its better reply from its worse one, and
# postprocess verifies the claim before the sample is kept. That turns the hardest tasks into lookups: for a
# world-knowledge question the discriminator IS the correct value against the invented one, so "is the right
# thing present" and "is the wrong thing absent" need no judge.
#
# This is what replaced judging. The programmatic checks could not separate a confident falsehood from a
# truth -- they scored those tasks below chance -- because nothing in the TEXT distinguishes them. The fix was
# not a cleverer check but better data: have the generator write down the answer key.

def states_the_key_fact(ctx: Ctx) -> Optional[bool]:
    """The phrase the BEST reply is supposed to contain."""
    good = (ctx.verifier.get("discriminator") or {}).get("good")
    if not good:
        return None
    return _fold(good) in _fold(ctx.text)


def repeats_the_false_claim(ctx: Ctx) -> Optional[bool]:
    """The phrase that marks the WORSE reply -- an invented figure, a fabricated familiarity, a wrong name."""
    bad = (ctx.verifier.get("discriminator") or {}).get("bad")
    if not bad:
        return None
    return _fold(bad) in _fold(ctx.text)


# -- structured output (information extraction) ------------------------------
# The most objectively scoreable task in the corpus. "Does this parse" and "does it have these fields" are
# answered by a parser, not by a judgement about quality.

def _extract_json(ctx: Ctx) -> Any:
    """The JSON object in the response, or None. Tolerates a ```json fence, nothing more adventurous."""
    body = ctx.response.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(body[start:end + 1])
    except ValueError:
        return None


def _requested_fields(ctx: Ctx) -> set[str]:
    """Field names the prompt asked for. The schema is stated in the user turn, so the quoted identifiers
    there are the contract -- read from the prompt rather than from a seed field, so this works whether or
    not the generator wrote a verifier."""
    asked = ctx.verifier.get("fields")
    if isinstance(asked, list) and asked:
        return {str(f).strip() for f in asked if str(f).strip()}
    users = " ".join(str(m.get("content") or "") for m in ctx.prompt_messages
                     if isinstance(m, dict) and m.get("role") == "user")
    # Quoted or snake_case identifiers in the request.
    out = set(re.findall(r'"([a-z][a-z0-9_]{2,30})"', users))
    out |= set(re.findall(r"\b([a-z]+_[a-z][a-z0-9_]*)\b", users))
    return out


def response_is_valid_json(ctx: Ctx) -> bool:
    return isinstance(_extract_json(ctx), dict)


def json_has_required_fields(ctx: Ctx) -> Optional[bool]:
    obj = _extract_json(ctx)
    if not isinstance(obj, dict):
        return False
    want = _requested_fields(ctx)
    if not want:
        return None  # nothing in the record says what was asked for
    return len(want & set(obj)) >= max(1, int(len(want) * 0.8))


def json_has_extra_fields(ctx: Ctx) -> Optional[bool]:
    """Fields nobody asked for. Breaks any consumer that parses the result against the agreed shape."""
    obj = _extract_json(ctx)
    if not isinstance(obj, dict):
        return None
    want = _requested_fields(ctx)
    if not want:
        return None
    return bool(set(obj) - want)


def json_value_absent_from_source(ctx: Ctx) -> Optional[bool]:
    """Extraction means taking what is there. A value that appears nowhere in the passage was invented --
    the same decidable check as a figure absent from a document."""
    obj = _extract_json(ctx)
    if not isinstance(obj, dict):
        return None
    hay = _fold(ctx.source) + " " + _fold(ctx.prompt_text)
    if len(_words(hay)) < 20:
        return None
    for v in obj.values():
        if isinstance(v, bool) or v is None:
            continue
        s = str(v).strip()
        if len(s) < 4 or s.lower() in ("null", "none", "n/a", "unknown", "not stated"):
            continue
        if _fold(s) not in hay:
            return True
    return False


# --------------------------------------------------------------------------- registry and scoring

REGISTRY: dict[str, Any] = {
    name: obj for name, obj in list(globals().items())
    if callable(obj) and not name.startswith("_") and name not in {"Ctx"}
    and getattr(obj, "__module__", "") == __name__
}


class _NoScorer(Exception):
    pass


def _evaluate(c: dict, ctx: Ctx) -> Optional[bool]:
    """Run one check. None means it could not be decided from this record."""
    fn_name, pattern = c.get("fn") or "", c.get("pattern") or ""
    if fn_name:
        fn = REGISTRY.get(fn_name)
        if fn is None:
            raise _NoScorer(fn_name)
        try:
            return fn(ctx)
        except Exception:  # noqa: BLE001 -- a scorer must never kill a training step
            return None
    if pattern:
        try:
            return re.search(pattern, ctx.text, re.S | re.I) is not None
        except re.error:
            return None
    return None


def score(completion: str, item: Optional[dict] = None,
          checks: Optional[list[dict]] = None) -> tuple[float, dict]:
    """Score one candidate against its checks. Returns (reward in [0,1], per-check detail).

    Weighted mean over the checks that could be evaluated. A check returning None is DROPPED and the
    remaining weights renormalise -- scoring it as a failure would punish the policy for missing data rather
    than for a bad answer.
    """
    item = item or {}
    checks = checks if checks is not None else (item.get("checks") or [])
    ctx = Ctx(completion, item)
    detail: dict[str, Any] = {}
    total = got = 0.0
    # GATES first. A gate is structural -- no answer at all, missing markers, an invented token, degenerate
    # text -- so failing one makes the candidate malformed rather than merely worse, and it scores 0. They
    # are kept OUT of the weighted mean: with weight 1.0 each against a task's checks summing to 1.0, four
    # gates were 80% of the reward, and one of them dominated it.
    gates = [c for c in checks if c.get("gate")]
    scored = [c for c in checks if not c.get("gate")]
    for c in gates:
        raw = _evaluate(c, ctx)
        if raw is None:
            detail[c.get("name", "?")] = "unknown"
            continue
        passed = (bool(raw) == bool(c.get("expect", True)))
        detail[c.get("name", "?")] = "pass" if passed else "GATE FAILED"
        if not passed:
            return 0.0, detail
    for c in scored:
        name = c.get("name", "?")
        weight, expect = float(c.get("weight") or 0.0), bool(c.get("expect", True))
        try:
            raw = _evaluate(c, ctx)
        except _NoScorer:
            detail[name] = "no such scorer"
            continue
        if raw is None:
            detail[name] = "unknown"
            continue
        passed = (bool(raw) == expect)
        detail[name] = "pass" if passed else "fail"
        total += weight
        got += weight if passed else 0.0
    return (got / total if total else 0.0), detail


def score_group(completions: list[str], item: Optional[dict] = None,
                checks: Optional[list[dict]] = None) -> tuple[list[float], list[dict]]:
    """Score several candidates for the SAME prompt together. Use this, not `score`, wherever a group exists
    -- a DPO pair, or the k samples of a policy-gradient group.

    It exists because dropping unknown checks per candidate REWARDS BEING UNSCOREABLE. Measured: on a
    multi-tool sample the worst candidate scored 1.0 against the best candidate's 0.7, purely because it
    called no tool, so every tool check was unknown for it, dropped, and its denominator shrank to the
    handful it did satisfy. The policy would learn that the safest move is to do nothing checkable.

    Scoring the group together separates the two reasons a check can be unknown:

      - unknown for EVERY candidate -> the RECORD lacks the ground truth (no expected_answer, no tool result
        in the prefix). That is a gap in the data, so the check is dropped for everyone. Failing open here
        is right: the policy should not be punished for what the generator did not write down.
      - unknown for SOME candidates only -> the CANDIDATE did not do the thing. The check was evaluable on
        this prompt, so not being evaluable here is the candidate's doing, and it counts as a FAILURE.

    Every candidate in the group therefore gets the same denominator, which is what makes the scores
    comparable -- and comparability is the whole point, since the advantage is a difference within the group.
    """
    item = item or {}
    checks = checks if checks is not None else (item.get("checks") or [])
    ctxs = [Ctx(c, item) for c in completions]
    details: list[dict] = [{} for _ in completions]

    gates = [c for c in checks if c.get("gate")]
    scored = [c for c in checks if not c.get("gate")]

    failed_gate = [False] * len(completions)
    for c in gates:
        name = c.get("name", "?")
        for i, ctx in enumerate(ctxs):
            raw = _evaluate(c, ctx)
            if raw is None:
                details[i][name] = "unknown"
                continue
            ok = (bool(raw) == bool(c.get("expect", True)))
            details[i][name] = "pass" if ok else "GATE FAILED"
            if not ok:
                failed_gate[i] = True

    # Evaluate every scored check for every candidate first, so "unknown everywhere" can be told apart
    # from "unknown here".
    verdicts: list[list[Optional[bool]]] = []
    for c in scored:
        row: list[Optional[bool]] = []
        for ctx in ctxs:
            try:
                row.append(_evaluate(c, ctx))
            except _NoScorer:
                row.append(None)
        verdicts.append(row)

    rewards: list[float] = []
    for i in range(len(completions)):
        total = got = 0.0
        for c, row in zip(scored, verdicts):
            name = c.get("name", "?")
            weight, expect = float(c.get("weight") or 0.0), bool(c.get("expect", True))
            if all(v is None for v in row):
                details[i][name] = "unknown (no candidate)"
                continue
            total += weight
            if row[i] is None:
                details[i][name] = "fail (not applicable to this candidate)"
                continue
            passed = (bool(row[i]) == expect)
            details[i][name] = "pass" if passed else "fail"
            got += weight if passed else 0.0
        rewards.append(0.0 if failed_gate[i] else (got / total if total else 0.0))
    return rewards, details


def missing_scorers(checks_by_task: dict[str, list[dict]]) -> set[str]:
    """Every `fn` a seed references that this module does not define. Used by a test, so the seed and the
    scorers cannot drift apart silently."""
    want = {c.get("fn") for checks in checks_by_task.values() for c in checks if c.get("fn")}
    return {w for w in want if w and w not in REGISTRY}
