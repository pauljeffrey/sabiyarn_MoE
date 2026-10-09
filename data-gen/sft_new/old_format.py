"""Bring the older SFT repos (BeardedMonster/african_sft, eng_sft, naija_sft, hate, intent, topic, tasks, summarization, pos) up to
the canonical assistant grammar (sft_new/grammar.py).

What the older turns look like, and what is wrong with them:
    <|input_lang|><hau><ewe><task_plan><intent></task_plan><|target_lang|><eng><response>alarm       two language tags (instruction, text)
    <|input_lang|>Hausa<ewe><task_plan><NER></task_plan><|target_lang|>Hausa<response> O O B-ORG   a language NAME instead of a tag
    <|input_lang|><eng><fon><task_plan><pos></task_plan><response>SCONJ PROPN ...                 no <|target_lang|>; tags without words
    <|input_lang|><hau><task_plan> <|generate|> </task_plan><|target_lang|><hau><response>Beyond    plan padded with spaces; and the language
                                                                                                  tags can simply be wrong (English text, <hau>)
Canonical:
    <|input_lang|><text language>[ unsure ...]<task_plan>VERB[LABEL]</task_plan><|target_lang|><answer language><response>CONTENT

So every row is re-derived rather than patched: the text language comes from detection (or from the config name when the
repo states it), the plan tokens are mapped onto real tokens, a label token implies <classify>, and word-less POS/NER/chunk tag
streams are rebuilt into `word <tag> LABEL` pairs when (and only when) the number of words equals the number of tags.
"""

from __future__ import annotations

import random
import re

from sft_new import langdetect as L
from sft_new.compose import A, Rec, Skip

OLD_TURN = re.compile(r"^<\|input_lang\|>(?P<inl>.*?)<task_plan>(?P<plan>.*?)</task_plan>(?:<\|target_lang\|>(?P<tgl>.*?))?<response>(?P<resp>.*?)(?:</s>)?\s*$", re.S)
LANG_ITEM = re.compile(r"<\|?([A-Za-z]{2,5})\|?>|([A-Za-z][A-Za-z ]{2,24})")
UPOS = frozenset("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split())
NER_TAG = re.compile(r"^(?:O|[BI]-[A-Z]+)$")
PLAN_TOK = re.compile(r"<\|?[A-Za-z_]+\|?>")
LABEL_TOKS = {"<intent>", "<|hate|>", "<toxic>", "<sentiment>", "<topic>", "<headline>", "<lang_ID>", "<score>", "<correct>", "<reason>",
              "<answer>", "<title>"}
EXTRA_MAP = {"<pos>": "<POS>", "<POS>": "<POS>", "<ner>": "<NER>", "<|summarize|>": "<summarize>", "<|classify|>": "<classify>",
             "<|translate|>": "<translate>", "<|NER|>": "<NER>", "<|hate|>": "<|hate|>"}


def parse_old(content: str) -> dict | None:
    m = OLD_TURN.match((content or "").strip())
    return m.groupdict() if m else None


def lang_codes(s: str | None) -> list[str]:
    """Every language mentioned in a tag field, in order, as our codes (None for a language we do not support)."""
    out = []
    for tag, name in LANG_ITEM.findall(s or ""):
        out.append(L.code_of(tag or name))
    return out


def plan_of(plan_s: str, extra_labels: bool = True) -> tuple[list[str], str | None]:
    """Old plan string -> (verbs, label token). A label token with no verb means classification."""
    verbs, label = [], None
    for raw in PLAN_TOK.findall(plan_s or ""):
        if raw == "<tag>":                       # the tagging marker is re-derived from the response by normalize_turn
            continue
        t = EXTRA_MAP.get(raw, raw)
        t = A.normalise_verb(t)
        if t in LABEL_TOKS:
            label = label or t
        elif t in A.VALID_VERBS:
            if t not in verbs:
                verbs.append(t)
        else:
            raise Skip("plan_token:" + raw)
    if label == "<headline>" and not verbs:
        verbs = ["<summarize>"]
    if label and not verbs:
        verbs = ["<classify>"]
    if not verbs:
        raise Skip("empty_plan")
    return verbs, label


def check_lang(text: str, candidate: str | None, trust: bool = False) -> tuple[str | None, bool]:
    """-> (code, unsure). With a candidate (stated by the repo or the old tag) the detector only has to *not contradict* it."""
    d = L.detect(text)
    if candidate in L.SUPPORTED_SET:
        sc, _ = L.stop_scores(text)
        if d.code == candidate and d.level == "certain":
            return candidate, False
        if candidate in d.glot or sc.get(candidate, 0.0) >= 0.03:
            return candidate, False
        if trust:                                     # the repo states the language of its text (config name): the detector may not overrule it
            return candidate, False
        if d.level == "none":
            return candidate, True
        return d.code, d.level != "certain"          # the text is clearly something else; believe the detector
    if d.code and d.level != "none":
        return d.code, d.level != "certain"
    return None, False


def split_prompt_text(user: str, n_tags: int, marker: str | None) -> tuple[str, str, list[str]]:
    """The word list a tag stream belongs to, plus the instruction left over: (instruction, text, words)."""
    toks = user.split()
    if marker and marker in user:
        before, after = user.split(marker, 1)
        w = after.split()
        if len(w) >= n_tags:
            return before.strip(" :"), " ".join(w[:n_tags]), w[:n_tags]
    for lo, hi in ((len(toks) - n_tags, len(toks)), (0, n_tags)):          # text at the end (usual) or at the start
        if lo >= 0 and hi <= len(toks) and n_tags > 0:
            w = toks[lo:hi]
            rest = " ".join(toks[:lo] + toks[hi:]).strip(" :")
            return rest, " ".join(w), w
    raise Skip("tags_not_alignable")


_MARKER = re.compile(r"<(?:summarize|NER|classify|translate|clean|identify|diacritize|qa|pos|POS)>\s?")


def convert(user: str, content: str, dataset: str, rng: random.Random, expect_src: str | None = None, task: str | None = None,
            meta: dict | None = None, src_hint: str | None = None, expect_tgt: str | None = None, text_first: bool = False) -> Rec:
    """One old-format (user, assistant) pair -> canonical Rec. Raises Skip when it cannot be made certain."""
    p = parse_old(content)
    if not p:
        raise Skip("grammar")
    user_raw = user
    user = _MARKER.sub("", user)                      # old prompts carry the verb token ("<summarize>**headline**..."); it now lives in the plan only
    verbs, label = plan_of(p["plan"])
    codes = lang_codes(p["inl"])
    resp = (p["resp"] or "").strip()
    if not resp:
        raise Skip("empty_response")
    cand = expect_src or src_hint or (codes[-1] if codes else None)
    meta = dict(meta or {})
    if len(codes) == 2:
        meta["instruction_language"] = codes[0] or "unknown"

    def hints_for(src_code: str) -> list:
        """Languages the old tags name for the user turn: (instruction, text) normally, (text, instruction) when text_first."""
        named = [c for c in codes if c in L.SUPPORTED_SET]
        if src_code not in named:
            named.append(src_code)
        if len(codes) == 2 and codes[0] in L.SUPPORTED_SET and codes[1] in L.SUPPORTED_SET:
            instr = codes[1] if text_first else codes[0]
            return [instr, src_code] if instr != src_code else [src_code]
        return named

    sequence = False
    clean_user = user
    if any(v in verbs for v in ("<NER>", "<POS>")) or p["plan"].strip().lower() in ("<pos>", "<chunk>", "<ner>") or "<chunk>" in p["plan"]:
        tags = resp.split()
        if "<tag>" not in resp:
            if "<NER>" in verbs and not all(NER_TAG.match(t) for t in tags):
                raise Skip("ner_labels_invalid")
            if "<POS>" in verbs and not all(t in UPOS for t in tags):
                raise Skip("pos_labels_invalid")           # e.g. NER tags written under a POS plan in the source
        if "<tag>" in resp:                                   # already word <tag> LABEL
            sequence = True
            text = re.sub(r"\s*<tag>\s*\S+", "", resp)
        else:
            instr, text, words = split_prompt_text(user_raw, len(tags), "<NER>" if "<NER>" in user_raw else None)
            resp = " ".join(f"{w} <tag> {t}" for w, t in zip(words, tags))
            sequence = True
            clean_user = user
            meta["tags_aligned"] = True
        src, unsure = check_lang(text, cand, trust=bool(expect_src))
        if not src:
            raise Skip("lang_input")
        if "<chunk>" in p["plan"]:
            verbs = ["<classify>"]
        return Rec(task=task or ("ner" if "<NER>" in verbs else "pos"), dataset=dataset, src=src, tgt=src, response=resp, plan=verbs,
                   prompt_override=clean_user.strip(), unsure_src=unsure, sequence_labels=sequence, lang_hints=hints_for(src), meta=meta)

    src, unsure_s = check_lang(user, cand, trust=bool(expect_src))
    if not src:
        raise Skip("lang_input")
    tgt_codes = lang_codes(p["tgl"])
    if label or verbs == ["<classify>"]:
        tgt, unsure_t = (tgt_codes[-1] if tgt_codes and tgt_codes[-1] else "eng"), False        # labels are written in English
    elif expect_tgt:
        tgt, unsure_t = expect_tgt, False
    else:
        tgt, unsure_t = check_lang(resp[:700], None)
        if not tgt:
            raise Skip("lang_output")
    if "<translate>" in verbs and src == tgt:
        raise Skip("translate_same_language")
    return Rec(task=task or _task_name(verbs, label), dataset=dataset, src=src, tgt=tgt, response=resp, plan=verbs, label=label,
               prompt_override=user.strip(), unsure_src=unsure_s, unsure_tgt=unsure_t, lang_hints=hints_for(src), meta=meta)


def _task_name(verbs: list[str], label: str | None) -> str:
    if label:
        return {"<intent>": "intent", "<|hate|>": "hate", "<toxic>": "toxicity", "<sentiment>": "sentiment", "<topic>": "topic",
                "<headline>": "headline", "<lang_ID>": "language_id", "<score>": "scoring", "<correct>": "correction"}.get(label, "classify")
    v = verbs[0].strip("<>|").lower()
    return {"chat": "chat", "generate": "generation", "summarize": "summarization", "translate": "translation", "qa": "qa", "code": "code",
            "math": "math", "explain": "explanation", "edit": "editing", "analyze": "analysis", "plan": "planning", "ner": "ner",
            "pos": "pos", "clean": "cleaning", "diacritize": "diacritization", "identify": "language_id"}.get(v, v)
