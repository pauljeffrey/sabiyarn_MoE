"""Artificial (programmatically corrupted) data for two tasks, built from real pretrain text:

  diacritize  real text with tone marks / diacritics / special letters stripped  -> the original text
  clean       real text with typical web/OCR/typing damage injected               -> the original text

Columns: instruction, input, response, language, task, messages, source_dataset  (+ `text` added by add_text.py).
Assistant turn: <|input_lang|><L>[...]<task_plan><diacritize|clean></task_plan><|target_lang|><L><response>TEXT
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sft_new import structured as S

SCHEMA = pa.schema([("instruction", pa.string()), ("input", pa.string()), ("response", pa.string()), ("language", pa.string()),
                    ("task", pa.string()), ("messages", pa.string()), ("source_dataset", pa.string())])

# --------------------------------------------------------------------------------------------- diacritics
LETTER_SUBS = {
    "ewe": {"ɛ": "e", "ɔ": "o", "ɖ": "d", "ƒ": "f", "ɣ": "g", "ŋ": "n", "ʋ": "v", "Ɛ": "E", "Ɔ": "O", "Ɖ": "D", "Ƒ": "F", "Ɣ": "G", "Ŋ": "N", "Ʋ": "V"},
    "twi": {"ɛ": "e", "ɔ": "o", "Ɛ": "E", "Ɔ": "O"},
    "efi": {"ɛ": "e", "ɔ": "o", "ŋ": "n", "ɣ": "g", "Ɛ": "E", "Ɔ": "O", "Ŋ": "N"},
    "fon": {"ɛ": "e", "ɔ": "o", "ɖ": "d", "Ɛ": "E", "Ɔ": "O", "Ɖ": "D"},
    "hau": {"ɓ": "b", "ɗ": "d", "ƙ": "k", "ɠ": "g", "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "’": "'"},
    "wol": {"ñ": "n", "ŋ": "n", "Ñ": "N", "Ŋ": "N"},
    "ibo": {}, "yor": {}, "afr": {}, "sot": {}, "tsn": {}, "nya": {},
}
DI_LANGS = list(LETTER_SUBS)
DI_MARKS = {"yor": "tone marks and underdots", "ibo": "tone marks and dot-below letters", "ewe": "tone marks", "fon": "tone marks",
            "twi": "tone marks", "efi": "tone marks", "hau": "tone marks", "afr": "accents", "sot": "accents and diacritics",
            "tsn": "accents and diacritics", "nya": "diacritics", "wol": "accents"}


def degrade(text: str, lang: str, letters: bool) -> str:
    t = unicodedata.normalize("NFD", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t)
    if letters and LETTER_SUBS[lang]:
        t = t.translate(str.maketrans(LETTER_SUBS[lang]))
    return t


def di_instruction(lang: str, letters: bool, rng: random.Random) -> str:
    n = S.NAMES[lang]
    sub = LETTER_SUBS[lang]
    if letters and sub:
        what = "special letters and diacritics"
    else:
        what = DI_MARKS[lang]
    return rng.choice([f"Restore {what} in this {n} text.", f"This {n} text was typed without {what}. Write it out correctly.",
                       f"Add the missing {what} to this {n} text.", f"Rewrite this {n} text with the correct {what}.",
                       f"Fix the {n} spelling by putting back {what}."])


# --------------------------------------------------------------------------------------------- noise injection
def _spaces(t, r):
    return re.sub(r" ", lambda m: " " * r.choice((1, 1, 1, 2, 3)), t)


def _punct_space(t, r):
    return re.sub(r"([.,;:!?])", lambda m: " " + m.group(1) if r.random() < 0.4 else m.group(1), t)


def _repeat_punct(t, r):
    return re.sub(r"([.!?])", lambda m: m.group(1) * r.choice((1, 2, 3, 4)) if r.random() < 0.3 else m.group(1), t)


def _html(t, r):
    parts = t.split(" ")
    if len(parts) < 4:
        return t
    for _ in range(r.randint(1, 3)):
        i = r.randrange(len(parts))
        parts[i] = r.choice(["<b>", "<br>", "<p>", "&nbsp;", "&amp;", "</div>", "<span>"]) + parts[i]
    return " ".join(parts)


def _mojibake(t, r):
    try:
        bad = t.encode("utf-8").decode("latin-1")
    except UnicodeError:
        return t
    return bad if any(ord(c) > 127 for c in t) else t


def _ocr(t, r):
    table = {"l": "1", "I": "l", "O": "0", "o": "0", "e": "c", "m": "rn", "h": "b", "i": "j"}
    return "".join(table[c] if c in table and r.random() < 0.04 else c for c in t)


def _typos(t, r):
    cs = list(t)
    for _ in range(max(1, len(cs) // 30)):
        i = r.randrange(len(cs))
        op = r.choice(("del", "dup", "swap"))
        if op == "del" and cs[i].isalpha():
            cs[i] = ""
        elif op == "dup" and cs[i].isalpha():
            cs[i] = cs[i] * 2
        elif op == "swap" and i + 1 < len(cs) and cs[i].isalpha() and cs[i + 1].isalpha():
            cs[i], cs[i + 1] = cs[i + 1], cs[i]
    return "".join(cs)


def _case(t, r):
    return t.upper() if r.random() < 0.3 else t.lower() if r.random() < 0.6 else "".join(c.upper() if r.random() < 0.15 else c for c in t)


def _junk(t, r):
    junk = r.choice(["Share on Facebook", "Click here to read more", "Copyright 2024 All rights reserved", "https://example.com/page?id=7",
                     "Subscribe to our newsletter", "[1]", "Advertisement", "Read also:", "Login | Register"])
    return (junk + " " + t) if r.random() < 0.5 else (t + " " + junk)


def _linebreaks(t, r):
    return re.sub(r" ", lambda m: "\n" if r.random() < 0.12 else " ", t)


def _nospace(t, r):
    return re.sub(r" ", lambda m: "" if r.random() < 0.08 else " ", t)


def _dupword(t, r):
    ws = t.split(" ")
    out = []
    for w in ws:
        out.append(w)
        if r.random() < 0.06:
            out.append(w)
    return " ".join(out)


NOISE = [_spaces, _punct_space, _repeat_punct, _html, _mojibake, _ocr, _typos, _case, _junk, _linebreaks, _nospace, _dupword]
# which corruptions are reversible by a plain clean-up (the response is the original text): drop none, but never let the
# junk line or lower-casing be the only change on very short inputs
CLEAN_INSTR = ["Clean up this text: fix the spacing and typos, remove stray markup and unrelated lines, and return only the cleaned text.",
               "The text below is messy. Rewrite it cleanly.", "Remove the noise from this text and give back the clean version.",
               "Fix the formatting and spelling errors in this text. Output only the corrected text.",
               "Clean this text (stray symbols, odd spacing, leftover markup, repeated words) and return the result.",
               "Tidy up the following text so it reads normally."]


def noisy(text: str, rng: random.Random) -> str:
    out = text
    for fn in rng.sample(NOISE, rng.choice((1, 2, 2, 3, 4))):
        out = fn(out, rng)
    return out


# --------------------------------------------------------------------------------------------- builders
def snippets(lang: str, token: str, rng: random.Random, n: int, lo: int, hi: int):
    from africomet_translation.text_utils import good_sentence, normalize, split_sentences
    got = 0
    for t in S.pretrain_docs(lang, token, 30000):
        sents = [normalize(s) for s in split_sentences(t)]
        i = 0
        while i < len(sents):
            k = rng.choice((1, 1, 2, 3))
            sn = " ".join(sents[i:i + k])
            i += k + rng.choice((0, 1, 2))
            nw = len(sn.split())
            if lo <= nw <= hi and good_sentence(sn, min_words=3, max_words=hi + 20, require_terminal=False, lenient=True):
                yield sn
                got += 1
                if got >= n:
                    return


def mk(task, instruction, text, response, lang, assistant):
    user = f"{instruction}\n\n{text}"
    return {"instruction": instruction, "input": text, "response": response, "language": lang, "task": task,
            "messages": json.dumps([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], ensure_ascii=False),
            "source_dataset": "BeardedMonster/pretrain-new (artificially corrupted)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["diacritize", "clean"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=3000)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--langs", nargs="*")
    a = ap.parse_args(argv)
    HERE = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(HERE))
    from africomet_translation.build import hf_token
    token = hf_token()
    out = Path(a.out)
    name = f"artificial-{a.task}"
    o = S.Out(out, name, schema=SCHEMA)
    stats = collections.defaultdict(int)
    rng = random.Random(f"artificial-{a.task}")
    langs = a.langs or (DI_LANGS if a.task == "diacritize" else sorted(S.NAMES.keys() - {"eng"}))
    for l in langs:
        k = 0
        for sn in snippets(l, token, rng, a.per_lang * 3, 5 if a.task == "diacritize" else 8, 60):
            if a.task == "diacritize":
                letters = rng.random() < 0.5
                bad = degrade(sn, l, letters)
                if bad == sn or sum(x != y for x, y in zip(sn, bad)) + abs(len(sn) - len(bad)) < 3:
                    continue
                ins = di_instruction(l, letters, rng)
                reply = S.turn(l, l, ["<diacritize>"], sn)
                o.add(mk("diacritize", ins, bad, sn, l, reply))
            else:
                bad = noisy(sn, rng)
                if bad == sn:
                    continue
                ins = rng.choice(CLEAN_INSTR)
                try:
                    reply = S.turn(l, l, ["<clean>"], sn)
                except Exception:
                    continue
                o.add(mk("clean", ins, bad, sn, l, reply))
            k += 1
            if k >= a.per_lang:
                break
        stats[l] = k
        print(a.task, l, k, file=sys.stderr, flush=True)
    o.flush()
    (out / "stats").mkdir(parents=True, exist_ok=True)
    (out / "stats" / f"{name}.json").write_text(json.dumps(dict(stats), indent=1))
    if a.push and o.total:
        S.push(out, name, token)


if __name__ == "__main__":
    main()
