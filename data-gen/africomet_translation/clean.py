"""Stage A2: clean candidate windows and keep only the coherent, human-looking ones.

Measured on the raw extraction: web Yoruba/Igbo (mC4, HPLT, GlotCC) is mostly machine-translated English (median
"looks like curated prose" score 0.02), with verse/section numbers glued to the front, English words left in, and
Title-Case spam. So, in order:

  1. CLEAN each sentence: strip leading verse / section numbers ("12:20 ", "7.1 ", "14 Josef ..."), scripture
     references "(Ìṣí 14: 9-11 NKJ)", bullets, spacing before punctuation, runs of punctuation.
  2. REJECT windows by rules: mojibake, symbols/emoji, too few letters, many digits, absurd word lengths, repetition,
     Title-Case spam, English function-word leakage (non-English languages other than Pidgin).
  3. SCORE web windows with a per-language hashed word 1-2-gram classifier trained to tell curated prose (news,
     encyclopedia, FLORES, SMOL) from random web text of the same language, and keep those above `threshold`.
     Curated, news, encyclopedia and Bible rows are not scored (they are the positives / a fixed register); they only
     get steps 1 and 2.
Output keeps the stage-A schema, with hashes recomputed on the cleaned sentences so stage B de-duplicates correctly.
"""

from __future__ import annotations

import json
import pickle
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

SCORED_KINDS = {"web"}
# Hand-translated or edited human text only. Encyclopedia is excluded: low-resource Wikipedias (Igbo especially) are
# themselves partly machine-translated, and a classifier trained on them cannot tell MT from prose.
POSITIVE_KINDS = {"curated", "news"}
# Languages where web text is dominated by machine-translated English, so a "looks like curated prose" classifier is
# trustworthy. In Hausa and Twi it rejected genuine news / religious prose (register mismatch), so those use rules only.
CLASSIFIED = ("yor", "ibo", "eng")
THRESHOLD = 0.5

_UPPER = "A-ZÀ-ÖØ-ÞẸỌṢƊƁƘƐƆ"
_LEAD_NUM = re.compile(rf"^\s*(?:\d{{1,3}}\s*[:.]\s*)*\d{{1,3}}\s*[:.)]?\s+(?=[{_UPPER}])")     # "12:20 ", "7.1 ", "14 ", "3) "
_SCRIPTURE = re.compile(
    r"\s*[\(\[][^()\[\]]{0,40}?\b\d{1,3}\s?[:.]\s?\d{1,3}(?:\s?[-–,]\s?\d{1,3})*[^()\[\]]{0,25}[\)\]]"   # (Ìṣí 14: 9-11 NKJ)
    r"|\s*[—–-]\s?(?:[1-3]\s)?[A-Z][a-zà-ÿ]{1,7}\.?\s\d{1,3}\s?:\s?\d{1,3}(?:[-–]\d{1,3})?\.?$"       # —Heb. 13:15
)
_LEAD_FRAG = re.compile(r"^\s*\d{1,3}\s*:\s*\d{1,3}(?:\s*,\s*\d{1,3})*\)\s*")        # "6:2, 3) Nanso ..."  (tail of a citation)
_TRAIL_CITE = re.compile(
    r"\s*[—–-]\s?(?:[1-3]\s)?[^\W\d_][^\W\d_.]{1,25}(?:\s[^\W\d_][^\W\d_.]{1,25}){0,2}\.?\s\d{1,3}\s?:\s?\d{1,3}(?:\s?[-–,]\s?\d{1,3})*\.?$")
_MIXED = re.compile(r"\b(?=\w*\d)(?=\w*[^\W\d_])\w+\b")                                    # letters+digits inside one word
_BULLET = re.compile(r"^\s*[•·▪►▶■□●○◦‣⁃\-–—*#>]+\s*")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?%)\]»”])")
_OPEN_SPACE = re.compile(r"([(\[«“])\s+")
_PUNCT_RUN = re.compile(r"([.!?,;:])\1{2,}")
_ELLIPSIS_SP = re.compile(r"(?:\.\s){2,}\.?")
_MOJIBAKE = re.compile(r"Ã[\u0080-\u00bf]|â€|Â[\u0080-\u00bf]|ï¿½|Ð[\u00b0-\u00bf]|Ñ[\u0080-\u00bf]|\ufffd")
_EN_STOP = frozenset(
    "the and of is are was were with that this from your our they will have has been their there which also more about "
    "would should could because when what who how all any these those into than then them".split())
_LET = r"(?:[^\W\d_]|[\u0300-\u036f])"
_WORD = re.compile(rf"{_LET}+(?:['’\-]{_LET}+)*", re.UNICODE)


def clean_sentence(s: str, lang: str) -> str:
    s = unicodedata.normalize("NFC", s).strip()
    s = _BULLET.sub("", s)
    if lang != "eng":
        s = _LEAD_NUM.sub("", s)
    s = _SCRIPTURE.sub("", s)
    s = _LEAD_FRAG.sub("", s)
    if s.rstrip(".").endswith(tuple("0123456789")):
        s = re.sub(r"\.{2,}$", ".", _TRAIL_CITE.sub(".", s))
    s = re.sub(r"(?<=[^\W\d_,.;:!?])\+(?=[\s,.;:!?]|$)|(?<=[,.;:!?])\+", "", s)       # footnote markers "akpakọrịta,+ bụ́"
    s = _ELLIPSIS_SP.sub("...", s)
    s = _PUNCT_RUN.sub(lambda m: m.group(1) * (3 if m.group(1) == "." else 1), s)
    s = _SPACE_PUNCT.sub(r"\1", s)
    s = _OPEN_SPACE.sub(r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def reject_reason(text: str, lang: str, kind: str = "web") -> str | None:
    """None if the window passes the rule-based checks, else a short reason (used for the stats report)."""
    n_chars = len(text)
    if n_chars < 20:
        return "short"
    if _MOJIBAKE.search(text):
        return "mojibake"
    # combining tone marks (Fon, Yoruba, Ewe...) are not isalpha() but are part of the letters
    letters = sum(c.isalpha() or unicodedata.category(c) == "Mn" for c in text)
    if letters / n_chars < 0.70:
        return "few_letters"
    if sum(c.isdigit() for c in text) / n_chars > 0.12:
        return "digits"
    if any(unicodedata.category(c) in ("So", "Sk", "Cs", "Co") for c in text):
        return "symbols"
    words = _WORD.findall(text)
    nw = len(words)
    if nw < 3:
        return "few_words"
    mean_len = sum(map(len, words)) / nw
    if not 2.4 <= mean_len <= 13:
        return "word_length"
    mixed = len(_MIXED.findall(text))
    if mixed >= 3 and mixed / nw > 0.08:
        return "mixed_digits"
    low = [w.lower() for w in words]
    if nw >= 12 and len(set(low)) / nw < 0.45:
        return "repetition"
    if any(low[i] == low[i + 1] == low[i + 2] for i in range(nw - 2)):
        return "repetition"
    if sum(not c.isalnum() and not c.isspace() for c in text) / n_chars > 0.18:
        return "punctuation"
    if lang != "eng" and kind == "web" and nw >= 8:
        # Title-Case spam: capitalised words that do not start a sentence
        toks, caps, prev_end = text.split(), 0, True
        for tk in toks:
            if not prev_end and tk[:1].isupper():
                caps += 1
            prev_end = tk.rstrip("\"'”’)]").endswith((".", "!", "?", "…"))
        if caps / max(1, len(toks)) > 0.30:
            return "title_case"
    if lang not in ("eng", "pcm") and nw >= 6:
        if sum(w in _EN_STOP for w in low) / nw > 0.08:
            return "english_leak"
    return None


# ------------------------------------------------------------------------------------------------ classifier

def _vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(n_features=2**20, alternate_sign=False, ngram_range=(1, 2), norm="l2")


def english_positive(source: str) -> bool:
    return source.startswith(("wikimedia/wikipedia", "HuggingFaceFW/fineweb-edu", "masakhane/masakhanews"))


def train_model(out: Path, lang: str, per_class: int = 60_000, seed: int = 0) -> dict | None:
    """Positives: curated prose of this language. Negatives: its raw web windows. English: wiki/edu vs C4."""
    import pyarrow.parquet as pq
    from sklearn.linear_model import SGDClassifier
    rng = random.Random(seed)
    pos, neg = [], []
    for f in sorted((out / "windows" / lang).glob("*.parquet")):
        for r in pq.read_table(f, columns=["text", "kind", "source"]).to_pylist():
            if lang == "eng":
                (pos if english_positive(r["source"]) else neg).append(r["text"])
            elif r["kind"] in POSITIVE_KINDS:
                pos.append(r["text"])
            elif r["kind"] in SCORED_KINDS:
                neg.append(r["text"])
        if lang == "eng" and len(pos) > 4 * per_class and len(neg) > 4 * per_class:
            break
    if lang not in CLASSIFIED or len(pos) < 3000 or len(neg) < 10_000:
        return None
    pos, neg = rng.sample(pos, min(len(pos), per_class)), rng.sample(neg, min(len(neg), per_class))
    hold = pos[: max(100, len(pos) // 10)]
    pos = pos[len(hold):]
    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]                                       # balanced, so 0.5 is the natural boundary
    hv = _vectorizer()
    clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=15, random_state=seed).fit(
        hv.transform(pos + neg), np.array([1] * n + [0] * n))
    held = clf.predict_proba(hv.transform(hold))[:, 1]
    return {"clf": clf, "n_pos": n, "n_neg": n, "heldout_pass": float((held >= THRESHOLD).mean())}


def load_models(out: Path) -> dict[str, dict]:
    d = out / "clean_models"
    return {p.stem: pickle.loads(p.read_bytes()) for p in d.glob("*.pkl")}


# ------------------------------------------------------------------------------------------------ English leakage

_ASCII_WORD = re.compile(r"[a-z]{4,}")


def build_lexicon(out: Path, lang: str, top_n: int = 40_000, min_clean: int = 3) -> dict | None:
    """English content words that this language's own clean text (news / curated / Bible) does not use, and the
    English-word share that 99.5% of that clean text stays under (so foreign names in real news are tolerated)."""
    import pyarrow.parquet as pq
    if lang in ("eng", "pcm"):
        return None
    eng = Counter()
    for f in sorted((out / "windows" / "eng").glob("*.parquet"))[:3]:
        for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()[:150_000]:
            eng.update(_ASCII_WORD.findall(t.lower()))
    top = {w for w, _ in eng.most_common(top_n)}
    clean_vocab, clean_texts = Counter(), []
    for f in sorted((out / "windows" / lang).glob("*.parquet")):
        for r in pq.read_table(f, columns=["text", "kind"]).to_pylist():
            if r["kind"] in ("news", "curated", "bible"):
                clean_vocab.update(_ASCII_WORD.findall(r["text"].lower()))
                if r["kind"] != "bible" and len(clean_texts) < 30_000:
                    clean_texts.append(r["text"])
    lex = frozenset(w for w in top if clean_vocab[w] < min_clean)
    shares = [english_share(t, lex) for t in clean_texts]
    thr = max(0.05, float(np.quantile(shares, 0.995))) if shares else 0.15
    return {"lexicon": lex, "threshold": thr, "n_clean_ref": len(clean_texts)}


def english_share(text: str, lex) -> float:
    words = _WORD.findall(text.lower())
    return sum(w in lex for w in words) / max(1, len(words))


# ------------------------------------------------------------------------------------------------ apply

def clean_file(path: str, lang: str, out_dir: str, threshold: float = THRESHOLD) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .text_utils import sent_hash, split_sentences
    out = Path(out_dir)
    model = None
    mp = out / "clean_models" / f"{lang}.pkl"
    if mp.exists():
        model = pickle.loads(mp.read_bytes())["clf"]
    hv = _vectorizer()
    lex = None
    lp = out / "clean_models" / f"{lang}.lex"
    if lp.exists():
        lex = pickle.loads(lp.read_bytes())
    t = pq.read_table(path)
    rows = t.to_pylist()
    stats = Counter()
    keep = []
    for r in rows:
        stats["in"] += 1
        sents = [clean_sentence(s, lang) for s in split_sentences(r["text"])]
        sents = [s for s in sents if s]
        if not sents or len(sents) > 5:
            stats["rej:sentences"] += 1
            continue
        text = " ".join(sents)
        why = reject_reason(text, lang, r["kind"])
        if why:
            stats["rej:" + why] += 1
            continue
        if lex is not None and r["kind"] != "bible" and english_share(text, lex["lexicon"]) > lex["threshold"]:
            stats["rej:english_words"] += 1
            continue
        r["_changed"] = text != r["text"]
        r["text"], r["n_sentences"], r["hashes"] = text, len(sents), [sent_hash(s) for s in sents]
        keep.append(r)
    if model is not None:
        idx = [i for i, r in enumerate(keep) if r["kind"] in SCORED_KINDS and (lang != "eng" or not english_positive(r["source"]))]
        if idx:
            scores = model.predict_proba(hv.transform([keep[i]["text"] for i in idx]))[:, 1]
            drop = {i for i, s in zip(idx, scores) if s < threshold}
            stats["rej:classifier"] += len(drop)
            keep = [r for i, r in enumerate(keep) if i not in drop]
    changed = [r for r in keep if r.pop("_changed")]
    if changed:
        from .extract import token_lengths
        lens = token_lengths([r["text"] for r in changed])
        for r, n in zip(changed, lens):
            r["tokens"] = n
    keep = [r for r in keep if r["tokens"] <= 256]
    stats["out"] = len(keep)
    dest = out / "clean" / lang / Path(path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = {k: [r[k] for r in keep] for k in ("text", "n_sentences", "tokens", "hashes", "source", "license", "kind")}
    from .extract import _schema
    pq.write_table(pa.table(cols, schema=_schema()), dest, compression="zstd")
    return {"file": Path(path).name, "lang": lang, **stats}
