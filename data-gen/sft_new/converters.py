"""One converter per source dataset: stream rows -> yield `Rec` (or `Skip(reason)`), nothing else.

Every converter obeys the same rules:
  * the languages of the input text and of the response are CHECKED (sft_new/langdetect.py), never taken on trust from a column name
    unless the repo states them (a config/column per language); a language that is not one of our 23 drops the row;
  * a translation pair is emitted in BOTH directions;
  * instruction wording is drawn from the template pools (translation bank in africomet_translation/instructions.py, the other tasks in
    sft_new/templates.py), in a randomly chosen instruction language, before/after the input at random;
  * degenerate machine output (repeated n-grams) is dropped.

Registry: CONVERTERS[name] = (function, one-line description). Every function has the signature (rng, limit) -> Iterator[Rec | Skip].
"""

from __future__ import annotations

import itertools
import json
import random
import re
from typing import Callable, Iterable, Iterator

from africomet_translation.instructions import make_instruction
from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import A, Late, Rec, Skip
from sft_new import old_format as O
from sft_new import tagged_parse as TP
from sft_new.textutil import aligned, blank, chunks, clean, degenerate, first_sentence, length_ratio_ok, real, recase, rep_ratio, strip_calc

CONVERTERS: dict[str, tuple[Callable, str]] = {}


def register(name: str, desc: str):
    def deco(fn):
        CONVERTERS[name] = (fn, desc)
        return fn
    return deco


def batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while True:
        b = list(itertools.islice(it, n))
        if not b:
            return
        yield b


def il_for(rng: random.Random, prefer: str | None = None) -> str:
    """Instruction language for the translation bank (7 languages), English-heavy."""
    from africomet_translation.instructions import instruction_languages
    have = instruction_languages()
    if prefer in have and rng.random() < 0.3:
        return prefer
    return "eng" if rng.random() < 0.5 else rng.choice(have)


def _expect(d: L.Detection, code: str):
    """Detector verdict against a language the repo states: None (contradicted) or unsure flag."""
    if d.code == code or code in d.glot:
        return not (d.code == code and d.level == "certain")
    return None


def translation(a: str, b: str, la: str, lb: str, dataset: str, rng: random.Random, ua=False, ub=False, both=True, meta=None) -> list[Rec]:
    out = []
    for x, y, s, t, u1, u2, direction in ((a, b, la, lb, ua, ub, "forward"), (b, a, lb, la, ub, ua, "backward")) if both else ((a, b, la, lb, ua, ub, "forward"),):
        ins = make_instruction(il_for(rng, s), s, t, rng)
        out.append(Rec("translation", dataset, s, t, y, input=x, instruction=ins.text, position=ins.position, plan=["<translate>"],
                       unsure_src=u1, unsure_tgt=u2, meta={**(meta or {}), "instruction_language": ins.language, "direction": direction}))
    return out


def pair_stream(rows: Iterable[dict], ka: str, kb: str, la: str, lb: str, dataset: str, rng: random.Random, limit: int | None = None,
                max_chars: int = 3600, meta=None, both=True) -> Iterator:
    """Aligned two-column translation rows with the languages stated by the repo; both sides verified, both directions emitted."""
    n = 0
    for chunk in batched(rows, 256):
        pairs = [(clean(r.get(ka)), clean(r.get(kb))) for r in chunk]
        keep = []
        for a, b in pairs:
            if not a or not b:
                yield Skip("empty")
            elif len(a) > max_chars or len(b) > max_chars or len(a) < 8 or len(b) < 8:
                yield Skip("length")
            elif not length_ratio_ok(a, b):
                yield Skip("length_ratio")
            elif degenerate(a) or degenerate(b):
                yield Skip("repetitive")
            elif not aligned(a, b):
                yield Skip("misaligned_names_numbers")
            else:
                keep.append((a, b))
        if not keep:
            continue
        da, db = L.detect_many([a[:1500] for a, _ in keep]), L.detect_many([b[:1500] for _, b in keep])
        for (a, b), x, y in zip(keep, da, db):
            ua, ub = _expect(x, la), _expect(y, lb)
            if ua is None or ub is None:
                yield Skip("language_mismatch")
                continue
            for r in translation(a, b, la, lb, dataset, rng, ua, ub, both=both, meta=meta):
                yield r
            n += 1
        if limit and n >= limit:
            return


def _rows(repo: str, limit: int | None, **kw):
    return hubio.iter_repo(repo, limit=limit, **kw)


# ============================================================ translation datasets

@register("english_to_igbo", "Aletheia-ng/english_to_igbo -> eng<->ibo translation")
def english_to_igbo(rng, limit=None):
    yield from pair_stream(_rows("Aletheia-ng/english_to_igbo", limit), "English", "Igbo", "eng", "ibo", "Aletheia-ng/english_to_igbo", rng, limit)


@register("webscraped_igbo", "Aletheia-ng/WebScraped_Igbo_To_Eng -> eng<->ibo translation")
def webscraped(rng, limit=None):
    yield from pair_stream(_rows("Aletheia-ng/WebScraped_Igbo_To_Eng", limit), "english", "igbo", "eng", "ibo", "Aletheia-ng/WebScraped_Igbo_To_Eng", rng, limit)


@register("efik_bible", "Aletheia-ng/English_efik_bible_translation -> eng<->efi translation (Bible)")
def efik_bible(rng, limit=None):
    yield from pair_stream(_rows("Aletheia-ng/English_efik_bible_translation", limit), "english", "efik", "eng", "efi", "Aletheia-ng/English_efik_bible_translation", rng, limit)


# Directions that AfriCOMET-QE measured as clearly unfaithful in the Aya translations (sft_new/quality_check.py, 60-80 pairs each): English <-> Hausa
# scores 0.32 with 93-95% of pairs under the 0.46 failure line, against 5-6% for the control datasets. Both directions are skipped.
AYA_SKIP = {("eng", "hau"), ("hau", "eng")}


def _multi(repo: str, cols: dict[str, str], per_row: int, dataset: str, rng, limit, skip_pairs=frozenset()):
    """Rows holding the same text in several of our languages: pick `per_row` random ordered pairs per row."""
    langs = list(cols)
    n = 0
    for chunk in batched(_rows(repo, limit), 128):
        texts = {c: [clean(r.get(k)) for r in chunk] for c, k in cols.items()}
        det = {c: L.detect_many([t[:1500] for t in v]) for c, v in texts.items()}
        for i in range(len(chunk)):
            ok = {}
            for c in langs:
                t, d = texts[c][i], det[c][i]
                if len(t) < 8 or degenerate(t):
                    continue
                u = _expect(d, c)
                if u is not None:
                    ok[c] = (t, u)
            if len(ok) < 2:
                yield Skip("fewer_than_two_languages")
                continue
            pairs = [(x, y) for x in ok for y in ok if x != y and (x, y) not in skip_pairs and aligned(ok[x][0], ok[y][0])]
            rng.shuffle(pairs)
            for x, y in pairs[:per_row]:
                for r in translation(ok[x][0], ok[y][0], x, y, dataset, rng, ok[x][1], ok[y][1], both=False):
                    yield r
            n += 1
        if limit and n >= limit:
            return


@register("aya_interlanguage", "Aletheia-ng/aya_interlanguage_translate (yoruba, ibo, hausa + english) -> interlanguage translation, 3 random directions per row")
def aya_inter(rng, limit=None):
    yield from _multi("Aletheia-ng/aya_interlanguage_translate", {"yor": "yoruba", "ibo": "ibo", "hau": "hausa"}, 3, "Aletheia-ng/aya_interlanguage_translate", rng, limit, AYA_SKIP)


@register("aya_translate", "Aletheia-ng/aya-translate (english, yoruba, ibo, hausa) -> translation, 3 random directions per row")
def aya_translate(rng, limit=None):
    yield from _multi("Aletheia-ng/aya-translate", {"eng": "english", "yor": "yoruba", "ibo": "ibo", "hau": "hausa"}, 3, "Aletheia-ng/aya-translate", rng, limit, AYA_SKIP)


@register("gsm8k_translate", "Aletheia-ng/gsm8k{,2,3} questions as interlanguage translation")
def gsm_translate(rng, limit=None):
    cols = {c: f"question_{c if c != 'ibo' else 'ibo'}" for c in ("eng", "pcm", "yor", "ibo", "hau")}
    for repo in ("Aletheia-ng/gsm8k", "Aletheia-ng/gsm8k2", "Aletheia-ng/gsm8k3"):
        yield from _multi(repo, cols, 2, repo, rng, limit)


# ============================================================ tagged-string datasets (naija, naija2, tds)

FILE_FAMILY: str | None = None           # exact source family (combine.py runs one family per step: converter@family)
FILE_MATCH: str | None = None            # debugging aid (samples --match): only files whose path contains this
SPLIT_SUFFIX = re.compile(r"-(train|test|validation|dev|devtest)\.parquet$")


def tagged_files(repo: str, include: str, train_only: bool = True) -> list[str]:
    """Files of a naija-style repo: `include` = "tagged" (everything that is not a *_monolingual file) or "monolingual". Evaluation splits
    (test / dev / devtest) are left out: these repos bundle benchmark data (FLORES, AfroLingu, ...) and training on a benchmark's own evaluation
    split would contaminate it. That is about 1.4% of the tagged bytes; validation splits (about 1.5%) are used."""
    out = []
    for p, _ in hubio.list_files(repo):
        mono = "_monolingual" in p
        if (include == "monolingual") != mono:
            continue
        m = SPLIT_SUFFIX.search(p)
        if train_only and m and m.group(1) not in ("train", "validation"):
            continue                       # test / dev / devtest are evaluation splits (~1.4% of the tagged bytes); validation is kept
        if FILE_MATCH and FILE_MATCH not in p:
            continue
        if FILE_FAMILY and TP.family_of(p) != FILE_FAMILY:
            continue
        out.append(p)
    return out


def late_docs(texts: list[str], source: str, min_chars: int = 150) -> Iterator:
    """Plain texts -> Late(text, lang, source) when the language is one of ours (certain or likely); cleaned with the pretraining rules."""
    from pretrain_new import docclean
    from sft_new.compose import Late
    texts = [t for t in (clean(x) for x in texts) if len(t) >= min_chars]
    for t, d in zip(texts, L.detect_many([x[:1500] for x in texts])):
        if d.code is None or d.level == "none":
            yield Skip("late:language_unsupported_or_undetermined")
            continue
        c = docclean.clean_document(t, d.code)
        if c is None:
            yield Skip("late:cleaned_out")
        else:
            yield Late(c, d.code, source)


def tagged_stream(repo: str, rng, limit, include: str = "tagged", match=None, min_quality: dict | None = None) -> Iterator:
    n = 0
    files = [f for f in tagged_files(repo, include) if not match or match in f] if repo.startswith("BeardedMonster") else None
    if files is None:                                              # a single-table repo such as tds
        files = [p for p, _ in hubio.list_files(repo)]
    for f in files:
        family = TP.family_of(f)
        if include != "monolingual" and any(x in family for x in TP.SKIP_FAMILIES):
            yield Skip("family_dropped_on_purpose")                  # not even read
            continue
        prior_state: dict = {}                                       # per file: which language its texts are (for short texts)
        for chunk in batched(hubio.iter_file(repo, f, columns=None), 256):
            texts = []
            for r in chunk:
                if min_quality and any((r.get(k) is None or float(r.get(k)) < v) for k, v in min_quality.items()):
                    yield Skip("quality_below_threshold")
                    continue
                if blank(r.get("text")):
                    texts.append(r["text"])
            if include == "monolingual":
                yield from late_docs(texts, family)
                n += len(texts)
            else:
                recs, left, stats = TP.process(texts, f"{repo}:{family}", rng, family=family, prior_state=prior_state)
                for k, v in stats.items():
                    for _ in range(v):
                        yield Skip(k)
                for r in recs:
                    yield r
                pend = [TP.plain_text(raw) for raw, why in left if TP.task_uncertain(why)]
                if pend:
                    yield from late_docs(pend, family + ":task_uncertain")
                n += len(recs)
            if limit and n >= limit:
                return


@register("naija", "BeardedMonster/naija: every tagged family (translation, prompt/response, summarisation, headline, sentiment/topic/intent, math) from the TRAIN files; language detected from the text, not the file name")
def naija(rng, limit=None):
    yield from tagged_stream("BeardedMonster/naija", rng, limit)


@register("naija2", "BeardedMonster/naija2: same families (plus MMLU dev with letters, MathScale, grade-school-math); Math-Shepherd dropped")
def naija2(rng, limit=None):
    yield from tagged_stream("BeardedMonster/naija2", rng, limit)


@register("naija_late", "BeardedMonster/naija *_monolingual files -> pretrain-new/late/<lang> (plain text, language detected)")
def naija_late(rng, limit=None):
    yield from tagged_stream("BeardedMonster/naija", rng, limit, include="monolingual")


@register("naija2_late", "BeardedMonster/naija2 *_monolingual files -> pretrain-new/late/<lang> (Inkuba-Mono, W2C, HPLT, C4, ...)")
def naija2_late(rng, limit=None):
    yield from tagged_stream("BeardedMonster/naija2", rng, limit, include="monolingual")


@register("tds", "Aletheia-ng/tds: <translate> A <lang> B, gated on its africomet_score >= 0.55")
def tds(rng, limit=None):
    yield from tagged_stream("Aletheia-ng/tds", rng, limit, min_quality={"africomet_score": 0.55})


# ============================================================ the older SFT repos (re-derived into the canonical grammar)

def _msgs(r: dict):
    m = r.get("chat_data") or r.get("data")
    if isinstance(m, str):
        m = json.loads(m)
    if not isinstance(m, list):                   # a null/blank row: no conversation at all
        return "", "", 0
    user = next((blank(x.get("content")) for x in m if x.get("role") == "user"), "")
    asst = next((blank(x.get("content")) for x in m if x.get("role") == "assistant"), "")
    return user, asst, len(m)


_SKIP_TOKENS = {"cnn", "extra", "all", "data", "train", "xlsum", "latn"}


def _config_lang(path: str) -> str | None:
    """'data/hau_hate-00000-of-00001.parquet' / 'Hausa_ner' / 'yor_Latn_topic' / 'cnn_eng_summarization' -> language code;
    'UNSUPPORTED' when the file is about a language we do not carry (knc, nso, fuv ...); None when the name carries no language."""
    stem = path.rsplit("/", 1)[-1].split("-0000")[0].split(".")[0]
    for tok in stem.split("_"):
        if tok.lower() in _SKIP_TOKENS:
            continue
        c = L.code_of(tok)
        if c:
            return c
        if re.fullmatch(r"[A-Za-z]{3}|[A-Z][a-z]{3,}", tok) and tok.lower() not in ("task", "tagging", "topic", "intent", "hate", "pos", "ner", "instruct"):
            return "UNSUPPORTED"
        return None
    return None


SIB200 = {"0": "entertainment", "1": "geography", "2": "health", "3": "politics", "4": "science/technology", "5": "sports", "6": "travel"}   # alphabetical, checked on 3 labelled sentences


def _sib_names(asst: str) -> str:
    """The *_Latn_topic files (SIB-200) answer with a class index; give it its name."""
    return re.sub(r"(<response>)(\d)(</s>)?\s*$", lambda m: m.group(1) + SIB200.get(m.group(2), m.group(2)), asst.strip())


def _old(repo: str, rng, limit, tasks: dict[str, str] | None = None, skip_configs=()):
    n = 0
    for r in _rows(repo, None):
        if limit and n >= limit:
            return
        f = r["__file"]
        cfg = _config_lang(f)
        if any(s in f for s in skip_configs):
            yield Skip("config_skipped")
            continue
        if cfg == "UNSUPPORTED":
            yield Skip("unsupported_language_config")
            continue
        user, asst, nmsg = _msgs(r)
        if not user or not asst:
            yield Skip("blank_row")                     # the repo contains empty rows; they are not counted against `limit`
            continue
        n += 1
        if "_Latn_topic" in f:
            asst = _sib_names(asst)
        if cfg is None and "tasks" not in repo:
            yield Skip("config_without_language")
            continue
        try:
            target = L.code_of(str(r.get("target_lang") or "")) if r.get("target_lang") else None
            gen_task = "tasks" in repo and ("summar" in f.lower() or "title" in f.lower())
            yield O.convert(user, asst, repo, rng, expect_src=cfg if "tasks" not in repo else (target or cfg),
                            expect_tgt=(target if gen_task else None), text_first=("tasks" in repo),
                            meta={"file": f.rsplit("/", 1)[-1], **({"turns_in_source": nmsg} if nmsg > 2 else {})})
        except Skip as e:
            yield e
        n += 1


@register("old_hate", "BeardedMonster/hate (AfriHate-style) -> <classify><|hate|>")
def old_hate(rng, limit=None):
    yield from _old("BeardedMonster/hate", rng, limit)


@register("old_intent", "BeardedMonster/intent -> <classify><intent>")
def old_intent(rng, limit=None):
    yield from _old("BeardedMonster/intent", rng, limit)


@register("old_topic", "BeardedMonster/topic (news topics by name; SIB-200 *_Latn_topic files: class index -> name; unsupported languages dropped)")
def old_topic(rng, limit=None):
    yield from _old("BeardedMonster/topic", rng, limit)


@register("old_pos", "BeardedMonster/pos -> <POS><tag> word/tag pairs (NER tags found under POS are dropped)")
def old_pos(rng, limit=None):
    yield from _old("BeardedMonster/pos", rng, limit)


@register("old_tasks", "BeardedMonster/tasks (NER, sentiment, summarisation, headline, instruct)")
def old_tasks(rng, limit=None):
    yield from _old("BeardedMonster/tasks", rng, limit, skip_configs=("data/all",))


@register("old_summarization", "BeardedMonster/summarization (English; CNN/DM etc.)")
def old_summ(rng, limit=None):
    yield from _old("BeardedMonster/summarization", rng, limit)


@register("old_african_sft", "BeardedMonster/african_sft (language tags are unreliable -> re-detected)")
def old_african(rng, limit=None):
    for r in _rows("BeardedMonster/african_sft", limit):
        user, asst, nmsg = _msgs(r)
        if not user or not asst:
            yield Skip("no_pair")
            continue
        try:
            yield O.convert(user, asst, "BeardedMonster/african_sft", rng, meta={"turns_in_source": nmsg} if nmsg > 2 else {})
        except Skip as e:
            yield e


@register("old_eng_sft", "BeardedMonster/eng_sft + naija_sft (Alpaca/TinyStories/CoT machine-translated into Yoruba etc.; duplicates removed by hash)")
def old_eng(rng, limit=None):
    for repo in ("BeardedMonster/eng_sft", "BeardedMonster/naija_sft"):
        for r in _rows(repo, limit):
            user, asst, nmsg = _msgs(r)
            if not user or not asst:
                yield Skip("no_pair")
                continue
            try:
                yield O.convert(user, asst, repo, rng, meta={"turns_in_source": nmsg} if nmsg > 2 else {})
            except Skip as e:
                yield e


# ============================================================ instruction following in Yoruba / Hausa

def _answer_in_variant(rng, base_instruction: str, tgt: str, prob: float = 0.35):
    """Optionally append "Provide your response in <language>." (in a random instruction language) to a request."""
    if rng.random() >= prob:
        return base_instruction, "append", {}
    ins = T.make("answer_in", rng, target=tgt, source=tgt, prefer=tgt, position="append")
    sep = rng.choice([" ", "\n", " "])
    return f"{base_instruction}{sep}{ins.text}", "prepend", {"conditioned_on_language": True, "suffix_language": ins.language}


def _langs_with_suffix(base: str, extra: dict) -> list:
    """The request is `base` language text plus (sometimes) an 'answer in X' sentence in another language that is appended to it."""
    suf = extra.get("suffix_language")
    return [base] if not suf or suf == base else [base, suf]


def _gate(chunk_rows, text_of, code: str):
    """Keep the rows whose text (per text_of) is detected as `code` (certain or likely). Yields (row, unsure)."""
    ds = L.detect_many([text_of(r)[:1500] for r in chunk_rows])
    for r, d in zip(chunk_rows, ds):
        yield r, d, (d.code == code or code in d.glot) and d.level != "none" or d.code == code


@register("gpt_teacher_yo", "Aletheia-ng/gpt-teacher-yo (Yoruba instruction/input/response) with optional 'answer in Yoruba' conditioning")
def gpt_teacher(rng, limit=None):
    ds = "Aletheia-ng/gpt-teacher-yo"
    for chunk in batched(_rows(ds, limit), 256):
        for r, d, ok in _gate(chunk, lambda r: real(r.get("response")), "yor"):
            ins, inp, out = real(r.get("instruction")), real(r.get("input")), real(r.get("response"))
            if not ins or len(out) < 20 or degenerate(out):
                yield Skip("empty_short_or_repetitive")
            elif not ok:
                yield Skip("response_not_yoruba")
            else:
                ins2, pos, extra = _answer_in_variant(rng, ins, "yor")
                yield Rec("instruction_following", ds, "yor", "yor", out, input=inp, instruction=ins2, position="prepend", plan=["<|chat|>"], unsure_tgt=d.level != "certain",
                          input_langs=_langs_with_suffix("yor", extra), meta=extra)


@register("nvidia_steer_yo", "Aletheia-ng/nvidia_steer_yo (Yoruba prompt/response; helpfulness, correctness, coherence >= 3, reply >= 8 words)")
def steer(rng, limit=None):
    ds = "Aletheia-ng/nvidia_steer_yo"
    for chunk in batched(_rows(ds, limit), 256):
        for r, d, ok in _gate(chunk, lambda r: real(r.get("response")), "yor"):
            p, out = real(r.get("input")), real(r.get("response"))
            try:
                good = all(float(r.get(k) or 0) >= 3 for k in ("helpfulness", "correctness", "coherence"))
            except ValueError:
                good = False
            if not good:
                yield Skip("low_quality_score")
            elif not p or len(out.split()) < 8 or degenerate(out):
                yield Skip("empty_short_or_repetitive")
            elif not ok:
                yield Skip("response_not_yoruba")
            else:
                yield Rec("instruction_following", ds, "yor", "yor", out, input=p, prompt_override=p, plan=["<|chat|>"], unsure_tgt=d.level != "certain",
                          meta={"helpfulness": r.get("helpfulness"), "correctness": r.get("correctness"), "coherence": r.get("coherence")})


@register("roleplay_hausa", "Aletheia-ng/Roleplay-Hausa: English, Hausa, and the two mixed (Hausa request -> English answer, English request -> Hausa answer)")
def roleplay(rng, limit=None):
    ds = "Aletheia-ng/Roleplay-Hausa"
    for r in _rows(ds, limit):
        ei, ein, eo = real(r.get("instruction")), real(r.get("input")), real(r.get("response"))
        hi, hin, ho = real(r.get("umarni")), real(r.get("shigarwa")), real(r.get("amsa"))
        variants = []
        if ei and eo:
            variants.append((ei, ein, eo, "eng", "eng", None))
        if hi and ho:
            variants.append((hi, hin, ho, "hau", "hau", None))
        if hi and eo:
            variants.append((hi, ein or hin, eo, "hau", "eng", "eng"))      # Hausa request, answer in English
        if ei and ho:
            variants.append((ei, hin or ein, ho, "eng", "hau", "hau"))      # English request, answer in Hausa
        if not variants:
            yield Skip("empty")
            continue
        for ins, inp, out, s, t, cond in variants:
            if rep_ratio(out) > 0.45:
                yield Skip("repetitive")
                continue
            ins2, extra = ins, {"mixed": s != t}
            if cond:
                ins2, _, e2 = _answer_in_variant(rng, ins, cond, prob=1.0)
                extra.update(e2)
            yield Rec("roleplay" if rng.random() < 0.5 else "instruction_following", ds, s, t, out, input=inp, instruction=ins2, plan=["<|chat|>"],
                      input_langs=_langs_with_suffix(s, extra), meta=extra)


# ============================================================ classification datasets without instructions

@register("igbo_sentiment", "Aletheia-ng/IgboSenti-BBC (Igbo headline sentiment) -> <classify><sentiment>")
def igbo_senti(rng, limit=None):
    names = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    for r in _rows("Aletheia-ng/IgboSenti-BBC", limit):
        s, lab = clean(r.get("Sentence")), names.get(str(r.get("label")).strip().lower())
        if not s or not lab:
            yield Skip("bad_row")
            continue
        ins = T.make("sentiment", rng, target="eng", source="ibo", prefer="ibo")
        yield Rec("sentiment", "Aletheia-ng/IgboSenti-BBC", "ibo", "eng", lab, input=s, instruction=ins.text, position=ins.position, plan=["<classify>"],
                  label="<sentiment>", meta={"instruction_language": ins.language})


HAUSA_VOA = {"0": "Africa", "1": "Health", "2": "Nigeria", "3": "Politics", "4": "World"}


@register("hausa_voa_topics", "Aletheia-ng/hausa_voa_topics (Hausa headline topic; 5 classes) -> <classify><topic>")
def hausa_voa(rng, limit=None):
    for r in _rows("Aletheia-ng/hausa_voa_topics", limit):
        s, lab = clean(r.get("news_title")), HAUSA_VOA.get(str(r.get("label")).strip())
        if not s or not lab:
            yield Skip("bad_row")
            continue
        ins = T.make("topic", rng, target="eng", source="hau", prefer="hau")
        yield Rec("topic", "Aletheia-ng/hausa_voa_topics", "hau", "eng", lab, input=s, instruction=ins.text + " (Africa, Health, Nigeria, Politics, World)" if ins.language == "eng" else ins.text,
                  position=ins.position, plan=["<classify>"], label="<topic>", meta={"instruction_language": ins.language})


# ============================================================ stories, math, multiple choice, wiki

@register("tinystories", "Aletheia-ng/TinyStories_{igbo,yoruba}: story generation (Igbo_AI and Igbo kept apart) + eng<->native translation of stories")
def tinystories(rng, limit=None):
    for repo, code, human, mt in (("Aletheia-ng/TinyStories_igbo", "ibo", "Igbo", "Igbo_AI"), ("Aletheia-ng/TinyStories_yoruba", "yor", "Yoruba", "Yoruba_AI")):
        for r in _rows(repo, limit):
            eng = clean(r.get("English"))
            for col, kind in ((human, "human"), (mt, "machine")):
                txt = recase(clean(r.get(col)))
                if len(txt) < 80 or degenerate(txt):
                    yield Skip("short_or_repetitive")
                    continue
                if rng.random() < 0.6:                                      # story generation, with or without the first sentence as a seed
                    seed = first_sentence(txt, 140)
                    if rng.random() < 0.4:
                        ins = T.make("story_seed", rng, target=code, source=code, prefer=code, position="prepend", seed=seed)
                    else:
                        ins = T.make("story_prompt", rng, target=code, source=code, prefer=code)
                    seeded = ins.text.endswith(seed)
                    yield Rec("story_generation", repo, ins.language if ins.language in L.SUPPORTED_SET else "eng", code, txt, instruction=ins.text, plan=["<|generate|>"],
                              input_langs=list(dict.fromkeys([ins.language, code])) if seeded else None,       # the seed sentence is in the story's language
                              meta={"text_source": kind, "instruction_language": ins.language})
                if eng and len(eng) > 80 and kind == "human":
                    for r2 in translation(eng, txt, "eng", code, repo, rng, meta={"text_source": kind}):
                        yield r2


@register("storyteller", "Aletheia-ng/AI_Storyteller_Dataset2: synopsis->story (same language and cross-language), story->synopsis, translation")
def storyteller(rng, limit=None):
    langs = ("eng", "pcm", "yor", "ibo", "hau")
    ds = "Aletheia-ng/AI_Storyteller_Dataset2"
    for r in _rows(ds, limit):
        syn = {l: clean(r.get(f"synopsis_{l}")) for l in langs}
        sto = {l: clean(r.get(f"short_story_{l}")) for l in langs}
        ls = [l for l in langs if len(syn[l]) > 15 and len(sto[l]) > 100 and not degenerate(sto[l])]
        if not ls:
            yield Skip("no_usable_language")
            continue
        for l in ls:
            tgt = l if rng.random() < 0.6 else rng.choice(ls)
            ins = T.make("story", rng, target=tgt, source=l, prefer=l)
            yield Rec("story_generation", ds, l, tgt, sto[tgt], input=syn[l], instruction=ins.text, position=ins.position, plan=["<|generate|>"],
                      meta={"instruction_language": ins.language, "cross_language": tgt != l})
            ins = T.make("synopsis", rng, target=l, source=l, prefer=l)
            yield Rec("summarization", ds, l, l, syn[l], input=sto[l], instruction=ins.text, position=ins.position, plan=["<summarize>"], meta={"kind": "synopsis", "instruction_language": ins.language})
        if len(ls) >= 2:
            a, b = rng.sample(ls, 2)
            for x in translation(sto[a], sto[b], a, b, ds, rng, meta={"unit": "story"}):
                yield x


@register("coqa_stories", "Aletheia-ng/coqa-stories (text_eng/pcm/yor/ibo/hau): story retelling in another language and interlanguage translation")
def coqa(rng, limit=None):
    ds = "Aletheia-ng/coqa-stories"
    yield from _multi(ds, {c: f"text_{c}" for c in ("eng", "pcm", "yor", "ibo", "hau")}, 2, ds, rng, limit)


def _gsm_answer(s: str) -> str:
    out = clean(strip_calc(s))
    out = re.sub(r"\s*#{2,}\s*([-\d][\d,.\s]*)\s*#*\s*$", lambda m: f"\n#### {m.group(1).strip()}", out)       # the final-answer marker survives machine translation as '## 15#' 
    return "" if re.search(r"<<|>>", out) else out                 # a broken calculator annotation that could not be removed cleanly


@register("gsm8k", "Aletheia-ng/gsm8k{,2,3}, reasoning-gsm-qna-oa2: math word problems, answer in the question's language or another")
def gsm8k(rng, limit=None):
    langs = ("eng", "pcm", "yor", "ibo", "hau")
    for repo, qk, ak in (("Aletheia-ng/gsm8k", "question_{}", "answer_{}"), ("Aletheia-ng/gsm8k2", "question_{}", "answer_{}"),
                         ("Aletheia-ng/gsm8k3", "question_{}", "answer_{}"), ("Aletheia-ng/reasoning-gsm-qna-oa2", "INSTRUCTION_{}", "RESPONSE_{}")):
        for chunk in batched(_rows(repo, limit), 64):
            for r in chunk:
                q = {l: clean(r.get(qk.format(l))) for l in langs}
                a = {l: _gsm_answer(r.get(ak.format(l)) or "") for l in langs}
                good = [l for l in langs if len(q[l]) > 20 and len(a[l]) > 20 and not degenerate(q[l]) and not degenerate(a[l])]
                if not good:
                    yield Skip("degenerate_translation")
                    continue
                l = rng.choice(good)
                tgt = l if rng.random() < 0.7 else rng.choice(good)
                ins = T.make("math", rng, target=tgt, source=l, prefer=l)
                yield Rec("math", repo, l, tgt, a[tgt], input=q[l], instruction=ins.text, position=ins.position, plan=["<|math|>"],
                          meta={"instruction_language": ins.language, "cross_language": tgt != l})


@register("mmmlu_yoruba", "Aletheia-ng/MMMLU_yoruba (its only split is called `test`; used as TRAINING data on the owner's instruction) -> multiple choice (<qa><answer>)")
def mmmlu_yor(rng, limit=None):
    for r in _rows("Aletheia-ng/MMMLU_yoruba", limit):
        q = clean(r.get("Question"))
        opts = {k: clean(str(r.get(k))) for k in "ABCD"}
        ans = str(r.get("Answer")).strip().upper()
        if not q or ans not in opts or not all(opts.values()):
            yield Skip("bad_row")
            continue
        body = q + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())
        ins = T.make("mcq", rng, target="yor", source="yor", prefer="yor")
        subj = str(r.get("Subject") or "").replace("_", " ")
        yield Rec("multiple_choice", "Aletheia-ng/MMMLU_yoruba", "yor", "yor", f"{ans}. {opts[ans]}", input=body, instruction=ins.text, position=ins.position,
                  plan=["<qa>"], label="<answer>", meta={"subject": subj, "source_split": "test (used as train)", "instruction_language": ins.language})


SEARCH_TOOLS = [("search_wikipedia", "query"), ("wikipedia_lookup", "query"), ("search_encyclopedia", "query")]
_SEARCH_DESC = {"search_wikipedia": "Search Wikipedia and return the text of the best matching article.",
                "wikipedia_lookup": "Look up a topic on Wikipedia and return the article text.",
                "search_encyclopedia": "Search the encyclopedia and return the text of the best matching entry."}


def search_tool_def(name: str, arg: str) -> dict:
    """The function definition shown to the model in the system turn (rendered by the chat template inside <tools>..</tools>)."""
    return {"type": "function", "function": {"name": name, "description": _SEARCH_DESC[name], "parameters": {
        "type": "object", "properties": {arg: {"type": "string", "description": "The topic or title to look up."},
                                         "language": {"type": "string", "description": "ISO 639-1 code of the Wikipedia edition, e.g. yo."}},
        "required": [arg]}}}


TOOLS_MARKER = "You have access to the following tools:"


def _as_tool_list(v) -> list | None:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return None
    return [t for t in v if isinstance(t, dict) and isinstance(t.get("function"), dict)] if isinstance(v, list) and v else None


def split_tools(system: str) -> tuple[str, list | None]:
    """Older data-gen rows embed the function list in the system text ('You have access to the following tools:' + a JSON array). Return the
    system text without it and the parsed list, so the template renders the definitions itself (<tools>..</tools>)."""
    a, b = system.find("<tools>"), system.find("</tools>")
    if 0 <= a < b:                                                    # the standard block: one JSON object per line
        try:
            tools = [json.loads(line) for line in system[a + 7:b].splitlines() if line.strip()]
        except ValueError:
            return system, None
        return (system[:a].rstrip() + "\n\n" + system[b + 8:].lstrip()).strip(), tools
    i = system.find(TOOLS_MARKER)
    if i < 0:
        return system, None
    rest = system[i + len(TOOLS_MARKER):].lstrip()
    try:
        tools, end = json.JSONDecoder().raw_decode(rest)
    except ValueError:
        return system, None
    if not isinstance(tools, list):
        return system, None
    return (system[:i].rstrip() + "\n\n" + rest[end:].lstrip()).strip(), tools


WIKI_SYSTEM = "You are a helpful assistant. When you need facts you do not already have, call one of the functions listed below, then answer from what it returns."


@register("wiki_yo", "Aletheia-ng/wiki-yo -> (a) title-as-question encyclopedic answer, (b) RAG with a tool call and tool result")
def wiki_yo(rng, limit=None):
    ds = "Aletheia-ng/wiki-yo"
    for r in _rows(ds, limit):
        title, text = clean(r.get("title")), clean(r.get("text"))
        body = text[len(title):].strip() if text.startswith(title) else text
        if len(body) < 200 or not title or rep_ratio(body) > 0.4:
            yield Skip("short_or_empty")
            continue
        tgt = "yor"                                                   # the article is Yoruba, so the request always asks for Yoruba
        ins = T.make("wiki_ask", rng, target=tgt, source=tgt, prefer="yor", position="prepend", title=title)
        qlang = ins.language                                          # the user's message is in the instruction language
        if rng.random() < 0.5:
            yield Rec("encyclopedic_qa", ds, qlang, tgt, body[:5000], instruction=ins.text, position="prepend", plan=["<|explain|>"],
                      meta={"instruction_language": ins.language, "title": title})
            continue
        # RAG: the assistant calls a search tool, the passage comes back, the assistant answers from it (and only from it)
        name, arg = rng.choice(SEARCH_TOOLS)
        passage = body[:rng.choice([1200, 2000, 3000])]
        sents = re.split(r"(?<=[.!?])\s+", passage)
        answer = " ".join(sents[:rng.randint(3, 7)]) if len(sents) > 3 else passage
        call = A.assistant_message({"input_lang": qlang, "task_plan": ["<|RAG|>"], "tool_call": {"name": name, "arguments": {arg: title, "language": "yo"}}})
        final = A.assistant_content(input_lang=qlang, target_lang=tgt, task_plan=["<|RAG|>", "<|explain|>"], response=answer)
        convo = [{"role": "system", "content": WIKI_SYSTEM}, {"role": "user", "content": ins.text}, call,
                 {"role": "tool", "content": passage}, {"role": "assistant", "content": final}]
        yield Rec("rag_tool_call", ds, qlang, tgt, answer, input=title, instruction=ins.text, convo=convo, tools=[search_tool_def(name, arg)],
                  meta={"instruction_language": ins.language, "tool": name})


# ============================================================ summarisation / headline from news with detected language

@register("sumnews", "Aletheia-ng/sumnews (title, resume, news; language detected, unsupported dropped) -> summarisation + headline")
def sumnews(rng, limit=None):
    ds = "Aletheia-ng/sumnews"
    for chunk in batched(_rows(ds, limit), 128):
        news = [clean(r.get("news")) for r in chunk]
        det = L.detect_many([n[:1500] for n in news])
        for r, n, d in zip(chunk, news, det):
            title, resume = clean(r.get("title")), clean(r.get("resume"))
            if d.code is None or d.level == "none" or len(n) < 300 or not resume:
                yield Skip("language_unsupported_or_undetermined")
                continue
            u = d.level != "certain"
            ins = T.make("summarize", rng, target=d.code, source=d.code, prefer=d.code)
            yield Rec("summarization", ds, d.code, d.code, resume, input=n[:12000], instruction=ins.text, position=ins.position, plan=["<summarize>"], unsure_src=u, unsure_tgt=u,
                      meta={"instruction_language": ins.language})
            if title:
                ins = T.make("headline", rng, target=d.code, source=d.code, prefer=d.code)
                yield Rec("headline", ds, d.code, d.code, title, input=n[:6000], instruction=ins.text, position=ins.position, plan=["<summarize>"], label="<headline>",
                          unsure_src=u, unsure_tgt=u, meta={"instruction_language": ins.language})


# ============================================================ RAG with context, Llama-templated, Apollo, Aya collection, xP3x-Kongo

RAG_TEXT = re.compile(r"^<s>\s*<\|system\|>(?P<sys>.*?)</s>\s*<\|user\|>(?P<user>.*?)</s>\s*<\|assistant\|>\s*(?P<asst><\|input_lang\|>.*?)</s>\s*$", re.S)


@register("rag_synthetic", "BeardedMonster/rag_synthetic_data (context + question; system/user frames in many languages) -> <|chat|><|RAG|><qa>")
def rag_synthetic(rng, limit=None):
    ds = "BeardedMonster/rag_synthetic_data"
    for chunk in batched(_rows(ds, limit), 64):
        parsed = []
        for r in chunk:
            m = RAG_TEXT.match(r["text"].strip())
            if not m:
                yield Skip("unparseable")
                continue
            parsed.append(m)
        dets = L.detect_many([m.group("sys")[:600] for m in parsed])
        for m, d in zip(parsed, dets):
            if d.code is None or d.level == "none":
                yield Skip("frame_language_unsupported")           # e.g. a Fula system prompt
                continue
            try:
                rec = O.convert(m.group("user").strip(), m.group("asst"), ds, rng, task="rag_qa", expect_src="eng", expect_tgt="eng",
                                meta={"system_language": d.code})             # context, question and answer are English; only the frame is multilingual
                rec.system = m.group("sys").strip()
                yield rec
            except Skip as e:
                yield e


LLAMA = re.compile(r"^(?:<s>)?\s*\[INST\]\s*(?:<<SYS>>(?P<sys>.*?)<</SYS>>)?\s*(?P<user>.*?)\s*\[/INST\]\s*(?P<resp>.*?)\s*(?:</s>)?\s*$", re.S)
TRANSLATE_LLAMA = re.compile(r"translat\w*\s+(?:the\s+)?(?P<a>\w+)\s+(?:input\s+)?text\s+(?:in)?to\s+(?P<b>\w+)", re.I)


@register("low_resource_multilingual", "Aletheia-ng/low_resource_multilingual{,_sft_short,_sft_short2}: Llama [INST] templating removed; translation and Igbo instruction following")
def low_resource(rng, limit=None):
    seen = set()
    for repo in ("Aletheia-ng/low_resource_multilingual_sft_short", "Aletheia-ng/low_resource_multilingual_sft_short2", "Aletheia-ng/low_resource_multilingual"):
        for chunk in batched(_rows(repo, limit, columns=["prompt_response"]), 256):
            items = []
            for r in chunk:
                m = LLAMA.match(r["prompt_response"].strip())
                if not m or not m.group("resp"):
                    yield Skip("unparseable_or_empty_response")
                    continue
                user = re.sub(r"###\s*[\w' ]+:?\s*\n", "", m.group("user")).strip()
                resp = re.sub(r"^###\s*[\w' ]+:?\s*\n", "", m.group("resp")).strip()
                key = (user, resp)
                if key in seen:
                    yield Skip("duplicate")
                    continue
                seen.add(key)
                items.append((m.group("sys") or "", user, resp))
            if not items:
                continue
            du, dr = L.detect_many([u[:1200] for _, u, _ in items]), L.detect_many([r[:1200] for _, _, r in items])
            for (sysm, user, resp), a, b in zip(items, du, dr):
                if b.code is None or b.level == "none":
                    yield Skip("response_language_undetermined")
                    continue
                if re.search(r"translat", sysm + " " + user[:80], re.I):
                    text = re.sub(r"^.*?(?:into|to)\s+\w+\.\s*", "", user, count=1, flags=re.S) if re.match(r"(?i)translate", user) else user
                    if a.code is None or a.code == b.code or len(text) < 10:
                        yield Skip("translation_languages_unclear")
                        continue
                    for r2 in translation(clean(text), clean(resp), a.code, b.code, repo, rng, a.level != "certain", b.level != "certain", both=True):
                        yield r2
                else:
                    if a.code is None or a.level == "none":
                        yield Skip("prompt_language_undetermined")
                        continue
                    yield Rec("instruction_following", repo, a.code, b.code, resp, input="", prompt_override=user, plan=["<|chat|>"], unsure_src=a.level != "certain", unsure_tgt=b.level != "certain",
                              meta={"llama_template_removed": True})


@register("apollo_moe", "Aletheia-ng/ApolloMoEDataset (medical; only supported-language rows) -> <|is_medical|> + chat/qa")
def apollo(rng, limit=None):
    import ast
    ds = "Aletheia-ng/ApolloMoEDataset"
    for chunk in batched(_rows(ds, limit), 256):
        keep = []
        for r in chunk:
            try:
                conv = r["conversations"] if isinstance(r["conversations"], list) else ast.literal_eval(r["conversations"])
            except Exception:
                yield Skip("unparseable")
                continue
            if len(conv) < 2 or conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
                yield Skip("not_human_then_gpt")
                continue
            keep.append((r, clean(conv[0]["value"]), clean(conv[1]["value"])))
        du, dr = L.detect_many([u[:1200] for _, u, _ in keep]), L.detect_many([a[:1200] for _, _, a in keep])
        for (r, u, a), x, y in zip(keep, du, dr):
            if x.code is None or y.code is None or x.level == "none" or y.level == "none":
                yield Skip("language_unsupported_or_undetermined")
                continue
            if len(a) < 2 or degenerate(a):
                yield Skip("short_or_repetitive")
                continue
            kind = str(r.get("type") or "")
            medical = kind.lower().startswith("medical")
            plan = (["<|is_medical|>"] if medical else []) + ["<qa>" if kind == "medicalExam" else "<|chat|>"]
            yield Rec(("medical_" + ("exam" if kind == "medicalExam" else "chat")) if medical else "instruction_following", ds, x.code, y.code, a, input="", prompt_override=u, plan=plan,
                      unsure_src=x.level != "certain", unsure_tgt=y.level != "certain", label="<answer>" if r.get("type") == "medicalExam" else None,
                      meta={"apollo_type": r.get("type"), "apollo_source": r.get("source")})


# machine-translated Flan numbered answer lists ("Na farko. ... Na biyu. ..."): 94% of the question-answering targets in a 3,000-row count (quality_check.artefacts)
AYA_LIST_MARKERS = ("Na farko.", "Na biyu.", "Ta yaya za mu iya yin hakan?", "Abin da ya faru.", "A'a ba haka ba")
AYA_PLAN = {"question-answering": ("qa", ["<qa>"]), "summarization": ("summarization", ["<summarize>"])}


@register("aya_collection", "Aletheia-ng/aya_collection_language_split (Hausa; QA and summarisation) -> <qa> / <summarize>")
def aya_collection(rng, limit=None):
    ds = "Aletheia-ng/aya_collection_language_split"
    for chunk in batched(_rows(ds, limit), 256):
        rows = [r for r in chunk if r.get("task_type") in AYA_PLAN and clean(r.get("inputs")) and clean(r.get("targets"))]
        for _ in range(len(chunk) - len(rows)):
            yield Skip("task_type_not_certain")
        if not rows:
            continue
        di = L.detect_many([clean(r["inputs"])[:1500] for r in rows])
        dt = L.detect_many([clean(r["targets"])[:1500] for r in rows])
        for r, a, b in zip(rows, di, dt):
            task, plan = AYA_PLAN[r["task_type"]]
            lang = L.code_of(str(r.get("language")))
            ua, ub = _expect(a, lang or "hau"), _expect(b, lang or "hau")
            out = clean(r["targets"])
            if lang is None or ua is None or ub is None:
                yield Skip("language_mismatch")
            elif degenerate(out) or len(out) < 3:
                yield Skip("repetitive_or_short")
            elif r["task_type"] == "question-answering" and any(m in out for m in AYA_LIST_MARKERS):
                yield Skip("machine_translated_numbered_list")
            else:
                yield Rec(task, ds, lang, lang, out, input="", prompt_override=clean(r["inputs"]), plan=plan, unsure_src=ua, unsure_tgt=ub,
                          meta={"aya_dataset": r.get("dataset_name"), "aya_template": r.get("template_id")})


@register("xp3x_kongo", "Aletheia-ng/xP3x-Kongo: every target is Kikongo (not a supported language) -> all rows dropped, counted for the report")
def xp3x(rng, limit=None):
    for r in _rows("Aletheia-ng/xP3x-Kongo", limit):
        tgt = L.code_of(str(r.get("language") or "").split("_")[0])
        if tgt is None:
            yield Skip("target_language_not_supported:" + str(r.get("language")))
        else:
            yield Skip("unexpected_supported_target")


# ============================================================ data-gen (our own LLM-written set), multilingual MMLU, and the old sft-new rows

@register("datagen_sft", "BeardedMonster/data-gen sft/*.jsonl (messages already canonical; supported languages only; io_markers_ok; confidence >= 0.7)")
def datagen_sft(rng, limit=None):
    from sft_new.grammar import normalize_turn
    ds = "BeardedMonster/data-gen"
    files = [p for p, _ in hubio.list_files(ds) if p.startswith("sft/") and "_documents" not in p]
    n = 0
    for f in files:
        for r in hubio.iter_file(ds, f):
            msgs = r.get("messages")
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            lang = L.code_of(str(r.get("lang")))
            if not msgs or lang is None:
                yield Skip("language_not_supported" if lang is None else "no_messages")
                continue
            if not r.get("io_markers_ok", True) or float(r.get("confidence") or 0) < 0.7:
                yield Skip("low_confidence_or_markers")
                continue
            fixed, ok, tools = [], True, None
            for m in msgs:
                m = dict(m)
                if m["role"] == "system" and isinstance(m.get("content"), str):
                    m["content"], tools = split_tools(m["content"])         # the function list moves out of the text; the template renders it
                    if not m["content"]:
                        continue
                if m["role"] == "assistant" and isinstance(m.get("content"), str) and "tool_calls" not in m:
                    m["content"] = normalize_turn(m["content"], (r.get("tags") or [None])[0])
                    mm = re.match(r"<\|input_lang\|>(?: unsure, but most likely )?<([a-z]{3})>", m["content"])
                    if not mm or L.code_of(mm.group(1)) is None:
                        ok = False
                fixed.append(m)
            if not ok:
                yield Skip("assistant_language_not_supported")
                continue
            last = next((m["content"] for m in reversed(fixed) if m["role"] == "assistant" and "tool_calls" not in m), None)
            if last is None:
                yield Skip("no_final_answer")
                continue
            mm = re.match(r"<\|input_lang\|>(?: unsure, but most likely )?<([a-z]{3})>.*?(?:<\|target_lang\|>(?: unsure, but most likely )?<([a-z]{3})>)?<response>", last, re.S)
            src = L.code_of(mm.group(1)) if mm else lang
            tgt = L.code_of(mm.group(2)) if mm and mm.group(2) else src
            task = (r.get("tags") or ["generated"])[0]
            yield Rec(str(task), ds, src or lang, tgt or lang, last.split("<response>", 1)[-1], input=str(r.get("input") or ""), instruction=str(r.get("instruction") or ""), convo=fixed,
                      tools=tools or _as_tool_list(r.get("tools")),
                      meta={"model": r.get("model"), "domain": r.get("domain"), "confidence": r.get("confidence"), "tags": r.get("tags")})
            n += 1
            if limit and n >= limit:
                return


GLOBAL_MMLU_LANG = {"ha": "hau", "ig": "ibo", "ny": "nya", "sn": "sna", "so": "som", "sw": "swh", "yo": "yor", "en": None}


@register("global_mmlu_dev", "CohereLabs/Global-MMLU dev split (ha ig ny sn so sw yo) -> multiple choice. No train split exists; 'test' is excluded")
def global_mmlu(rng, limit=None):
    ds = "CohereLabs/Global-MMLU"
    for code2, code in GLOBAL_MMLU_LANG.items():
        if code is None:
            continue
        for r in hubio.iter_file(ds, f"{code2}/dev-00000-of-00001.parquet"):
            q = clean(r.get("question"))
            opts = {k: clean(str(r.get(f"option_{k.lower()}"))) for k in "ABCD"}
            ans = str(r.get("answer")).strip().upper()
            if not q or ans not in opts or not all(opts.values()):
                yield Skip("bad_row")
                continue
            ins = T.make("mcq", rng, target=code, source=code, prefer=code)
            body = q + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())
            yield Rec("multiple_choice", ds, code, code, f"{ans}. {opts[ans]}", input=body, instruction=ins.text, position=ins.position, plan=["<qa>"], label="<answer>",
                      meta={"subject": r.get("subject"), "split": "dev", "instruction_language": ins.language})


MMLU_PROX_LANG = {"af": "afr", "sw": "swh", "wo": "wol", "yo": "yor", "zu": "zul"}


@register("mmlu_prox_validation", "li-lab/MMLU-ProX validation split (af sw wo yo zu; 10 options) -> multiple choice")
def mmlu_prox(rng, limit=None):
    ds = "li-lab/MMLU-ProX"
    for code2, code in MMLU_PROX_LANG.items():
        for r in hubio.iter_file(ds, f"{code2}/validation-00000-of-00001.parquet"):
            q = clean(r.get("question"))
            opts = [(chr(65 + i), clean(str(r.get(f"option_{i}")))) for i in range(10) if r.get(f"option_{i}") not in (None, "", "N/A")]
            ans = str(r.get("answer")).strip().upper()
            if not q or ans not in dict(opts) or len(opts) < 2:
                yield Skip("bad_row")
                continue
            ins = T.make("mcq", rng, target=code, source=code, prefer=code)
            body = q + "\n" + "\n".join(f"{k}. {v}" for k, v in opts)
            yield Rec("multiple_choice", ds, code, code, f"{ans}. {dict(opts)[ans]}", input=body, instruction=ins.text, position=ins.position, plan=["<qa>"], label="<answer>",
                      meta={"subject": r.get("category"), "split": "validation", "instruction_language": ins.language})


def _sft_new_task(r: dict) -> str:
    """The translation-* and instruct-* folders of sft-new have no `task` column: take the task from the folder, never a catch-all.
    (The first full build wrote 5.17M rows as task 'unknown', which also kept them out of the QE audit sample.)"""
    if r.get("task"):
        return str(r["task"])
    folder = str(r.get("__file") or "").split("/")[1:2]
    folder = folder[0] if folder else ""
    if folder.startswith("translation"):
        return "translation"
    if folder.startswith("instruct"):
        return "instruct"
    return "unknown"


@register("sft_new_existing", "BeardedMonster/sft-new (the first build: translation, structured tasks, instruct) mapped to the combined schema")
def sft_new_existing(rng, limit=None):
    ds = "BeardedMonster/sft-new"
    pat = re.compile(r"<\|input_lang\|>(?: unsure, but most likely )?<([a-z]{3})>.*?(?:<\|target_lang\|>(?: unsure, but most likely )?<([a-z]{3})>)?<response>", re.S)
    for r in _rows(ds, limit, columns=["instruction", "input", "response", "language", "task", "messages", "source_dataset"]):
        msgs = json.loads(r["messages"])
        last = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
        m = pat.match(last)
        if not m:
            yield Skip("unparseable_assistant")
            continue
        yield Rec(_sft_new_task(r), ds, m.group(1), m.group(2) or m.group(1), str(r.get("response") or ""), input=str(r.get("input") or ""),
                  instruction=str(r.get("instruction") or ""), convo=msgs, meta={"origin": r.get("source_dataset")})


def families(base: str) -> list[str]:
    """The source families a naija-style converter works through, one build step each: ['naija@AfroLingu-MT_ibo_prompt-response', ...]."""
    repo, include = {"naija": ("BeardedMonster/naija", "tagged"), "naija2": ("BeardedMonster/naija2", "tagged"),
                     "naija_late": ("BeardedMonster/naija", "monolingual"), "naija2_late": ("BeardedMonster/naija2", "monolingual")}[base]
    return [f"{base}@{fam}" for fam in sorted({TP.family_of(p) for p in tagged_files(repo, include)})]
