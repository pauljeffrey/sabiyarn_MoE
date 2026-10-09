"""Samples written as ONE string with two special tags in the middle (BeardedMonster/naija, naija2, Aletheia-ng/tds), e.g.

    <prompt> Translate the following sentence to Zulu (zul). Return the translated sentence only.\\nLayout  <response>: Ukubonakala ngaphandle
    <translate> Ị ma na ndị uweojii ga-anwụchị gị ...?<eng>\\nYou know that the police will arrest and jail you ...
    <classify> Anotanga kaita denderedzwa riri kubva mumakore ... <topic>: Weather
    <text> long article ... <summarize>: the summary            <title> long article ... <headline>: the headline
    <prompt> question <response>: <reason> worked solution     <prompt> utterance <response>: <intent>alarm_set

A sample is valid only if it has TWO such tags (first = what is asked, second = what comes back) AND the language of the text after
the first tag and of the text after the second tag can each be determined and is one of the languages we support. Anything else is
returned as `left over` so the caller can send it to the pretraining pile instead of guessing.

The LANGUAGE NAME IN A FILE NAME IS NOT TRUSTED (in these repos `sib200_fon_Latn` is Afrikaans, `HPLT2.0_cleaned_twi_Latn` is Fulfulde,
`xP3x_ibo_Latn` is Hausa): every side is language-identified from its own text.

Task rules (anything not matched is dropped, never guessed):
    <prompt> ... <response>   "Translate ... to X (iso)"  -> translation (target = X, must be supported)
                              response starts <reason>    -> math word problem / multiple choice (by `family`)
                              response starts <intent>    -> intent classification
                              otherwise                   -> instruction following (<|chat|>)
    <translate> A <lang>  B   -> translation A -> B, target = <lang>
    <classify>  A <label>: B  -> classification, label token from the second tag (topic, sentiment, intent, toxic, hate)
    <text> A <summarize>: B   -> summarisation         <title> A <headline>: B -> headline
"""

from __future__ import annotations

import random
import re

from africomet_translation.instructions import make_instruction
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import Rec, Skip
from sft_new.textutil import clean, degenerate

FIRST = re.compile(r"^\s*<(prompt|translate|classify|summarize|NER|clean|qa|text|title)>\s*")
SECOND = re.compile(r"\s*<(response|topic|sentiment|intent|toxic|hate|answer|summary|summarize|headline|[a-z]{3})>:?[ \t]*\n?")
LABEL_TOKEN = {"topic": "<topic>", "sentiment": "<sentiment>", "intent": "<intent>", "toxic": "<toxic>", "hate": "<|hate|>"}
TRANSLATE_PROMPT = re.compile(r"^Translate the following (?:sentence|text|passage)s? (?:to|into) (?P<name>[^()\n]+?)(?: \((?P<code>[a-z]{3})\))?\.(?: Return the translated (?:sentence|text) only\.)?\s*\n(?P<src>.+)$", re.S)
# a passage followed by "Question: ..." (multi-wiki-qa, Pidgin Wikipedia): context-grounded QA, i.e. RAG without the tool call
CONTEXT_QUESTION = re.compile(r"\n\s*(?:Question|Tambaya|Ajụjụ|Ibéèrè|Ìbéèrè|Vraag)\s*:\s*\S", re.I)
ALPACA_WRAP = re.compile(r"^Below is an instruction that describes a task\. Write a response that appropriately completes the request\.\s*Instruction:\s*(?P<q>.*?)\s*Response:\s*$", re.S)
# Families that carry process-supervision or system-prompt scaffolding rather than a clean sample: dropped on purpose
SKIP_FAMILIES = ("Math-Shepherd", "uhura-instruct")
# Sources with a licence that is not commercial (flagged in the row's metadata so they can be filtered with one test)
NC_FAMILIES = {"xlsum": "CC-BY-NC-SA-4.0 (XL-Sum)"}


def family_of(path: str) -> str:
    return re.sub(r"_batch_\d+_\d+.*", "", path.rsplit("/", 1)[-1]).replace(".parquet", "")


def split(t: str):
    """-> (verb, second, A, B) or raise Skip."""
    m = FIRST.match(t)
    if not m:
        raise Skip("no_first_tag")
    rest = t[m.end():]
    m2 = SECOND.search(rest)
    if not m2:
        raise Skip("no_second_tag")
    a, b = clean(rest[:m2.start()]), clean(rest[m2.end():])
    if not a or not b:
        raise Skip("empty_side")
    return m.group(1), m2.group(1), a, b


def _lang_of(d: L.Detection, expect: str | None = None):
    """-> (code, unsure) or None. An expected language only has to appear in the detector's top-3."""
    if expect:
        if d.code == expect or expect in d.glot:
            return expect, d.level != "certain" and d.code != expect
        return None
    if d.code and d.level != "none":
        return d.code, d.level != "certain"
    return None


def parse_mcq(a: str, letter: str):
    """Global-MMLU-style prompt: question lines, then the four options as bare lines. -> (question, {A:..,B:..,C:..,D:..}, letter) or None."""
    lines = [x.strip() for x in a.split("\n") if x.strip()]
    if len(lines) < 5 or letter.strip().upper() not in "ABCD" or len(letter.strip()) != 1:
        return None
    return " ".join(lines[:-4]), dict(zip("ABCD", lines[-4:])), letter.strip().upper()


# Reasons for which the TASK could not be determined (the text itself is fine and may still be used for pretraining), as opposed to a
# language we do not support or cannot identify (those rows are dropped for good).
TASK_UNCERTAIN = ("no_first_tag", "no_second_tag", "empty_side", "mcq_unparseable", "reason_family_unknown", "pattern:", "family_dropped_on_purpose")


def task_uncertain(reason: str) -> bool:
    return any(reason.startswith(r) for r in TASK_UNCERTAIN)


def plain_text(raw: str) -> str:
    """The tagged string with its special tags removed: what it is worth as ordinary pretraining text."""
    t = FIRST.sub("", raw, count=1)
    t = SECOND.sub("\n", t, count=1)
    return clean(re.sub(r"<(?:reason|intent|topic|sentiment|response|prompt)>", " ", t))


def family_prior(state: dict | None) -> str | None:
    """The language most of a file's A-side texts are certainly in (needs >= 40 certain detections and a 60% share), else None."""
    if not state or not state.get("counts"):
        return None
    total = sum(state["counts"].values())
    code, n = state["counts"].most_common(1)[0]
    return code if total >= 40 and n / total >= 0.6 else None


def process(texts: list[str], dataset: str, rng: random.Random, both_ways: bool = True, family: str = "",
            prior_state: dict | None = None) -> tuple[list[Rec], list[tuple[str, str]], dict]:
    """-> (records, [(raw text, reason)] for samples that did not become records, skip counts). One detector pass for the whole batch."""
    parsed, stats = [], {}
    if any(s in family for s in SKIP_FAMILIES):
        return [], [(t, "family_dropped_on_purpose") for t in texts], {"family_dropped_on_purpose": len(texts)}
    for t in texts:
        try:
            v, s, a, b = split(t)
            if v == "prompt" and s == "response":
                m = TRANSLATE_PROMPT.match(a)
                if m:
                    code = L.code_of(m.group("code") or m.group("name"))
                    if code is None:
                        raise Skip("target_not_supported")
                    parsed.append(("translate", clean(m.group("src")), b, code, t))
                elif b.startswith("<reason>"):
                    body = clean(b[len("<reason>"):])
                    if "Global-MMLU" in family:
                        q = parse_mcq(a, body[:3])
                        if not q:
                            raise Skip("mcq_unparseable")
                        parsed.append(("mcq", q, body, None, t))
                    elif "MathScale" in family or "grade-school" in family:
                        w = ALPACA_WRAP.match(a)
                        parsed.append(("math", clean(w.group("q")) if w else a, body, None, t))
                    else:
                        raise Skip("reason_family_unknown")
                elif b.startswith("<intent>"):
                    parsed.append(("classify:intent", a, clean(b[len("<intent>"):]), None, t))
                elif "Injongo" in family:
                    parsed.append(("classify:intent", a, b.split("\n")[0].strip(), None, t))
                else:
                    if L.has_foreign_segment(a):
                        raise Skip("foreign_language_inside_prompt")
                    parsed.append(("rag_qa" if CONTEXT_QUESTION.search(a) else "chat", a, b, None, t))
            elif v == "translate" and len(s) == 3 and s not in LABEL_TOKEN:
                code = L.code_of(s)
                if code is None:
                    raise Skip("target_not_supported")
                parsed.append(("translate", a, b, code, t))
            elif v == "classify" and s in LABEL_TOKEN:
                parsed.append(("classify:" + s, a, b, None, t))
            elif v == "text" and s == "summarize":
                parsed.append(("summarize", a, b, None, t))
            elif v == "title" and s == "headline":
                parsed.append(("headline", a, b, None, t))
            elif v == "summarize":
                parsed.append(("summarize", a, b, None, t))
            else:
                raise Skip(f"pattern:{v}+{s}")
        except Skip as e:
            stats[str(e)] = stats.get(str(e), 0) + 1
            parsed.append(("skip", None, None, str(e), t))

    live = [p for p in parsed if p[0] != "skip"]
    side_a = [p[1][0] if p[0] == "mcq" else p[1] for p in live]
    da = L.detect_many([x[:1500] for x in side_a])
    db = L.detect_many([(p[2] if p[0] != "mcq" else p[1][0])[:1500] for p in live])
    recs, left = [], []
    prior = family_prior(prior_state)
    nc = next((v for k, v in NC_FAMILIES.items() if k in family), None)
    it = iter(zip(live, da, db))
    for kind, a, b, code, raw in parsed:
        if kind == "skip":
            left.append((raw, code))                              # `code` holds the reason for a skipped sample
            continue
        _, d_a, d_b = next(it)
        la = _lang_of(d_a)
        if prior_state is not None and kind != "translate":
            if d_a.code and d_a.level == "certain":
                prior_state.setdefault("counts", __import__("collections").Counter())[d_a.code] += 1
            elif la is None and prior and prior in d_a.glot and not d_a.foreign:
                la = (prior, True)                                   # a short text of a file whose texts are all in `prior`: accepted as "unsure, but most likely"
        if kind == "translate":
            lb = _lang_of(d_b, expect=code)
        elif kind.startswith("classify") or kind == "mcq":
            lb = la if kind == "mcq" else ("eng", False)           # labels are English words; an MCQ answer is in the question's language
        else:
            lb = _lang_of(d_b)
        if la is None or lb is None:
            stats["language_undetermined"] = stats.get("language_undetermined", 0) + 1
            left.append((raw, "language_undetermined"))
            continue
        text_a = a[0] if kind == "mcq" else a
        if degenerate(text_a) or degenerate(b):
            stats["repetitive"] = stats.get("repetitive", 0) + 1
            continue
        (src, us), (tgt, ut) = la, lb
        meta = {"format": "tagged", "family": family, **({"license_flag": nc} if nc else {})}
        if kind == "translate":
            pairs = [(a, b, src, tgt, us, ut)] + ([(b, a, tgt, src, ut, us)] if both_ways else [])
            for x, y, s_, t_, u1, u2 in pairs:
                il = rng.choice(["eng", "eng", "pcm", "yor", "hau", "ibo"]) if rng.random() < 0.5 else "eng"
                ins = make_instruction(il, s_, t_, rng)
                recs.append(Rec("translation", dataset, s_, t_, y, input=x, instruction=ins.text, position=ins.position, plan=["<translate>"],
                                unsure_src=u1, unsure_tgt=u2, meta={**meta, "instruction_language": ins.language, "direction": "forward" if x == a else "backward"}))
        elif kind == "rag_qa":
            ins = T.make("qa_context", rng, target=tgt, source=src, prefer=src)
            recs.append(Rec("rag_qa", dataset, src, tgt, b, input=a, instruction=ins.text, position=ins.position, plan=["<|RAG|>", "<qa>"], unsure_src=us, unsure_tgt=ut,
                            meta={**meta, "instruction_language": ins.language}))
        elif kind == "chat":
            recs.append(Rec("instruction_following", dataset, src, tgt, b, input=a, plan=["<|chat|>"], prompt_override=a, unsure_src=us, unsure_tgt=ut, meta=meta))
        elif kind.startswith("classify"):
            lab = kind.split(":")[1]
            task = {"topic": "topic", "sentiment": "sentiment", "intent": "intent", "toxic": "toxicity", "hate": "hate"}[lab]
            pool = {"toxic": "hate"}.get(lab, lab)
            ins = T.make(pool, rng, target="eng", source=src, prefer=src, position="prepend" if rng.random() < .6 else "append")
            recs.append(Rec(task, dataset, src, "eng", b, input=a, instruction=ins.text, position=ins.position, plan=["<classify>"], label=LABEL_TOKEN[lab],
                            unsure_src=us, meta={**meta, "instruction_language": ins.language}))
        elif kind in ("summarize", "headline"):
            ins = T.make("summarize" if kind == "summarize" else "headline", rng, target=tgt, source=src, prefer=src)
            recs.append(Rec("summarization" if kind == "summarize" else "headline", dataset, src, tgt, b, input=a, instruction=ins.text, position=ins.position,
                            plan=["<summarize>"], label="<headline>" if kind == "headline" else None, unsure_src=us, unsure_tgt=ut,
                            meta={**meta, "instruction_language": ins.language}))
        elif kind == "mcq":
            question, opts, letter = a
            body = question + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())
            ins = T.make("mcq", rng, target=src, source=src, prefer=src)
            recs.append(Rec("multiple_choice", dataset, src, src, f"{letter}. {opts[letter]}", input=body, instruction=ins.text, position=ins.position, plan=["<qa>"],
                            label="<answer>", unsure_src=us, meta={**meta, "split": "dev", "instruction_language": ins.language}))
        elif kind == "math":
            ins = T.make("math", rng, target=src, source=src, prefer=src)
            recs.append(Rec("math", dataset, src, tgt, b, input=a, instruction=ins.text, position=ins.position, plan=["<|math|>"], unsure_src=us, unsure_tgt=ut,
                            meta={**meta, "instruction_language": ins.language}))
    return recs, left, stats
