"""Two-stage (really three-stage) generation for samples that need a LONG document.

The one-stage version did not work. Asking a model for "a conversation in which the user pastes a 900-3,500
word document and the assistant summarises it" produced documents of 635-647 words however firmly the prompt
insisted, because the document and the whole conversation compete for the SAME completion budget and a model
under output pressure quietly shortens the expensive part. Nor does one request reach 14,000 words even with
the whole budget to itself: models write to a habitual length, not to an instruction.

So a document is built the way a long document is actually written -- outline first, then sections:

  STAGE 0  ONE cheap request -> {"title", "setting", "cast", "facts", "sections":[{heading, brief}]}
           The cast and the facts are the point: every part is given the same named people, places and
           numbers, so the parts AGREE without having to be written in sequence.
  STAGE 1  ceil(sections / 6) requests IN PARALLEL, each writing its own slice of the outline.
  STAGE 2  the conversation, with the finished document passed in as INPUT and referred to by the
           placeholder __DOCUMENT__ in the user turn.

Stage 2 never re-emits the document, so it costs input tokens (~4x cheaper) instead of output tokens, and its
completion budget goes entirely to the conversation. `splice()` puts the real text back where the placeholder
was.

LENGTH
4,000-16,000 tokens per document, in four bands weighted towards the short end, because that is the real
distribution of documents anyone pastes into a chat. Sizes are in TOKENS and converted to words per language:
Yoruba costs ~2.5 tokens/word against English's ~1.15, so one word figure would either truncate Yoruba or
spend two thirds of the budget on nothing in English.

These samples need an SFT/RL training context above the pretrain block_size of 4096 -- a 16,000-token document
truncated at 4096 loses the summary, which is the only part carrying a training signal. Pin the whole band to
one size with DATA_GEN_DOC_TOKENS=N if you need to generate for a shorter context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from assemble import _TOKENS_PER_WORD
from providers.base import Request, Response

# Tasks whose sample contains a long generated document, and which therefore go through stages 0 and 1 first.
DOCUMENT_TASKS = {"long_document_summarization"}

# What the generator writes in the user turn instead of the document, and what splice() replaces.
PLACEHOLDER = "__DOCUMENT__"

# Document size in TOKENS, weighted towards the short end. A 16,000-token document costs ~7x a 4,000-token one
# and is ~7x rarer in practice, so weighting it equally would spend most of the budget on the rarest case.
#
# ELEVEN buckets, not ten, and that is the whole reason for the odd weight on the first band. The document
# LANGUAGE cycle below is 10 long; a size cycle that is also 10 long has the same period in `index`, so the two
# are locked together and only 10 of the 30 (language, size) pairs ever occur -- measured: no Fon row ever drew
# an English document above 12,000 tokens. 11 is coprime with 10, so the pair cycles over all 110.
_TOKEN_BANDS = [(4000, 6000)] * 5 + [(6000, 9000)] * 3 + [(9000, 12000)] * 2 + [(12000, 16000)]
_BAND_STRIDE = 7          # coprime with len(_TOKEN_BANDS) == 11, or most bands are unreachable
_PINNED = int(os.environ.get("DATA_GEN_DOC_TOKENS", "0")) or None

_CHARS_PER_WORD = 5.5
# Sections per stage-1 request. Each section runs ~400 words, so six is ~3,400 output tokens -- comfortably
# inside the range where a model still writes to length rather than trailing off.
SECTIONS_PER_PART = 6
# Words per section, by what the section is written in. A section of Fon is asked to be shorter because 220
# words of fluent Fon is worth more than 400 that drift into French halfway through.
_SECTION_WORDS = {"eng": 420, "mixed": 400, "native": 300, "native_low": 220}
# Ceiling on a document written entirely in the target language, by that language's tier.
_NATIVE_TOKEN_CAP = {"high": 12000, "medium": 9000, "low": 6000}
# The floor for KEEPING a document, by what it is written in. Deliberately absolute rather than a fraction of
# target: a 3,952-word document asked to be 6,686 is still an excellent long document and 6x what one-stage
# generation managed, so dropping it for missing its target throws away the good case with the bad. What is
# NOT acceptable is a document too short to be worth summarising -- measured: one Hausa attempt came back at
# 529 words, which is a passage, not a document.
_MIN_WORDS = {"eng": 1200, "mixed": 1100, "native": 900, "native_low": 650}
# More sections than this in one outline and the model starts dropping them; the word target is preserved by
# making each section longer instead.
MAX_SECTIONS = 20
_MAX_SECTION_WORDS = 700


def _tokens_for(index: int, lang: str) -> tuple[int, int]:
    if _PINNED:
        return _PINNED, _PINNED
    return _TOKEN_BANDS[(index * _BAND_STRIDE + _offset(lang)) % len(_TOKEN_BANDS)]


def words_for_tokens(tokens: int, doc_lang: str) -> int:
    return int(tokens / _TOKENS_PER_WORD.get(doc_lang, 2.5))


# Long-document FORMS. Deliberately not the 28-genre taxonomy: a poem, a social-media thread and a folktale
# are not documents anyone pastes in and asks to have summarised. These are. The stages below are the document
# skeleton -- used directly as the section list for a short document, and as the outline request's brief for a
# long one, where the model expands each stage into several sections.
_FORMS: list[tuple[str, str, list[str]]] = [
    ("field_report", "a field/monitoring report written by someone who visited the place",
     ["Background and why the visit happened", "What was observed, with specifics",
      "Numbers, measurements and costs", "Problems encountered", "Recommendations",
      "Next steps and who is responsible"]),
    ("news_feature", "a long newspaper feature, not a short bulletin",
     ["The situation now", "How it started", "What ordinary people say about it",
      "What officials and experts say", "What is being tried", "What happens next"]),
    ("meeting_minutes", "the minutes of a real working meeting, with named roles and decisions",
     ["Attendance, apologies and opening", "Matters arising from the last meeting",
      "Main item: discussion in detail", "Second item and the disagreement about it",
      "Decisions taken", "Action points with owners and dates"]),
    ("guideline", "an official plain-language guideline or standard operating procedure",
     ["Purpose and scope", "Who this applies to and who it does not",
      "The procedure, step by step", "Warning signs, exceptions and what to do instead",
      "Records to keep", "Where to get help and who to escalate to"]),
    ("training_manual", "a chapter from a practical training manual",
     ["Why this matters in practice", "What you need before you start",
      "The method, step by step", "The mistakes people make most often",
      "How to check your own work", "A practice exercise with the expected result"]),
    ("interview_transcript", "the transcript of a recorded interview, with speaker labels throughout",
     ["Introductions and how the subject came to this work", "The first substantive question and a long answer",
      "A follow-up that presses on a weak point", "A question about money or resources",
      "A disagreement or correction", "Closing question and what the subject wants readers to know"]),
    ("case_study", "a written case study of one organisation, farm, clinic or business",
     ["The setting and the people", "The problem as it was first understood",
      "What was tried and why", "What actually happened", "What it cost and who paid",
      "Lessons, including what would be done differently"]),
    ("annual_review", "an organisation's review of its year, written for members",
     ["Overview of the year", "Activities and what they achieved", "Money in and money out",
      "Staffing and volunteers", "Challenges, honestly stated", "Plans for next year"]),
    ("radio_transcript", "the transcript of a radio programme including caller segments",
     ["Presenter's opening and the day's topic", "The guest's explanation",
      "First caller and the answer", "Second caller, who disagrees",
      "A point the presenter asks the guest to clarify", "Wrap-up and what listeners should do"]),
    ("research_summary", "a plain-language summary of a piece of applied research",
     ["The question and why it matters here", "How the work was done",
      "What was measured and over how long", "What was found",
      "What the work cannot tell us", "What it means for practice"]),
]

_REGISTERS = ["plain and practical", "semi-formal", "formal and official", "conversational but organised"]

# Document language, per the owner's split: English 70%, mixed English + target 10%, target language 20%.
# Ten buckets so the split is exact, and the stride must be COPRIME with 10 or most buckets are unreachable
# (3 is; 2, 4, 5 and 6 are not) -- the same arithmetic trap that once locked every sub-topic to one genre.
_DOC_LANG_CYCLE = ["eng"] * 7 + ["mixed"] + ["native"] * 2
_DOC_LANG_STRIDE = 3


def _offset(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class DocSpec:
    """Everything about one document, derived deterministically from its plan row."""

    __slots__ = ("custom_id", "lang", "doc_lang", "doc_lang_mode", "form", "form_desc", "stages",
                 "tokens", "words", "n_sections", "section_words", "min_words", "register", "domain",
                 "subtopic", "parts")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


def plan_document(seed: Any, row: dict, *, domain: str, subtopic: str) -> DocSpec:
    """Decide form, language, length and section count for this row's document. Pure function of the row."""
    lang = row["lang"]
    i = row["index"] + _offset(lang)
    form, form_desc, stages = _FORMS[i % len(_FORMS)]
    mode = _DOC_LANG_CYCLE[(row["index"] * _DOC_LANG_STRIDE + _offset(lang)) % len(_DOC_LANG_CYCLE)]
    if lang == "eng":
        mode = "eng"
    doc_lang = "eng" if mode == "eng" else lang

    tier = next((l.tier for l in seed.languages if l.code == lang), "medium")
    lo, hi = _tokens_for(row["index"], lang)
    tokens = lo + (_offset(row["custom_id"]) % max(1, hi - lo))
    # A document written ENTIRELY in a low-resource language is capped short. The full band asked Fon for
    # 4,775 words, and a generator that half-knows Fon does not produce 4,775 good ones -- it drifts into
    # French or starts looping. English and mixed documents keep the whole 4k-16k band, which also matches
    # reality: the very long documents people actually paste in are mostly English.
    if mode == "native" and not _PINNED:
        tokens = min(tokens, _NATIVE_TOKEN_CAP.get(tier, 9000))
    if mode == "mixed":
        words = int((words_for_tokens(tokens, "eng") + words_for_tokens(tokens, doc_lang)) / 2)
    else:
        words = words_for_tokens(tokens, doc_lang)

    key = mode if mode != "native" else ("native_low" if tier == "low" else "native")
    n_sections = min(MAX_SECTIONS, max(len(stages), -(-words // _SECTION_WORDS[key])))
    section_words = min(_MAX_SECTION_WORDS, max(_SECTION_WORDS[key], -(-words // n_sections)))
    return DocSpec(custom_id=f"doc__{row['custom_id']}", lang=lang, doc_lang=doc_lang, doc_lang_mode=mode,
                   form=form, form_desc=form_desc, stages=stages, tokens=tokens, words=words,
                   n_sections=n_sections, section_words=section_words, min_words=_MIN_WORDS[key],
                   register=_REGISTERS[i % len(_REGISTERS)], domain=domain, subtopic=subtopic,
                   parts=max(1, -(-n_sections // SECTIONS_PER_PART)))


def _lang_line(seed: Any, spec: DocSpec) -> str:
    name = next((l.name for l in seed.languages if l.code == spec.doc_lang), spec.doc_lang)
    guidance = next((l.guidance for l in seed.languages if l.code == spec.doc_lang), "")
    if spec.doc_lang_mode == "eng":
        return ("LANGUAGE OF THE DOCUMENT: English. Real documents in this setting are mostly written in "
                "English, so this is the common case.")
    if spec.doc_lang_mode == "mixed":
        return (f"LANGUAGE OF THE DOCUMENT: mixed English and {name}, the way a real bilingual document is "
                f"mixed -- headings and official wording in English, quotations, examples and asides in "
                f"{name}. Do not translate; switch where a real writer would. {guidance}")
    return (f"LANGUAGE OF THE DOCUMENT: entirely {name}. No English except words the language genuinely "
            f"borrows. Keep sentences plain and concrete over a long stretch. {guidance}")


# --------------------------------------------------------------------------- stage 0: the outline


def outline_request(seed: Any, spec: DocSpec) -> Request:
    """STAGE 0. One cheap request producing the title, the cast, the numbers and the section list.

    The cast and facts are what make PARALLEL section writing possible. Without them, part 3 invents a
    different chairperson and a different budget from part 1, and the document reads as three documents. With
    them, every part is writing about the same people and the same numbers and no part has to wait for
    another.
    """
    stages = "\n".join(f"  - {s}" for s in spec.stages)
    system = ("You plan long documents before they are written. You return an outline only -- never the prose. "
              "Return strict JSON only, no commentary.")
    user = f"""Plan {spec.form_desc} on this subject: {spec.domain} / {spec.subtopic}.

The finished document will be about {spec.words:,} words, written by several writers working in parallel from
your outline. It must read as ONE document, so your outline has to fix everything they could otherwise
disagree about.

{_lang_line(seed, spec)}
Register: {spec.register}

It must move through these stages, in this order, expanded into EXACTLY {spec.n_sections} sections
(so several consecutive sections will belong to the same stage, going deeper rather than repeating):
{stages}

Return JSON:
{{"title": "<the document's own title>",
  "setting": "<one sentence: the place, the organisation, the period. Locally West African and specific.>",
  "cast": [{{"name": "<a plausible local name>", "role": "<their role, e.g. 'PHC nurse, Ìbàdàn North'>"}}],
  "facts": ["<a concrete figure or fact every writer must use consistently: a budget in naira/cedi/CFA, a
             headcount, a date, a distance, a price. 6-10 of them.>"],
  "tension": "<the one thing that went wrong or is disputed, which the whole document circles -- a figure
              that does not add up, a decision someone objected to, a promise not kept>",
  "sections": [{{"heading": "<the section's own heading, in the document's language>",
                "brief": "<one line: what happens in this section and which facts it uses>"}}]}}

Rules: 4-8 people in `cast`, all named and given roles. EXACTLY {spec.n_sections} entries in `sections`, in
reading order. No real named people, no invented published research, no fake citations."""
    return Request(
        custom_id=f"{spec.custom_id}__outline",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=min(8192, 900 + spec.n_sections * 90), temperature=0.9, top_p=0.95,
        response_format={"type": "json_object"},
        metadata={"stage": "outline", "doc_id": spec.custom_id, "spec": spec.as_dict()},
    )


def local_outline(spec: DocSpec) -> dict:
    """The outline for a document that needs no stage 0, and the FALLBACK when stage 0 fails.

    A single-part document cannot disagree with itself, so its own stages are outline enough. The fallback
    matters more: one transient outline failure used to cost an entire 9,700-word document, five section
    requests unspent. Falling back to the form's own stages yields a shorter document -- the stages are six, so
    it is one part -- which is still a real long document and vastly better than nothing.
    """
    return {"title": "", "setting": "", "cast": [], "facts": [], "tension": "",
            "sections": [{"heading": s, "brief": ""} for s in spec.stages[:spec.n_sections]]}


def parse_outline(resp: Response, spec: DocSpec) -> Optional[dict]:
    """A stage-0 response -> an outline, or None. A SHORT outline is accepted rather than rejected: if the
    model returns 10 sections where 20 were asked for, each section is simply asked to be longer (see
    `rescale`), which keeps the word target instead of discarding the plan."""
    data = _loose_json(resp.text) if resp.ok else None
    if not data:
        return None
    sections = [s for s in (data.get("sections") or [])
                if isinstance(s, dict) and str(s.get("heading") or "").strip()]
    if len(sections) < min(4, spec.n_sections):
        return None
    return {"title": str(data.get("title") or "").strip(),
            "setting": str(data.get("setting") or "").strip(),
            "cast": [c for c in (data.get("cast") or []) if isinstance(c, dict)],
            "facts": [str(f) for f in (data.get("facts") or []) if str(f).strip()],
            "tension": str(data.get("tension") or "").strip(),
            "sections": sections}


def rescale(spec: DocSpec, outline: dict) -> int:
    """Words per section for the outline we actually got, which may have fewer sections than were asked for.
    Preserves the document's word target where it can, bounded so no single section becomes an essay."""
    n = max(1, len(outline["sections"]))
    return min(_MAX_SECTION_WORDS, max(spec.section_words, -(-spec.words // n)))


# --------------------------------------------------------------------------- stage 1: the sections


def _shared_block(outline: dict) -> str:
    if not (outline.get("cast") or outline.get("facts")):
        return ""
    cast = "\n".join(f"  - {c.get('name','')}: {c.get('role','')}" for c in outline["cast"])
    facts = "\n".join(f"  - {f}" for f in outline["facts"])
    parts = [f"\nSETTING: {outline['setting']}" if outline.get("setting") else ""]
    if cast:
        parts.append(f"\nTHE PEOPLE -- use these names and roles, invent no others:\n{cast}")
    if facts:
        parts.append(f"\nFACTS THAT MUST STAY CONSISTENT -- use them, never contradict them:\n{facts}")
    if outline.get("tension"):
        parts.append(f"\nTHE TENSION running through the document: {outline['tension']}")
    return "".join(parts) + ("\nOther parts of this document are being written from the same sheet, so "
                             "anything not fixed above must not be presented as established fact.\n")


def document_request(seed: Any, spec: DocSpec, outline: dict, part: int = 0) -> Request:
    """STAGE 1. One request writing this part's slice of the outline, and nothing else."""
    sections = outline["sections"]
    section_words = rescale(spec, outline)
    lo = part * SECTIONS_PER_PART
    mine = sections[lo:lo + SECTIONS_PER_PART]
    plan = "\n".join(
        f"  {lo + n + 1}. {s['heading']} -- about {section_words} words"
        + (f"\n     ({s['brief']})" if s.get("brief") else "")
        for n, s in enumerate(mine))
    words_here = section_words * len(mine)
    n_parts = max(1, -(-len(sections) // SECTIONS_PER_PART))
    title = outline.get("title") or ""

    system = (
        "You write realistic long-form documents that will be used as INPUT for a summarisation exercise. "
        "The prose is the whole product: no preamble, no commentary, no summary of your own, no questions "
        "back, no note about what you have or have not covered. Return strict JSON only."
    )
    where = ("" if n_parts == 1 else
             f"\nThis is PART {part + 1} OF {n_parts} of one document titled \"{title}\". Write ONLY the "
             f"sections listed below. Do not introduce the document, do not recap earlier parts, do not "
             f"conclude it unless a closing section is in your list. Begin directly with your first heading.\n")
    user = f"""Write {spec.form_desc}.

Subject: {spec.domain} / {spec.subtopic}
Register: {spec.register}
{_lang_line(seed, spec)}
{_shared_block(outline)}{where}
YOUR SECTIONS -- write every one, in this order, each with its heading on its own line:
{plan}

LENGTH IS THE POINT: about {words_here:,} words for these {len(mine)} sections. A short document makes the
sample worthless. Write each section to its word target instead of summarising it, and if you find yourself
running out of things to say, go deeper into a specific -- one person's account, one week's figures, one
argument in full -- rather than stopping early or repeating yourself.

What makes it read as real:
- Concrete numbers that recur and stay consistent with the facts sheet: quantities, prices in naira/cedi/CFA,
  dates, distances, headcounts. Refer back to earlier figures.
- Specifics that are NOT in the headings, so a summary has to select rather than copy.
- Locally grounded: real West African settings, institutions, seasons, foods, transport, prices.
- No invented statistics presented as published research, no real named people, no fake citations.
- No filler, no repetition, no sentence that only restates the heading.

Plain text only: headings as their own lines, paragraphs, and speaker labels where the form calls for them.
NO markdown (`#`, `**`, bullet characters), NO html tags, NO special tokens.

Return JSON: {{"text": "<your sections, in full>"}}"""
    return Request(
        custom_id=f"{spec.custom_id}__p{part}",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        # ~3.2 tokens/word is generous for English and about right for diacritic-heavy languages. This
        # completion holds nothing but prose, so there is nothing to crowd it out.
        max_tokens=min(32768, max(2048, int(words_here * 3.2))),
        temperature=0.9, top_p=0.95,
        response_format={"type": "json_object"},
        metadata={"stage": "document", "doc_id": spec.custom_id, "part": part, "spec": spec.as_dict()},
    )


# --------------------------------------------------------------------------- parsing / cleaning


def _loose_json(text: str) -> Optional[dict]:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except ValueError:  # JSONDecodeError, and the bare ValueError for >4300-digit integer literals
        s, e = t.find("{"), t.rfind("}")
        if 0 <= s < e:
            try:
                v = json.loads(t[s:e + 1])
                return v if isinstance(v, dict) else None
            except ValueError:
                return None
        return None


# Document text is prose by contract. Markdown emphasis and stray html are formatting noise rather than
# meaning, so they are stripped -- unlike a marker inside a response field, where stripping corrupts the text
# ("<lang>pcm<target_lang>yorubaOwo tan." -> "pcmyorubaOwo tan.") and the sample is dropped instead. Removing
# "<h1>" from "<h1>Background</h1>" leaves "Background", which is correct.
_MD_BOLD = re.compile(r"\*{1,3}([^*\n]+)\*{1,3}")
_MD_HEAD = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.M)
# A tag is either a bare name (<h1>, </nav>, <br/>) or a name with at least one name="value" attribute
# (<div class='x'>). Requiring the `=` is what keeps prose safe: "if x<y then z>0" would match a looser
# pattern that allows valueless attributes, and stripping it would silently corrupt the sentence.
_HTMLISH = re.compile(
    r"""</?[A-Za-z][A-Za-z0-9]{0,20}\s*/?>"""
    r"""|<[A-Za-z][A-Za-z0-9]{0,20}(?:\s+[A-Za-z_:][-\w:.]*\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))+\s*/?>""")


def _distinct_ngram(text: str, n: int = 4) -> float:
    w = text.split()
    if len(w) <= n:
        return 1.0
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / len(g)


def clean_document(text: str) -> str:
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_HEAD.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _HTMLISH.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _text_from(data: dict) -> str:
    """`text` as asked for, but a model that decides to return the sections as a list or as an object of
    heading->prose is giving the same document in a different container, and throwing away a 3,000-word
    completion over the container is expensive. Anything else is a failure."""
    v = data.get("text")
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n\n".join(_stringify(x) for x in v)
    if isinstance(v, dict):
        return "\n\n".join(f"{k}\n{_stringify(x)}" for k, x in v.items())
    for key in ("document", "sections", "content", "body"):
        if key in data:
            return _text_from({"text": data[key]})
    return ""


def _stringify(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        head = x.get("heading") or x.get("title") or ""
        body = x.get("text") or x.get("content") or x.get("body") or ""
        return f"{head}\n{_stringify(body)}".strip() if head else _stringify(body)
    if isinstance(x, list):
        return "\n\n".join(_stringify(i) for i in x)
    return ""


def parse_document(resp: Response) -> Optional[dict]:
    """A stage-1 response -> {"title", "text", "words"}, or None if it is not usable."""
    if not resp.ok:
        return None
    data = _loose_json(resp.text)
    if not data:
        return None
    text = clean_document(_text_from(data))
    if not text:
        return None
    return {"title": clean_document(str(data.get("title") or "")), "text": text, "words": len(text.split())}


# --------------------------------------------------------------------------- store


class DocumentStore:
    """Documents on disk, keyed by doc_id. Stages 0 and 1 are the expensive half, so they are never paid
    twice: a resumed run, a second model being compared, or a re-run after a stage-2 prompt change all reuse
    what is already there."""

    def __init__(self, root: Path):
        self.root = root
        self.docs: dict[str, dict] = {}
        self._fh = None
        if root.exists():
            for shard in sorted(root.glob("*.jsonl")):
                with shard.open(encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                        except ValueError:
                            continue  # a half-written last line after a kill
                        if d.get("doc_id"):
                            self.docs[d["doc_id"]] = d

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def get(self, doc_id: str) -> Optional[dict]:
        return self.docs.get(doc_id)

    def put(self, doc: dict) -> None:
        self.docs[doc["doc_id"]] = doc
        if self._fh is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._fh = (self.root / "documents.jsonl").open("a", encoding="utf-8")
        self._fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def generate_documents(seed: Any, specs: list[DocSpec], provider: Any,
                       store: DocumentStore) -> dict[str, dict]:
    """Run stages 0 and 1 for every spec not already in the store. Returns {doc_id: document}.

    Deliberately forgiving about SIZE and strict about CONTINUITY. A document that came back shorter than its
    target is still a long document and is kept; a document missing its FIRST part is not, because it then
    starts mid-argument. Both rules come from measurement: the strict version dropped a perfectly good
    3,952-word document for missing a 6,686-word target, and lost a 9,700-word document entirely to one
    transient outline failure.
    """
    pending = [s for s in specs if s.custom_id not in store]
    if not pending:
        return {s.custom_id: store.get(s.custom_id) for s in specs if s.custom_id in store}
    by_spec = {s.custom_id: s for s in pending}

    # -- stage 0. Only for multi-part documents: a single-part document cannot contradict itself, so its own
    # stage list is outline enough and the request is not worth making.
    outlines = {s.custom_id: local_outline(s) for s in pending if s.parts == 1}
    need_outline = [s for s in pending if s.parts > 1]
    if need_outline:
        print(f"  stage 0: {len(need_outline):,} outlines", flush=True)
        for resp in provider.complete_many([outline_request(seed, s) for s in need_outline]):
            spec = by_spec[(resp.metadata or {})["doc_id"]]
            o = parse_outline(resp, spec)
            if o:
                outlines[spec.custom_id] = o
        fell_back = [s for s in need_outline if s.custom_id not in outlines]
        for s in fell_back:
            outlines[s.custom_id] = local_outline(s)     # shorter document, not no document
        if fell_back:
            print(f"    {len(fell_back):,} outlines failed -> falling back to the form's own stages "
                  f"(a shorter document rather than none)", flush=True)

    # -- stage 1. Every part of every document, in parallel: the shared cast and facts sheet is what makes
    # that safe, so nothing has to wait for the part before it.
    requests = []
    for s in pending:
        o = outlines[s.custom_id]
        n_parts = max(1, -(-len(o["sections"]) // SECTIONS_PER_PART))
        requests.extend(document_request(seed, s, o, p) for p in range(n_parts))
    print(f"  stage 1: {len(pending):,} documents, {len(requests):,} section requests", flush=True)
    parts: dict[str, dict[int, dict]] = {}
    for resp in provider.complete_many(requests):
        md = resp.metadata or {}
        doc = parse_document(resp)
        if doc:
            parts.setdefault(md["doc_id"], {})[md["part"]] = doc

    short = gapped = ended_early = 0
    for doc_id, got in parts.items():
        spec, o = by_spec[doc_id], outlines[doc_id]
        want = max(1, -(-len(o["sections"]) // SECTIONS_PER_PART))
        # Keep the LONGEST CONTIGUOUS PREFIX. A prefix is acceptable -- the document covers fewer sections than
        # planned but ends at a section boundary, so it reads as a document about less rather than one cut off
        # mid-sentence. A GAP is not: parts [0, 2] jump from the background straight to the recommendations,
        # and a summary of that is a summary of something incoherent. Truncating AT the gap gives the good half
        # instead of discarding both: measured at 36 of 160 documents in one tranche, all of them already paid
        # for. (The first version of this rule had it backwards -- it tolerated gaps and so accepted two-part
        # documents that had simply lost their ending.)
        have = []
        for p in range(want):
            if p not in got:
                break
            have.append(p)
        if not have:
            gapped += 1          # part 1 missing: the document would open mid-argument
            continue
        if len(have) < len(got):
            gapped += 1          # a gap was present; what follows it is dropped
        if len(have) < want:
            ended_early += 1
        text = "\n\n".join(got[p]["text"] for p in have)
        words = len(text.split())
        if words < spec.min_words:
            short += 1
            continue
        if _distinct_ngram(text) < 0.55:
            # A looping document is a generator that does not know the language well enough to sustain 3,000
            # words. Caught HERE, where it is the document's problem, rather than in stage 2, where it would
            # discard a sound conversation for the sake of its input.
            short += 1
            print(f"    looping document {doc_id} ({spec.doc_lang})", flush=True)
            continue
        store.put({"doc_id": doc_id, "title": o.get("title") or got[0].get("title") or "",
                   "text": text, "words": words, "lang": spec.lang, "doc_lang": spec.doc_lang,
                   "doc_lang_mode": spec.doc_lang_mode, "form": spec.form,
                   "target_words": spec.words, "target_tokens": spec.tokens,
                   "sections": len(o["sections"]), "parts": len(have), "parts_wanted": want,
                   "model": provider.model})
    if short or gapped or ended_early:
        print(f"    dropped: {short} too short or looping, {gapped} with a gap between parts; "
              f"{ended_early} kept as a shorter contiguous document", flush=True)
    return {s.custom_id: store.get(s.custom_id) for s in specs if s.custom_id in store}


# --------------------------------------------------------------------------- stage 2 splice


def splice(turns: Any, document: dict) -> tuple[Any, Optional[str]]:
    """Put the real document where the generator wrote __DOCUMENT__. Returns (turns, error).

    The placeholder is only legal in a USER turn. In an assistant response it would mean the summary REFERS to
    the document rather than summarising it, which is exactly the sample being worthless.
    """
    if not isinstance(turns, list):
        return turns, "turns_not_a_list"
    body = (f"{document['title']}\n\n{document['text']}" if document.get("title") else document["text"])
    found = False
    for t in turns:
        if not isinstance(t, dict):
            continue
        content = t.get("content")
        if not isinstance(content, str) or PLACEHOLDER not in content:
            if t.get("role") == "assistant" and PLACEHOLDER in str(t.get("response") or ""):
                return turns, "placeholder_in_response"
            continue
        if t.get("role") != "user":
            return turns, f"placeholder_in_{t.get('role')}_turn"
        t["content"] = content.replace(PLACEHOLDER, body)
        found = True
    if not found:
        return turns, "placeholder_missing"
    return turns, None


def document_brief(document: dict) -> str:
    """The stage-2 instruction block: the document is given, and must be referenced, never repeated."""
    return f"""THE DOCUMENT IS ALREADY WRITTEN -- it is below, {document.get('words') or 0:,} words of it, and it
is what the user pastes in. You must NOT retype it, shorten it, or paraphrase it into the user turn. In the
user turn where the user pastes it, write exactly the placeholder

    {PLACEHOLDER}

plus whatever the user says around it ("Abeg help me, {PLACEHOLDER} -- wetin be the main point?"). The real text
is substituted for the placeholder afterwards. Use the placeholder EXACTLY ONCE, in a user turn, and nowhere
else -- not in an assistant response, not in a tool result.

Your job is the CONVERSATION about it. The summary must be a real summary of the document below -- its specific
numbers, names, decisions and the point of tension in it -- not a generic description of what kind of document
it is. Read it before you write anything.

--- THE DOCUMENT (read it, summarise it, do not reproduce it) ---
{document.get('title') or ''}

{document.get('text') or ''}
--- END OF DOCUMENT ---"""
