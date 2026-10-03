"""Real documents for the RAG and summarisation tasks, sourced from Wikipedia instead of generated.

WHY SOURCE AT ALL
-----------------
Generating a unique long document per sample was the largest single cost in the RL run -- measured at ~$27
of documents against $50 of samples, plus hours of synchronous calls before a batch could be submitted.
Wikipedia is free.

But the better argument is quality. A generated document is written by the same model family that will be
questioned about it, so it contains exactly the facts the question expects, phrased the way the question
expects them. That makes retrieval artificially easy. Real encyclopaedia prose has irrelevant sections,
hedged claims, citation clutter, tables rendered as runs of numbers, and whole paragraphs about something
adjacent -- which is what a real retrieval returns, and what the model has to learn to read past.

WHAT WE SELECT, AND WHY IT IS "TECHNICAL"
-----------------------------------------
The corpus is teaching a DIVISION OF LABOUR: answer general knowledge from the weights, reach for a tool
when the question is technical. For that split to be learnable, the documents behind the tool have to be
the kind of thing a 306M-parameter model genuinely cannot hold -- a discharge rate, a dosage schedule, a
tolerance, a statutory fee, a varietal maturation period. If the retrieved document says what everyone
knows, the sample teaches the model to call a tool for things it already knows, which is the exact
over-retrieval failure the knowledge-boundary tasks exist to prevent.

So selection is biased towards SPECIFIC, CHECKABLE, NUMERIC content, scored by `technicality`:

  - numbers carrying units (mm, kg/ha, mg, kPa, °C, %, naira) -- the single strongest signal, because a
    figure with a unit is a fact that must be looked up rather than recalled
  - specification and procedure vocabulary (dosage, tolerance, rating, calibrate, threshold, protocol)
  - structural markers of a reference document (sections, ranges, per-unit rates)
  and penalised for the markers of general-interest prose: biography, plot, discography, filmography.

The pool is deliberately drawn across the corpus's own domains (agriculture, health, law, mining,
finance...) rather than from "technical" categories in general, so a retrieved document is about something
a West African user would plausibly ask about.

LICENCE
-------
Wikipedia text is CC BY-SA 4.0. Every pooled document keeps its title, page id, url and a licence field,
so a published corpus can attribute properly. Share-alike applies to a redistributed corpus -- that is a
decision for whoever publishes it, and the metadata is here so the decision can be made.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks for a descriptive agent with a contact. A generic one gets rate-limited or blocked.
UA = "SabiYarn-datagen/0.1 (West African LM research; https://huggingface.co/BeardedMonster)"
LICENCE = "CC BY-SA 4.0"

# Search seeds, grouped by the corpus's own domains (prompts.DOMAINS). Each is phrased to surface REFERENCE
# pages rather than overviews: the words are the ones a specification uses.
TOPIC_SEEDS: dict[str, tuple[str, ...]] = {
    "agriculture_crops": (
        "drip irrigation emitter discharge rate", "cassava mosaic disease resistant variety",
        "maize seed spacing plant population", "fertiliser application rate nitrogen phosphorus",
        "grain moisture content storage", "cocoa black pod disease fungicide",
        "rice paddy water management", "soil pH lime requirement", "post-harvest loss grain drying",
    ),
    "livestock_fisheries": (
        "poultry feed conversion ratio", "newcastle disease vaccination schedule",
        "tilapia stocking density pond", "cattle dipping acaricide concentration",
        "goat gestation period kidding", "fish feed protein requirement",
    ),
    "health_medicine": (
        "oral rehydration therapy composition", "artemisinin combination therapy dosing",
        "tuberculosis treatment regimen duration", "snakebite antivenom administration",
        "blood pressure classification hypertension stage", "anaemia haemoglobin threshold",
        "insulin dosing regimen", "burn depth classification treatment",
    ),
    "public_health": (
        "chlorination water treatment residual", "cold chain vaccine storage temperature",
        "cholera outbreak case definition", "measles vaccination coverage threshold",
        "malaria vector control insecticide", "sanitation latrine construction specification",
    ),
    "maternal_child_health": (
        "antenatal care visit schedule", "growth chart weight for age z-score",
        "exclusive breastfeeding duration recommendation", "neonatal resuscitation steps",
    ),
    "nutrition": (
        "dietary reference intake micronutrient", "food fortification level iron",
        "protein energy malnutrition classification",
    ),
    "law_governance": (
        "land registration procedure statute", "tenancy notice period law",
        "company incorporation requirement", "traffic offence penalty schedule",
        "employment termination notice statutory",
    ),
    "public_services_documents": (
        "passport application requirement", "national identification enrolment",
        "birth registration procedure", "driving licence class category",
    ),
    "extractives_mining": (
        "artisanal gold mining mercury", "quarry blasting safety distance",
        "mineral licence tenement", "tailings dam design",
    ),
    "technology_digital": (
        "GSM network frequency band", "solar panel rating inverter sizing",
        "transformer rating voltage regulation", "USSD session protocol",
        "lithium battery charging profile", "borehole pump head discharge",
    ),
    "finance_mobile_money": (
        "interest rate calculation amortisation", "mobile money transaction limit tier",
        "exchange rate spread calculation", "value added tax rate threshold",
    ),
    "housing_land": (
        "concrete mix ratio compressive strength", "reinforcement bar spacing slab",
        "septic tank sizing design", "roof truss span spacing",
    ),
    "disasters_emergencies": (
        "flood return period design", "earthquake magnitude intensity scale",
        "fire extinguisher class rating",
    ),
    "trade_markets": (
        "shipping container dimensions capacity", "customs tariff classification",
        "weights and measures standard", "cold storage temperature produce",
    ),
    "education": (
        "examination grading scale", "curriculum contact hours credit",
    ),
}

# --- technicality scoring -------------------------------------------------------------------------------
# A number WITH a unit. The heart of the score: a figure carrying a unit is a fact that has to be looked
# up, which is exactly the content the tool-using behaviour is for.
_UNIT = (r"(?:mm|cm|m|km|m2|m3|ha|kg|g|mg|µg|mcg|t|l|ml|kw|kwh|w|v|a|hz|mhz|ghz|kpa|mpa|bar|psi|"
         r"°c|°f|%|ppm|mol|kcal|kj|rpm|bps|kbps|mbps|mg/kg|kg/ha|l/s|m/s|days?|weeks?|months?|years?|"
         r"hours?|minutes?|naira|cedis?|usd|ngn|ghs)")
R_NUM_UNIT = re.compile(rf"\b\d[\d,.]*\s?{_UNIT}\b", re.I)
R_RANGE = re.compile(r"\b\d[\d,.]*\s?(?:-|–|to)\s?\d[\d,.]*\b")
R_NUMBER = re.compile(r"\b\d[\d,.]*\b")

_SPEC_WORDS = (
    "dosage", "dose", "dosing", "tolerance", "rating", "rated", "calibrat", "threshold", "protocol",
    "specification", "standard", "requirement", "procedure", "schedule", "regimen", "concentration",
    "ratio", "coefficient", "capacity", "efficiency", "yield", "density", "diameter", "thickness",
    "interval", "frequency", "classification", "grade", "class", "formula", "equation", "measured",
    "recommended", "minimum", "maximum", "per hectare", "per litre", "per kilogram", "statutory",
)
# Markers of general-interest prose: a 306M model is expected to handle this from its weights, and a
# retrieved document about it would teach over-retrieval.
_GENERAL_WORDS = (
    "discography", "filmography", "plot", "album", "band", "footballer", "actor", "actress", "singer",
    "novel", "character", "episode", "season", "tournament", "championship", "born on", "died on",
    "his career", "her career", "biography", "awards and nominations",
)


def technicality(text: str, title: str = "") -> float:
    """0..1. How much of this document is specific, checkable, looked-up fact.

    A proxy, and a deliberately crude one -- it is a filter on a free corpus, not a measurement. It only
    has to be good enough to prefer a page about emitter discharge rates over one about a footballer.
    """
    words = text.split()
    n = len(words)
    if n < 200:
        return 0.0
    low = text.lower()
    per_1k = 1000.0 / n
    units = len(R_NUM_UNIT.findall(text)) * per_1k      # figures with units, per 1k words
    ranges = len(R_RANGE.findall(text)) * per_1k
    nums = len(R_NUMBER.findall(text)) * per_1k
    spec = sum(low.count(w) for w in _SPEC_WORDS) * per_1k
    general = sum(low.count(w) for w in _GENERAL_WORDS) * per_1k

    score = (min(units / 12.0, 1.0) * 0.40
             + min(spec / 14.0, 1.0) * 0.25
             + min(ranges / 4.0, 1.0) * 0.15
             + min(nums / 35.0, 1.0) * 0.10
             + 0.10)  # a floor, so a sound reference page with few numbers is not excluded outright
    score -= min(general / 6.0, 1.0) * 0.45
    tl = title.lower()
    if any(w in tl for w in ("list of", "discography", "filmography", "timeline of")):
        score -= 0.30
    return max(0.0, min(1.0, score))


# --- the API ---------------------------------------------------------------------------------------------

class Wikipedia:
    """Minimal Action API client: search for candidates, then fetch plain text for the keepers.

    `list=search` returns `wordcount`, which is what makes this cheap -- length filtering happens on the
    search result and only the documents that survive it are downloaded.
    """

    def __init__(self, *, lang: str = "en", pause: float = 0.12, retries: int = 3):
        self.base = f"https://{lang}.wikipedia.org/w/api.php"
        self.pause = pause
        self.retries = retries

    def _get(self, **params: Any) -> dict:
        params.setdefault("format", "json")
        params.setdefault("action", "query")
        url = self.base + "?" + urllib.parse.urlencode(params)
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=45) as fh:
                    out = json.load(fh)
                time.sleep(self.pause)  # be a polite client; this is someone else's free service
                return out
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"wikipedia api failed after {self.retries} attempts: {last}")

    def search(self, query: str, *, limit: int = 50) -> list[dict]:
        d = self._get(list="search", srsearch=query, srlimit=limit, srnamespace=0,
                      srinfo="", srprop="wordcount")
        return d.get("query", {}).get("search", [])

    def plaintext(self, title: str) -> Optional[dict]:
        d = self._get(prop="extracts|info", explaintext=1, exlimit=1, titles=title,
                      redirects=1, inprop="url")
        pages = d.get("query", {}).get("pages", {})
        for pid, pg in pages.items():
            if str(pid) == "-1":
                return None
            text = pg.get("extract") or ""
            if not text.strip():
                return None
            return {"title": pg.get("title") or title, "pageid": pg.get("pageid"),
                    "url": pg.get("fullurl") or "", "text": text}
        return None


# --- cleaning --------------------------------------------------------------------------------------------

# Sections that are navigation rather than content. Kept deliberately short: the point of a real document
# is that it HAS clutter, so only the parts with no prose at all are cut.
_DROP_SECTIONS = ("references", "external links", "see also", "further reading", "bibliography",
                  "notes", "citations", "sources")


def clean(text: str) -> str:
    """Drop the trailing navigation sections and collapse blank runs. Nothing else.

    Over-cleaning would defeat the purpose: the irrelevant-but-plausible paragraph is the thing the model
    must learn to read past, so it stays.
    """
    out: list[str] = []
    skipping = False
    for line in text.splitlines():
        s = line.strip()
        m = re.fullmatch(r"=+\s*(.+?)\s*=+", s)
        if m:
            skipping = m.group(1).strip().lower() in _DROP_SECTIONS
            if not skipping:
                out.append(s)
            continue
        if not skipping:
            out.append(line)
    cleaned = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# --- pool building ---------------------------------------------------------------------------------------

# Calibrated against known pages rather than picked: general-interest articles score 0.00 (Fela Kuti,
# Nigeria national football team) to 0.30 (Lagos), reference pages 0.41 (Drip irrigation) to 0.68 (Oral
# rehydration therapy). 0.36 sits in the gap with room on both sides. A threshold of 0.42 -- my first guess
# -- excluded Drip irrigation, which is exactly the kind of page this pool exists for.
MIN_TECHNICALITY = 0.36


def harvest(*, min_words: int, max_words: int, min_technicality: float = MIN_TECHNICALITY,
            per_seed: int = 40, domains: Optional[Iterable[str]] = None,
            want: int = 0, verbose: bool = True) -> Iterator[dict]:
    """Yield cleaned, length-filtered, technicality-scored documents.

    Length is filtered on the SEARCH result (`wordcount`), so a page is only downloaded once it is already
    known to be the right size -- that is the difference between a few hundred requests and a few thousand.
    """
    wiki = Wikipedia()
    seeds = TOPIC_SEEDS if domains is None else {k: v for k, v in TOPIC_SEEDS.items() if k in set(domains)}
    seen_titles: set[str] = set()
    produced = 0
    for domain, queries in seeds.items():
        for q in queries:
            try:
                hits = wiki.search(q, limit=per_seed)
            except RuntimeError as exc:
                if verbose:
                    print(f"  [{domain}] search failed for {q!r}: {exc}")
                continue
            # Filter on wordcount BEFORE downloading. Wikipedia's wordcount runs a little high against
            # str.split(), so the window is widened slightly rather than tuned.
            cands = [h for h in hits
                     if h.get("title") not in seen_titles
                     and int(h.get("wordcount") or 0) >= min_words * 0.9]
            for h in cands:
                title = h["title"]
                seen_titles.add(title)
                try:
                    pg = wiki.plaintext(title)
                except RuntimeError:
                    continue
                if not pg:
                    continue
                body = clean(pg["text"])
                n = len(body.split())
                if not (min_words <= n <= max_words):
                    continue
                score = technicality(body, title)
                if score < min_technicality:
                    continue
                produced += 1
                if verbose:
                    print(f"  [{domain}] {title[:52]:<52} {n:>6,}w  tech {score:.2f}")
                yield {"title": pg["title"], "pageid": pg["pageid"], "url": pg["url"],
                       "text": body, "words": n, "technicality": round(score, 3),
                       "domain": domain, "source": "wikipedia", "licence": LICENCE}
                if want and produced >= want:
                    return


class DocumentPool:
    """Sourced documents on disk, drawn by (words, domain) and never handed out twice in a run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.docs: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.docs.append(json.loads(line))
                    except ValueError:
                        continue
        self._used: set[int] = set()

    def __len__(self) -> int:
        return len(self.docs)

    def extend(self, docs: Iterable[dict]) -> int:
        have = {d.get("pageid") for d in self.docs}
        added = 0
        with self.path.open("a", encoding="utf-8") as fh:
            for d in docs:
                if d.get("pageid") in have:
                    continue
                have.add(d.get("pageid"))
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                self.docs.append(d)
                added += 1
        return added

    def take(self, *, words: int, domain: str = "", key: str = "") -> Optional[dict]:
        """The unused document closest to `words`, preferring the requested domain.

        `key` makes the choice deterministic for a given caller, so a re-run picks the same document for the
        same sample and the corpus is reproducible.
        """
        pool = [(i, d) for i, d in enumerate(self.docs) if i not in self._used]
        if not pool:
            return None
        same = [(i, d) for i, d in pool if d.get("domain") == domain]
        candidates = same or pool
        # Closest length first, then a deterministic tie-break on the caller's key.
        candidates.sort(key=lambda it: (abs(int(it[1].get("words") or 0) - words),
                                        hash((key, it[1].get("pageid"))) & 0xffff))
        i, d = candidates[0]
        self._used.add(i)
        return d


# --- handing sourced documents to the generation path ----------------------------------------------------

def fill_store(specs: Iterable[Any], store: Any, pool: "DocumentPool") -> int:
    """Write pooled documents into the DocumentStore under the specs' own ids, and return how many.

    This is the whole integration. The store is already the cache the generator consults before building
    anything, so a sourced document placed there is simply never generated -- no change to stage 0, stage 1
    or the splice. Two consequences worth stating:

      - only ENGLISH-language documents can be sourced, because the low-resource Wikipedias are stubs
        (Yoruba's median article is 9 words). A native-language document still has to be written, and those
        specs are left alone here.
      - the record keeps `source`, `url` and `licence`, so a published corpus can attribute per document
        rather than with a blanket notice.
    """
    filled = 0
    for spec in specs:
        cid = getattr(spec, "custom_id", None)
        if not cid or cid in store:
            continue
        if getattr(spec, "doc_lang", None) != "eng":
            continue  # nothing to source: see above
        picked = pool.take(words=int(getattr(spec, "words", 0) or 0),
                           domain=getattr(spec, "domain", "") or "", key=cid)
        if picked is None:
            continue
        store.put({
            "doc_id": cid,
            "title": picked["title"],
            "text": picked["text"],
            "words": picked["words"],
            "lang": getattr(spec, "lang", "eng"),
            "doc_lang": "eng",
            "doc_lang_mode": "eng",
            # The form is what it is -- an encyclopaedia article -- rather than one of the generated forms.
            # Stage 2 reads this only to describe the document to the generator, so saying so is honest and
            # also tells the generator not to expect a report's section structure.
            "form": "encyclopaedia_article",
            "target_words": getattr(spec, "words", picked["words"]),
            "target_tokens": getattr(spec, "tokens", 0),
            "sections": 0,
            "parts": 1,
            "parts_wanted": 1,
            "rag": bool(getattr(spec, "rag", False)),
            "model": "wikipedia",
            "source": picked.get("source", "wikipedia"),
            "source_url": picked.get("url", ""),
            "licence": picked.get("licence", LICENCE),
            "technicality": picked.get("technicality"),
        })
        filled += 1
    return filled
