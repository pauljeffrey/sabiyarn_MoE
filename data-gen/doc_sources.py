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
        "school building classroom size standard", "teacher qualification requirement",
    ),
    "energy_power": (
        "solar photovoltaic module efficiency rating", "inverter sizing surge capacity",
        "deep cycle battery depth of discharge", "generator fuel consumption rating",
        "transmission line voltage drop", "earthing resistance specification",
        "power factor correction capacitor", "diesel generator derating altitude",
    ),
    "engineering_construction": (
        "concrete curing time strength gain", "soil bearing capacity foundation design",
        "steel reinforcement lap length", "block wall mortar mix proportion",
        "borehole casing screen design", "culvert sizing design discharge",
        "scaffolding load capacity standard",
    ),
    "water_sanitation": (
        "slow sand filter loading rate", "borehole yield pumping test",
        "chlorine dose contact time disinfection", "pit latrine setback distance groundwater",
        "rainwater harvesting tank sizing", "water quality guideline value turbidity",
        "greywater reuse irrigation standard",
    ),
    "transport_mobility": (
        "tyre load index speed rating", "axle load limit road",
        "brake pad wear specification", "vehicle emission standard limit",
        "motorcycle helmet safety standard", "road camber gradient design",
    ),
    "telecom_phones": (
        "LTE band frequency duplex", "antenna gain radiation pattern",
        "fibre optic attenuation splice loss", "VSAT link budget",
        "SIM card authentication protocol", "mobile battery charging current",
    ),
    "environment_climate": (
        "rainfall intensity duration frequency", "evapotranspiration reference crop coefficient",
        "air quality index particulate matter", "carbon sequestration rate",
        "desertification land degradation indicator", "mangrove restoration planting density",
    ),
    "natural_science": (
        "soil texture classification triangle", "groundwater aquifer transmissivity",
        "mosquito larval habitat control", "plant nutrient deficiency symptom",
        "geological formation stratigraphy", "lightning protection zone",
    ),
    "weather_seasons": (
        "harmattan dust haze visibility", "rainy season onset cessation criteria",
        "tropical cyclone intensity scale", "weather station instrument siting",
    ),
    "security_safety": (
        "personal protective equipment standard", "fire resistance rating wall",
        "first aid burn treatment protocol", "occupational exposure limit noise",
        "pesticide re-entry interval",
    ),
    "animals_wildlife": (
        "rabies post-exposure prophylaxis", "tsetse fly trypanosomiasis control",
        "veterinary drug withdrawal period", "wildlife stocking carrying capacity",
    ),
    "ai_data": (
        "sampling error confidence interval", "database normalisation form",
        "data retention regulation requirement", "cryptographic key length recommendation",
    ),
    "careers_work": (
        "professional certification requirement hours", "minimum wage statutory rate",
        "pension contribution rate statutory", "apprenticeship duration requirement",
    ),
    "childhood_parenting": (
        "childhood immunisation schedule dose", "infant growth monitoring indicator",
        "deworming dose schedule children", "iron supplementation children dose",
    ),
    "consumer_life": (
        "electrical appliance energy label", "warranty period statutory consumer",
        "food labelling requirement nutrition", "weighing instrument accuracy class",
    ),
    "disability_inclusion": (
        "wheelchair ramp gradient standard", "hearing aid fitting specification",
        "accessible toilet dimension standard",
    ),
    "entrepreneurship": (
        "business name registration fee", "small business tax threshold",
        "cooperative society registration requirement",
    ),
}

# DELIBERATELY NOT SEEDED: music, film_nollywood, sports, history, festivals_customs,
# proverbs_oral_tradition, arts_literature, geography_places, rural_life, urban_life, religion_ethics,
# family_relationships, language_linguistics and the rest of the general-interest domains.
#
# That is the point rather than an omission. The corpus is teaching the model to answer general knowledge
# FROM ITS WEIGHTS and to reach for a tool only when the question turns on a looked-up specific. Putting a
# retrieved document about Nollywood or a proverb behind the tool would teach exactly the opposite -- that
# the right response to something it already knows is to go and search. Those domains still appear
# throughout the corpus; they appear in the tasks that answer WITHOUT a tool.

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


# --- relevance and locality ------------------------------------------------------------------------------
# Technicality alone is not enough, and the first harvest proved it: "Toyota Prius (XW20)" scored 0.95,
# "BMW 7 Series (G70)" 0.93, alongside "Parliament Hill", "Coal companies of Australia" and "Value-added tax
# in the United Kingdom". All genuinely technical; all useless here. A retrieved document about a BMW trim
# level or UK VAT teaches retrieval on content no SabiYarn user will ever ask about, and mis-teaches the
# jurisdiction for anything legal or fiscal.
#
# So a candidate must also be ON TOPIC for the query that found it, and must not be anchored to a place or
# product the corpus does not serve.

# A specific product or model: a parenthesised model code, a trim designation, a model year.
R_MODEL = re.compile(r"\((?:[A-Z]{1,3}\d{2,3}|\d{4})\)|\b(?:Mk\s?\d|Series\s*\(|generation\))", re.I)
# Anchored to a jurisdiction or place the corpus does not serve. Legal, fiscal and standards pages are the
# ones that matter: the wrong jurisdiction is worse than no document.
# Both forms matter. "of Canada" and "Canadian" anchor a page just as firmly, and the first version caught
# only the prepositional one -- which let "Canadian passport", "Argentine identity card" and "Malaysian
# identity card" into the pool, each teaching the wrong country's procedure for a document question.
_FOREIGN_PLACES = (
    "United Kingdom", "United States", "Australia", "Canada", "Britain", "England", "Scotland", "Wales",
    "Ireland", "Germany", "France", "Japan", "China", "India", "Brazil", "Russia", "Sweden", "Norway",
    "Denmark", "Finland", "Italy", "Spain", "Poland", "Mexico", "Netherlands", "Philippines", "Argentina",
    "Malaysia", "Indonesia", "Pakistan", "Bangladesh", "Turkey", "Korea", "Vietnam", "Thailand",
)
_FOREIGN_ADJECTIVES = (
    "British", "American", "Australian", "Canadian", "English", "Scottish", "Welsh", "Irish", "German",
    "French", "Japanese", "Chinese", "Indian", "Brazilian", "Russian", "Swedish", "Norwegian", "Danish",
    "Finnish", "Italian", "Spanish", "Polish", "Mexican", "Dutch", "Filipino", "Argentine", "Argentinian",
    "Malaysian", "Indonesian", "Pakistani", "Bangladeshi", "Turkish", "Korean", "Vietnamese", "Thai",
)
R_FOREIGN = re.compile(
    r"\b(?:in|of)\s+(?:the\s+)?(?:" + "|".join(re.escape(p) for p in _FOREIGN_PLACES) + r")\b"
    r"|\b(?:" + "|".join(_FOREIGN_ADJECTIVES) + r")\b", re.I)
# Places and lineages, which arrive via search noise and carry no transferable fact.
R_PROPER_SUBJECT = re.compile(r"\b(?:family|dynasty|dam|bridge|stadium|cathedral|castle|palace|"
                              r"constituency|county|province|prefecture)\b", re.I)
# Worth keeping even if it mentions elsewhere: the corpus's own region.
R_LOCAL = re.compile(
    r"\b(?:africa|african|nigeria|nigerian|ghana|ghanaian|benin|togo|niger|cameroon|senegal|mali|"
    r"burkina|ivory\s+coast|c[oô]te\s+d'ivoire|liberia|sierra\s+leone|gambia|guinea|chad|"
    r"sub-saharan|west\s+africa|tropical|sahel|savanna)\b", re.I)


# A BIOGRAPHY. Wikipedia opens these with a birth date in parentheses, which makes them exactly detectable
# rather than guessed at. A person's page carries no transferable fact for a retrieval corpus, and these
# arrive through search noise -- "Eddie Obeid" came back for a query about mineral licences.
R_BIOGRAPHY = re.compile(
    r"\((?:born\s|b\.\s|\d{1,2}\s+\w+\s+\d{4}\s*[-–]\s*)"
    r"|\bwas\s+an?\s+(?:[a-z]+\s+){0,3}(?:politician|businessman|footballer|singer|actor|writer|poet|"
    r"journalist|lawyer|engineer|physician|economist|academic|activist)\b"
    r"|\bis\s+an?\s+(?:[a-z]+\s+){0,3}(?:politician|businessman|footballer|singer|actor|writer|poet|"
    r"journalist|lawyer|economist|activist)\b", re.I)


def _content_terms(q: str) -> set[str]:
    stop = {"the", "a", "an", "of", "and", "for", "in", "to", "on", "rate", "per"}
    return {w for w in re.findall(r"[a-z]+", q.lower()) if len(w) > 3 and w not in stop}


def relevance(text: str, title: str, query: str) -> float:
    """0..1. Is this document about what the query asked for, and is it usable for this corpus?

    Deliberately a gate rather than a ranking: the pool is drawn from a free corpus, so it costs nothing to
    be strict and a lot to be wrong.
    """
    terms = _content_terms(query)
    if not terms:
        return 0.0
    low = text.lower()
    lead = low[:2000]  # the lead section defines the subject; a passing mention later does not
    hit_any = sum(1 for w in terms if w in low) / len(terms)
    hit_lead = sum(1 for w in terms if w in lead) / len(terms)
    # The LEAD is what decides the subject. A document whose lead mentions none of the query's terms is
    # about something else, however many times they appear further down -- measured: a page with the terms
    # only in its last sentence scored 0.55 and passed the gate, because the any-mention term dominated.
    if hit_lead == 0.0:
        return 0.0
    score = 0.45 * hit_any + 0.55 * hit_lead

    # A jurisdiction in the TITLE is a hard reject, not a penalty. "Canadian passport" is a perfect match
    # for a query about identity enrolment and still the wrong answer: a document question answered from
    # another country's procedure is worse than one answered with "I could not find it". A penalty left it
    # at 0.50, above the gate.
    if R_FOREIGN.search(title):
        return 0.0
    # Likewise a specific product or model year. "Toyota Prius (XW20)" matches a query about battery
    # charging perfectly and is still never a document a SabiYarn user retrieves. A penalty left it at 0.55.
    if R_MODEL.search(title):
        return 0.0
    # Checked against the LEAD, which is where Wikipedia states a birth date.
    if R_BIOGRAPHY.search(text[:400]):
        return 0.0
    if R_PROPER_SUBJECT.search(title):
        score -= 0.35           # a named place or lineage
    if R_FOREIGN.search(lead) and not R_LOCAL.search(low):
        score -= 0.20
    if R_LOCAL.search(low):
        score += 0.15           # actually about somewhere the corpus serves
    return max(0.0, min(1.0, score))


MIN_RELEVANCE = 0.40


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
                rel = relevance(body, title, q)
                if rel < MIN_RELEVANCE:
                    continue
                produced += 1
                if verbose:
                    print(f"  [{domain}] {title[:46]:<46} {n:>6,}w  tech {score:.2f} rel {rel:.2f}")
                yield {"title": pg["title"], "pageid": pg["pageid"], "url": pg["url"],
                       "text": body, "words": n, "technicality": round(score, 3),
                       "relevance": round(rel, 3), "query": q,
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


# --- chunking --------------------------------------------------------------------------------------------
# A real retriever returns PASSAGES, not whole documents. Handing the model a 3,000-word article in one tool
# result teaches it to read an article; handing it four labelled passages, two of which are from the wrong
# section, teaches it to retrieve -- which is the behaviour this corpus exists for.
#
# Four things become possible that whole-document retrieval cannot express at all:
#
#   1. DISTRACTOR passages from the SAME document -- plausible, adjacent, and wrong. Much harder than an
#      unrelated document, and much closer to what a real index returns.
#   2. Honest insufficiency: return passages that genuinely do not contain the answer, instead of hiding the
#      answer inside a document that does contain it. retrieval_insufficient stops being a fiction.
#   3. Multi-hop: the answer needs passage 4 AND passage 9.
#   4. A follow-up call for the NEIGHBOURING passage when one is cut off mid-point.
#
# It is also cheaper: ~800 words of passages in the prompt instead of ~3,000.

CHUNK_WORDS = 190          # ~250 tokens: large enough to carry a fact with its qualifier
CHUNK_MIN_WORDS = 60       # below this a passage is a heading with nothing under it


def chunk_document(text: str, *, target_words: int = CHUNK_WORDS) -> list[dict]:
    """Split a document into labelled passages, respecting its own section headings.

    Returns [{"id", "section", "text", "words"}]. The id is stable and sequential so a sample can record
    WHICH passages it was shown and which actually held the answer -- that is what makes looking in the wrong
    place scoreable rather than invisible.
    """
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in (text or "").splitlines():
        s = line.strip()
        m = re.fullmatch(r"=+\s*(.+?)\s*=+", s)
        if m:
            sections.append((m.group(1).strip(), []))
        elif s:
            sections[-1][1].append(s)

    def _split_long(para: str, limit: int) -> list[str]:
        """Break a paragraph that is itself longer than a passage, on sentence boundaries.

        Needed because paragraph breaks are not guaranteed: a 2,000-word unbroken paragraph would otherwise
        become one 2,000-word "passage", which defeats retrieving a passage at all.
        """
        if len(para.split()) <= limit * 1.4:
            return [para]
        out, buf, n = [], [], 0
        for sent in re.split(r"(?<=[.!?])\s+", para):
            w = len(sent.split())
            if n and n + w > limit:
                out.append(" ".join(buf))
                buf, n = [], 0
            buf.append(sent)
            n += w
        if buf:
            out.append(" ".join(buf))
        # A "sentence" can still be longer than a passage -- table dumps and run-on text have no terminal
        # punctuation at all. Fall back to a hard word split so passage size is bounded whatever the input.
        bounded: list[str] = []
        for piece in out:
            words = piece.split()
            if len(words) <= limit * 1.4:
                bounded.append(piece)
                continue
            for i in range(0, len(words), limit):
                bounded.append(" ".join(words[i:i + limit]))
        return bounded

    chunks: list[dict] = []
    for heading, paras in sections:
        paras = [piece for para in paras for piece in _split_long(para, target_words)]
        buf: list[str] = []
        n = 0
        for para in paras:
            w = len(para.split())
            # Keep paragraphs whole where possible: a fact split across a boundary is unanswerable from
            # either half, which would make the sample unfair rather than hard.
            if n and n + w > target_words * 1.4:
                chunks.append({"section": heading, "text": "\n".join(buf), "words": n})
                buf, n = [], 0
            buf.append(para)
            n += w
            if n >= target_words:
                chunks.append({"section": heading, "text": "\n".join(buf), "words": n})
                buf, n = [], 0
        if buf and n >= CHUNK_MIN_WORDS:
            chunks.append({"section": heading, "text": "\n".join(buf), "words": n})
        elif buf and chunks:
            chunks[-1]["text"] += "\n" + "\n".join(buf)
            chunks[-1]["words"] += n
    for i, c in enumerate(chunks, start=1):
        c["id"] = f"c{i}"
    return [c for c in chunks if c["words"] >= CHUNK_MIN_WORDS]


def chunks_containing(chunks: list[dict], phrases: Iterable[str]) -> list[str]:
    """Which passage ids contain any of `phrases`. Used to record the ANSWER-BEARING passages, so a reply
    that cites the wrong one can be scored down."""
    want = [_norm(p) for p in phrases if p and str(p).strip()]
    if not want:
        return []
    out = []
    for c in chunks:
        hay = _norm(c["text"])
        if any(w in hay for w in want):
            out.append(c["id"])
    return out


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", (s or "").casefold())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def render_retrieval(chunks: list[dict], *, scores: Optional[list[float]] = None) -> str:
    """Format passages as a retriever would return them: id, section, a score, then the text.

    The score is cosmetic but load-bearing for the lesson -- a real result is ranked, and the top hit is
    often not the one that answers the question.
    """
    out = []
    for i, c in enumerate(chunks):
        sc = scores[i] if scores and i < len(scores) else max(0.30, 0.92 - 0.11 * i)
        head = f"[{c['id']} | {c['section'] or 'untitled'} | score {sc:.2f}]"
        out.append(f"{head}\n{c['text']}")
    return "\n\n".join(out)
