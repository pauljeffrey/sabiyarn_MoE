"""Retrieval for the agent loop: find passages, hand back a ranked result a model can read.

TWO IMPLEMENTATIONS, one interface.

`LocalRetriever` scores chunks with BM25 over the document pool already on disk. No service, no keys, no
network, deterministic, and it is what the tests and the offline path use.

`PineconeRetriever` queries a Pinecone index with integrated embedding (llama-text-embed-v2, which is
multilingual -- the reason it is worth having, since half the queries are Yoruba or Hausa). What it buys over
the local path is CROSS-DOCUMENT retrieval: the distractors come from other documents that are genuinely
near in embedding space, instead of other sections of the same article. "The retrieved passage does not
answer the question" is a lesson this corpus is built to teach, and it is only honest if the near-miss is a
real near-miss.

WHY THE RETRIEVED TEXT IS RECORDED SEPARATELY. During generation the passages are evicted from the agent's
history after the turn that cited them -- that eviction is what keeps the loop cheap (a document left in
context is a 2x cost swing, measured). But the published sample must still CONTAIN the context, or an SFT
record teaches the model to answer from nowhere. So every turn records what it retrieved in `Turn.retrieved`,
and the record builder writes those passages back into the published conversation as the tool result they
were. Evicted from the generator's working memory, present in the data.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol

_HERE = Path(__file__).resolve().parent
_POOL = _HERE.parent / "data" / "document_pool.jsonl"

_WORD = re.compile(r"[0-9A-Za-zÀ-ɏĀ-ſ']+")


def _tok(s: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(s or "")]


@dataclass
class Passage:
    """One retrievable unit. `doc_id`/`section` are what a citation points at."""
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float = 0.0
    lang: str = ""
    url: str = ""

    def words(self) -> int:
        return len(self.text.split())


class Retriever(Protocol):
    def search(self, query: str, *, k: int = 4, lang: str = "") -> list[Passage]: ...


# ---------------------------------------------------------------------------- chunking


def split_passages(text: str, *, target_words: int = 110, doc_id: str = "", title: str = "",
                   lang: str = "", url: str = "") -> list[Passage]:
    """Split on the document's own headings, then pack paragraphs up to `target_words`.

    Paragraph-level, not sentence-level: a sentence ripped out of a paragraph loses the referent of every
    pronoun in it, and a model asked to answer from it either invents the referent or refuses. 110 words is
    about the size at which a passage still answers a question on its own.
    """
    out: list[Passage] = []
    section = ""
    buf: list[str] = []
    n = 0

    def flush() -> None:
        nonlocal buf, n
        if buf:
            out.append(Passage(id=f"p{len(out) + 1}", doc_id=doc_id, title=title, section=section,
                               text="\n".join(buf).strip(), lang=lang, url=url))
        buf, n = [], 0

    for block in re.split(r"\n\s*\n", text or ""):
        b = block.strip()
        if not b:
            continue
        # a short line with no terminal punctuation is a heading, not a paragraph
        if len(b.split()) <= 12 and not b.endswith((".", "!", "?", ":", ";")) and "\n" not in b:
            flush()
            section = b
            continue
        w = len(b.split())
        if n and n + w > target_words:
            flush()
        buf.append(b)
        n += w
    flush()
    return out


# ---------------------------------------------------------------------------- local BM25


@dataclass
class LocalRetriever:
    """BM25 over the document pool. Deterministic, offline, and the fallback when Pinecone is unavailable."""

    passages: list[Passage] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75
    _df: Counter = field(default_factory=Counter, repr=False)
    _toks: list[list[str]] = field(default_factory=list, repr=False)
    _avg: float = 0.0

    @classmethod
    def from_pool(cls, path: Path = _POOL, *, limit: int = 0, target_words: int = 110) -> "LocalRetriever":
        passages: list[Passage] = []
        if path.exists():
            for i, line in enumerate(path.open(encoding="utf-8")):
                if limit and i >= limit:
                    break
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                doc_id = str(d.get("pageid") or d.get("doc_id") or i)
                # `doc_lang`, not `lang` -- the pool carries 2,813 language-tagged documents (yor 548,
                # som 540, fra 468, hau 450, ibo 446, twi 361, eng 260) harvested from native Wikipedias,
                # and reading the wrong key made every one of them look untagged.
                for p in split_passages(d.get("text") or "", target_words=target_words, doc_id=doc_id,
                                        title=str(d.get("title") or ""),
                                        lang=str(d.get("doc_lang") or d.get("lang") or ""),
                                        url=str(d.get("url") or "")):
                    p.id = f"{doc_id}#{p.id}"
                    passages.append(p)
        return cls.build(passages)

    @classmethod
    def build(cls, passages: list[Passage]) -> "LocalRetriever":
        r = cls(passages=passages)
        r._toks = [_tok(f"{p.title} {p.section} {p.text}") for p in passages]
        for t in r._toks:
            r._df.update(set(t))
        r._avg = (sum(len(t) for t in r._toks) / len(r._toks)) if r._toks else 0.0
        return r

    def search(self, query: str, *, k: int = 4, lang: str = "") -> list[Passage]:
        q = _tok(query)
        if not q or not self.passages:
            return []
        n = len(self.passages)
        scored: list[tuple[float, int]] = []
        for i, toks in enumerate(self._toks):
            if lang and self.passages[i].lang and self.passages[i].lang != lang:
                continue
            tf = Counter(toks)
            dl = len(toks) or 1
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[w] + 0.5) / (self._df[w] + 0.5))
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self._avg or 1)))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        hits = []
        top = scored[0][0] if scored else 1.0
        for s, i in scored[:k]:
            p = self.passages[i]
            hits.append(Passage(p.id, p.doc_id, p.title, p.section, p.text,
                                score=round(s / (top or 1), 3), lang=p.lang, url=p.url))
        return hits


# ---------------------------------------------------------------------------- pinecone


class PineconeRetriever:
    """Pinecone with integrated embedding. Imported lazily so the module works without the SDK installed."""

    def __init__(self, index: str = "", namespace: str = "docs", model: str = ""):
        # The key lives in the one repo-root .env, and providers.base.load_env is what knows that. Reading
        # os.environ without it works only when the caller happened to load it first, which the ingest CLI
        # did not -- so a correct key looked like a missing one.
        from providers.base import load_env  # noqa: PLC0415

        load_env()
        try:
            from pinecone import Pinecone  # noqa: PLC0415 -- optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "the pinecone SDK is not installed: pip install -r requirements.txt") from exc

        key = os.environ.get("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("PINECONE_API_KEY is not set in the repo-root .env")
        self.model = model or os.environ.get("PINECONE_MODEL", "llama-text-embed-v2")
        self.name = index or os.environ.get("PINECONE_INDEX", "hera")
        self.namespace = namespace
        self._pc = Pinecone(api_key=key)
        self._index = self._pc.Index(self.name)

    def search(self, query: str, *, k: int = 4, lang: str = "") -> list[Passage]:
        flt = {"lang": {"$eq": lang}} if lang else None
        res = self._index.search(
            namespace=self.namespace,
            query={"inputs": {"text": query}, "top_k": k, **({"filter": flt} if flt else {})},
            fields=["text", "title", "section", "doc_id", "lang", "url"],
        )
        d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        out = []
        for hit in ((d.get("result") or {}).get("hits") or []):
            h = dict(hit)
            f = dict(h.get("fields") or {})
            # SDK 10 names these `id_` and `score_`; earlier versions used `_id` and `_score`. Reading both
            # is three characters of defence against an upgrade silently returning every score as 0.0 --
            # which is exactly what the first live query did.
            pid = h.get("id_") or h.get("_id") or h.get("id") or ""
            score = h.get("score_") or h.get("_score") or h.get("score") or 0.0
            out.append(Passage(id=str(pid), doc_id=str(f.get("doc_id") or ""),
                               title=str(f.get("title") or ""), section=str(f.get("section") or ""),
                               text=str(f.get("text") or ""), score=round(float(score), 3),
                               lang=str(f.get("lang") or ""), url=str(f.get("url") or "")))
        return out

    # ---- ingest

    def ensure_index(self, *, region: str = "") -> None:
        from pinecone import ServerlessSpec  # noqa: PLC0415

        region = region or os.environ.get("PINECONE_REGION", "us-east-1")
        if self.name in [i["name"] for i in self._pc.list_indexes()]:
            return
        self._pc.create_index_for_model(
            name=self.name,
            cloud="aws", region=region,
            embed={"model": self.model, "field_map": {"text": "text"}},
            spec=ServerlessSpec(cloud="aws", region=region),
        )

    def upsert(self, passages: Iterable[Passage], *, batch: int = 90, progress_every: int = 20,
               max_retries: int = 4) -> int:
        """Upsert with integrated embedding -- Pinecone embeds server-side, so nothing is embedded here.

        The document pool is 51,807 passages, which is ~576 batched requests against a service that embeds
        each one as it arrives. At that length a silent loop is indistinguishable from a hang and one
        rate-limit response should not cost the whole ingest, so: progress, and bounded retry with backoff.
        """
        import time  # noqa: PLC0415

        def flush(rows: list[dict]) -> int:
            for attempt in range(max_retries):
                try:
                    # keyword-only since SDK 10; passing positionally raises PineconeValueError
                    self._index.upsert_records(records=rows, namespace=self.namespace)
                    return len(rows)
                except Exception as exc:  # noqa: BLE001 -- rate limits and transient 5xx both land here
                    # A VALIDATION error is not transient: retrying client-side misuse four times with
                    # backoff just turns a clear failure into a slow one. Only retry what can change.
                    if "ValueError" in type(exc).__name__ or isinstance(exc, (TypeError, ValueError)):
                        print(f"  [pinecone] not retryable: {type(exc).__name__}: {exc}")
                        return 0
                    if attempt == max_retries - 1:
                        print(f"  [pinecone] giving up on a batch of {len(rows)}: "
                              f"{type(exc).__name__}: {exc}")
                        return 0
                    wait = 2 ** attempt
                    print(f"  [pinecone] {type(exc).__name__}, retrying in {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})", flush=True)
                    time.sleep(wait)
            return 0

        buf: list[dict] = []
        total = batches = 0
        t0 = time.time()
        for p in passages:
            if not (p.text or "").strip():
                continue
            buf.append({"_id": p.id, "text": p.text, "title": p.title, "section": p.section,
                        "doc_id": p.doc_id, "lang": p.lang or "", "url": p.url or ""})
            if len(buf) >= batch:
                total += flush(buf)
                batches += 1
                buf = []
                if progress_every and batches % progress_every == 0:
                    rate = total / max(1e-6, time.time() - t0)
                    print(f"  upserted {total:,} passages ({rate:,.0f}/s)", flush=True)
        if buf:
            total += flush(buf)
        return total


def get_retriever(prefer_pinecone: bool = True, **kw) -> Retriever:
    """Pinecone when it is configured AND reachable, otherwise the local index.

    Falling back rather than failing is deliberate: retrieval quality changes how realistic the sample is,
    but a missing service should not stop a run, and the local path is good enough to generate against.
    """
    if prefer_pinecone and os.environ.get("PINECONE_API_KEY"):
        try:
            return PineconeRetriever(**kw)
        except Exception as exc:  # noqa: BLE001 -- any SDK/network/auth failure means "use local"
            print(f"  [retrieval] pinecone unavailable ({type(exc).__name__}), using the local index")
    return LocalRetriever.from_pool()


def render(passages: list[Passage], *, note: str = "") -> str:
    """Format passages the way a retriever returns them: ranked, labelled, scored.

    The id is what a citation points at, and the score is what makes "the top hit is not always the answer"
    visible to the model reading it.
    """
    if not passages:
        return "No passages matched." + (f" {note}" if note else "")
    out = []
    for p in passages:
        head = f"[{p.id}] score={p.score:.2f}"
        if p.title:
            head += f" | {p.title}"
        if p.section:
            head += f" > {p.section}"
        out.append(f"{head}\n{p.text}")
    return "\n\n".join(out)


def _ingest_cli() -> None:
    """python -m agents.retrieval --ingest [--limit N] [--dry-run]

    Chunks the document pool and upserts it with integrated embedding. Pinecone embeds server-side, so there
    is no embedding model or GPU on this side and no vector ever crosses the wire.
    """
    import argparse

    ap = argparse.ArgumentParser(description="ingest the document pool into Pinecone")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="documents to read (0 = all)")
    ap.add_argument("--target-words", type=int, default=110)
    ap.add_argument("--namespace", default="docs")
    ap.add_argument("--dry-run", action="store_true", help="chunk and report, upsert nothing")
    a = ap.parse_args()

    rows = []
    if _POOL.exists():
        for i, line in enumerate(_POOL.open(encoding="utf-8")):
            if a.limit and i >= a.limit:
                break
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    passages: list[Passage] = []
    for i, d in enumerate(rows):
        doc_id = str(d.get("pageid") or d.get("doc_id") or i)
        for p in split_passages(d.get("text") or "", target_words=a.target_words, doc_id=doc_id,
                                title=str(d.get("title") or ""), lang=str(d.get("lang") or "")):
            p.id = f"{doc_id}#{p.id}"
            passages.append(p)
    words = sum(p.words() for p in passages)
    print(f"{len(rows):,} documents -> {len(passages):,} passages, {words:,} words "
          f"(median {sorted(p.words() for p in passages)[len(passages) // 2] if passages else 0} per passage)")
    if a.dry_run or not a.ingest:
        print("dry run: nothing upserted")
        return
    r = PineconeRetriever(namespace=a.namespace)
    r.ensure_index()
    n = r.upsert(passages)
    print(f"upserted {n:,} passages into {r.name}/{a.namespace} with {r.model}")


if __name__ == "__main__":
    _ingest_cli()


# ---------------------------------------------------------------------------- routing and the web

# Languages the Pinecone embedder actually handles. Measured against llama-text-embed-v2 over this corpus:
# an English query returns the right document at 0.39-0.60 similarity, Pidgin at 0.47 (it is
# English-lexified), and the same question in Hausa or Yoruba returns the WRONG document at ~0.17 -- the
# noise floor -- with the right document sitting in the index. So a query in anything else goes to BM25,
# where exact word overlap with a same-language document is a real signal rather than a dead one.
EMBEDDABLE = {"eng", "pcm", "fra", ""}


@dataclass
class HybridRetriever:
    """BM25 for local-language corpora, Pinecone for English. Routed by the LANGUAGE OF THE CORPUS.

    Not by the language of the conversation: a Hausa speaker asking about Nigerian tax law is best served by
    the English corpus, because that is where the document exists. What the language decides is which INDEX
    can find it -- and for Hausa or Yoruba text the embedder is blind, so lexical matching against
    same-language documents is the only thing that works.
    """

    local: LocalRetriever
    remote: Optional[Retriever] = None
    remote_failures: int = 0
    max_remote_failures: int = 3

    def search(self, query: str, *, k: int = 4, lang: str = "") -> list[Passage]:
        if lang and lang not in EMBEDDABLE:
            return self.local.search(query, k=k, lang=lang)
        if self.remote is not None:
            try:
                hits = self.remote.search(query, k=k, lang=lang if lang in EMBEDDABLE else "")
                if hits:
                    return hits
            except Exception as exc:  # noqa: BLE001
                # A CIRCUIT BREAKER, not a retry. The failure that actually happened was the org's monthly
                # embedding quota running out mid-run -- 5M tokens for llama-text-embed-v2, against a corpus
                # that needs ~7.8M to ingest -- and a quota does not come back on the next call. Retrying it
                # per query would turn one hard stop into thousands of slow ones, so after a few failures
                # the remote is switched off for the rest of the process and BM25 carries the run.
                self.remote_failures += 1
                print(f"  [retrieval] pinecone failed ({type(exc).__name__}), "
                      f"{self.remote_failures}/{self.max_remote_failures} before falling back to BM25",
                      flush=True)
                if self.remote_failures >= self.max_remote_failures:
                    print("  [retrieval] pinecone disabled for this run; BM25 only", flush=True)
                    self.remote = None
        return self.local.search(query, k=k, lang=lang if lang else "")

    def search_many(self, queries: list[tuple[str, str]], *, k: int = 4) -> list[Passage]:
        """Several (query, lang) pairs, merged and re-ranked.

        The assistant is allowed to issue more than one query because one question often spans two corpora:
        the figure is in an English document and the local practice is described in a Hausa one. Merging by
        score keeps the best of each rather than privileging whichever ran first.
        """
        seen: dict[str, Passage] = {}
        for q, lg in queries[:4]:
            for p in self.search(q, k=k, lang=lg):
                if p.id not in seen or p.score > seen[p.id].score:
                    seen[p.id] = p
        return sorted(seen.values(), key=lambda p: -p.score)[:k]


@dataclass
class Library:
    """Documents by url, for the FETCH half of a two-step web search."""

    by_url: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_pool(cls, path: Path = _POOL, *, limit: int = 0) -> "Library":
        by_url: dict[str, dict] = {}
        if path.exists():
            for i, line in enumerate(path.open(encoding="utf-8")):
                if limit and i >= limit:
                    break
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("url"):
                    by_url[str(d["url"])] = d
        return cls(by_url=by_url)

    def page(self, url: str, *, max_words: int = 1400) -> str:
        """A fetched page: the real article, with the clutter a real page carries.

        The clutter is not decoration. A fetched page is a whole article with navigation and a cookie notice
        in it, and reading past that to the paragraph that answers the question is half of what the sample
        teaches -- a tidy three-line extract teaches nothing, because no real page looks like that.
        """
        d = self.by_url.get(url)
        if not d:
            return (f"HTTP 404 -- {url} could not be fetched.\n"
                    "Nothing was returned for this address.")
        text = " ".join(str(d.get("text") or "").split()[:max_words])
        title = str(d.get("title") or "")
        return (f"{title}\n{'=' * len(title)}\n"
                f"Home | Contents | Current events | Random article | About\n"
                f"[This site uses cookies. Accept | Manage preferences]\n\n"
                f"{text}\n\n"
                f"Categories: {title} | Retrieved from {url}\n"
                f"This page was last edited on 14 February 2026.")


def search_results(hits: list[Passage], *, library: Optional[Library] = None) -> str:
    """A SERP, not an answer: title, url, and a snippet per result -- no bodies.

    The two-step is the lesson. A real assistant searches, reads the titles and snippets, decides which
    result is most likely to hold the answer, and fetches THAT one. Handing back the full text of four
    documents at once removes the decision, and the decision is the behaviour being taught -- including the
    chance to pick wrong, which is what makes picking right scoreable.
    """
    if not hits:
        return "No results."
    out = []
    for i, p in enumerate(hits, 1):
        snippet = " ".join(p.text.split()[:28])
        out.append(f"{i}. {p.title or '(untitled)'}{' > ' + p.section if p.section else ''}\n"
                   f"   {p.url or '(no url)'}\n"
                   f"   {snippet}...")
    return "\n".join(out) + "\n\nFetch one of these addresses to read it."
