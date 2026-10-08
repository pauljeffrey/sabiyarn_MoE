"""Drive the three-agent loop against the same plan rows the single-call generator uses.

Deliberately thin: the loop is in loop.py, the retrieval in retrieval.py, the prompts in agent_prompts.py.
This file only wires a provider to them and emits records in the schema postprocess already reads, so the
two generation strategies can be compared on identical rows.
"""

from __future__ import annotations

import os
import random
from typing import Any, Optional

from agents.loop import Loop, records_from, sft_record
from agents.retrieval import (HybridRetriever, Library, LocalRetriever, PineconeRetriever,
                              Retriever)
from prompts import _lang_spec
from schemas.seed import Seed


def build_ask(provider, *, temperature: float = 0.85, tag: str = "agent"):
    """A (messages, max_tokens) -> text callable over a provider.

    Temperature is 0.85 and not lower on purpose: the user agent and the assistant are being asked for
    VARIETY across a corpus, and a near-greedy decode gives the same careful question and the same flat
    answer every time.
    """
    from providers.base import Request

    n = {"i": 0}

    def ask(messages: list[dict], max_tokens: int = 700) -> str:
        """Retried HERE, at the single call, because one empty reply would otherwise end a conversation.

        Measured on the first live run: two of five conversations produced ZERO turns because the very first
        user-agent call came back empty, and the loop treats an empty turn as the end of the conversation.
        The provider already retries transport errors and empty 200s, but it gives up after its own budget
        and hands back ""; a conversation is worth more than that. Three attempts, and the temperature is
        nudged up slightly each time so a model stuck on an empty completion is not asked the identical
        question again.
        """
        n["i"] += 1
        for attempt in range(3):
            r = provider.complete(Request(custom_id=f"{tag}-{n['i']}-{attempt}", messages=messages,
                                          max_tokens=max_tokens,
                                          temperature=min(1.0, temperature + 0.05 * attempt)))
            if r.ok and (r.text or "").strip():
                return r.text
        return ""

    return ask


def tool_specs(seed: Seed, names: list[str]) -> list[dict]:
    """The seed's own tool definitions, so the agent sees exactly what the single-call path advertises."""
    out = []
    for t in (seed.tools or []):
        name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if name not in names:
            continue
        if isinstance(t, dict):
            params = t.get("parameters") or {}
            desc = t.get("description") or ""
        else:
            params = getattr(t, "parameters", None) or {}
            desc = getattr(t, "description", "") or ""
        props = (params.get("properties") if isinstance(params, dict) else None) or {}
        out.append({"name": name, "description": desc, "arguments": props})
    return out


def build_retrieval(*, limit: int = 0, prefer_pinecone: bool = True) -> tuple[Retriever, Library]:
    """The routed retriever and the page library, built once and shared across conversations.

    BM25 locally for the language-tagged corpora -- 29,535 of the pool's 51,807 passages carry a `doc_lang`
    (fra 10,675, eng 4,427, som 3,427, hau 3,207, ibo 2,968, yor 2,908, twi 1,923) -- and Pinecone for
    English, where its embedder is actually competent. Pinecone is optional: without it the local index
    serves everything, which is worse for English and identical for everything else.
    """
    local = LocalRetriever.from_pool(limit=limit)
    remote = None
    if prefer_pinecone and os.environ.get("PINECONE_API_KEY"):
        try:
            remote = PineconeRetriever()
        except Exception as exc:  # noqa: BLE001 -- any SDK/auth/network failure means "local only"
            print(f"  [retrieval] pinecone unavailable ({type(exc).__name__}), BM25 only")
    return HybridRetriever(local=local, remote=remote), Library.from_pool(limit=limit)


def ground_subject(retriever, lang: str, tags: list[str], rng: random.Random, fallback: str = "") -> str:
    """For a retrieval task, make the conversation about something the corpus actually holds.

    Measured: left to invent its own specifics, the user agent asked about Nigerian micro-savings apps,
    health insurance for elderly parents and pension claim forms -- all perfectly realistic, and none of them
    in a corpus of general Wikipedia. Every search returned nothing, the assistant correctly said so, and 6
    of 6 tool-using turns became "I could not find that". Honest behaviour, and a corpus made of it would
    teach the model that looking things up does not work.

    So a retrieval task is seeded with a real document's title in the right language, and the user agent is
    pointed at that subject. The question is still the user's to invent -- only the territory is fixed.
    """
    if not ({"rag", "tool-calling"} & set(tags or [])):
        return fallback
    local = getattr(retriever, "local", retriever)
    pool = [p for p in getattr(local, "passages", []) if p.lang == lang and p.title]
    if not pool:
        pool = [p for p in getattr(local, "passages", []) if p.title]
    if not pool:
        return fallback
    p = pool[rng.randrange(len(pool))]
    return f"{p.title}{' -- ' + p.section if p.section and len(p.section) < 40 else ''}"


def run_row(seed: Seed, row: dict, provider, *, retriever: Optional[Retriever] = None,
            library: Optional[Library] = None,
            n_turns: int = 4, mode: str = "rl", today: str = "2026-10-08",
            model: str = "", findings_for=None) -> dict[str, Any]:
    """One conversation for one plan row. Returns {"records": [...], "conversation": Conversation}."""
    lang = _lang_spec(seed, row["lang"])
    names = list(row.get("tools") or [])
    rng = random.Random(f"{row.get('custom_id') or row.get('id')}")
    if retriever is None:
        retriever, library = build_retrieval()
    subject = ground_subject(retriever, lang.code, list(row.get("tags") or []), rng,
                             row.get("subtopic") or "")
    loop = Loop(build_ask(provider), retriever, lang=lang.code, lang_name=lang.name,
                today=today, tools=tool_specs(seed, names), plan=list(row.get("task_plan") or []),
                subject=subject, rng=rng, mode=mode,
                tags=list(row.get("tags") or []), library=library)
    conv = loop.run(n_turns, findings_for=findings_for)
    kw = dict(row=row, lang=lang.code, target_lang=row.get("target_lang") or lang.code,
              model=model or getattr(provider, "model", ""), subject=subject,
              tasks=list(row.get("tasks") or ([row["task"]] if row.get("task") else [])),
              tools=names)
    if mode == "sft":
        rec = sft_record(conv, **kw)
        return {"records": [rec] if rec else [], "conversation": conv}
    return {"records": records_from(conv, **kw), "conversation": conv}
