"""Length budgets per phase. ONE place where the target model's context length enters the pipeline.

Two different kinds of limit used to be scattered across the codebase and were easy to confuse:

  GENERATION limits  -- how big a sample is when it is created (a document's length, a response's ceiling).
  TRAINING limits    -- how much of an existing sample the trainer feeds the model (block_size,
                        max_prompt_len). Anything past those is truncated off the END, which for a
                        [document -> summary] sample removes the summary, the only part with a signal.

This module owns the GENERATION side, and it derives everything from the target model's context so the two
cannot drift. Training-side numbers live in training/train_config.yaml (sft_block_size) and rl/config.py; they
must be >= `Budget.context` or generated samples get truncated.

SabiYarn's block_size is 32,768 (sabiyarn/model/configuration.py, and the published config.json), so that is
the default. 16,384 is kept as a supported setting because it halves KV-cache cost during generation and
quarters attention cost during training, which is a real trade, not a legacy value.

    from budgets import budget_for
    b = budget_for("sft")                 # the default context
    b = budget_for("sft", context=16384)  # or pinned

Every number below is in TOKENS and is converted to words per language where prose is being asked for, because
Yoruba costs ~2.5 tokens/word against English's ~1.15 and a single word figure either truncates one or wastes
two thirds of the budget on the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Any value in this range, not a fixed pair. The ceiling is the model's own block_size (32,768 in
# sabiyarn/model/configuration.py and the published config.json); the learned absolute position embedding has
# no rows past it, so a larger number would be untrainable rather than merely expensive.
MIN_CONTEXT = 512
MAX_CONTEXT = 32_768

# The DEFAULT differs by phase because `context` means something different for each, and for pretraining the two
# meanings coincide:
#   sft / rl   the model's training window. A sample is a conversation that must fit inside it, so this is
#              32,768 and must match training/train_config.yaml's sft_block_size and rl/config.py's max_seq_len.
#   pretrain   a sample IS one document, so the length of a sample and the context it needs are the same
#              number. 1,024 tokens is about the 300-500 words this corpus was specified at for the mid-cost
#              languages -- Yoruba at ~2.5 tokens/word lands at ~400 words, Pidgin at ~1.35 runs longer,
#              because fixing tokens necessarily varies words and vice versa. It is NOT tied to the SFT window:
#              pretraining trains at block_size 4,096, and a 32,768-token pretraining document would be four
#              windows' worth of text in one sample.
_DEFAULT_BY_KIND = {"pretrain": 1_024, "sft": 32_768, "rl": 32_768}
DEFAULT_CONTEXT = int(os.environ.get("DATA_GEN_CONTEXT", "32768"))
# Tokens reserved for the prompt when sizing the engine window. Measured: a pretrain prompt is ~1,106 tokens.
_PROMPT_RESERVE = 1_536


@dataclass(frozen=True)
class Budget:
    """Every generation-side length for one (kind, context)."""

    kind: str
    context: int
    # Ceiling on ONE assistant response. Not a stylistic preference: a sample whose response alone fills the
    # context leaves no room for the conversation that prompted it.
    max_response_tokens: int
    # (lo, hi) token bands a long generated document is drawn from, weighted towards the short end.
    doc_token_bands: tuple[tuple[int, int], ...]
    # (lo, hi) tokens for a RAG context -- the passage a retrieval tool returns. Substantial on purpose: a
    # 200-token "passage" teaches nothing about reading a real document, and the owner's spec is 2,048-8,192.
    rag_context_tokens: tuple[int, int]
    # Completion ceiling for one generation request (vLLM max_tokens / API max_tokens).
    max_output_tokens: int
    # Target size of ONE pretraining document, in tokens. Converted to words per language at request time,
    # because Yoruba costs ~2.5 tokens/word against English's ~1.15.
    pretrain_tokens: int = 1_536
    # The window the ENGINE needs, which is not always the model's context. vLLM reserves KV cache for
    # max_model_len per sequence, so asking for 32,768 when the phase's longest sample is 5,200 tokens throws
    # away most of the cache and most of the concurrency: on a GB10 that is 25 concurrent sequences instead of
    # 59. Set per phase from what the phase actually produces.
    engine_len: int = 0
    notes: str = ""

    @property
    def doc_token_range(self) -> tuple[int, int]:
        return min(b[0] for b in self.doc_token_bands), max(b[1] for b in self.doc_token_bands)


def _sft(context: int) -> Budget:
    # A long-document sample is [system] + [document] + [request] + [summary] (+ follow-ups). Reserve room for
    # everything that is not the document, then give the document the rest:
    #   system message with the tool catalogue   ~1,200
    #   the user's own words around the document   ~200
    #   think + task_plan + markers                ~300
    #   the summary and any follow-up answers    = max_response_tokens x 2
    resp = 4_096 if context >= 32_768 else 2_048
    overhead = 1_700 + resp * 2
    doc_ceiling = context - overhead
    # Four bands weighted 5:3:2:1 towards the short end -- a document at the ceiling costs ~6x one at the floor
    # and is ~6x rarer in practice, so weighting them equally would spend most of the budget on the rarest case.
    # ELEVEN entries, not ten: the document LANGUAGE cycle in longdocs.py is 10 long, and a size cycle that is
    # also 10 long has the same period in `index`, which locks the two together (measured: no Fon row ever drew
    # a long English document). 11 is coprime with 10.
    lo = 4_000
    q = (doc_ceiling - lo) / 4.0
    bands = tuple([(int(lo), int(lo + q))] * 5
                  + [(int(lo + q), int(lo + 2 * q))] * 3
                  + [(int(lo + 2 * q), int(lo + 3 * q))] * 2
                  + [(int(lo + 3 * q), int(doc_ceiling))])
    return Budget(
        kind="sft", context=context, max_response_tokens=resp, doc_token_bands=bands,
        rag_context_tokens=(2_048, 8_192),
        # One request produces the conversation only -- the document arrives as INPUT via the placeholder --
        # so this holds the summary, the follow-ups and the JSON scaffolding, not the document.
        max_output_tokens=min(context, resp * 2 + 2_048),
        # A summarisation sample fills the context by construction: doc_ceiling was derived as
        # context - overhead, so document + overhead is exactly the context.
        engine_len=context,
        notes=f"documents {bands[0][0]:,}-{doc_ceiling:,} tokens inside a {context:,} context",
    )


def _rl(context: int) -> Budget:
    # RL is a conversation PREFIX plus two candidate final responses. Both candidates must fit alongside the
    # prefix, so the per-response ceiling is half what SFT allows at the same context.
    resp = 2_048 if context >= 32_768 else 1_024
    # RL has no long_document_summarization task (see build_seeds._RL_WEIGHTS), so its documents exist only to
    # back a RAG context; the band is the RAG band.
    return Budget(
        kind="rl", context=context, max_response_tokens=resp,
        doc_token_bands=((2_048, 8_192),), rag_context_tokens=(2_048, 8_192),
        max_output_tokens=min(context, resp * 2 + 2_048),
        # A prefix can carry an 8,192-token RAG context on top of a conversation that measured p99 6,762, plus
        # two candidates: it genuinely can approach the full context.
        engine_len=context,
        notes="prefix + two rankable candidates",
    )


def _pretrain(context: int) -> Budget:
    # A pretraining sample is one document, so `context` IS the document's target size. The goal is breadth of
    # world model across 636 (domain, sub-topic) pairs, not long-context practice, and short documents pack
    # cleanly into the block_size 4,096 windows pretraining actually uses.
    #
    # The engine window is computed from what a request needs -- prompt plus completion -- and not from the
    # document size alone, so a 1,000-token document does not ask for a 1,000-token window that its own
    # ~1,100-token prompt would not fit in. Every token saved here is concurrency: on a GB10, a 3,072-token
    # window holds ~2.7x the sequences of an 8,192-token one.
    # A completion has to hold the document plus its title and the JSON scaffolding, and 4,096 is the ceiling
    # worth paying for: pretraining trains at block_size 4,096, so a longer document is split across windows
    # anyway. The document target is clamped to what the completion can actually deliver, so the reported band
    # is never a size the run cannot produce.
    doc = min(context, int((4_096 - 256) / 1.35))
    out = _round_up(int(doc * 1.35) + 256, 128)
    engine = _round_up(_PROMPT_RESERVE + out, 256)
    return Budget(
        kind="pretrain", context=context, max_response_tokens=min(2_048, out),
        doc_token_bands=((max(256, doc // 2), doc),), rag_context_tokens=(2_048, 8_192),
        max_output_tokens=out, pretrain_tokens=doc, engine_len=engine,
        notes=(f"one ~{doc:,}-token document per sample; {engine:,}-token engine window"
               + ("" if doc == context else f" (clamped from {context:,}: block_size 4,096 splits longer ones)")),
    )


def _round_up(n: int, to: int) -> int:
    return -(-n // to) * to


_BUILDERS = {"sft": _sft, "rl": _rl, "pretrain": _pretrain}


def budget_for(kind: str, context: Optional[int] = None) -> Budget:
    if context:
        ctx = int(context)
    elif os.environ.get("DATA_GEN_CONTEXT"):
        ctx = int(os.environ["DATA_GEN_CONTEXT"])
    else:
        ctx = _DEFAULT_BY_KIND.get(kind, DEFAULT_CONTEXT)
    if not MIN_CONTEXT <= ctx <= MAX_CONTEXT:
        raise SystemExit(
            f"context {ctx:,} is outside {MIN_CONTEXT:,}-{MAX_CONTEXT:,}.\n"
            f"{MAX_CONTEXT:,} is the model's own block_size -- its learned absolute position embedding has no "
            f"rows past it, so a larger context is untrainable, not merely expensive.")
    if kind in ("sft", "rl") and ctx < 8_192:
        raise SystemExit(
            f"context {ctx:,} is too small for {kind}: a conversation carries a system message with the tool "
            f"catalogue (~1,200 tokens) plus several tool results before any answer. Use 8,192 or more, and "
            f"remember the TRAINING side must match (training/train_config.yaml sft_block_size, "
            f"rl/config.py max_seq_len).")
    build = _BUILDERS.get(kind)
    if build is None:
        # `judge` and any future read-only phase: give it the sft shape, which is the largest.
        build = _sft
    return build(ctx)


def describe(context: Optional[int] = None) -> str:
    lines = [f"generation budgets at context {context:,}:" if context else
             "generation budgets at each phase's default context:"]
    for kind in ("pretrain", "sft", "rl"):
        b = budget_for(kind, context)
        lo, hi = b.doc_token_range
        lines.append(
            f"  {kind:9s} response<={b.max_response_tokens:>6,}  documents {lo:>6,}-{hi:<6,}  "
            f"rag ctx {b.rag_context_tokens[0]:,}-{b.rag_context_tokens[1]:,}  "
            f"completion<={b.max_output_tokens:>6,}  engine window {b.engine_len:,}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    for c in (1_000, 2_048, 16_384, 32_768):
        try:
            print(describe(c))
        except SystemExit as exc:
            print(f"context {c:,}: {exc}")
        print()
