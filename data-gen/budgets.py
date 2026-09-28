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

SUPPORTED_CONTEXTS = (16_384, 32_768)
# The target model's real context: block_size 32768 in the published config. Override with DATA_GEN_CONTEXT.
DEFAULT_CONTEXT = int(os.environ.get("DATA_GEN_CONTEXT", "32768"))


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
    # Prose documents for pretraining: (lo, hi) words. Independent of context; these are short by design.
    pretrain_words: tuple[int, int] = (300, 500)
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
        notes="prefix + two rankable candidates",
    )


def _pretrain(context: int) -> Budget:
    # Pretraining documents are deliberately short (300-500 words): the goal is breadth of world model across
    # 636 (domain, sub-topic) pairs, not long-context practice, and short documents pack cleanly into
    # block_size 4096 windows during pretraining.
    return Budget(
        kind="pretrain", context=context, max_response_tokens=2_048,
        doc_token_bands=((1_000, 2_000),), rag_context_tokens=(2_048, 8_192),
        max_output_tokens=4_096, pretrain_words=(300, 500),
        notes="short prose documents; context is not the binding constraint here",
    )


_BUILDERS = {"sft": _sft, "rl": _rl, "pretrain": _pretrain}


def budget_for(kind: str, context: Optional[int] = None) -> Budget:
    ctx = int(context or DEFAULT_CONTEXT)
    if ctx not in SUPPORTED_CONTEXTS:
        raise SystemExit(
            f"context {ctx:,} is not one of {', '.join(f'{c:,}' for c in SUPPORTED_CONTEXTS)}.\n"
            f"These are the two the whole pipeline is sized and tested for; anything else needs the training "
            f"side moved too (training/train_config.yaml sft_block_size, rl/config.py max_seq_len).")
    build = _BUILDERS.get(kind)
    if build is None:
        # `judge` and any future read-only phase: give it the sft shape, which is the largest.
        build = _sft
    return build(ctx)


def describe(context: Optional[int] = None) -> str:
    lines = [f"generation budgets at context {int(context or DEFAULT_CONTEXT):,}:"]
    for kind in ("pretrain", "sft", "rl"):
        b = budget_for(kind, context)
        lo, hi = b.doc_token_range
        lines.append(
            f"  {kind:9s} response<={b.max_response_tokens:>6,}  documents {lo:>6,}-{hi:<6,}  "
            f"rag context {b.rag_context_tokens[0]:,}-{b.rag_context_tokens[1]:,}  "
            f"completion<={b.max_output_tokens:,}")
    return "\n".join(lines)


if __name__ == "__main__":
    for c in SUPPORTED_CONTEXTS:
        print(describe(c))
        print()
