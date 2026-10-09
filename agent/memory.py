"""Chat history under a token budget.

Three strategies, same interface, swap freely:

    WindowMemory       keep what fits, drop the oldest
    SummarizingMemory  keep what fits; summarise what falls out into a digest
    RecallMemory       as above, plus retrieve archived turns by relevance

The system turn (which carries the tool catalogue) is always pinned, and a
tool-call turn is never separated from its <tool_response>: dropping one of a
pair leaves the model looking at a call with no result, or a result with no
call, which is a shape it never saw in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

from .protocol import render

Message = dict[str, Any]
CountTokens = Callable[[str], int]

__all__ = ["Memory", "WindowMemory", "SummarizingMemory", "RecallMemory", "approx_tokens"]


def approx_tokens(text: str) -> int:
    """Rough token count for when no tokenizer is handy: ~4 chars per token.

    Only a fallback. Pass the real tokenizer's encoder as `count_tokens` and the
    budget means what it says; with this you should leave more headroom.
    """
    return max(1, len(text) // 4)


class Memory(Protocol):
    """What the agent needs from a history store."""

    def add(self, message: Message) -> None: ...
    def extend(self, messages: Iterable[Message]) -> None: ...
    def messages(self) -> list[Message]:
        """The messages to render for the next prompt, oldest first."""
        ...


def _is_pair_start(messages: Sequence[Message], index: int) -> bool:
    """True if messages[index] is an assistant tool-call turn whose result follows."""
    msg = messages[index]
    if msg.get("role") != "assistant":
        return False
    nxt = messages[index + 1] if index + 1 < len(messages) else None
    return bool(nxt and nxt.get("role") == "tool")


@dataclass
class WindowMemory:
    """Keep the newest turns that fit in `max_tokens`; drop the rest.

    Dropped messages go to `archived` so another strategy can use them.
    """

    max_tokens: int = 3072
    count_tokens: CountTokens = approx_tokens
    system: Message | None = None
    history: list[Message] = field(default_factory=list)
    archived: list[Message] = field(default_factory=list)
    # Headroom for the model's own answer, so the budget is not spent entirely
    # on history and the generation immediately runs into the context limit.
    reserve_tokens: int = 512

    def add(self, message: Message) -> None:
        if message.get("role") == "system":
            self.system = message
            return
        self.history.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        for m in messages:
            self.add(m)

    def _cost(self, messages: Sequence[Message]) -> int:
        if not messages:
            return 0
        return self.count_tokens(render(list(messages), add_generation_prompt=False))

    def _budget(self) -> int:
        fixed = self._cost([self.system]) if self.system else 0
        return max(0, self.max_tokens - self.reserve_tokens - fixed)

    def _prefix(self) -> list[Message]:
        """Messages that sit between the system turn and the live window."""
        return []

    def messages(self) -> list[Message]:
        kept = self._fit()
        out: list[Message] = []
        if self.system:
            out.append(self.system)
        out.extend(self._prefix())
        out.extend(kept)
        return out

    def _fit(self) -> list[Message]:
        """Walk backwards keeping whole turns until the budget is spent.

        Returns the kept window and moves anything evicted to `archived`
        (eviction is recorded once, so repeated calls stay idempotent).
        """
        budget = self._budget()
        kept: list[Message] = []
        index = len(self.history) - 1
        while index >= 0:
            # Keep a tool-call turn and its result together.
            start = index - 1 if (index > 0 and self.history[index].get("role") == "tool"
                                  and _is_pair_start(self.history, index - 1)) else index
            chunk = self.history[start : index + 1]
            if self._cost(chunk + kept) > budget and kept:
                break
            kept = chunk + kept
            index = start - 1
        evicted = self.history[: len(self.history) - len(kept)]
        for m in evicted:
            if m not in self.archived:
                self.archived.append(m)
        return kept

    def clear(self) -> None:
        self.history.clear()
        self.archived.clear()


@dataclass
class SummarizingMemory(WindowMemory):
    """WindowMemory that folds evicted turns into a running digest.

    Older turns are chunked (`chunk_messages` at a time) and summarised by the
    model itself; the digest is injected as a system-adjacent turn right after
    the real system message, so the tool catalogue stays first and the model
    still sees what happened earlier.

    `summarize` is called as summarize(prompt) -> str. Pass the agent's own LLM
    (`llm.complete`) or anything with the same shape.
    """

    summarize: Callable[[str], str] | None = None
    chunk_messages: int = 6
    digest: str = ""
    digest_label: str = "Earlier in this conversation"
    max_digest_tokens: int = 512
    _summarized: int = 0

    SUMMARY_PROMPT = (
        "Summarise the conversation below in at most 120 words. Keep facts, names, "
        "numbers, decisions and anything the user asked for that is still pending. "
        "Drop pleasantries. Write plain prose, no tags.\n\n{body}"
    )

    def _prefix(self) -> list[Message]:
        if not self.digest:
            return []
        return [{"role": "system", "content": f"{self.digest_label}: {self.digest}"}]

    def messages(self) -> list[Message]:
        # Fit first so eviction happens, then summarise whatever is newly archived.
        kept = self._fit()
        self._maybe_summarize()
        out: list[Message] = []
        if self.system:
            out.append(self.system)
        out.extend(self._prefix())
        out.extend(kept)
        return out

    def _maybe_summarize(self) -> None:
        pending = self.archived[self._summarized :]
        if self.summarize is None or len(pending) < self.chunk_messages:
            return
        take = len(pending) - (len(pending) % self.chunk_messages)
        chunk, self._summarized = pending[:take], self._summarized + take
        body = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in ([{"role": "digest", "content": self.digest}] if self.digest else []) + chunk
        )
        try:
            summary = self.summarize(self.SUMMARY_PROMPT.format(body=body)).strip()
        except Exception:
            # A failed summary must not break the turn: the digest simply does
            # not grow, and those messages stay only in `archived`.
            return
        if summary:
            self.digest = self._truncate(summary)

    def _truncate(self, text: str) -> str:
        if self.count_tokens(text) <= self.max_digest_tokens:
            return text
        words = text.split()
        while words and self.count_tokens(" ".join(words)) > self.max_digest_tokens:
            words = words[: int(len(words) * 0.9)] or words[:-1]
        return " ".join(words)


@dataclass
class RecallMemory(SummarizingMemory):
    """SummarizingMemory plus retrieval over archived turns.

    `embed` maps text to a vector (list of floats); archived messages scoring
    above `min_score` against the latest user turn are re-inserted after the
    digest. Bring your own embedder -- no dependency is added here.
    """

    embed: Callable[[str], Sequence[float]] | None = None
    recall_k: int = 3
    min_score: float = 0.25
    _vectors: list[tuple[Message, Sequence[float]]] = field(default_factory=list)

    def _prefix(self) -> list[Message]:
        out = super()._prefix()
        out.extend(self._recalled())
        return out

    def _recalled(self) -> list[Message]:
        if self.embed is None or not self.history:
            return []
        last_user = next(
            (m for m in reversed(self.history) if m.get("role") == "user"), None
        )
        if not last_user:
            return []
        self._index()
        try:
            query = self.embed(last_user.get("content", ""))
        except Exception:
            return []
        scored = [(self._cosine(query, vec), msg) for msg, vec in self._vectors]
        hits = [m for score, m in sorted(scored, key=lambda s: s[0], reverse=True)[: self.recall_k] if score >= self.min_score]
        if not hits:
            return []
        body = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in hits)
        return [{"role": "system", "content": f"Possibly relevant earlier turns: {body}"}]

    def _index(self) -> None:
        known = {id(m) for m, _ in self._vectors}
        for m in self.archived:
            if id(m) in known or not m.get("content"):
                continue
            try:
                self._vectors.append((m, self.embed(m["content"])))  # type: ignore[misc]
            except Exception:
                continue

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0
