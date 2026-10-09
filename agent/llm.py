"""Model backends.

The agent only needs one method:

    stream(prompt, *, stop, max_new_tokens, **kw) -> Iterator[str]

Everything else (non-streaming `complete`) is derived from it. Implement that
and any runtime works: HF transformers, vLLM, an HTTP endpoint, a stub.

Included:
    HFBackend      local transformers model (uses the model's own KV cache)
    ScriptedLLM    returns canned text; for tests and for driving the loop dry
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol, Sequence

from .protocol import EOS, STOP_STRINGS

__all__ = ["LLM", "HFBackend", "ScriptedLLM", "GenerationSettings"]


@dataclass
class GenerationSettings:
    """Decoding defaults. Chosen to match how the model is sampled in training
    logs; override per call with keyword arguments to stream()/complete()."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    do_sample: bool = True

    def merged(self, **overrides: Any) -> dict[str, Any]:
        out = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }
        out.update({k: v for k, v in overrides.items() if v is not None})
        if not out.get("do_sample") or out.get("temperature") == 0:
            # Greedy: leave the sampling knobs out entirely rather than passing
            # values transformers will warn about as unused.
            for key in ("temperature", "top_k", "top_p"):
                out.pop(key, None)
            out["do_sample"] = False
        return out


class LLM(Protocol):
    def stream(
        self, prompt: str, *, stop: Sequence[str] = (), max_new_tokens: int | None = None, **kw: Any
    ) -> Iterator[str]: ...

    def complete(
        self, prompt: str, *, stop: Sequence[str] = (), max_new_tokens: int | None = None, **kw: Any
    ) -> str: ...

    def count_tokens(self, text: str) -> int: ...


def _truncate_at_stop(text: str, stop: Sequence[str]) -> str:
    """Cut at the earliest stop string, keeping the stop string itself.

    Keeping it matters: the agent needs to see `</tool_call>` to know the call is
    complete rather than cut off by the token budget.
    """
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i + len(s))
    return text[:cut]


class _StopBuffer:
    """Stop the stream at the first stop string, emitting it and nothing after.

    The stop string is INCLUDED on purpose: the agent has to see a complete
    `</tool_call>` to tell a finished call from one the token budget cut off.
    Hiding scaffolding from the user is a separate concern, handled by the
    agent's visible-text filter, which needs whole tags to do its job.

    The holdback exists so a stop string split across two chunks ("</tool_" +
    "call>") is still recognised rather than streamed straight through.
    """

    def __init__(self, stop: Sequence[str]) -> None:
        self.stop = [s for s in stop if s]
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.buffer = ""
        self.done = False

    def push(self, chunk: str) -> str:
        """Add generated text; return what is safe to emit now."""
        if self.done:
            return ""
        self.buffer += chunk
        best = min(
            (i for i in (self.buffer.find(s) for s in self.stop) if i != -1),
            default=-1,
        )
        if best != -1:
            end = best + max(len(s) for s in self.stop if self.buffer.find(s) == best)
            out, self.buffer, self.done = self.buffer[:end], "", True
            return out
        if self.hold and len(self.buffer) > self.hold:
            out, self.buffer = self.buffer[: -self.hold], self.buffer[-self.hold :]
            return out
        return ""

    def flush(self) -> str:
        if self.done:
            return ""
        out, self.buffer = self.buffer, ""
        return out


class HFBackend:
    """A local transformers model.

    Generation goes through model.generate() rather than a hand-rolled decode
    loop, because this model's positions are learned and ABSOLUTE: its own
    prepare_inputs_for_generation recomputes position_ids from the attention
    mask, and reimplementing the KV-cache plumbing would mean duplicating that
    correctly. Streaming uses TextIteratorStreamer on a worker thread.

    skip_special_tokens is False throughout -- the tags ARE the protocol.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        settings: GenerationSettings | None = None,
        max_ctx: int | None = None,
        device: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings or GenerationSettings()
        cfg_ctx = getattr(getattr(model, "config", None), "block_size", None) or 4096
        self.max_ctx = max_ctx or min(int(cfg_ctx), 4096)
        self.device = device or getattr(model, "device", None)

    # -- helpers ------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def eos_token(self) -> str:
        return getattr(self.tokenizer, "eos_token", None) or EOS

    def _encode(self, prompt: str) -> dict[str, Any]:
        # add_special_tokens=False: render() already emitted <s>, and a second
        # bos shifts every absolute position by one.
        enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        if len(enc["input_ids"][0]) > self.max_ctx:
            raise ValueError(
                f"prompt is {len(enc['input_ids'][0])} tokens, over the {self.max_ctx} limit. "
                "Lower Memory.max_tokens or summarise more aggressively -- truncating here would "
                "cut <s>/the system turn and the tool catalogue with it."
            )
        if self.device is not None:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        return dict(enc)

    def _generate_kwargs(self, stop: Sequence[str], max_new_tokens: int | None, kw: dict[str, Any]) -> dict[str, Any]:
        tok = self.tokenizer
        out = self.settings.merged(max_new_tokens=max_new_tokens, **kw)
        out.update(
            eos_token_id=getattr(tok, "eos_token_id", None),
            pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
            use_cache=True,
        )
        stops = [s for s in stop if s]
        if stops:
            # transformers stops ON these and leaves them off the output, so the
            # agent re-appends the matched stop string below.
            out.update(stop_strings=list(stops), tokenizer=tok)
        return out

    # -- generation ---------------------------------------------------------
    def complete(
        self, prompt: str, *, stop: Sequence[str] = (), max_new_tokens: int | None = None, **kw: Any
    ) -> str:
        import torch

        enc = self._encode(prompt)
        kwargs = self._generate_kwargs(stop, max_new_tokens, kw)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        new = out[0][enc["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new, skip_special_tokens=False)
        return self._finish(text, stop)

    def stream(
        self, prompt: str, *, stop: Sequence[str] = (), max_new_tokens: int | None = None, **kw: Any
    ) -> Iterator[str]:
        import torch
        from threading import Thread

        from transformers import TextIteratorStreamer

        enc = self._encode(prompt)
        kwargs = self._generate_kwargs(stop, max_new_tokens, kw)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        error: list[BaseException] = []

        def run() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(**enc, streamer=streamer, **kwargs)
            except BaseException as exc:  # surfaced on the consumer thread below
                error.append(exc)
                streamer.end()

        thread = Thread(target=run, daemon=True)
        thread.start()
        buf = _StopBuffer(stop)
        for chunk in streamer:
            emit = buf.push(chunk)
            if emit:
                yield self._clean(emit)
            if buf.done:
                break
        tail = buf.flush()
        if tail:
            yield self._clean(tail)
        thread.join(timeout=1.0)
        if error:
            raise error[0]

    # -- output hygiene -----------------------------------------------------
    _PAD = re.compile(r"(<pad>)+")

    def _clean(self, text: str) -> str:
        return self._PAD.sub("", text)

    def _finish(self, text: str, stop: Sequence[str]) -> str:
        """Trim anything generated past the first stop string (or EOS).

        _truncate_at_stop keeps the stop string itself. transformers' own
        stop_strings handling also stops ON the string rather than before it,
        so the two paths agree and the parser always sees a closed tag.
        """
        return _truncate_at_stop(self._clean(text), [*stop, self.eos_token])


@dataclass
class ScriptedLLM:
    """Returns pre-set replies, in order, ignoring the prompt.

    For tests, and for exercising an agent's tool wiring without a GPU. Records
    every prompt it was given in `prompts`, which is how tests assert that
    history and tool results were rendered the way training saw them.
    """

    replies: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    chunk_size: int = 7
    default: str = "<response>(no scripted reply left)"
    calls: int = 0

    def _next(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies[self.calls] if self.calls < len(self.replies) else self.default
        self.calls += 1
        return reply

    def complete(
        self, prompt: str, *, stop: Sequence[str] = (), max_new_tokens: int | None = None, **kw: Any
    ) -> str:
        return _truncate_at_stop(self._next(prompt), [*stop, EOS])

    def stream(
        self, prompt: str, *, stop: Sequence[str] = (), max_new_tokens: int | None = None, **kw: Any
    ) -> Iterator[str]:
        text = self._next(prompt)
        buf = _StopBuffer([*stop, EOS])
        for i in range(0, len(text), self.chunk_size):
            emit = buf.push(text[i : i + self.chunk_size])
            if emit:
                yield emit
            if buf.done:
                return
        tail = buf.flush()
        if tail:
            yield tail

    def count_tokens(self, text: str) -> int:
        from .memory import approx_tokens

        return approx_tokens(text)


def join(chunks: Iterable[str]) -> str:
    return "".join(chunks)
