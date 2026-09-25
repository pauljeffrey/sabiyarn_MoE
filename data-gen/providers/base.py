"""Provider abstraction: one interface over Together AI, OpenRouter and anything else OpenAI-compatible.

Two execution modes, because they have very different economics:

  * `complete_many(...)` -- synchronous, many requests in flight at once. Works everywhere, results come
    back in minutes, full price. This is what runs on Modal / RunPod / vast, where you are already paying
    for a box and want the corpus now.
  * `submit_batch(...)` / `poll_batch(...)` -- the provider's batch queue. Roughly half price with a 24h
    completion window. Only some providers have it (Together does, OpenRouter does not).

Both modes take and return the same `Request` / `Response` objects, so a generation job can switch between
them by changing one flag, and a job interrupted in either mode resumes from the same shard files.

Keys come from the repo-root .env and nowhere else (see data-gen/README): TOGETHER_API_KEY,
OPENROUTER_API_KEY, HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parents[2]  # repo root -- the single .env lives here


def load_env() -> None:
    """Load the ONE .env at the repo root. Idempotent."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


@dataclass
class Request:
    """One generation request. `custom_id` is how a response is matched back to its plan row."""
    custom_id: str
    messages: list[dict[str, Any]]
    max_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 0.95
    response_format: Optional[dict[str, Any]] = None   # {"type": "json_object"} or a json_schema
    tools: Optional[list[dict[str, Any]]] = None
    metadata: dict[str, Any] = field(default_factory=dict)  # carried through untouched (lang, task, ...)


@dataclass
class Response:
    custom_id: str
    text: str
    ok: bool = True
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    raw_tool_calls: Optional[list[dict]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = {"custom_id": self.custom_id, "ok": self.ok, "text": self.text, "model": self.model,
             "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
             "metadata": self.metadata}
        if self.error:
            d["error"] = self.error
        if self.raw_tool_calls:
            d["tool_calls"] = self.raw_tool_calls
        return json.dumps(d, ensure_ascii=False)


class RateLimit(Exception):
    pass


class Provider:
    """Subclass and set `name`, `base_url`, `env_key`. Everything else is shared."""

    name: str = "base"
    base_url: str = ""
    env_key: str = ""
    supports_batch: bool = False
    # USD per 1M tokens (input, output). VERIFY before trusting any cost figure -- provider pricing moves.
    pricing: dict[str, tuple[float, float]] = {}
    default_pricing: tuple[float, float] = (0.20, 0.60)

    def __init__(self, model: str, *, api_key: Optional[str] = None, max_concurrency: int = 16,
                 max_retries: int = 5, timeout: float = 180.0):
        load_env()
        self.model = model
        self.api_key = api_key or os.environ.get(self.env_key, "")
        if not self.api_key:
            raise SystemExit(
                f"{self.name}: no API key. Put {self.env_key}=... in {ROOT/'.env'} "
                f"(one .env for the whole repo).")
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.timeout = timeout
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- wire ---------------------------------------------------------------
    def _client(self):
        from openai import OpenAI  # OpenAI-compatible: works for Together and OpenRouter alike
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout,
                      max_retries=0, default_headers=self.extra_headers())

    def extra_headers(self) -> dict[str, str]:
        return {}

    def _payload(self, req: Request) -> dict[str, Any]:
        p: dict[str, Any] = {"model": self.model, "messages": req.messages,
                             "max_tokens": req.max_tokens, "temperature": req.temperature, "top_p": req.top_p}
        if req.response_format:
            p["response_format"] = req.response_format
        if req.tools:
            p["tools"] = req.tools
        return p

    def complete(self, req: Request) -> Response:
        """One request, with retry/backoff. Never raises for an API failure -- returns ok=False."""
        client = self._client()
        delay = 1.5
        last = ""
        for attempt in range(self.max_retries):
            try:
                r = client.chat.completions.create(**self._payload(req))
                choice = r.choices[0]
                calls = [tc.model_dump() for tc in (choice.message.tool_calls or [])] or None
                usage = getattr(r, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                with self._lock:
                    self.prompt_tokens += pt
                    self.completion_tokens += ct
                return Response(req.custom_id, choice.message.content or "", True, None, pt, ct,
                                getattr(r, "model", self.model), calls, req.metadata)
            except Exception as exc:  # noqa: BLE001 -- provider SDKs raise a wide zoo of errors
                last = f"{type(exc).__name__}: {exc}"
                transient = any(s in last.lower() for s in
                                ("rate", "429", "timeout", "timed out", "502", "503", "504",
                                 "overload", "connection", "temporarily"))
                if not transient or attempt == self.max_retries - 1:
                    break
                time.sleep(delay + random.uniform(0, delay))  # jitter: 100s of workers must not sync up
                delay = min(delay * 2, 60)
        return Response(req.custom_id, "", False, last, metadata=req.metadata)

    def complete_many(self, requests: Iterable[Request], *, on_result: Optional[Callable[[Response], None]] = None,
                      progress: bool = True) -> Iterator[Response]:
        """Run many requests concurrently, yielding as they finish (order not preserved)."""
        reqs = list(requests)
        if not reqs:
            return
        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {pool.submit(self.complete, r): r for r in reqs}
            for fut in as_completed(futures):
                resp = fut.result()
                done += 1
                if on_result:
                    on_result(resp)
                if progress and (done % 25 == 0 or done == len(reqs)):
                    rate = done / max(time.time() - t0, 1e-9)
                    eta = (len(reqs) - done) / max(rate, 1e-9)
                    extra = ""
                    try:  # surface systematic validation failures while there is still time to stop
                        from postprocess_gen import summary as _drops
                        d = _drops()
                        extra = f"  drops: {d}" if d != "no drops" else ""
                    except Exception:
                        pass
                    print(f"  [{self.name}] {done}/{len(reqs)}  {rate:.1f}/s  eta {eta/60:.1f}m  "
                          f"${self.cost_usd():.4f}{extra}", flush=True)
                yield resp

    # -- batch (providers that have a queue override these) ------------------
    def submit_batch(self, requests: Iterable[Request], workdir: Path) -> str:
        raise NotImplementedError(f"{self.name} has no batch API; use complete_many()")

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} has no batch API")

    def fetch_batch(self, batch_id: str, out_path: Path) -> list[Response]:
        raise NotImplementedError(f"{self.name} has no batch API")

    # -- accounting ---------------------------------------------------------
    def rates(self) -> tuple[float, float]:
        for key, val in self.pricing.items():
            if key in self.model:
                return val
        return self.default_pricing

    def cost_usd(self, *, batch: bool = False) -> float:
        pin, pout = self.rates()
        cost = self.prompt_tokens / 1e6 * pin + self.completion_tokens / 1e6 * pout
        return cost * (0.5 if batch else 1.0)

    def usage_line(self) -> str:
        return (f"{self.name}/{self.model}: {self.prompt_tokens:,} in + {self.completion_tokens:,} out "
                f"= ${self.cost_usd():.2f} (verify pricing in providers/{self.name}.py)")
