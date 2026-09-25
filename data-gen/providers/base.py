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


# Models whose chain of thought goes to a separate `reasoning` field and which STRIP <think> from the message
# content. Measured: asked to emit "<think>I am thinking</think>DONE" literally, gpt-oss-120b returns "DONE".
# They cannot produce this corpus's assistant format, whatever the prompt says, so warn rather than let a
# 200k-request run come back with zero think blocks.
REASONING_MODELS_THAT_STRIP_THINK = ("gpt-oss", "o1-", "o3-", "o4-", "deepseek-r1", "qwq")


def warn_if_reasoning_model(model: str) -> None:
    if any(k in model.lower() for k in REASONING_MODELS_THAT_STRIP_THINK):
        print(f"  WARNING: {model} is a reasoning model. It routes its chain of thought to a separate "
              f"`reasoning` field and strips <think> from the content, so samples will have NO <think> "
              f"blocks -- measured at 0 of 14 on a pilot. Use a non-reasoning model "
              f"(e.g. google/gemma-4-31b-it) for sft/rl.", flush=True)


def _parse_batch_line(row: dict, meta: dict, model: str) -> Response:
    cid = row.get("custom_id", "")
    md = meta.get(cid, {})
    body = (row.get("response") or {}).get("body") or {}
    if row.get("error") or not body.get("choices"):
        return Response(cid, "", False, json.dumps(row.get("error") or "no choices")[:400], metadata=md)
    msg = body["choices"][0].get("message", {})
    usage = body.get("usage", {})
    return Response(cid, msg.get("content") or "", True, None, usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0), body.get("model", model),
                    msg.get("tool_calls"), md)


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
                 max_retries: int = 5, timeout: float = 180.0,
                 retry_base_s: Optional[float] = None, retry_max_s: Optional[float] = None):
        load_env()
        self.model = model
        self.api_key = api_key or os.environ.get(self.env_key, "")
        if not self.api_key:
            raise SystemExit(
                f"{self.name}: no API key. Put {self.env_key}=... in {ROOT/'.env'} "
                f"(one .env for the whole repo).")
        warn_if_reasoning_model(model)
        # A ':free' endpoint rate-limits on a scale of minutes, not seconds: a 1.5s-doubling-to-60s backoff
        # just burns retries against a wall. Free models therefore wait 2-5 minutes between attempts and get
        # more of them, which is slow but is the difference between eventually succeeding and never.
        free = model.endswith(":free")
        self.is_free = free
        self.retry_base_s = retry_base_s if retry_base_s is not None else (120.0 if free else 1.5)
        self.retry_max_s = retry_max_s if retry_max_s is not None else (300.0 if free else 60.0)
        self.max_concurrency = 2 if (free and max_concurrency > 2) else max_concurrency
        self.max_retries = max_retries if not free else max(max_retries, 8)
        self.timeout = timeout
        if free:
            print(f"  [{self.name}] {model} is a free endpoint: concurrency capped at "
                  f"{self.max_concurrency}, backoff {self.retry_base_s:.0f}-{self.retry_max_s:.0f}s, "
                  f"{self.max_retries} attempts. Expect this to be slow.", flush=True)
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
        delay = self.retry_base_s
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
                # jitter so many workers do not retry in lockstep against the same upstream
                wait = delay + random.uniform(0, min(delay, 30.0))
                if wait > 30:
                    print(f"  [{self.name}] rate-limited, cooling down {wait/60:.1f}m "
                          f"(attempt {attempt + 1}/{self.max_retries})", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, self.retry_max_s)
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

    # -- batch --------------------------------------------------------------
    # Together and OpenRouter both expose OpenAI-shaped /files and /batches, at roughly half price with a
    # 24h window. On OpenRouter the cheapest models (e.g. openai/gpt-oss-120b:batch at $0.03/$0.14) are
    # reachable ONLY this way -- chat/completions returns 404 for them.
    batch_file_purpose = "batch"

    def _batch_line(self, req: Request) -> dict[str, Any]:
        return {"custom_id": req.custom_id, "method": "POST", "url": "/v1/chat/completions",
                "body": self._payload(req)}

    def submit_batch(self, requests: Iterable[Request], workdir: Path) -> str:
        workdir.mkdir(parents=True, exist_ok=True)
        reqs = list(requests)
        path = workdir / "batch_input.jsonl"
        meta = {r.custom_id: r.metadata for r in reqs}
        path.write_text("".join(json.dumps(self._batch_line(r), ensure_ascii=False) + "\n" for r in reqs),
                        encoding="utf-8")
        (workdir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        client = self._client()
        with path.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose=self.batch_file_purpose)
        batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                      completion_window="24h")
        (workdir / "batch.json").write_text(
            json.dumps({"batch_id": batch.id, "input_file_id": uploaded.id, "n": len(reqs),
                        "model": self.model}, indent=2), encoding="utf-8")
        print(f"[{self.name}] submitted {len(reqs)} requests -> batch {batch.id} (~50% price, 24h window)")
        return batch.id

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        b = self._client().batches.retrieve(batch_id)
        counts = getattr(b, "request_counts", None)
        return {"id": b.id, "status": getattr(b, "status", "unknown"),
                "completed": getattr(counts, "completed", 0) if counts else 0,
                "failed": getattr(counts, "failed", 0) if counts else 0,
                "total": getattr(counts, "total", 0) if counts else 0,
                "output_file_id": getattr(b, "output_file_id", None),
                "error_file_id": getattr(b, "error_file_id", None)}

    def fetch_batch(self, batch_id: str, out_path: Path) -> list[Response]:
        st = self.poll_batch(batch_id)
        if st["status"] != "completed":
            raise RuntimeError(f"batch {batch_id} is {st['status']}, not completed")
        client = self._client()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = client.files.content(st["output_file_id"])
        data = content.read() if hasattr(content, "read") else content.content
        out_path.write_bytes(data)
        meta_path = out_path.parent / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return [_parse_batch_line(json.loads(l), meta, self.model)
                for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]

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
