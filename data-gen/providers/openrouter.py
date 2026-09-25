"""OpenRouter. One key, many models, OpenAI-compatible, with an OpenAI-shaped batch queue.

    python -m providers.openrouter --check
    from providers import get_provider; p = get_provider("openrouter", "openai/gpt-oss-120b")

Useful extras over the plain OpenAI shape:
  * `provider_order` / `allow_fallbacks` pin or spread which upstream serves the model, which matters when
    one upstream is slow or rate-limits hard mid-run.
  * HTTP-Referer / X-Title headers are what OpenRouter shows on your dashboard; harmless but they make a
    long generation run identifiable when you are trying to work out what spent the credit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from providers.base import Provider, Request, Response, _parse_batch_line


class OpenRouterProvider(Provider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    env_key = "OPENROUTER_API_KEY"
    # Only the ~72 models carrying an explicit ':batch' suffix have a batch endpoint. Gemma and Llama do NOT
    # ("Model 'google/gemma-4-31b-it' does not have a :batch endpoint"), so for those the only options are
    # synchronous concurrency here or self-hosting with vllm_gen.py.
    supports_batch = True
    batch_file_purpose = "batch"
    # OpenRouter caps a single batch; chunk conservatively and submit several.
    MAX_REQUESTS_PER_BATCH = 10_000
    # USD per 1M (input, output). VERIFY at https://openrouter.ai/models -- OpenRouter quotes per upstream
    # and the cheapest upstream changes week to week.
    # USD per 1M (input, output), read from OpenRouter's /models endpoint 2026-09-24. Re-check with
    # `python -m providers.openrouter --prices` -- the cheapest upstream for a model changes week to week.
    pricing = {
        "gemma-4-31b-it": (0.09, 0.34),
        "gemma-4-26b-a4b-it": (0.09, 0.30),
        "gemma-3-27b-it": (0.08, 0.45),
        "gemma-3-12b-it": (0.05, 0.15),
        "llama-3.3-70b-instruct": (0.10, 0.32),
        "llama-3.1-70b-instruct": (0.40, 0.40),
        # :batch is markedly cheaper than Together's batch tier -- see README.
        "gpt-oss-120b:batch": (0.03, 0.14),
        "gpt-oss-120b": (0.15, 0.60),
        "gpt-oss-20b": (0.02, 0.09),
    }
    # A ':free' suffix means exactly that, but the endpoints are heavily rate-limited upstream and will
    # 429 for long stretches; useful for smoke tests, not for a 225k-request run.
    FREE_SUFFIX = ":free"

    default_pricing = (0.30, 0.80)

    def __init__(self, model: str, *, provider_order: Optional[list[str]] = None,
                 allow_fallbacks: bool = True, **kw):
        super().__init__(model, **kw)
        self.provider_order = provider_order
        self.allow_fallbacks = allow_fallbacks

    def rates(self) -> tuple[float, float]:
        if self.model.endswith(self.FREE_SUFFIX):
            return (0.0, 0.0)
        # longest matching key wins, so "gpt-oss-120b:batch" is not shadowed by "gpt-oss-120b"
        for key in sorted(self.pricing, key=len, reverse=True):
            if key in self.model:
                return self.pricing[key]
        return self.default_pricing

    def extra_headers(self) -> dict[str, str]:
        return {"HTTP-Referer": "https://github.com/pauljeffrey/sabiyarn_MoE",
                "X-Title": "SabiYarn data-gen"}

    def _payload(self, req: Request) -> dict[str, Any]:
        p = super()._payload(req)
        if self.provider_order:
            p["extra_body"] = {"provider": {"order": self.provider_order,
                                            "allow_fallbacks": self.allow_fallbacks}}
        return p


    # -- batch: NOT OpenAI-shaped ------------------------------------------------------------------
    # OpenRouter takes the whole batch inline as JSON on POST /batches -- no file upload -- and its parser is
    # KEY-ORDER SENSITIVE: `endpoint` and `model` must appear before `requests`, or it returns
    # "`endpoint` and `model` are required and must appear before `requests`". Results come back on the batch
    # object itself rather than as a downloadable output file. None of that matches the base implementation,
    # so all three methods are overridden.
    def _http(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        import json as _json
        import urllib.error
        import urllib.request

        data = _json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     **self.extra_headers()})
        try:
            return _json.loads(urllib.request.urlopen(req, timeout=self.timeout).read() or b"{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:300].decode()}") from None

    def submit_batch(self, requests: Iterable[Request], workdir: Path) -> str:
        reqs = list(requests)
        if len(reqs) > self.MAX_REQUESTS_PER_BATCH:
            raise SystemExit(
                f"{len(reqs):,} requests exceeds OpenRouter's per-batch cap of "
                f"{self.MAX_REQUESTS_PER_BATCH:,}. Split the job with --limit, or shard it with "
                f"DATA_GEN_SHARDS and submit one batch per shard.")
        workdir.mkdir(parents=True, exist_ok=True)
        # `model` lives at the batch level, so it must NOT be repeated in each request body.
        lines = []
        for r in reqs:
            body = {k: v for k, v in self._payload(r).items() if k != "model"}
            lines.append({"custom_id": r.custom_id, "body": body})
        payload = {"endpoint": "/v1/chat/completions", "model": self.model, "requests": lines}
        (workdir / "metadata.json").write_text(
            __import__("json").dumps({r.custom_id: r.metadata for r in reqs}, ensure_ascii=False),
            encoding="utf-8")
        batch = self._http("POST", "/batches", payload)
        bid = batch["id"]
        (workdir / "batch.json").write_text(
            __import__("json").dumps({"batch_id": bid, "n": len(reqs), "model": self.model}, indent=2),
            encoding="utf-8")
        print(f"[openrouter] submitted {len(reqs)} requests -> batch {bid} "
              f"(${self.rates()[0]}/${self.rates()[1]} per 1M, 24h window)")
        return bid

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        # A freshly created batch 404s for a few seconds before it is queryable, so a poll immediately after
        # submit_batch would otherwise look like a lost job.
        import time as _time

        for attempt in range(6):
            try:
                b = self._http("GET", f"/batches/{batch_id}")
                break
            except RuntimeError as exc:
                if "404" not in str(exc) or attempt == 5:
                    raise
                _time.sleep(2 * (attempt + 1))
        c = b.get("request_counts") or {}
        return {"id": b.get("id", batch_id), "status": b.get("status", "unknown"),
                "completed": c.get("completed", 0), "failed": c.get("failed", 0),
                "total": c.get("total", 0), "usage": b.get("usage"),
                "output_file_id": None, "error_file_id": None}

    def fetch_batch(self, batch_id: str, out_path: Path) -> list[Response]:
        import json as _json

        b = self._http("GET", f"/batches/{batch_id}")
        if b.get("status") != "completed":
            raise RuntimeError(f"batch {batch_id} is {b.get('status')}, not completed")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(b.get("results") or [], ensure_ascii=False), encoding="utf-8")
        meta_path = out_path.parent / "metadata.json"
        meta = _json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return [_parse_batch_line(row, meta, self.model) for row in (b.get("results") or [])]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = OpenRouterProvider(a.model)
    if a.check:
        r = p.complete(Request("check", [{"role": "user", "content": "Reply with exactly: ready"}], max_tokens=16))
        print("ok" if r.ok else "FAILED", "->", (r.text or r.error)[:200])
        print(p.usage_line())
