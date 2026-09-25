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

from typing import Any, Optional

from providers.base import Provider, Request


class OpenRouterProvider(Provider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    env_key = "OPENROUTER_API_KEY"
    # Its ':batch' models (the cheapest tier) are reachable ONLY through /files + /batches; chat/completions
    # 404s for them with "cannot be used with the chat/completions endpoint".
    supports_batch = True
    batch_file_purpose = "batch"
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
