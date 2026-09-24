"""OpenRouter. One key, many models, OpenAI-compatible. No batch queue -- throughput comes from concurrency.

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
    supports_batch = False
    # USD per 1M (input, output). VERIFY at https://openrouter.ai/models -- OpenRouter quotes per upstream
    # and the cheapest upstream changes week to week.
    pricing = {
        "gpt-oss-120b": (0.15, 0.60),
        "gpt-oss-20b": (0.05, 0.20),
        "gemma-3-27b": (0.20, 0.30),
        "qwen-3-235b": (0.20, 0.60),
        "llama-3.3-70b": (0.60, 0.60),
    }
    default_pricing = (0.30, 0.80)

    def __init__(self, model: str, *, provider_order: Optional[list[str]] = None,
                 allow_fallbacks: bool = True, **kw):
        super().__init__(model, **kw)
        self.provider_order = provider_order
        self.allow_fallbacks = allow_fallbacks

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
