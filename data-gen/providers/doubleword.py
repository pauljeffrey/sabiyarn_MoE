"""Doubleword. OpenAI-compatible, with a real OpenAI-shaped batch queue at ~80% off realtime.

    python -m providers.doubleword --check
    python generate.py --kind rl --provider doubleword --batch

VERIFIED 2026-10-01 against the live API with a working key:
  * base url https://api.doubleword.ai/v1, Bearer auth, /v1/models returns an OpenAI-shaped list;
  * GET /v1/batches and GET /v1/files both answer, so the base class's file-upload-then-create-batch flow
    works unchanged -- unlike OpenRouter, whose batch API takes the whole job inline and is key-order
    sensitive, and so needed all three methods overridden;
  * 36 models, including google/gemma-4-31B-it (note the capital B) and google/gemma-4-26B-A4B-it.

google/gemma-3-27b-it IS NOT OFFERED HERE, and that decides what this provider is for. gemma-3-27b measured
89% clean on low-resource pretraining prose against gemma-4-31b's 6%, so pretraining for those languages
cannot be bought here. SFT and RL are a different task -- conversation structure and tool use -- where
gemma-4-31b was chosen on a much larger sample, and this is the cheapest place to buy them.
"""

from __future__ import annotations

from typing import Any, Optional

from providers.base import Provider, Request


class DoublewordProvider(Provider):
    name = "doubleword"
    base_url = "https://api.doubleword.ai/v1"
    env_key = "DOUBLEWORD_API_KEY"
    supports_batch = True
    batch_file_purpose = "batch"

    # USD per 1M (input, output) on the BATCH tier, from doubleword.ai/pricing (checked 2026-10-01). The site
    # lists three SLAs per model -- realtime, async (~50% off), batch (~80% off) -- and these are the batch
    # column. Re-check before trusting a total: provider pricing moves.
    pricing = {
        "gemma-4-31B-it": (0.06, 0.18),
        "gemma-4-26B-A4B-it": (0.05, 0.17),
        "gpt-oss-120b": (0.08, 0.30),
        "gpt-oss-20b": (0.02, 0.09),
        "Qwen3.8-27B-FP8": (0.25, 1.50),
        "Qwen3.6-35B-A3B-FP8": (0.07, 0.50),
        "DeepSeek-V4-Flash": (0.05, 0.09),
        "DeepSeek-V4.1-Flash": (0.08, 0.30),
        "DeepSeek-V4-Pro": (0.66, 1.98),
    }
    # Anything unlisted: assume the gemma-4-31b batch rate rather than something optimistic.
    default_pricing = (0.06, 0.18)

    # The model ids here are case-sensitive and differ from OpenRouter's. Accept the OpenRouter spelling and
    # translate, so --model can be copied between providers without a silent 404.
    ALIASES = {
        "google/gemma-4-31b-it": "google/gemma-4-31B-it",
        "google/gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B-it",
        "gemma-4-31b-it": "google/gemma-4-31B-it",
    }

    def __init__(self, model: str, **kw):
        canonical = self.ALIASES.get(model.lower(), model)
        if canonical != model:
            print(f"[doubleword] {model} -> {canonical} (this provider's ids are case-sensitive)")
        super().__init__(canonical, **kw)
        if "gemma-3" in canonical.lower():
            print(f"[doubleword] WARNING: {canonical} is not offered here -- only gemma-4. For low-resource "
                  f"PRETRAINING use gemma-3-27b elsewhere: it measured 89% clean against gemma-4-31b's 6%.")

    def extra_headers(self) -> dict[str, str]:
        return {"HTTP-Referer": "https://github.com/pauljeffrey/sabiyarn_MoE",
                "X-Title": "SabiYarn data-gen"}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = DoublewordProvider(a.model)
    if a.check:
        r = p.complete(Request("check", [{"role": "user", "content": "Reply with exactly: ready"}],
                               max_tokens=16))
        print("ok" if r.ok else "FAILED", "->", (r.text or r.error)[:200])
        print(p.usage_line())
