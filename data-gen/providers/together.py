"""Together AI. OpenAI-compatible chat completions, plus a batch queue at ~50% price.

    python -m providers.together --check                      # 1 live request, prints the reply

The batch flow lives in providers/base.py -- Together and OpenRouter share it. Together's file upload wants
purpose="batch-api" where OpenAI and OpenRouter want "batch", which is the only difference.
"""

from __future__ import annotations

from providers.base import Provider, Request


class TogetherProvider(Provider):
    name = "together"
    base_url = "https://api.together.xyz/v1"
    env_key = "TOGETHER_API_KEY"
    supports_batch = True
    batch_file_purpose = "batch-api"
    # USD per 1M (input, output). VERIFY at https://www.together.ai/pricing -- these move.
    pricing = {
        "gpt-oss-120b": (0.15, 0.60),
        "gpt-oss-20b": (0.05, 0.20),
        "Qwen3-235B": (0.20, 0.60),
        "Llama-3.3-70B": (0.88, 0.88),
        "gemma-3-27b": (0.20, 0.30),
        "gemma-2-27b": (0.80, 0.80),
    }
    default_pricing = (0.20, 0.60)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = TogetherProvider(a.model)
    if a.check:
        r = p.complete(Request("check", [{"role": "user", "content": "Reply with exactly: ready"}], max_tokens=16))
        print("ok" if r.ok else "FAILED", "->", (r.text or r.error)[:200])
        print(p.usage_line())
