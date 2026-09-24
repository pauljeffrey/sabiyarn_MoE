"""Provider registry.  get_provider("together", "openai/gpt-oss-120b")"""

from __future__ import annotations

from providers.base import Provider, Request, Response, load_env  # noqa: F401
from providers.openrouter import OpenRouterProvider
from providers.together import TogetherProvider

PROVIDERS = {"together": TogetherProvider, "openrouter": OpenRouterProvider}

# Sensible model per provider for this project. gpt-oss-120b is strong enough for the reasoning/tool-call
# structure; for the low-resource languages a gemma-3-27b pass is worth A/B-ing (see README).
DEFAULT_MODELS = {"together": "openai/gpt-oss-120b", "openrouter": "openai/gpt-oss-120b"}


def get_provider(name: str, model: str | None = None, **kw) -> Provider:
    if name not in PROVIDERS:
        raise SystemExit(f"unknown provider {name!r}; available: {sorted(PROVIDERS)}")
    return PROVIDERS[name](model or DEFAULT_MODELS[name], **kw)
