"""Provider registry.  get_provider("together", "openai/gpt-oss-120b")"""

from __future__ import annotations

from providers.base import Provider, Request, Response, load_env  # noqa: F401
from providers.doubleword import DoublewordProvider
from providers.openrouter import OpenRouterProvider
from providers.together import TogetherProvider

PROVIDERS = {"together": TogetherProvider, "doubleword": DoublewordProvider,
             "openrouter": OpenRouterProvider}

# Sensible model per provider. These are MEASURED choices, not guesses -- see providers.base.RECOMMENDED_MODEL
# for the per-phase figures. gemma-3-27b is the pretraining model (89% clean on low-resource prose against
# gemma-4-31b's 6%) and gemma-4-31b is the sft/rl model; Doubleword carries only gemma-4, which is why it is
# the place to buy sft and rl rather than pretraining.
DEFAULT_MODELS = {"together": "openai/gpt-oss-120b",
                  "doubleword": "google/gemma-4-31B-it",
                  "openrouter": "google/gemma-4-31b-it"}


def get_provider(name: str, model: str | None = None, **kw) -> Provider:
    if name not in PROVIDERS:
        raise SystemExit(f"unknown provider {name!r}; available: {sorted(PROVIDERS)}")
    return PROVIDERS[name](model or DEFAULT_MODELS[name], **kw)
