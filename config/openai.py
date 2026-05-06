from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5-mini"


def _clean(value: object | None) -> str:
    return str(value or "").strip()


def load_openai_config(
    model_env: str | None = None,
    default_model: str = DEFAULT_OPENAI_CHAT_MODEL,
) -> tuple[str, str, str | None]:
    """Return (api_key, model, base_url) from .env / environment.

    model_env lets each agent/tool choose its own model variable while still
    falling back to OPENAI_MODEL and then the app default.
    """
    env = dotenv_values(_ENV_PATH)
    load_dotenv(_ENV_PATH, override=True)

    api_key = _clean(env.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))

    model = ""
    if model_env:
        model = _clean(env.get(model_env) or os.getenv(model_env))
    model = model or _clean(env.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or default_model)

    base_url = _clean(env.get("OPENAI_BASE_URL"))
    if not base_url and _ENV_PATH.exists():
        # Prevent stale shell/data env values from making LangChain call OpenRouter.
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_API_BASE", None)
    else:
        base_url = _clean(
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )

    base_url = base_url or None
    return api_key, model, base_url


def openai_chat_kwargs(
    model: str,
    api_key: str,
    base_url: str | None,
    *,
    temperature: float | None = None,
    timeout: int = 30,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
) -> dict:
    kwargs: dict = {
        "model": model,
        "api_key": api_key,
        "timeout": timeout,
    }
    if temperature is not None and not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if model.startswith("gpt-5"):
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if verbosity:
            kwargs["verbosity"] = verbosity
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs
