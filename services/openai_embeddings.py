from __future__ import annotations

import logging
from functools import lru_cache

from openai import OpenAI

from config.openai import load_openai_config

log = logging.getLogger(__name__)

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


@lru_cache(maxsize=1)
def _client() -> OpenAI | None:
    api_key, _, base_url = load_openai_config()
    if not api_key:
        return None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned = [text.strip() for text in texts if text and text.strip()]
    if not cleaned:
        return []

    client = _client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings")

    _, model, _ = load_openai_config("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
    response = client.embeddings.create(model=model, input=cleaned)
    return [list(item.embedding) for item in response.data]


def embed_query(text: str) -> list[float] | None:
    try:
        embeddings = embed_texts([text])
        return embeddings[0] if embeddings else None
    except Exception as exc:
        log.warning("OpenAI query embedding failed: %s", exc)
        return None


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embed_texts(texts)
