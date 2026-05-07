from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from config.openai import load_openai_config, openai_chat_kwargs
from graph.tracing import traceable
from schemas import GraphState

log = logging.getLogger(__name__)

_CLASSIFY_SYSTEM = """
Classify the user message into exactly one mode:
- analyse  : questions about their current routine, ingredient conflicts,
             irritation causes, product compatibility, or reviewing what they use
- build    : wanting to start or build a routine from scratch
- learn    : asking what an ingredient does, how something works, or educational
             questions about skincare science
- recommend: asking for product recommendations or what to buy next

Reply with ONE word only: analyse, build, learn, or recommend."""

_VALID_MODES = {"analyse", "build", "learn", "recommend"}

_PRODUCT_CATEGORY_RE = re.compile(
    r"\b(?:cleanser|face wash|moisturi[sz]er|serum|toner|essence|sunscreen|spf|"
    r"exfoliant|bha|aha|retinol|retinal|eye cream|spot treatment|mask|face oil)\b",
    re.IGNORECASE,
)


def _asks_for_products_with_ingredient(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(?:products?|serums?|creams?|moisturi[sz]ers?|toners?|cleansers?|eye creams?)\s+"
            r"(?:with|containing|that contain|featuring|for)\b",
            lower,
        )
    )


def _asks_for_product_category(text: str) -> bool:
    lower = text.lower()
    if not _PRODUCT_CATEGORY_RE.search(lower):
        return False
    return bool(
        re.search(
            r"\b(?:how about|what about|any|a|an|some|gentle|good|best|recommend|"
            r"suggest|need|want|looking for|add)\b",
            lower,
        )
    )


@lru_cache(maxsize=128)
def _keyword_fallback(text: str) -> str:
    lower = text.lower()

    recommend_signals = (
        "recommend", "suggest", "best", "which product", "what to buy",
        "should i get", "looking for a", "find me", "good product for",
        "what should i add", "should i add", "add to my routine",
        "add to routine", "addition to my routine", "add next",
        "product options", "products with", "products containing",
    )
    learn_signals = (
        "what is", "what does", "explain", "how does", "tell me about",
        "what are", "how do", "does it work", "benefits of", "effects of",
        "literature", "evidence", "research", "studies", "study", "papers",
        "what do people say", "what are people saying", "what does reddit say",
        "community thoughts", "reviews of", "review of",
        "topical ingredient", "skincare ingredient",
    )
    build_signals = (
        "build", "start", "create", "new routine", "begin", "from scratch",
        "help me make", "put together",
    )

    if (
        _asks_for_products_with_ingredient(text)
        or _asks_for_product_category(text)
        or any(s in lower for s in recommend_signals)
    ):
        return "recommend"
    if any(s in lower for s in learn_signals):
        return "learn"
    if any(s in lower for s in build_signals):
        return "build"
    return "analyse"


@traceable(name="classify_intent")
def classify_intent(state: GraphState) -> GraphState:
    """Classify the user's message into a conversation mode."""
    if state.get("mode_locked") and state.get("mode") in _VALID_MODES:
        return state

    message = state.get("message", "")

    if _asks_for_products_with_ingredient(message) or _asks_for_product_category(message):
        return {**state, "mode": "recommend"}

    if os.getenv("SKINIQ_LLM_INTENT_CLASSIFIER", "").lower() not in {"1", "true", "yes"}:
        return {**state, "mode": _keyword_fallback(message)}

    api_key, model, base_url = load_openai_config("SKINIQ_INTENT_CLASSIFIER_MODEL")

    if api_key:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                **openai_chat_kwargs(
                    model,
                    api_key,
                    base_url,
                    temperature=0,
                    timeout=6,
                    max_completion_tokens=8,
                    reasoning_effort="minimal",
                    verbosity="low",
                )
            )
            response = llm.invoke(
                [SystemMessage(content=_CLASSIFY_SYSTEM), HumanMessage(content=message)]
            )
            candidate = response.content.strip().lower()
            if candidate in _VALID_MODES:
                return {**state, "mode": candidate}
            log.warning("LLM returned unexpected mode %r — falling back", candidate)
        except Exception as exc:
            log.warning("Intent classification LLM call failed: %s", exc)

    return {**state, "mode": _keyword_fallback(message)}
