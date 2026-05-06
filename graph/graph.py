from __future__ import annotations

from collections.abc import Callable
import logging
import os
import re

from langgraph.graph import END, StateGraph

from graph.nodes.analyse_routine import analyse_routine
from graph.nodes.classify_intent import _keyword_fallback, classify_intent
from graph.nodes.generate_response import generate_response
from graph.nodes.retrieve_context import retrieve_context
from graph.nodes.skincare_agent import skincare_agent
from graph.nodes.validate_response import validate_response
from graph.tracing import traceable
from graph.recommendation_cache import get_prepared_answer
from schemas import GraphState
from tools.product_tools import select_recommended_product_cards
from utils.product_match import product_family_name, product_identity_key

log = logging.getLogger(__name__)

RECOMMENDATION_LIMIT = 2


def _build_pg_connstr() -> str:
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "skincare_db")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def _make_checkpointer():
    """Use PostgresSaver for persistent, multi-user-safe checkpointing.
    Falls back to MemorySaver if the Postgres dependency is missing or the DB is unreachable.
    """
    try:
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        pool = ConnectionPool(conninfo=_build_pg_connstr(), max_size=5, open=True)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # creates langgraph checkpoint tables if they don't exist
        log.info("Using PostgresSaver for graph checkpointing")
        return checkpointer
    except Exception as exc:
        log.warning("PostgresSaver unavailable (%s), falling back to MemorySaver", exc)
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


_checkpointer = _make_checkpointer()


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("analyse_routine", analyse_routine)
    workflow.add_node("skincare_agent", skincare_agent)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("validate_response", validate_response)

    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "retrieve_context")
    workflow.add_edge("retrieve_context", "analyse_routine")
    workflow.add_edge("analyse_routine", "skincare_agent")
    workflow.add_edge("skincare_agent", "generate_response")
    workflow.add_edge("generate_response", "validate_response")
    workflow.add_edge("validate_response", END)

    return workflow.compile(checkpointer=_checkpointer)


advisor_graph = build_graph()


def _product_key(product: dict) -> tuple[str, str]:
    return product_identity_key(product)


def _product_family_key(product: dict) -> tuple[str, str]:
    return product_identity_key(product, family=True)


def _dedupe_products(products: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for product in products:
        key = _product_family_key(product)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(product)
    return deduped


def _routine_product_keys(user_routine: dict | None) -> set[tuple[str, str]]:
    if not isinstance(user_routine, dict):
        return set()

    keys = set()
    for item in user_routine.get("items") or user_routine.get("routine") or []:
        if isinstance(item, dict) and item.get("product_name"):
            keys.add(_product_key(item))
    return keys


def _exclude_routine_products(products: list[dict], user_routine: dict | None) -> list[dict]:
    routine_keys = _routine_product_keys(user_routine)
    routine_names = {name for name, _brand in routine_keys if name}
    return [
        product
        for product in products
        if _product_key(product) not in routine_keys
        and _product_key(product)[0] not in routine_names
    ]


def _cards_for_selected_product_names(
    selected_names: list[str],
    candidate_products: list[dict],
    routine_items: list[dict],
    limit: int = RECOMMENDATION_LIMIT,
) -> list[dict]:
    """Resolve agent-selected names into UI card payloads.

    Prefer the catalog objects already in hand. The selector returns names as
    text, so we map those names back onto candidate objects by product family
    before falling back to catalog lookup for genuinely missing products.
    """
    names = [
        re.split(r"\s+[·|]\s+", str(name).strip(), maxsplit=1)[0].strip()
        for name in selected_names
        if str(name).strip()
    ]
    if not names:
        return []

    limit = max(1, min(limit, RECOMMENDATION_LIMIT, len(names)))
    candidates = _dedupe_products(candidate_products)
    family_index = {
        product_family_name(str(product.get("product_name") or "")): product
        for product in reversed(candidates)
        if isinstance(product, dict) and product.get("product_name")
    }

    resolved = []
    seen_keys: set[tuple[str, str]] = set()
    for name in names:
        product = family_index.get(product_family_name(name))
        if not product:
            continue
        key = _product_family_key(product)
        if key in seen_keys:
            continue
        resolved.append(product)
        seen_keys.add(key)
        if len(resolved) >= limit:
            return resolved

    synthetic_response = "\n".join(f"#### {name}" for name in names)
    fallback = select_recommended_product_cards.invoke(
        {
            "response": synthetic_response,
            "candidate_products": candidates,
            "routine_items": routine_items,
            "limit": limit,
        }
    )
    for product in fallback:
        key = _product_family_key(product)
        if key not in seen_keys:
            resolved.append(product)
            seen_keys.add(key)
        if len(resolved) >= limit:
            break
    for name in names:
        if len(resolved) >= limit:
            break
        minimal = {
            "product_name": name,
            "brand": "",
            "quantity": "",
            "image_url": "",
            "key_ingredients": [],
            "active_ingredients": [],
            "ingredients": [],
            "categories": [],
            "source": "assistant_recommendation",
        }
        key = _product_family_key(minimal)
        if key not in seen_keys:
            resolved.append(minimal)
            seen_keys.add(key)
    return resolved


def _community_search_query(message: str) -> str | None:
    lower = message.lower()
    if not any(
        signal in lower
        for signal in (
            "what do people say",
            "what are people saying",
            "community",
            "reddit",
            "reviews",
            "review",
            "real-world",
            "real world",
        )
    ):
        return None

    cleaned = re.sub(
        r"(?i)\b(?:what\s+do\s+people\s+say\s+about|what\s+are\s+people\s+saying\s+about|"
        r"what\s+does\s+reddit\s+say\s+about|reviews?\s+(?:for|of)|community\s+thoughts\s+(?:on|about))\b",
        "",
        message,
    )
    cleaned = cleaned.strip(" ?.!:")
    return cleaned or message.strip()


def _extract_recommended_ingredients(response: str) -> list[str]:
    """Extract ingredient names from a learn-mode response.

    Handles H4 headings (`#### Name`), numbered list format (`1. **Name** — ...`),
    and falls back to any short bolded term.
    """
    names: list[str] = []
    seen: set[str] = set()
    generic_headings = {
        "what it is",
        "how it works",
        "how it's thought to work",
        "how it is thought to work",
        "mechanism",
        "evidence",
        "evidence and safety",
        "safety",
        "practical guidance",
        "how to use it",
        "where it fits",
    }

    def add_name(raw_name: str) -> None:
        name = raw_name.strip(" -:;.")
        key = name.lower()
        if not name or key in generic_headings or key in seen:
            return
        names.append(name)
        seen.add(key)

    # Primary: H4 headings (#### Name) — format used by generate_response SYSTEM_PROMPT
    h4_re = re.compile(r"^#{4}\s+(.{3,60})$")
    for line in response.splitlines():
        m = h4_re.match(line.strip())
        if m:
            add_name(m.group(1))

    if names:
        return names[:RECOMMENDATION_LIMIT]

    # Secondary: numbered list lines with a leading bold term
    numbered_re = re.compile(r"^\d+\.\s+\*\*([^*]{3,50})\*\*")
    for line in response.splitlines():
        m = numbered_re.match(line.strip())
        if m:
            add_name(m.group(1))

    if names:
        return names[:RECOMMENDATION_LIMIT]

    # Fallback: any short bolded term (≤4 words) that looks like an ingredient
    bold_re = re.compile(r"\*\*([A-Z][A-Za-z0-9 '\-]{2,40})\*\*")
    for m in bold_re.finditer(response):
        name = m.group(1).strip()
        if len(name.split()) <= 4:
            add_name(name)
    return names[:RECOMMENDATION_LIMIT]


def _ingredient_evidence_query(message: str, result: GraphState | None = None) -> str | None:
    result = result or {}
    if result.get("mode") != "learn":
        return None

    # Prefer ingredients parsed from the actual response — the message rarely
    # names them (e.g. "what ingredients should I try?").
    response_text = result.get("final_response") or result.get("draft_response") or ""
    if response_text:
        ingredients = _extract_recommended_ingredients(response_text)
        if ingredients:
            return ", ".join(ingredients)

    message_text = message.lower()
    for ing in result.get("retrieved_ingredients") or []:
        name = str(
            ing.get("display_name")
            or ing.get("inci_name")
            or ""
        ).strip()
        if name and name.lower() in message_text:
            return name

    for output in result.get("tool_outputs") or []:
        if output.get("tool") != "extract_ingredient_terms":
            continue
        terms = output.get("output") or []
        if terms:
            return str(terms[0]).strip() or None
    named_term = _named_learn_term(message)
    if named_term:
        return named_term
    return None


def _named_learn_term(message: str) -> str | None:
    text = " ".join(str(message or "").strip().split())
    if not text:
        return None
    patterns = (
        r"(?i)\bwhat\s+is\s+([a-z0-9][a-z0-9 +/'-]{1,60}?)(?:\s+and\b|\?|$)",
        r"(?i)\btell\s+me\s+about\s+([a-z0-9][a-z0-9 +/'-]{1,60}?)(?:\?|$)",
        r"(?i)\bwhat\s+does\s+([a-z0-9][a-z0-9 +/'-]{1,60}?)\s+do\b",
        r"(?i)\bis\s+([a-z0-9][a-z0-9 +/'-]{1,60}?)\s+(?:good|safe|worth|effective)\b",
    )
    blocked = {
        "a",
        "an",
        "it",
        "this",
        "that",
        "ingredient",
        "topical ingredient",
        "skincare ingredient",
    }
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        term = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;!?")
        if term.lower() not in blocked:
            return term
    return None


def _community_query_for_evidence(
    community_request: dict | None,
    ingredient_query: str | None,
) -> str | None:
    if community_request:
        return community_request.get("query") or None
    return None


def _explicit_literature_followup(message: str) -> bool:
    lower = message.lower()
    return any(term in lower for term in ("literature", "evidence", "research", "studies", "study", "papers"))


def _community_placeholder_response(message: str, response: str) -> str:
    """Avoid follow-up prompts when the user already requested community signals."""
    lower = (response or "").lower()
    blocked = (
        "yes/no",
        "confirm you want",
        "do you want",
        "i can share community",
        "i can summarize community",
        "reddit feedback only",
        "general summary",
    )
    if any(phrase in lower for phrase in blocked):
        query = _community_search_query(message) or "that product"
        return f"Here is the plain-language community signal I found for **{query}**."
    return response


def _report_progress(on_progress: Callable[[str], None] | None, message: str) -> None:
    if not on_progress:
        return
    try:
        on_progress(message)
    except BaseException as exc:
        # Streamlit may raise StopException while handling an enqueue request
        # during reruns/stops. Progress text is cosmetic, so do not let that
        # abort a valid advisor response.
        if exc.__class__.__name__ == "StopException":
            log.debug("Ignoring Streamlit StopException from progress callback")
            return
        raise


def _initial_state(
    message: str,
    user_profile: dict | None = None,
    user_routine: dict | None = None,
    explicit_mode: str | None = None,
) -> GraphState:
    from langchain_core.messages import HumanMessage

    mode = explicit_mode if explicit_mode in {"build", "analyse", "learn", "recommend"} else "build"

    return {
        "user_id": (user_profile or {}).get("user_id", "session"),
        "mode": mode,
        "mode_locked": explicit_mode in {"build", "analyse", "learn", "recommend"},
        "message": message,
        "messages": [HumanMessage(content=message)],
        "user_profile": user_profile or {},
        "user_routine": user_routine or {},
        "retrieved_products": [],
        "retrieved_ingredients": [],
        "tool_outputs": [],
        "community_search_request": None,
        "literature_search_request": None,
    }


@traceable(name="run_advisor")
def run_advisor(
    message: str,
    user_profile: dict | None = None,
    user_routine: dict | None = None,
    thread_id: str = "default",
    explicit_mode: str | None = None,
) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = _initial_state(message, user_profile, user_routine, explicit_mode)
    result = advisor_graph.invoke(initial_state, config=config)
    return result.get("final_response") or result.get("draft_response") or ""


@traceable(name="run_advisor_with_progress")
def run_advisor_with_progress(
    message: str,
    user_profile: dict | None = None,
    user_routine: dict | None = None,
    thread_id: str = "default",
    on_progress: Callable[[str], None] | None = None,
    explicit_mode: str | None = None,
    allow_prepared_answer: bool = True,
) -> dict:
    """Returns {"response": str, "matched_products": list[dict]}."""
    initial_mode = explicit_mode if explicit_mode in {"build", "analyse", "learn", "recommend"} else _keyword_fallback(message)
    if allow_prepared_answer and initial_mode in {"recommend", "learn"}:
        _report_progress(on_progress, "Checking recommendations")
        prepared = get_prepared_answer(
            message,
            initial_mode,
            user_profile,
            user_routine,
            wait_timeout=45.0,
        )
        if prepared is not None:
            _report_progress(on_progress, "Getting recommendation")
            return prepared

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = _initial_state(message, user_profile, user_routine, explicit_mode)
    result: GraphState = {}

    _report_progress(on_progress, "Routing your question")

    next_status = {
        "classify_intent": "Gathering relevant information",
        "retrieve_context": "Analysing your routine",
        "analyse_routine": "Analysing",
        "skincare_agent": "Choosing supporting tools",
        "generate_response": "Drafting the response",
        "validate_response": "Ready",
    }

    for update in advisor_graph.stream(initial_state, config=config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, node_state in update.items():
            if isinstance(node_state, dict):
                # Only merge non-None values so a later node can't silently
                # overwrite real data from an earlier node with None.
                result = {**result, **{k: v for k, v in node_state.items() if v is not None}}
            if node_name in next_status:
                _report_progress(on_progress, next_status[node_name])

    response = result.get("final_response") or result.get("draft_response") or ""
    routine_items = []
    if isinstance(user_routine, dict):
        routine_items = user_routine.get("items") or user_routine.get("routine") or []
    matched_products = []
    if result.get("mode") in {"recommend", "build"}:
        product_candidates = _dedupe_products(
            (result.get("matched_products") or [])
            + (result.get("agent_searched_products") or [])
        )
        selected_names = result.get("agent_selected_product_names") or []
        if selected_names:
            matched_products = _cards_for_selected_product_names(
                selected_names,
                product_candidates,
                routine_items,
                limit=RECOMMENDATION_LIMIT,
            )
        else:
            matched_products = select_recommended_product_cards.invoke(
                {
                    "response": response,
                    "candidate_products": product_candidates,
                    "routine_items": routine_items,
                    "limit": RECOMMENDATION_LIMIT,
                }
            )
        matched_products = _exclude_routine_products(matched_products, user_routine)

    community_request = None
    community_query = _community_search_query(message)
    if community_query:
        community_request = {
            "query": community_query,
            "reason": "User asked for community opinions or reviews.",
        }
        response = _community_placeholder_response(message, response)

    ingredient_query = None if community_request else _ingredient_evidence_query(message, result)
    literature_request = None
    if ingredient_query:
        literature_request = {
            "query": ingredient_query,
            "reason": "Ingredient question; surface relevant scientific literature.",
        }
    elif _explicit_literature_followup(message) and result.get("literature_search_request"):
        literature_request = result.get("literature_search_request")

    evidence_summary = None
    if literature_request or community_request:
        _report_progress(on_progress, "Summarising evidence sources")
        from services.evidence_summary_service import summarize_evidence

        evidence_summary = summarize_evidence(
            literature_query=(literature_request or {}).get("query"),
            community_query=_community_query_for_evidence(community_request, ingredient_query),
        )

    return {
        "response": response,
        "matched_products": matched_products,
        "updated_user_profile": result.get("updated_user_profile"),
        "community_search_request": community_request,
        "literature_search_request": literature_request,
        "evidence_summary": evidence_summary,
    }
