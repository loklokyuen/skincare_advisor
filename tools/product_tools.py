from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import requests
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from config.openai import load_openai_config, openai_chat_kwargs
from services.product_service import identify_key_ingredients
from services.product_service import save_external_product
from services.product_service import search_products
from utils.product_match import normalise_match_text, product_family_name, product_identity_key


def _split_ingredients(text: str) -> list[str]:
    cleaned = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return [item.strip() for item in re.split(r",\s*(?!\d)", cleaned) if item.strip()]


def _identify_key_ingredients_safe(ingredients: list[str]) -> list[str]:
    try:
        return identify_key_ingredients(ingredients)
    except Exception:
        return []


def _normalise_match_text(value: str) -> str:
    return normalise_match_text(value)


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


def _routine_product_keys(routine_items: list[dict] | None) -> set[tuple[str, str]]:
    return {
        _product_key(item)
        for item in routine_items or []
        if isinstance(item, dict) and item.get("product_name")
    }


def _exclude_routine_products(products: list[dict], routine_items: list[dict] | None) -> list[dict]:
    routine_keys = _routine_product_keys(routine_items)
    routine_names = {name for name, _brand in routine_keys if name}
    return [
        product
        for product in products
        if _product_key(product) not in routine_keys
        and _product_key(product)[0] not in routine_names
    ]


def _normalise_obf_product(raw: dict[str, Any]) -> dict:
    ingredients = _split_ingredients(raw.get("ingredients_text") or "")
    categories = raw.get("categories_tags") or []
    brand = raw.get("brands") or ""

    return {
        "product_name": raw.get("product_name") or raw.get("product_name_en") or "Unknown product",
        "brand": brand.split(",")[0].strip() if isinstance(brand, str) else "",
        "quantity": raw.get("quantity") or "",
        "image_url": raw.get("image_url") or "",
        "key_ingredients": _identify_key_ingredients_safe(ingredients),
        "active_ingredients": _identify_key_ingredients_safe(ingredients),
        "categories": categories if isinstance(categories, list) else [],
        "ingredients": ingredients,
        "ingredients_raw": raw.get("ingredients_text") or "",
        "source": "open_beauty_facts",
        "barcode": raw.get("code") or "",
    }


def _search_open_beauty_facts(query: str, limit: int = 5) -> list[dict]:
    if not query.strip():
        return []

    response = requests.get(
        "https://world.openbeautyfacts.org/cgi/search.pl",
        params={
            "search_terms": query.strip(),
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": max(1, min(limit, 10)),
            "fields": "code,product_name,product_name_en,brands,categories_tags,ingredients_text,image_url,quantity",
        },
        timeout=10,
    )
    response.raise_for_status()
    products = response.json().get("products") or []
    results = [
        _normalise_obf_product(item)
        for item in products
        if isinstance(item, dict) and (item.get("product_name") or item.get("product_name_en"))
    ][:limit]
    for product in results:
        save_external_product(product)
    return results


@tool
def search_open_beauty_facts(query: str, limit: int = 5) -> list[dict]:
    """
    Search Open Beauty Facts online for skincare products missing from the local DB.
    Use this only after the local product catalog has no matching results.
    """
    try:
        return _search_open_beauty_facts(query, limit=limit)
    except Exception as exc:
        return [{"error": f"Open Beauty Facts search failed: {exc}"}]


ONLINE_PRODUCT_SYSTEM_PROMPT = """You are a product research sub-agent for SkinIQ.
Your job is to fetch skincare product data from online sources when the local database has no match.
Always call search_open_beauty_facts. Return ONLY a JSON array of product objects from the tool output.
Do not add commentary or skincare advice."""


@lru_cache(maxsize=1)
def _online_product_agent():
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    api_key, model_name, base_url = load_openai_config("SKINIQ_ONLINE_PRODUCT_AGENT_MODEL")
    model = ChatOpenAI(
        **openai_chat_kwargs(
            model_name,
            api_key,
            base_url,
            temperature=0.0,
            timeout=20,
            max_completion_tokens=600,
            reasoning_effort="minimal",
            verbosity="low",
        )
    )
    return create_agent(
        model,
        tools=[search_open_beauty_facts],
        system_prompt=ONLINE_PRODUCT_SYSTEM_PROMPT,
    )


def _ask_online_product_agent(query: str, limit: int = 5) -> list[dict]:
    api_key, _, _ = load_openai_config("SKINIQ_ONLINE_PRODUCT_AGENT_MODEL")
    if not api_key:
        return _search_open_beauty_facts(query, limit=limit)

    try:
        result = _online_product_agent().invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Find up to {limit} skincare products matching: {query}",
                    }
                ]
            }
        )
        content = result["messages"][-1].content
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return _search_open_beauty_facts(query, limit=limit)


PRODUCT_CARD_SELECTION_PROMPT = """You select product cards for the SkinIQ UI.
Return ONLY a JSON array.

Given the assistant response, return ONLY the products whose exact name appears verbatim
(or near-verbatim) in the assistant response. The assistant response is the source of truth.
Do NOT include products that are merely similar, alternative sizes, or in the same product
family if not explicitly named. When in doubt, exclude.
Do not include products that are only part of the user's existing routine. Do not include
generic category advice such as "use a broad-spectrum SPF" unless a specific product is named.

Each item must use this shape:
{"index": 0, "product_name": "name", "brand": "brand", "search_query": "optional query"}

Use "index" only when a candidate is clearly the same product named in the response.
Otherwise return product_name, brand if known, and a search_query for that exact product."""


def _safe_json_array(content: str) -> list[dict]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _candidate_payload(products: list[dict]) -> list[dict]:
    payload = []
    for idx, product in enumerate(products):
        payload.append(
            {
                "index": idx,
                "product_name": product.get("product_name") or "",
                "brand": product.get("brand") or "",
                "quantity": product.get("quantity") or "",
                "key_ingredients": (
                    product.get("key_ingredients")
                    or product.get("active_ingredients")
                    or product.get("ingredients")
                    or []
                )[:8],
                "categories": (product.get("categories") or [])[:6],
                "source": product.get("source") or "",
            }
        )
    return payload


def _ask_product_card_llm(
    response: str,
    candidate_products: list[dict],
    limit: int,
) -> list[dict]:
    api_key, model, base_url = load_openai_config("SKINIQ_PRODUCT_CARD_SELECTION_MODEL")
    if not api_key:
        return []

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        **openai_chat_kwargs(
            model,
            api_key,
            base_url,
            temperature=0.0,
            timeout=20,
            max_completion_tokens=300,
            reasoning_effort="minimal",
            verbosity="low",
        )
    )
    user_payload = {
        "limit": limit,
        "assistant_response": response,
        "candidate_products": _candidate_payload(candidate_products),
    }
    result = llm.invoke(
        [
            SystemMessage(content=PRODUCT_CARD_SELECTION_PROMPT),
            HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
        ]
    )
    content = result.content if isinstance(result.content, str) else json.dumps(result.content)
    return _safe_json_array(content)


def _has_catalog_data(product: dict) -> bool:
    """True when the product has been loaded from a catalog source (not a bare name-only entry)."""
    return bool(
        product.get("image_url")
        or product.get("ingredients")
        or product.get("key_ingredients")
        or product.get("categories")
        or product.get("ingredients_raw")
    )


def _same_product(left: dict, right: dict) -> bool:
    left_name, left_brand = _product_key(left)
    right_name, right_brand = _product_key(right)
    if not left_name or not right_name:
        return False
    return left_name == right_name and (not left_brand or not right_brand or left_brand == right_brand)


_GENERIC_PRODUCT_TOKENS = {
    "serum", "cream", "lotion", "moisturiser", "moisturizer", "gel", "balm",
    "mask", "toner", "cleanser", "sunscreen", "oil", "treatment", "essence",
    "skin", "face", "day", "night", "spray", "wash",
}


def _product_named_in_response(product: dict, response: str) -> bool:
    raw_name = str(product.get("product_name") or "")
    name = _normalise_match_text(raw_name)
    if not name:
        return False
    response_text = _normalise_match_text(response)
    if name in response_text:
        return True

    family_name = product_family_name(raw_name)
    response_family = product_family_name(response)
    if family_name and len(family_name) > 12 and family_name in response_family:
        return True

    # Catalog products often have variant suffixes after a comma, e.g.
    #   "Skin + Me Breakouts + Visible Pores Serum, for Dry to Normal Skin, ..."
    # The response writer typically only quotes the part before the first
    # comma. Match if that core prefix family is contained in the response.
    core = raw_name.split(",", 1)[0]
    core_family = product_family_name(core)
    if core_family and len(core_family) > 12 and core_family in response_family:
        return True

    # DB name may differ from what LLM wrote (extra suffix, missing "with",
    # different size, etc.). Compare distinctive tokens only — drop generic
    # product-category words (serum, cream, ...) so two products in the same
    # category aren't treated as the same.
    stop = {"for", "the", "and", "with", "a", "an", "to", "of", "in", "by", "on", "or"}
    name_tokens = {
        t for t in family_name.split()
        if t not in stop and t not in _GENERIC_PRODUCT_TOKENS and len(t) > 1
    }
    if len(name_tokens) >= 3:
        matched = sum(1 for t in name_tokens if t in response_text)
        if matched / len(name_tokens) >= 0.85:
            return True

    return False


_STOP = {"for", "the", "and", "with", "a", "an", "to", "of", "in", "by", "on", "or"}


def _query_match_ratio(query: str, candidate_name: str) -> tuple[int, float]:
    """Return (overlap_count, overlap_ratio) of distinctive tokens between query and candidate."""
    query_tokens = {
        t for t in product_family_name(query).split()
        if t not in _STOP and len(t) > 1
    }
    if not query_tokens:
        return 0, 0.0
    candidate_tokens = {
        t for t in product_family_name(candidate_name).split()
        if t not in _STOP and len(t) > 1
    }
    overlap = query_tokens & candidate_tokens
    return len(overlap), len(overlap) / len(query_tokens)


def _find_catalog_product(
    query: str,
    brand: str = "",
    allow_first: bool = True,
    use_obf: bool = False,
) -> dict | None:
    if not query.strip():
        return None
    try:
        matches = search_products(query.strip(), limit=5, use_obf_fallback=use_obf)
    except TypeError:
        matches = search_products(query.strip(), limit=5)
    except Exception:
        return None

    target = {"product_name": query, "brand": brand}
    for match in matches:
        if _same_product(target, match):
            return match

    if allow_first and matches:
        first = matches[0]
        first_name = str(first.get("product_name") or "")
        overlap, ratio = _query_match_ratio(query, first_name)
        # For specific multi-token queries, require strong overlap so a
        # related-but-different product (e.g. "Copper Peptides" returned for
        # a "Hyaluronic Acid" query) is not silently substituted.
        query_distinctive_count = len({
            t for t in product_family_name(query).split()
            if t not in _STOP and len(t) > 1
        })
        if query_distinctive_count >= 4:
            if ratio < 0.7:
                return None
        elif use_obf and first.get("source") == "open_beauty_facts":
            if overlap < 2:
                return None
        return first
    return None


def _explicit_product_suggestion_queries(response: str) -> list[str]:
    queries = []
    for line in response.splitlines():
        match = re.search(
            r"\b(?:product suggestion|recommendation|suggested product|suggested sunscreen)\b\s*\**\s*:\s*\**\s*(.+?)(?:[.;]|$)",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        query = re.sub(r"^[\s*\-:]+|[\s*\-:]+$", "", match.group(1)).strip()
        query = re.split(r"\s+[·|]\s+", query, maxsplit=1)[0].strip()
        if query:
            queries.append(query)
    return queries


def _heading_product_suggestion_queries(response: str) -> list[str]:
    queries = []
    markdown_heading_re = re.compile(r"^\s*#{4}\s+(?P<name>[^\n]{4,160})\s*$")
    heading_slot_re = re.compile(
        r"^\s*(?:\d+\.\s*)?(?P<name>[A-Z][^\n:]{8,120}?)\s*"
        r"\((?:Active|Rest)?\s*(?:AM|PM)\)\s*$"
    )
    product_like_re = re.compile(
        r"^\s*(?:\d+\.\s*)?(?P<name>[A-Z][^\n:]{8,180}?\b"
        r"(?:SPF\s*\d+\+?|PA\+{2,4}|\d+(?:\.\d+)?\s*(?:ml|g|oz|fl\s*oz))"
        r"[^\n:]*?)\s*$",
        flags=re.IGNORECASE,
    )
    # Lines that look like a product title and end on a category keyword,
    # e.g. "Olay Hyaluronic Acid 24 + Vitamin B5 Day Gel Moisturiser with Niacinamide"
    category_heading_re = re.compile(
        r"^\s*(?:\d+\.\s*)?(?P<name>[A-Z][^\n:]{12,180}?\b"
        r"(?:Moisturi[sz]er|Serum|Cleanser|Toner|Sunscreen|Mask|Essence|"
        r"Balm|Lotion|Treatment|Eye\s+Cream|Day\s+Cream|Night\s+Cream|"
        r"Face\s+Oil|Face\s+Wash|Spot\s+Treatment)"
        r"\b[^\n:]{0,80}?)\s*$"
    )
    excluded_prefixes = (
        "Based on ",
        "Why it fits",
        "Gap it fills",
        "Usage",
        "These additions",
        "Starting from ",
        "Use ",
        "Apply ",
        "Reapply ",
        "Start ",
        "Patch test ",
        "Next step",
        "Which would ",
        "Lightweight ",
        "Serum texture",
        "Gentle foaming",
        "Contains ",
        "Includes ",
        "Fits ",
        "Fills ",
        "Very ",
    )

    for line in response.splitlines():
        stripped = line.strip().strip("*")
        if not stripped or stripped.startswith(excluded_prefixes):
            continue
        # Sentence-style lines ending in a period are descriptions, not headings.
        # Markdown headings handled separately so they keep working.
        is_markdown_heading = stripped.startswith("#")
        if not is_markdown_heading and stripped.endswith((".", "!", "?")):
            continue
        match = (
            markdown_heading_re.match(stripped)
            or heading_slot_re.match(stripped)
            or product_like_re.match(stripped)
            or category_heading_re.match(stripped)
        )
        if not match:
            continue
        query = re.sub(r"\s+", " ", match.group("name")).strip(" -")
        if query:
            queries.append(query)
            size_variant_query = re.sub(
                r"\s*[-–—]?\s*\b\d+(?:\.\d+)?\s*(?:ml|l|g|kg|oz|fl\s*oz)\s*/\s*"
                r"\d+(?:\.\d+)?\s*(?:ml|l|g|kg|oz|fl\s*oz)\b\s*$",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip(" -")
            if size_variant_query and size_variant_query != query:
                queries.append(size_variant_query)
    return queries


def _response_product_suggestion_queries(response: str, limit: int | None = None) -> list[str]:
    queries = []
    seen = set()
    for query in _explicit_product_suggestion_queries(response) + _heading_product_suggestion_queries(response):
        cleaned = re.sub(r"\s+", " ", str(query or "")).strip(" -:;.")
        key = product_family_name(cleaned)
        if not cleaned or not key or key in seen:
            continue
        queries.append(cleaned)
        seen.add(key)
        if limit and len(queries) >= limit:
            break
    return queries


@tool
def extract_recommended_product_names_from_response(response: str, limit: int = 2) -> list[str]:
    """
    Deterministically extract specific product names that appear as recommendations
    in an assistant response. This does not call an LLM or external services.
    """
    if not response.strip():
        return []
    return _response_product_suggestion_queries(response, limit=max(1, min(limit, 10)))


def _minimal_product_card(name: str, brand: str = "") -> dict | None:
    product_name = re.sub(r"\s+", " ", str(name or "")).strip(" -:;.")
    if not product_name:
        return None
    return {
        "product_name": product_name,
        "brand": re.sub(r"\s+", " ", str(brand or "")).strip(),
        "quantity": "",
        "image_url": "",
        "key_ingredients": [],
        "active_ingredients": [],
        "ingredients": [],
        "categories": [],
        "source": "assistant_recommendation",
    }


def _fallback_named_product_cards(
    response: str,
    limit: int,
    existing_products: list[dict] | None = None,
) -> list[dict]:
    cards = []
    seen = {
        product_family_name(str(product.get("product_name") or ""))
        for product in existing_products or []
        if isinstance(product, dict)
    }
    for query in _response_product_suggestion_queries(response):
        card = _minimal_product_card(query)
        if not card:
            continue
        key = product_family_name(card["product_name"])
        if key in seen:
            continue
        cards.append(card)
        seen.add(key)
        if len(cards) >= limit:
            break
    return cards


def _resolve_catalog_product_cards_first(
    response: str,
    candidate_products: list[dict],
    limit: int,
) -> list[dict]:
    resolved = []

    for candidate in candidate_products:
        if not _product_named_in_response(candidate, response):
            continue
        product = _find_catalog_product(
            str(candidate.get("product_name") or ""),
            brand=str(candidate.get("brand") or ""),
            allow_first=False,
        )
        resolved.append(product or candidate)
        if len(resolved) >= limit:
            return resolved

    for query in _response_product_suggestion_queries(response):
        product = _find_catalog_product(query, allow_first=True, use_obf=True)
        if product and _product_named_in_response(product, response):
            resolved.append(product)
        if len(resolved) >= limit:
            break

    return resolved


def _resolve_llm_product_selection(
    selected: list[dict],
    candidate_products: list[dict],
    response: str,
    limit: int,
) -> list[dict]:
    resolved = []
    for item in selected:
        product = None
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(candidate_products):
            candidate = candidate_products[idx]
            if _product_named_in_response(candidate, response):
                product = candidate
        else:
            query = str(item.get("search_query") or item.get("product_name") or "").strip()
            brand = str(item.get("brand") or "").strip()
            if query and _normalise_match_text(query) in _normalise_match_text(response):
                candidate = _find_catalog_product(query, brand=brand, allow_first=True, use_obf=True)
                # Only accept the catalog match if its name actually appears
                # in the assistant response — guards against the search
                # returning a near-name match that the response never named.
                if candidate and _product_named_in_response(candidate, response):
                    product = candidate

        if product:
            resolved.append(product)
        if len(resolved) >= limit:
            break
    return resolved


@tool
def ask_online_product_research_agent(query: str, limit: int = 15) -> list[dict]:
    """
    Delegate to the online product research sub-agent for products missing from the DB.
    Use this when search_product_catalog returns no local catalog matches.
    """
    try:
        return _ask_online_product_agent(query.strip(), limit=limit)
    except Exception as exc:
        return [{"error": f"Online product research failed: {exc}"}]


@tool
def search_product_catalog(query: str, limit: int = 5) -> list[dict]:
    """
    Search the local skincare product catalog by product name or brand, then
    delegate to the online product research sub-agent if the product is not in DB.
    Use this when the user asks for product recommendations or names a product.
    """
    if not query.strip():
        return []

    try:
        local_results = list(search_products(query.strip(), limit=limit) or [])
    except Exception as exc:
        local_results = [{"error": f"Product search failed: {exc}"}]

    if local_results and not any("error" in item for item in local_results if isinstance(item, dict)):
        return local_results

    try:
        online_results = _ask_online_product_agent(query.strip(), limit=limit)
    except Exception as exc:
        online_results = [{"error": f"Online product research failed: {exc}"}]

    if online_results:
        for product in online_results:
            if isinstance(product, dict) and "error" not in product:
                save_external_product(product)
        return online_results
    return local_results


@tool
def select_recommended_product_cards(
    response: str,
    candidate_products: list[dict] | None = None,
    routine_items: list[dict] | None = None,
    limit: int = 2,
) -> list[dict]:
    """
    Use an LLM to select the product recommendations that should be rendered as
    UI cards. Excludes products already in the user's routine.
    """
    if not response.strip():
        return []

    candidates = _exclude_routine_products(
        _dedupe_products(candidate_products or []),
        routine_items,
    )
    resolved = _resolve_catalog_product_cards_first(response, candidates, limit=limit)
    if len(resolved) >= limit:
        result = _exclude_routine_products(_dedupe_products(resolved), routine_items)[:limit]
        return result

    try:
        selected = _ask_product_card_llm(response, candidates, limit=limit)
    except Exception:
        selected = []

    resolved.extend(
        _resolve_llm_product_selection(
            selected,
            candidates,
            response=response,
            limit=limit - len(resolved),
        )
    )
    if len(resolved) < limit:
        resolved.extend(
            _fallback_named_product_cards(
                response,
                limit - len(resolved),
                existing_products=resolved,
            )
        )
    result = _exclude_routine_products(_dedupe_products(resolved), routine_items)[:limit]
    return result


product_tools = [
    search_product_catalog,
    select_recommended_product_cards,
    extract_recommended_product_names_from_response,
    ask_online_product_research_agent,
    search_open_beauty_facts,
]
