from __future__ import annotations

import re

from graph.tracing import traceable
from graph.context_status import input_status
from schemas import GraphState
from services.product_service import search_products as _search_products_full
from services.product_service import search_products_for_routine
from services.rag_service import retrieve_semantic_context
from tools.ingredient_tools import extract_ingredient_terms, search_ingredient_catalog
from utils.product_match import product_identity_key


_CONCERN_INGREDIENTS = {
    "Acne / Breakouts": ["salicylic acid", "benzoyl peroxide", "niacinamide", "tea tree"],
    "Hyperpigmentation": ["vitamin c", "niacinamide", "kojic acid", "alpha arbutin", "aha"],
    "Anti-aging / Wrinkles": ["retinol", "peptides", "vitamin c", "hyaluronic acid"],
    "Dryness / Dehydration": ["hyaluronic acid", "glycerin", "ceramides", "squalane"],
    "Redness / Rosacea": ["centella asiatica", "azelaic acid", "niacinamide", "ceramides"],
    "Large Pores": ["niacinamide", "salicylic acid", "retinol", "vitamin c"],
    "Uneven Skin Tone": ["vitamin c", "alpha arbutin", "niacinamide", "aha"],
    "Dark Circles": ["caffeine", "vitamin c", "retinol", "peptides"],
    "Sensitivity / Irritation": ["ceramides", "centella asiatica", "aloe vera", "oat extract"],
    "Oiliness / Shine": ["niacinamide", "salicylic acid", "zinc", "clay"],
    "Texture / Roughness": ["aha", "bha", "retinol", "vitamin c"],
    "Sun Damage": ["vitamin c", "niacinamide", "retinol", "alpha arbutin"],
}

_SKIN_TYPE_ALIASES = {
    "oily": "oily",
    "oiliness": "oily",
    "combination": "combination",
    "combo": "combination",
    "dry": "dry",
    "normal": "normal",
    "sensitive": "sensitive",
}

_OIL_BALANCE_SKIN_TYPES = {"oily", "combination", "dry", "normal"}

_RETINOID_RE = re.compile(
    r"\b(?:retinol|retinal|retinoid|retinoids|retinoate|retinaldehyde|retinyl)\b",
    re.IGNORECASE,
)


def _dedupe_by_key(items: list[dict], key: str) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        value = str(item.get(key) or "").lower().strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(item)
    return deduped


def _product_key(item: dict) -> tuple[str, str]:
    return product_identity_key(item, family=True)


def _exclude_routine_products(products: list[dict], routine_items: list[dict]) -> list[dict]:
    routine_keys = {
        _product_key(item)
        for item in routine_items
        if isinstance(item, dict) and item.get("product_name")
    }
    routine_names = {name for name, _brand in routine_keys if name}

    filtered = []
    for product in products:
        name, brand = _product_key(product)
        if (name, brand) in routine_keys or name in routine_names:
            continue
        filtered.append(product)
    return filtered


def _text_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _target_ingredients_for_profile(profile: dict) -> list[str]:
    concerns = _text_list(profile.get("concerns") or [])
    if profile.get("sensitive_skin") and "Sensitivity / Irritation" not in concerns:
        concerns.append("Sensitivity / Irritation")
    targets = []
    seen = set()
    for concern in concerns:
        for ingredient in _CONCERN_INGREDIENTS.get(concern, []):
            key = ingredient.lower().strip()
            if key and key not in seen:
                targets.append(ingredient)
                seen.add(key)
    return targets


def _avoid_terms_for_profile(profile: dict) -> list[str]:
    terms = []
    if profile.get("avoid_fragrance"):
        terms.extend(["fragrance", "parfum"])
    if profile.get("avoid_alcohol"):
        terms.extend(["alcohol denat", "denatured alcohol", "sd alcohol"])
    if profile.get("avoid_parabens"):
        terms.append("paraben")
    if profile.get("avoid_silicones"):
        terms.extend(["dimethicone", "silicone"])
    terms.extend(term.lower() for term in _text_list(profile.get("allergens") or []))
    return terms


def _profile_skin_type(profile: dict) -> str:
    raw = str(profile.get("skin_type") or "").lower().strip()
    skin_type = _SKIN_TYPE_ALIASES.get(raw, raw)
    return "" if skin_type == "sensitive" else skin_type


def _explicit_product_skin_types(product: dict) -> set[str]:
    text = " ".join(
        [str(product.get("product_name") or "")]
        + _text_list(product.get("categories") or [])
    ).lower()
    if not text:
        return set()

    targets: set[str] = set()
    connector = r"(?:to|and|&|/|-)"
    for left, right in re.findall(
        rf"\b(oily|combination|combo|dry|normal|sensitive)\s+{connector}\s+"
        r"(oily|combination|combo|dry|normal|sensitive)\s+skin\b",
        text,
    ):
        targets.add(_SKIN_TYPE_ALIASES[left])
        targets.add(_SKIN_TYPE_ALIASES[right])

    for label, canonical in _SKIN_TYPE_ALIASES.items():
        if re.search(rf"\b(?:for\s+)?{re.escape(label)}\s+skin\b", text):
            targets.add(canonical)
        if re.search(rf"\bskin\s+(?:type\s+)?(?:for\s+)?{re.escape(label)}\b", text):
            targets.add(canonical)

    return targets


def _is_product_skin_type_mismatch(product: dict, profile: dict) -> bool:
    skin_type = _profile_skin_type(profile)
    explicit_targets = _explicit_product_skin_types(product)
    if skin_type not in _OIL_BALANCE_SKIN_TYPES:
        return False
    oil_balance_targets = explicit_targets & _OIL_BALANCE_SKIN_TYPES
    return bool(oil_balance_targets and skin_type not in oil_balance_targets)


def _filter_skin_type_mismatches(products: list[dict], profile: dict) -> list[dict]:
    if not isinstance(profile, dict) or not products:
        return products
    return [
        product
        for product in products
        if not _is_product_skin_type_mismatch(product, profile)
    ]


def _product_search_text(product: dict) -> str:
    return " ".join(
        _text_list(product.get("key_ingredients") or [])
        + _text_list(product.get("active_ingredients") or [])
        + _text_list(product.get("ingredients") or [])
        + _text_list(product.get("categories") or [])
        + [str(product.get("ingredients_raw") or "")]
        + [str(product.get("product_name") or "")]
    ).lower()


def _routine_has_retinoid(routine_items: list[dict]) -> bool:
    for item in routine_items:
        if not isinstance(item, dict):
            continue
        if _RETINOID_RE.search(_product_search_text(item)):
            return True
    return False


def _explicitly_requests_retinoid(message: str) -> bool:
    lower = message.lower()
    return bool(_RETINOID_RE.search(lower)) or any(
        phrase in lower
        for phrase in (
            "stronger vitamin a",
            "replace my retinol",
            "replace retinol",
            "retinol alternative",
            "retinoid alternative",
            "compare retinoids",
        )
    )


def _filter_duplicate_retinoids(
    products: list[dict],
    routine_items: list[dict],
    message: str,
) -> list[dict]:
    if not products or not _routine_has_retinoid(routine_items) or _explicitly_requests_retinoid(message):
        return products
    return [
        product
        for product in products
        if not _RETINOID_RE.search(_product_search_text(product))
    ]


def _build_recommendation_query(state: GraphState) -> str:
    """Build the product retrieval query from user intent plus saved personalization."""
    message = (state.get("message") or "").strip()
    profile = state.get("user_profile") or {}
    routine = state.get("user_routine") or {}
    if not isinstance(profile, dict):
        profile = {}
    if not isinstance(routine, dict):
        routine = {}

    parts = [message]
    skin_type = str(profile.get("skin_type") or "").strip()
    if skin_type:
        parts.append(f"Skin type: {skin_type}")
    if profile.get("sensitive_skin"):
        parts.append("Sensitive skin: yes")

    concerns = _text_list(profile.get("concerns") or [])
    if concerns:
        parts.append(f"Skin concerns: {', '.join(concerns)}")

    goals = _text_list(profile.get("goals") or [])
    if goals:
        parts.append(f"Goals: {', '.join(goals)}")

    notes = str(profile.get("notes") or "").strip()
    if notes:
        parts.append(f"User notes: {notes[:500]}")

    avoid = [
        key.replace("avoid_", "").replace("_", " ")
        for key in ("avoid_fragrance", "avoid_alcohol", "avoid_parabens", "avoid_silicones")
        if profile.get(key)
    ]
    if avoid:
        parts.append(f"Avoid: {', '.join(avoid)}")

    target_ingredients = _target_ingredients_for_profile(profile)
    if target_ingredients:
        parts.append(f"Helpful ingredients: {', '.join(target_ingredients)}")


    routine_items = [
        item
        for item in routine.get("items") or routine.get("routine") or []
        if isinstance(item, dict)
    ]
    if routine_items:
        routine_names = [
            str(item.get("product_name") or "").strip()
            for item in routine_items[:8]
            if str(item.get("product_name") or "").strip()
        ]
        routine_ingredients = []
        seen = set()
        for item in routine_items:
            for ingredient in _text_list(
                item.get("key_ingredients")
                or item.get("active_ingredients")
                or item.get("ingredients")
                or []
            ):
                key = ingredient.lower().strip()
                if key and key not in seen:
                    routine_ingredients.append(ingredient)
                    seen.add(key)
        if routine_names:
            parts.append(f"Current routine products: {', '.join(routine_names)}")
        if routine_ingredients:
            parts.append(
                f"Current routine key ingredients: {', '.join(routine_ingredients[:18])}"
            )

    return "\n".join(part for part in parts if part).strip() or message


def _product_profile_score(product: dict, profile: dict) -> int:
    target_ingredients = _target_ingredients_for_profile(profile)

    searchable = _product_search_text(product)

    score = sum(2 for ingredient in target_ingredients if ingredient in searchable)
    skin_type = _profile_skin_type(profile)
    if skin_type and skin_type in searchable:
        score += 1
    if profile.get("sensitive_skin") and "sensitive" in searchable:
        score += 1
    if _is_product_skin_type_mismatch(product, profile):
        score -= 8
    score -= sum(3 for term in _avoid_terms_for_profile(profile) if term and term in searchable)
    return score


def _rank_products_for_profile(products: list[dict], profile: dict) -> list[dict]:
    if not isinstance(profile, dict) or not products:
        return products
    products = _filter_skin_type_mismatches(products, profile)
    return sorted(
        products,
        key=lambda product: _product_profile_score(product, profile),
        reverse=True,
    )


def _find_matched_products(message: str) -> list[dict]:
    """Return ProductSearchResult dicts for products/brands mentioned in the message."""
    results = []
    try:
        results.extend(_search_products_full(message, limit=5, use_obf_fallback=False))
    except Exception:
        pass

    msg_lower = message.lower()
    matched = []
    for p in results:
        name = (p.get("product_name") or "").lower()
        brand = (p.get("brand") or "").lower()
        name_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", name)
            if len(token) > 3 and token not in {"serum", "skin", "solution", "control"}
        }
        mentioned_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", msg_lower)
            if len(token) > 3
        }
        if (
            len(name) > 4
            and name in msg_lower
            or (
                len(brand) > 3
                and brand in msg_lower
                and bool(name_tokens & mentioned_tokens)
            )
        ):
            matched.append(p)
    return matched


def _find_recommended_products(
    message: str,
    mode: str | None,
    embedding: list[float] | None = None,
) -> list[dict]:
    """Return UI-facing product cards for modes where the assistant may recommend products."""
    if mode not in {"recommend", "build"}:
        return []
    try:
        return search_products_for_routine(message, limit=15, embedding=embedding)
    except Exception:
        return []


def _card_from_search_result(product: dict) -> dict | None:
    name = str(product.get("product_name") or "").strip()
    if not name:
        return None

    if "image_url" in product or "key_ingredients" in product or "categories" in product:
        return {
            "product_name": name,
            "brand": product.get("brand") or "",
            "quantity": product.get("quantity") or "",
            "image_url": product.get("image_url") or "",
            "key_ingredients": product.get("key_ingredients") or product.get("ingredients") or [],
            "active_ingredients": product.get("active_ingredients") or product.get("key_ingredients") or [],
            "ingredients": product.get("ingredients") or [],
            "ingredients_raw": product.get("ingredients_raw") or "",
            "categories": product.get("categories") or [],
            "source": product.get("source") or "",
        }

    try:
        results = _search_products_full(name, limit=3, use_obf_fallback=False)
    except Exception:
        return None

    name_lower = name.lower()
    brand_lower = str(product.get("brand") or "").lower().strip()
    for result in results:
        result_name = str(result.get("product_name") or "").lower().strip()
        result_brand = str(result.get("brand") or "").lower().strip()
        if result_name == name_lower and (not brand_lower or result_brand == brand_lower):
            return result
    return results[0] if results else None


def _cards_from_product_context(products: list[dict]) -> list[dict]:
    cards = []
    for product in products:
        if product.get("source") == "session":
            continue
        card = _card_from_search_result(product)
        if card:
            cards.append(card)
    return cards


@traceable(name="retrieve_context")
def retrieve_context(state: GraphState) -> GraphState:
    """Retrieve product and ingredient context from Cloud SQL."""
    message = state.get("message", "")
    mode = state.get("mode")
    profile = state.get("user_profile") or {}
    routine = state.get("user_routine") or {}
    products = []
    ingredients = []
    product_card_candidates = []
    tool_outputs = list(state.get("tool_outputs") or [])
    status = input_status(mode, state.get("user_profile") or {}, routine, message)
    raw_items = []

    if not status["ready"]:
        return {
            **state,
            "input_status": status,
            "retrieved_products": [],
            "retrieved_ingredients": [],
            "matched_products": [],
            "tool_outputs": tool_outputs,
        }

    if isinstance(routine, dict):
        raw_items = routine.get("items") or routine.get("routine") or []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            products.append(
                {
                    "product_id": 0,
                    "product_name": item.get("product_name", ""),
                    "brand": item.get("brand") or "",
                    "quantity": item.get("quantity") or "",
                    "image_url": item.get("image_url") or "",
                    "ingredients": item.get("ingredients") or item.get("key_ingredients") or [],
                    "key_ingredients": item.get("key_ingredients") or [],
                    "active_ingredients": item.get("active_ingredients") or item.get("key_ingredients") or [],
                    "categories": item.get("categories") or [],
                    "source": "session",
                }
            )

    include_products = mode in {"analyse", "recommend", "build"}
    product_query = (
        _build_recommendation_query(state)
        if mode in {"recommend", "build"}
        else message
    )
    semantic_products, semantic_ingredients, query_embedding = retrieve_semantic_context(
        product_query,
        product_limit=15,
        ingredient_limit=15,
        include_products=include_products,
    )
    semantic_products = _filter_duplicate_retinoids(semantic_products, raw_items, message)
    semantic_products = _rank_products_for_profile(semantic_products, profile)

    if semantic_products:
        products.extend(semantic_products)
        product_card_candidates.extend(_cards_from_product_context(semantic_products))
    if include_products:
        tool_outputs.append(
            {
                "tool": "search_products_semantic",
                "input": product_query,
                "output_count": len(semantic_products),
            }
        )

    if semantic_ingredients:
        ingredients.extend(semantic_ingredients)
    tool_outputs.append(
        {
            "tool": "search_ingredients_semantic",
            "input": product_query,
            "output_count": len(semantic_ingredients),
        }
    )

    ingredient_terms = extract_ingredient_terms.invoke({"text": message})
    tool_outputs.append({"tool": "extract_ingredient_terms", "output": ingredient_terms})

    if mode in {"learn", "analyse"} and len(ingredients) < 3:
        for term in ingredient_terms[:3]:
            result = search_ingredient_catalog.invoke({"query": term, "limit": 3})
            ingredients.extend(result)
            tool_outputs.append({"tool": "search_ingredient_catalog", "input": term, "output": result})

    if mode == "recommend" and len([p for p in products if p.get("source") != "session"]) < 3:
        try:
            result = search_products_for_routine(product_query, limit=15, embedding=query_embedding)
        except Exception:
            result = []
        result = _filter_duplicate_retinoids(result, raw_items, message)
        result = _rank_products_for_profile(result, profile)
        product_card_candidates.extend(
            item
            for item in result
            if isinstance(item, dict) and "error" not in item and item.get("product_name")
        )
        products.extend(
            {
                "product_id": 0,
                "product_name": item.get("product_name", ""),
                "brand": item.get("brand") or "",
                "quantity": item.get("quantity") or "",
                "image_url": item.get("image_url") or "",
                "ingredients": item.get("ingredients") or item.get("key_ingredients") or [],
                "key_ingredients": item.get("key_ingredients") or [],
                "active_ingredients": item.get("active_ingredients") or item.get("key_ingredients") or [],
                "categories": item.get("categories") or [],
                "source": item.get("source") or "catalog_search",
            }
            for item in result
            if isinstance(item, dict) and "error" not in item
        )
        tool_outputs.append(
            {"tool": "search_products_for_routine", "input": product_query, "output": result}
        )

    matched_source_products = _find_matched_products(message)
    if mode == "analyse":
        matched_products = _dedupe_by_key(matched_source_products, "product_name")
    else:
        matched_products = _dedupe_by_key(
            matched_source_products
            + _find_recommended_products(product_query, mode, query_embedding)
            + product_card_candidates
            + _cards_from_product_context(products),
            "product_name",
        )
    if mode in {"recommend", "build"}:
        matched_products = _filter_skin_type_mismatches(matched_products, profile)
        matched_products = _filter_duplicate_retinoids(matched_products, raw_items, message)
    matched_products = _exclude_routine_products(matched_products, raw_items if isinstance(routine, dict) else [])

    return {
        **state,
        "retrieved_products": _dedupe_by_key(products, "product_name"),
        "retrieved_ingredients": _dedupe_by_key(ingredients, "inci_name"),
        "matched_products": matched_products,
        "tool_outputs": tool_outputs,
        "ingredient_terms": ingredient_terms,
    }
