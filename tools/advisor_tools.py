from __future__ import annotations

import re

from langchain_core.tools import tool

from utils.product_match import product_family_name, product_identity_key


_RETINOID_RE = re.compile(
    r"\b(?:retinol|retinal|retinoid|retinoids|retinoate|retinaldehyde|retinyl)\b",
    re.IGNORECASE,
)


def _text_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _product_text(product: dict) -> str:
    return " ".join(
        _text_list(product.get("key_ingredients") or [])
        + _text_list(product.get("active_ingredients") or [])
        + _text_list(product.get("ingredients") or [])
        + _text_list(product.get("categories") or [])
        + [str(product.get("ingredients_raw") or "")]
        + [str(product.get("product_name") or "")]
    ).lower()


def _routine_items(user_routine: dict | None) -> list[dict]:
    if not isinstance(user_routine, dict):
        return []
    return [
        item
        for item in user_routine.get("items") or user_routine.get("routine") or []
        if isinstance(item, dict)
    ]


def _routine_product_keys(user_routine: dict | None) -> set[tuple[str, str]]:
    return {
        product_identity_key(item, family=True)
        for item in _routine_items(user_routine)
        if item.get("product_name")
    }


def _routine_has_retinoid(user_routine: dict | None) -> bool:
    return any(_RETINOID_RE.search(_product_text(item)) for item in _routine_items(user_routine))


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


def _avoid_terms_for_profile(profile: dict | None) -> list[str]:
    if not isinstance(profile, dict):
        return []
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


def _score_product(product: dict, profile: dict | None, message: str) -> tuple[int, list[str], list[str]]:
    text = _product_text(product)
    profile = profile if isinstance(profile, dict) else {}
    positive: list[str] = []
    cautions: list[str] = []
    score = 0

    for concern in _text_list(profile.get("concerns") or []):
        if concern.lower() in text:
            score += 2
            positive.append(f"matches concern: {concern}")
    for goal in _text_list(profile.get("goals") or []):
        if goal.lower() in text:
            score += 2
            positive.append(f"matches goal: {goal}")

    message_tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", message.lower())
        if token not in {"skin", "product", "recommend", "routine", "please", "what"}
    }
    matched_tokens = sorted(token for token in message_tokens if token in text)
    if matched_tokens:
        score += min(len(matched_tokens), 5)
        positive.append("matches request terms: " + ", ".join(matched_tokens[:5]))

    avoid_terms = [term for term in _avoid_terms_for_profile(profile) if term and term in text]
    if avoid_terms:
        score -= 8
        cautions.append("contains avoided/allergen terms: " + ", ".join(avoid_terms[:4]))

    if _RETINOID_RE.search(text) and _routine_has_retinoid(profile.get("routine")):
        score -= 4

    return score, positive, cautions


def analyse_product_candidates(
    *,
    candidate_products: list[dict],
    user_profile: dict | None,
    user_routine: dict | None,
    message: str,
    previously_recommended: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Rank product candidates for fit before the answer writer sees them."""
    routine_keys = _routine_product_keys(user_routine)
    previous_families = {
        product_family_name(name)
        for name in previously_recommended or []
        if str(name).strip()
    }
    routine_retinoid = _routine_has_retinoid(user_routine)
    explicit_retinoid = _explicitly_requests_retinoid(message)

    ranked = []
    seen = set()
    profile_with_routine = dict(user_profile or {})
    profile_with_routine["routine"] = user_routine or {}
    for product in candidate_products or []:
        if not isinstance(product, dict) or not product.get("product_name"):
            continue
        key = product_identity_key(product, family=True)
        family = product_family_name(str(product.get("product_name") or ""))
        if key in seen or key in routine_keys or family in previous_families:
            continue
        seen.add(key)
        text = _product_text(product)
        if routine_retinoid and not explicit_retinoid and _RETINOID_RE.search(text):
            continue
        score, reasons, cautions = _score_product(product, profile_with_routine, message)
        ranked.append(
            {
                "product_name": product.get("product_name") or "",
                "brand": product.get("brand") or "",
                "quantity": product.get("quantity") or "",
                "score": score,
                "reasons": reasons[:4],
                "cautions": cautions[:4],
                "key_ingredients": (
                    product.get("key_ingredients")
                    or product.get("active_ingredients")
                    or product.get("ingredients")
                    or []
                )[:8],
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(1, min(limit, 10))]


def analyse_ingredient_candidates(
    *,
    candidate_ingredients: list[dict],
    user_profile: dict | None,
    user_routine: dict | None,
    limit: int = 5,
) -> list[dict]:
    """Rank ingredient candidates and mark ingredients already present in routine."""
    profile = user_profile if isinstance(user_profile, dict) else {}
    routine_text = " ".join(_product_text(item) for item in _routine_items(user_routine))
    concerns_goals = " ".join(
        _text_list(profile.get("concerns") or []) + _text_list(profile.get("goals") or [])
    ).lower()

    ranked = []
    seen = set()
    for ingredient in candidate_ingredients or []:
        if not isinstance(ingredient, dict):
            continue
        name = str(ingredient.get("inci_name") or ingredient.get("display_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        functions = _text_list(ingredient.get("functions") or [])
        conditions = _text_list(ingredient.get("conditions") or [])
        score = 0
        reasons = []
        haystack = " ".join(functions + conditions + _text_list(ingredient.get("common_names") or [])).lower()
        for term in re.findall(r"[a-z0-9]{4,}", concerns_goals):
            if term in haystack:
                score += 2
        if key in routine_text:
            score -= 6
            reasons.append("already present in routine")
        if functions:
            reasons.append("functions: " + ", ".join(functions[:4]))
        ranked.append(
            {
                "ingredient": name,
                "score": score,
                "already_in_routine": key in routine_text,
                "reasons": reasons[:4],
                "cautions": _text_list(ingredient.get("cautions") or [])[:3],
                "usage_guidance": ingredient.get("usage_guidance") or "",
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(1, min(limit, 10))]


advisor_tools = [tool(analyse_product_candidates), tool(analyse_ingredient_candidates)]
