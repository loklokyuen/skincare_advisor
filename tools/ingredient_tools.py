from __future__ import annotations

import re

from langchain_core.tools import tool

from services.ingredient_service import save_external_ingredient
from services.ingredient_service import search_ingredients


COMMON_INGREDIENT_TERMS = [
    "aha",
    "azelaic acid",
    "benzoyl peroxide",
    "bha",
    "ceramide",
    "ceramides",
    "glycolic acid",
    "glycerin",
    "hyaluronic acid",
    "lactic acid",
    "niacinamide",
    "peptide",
    "peptides",
    "retinal",
    "retinol",
    "salicylic acid",
    "squalane",
    "vitamin c",
]


@tool
def extract_ingredient_terms(text: str) -> list[str]:
    """
    Extract likely skincare ingredient terms from free text.
    Use this before ingredient lookup or conflict analysis.
    """
    lower_text = text.lower()
    terms = [term for term in COMMON_INGREDIENT_TERMS if term in lower_text]

    inci_like = re.findall(r"\b[a-z]+(?:\s+acid|ol|amide|in|ate)\b", lower_text)
    for term in inci_like:
        if term not in terms:
            terms.append(term)

    return terms


@tool
def search_ingredient_catalog(query: str, limit: int = 5) -> list[dict]:
    """
    Search the local skincare ingredient catalog by INCI name, display name, or alias.
    Use this when the user asks what an ingredient does or whether it suits them.
    """
    if not query.strip():
        return []

    try:
        return list(search_ingredients(query.strip(), limit=limit) or [])
    except Exception as exc:
        return [{"error": f"Ingredient search failed: {exc}"}]


@tool
def save_ingredient_to_catalog(ingredient: dict) -> dict:
    """
    Save externally researched ingredient details to Cloud SQL.
    Use this after fetching an ingredient that is missing from the local catalog.
    """
    try:
        saved = save_external_ingredient(ingredient)
        return {"saved": saved}
    except Exception as exc:
        return {"saved": False, "error": str(exc)}


ingredient_tools = [extract_ingredient_terms, search_ingredient_catalog, save_ingredient_to_catalog]
