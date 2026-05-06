from __future__ import annotations

import logging

from langchain_core.tools import tool

log = logging.getLogger(__name__)

# ── Conflict rules — loaded from DB, with in-process cache ────────────────────

_conflict_rules_cache: list[dict] | None = None

# Hardcoded fallback used when the DB table doesn't exist yet.
_FALLBACK_RULES = [
    {
        "a": "retinol", "b": "benzoyl peroxide", "risk_level": "high",
        "reason": "Benzoyl peroxide can oxidise retinoids and increase irritation.",
        "suggestion": "Separate into different routines or alternate nights.",
    },
    {
        "a": "retinol", "b": "glycolic acid", "risk_level": "medium",
        "reason": "Retinoids plus strong AHAs can cause dryness and barrier disruption.",
        "suggestion": "Alternate nights; use AHA in the AM and retinol at night.",
    },
    {
        "a": "retinol", "b": "salicylic acid", "risk_level": "medium",
        "reason": "Retinoids plus BHA can be too drying, especially on sensitive skin.",
        "suggestion": "Introduce slowly; separate use if sensitivity appears.",
    },
    {
        "a": "vitamin c", "b": "glycolic acid", "risk_level": "medium",
        "reason": "Both are low-pH; layering them can sting and irritate sensitised skin.",
        "suggestion": "Use vitamin C in the AM and glycolic acid in the PM.",
    },
]


def _load_conflict_rules() -> list[dict]:
    """Load conflict rules from Cloud SQL, falling back to the hardcoded set."""
    global _conflict_rules_cache
    if _conflict_rules_cache is not None:
        return _conflict_rules_cache

    try:
        from utils.db import run_query
        rows = run_query(
            """
            SELECT ingredient_a, ingredient_b, risk_level, reason, suggestion
            FROM skincare_advisor.ingredient_conflicts
            """
        )
        if rows:
            _conflict_rules_cache = [
                {
                    "a": r["ingredient_a"].lower(),
                    "b": r["ingredient_b"].lower(),
                    "risk_level": r["risk_level"],
                    "reason": r["reason"],
                    "suggestion": r["suggestion"],
                }
                for r in rows
            ]
            log.debug("Loaded %d conflict rules from DB", len(_conflict_rules_cache))
            return _conflict_rules_cache
    except Exception as exc:
        log.warning("Could not load conflict rules from DB: %s — using fallback", exc)

    _conflict_rules_cache = _FALLBACK_RULES
    return _conflict_rules_cache


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _normalize(value: str) -> str:
    return " ".join((value or "").lower().split())


def _item_text(item: dict) -> str:
    ingredients = item.get("key_ingredients") or item.get("ingredients") or []
    categories = item.get("categories") or []
    return " ".join([
        item.get("product_name", ""),
        item.get("brand", ""),
        " ".join(str(i) for i in ingredients),
        " ".join(str(c) for c in categories),
    ]).lower()


def _item_contains_term(item: dict, term: str) -> bool:
    return _normalize(term) in _item_text(item)


def _schedule_key(item: dict) -> tuple[str, str, str]:
    scope = item.get("scope") or "unknown"
    group = item.get("group") or ("Daily" if scope == "daily" else "unknown")
    time = item.get("time") or "unknown"
    return (scope, group, time)


def _scheduled_together(left_items: list[dict], right_items: list[dict]) -> bool:
    if not left_items or not right_items:
        return True
    for left in left_items:
        for right in right_items:
            if left is right:
                return True
            lk, rk = _schedule_key(left), _schedule_key(right)
            if "unknown" in lk or "unknown" in rk:
                return True
            if lk == rk:
                return True
    return False


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def analyse_routine_basics(profile: dict, routine_items: list[dict]) -> dict:
    """
    Analyse the user's routine for broad gaps and suitability notes.
    Use this whenever the user asks about their routine.
    """
    concerns = profile.get("concerns") or []
    skin_type = profile.get("skin_type") or ""
    routine_text = " ".join(_item_text(i) for i in routine_items if isinstance(i, dict))
    ingredients = " ".join(
        ing.lower()
        for item in routine_items if isinstance(item, dict)
        for ing in (item.get("key_ingredients") or item.get("ingredients") or [])
    )

    gaps: list[str] = []
    suitability_notes: list[str] = []

    if "Dryness / Dehydration" in concerns and not any(
        t in ingredients for t in ("hyaluronic", "glycerin", "ceramide", "squalane")
    ):
        gaps.append("Routine may be light on humectants or barrier-supporting lipids.")
    if skin_type == "Sensitive" and any(
        t in ingredients for t in ("retinol", "glycolic", "salicylic", "benzoyl peroxide")
    ):
        suitability_notes.append("Sensitive skin may need slower introduction of strong actives.")

    return {"gaps": gaps, "suitability_notes": suitability_notes}


@tool
def flag_common_conflicts(ingredient_terms: list[str], routine_items: list[dict]) -> list[dict]:
    """
    Flag common skincare active conflicts from ingredient terms and routine products.
    Use this when the user asks about conflicts, irritation, or product combinations.
    """
    rules = _load_conflict_rules()
    terms: set[str] = {t.lower() for t in ingredient_terms}
    product_lookup: dict[str, list[dict]] = {}

    for item in routine_items:
        if not isinstance(item, dict):
            continue
        for rule in rules:
            for term in (rule["a"], rule["b"]):
                if _item_contains_term(item, term):
                    terms.add(term)
                    product_lookup.setdefault(term, []).append(item)

    conflicts = []
    for rule in rules:
        if rule["a"] not in terms or rule["b"] not in terms:
            continue
        left_items = product_lookup.get(rule["a"], [])
        right_items = product_lookup.get(rule["b"], [])
        if not _scheduled_together(left_items, right_items):
            continue

        involved_products = sorted({
            item.get("product_name", "Routine product")
            for item in left_items + right_items
        })
        conflicts.append({
            "risk_level": rule["risk_level"],
            "products_involved": involved_products,
            "ingredients_involved": [rule["a"], rule["b"]],
            "reason": rule["reason"],
            "suggestion": rule["suggestion"],
        })

    return conflicts


analysis_tools = [analyse_routine_basics, flag_common_conflicts]
