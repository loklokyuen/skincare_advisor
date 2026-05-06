from __future__ import annotations

from psycopg2.extras import Json

from utils.db import run_query
from utils.db import run_mutation
from utils.ingredient_format import format_ingredient_name, format_ingredient_names
from schemas import IngredientDetail


def _format_ingredient(row: dict) -> IngredientDetail:
    inci_name = format_ingredient_name(row.get("inci_name", ""))
    # Combine safety_notes and any conflict summary into cautions
    cautions: list[str] = []
    if row.get("safety_notes"):
        cautions.append(row["safety_notes"])
    for conflict in row.get("known_conflicts") or []:
        cautions.append(f"Caution with {conflict}")

    return {
        "ingredient_id": row.get("id") or 0,
        "inci_name": inci_name,
        # No separate display_name column — use inci_name as the label
        "display_name": inci_name,
        "common_names": format_ingredient_names(row.get("common_names") or []),
        "category": "",  # not stored in this schema
        "functions": row.get("functions") or [],
        "cautions": cautions,
        "suitable_for": row.get("suitable_skin_types") or [],
        "avoid_for": row.get("avoid_skin_types") or [],
        # Extra fields available but not in IngredientDetail — surface via usage_guidance
        "usage_guidance": row.get("usage_guidance") or "",
        "conditions": row.get("conditions") or [],
    }


def get_ingredient_by_name(name: str) -> IngredientDetail | None:
    rows = run_query(
        """
        SELECT *
        FROM skincare_advisor.ingredients
        WHERE inci_name ILIKE %s
           OR %s = ANY(common_names)
        LIMIT 1
        """,
        (name, name.lower().strip()),
    )
    return _format_ingredient(rows[0]) if rows else None


def get_ingredients_by_names(names: list[str]) -> list[IngredientDetail]:
    return [ing for name in names for ing in [get_ingredient_by_name(name)] if ing]


def search_ingredients(query: str, limit: int = 10) -> list[IngredientDetail]:
    term = f"%{query}%"
    normalized_query = query.lower().strip()
    rows = run_query(
        """
        SELECT *
        FROM skincare_advisor.ingredients
        WHERE inci_name ILIKE %s
           OR %s = ANY(common_names)
        ORDER BY inci_name
        LIMIT %s
        """,
        (term, normalized_query, limit),
    )
    return [_format_ingredient(row) for row in rows or []]


def _clean_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def save_external_ingredient(ingredient: dict) -> bool:
    """Persist externally researched ingredient details in the catalog."""
    inci_name = str(
        ingredient.get("inci_name")
        or ingredient.get("display_name")
        or ingredient.get("name")
        or ""
    ).strip().lower()
    if not inci_name:
        return False

    return run_mutation(
        """
        INSERT INTO skincare_advisor.ingredients (
            inci_name, common_names, functions, suitable_skin_types,
            avoid_skin_types, conditions, known_conflicts,
            known_conflicts_detail, safety_notes, usage_guidance,
            confidence, sources, normalized_name
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (inci_name) DO UPDATE SET
            common_names = CASE
                WHEN array_length(EXCLUDED.common_names, 1) > 0 THEN EXCLUDED.common_names
                ELSE ingredients.common_names
            END,
            functions = CASE
                WHEN array_length(EXCLUDED.functions, 1) > 0 THEN EXCLUDED.functions
                ELSE ingredients.functions
            END,
            suitable_skin_types = CASE
                WHEN array_length(EXCLUDED.suitable_skin_types, 1) > 0 THEN EXCLUDED.suitable_skin_types
                ELSE ingredients.suitable_skin_types
            END,
            avoid_skin_types = CASE
                WHEN array_length(EXCLUDED.avoid_skin_types, 1) > 0 THEN EXCLUDED.avoid_skin_types
                ELSE ingredients.avoid_skin_types
            END,
            conditions = CASE
                WHEN array_length(EXCLUDED.conditions, 1) > 0 THEN EXCLUDED.conditions
                ELSE ingredients.conditions
            END,
            known_conflicts = CASE
                WHEN array_length(EXCLUDED.known_conflicts, 1) > 0 THEN EXCLUDED.known_conflicts
                ELSE ingredients.known_conflicts
            END,
            known_conflicts_detail = CASE
                WHEN jsonb_typeof(EXCLUDED.known_conflicts_detail) = 'array'
                 AND jsonb_array_length(EXCLUDED.known_conflicts_detail) > 0
                THEN EXCLUDED.known_conflicts_detail
                ELSE ingredients.known_conflicts_detail
            END,
            safety_notes = COALESCE(NULLIF(EXCLUDED.safety_notes, ''), ingredients.safety_notes),
            usage_guidance = COALESCE(NULLIF(EXCLUDED.usage_guidance, ''), ingredients.usage_guidance),
            confidence = GREATEST(COALESCE(ingredients.confidence, 0), COALESCE(EXCLUDED.confidence, 0)),
            sources = CASE
                WHEN jsonb_typeof(EXCLUDED.sources) = 'array'
                 AND jsonb_array_length(EXCLUDED.sources) > 0
                THEN EXCLUDED.sources
                ELSE ingredients.sources
            END,
            normalized_name = COALESCE(NULLIF(EXCLUDED.normalized_name, ''), ingredients.normalized_name)
        """,
        (
            inci_name,
            _clean_list(ingredient.get("common_names")),
            _clean_list(ingredient.get("functions")),
            _clean_list(ingredient.get("suitable_for") or ingredient.get("suitable_skin_types")),
            _clean_list(ingredient.get("avoid_for") or ingredient.get("avoid_skin_types")),
            _clean_list(ingredient.get("conditions")),
            _clean_list(ingredient.get("known_conflicts")),
            Json(ingredient.get("known_conflicts_detail") or []),
            str(ingredient.get("safety_notes") or ""),
            str(ingredient.get("usage_guidance") or ""),
            int(ingredient.get("confidence") or 0),
            Json(ingredient.get("sources") or []),
            str(ingredient.get("normalized_name") or inci_name),
        ),
    )
