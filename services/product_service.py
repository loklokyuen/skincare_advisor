from __future__ import annotations

import logging
import json
import re
import time
from typing import TYPE_CHECKING

from psycopg2.extras import Json

from services.openai_embeddings import embed_query
from utils.db import run_mutation, run_query
from utils.ingredient_format import format_ingredient_name, format_ingredient_names
from utils.product_match import product_identity_key

if TYPE_CHECKING:
    from schemas import Product, ProductSearchResult

log = logging.getLogger(__name__)

# ── Skincare product guard ─────────────────────────────────────────────────────

# OBF category tags that confirm a cosmetic/skincare product
_COSMETIC_TAGS: frozenset[str] = frozenset({
    "en:face-care-products", "en:face-care", "en:skin-care", "en:skincare",
    "en:cosmetics", "en:beauty", "en:face-creams", "en:moisturisers",
    "en:serums", "en:cleansers", "en:toners", "en:sunscreens",
    "en:face-masks", "en:facial-scrubs", "en:eye-care", "en:lip-care",
    "en:body-care", "en:shower-gels", "en:shampoos", "en:hair-care",
})

# OBF category tags that disqualify a product
_NON_COSMETIC_TAGS: frozenset[str] = frozenset({
    "en:dietary-supplements", "en:food", "en:beverages", "en:drinks",
    "en:vitamins", "en:minerals", "en:tablets", "en:capsules",
    "en:drops", "en:syrups", "en:lozenges", "en:confectionery",
    "en:plant-based-foods", "en:grocery",
})

# Name-level keywords that flag non-skincare items regardless of category
_NON_COSMETIC_NAME_RE = re.compile(
    r"\b("
    r"tablet|capsule|lozenge|gummie?s?|gummy|chewable|softgel|"
    r"supplement|multivitamin|probiotic|prebiotic|omega|collagen\s+shot|"
    r"energy\s+(drink|bar|gel|chew)|electrolyte|protein\s+(powder|bar|shake)|"
    r"(?:vitamin\s+[a-z]\d*\s+(?:drops?|tablets?|capsules?|gummies?))|"
    r"\d+\s*(?:tablets?|capsules?|softgels?|lozenges?)\b"
    r")",
    re.IGNORECASE,
)


def _is_skincare_product(name: str, categories_tags: list[str]) -> bool:
    """
    Return True if this OBF product is likely a skincare/cosmetic item.
    Logic:
      1. Reject outright if the name contains supplement/tablet keywords.
      2. Reject if any non-cosmetic OBF category tag is present.
      3. Accept if at least one cosmetic OBF category tag is present.
      4. Accept otherwise (OBF is a beauty DB — default is probably fine).
    """
    if _NON_COSMETIC_NAME_RE.search(name):
        return False
    tags = set(categories_tags or [])
    if tags & _NON_COSMETIC_TAGS:
        return False
    if tags & _COSMETIC_TAGS:
        return True
    # No positive cosmetic tag but also no disqualifying tag — allow through;
    # OBF is beauty-focused so untagged products are usually cosmetics.
    return True

# ── Ingredient alias cache ─────────────────────────────────────────────────────
_alias_cache: dict[str, str] | None = None
_alias_cache_ts: float = 0.0
_ALIAS_CACHE_TTL = 3600.0  # refresh every hour


def _get_alias_map() -> dict[str, str]:
    """Return {normalized_alias: inci_name} for all ingredients in the DB."""
    global _alias_cache, _alias_cache_ts
    now = time.monotonic()
    if _alias_cache is not None and now - _alias_cache_ts < _ALIAS_CACHE_TTL:
        return _alias_cache
    rows = run_query("SELECT inci_name, common_names FROM skincare_advisor.ingredients")
    cache: dict[str, str] = {}
    for row in rows or []:
        inci = row["inci_name"]
        cache[inci.lower().strip()] = inci
        for alias in row.get("common_names") or []:
            if alias:
                cache[alias.lower().strip()] = inci
    _alias_cache = cache
    _alias_cache_ts = now
    return cache


def identify_key_ingredients(raw_names: list[str]) -> list[str]:
    """Return only ingredient names that match a known skincare active."""
    alias_map = _get_alias_map()
    return [
        format_ingredient_name(name)
        for name in raw_names
        if name.lower().strip() in alias_map
    ]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _parse_raw_ingredients(row: dict, max_count: int = 50) -> list[str]:
    raw = (row.get("ingredients_raw") or "").strip()
    if not raw:
        return []
    return format_ingredient_names(
        [p.strip() for p in re.split(r",\s*(?!\d)", raw) if p.strip()][:max_count]
    )


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _active_ingredients(row: dict) -> list[str]:
    return format_ingredient_names(
        [str(item).strip() for item in _as_list(row.get("active_ingredients")) if str(item).strip()]
    )


def _category_list(row: dict) -> list[str]:
    return [str(item).strip() for item in _as_list(row.get("categories")) if str(item).strip()]


def _key_ingredients_for_row(row: dict) -> list[str]:
    raw_ingredients = _parse_raw_ingredients(row)
    key_ingredients = identify_key_ingredients(raw_ingredients)
    seen = {item.lower().strip() for item in key_ingredients}
    for item in _active_ingredients(row):
        key = item.lower().strip()
        if key and key not in seen:
            key_ingredients.append(item)
            seen.add(key)
    return key_ingredients


def _format_search_result(row: dict) -> ProductSearchResult:
    all_ingredients = _parse_raw_ingredients(row, max_count=200)
    active_ingredients = _active_ingredients(row)
    return {
        "product_name": row.get("product_name", ""),
        "brand": row.get("brand") or "",
        "quantity": row.get("quantity") or "",
        "image_url": row.get("image_url") or "",
        "key_ingredients": _key_ingredients_for_row(row),
        "active_ingredients": active_ingredients,
        "ingredients": all_ingredients,
        "ingredients_raw": row.get("ingredients_raw") or "",
        "categories": _category_list(row),
        "source": row.get("source") or "db",
    }


def _dedupe_search_results(items: list[ProductSearchResult]) -> list[ProductSearchResult]:
    seen = set()
    deduped: list[ProductSearchResult] = []
    for item in items:
        key = product_identity_key(item, family=True)
        if key in seen or not key[0]:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_product(row: dict) -> Product:
    return {
        "product_id": row.get("id") or 0,
        "product_name": row.get("product_name", ""),
        "brand": row.get("brand") or "",
        "quantity": row.get("quantity") or "",
        "image_url": row.get("image_url") or "",
        "ingredients": _parse_raw_ingredients(row, max_count=200),
        "key_ingredients": _key_ingredients_for_row(row),
        "active_ingredients": _active_ingredients(row),
        "categories": _category_list(row),
        "source": "db",
    }


def _normalise_external_product(product: dict) -> dict:
    raw_ingredients = (
        product.get("ingredients_raw")
        or product.get("ingredients_text")
        or ", ".join(product.get("ingredients") or [])
    )
    categories = product.get("categories") or product.get("categories_tags") or []
    brand = product.get("brand") or product.get("brands") or ""
    if isinstance(brand, str) and "," in brand:
        brand = brand.split(",")[0].strip()

    active_ingredients = product.get("active_ingredients") or product.get("key_ingredients") or []
    if not active_ingredients and raw_ingredients:
        active_ingredients = identify_key_ingredients(_parse_raw_ingredients({"ingredients_raw": raw_ingredients}))

    return {
        "external_product_id": product.get("external_product_id") or product.get("obf_code") or product.get("barcode") or product.get("code") or None,
        "product_name": product.get("product_name") or product.get("product_name_en") or product.get("name") or "",
        "brand": brand or "",
        "quantity": product.get("quantity") or "",
        "image_url": product.get("image_url") or "",
        "ingredients_raw": raw_ingredients or "",
        "categories": categories if isinstance(categories, list) else [],
        "country": product.get("country") or "Unknown",
        "source": product.get("source") or "open_beauty_facts",
        "active_ingredients": active_ingredients if isinstance(active_ingredients, list) else [],
    }


def _generate_product_embedding(row: dict) -> str | None:
    """Generate a pgvector-ready embedding string for a product row, or None on failure."""
    try:
        parts = []
        if row.get("product_name"):
            parts.append(f"Product: {row['product_name']}")
        if row.get("brand"):
            parts.append(f"Brand: {row['brand']}")
        cats = row.get("categories") or []
        if cats:
            parts.append(f"Categories: {', '.join(str(c) for c in cats[:20] if c)}")
        if row.get("ingredients_raw"):
            parts.append(f"Ingredients: {row['ingredients_raw']}")
        text = "\n".join(parts).strip()
        if not text:
            return None
        vector = embed_query(text)
        if vector is None:
            return None
        return json.dumps(vector, separators=(",", ":"))
    except Exception as exc:
        log.debug("Embedding generation skipped for %r: %s", row.get("product_name"), exc)
        return None


def save_external_product(product: dict) -> bool:
    """Persist an externally fetched product so future searches can use Cloud SQL."""
    row = _normalise_external_product(product)
    if not row["product_name"]:
        return False

    embedding = _generate_product_embedding(row)
    base_obf_params = (
        row["external_product_id"],
        row["product_name"],
        row["brand"],
        row["quantity"],
        row["image_url"],
        row["ingredients_raw"],
        Json(row["categories"]),
        row["country"],
        row["source"],
        Json(row["active_ingredients"]),
    )
    base_no_obf_params = (
        row["product_name"],
        row["brand"],
        row["quantity"],
        row["image_url"],
        row["ingredients_raw"],
        Json(row["categories"]),
        row["country"],
        row["source"],
        Json(row["active_ingredients"]),
    )
    emb_col = ", embedding" if embedding else ""
    emb_val = ", %s::vector" if embedding else ""
    emb_tuple = (embedding,) if embedding else ()

    if row["external_product_id"]:
        return run_mutation(
            f"""
            INSERT INTO skincare_advisor.products (
                external_product_id, product_name, brand, quantity, image_url,
                ingredients_raw, categories, country, last_synced_at, source,
                active_ingredients{emb_col}
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now(), %s, %s{emb_val})
            ON CONFLICT (external_product_id) DO UPDATE SET
                product_name = COALESCE(NULLIF(EXCLUDED.product_name, ''), products.product_name),
                brand = COALESCE(NULLIF(EXCLUDED.brand, ''), products.brand),
                quantity = COALESCE(NULLIF(EXCLUDED.quantity, ''), products.quantity),
                image_url = COALESCE(NULLIF(EXCLUDED.image_url, ''), products.image_url),
                ingredients_raw = COALESCE(NULLIF(EXCLUDED.ingredients_raw, ''), products.ingredients_raw),
                categories = CASE
                    WHEN jsonb_array_length(EXCLUDED.categories) > 0 THEN EXCLUDED.categories
                    ELSE products.categories
                END,
                country = COALESCE(NULLIF(EXCLUDED.country, ''), products.country),
                last_synced_at = now(),
                source = COALESCE(NULLIF(EXCLUDED.source, ''), products.source),
                active_ingredients = CASE
                    WHEN jsonb_array_length(EXCLUDED.active_ingredients) > 0 THEN EXCLUDED.active_ingredients
                    ELSE products.active_ingredients
                END,
                embedding = COALESCE(EXCLUDED.embedding, products.embedding)
            """,
            base_obf_params + emb_tuple,
        )

    existing = run_query(
        """
        SELECT id
        FROM skincare_advisor.products
        WHERE lower(product_name) = lower(%s)
          AND lower(COALESCE(brand, '')) = lower(%s)
        LIMIT 1
        """,
        (row["product_name"], row["brand"]),
    )
    if existing:
        return True

    return run_mutation(
        f"""
        INSERT INTO skincare_advisor.products (
            product_name, brand, quantity, image_url, ingredients_raw,
            categories, country, last_synced_at, source, active_ingredients{emb_col}
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, now(), %s, %s{emb_val})
        """,
        base_no_obf_params + emb_tuple,
    )


# ── OBF live fallback ──────────────────────────────────────────────────────────

def _search_obf(query: str, limit: int = 5) -> list[ProductSearchResult]:
    """
    Query the Open Beauty Facts API and return skincare-only results.
    Fetches more than needed so filtering doesn't leave us short.
    Called only when the local DB has too few matches.
    """
    try:
        from utils.api_client import search_products as obf_search
        raw = obf_search(query, page_size=limit * 3)  # fetch extra to absorb filter losses
        products = raw.get("products") or []
        results: list[ProductSearchResult] = []
        for p in products:
            name = (p.get("product_name") or "").strip()
            if not name:
                continue
            categories_tags = p.get("categories_tags") or []
            if not _is_skincare_product(name, categories_tags):
                log.debug("OBF filter excluded: %r", name)
                continue
            ingredients_text = p.get("ingredients_text") or ""
            raw_ings = [i.strip() for i in ingredients_text.split(",") if i.strip()]
            result = {
                "product_name": name,
                "brand": (p.get("brands") or "").split(",")[0].strip(),
                "quantity": p.get("quantity") or "",
                "image_url": p.get("image_url") or "",
                "key_ingredients": identify_key_ingredients(raw_ings[:60]),
                "active_ingredients": identify_key_ingredients(raw_ings[:60]),
                "ingredients": raw_ings,
                "ingredients_raw": ingredients_text,
                "categories": categories_tags,
                "source": "open_beauty_facts",
                "barcode": p.get("code") or "",
            }
            save_external_product(result)
            results.append(result)
            if len(results) >= limit:
                break
        return results
    except Exception as exc:
        log.warning("OBF fallback search failed: %s", exc)
        return []


# ── Public API ─────────────────────────────────────────────────────────────────

def search_products(
    query: str,
    limit: int = 20,
    use_obf_fallback: bool = True,
) -> list[ProductSearchResult]:
    """
    Full-text search over local product DB.
    Falls back to live Open Beauty Facts API when local results are sparse (< 3).
    Non-skincare items (supplements, tablets, food) are filtered out at both layers.
    """
    term = f"%{query}%"
    rows = run_query(
        """
        SELECT product_name, brand, quantity, image_url, ingredients_raw, categories,
               source, active_ingredients
        FROM skincare_advisor.products
        WHERE product_name ILIKE %s OR brand ILIKE %s
        ORDER BY product_name
        LIMIT %s
        """,
        (term, term, limit * 2),  # fetch extra so name-filter doesn't leave us short
    )
    results = [
        _format_search_result(r)
        for r in rows or []
        if not _NON_COSMETIC_NAME_RE.search(r.get("product_name") or "")
    ]
    results = results[:limit]

    if use_obf_fallback and len(results) < 3:
        log.debug("Local DB returned %d results for %r — querying OBF", len(results), query)
        obf = _search_obf(query, limit=limit - len(results))
        seen = {r["product_name"].lower() for r in results}
        for item in obf:
            if item["product_name"].lower() not in seen:
                results.append(item)
                seen.add(item["product_name"].lower())

    return results[:limit]


def search_products_semantic(
    query: str,
    limit: int = 20,
    embedding: list[float] | None = None,
) -> list[ProductSearchResult]:
    """
    Semantic product search over Cloud SQL pgvector embeddings.
    Best for intent-style searches like "gentle cleanser for dry sensitive skin".
    Pass a pre-computed embedding to avoid a redundant API call.
    """
    if embedding is None:
        embedding = embed_query(query)
    if embedding is None:
        return []

    vector = json.dumps(embedding, separators=(",", ":"))
    rows = run_query(
        """
        SELECT product_name, brand, quantity, image_url, ingredients_raw, categories,
               source, active_ingredients,
               1 - (embedding <=> %s::vector) AS similarity
        FROM skincare_advisor.products
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vector, vector, limit * 2),
    )
    results = [
        _format_search_result(r)
        for r in rows or []
        if not _NON_COSMETIC_NAME_RE.search(r.get("product_name") or "")
    ]
    return results[:limit]


def search_products_for_routine(
    query: str,
    limit: int = 20,
    embedding: list[float] | None = None,
) -> list[ProductSearchResult]:
    """
    Product search tuned for the Routine page.
    Exact name/brand matches stay first; semantic matches fill in for broader needs.
    Pass a pre-computed embedding to avoid a redundant API call.
    """
    cleaned = query.strip()
    if not cleaned:
        return []

    exact_limit = max(3, limit // 2)
    exact = search_products(cleaned, limit=exact_limit, use_obf_fallback=False)
    semantic = search_products_semantic(cleaned, limit=limit, embedding=embedding)
    return _dedupe_search_results(exact + semantic)[:limit]


def get_product_by_id(product_id: int) -> Product | None:
    """Fetch a single product by its integer PK."""
    rows = run_query(
        """
        SELECT id, product_name, brand, quantity, image_url, ingredients_raw,
               categories, source, active_ingredients
        FROM skincare_advisor.products
        WHERE id = %s
        """,
        (product_id,),
    )
    return _format_product(rows[0]) if rows else None


def get_products_by_ids(product_ids: list[int]) -> list[Product]:
    """Batch-fetch products by their integer PKs."""
    if not product_ids:
        return []
    rows = run_query(
        """
        SELECT id, product_name, brand, quantity, image_url, ingredients_raw,
               categories, source, active_ingredients
        FROM skincare_advisor.products
        WHERE id = ANY(%s)
        """,
        (product_ids,),
    )
    return [_format_product(r) for r in rows or []]
