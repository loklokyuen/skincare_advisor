from __future__ import annotations

import json
import logging

from schemas import IngredientDetail, Product
from services.ingredient_service import _format_ingredient
from services.openai_embeddings import embed_query
from services.product_service import _format_product
from utils.db import run_query

log = logging.getLogger(__name__)


def _vector_literal(values: list[float]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _search_products_by_embedding(embedding: list[float], limit: int) -> list[Product]:
    vector = _vector_literal(embedding)
    rows = run_query(
        """
        SELECT id, product_name, brand, ingredients_raw,
               1 - (embedding <=> %s::vector) AS similarity
        FROM skincare_advisor.products
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vector, vector, limit),
    )

    products = [_format_product(row) for row in rows or []]
    for product in products:
        product["source"] = "semantic_catalog"
    return products


def _search_ingredients_by_embedding(embedding: list[float], limit: int) -> list[IngredientDetail]:
    vector = _vector_literal(embedding)
    rows = run_query(
        """
        SELECT *,
               1 - (embedding <=> %s::vector) AS similarity
        FROM skincare_advisor.ingredients
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vector, vector, limit),
    )
    return [_format_ingredient(row) for row in rows or []]


def search_products_semantic(query: str, limit: int = 15) -> list[Product]:
    embedding = embed_query(query)
    return _search_products_by_embedding(embedding, limit) if embedding is not None else []


def search_ingredients_semantic(query: str, limit: int = 15) -> list[IngredientDetail]:
    embedding = embed_query(query)
    return _search_ingredients_by_embedding(embedding, limit) if embedding is not None else []


def retrieve_semantic_context(
    query: str,
    product_limit: int = 15,
    ingredient_limit: int = 15,
    include_products: bool = True,
) -> tuple[list[Product], list[IngredientDetail], list[float] | None]:
    """Returns (products, ingredients, embedding) so callers can reuse the embedding."""
    embedding = embed_query(query)
    if embedding is None:
        return [], [], None

    products = _search_products_by_embedding(embedding, product_limit) if include_products else []
    ingredients = _search_ingredients_by_embedding(embedding, ingredient_limit)
    return products, ingredients, embedding
