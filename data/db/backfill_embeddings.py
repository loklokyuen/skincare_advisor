from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "data" / ".env", override=False)
load_dotenv(ROOT / ".env", override=True)

from services.openai_embeddings import embed_documents  # noqa: E402

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "skincare_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def _connect():
    return psycopg2.connect(
        **DB_CONFIG,
        cursor_factory=psycopg2.extras.RealDictCursor,
        keepalives=1,
        keepalives_idle=60,
        keepalives_interval=10,
        keepalives_count=5,
    )


def _join(values: object, limit: int | None = None) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, Iterable):
        items = [str(item).strip() for item in values if str(item).strip()]
        if limit is not None:
            items = items[:limit]
        return ", ".join(items)
    return str(values)


def _product_text(row: dict) -> str:
    return "\n".join(
        part
        for part in (
            f"Product: {row.get('product_name') or ''}",
            f"Brand: {row.get('brand') or ''}",
            f"Categories: {_join(row.get('categories'), limit=20)}",
            f"Ingredients: {_join(row.get('ingredients_raw'), limit=None)}",
        )
        if part.split(": ", 1)[-1].strip()
    )


def _ingredient_text(row: dict) -> str:
    return "\n".join(
        part
        for part in (
            f"Ingredient: {row.get('inci_name') or ''}",
            f"Aliases: {_join(row.get('common_names'), limit=20)}",
            f"Functions: {_join(row.get('functions'), limit=20)}",
            f"Suitable skin types: {_join(row.get('suitable_skin_types'), limit=20)}",
            f"Avoid skin types: {_join(row.get('avoid_skin_types'), limit=20)}",
            f"Conditions: {_join(row.get('conditions'), limit=20)}",
            f"Known conflicts: {_join(row.get('known_conflicts'), limit=20)}",
            f"Safety notes: {row.get('safety_notes') or ''}",
            f"Usage guidance: {row.get('usage_guidance') or ''}",
        )
        if part.split(": ", 1)[-1].strip()
    )


def _fetch_products(conn, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, product_name, brand, categories, ingredients_raw
            FROM skincare_advisor.products
            WHERE embedding IS NULL
            ORDER BY id
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def _fetch_ingredients(conn, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, inci_name, common_names, functions, suitable_skin_types,
                   avoid_skin_types, conditions, known_conflicts, safety_notes,
                   usage_guidance
            FROM skincare_advisor.ingredients
            WHERE embedding IS NULL
            ORDER BY id
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def _embed_batch(texts: list[str]) -> list[str]:
    vectors = embed_documents(texts)
    if len(vectors) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
    return [json.dumps(vector, separators=(",", ":")) for vector in vectors]


def _update_embeddings(conn, table: str, rows: list[tuple[int, str]]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            UPDATE skincare_advisor.{table} AS target
            SET embedding = data.embedding::vector
            FROM (VALUES %s) AS data(id, embedding)
            WHERE target.id = data.id
            """,
            rows,
            template="(%s, %s)",
        )
    conn.commit()


def _backfill_table(conn, kind: str, batch_size: int, limit: int) -> int:
    total = 0
    remaining = limit
    while remaining > 0:
        page_size = min(batch_size, remaining)
        rows = _fetch_products(conn, page_size) if kind == "products" else _fetch_ingredients(conn, page_size)
        if not rows:
            break

        texts = [_product_text(row) if kind == "products" else _ingredient_text(row) for row in rows]
        vectors = _embed_batch(texts)
        _update_embeddings(conn, kind, [(row["id"], vector) for row, vector in zip(rows, vectors)])

        total += len(rows)
        remaining -= len(rows)
        print(f"{kind}: embedded {total}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pgvector embeddings in Cloud SQL.")
    parser.add_argument("--kind", choices=["products", "ingredients", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()

    with _connect() as conn:
        if args.kind in {"products", "all"}:
            _backfill_table(conn, "products", args.batch_size, args.limit)
        if args.kind in {"ingredients", "all"}:
            _backfill_table(conn, "ingredients", args.batch_size, args.limit)


if __name__ == "__main__":
    main()
