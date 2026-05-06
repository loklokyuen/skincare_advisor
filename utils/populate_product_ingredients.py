import os
import re

import psycopg2
from psycopg2.extras import execute_values


SCHEMA = "skincare_advisor"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "34.147.213.190"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "skincare_advisor"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def normalize_ingredient(name: str) -> str:
    name = name.lower().strip()

    # remove text inside brackets, e.g. "glycerin (plant origin)" -> "glycerin"
    name = re.sub(r"\([^)]*\)", "", name)

    # remove label noise
    name = re.sub(r"\bingredients?\b\s*:?", "", name)

    # keep simple ingredient-safe characters
    name = re.sub(r"[^a-z0-9\s\/\-]", " ", name)

    # collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name


def split_ingredients(raw: str) -> list[str]:
    if not raw:
        return []

    parts = raw.split(",")

    cleaned = []
    for part in parts:
        part = part.strip()

        if len(part) < 2:
            continue

        cleaned.append(part)

    return cleaned


def main():
    print("Starting populate_product_ingredients...")

    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"TRUNCATE TABLE {SCHEMA}.product_ingredients RESTART IDENTITY;"
                )

                cur.execute(
                    f"""
                    SELECT id, ingredients_raw
                    FROM {SCHEMA}.products
                    WHERE ingredients_raw IS NOT NULL
                      AND trim(ingredients_raw) <> '';
                    """
                )

                products = cur.fetchall()
                print(f"Fetched {len(products)} products.")

                rows = []

                for product_id, ingredients_raw in products:
                    ingredients = split_ingredients(ingredients_raw)

                    for index, ingredient_raw in enumerate(ingredients, start=1):
                        normalized = normalize_ingredient(ingredient_raw)

                        if not normalized:
                            continue

                        rows.append(
                            (
                                product_id,
                                None,
                                ingredient_raw,
                                normalized,
                                index,
                                index <= 5,
                            )
                        )

                if not rows:
                    print("No ingredient rows generated.")
                    return

                execute_values(
                    cur,
                    f"""
                    INSERT INTO {SCHEMA}.product_ingredients
                    (
                        product_id,
                        ingredient_id,
                        ingredient_name_raw,
                        normalized_name,
                        position,
                        is_prominent
                    )
                    VALUES %s;
                    """,
                    rows,
                )

                print(f"Inserted {len(rows)} product ingredient rows.")

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

    finally:
        conn.close()


if __name__ == "__main__":
    main()