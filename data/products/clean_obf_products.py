#!/usr/bin/env python3
"""
Clean Open Beauty Facts TSV exports into Cloud-SQL-ready CSV and JSONL files.

The output is intentionally skincare-focused for the Skincare Advisor MVP:
- keeps products that are useful for search + ingredient analysis
- drops rows with no parsed ingredients
- drops obvious test/demo rows
- drops non-skincare categories such as haircare, makeup, perfume, oral care, etc.

Usage:
  python clean_obf_products.py \
    --input en.openbeautyfacts.org.products.tsv \
    --csv products_clean.csv \
    --jsonl products_clean.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

OUTPUT_COLUMNS = [
    "obf_code",
    "product_name",
    "display_name",
    "brand",
    "quantity",
    "image_url",
    "ingredients_raw",
    "ingredients_parsed_json",
    "categories_json",
    "country",
    "is_named",
    "is_analyzable",
    "ingredient_count",
    "search_score",
    "search_quality",
    "source",
]

PREFERRED_CATEGORY_FIELD = "categories_en"
SOURCE_NAME = "obf_dump"

# Broad allow-list. This keeps skincare, sun care, body care, lip care, and acne-related products.
SKINCARE_INCLUDE_KEYWORDS = {
    "skin", "skincare", "skin care", "face", "facial", "body", "hand", "hands",
    "foot", "feet", "moisturizer", "moisturiser", "moisturizing", "moisturising",
    "cream", "creams", "lotion", "lotions", "serum", "serums", "cleanser", "cleansers",
    "cleansing", "wash", "toner", "toners", "essence", "essences", "ampoule", "ampoules",
    "exfoliant", "exfoliants", "exfoliator", "exfoliators", "peel", "peels", "mask", "masks",
    "sunscreen", "sun screen", "sun care", "spf", "uv", "after sun", "aftersun",
    "balm", "balms", "lip balm", "lip care", "eye cream", "acne", "blemish", "spot treatment",
    "retinol", "retinoid", "niacinamide", "azelaic", "hyaluronic", "vitamin c", "aqua cream",
}

# Things that may appear in Open Beauty Facts cosmetics data but are outside the MVP scope.
NON_SKINCARE_EXCLUDE_KEYWORDS = {
    # Haircare
    "hair", "shampoo", "shampoos", "conditioner", "conditioners", "hair mask", "hair masks",
    "hair dye", "hair color", "hair colour", "hairspray", "hair spray", "scalp", "styling gel",
    # Makeup / nails
    "makeup", "make-up", "mascara", "foundation", "concealer", "blush", "bronzer", "highlighter",
    "eyeshadow", "eye shadow", "eyeliner", "lipstick", "lip gloss", "lip liner", "powder", "primer",
    "nail", "nails", "varnish", "polish", "manicure", "pedicure",
    # Fragrance / deodorant / hygiene / oral care / household-ish items
    "perfume", "fragrance", "eau de parfum", "eau de toilette", "cologne", "deodorant", "anti-perspirant",
    "antiperspirant", "toothpaste", "mouthwash", "mouth wash", "soap", "hand soap", "sanitizer", "sanitiser",
    "disinfectant", "wipes", "tissues", "mouchoirs", "cotton pads", "cotton buds",
    # Baby or miscellaneous personal care outside skincare routine analysis
    "diaper", "nappy", "razor", "shaving", "depilatory", "wax strips",
}

TEST_PRODUCT_PATTERNS = [
    re.compile(r"\btest\b", re.IGNORECASE),
    re.compile(r"\bdemo\b", re.IGNORECASE),
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"\bproduct\s+for\s+parsing\b", re.IGNORECASE),
    re.compile(r"\bfor\s+parsing\s+ingredients\b", re.IGNORECASE),
]

NOISE_INGREDIENT_PATTERNS = [
    re.compile(r"\btest\b", re.IGNORECASE),
    re.compile(r"\bexample\b", re.IGNORECASE),
    re.compile(r"\bdemo\b", re.IGNORECASE),
]


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).replace("\x00", "").strip()
    value = " ".join(value.split())
    return value or None


def normalize_for_keyword(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^a-z0-9+%./ -]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value




def _compile_keyword_regex(keywords: Sequence[str]) -> re.Pattern[str]:
    escaped = []
    for keyword in keywords:
        key = normalize_for_keyword(keyword)
        if key:
            escaped.append(re.escape(key).replace(r"\ ", r"\s+"))
    # Match at non-alphanumeric boundaries so "test" does not match "contest".
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(sorted(escaped, key=len, reverse=True)) + r")(?![a-z0-9])", re.IGNORECASE)

INCLUDE_REGEX = _compile_keyword_regex(tuple(SKINCARE_INCLUDE_KEYWORDS))
EXCLUDE_REGEX = _compile_keyword_regex(tuple(NON_SKINCARE_EXCLUDE_KEYWORDS))

def contains_keyword(text: str, keyword_regex: re.Pattern[str]) -> bool:
    normalized = normalize_for_keyword(text)
    return bool(keyword_regex.search(normalized))


def split_multi_value(value: Optional[str]) -> List[str]:
    value = clean_text(value)
    if not value:
        return []

    parts = [clean_text(part) for part in value.split(",")]
    seen = set()
    result = []
    for part in parts:
        if not part:
            continue
        # OBf tag fields can include values such as fr:Shampoings.
        part = re.sub(r"^[a-z]{2}:", "", part, flags=re.IGNORECASE).strip()
        part = clean_text(part)
        if not part:
            continue
        key = part.casefold()
        if key not in seen:
            seen.add(key)
            result.append(part)
    return result


def split_ingredients(raw: str) -> List[str]:
    """Split ingredient text on separators while avoiding commas inside brackets."""
    text = raw.replace("•", ",").replace(";", ",").replace("\n", ",")
    parts: List[str] = []
    current: List[str] = []
    depth = 0

    for char in text:
        if char in "([{" :
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)

        if char == "," and depth == 0:
            part = clean_text("".join(current))
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)

    final_part = clean_text("".join(current))
    if final_part:
        parts.append(final_part)
    return parts


def normalize_ingredient_name(name: str) -> str:
    normalized = name.casefold()
    normalized = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", " ", normalized)
    normalized = re.sub(r"[\(\[\{].*$", " ", normalized)
    normalized = re.sub(r"\bmay contain\b.*$", " ", normalized)
    normalized = re.sub(r"\bcontains?\b:?", " ", normalized)
    normalized = re.sub(r"[^a-z0-9%+./' -]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .,-_/+")
    return normalized


def parse_ingredients(raw: Optional[str]) -> List[Dict[str, object]]:
    raw = clean_text(raw)
    if not raw:
        return []

    candidates = split_ingredients(raw)
    parsed = []
    seen = set()
    for idx, item in enumerate(candidates, start=1):
        name = clean_text(item.strip(" \t\r\n.,;:()[]{}*_"))
        if not name:
            continue
        normalized = normalize_ingredient_name(name)
        if not normalized:
            continue
        # Reject obvious non-ingredient noise from test/demo rows.
        if any(pattern.search(normalized) for pattern in NOISE_INGREDIENT_PATTERNS):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        parsed.append({"name": name, "normalized": normalized, "position": idx})
    return parsed


def choose_display_name(row: Dict[str, str], categories: List[str]) -> str:
    product_name = clean_text(row.get("product_name"))
    generic_name = clean_text(row.get("generic_name"))
    brand = clean_text(row.get("brands"))
    code = clean_text(row.get("code"))

    base = product_name or generic_name
    if base and brand and brand.casefold() not in base.casefold():
        return f"{brand} — {base}"
    if base:
        return base
    if brand and categories:
        return f"{brand} — {categories[-1]}"
    if brand:
        return brand
    if categories:
        return f"Unnamed product — {categories[-1]}"
    return f"Unnamed product — {code or 'unknown code'}"


def is_test_product(*values: Optional[str]) -> bool:
    joined = " ".join(v for v in values if v)
    if not joined:
        return False
    normalized = normalize_for_keyword(joined)
    # Avoid dropping real names merely containing "test" as part of a word.
    return any(pattern.search(normalized) for pattern in TEST_PRODUCT_PATTERNS)


def is_skincare_product(
    product_name: Optional[str],
    brand: Optional[str],
    categories: List[str],
    ingredients_raw: Optional[str],
) -> bool:
    """Return True only for products likely relevant to a skincare routine.

    Categories are trusted more than names. If a product has explicit non-skincare
    categories such as shampoo/makeup/perfume, it is dropped even if the name has
    generic words like "cream".
    """
    category_text = " ".join(categories)
    searchable_text = " ".join(v for v in [product_name, brand, category_text] if v)

    if contains_keyword(category_text, EXCLUDE_REGEX):
        return False

    # If categories are missing/weak, use product text as a fallback, but still
    # drop obvious non-skincare product names.
    if contains_keyword(searchable_text, EXCLUDE_REGEX):
        return False

    if contains_keyword(searchable_text, INCLUDE_REGEX):
        return True

    # Last-resort fallback: ingredient lists with common skincare actives can be kept.
    # This prevents losing useful rows with sparse categories.
    if ingredients_raw and contains_keyword(ingredients_raw, INCLUDE_REGEX):
        return True

    return False


def calculate_search_score(
    product_name: Optional[str],
    brand: Optional[str],
    code: Optional[str],
    ingredients_parsed: List[Dict[str, object]],
    categories: List[str],
) -> int:
    score = 0

    if product_name and len(product_name.strip()) >= 4:
        score += 4
    elif product_name:
        score += 1

    if brand:
        score += 2
    if code:
        score += 1

    ingredient_count = len(ingredients_parsed)
    if ingredient_count >= 15:
        score += 4
    elif ingredient_count >= 8:
        score += 3
    elif ingredient_count >= 3:
        score += 2
    elif ingredient_count > 0:
        score += 1

    if categories:
        score += 1

    return score


def classify_search_quality(search_score: int, ingredient_count: int, is_named: bool) -> str:
    # Keep the label useful for ranking: unnamed or thin ingredient rows should not be "high".
    if search_score >= 10 and ingredient_count >= 8 and is_named:
        return "high"
    if search_score >= 6 and ingredient_count >= 3:
        return "medium"
    return "low"


def should_keep_row(
    product_name: Optional[str],
    brand: Optional[str],
    code: Optional[str],
    ingredients_raw: Optional[str],
    ingredients_parsed: List[Dict[str, object]],
    categories: List[str],
    image_url: Optional[str],
    quantity: Optional[str],
) -> bool:
    # Must be analyzable.
    if not ingredients_raw or len(ingredients_parsed) == 0:
        return False

    # Must be findable/confirmable somehow.
    if not any([product_name, brand, code, categories, image_url, quantity]):
        return False

    # Remove test/demo rows.
    if is_test_product(product_name, brand, ingredients_raw, " ".join(categories)):
        return False

    # Keep only products relevant to a skincare routine.
    if not is_skincare_product(product_name, brand, categories, ingredients_raw):
        return False

    return True


def clean_row(row: Dict[str, str]) -> Optional[Dict[str, str]]:
    code = clean_text(row.get("code"))
    if not code:
        return None

    product_name = clean_text(row.get("product_name"))
    brand = clean_text(row.get("brands"))
    quantity = clean_text(row.get("quantity"))
    image_url = clean_text(row.get("image_url"))
    ingredients_raw = clean_text(row.get("ingredients_text"))

    category_source = row.get(PREFERRED_CATEGORY_FIELD) or row.get("categories")
    categories = split_multi_value(category_source)
    country = clean_text(row.get("countries_en"))
    display_name = choose_display_name(row, categories)
    ingredients_parsed = parse_ingredients(ingredients_raw)

    if not should_keep_row(
        product_name=product_name,
        brand=brand,
        code=code,
        ingredients_raw=ingredients_raw,
        ingredients_parsed=ingredients_parsed,
        categories=categories,
        image_url=image_url,
        quantity=quantity,
    ):
        return None

    is_named = bool(product_name)
    ingredient_count = len(ingredients_parsed)
    search_score = calculate_search_score(product_name, brand, code, ingredients_parsed, categories)
    search_quality = classify_search_quality(search_score, ingredient_count, is_named)

    return {
        "obf_code": code,
        "product_name": product_name or "",
        "display_name": display_name,
        "brand": brand or "",
        "quantity": quantity or "",
        "image_url": image_url or "",
        "ingredients_raw": ingredients_raw or "",
        "ingredients_parsed_json": json.dumps(ingredients_parsed, ensure_ascii=False),
        "categories_json": json.dumps(categories, ensure_ascii=False),
        "country": country or "",
        "is_named": str(is_named).lower(),
        "is_analyzable": "true",
        "ingredient_count": ingredient_count,
        "search_score": search_score,
        "search_quality": search_quality,
        "source": SOURCE_NAME,
    }


def iter_clean_rows(input_path: Path) -> Iterable[Dict[str, str]]:
    seen_codes = set()
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        header_line = f.readline().rstrip("\n\r")
        headers = header_line.split("\t")
        for line in f:
            values = line.rstrip("\n\r").split("\t")
            if len(values) < len(headers):
                values.extend([""] * (len(headers) - len(values)))
            row = dict(zip(headers, values))
            cleaned = clean_row(row)
            if not cleaned:
                continue
            code = cleaned["obf_code"]
            if code in seen_codes:
                continue
            seen_codes.add(code)
            yield cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Open Beauty Facts TSV file")
    parser.add_argument("--csv", default="products_clean.csv", help="Output CSV path")
    parser.add_argument("--jsonl", default="products_clean.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for testing/sample exports")
    args = parser.parse_args()

    input_path = Path(args.input)
    csv_path = Path(args.csv)
    jsonl_path = Path(args.jsonl)

    total = 0
    with csv_path.open("w", encoding="utf-8", newline="") as csv_f, jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        writer = csv.DictWriter(csv_f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for cleaned in iter_clean_rows(input_path):
            writer.writerow(cleaned)
            jsonl_f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            total += 1
            if args.limit is not None and total >= args.limit:
                break

    print(f"Wrote {total:,} cleaned products")
    print(f"CSV:   {csv_path}")
    print(f"JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
