import logging

import requests

from config.settings import OPEN_BEAUTY_FACTS_BASE

log = logging.getLogger(__name__)


def search_products(query: str, page: int = 1, page_size: int = 20) -> dict:
    """
    Search Open Beauty Facts for products by name/keyword.
    Returns the raw API response dict, or empty dict on failure.
    """
    try:
        url = f"{OPEN_BEAUTY_FACTS_BASE}/search"
        params = {
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page": page,
            "page_size": page_size,
            "fields": (
                "code,product_name,brands,categories_tags,"
                "ingredients_text,image_url,labels_tags,"
                "countries_tags,quantity"
            ),
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.warning("OBF API request failed: %s", e)
        return {}


def get_product_by_barcode(barcode: str) -> dict | None:
    """
    Fetch a single product from Open Beauty Facts by barcode.
    Returns the product dict, or None if not found.
    """
    try:
        url = f"{OPEN_BEAUTY_FACTS_BASE}/product/{barcode}.json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == 1:
            return data.get("product")
        return None
    except requests.RequestException as e:
        log.warning("OBF barcode lookup failed: %s", e)
        return None


def parse_product(raw: dict) -> dict:
    """
    Normalise a raw Open Beauty Facts product dict into a
    consistent internal format.
    """
    ingredients_raw = raw.get("ingredients_text", "") or ""
    ingredients = [i.strip() for i in ingredients_raw.split(",") if i.strip()]

    return {
        "barcode": raw.get("code", ""),
        "name": raw.get("product_name", "Unknown"),
        "brand": raw.get("brands", "Unknown"),
        "category": raw.get("categories_tags", []),
        "ingredients": ingredients,
        "ingredient_count": len(ingredients),
        "image_url": raw.get("image_url", ""),
        "labels": raw.get("labels_tags", []),
        "countries": raw.get("countries_tags", []),
        "quantity": raw.get("quantity", ""),
    }


def format_labels(labels: list[str]) -> list[str]:
    """
    Decode Open Beauty Facts label tags into human-readable strings.
    e.g. 'en:vegan' → 'Vegan'
    """
    out = []
    for label in labels:
        parts = label.split(":")
        readable = parts[-1].replace("-", " ").title()
        out.append(readable)
    return out
