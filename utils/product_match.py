from __future__ import annotations

import re

_SIZE_PATTERN = re.compile(
    r"\b(?:"
    r"\d+(?:\.\d+)?\s*(?:ml|l|g|kg|oz|fl\s*oz|count|ct|pcs?|pack|wipes?|pads?|capsules?|tablets?)"
    r"|(?:30|60|100|150|200|250|300|500)\s*$"
    r")\b",
    flags=re.IGNORECASE,
)


def normalise_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def product_family_name(value: str) -> str:
    cleaned = re.sub(r"\s*[-–—]\s*", " - ", value)
    cleaned = _SIZE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s*-\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return normalise_match_text(cleaned)


def product_identity_key(product: dict, *, family: bool = False) -> tuple[str, str]:
    name = str(product.get("product_name") or "")
    return (
        product_family_name(name) if family else normalise_match_text(name),
        normalise_match_text(str(product.get("brand") or "")),
    )
