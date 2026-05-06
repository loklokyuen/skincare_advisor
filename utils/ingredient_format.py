_INGREDIENT_ACRONYMS = {
    "aha",
    "bha",
    "bha/aha",
    "bht",
    "ci",
    "edta",
    "peg",
    "ppg",
    "spf",
    "uv",
}


def format_ingredient_name(ingredient: str) -> str:
    """Format ingredient names for display while preserving common INCI acronyms."""
    if not ingredient:
        return ""

    def format_part(part: str) -> str:
        if part.lower() in _INGREDIENT_ACRONYMS:
            return part.upper()
        return part[:1].upper() + part[1:].lower()

    words = []
    for word in str(ingredient).strip().split():
        normalized = word.strip()
        if not normalized:
            continue
        bare = normalized.strip("()[]{}.,;:")
        if bare.lower() in _INGREDIENT_ACRONYMS:
            words.append(normalized.replace(bare, bare.upper(), 1))
            continue
        words.append("-".join(format_part(part) for part in normalized.split("-")))
    return " ".join(words)


def format_ingredient_names(ingredients: list[str]) -> list[str]:
    """Format a list of ingredient names for display."""
    return [formatted for ing in ingredients if (formatted := format_ingredient_name(str(ing)))]
