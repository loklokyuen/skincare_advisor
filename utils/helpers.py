import streamlit as st
from utils.ingredient_format import format_ingredient_name


def get_user_profile() -> dict:
    """Return the current user profile from session state."""
    return st.session_state.get("user_profile", {})


def profile_is_complete() -> bool:
    """True if the user has at least set their skin type."""
    profile = get_user_profile()
    return bool(profile.get("skin_type"))


def is_suitable_skin_type(suitable_for: str, skin_type: str) -> bool:
    """
    Return True if the given skin_type matches the suitable_for string.
    Handles 'All', comma-separated values, and partial matches.
    Example: is_suitable_skin_type("All, esp. Oily", "Oily") -> True
    """
    if not skin_type or not suitable_for:
        return True
    suitable_lower = suitable_for.lower()
    if "all" in suitable_lower:
        return True
    return skin_type.lower() in suitable_lower


def concern_to_ingredients(concern: str) -> list[str]:
    """
    Return a list of ingredient names commonly associated with a skin concern.
    Used to surface relevant ingredients when a user has set concerns.
    """
    concern_map = {
        "Acne / Breakouts": ["Salicylic Acid", "Benzoyl Peroxide", "Niacinamide", "Tea Tree"],
        "Hyperpigmentation": ["Vitamin C", "Niacinamide", "Kojic Acid", "Alpha Arbutin", "AHA"],
        "Anti-aging / Wrinkles": ["Retinol", "Peptides", "Vitamin C", "Hyaluronic Acid"],
        "Dryness / Dehydration": ["Hyaluronic Acid", "Glycerin", "Ceramides", "Squalane"],
        "Redness / Rosacea": ["Centella Asiatica", "Azelaic Acid", "Niacinamide", "Ceramides"],
        "Large Pores": ["Niacinamide", "Salicylic Acid", "Retinol"],
        "Uneven Skin Tone": ["Vitamin C", "Alpha Arbutin", "Niacinamide", "AHA"],
        "Dark Circles": ["Caffeine", "Vitamin C", "Retinol", "Peptides"],
        "Sensitivity / Irritation": ["Ceramides", "Centella Asiatica", "Aloe Vera", "Oat Extract"],
        "Oiliness / Shine": ["Niacinamide", "Salicylic Acid", "Zinc", "Clay"],
        "Texture / Roughness": ["AHA", "BHA", "Retinol", "Vitamin C"],
        "Sun Damage": ["Vitamin C", "Niacinamide", "Retinol", "Alpha Arbutin"],
    }
    return concern_map.get(concern, [])


def get_recommended_ingredients_for_profile(profile: dict) -> list[str]:
    """
    Return a deduplicated list of recommended ingredient names
    based on the user's concerns list.
    """
    concerns = profile.get("concerns", [])
    seen = set()
    result = []
    for concern in concerns:
        for ing in concern_to_ingredients(concern):
            if ing not in seen:
                seen.add(ing)
                result.append(ing)
    return result


def skin_type_description(skin_type: str) -> str:
    """Return a short description of a skin type for display."""
    descriptions = {
        "Normal": "Balanced — not too oily or dry, minimal sensitivity",
        "Oily": "Excess sebum, enlarged pores, prone to shine and breakouts",
        "Dry": "Tight, flaky, or rough texture — lacks moisture and lipids",
        "Combination": "Oily T-zone (forehead, nose, chin), drier cheeks",
        "Sensitive": "Reacts easily to products, prone to redness and irritation",
    }
    return descriptions.get(skin_type, "")


def skin_type_badge(skin_type: str) -> str:
    """Return an emoji badge for a skin type."""
    badges = {
        "Normal": "🟢",
        "Oily": "💧",
        "Dry": "🏜️",
        "Combination": "⚖️",
        "Sensitive": "🌸",
    }
    return badges.get(skin_type, "❓")


def ingredient_safety_colour(safety_score: float) -> str:
    """
    Return a hex colour for a normalised safety score (0–1).
    Green = safe, Yellow = moderate, Red = concern.
    """
    if safety_score >= 0.7:
        return "#4CAF50"
    elif safety_score >= 0.4:
        return "#FFC107"
    else:
        return "#F44336"


def paginate(items: list, page: int, per_page: int = 20) -> list:
    """Slice a list for pagination."""
    start = (page - 1) * per_page
    return items[start : start + per_page]


def highlight_concerns(concerns: list[str]) -> str:
    """Render a list of skin concerns as coloured tags (HTML)."""
    tags = "".join(
        f'<span style="background:#fde8f0;color:#c2185b;'
        f'padding:2px 8px;border-radius:12px;margin:2px;'
        f'font-size:0.8rem">{c}</span>'
        for c in concerns
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px">{tags}</div>'


def format_ingredient_list(ingredients: list[str], highlight: list[str] = None) -> str:
    """
    Format an ingredient list as an HTML string,
    optionally highlighting specific ingredients.
    """
    highlight_lower = {h.lower() for h in (highlight or [])}
    parts = []
    for ing in ingredients:
        display = format_ingredient_name(ing)
        if ing.lower() in highlight_lower:
            parts.append(
                f'<mark style="background:#fff176;padding:0 3px">{display}</mark>'
            )
        else:
            parts.append(display)
    return ", ".join(parts)
