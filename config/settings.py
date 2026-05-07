import os
from dotenv import load_dotenv

load_dotenv()

APP_CONFIG = {
    "name": "SkinIQ",
    "version": "0.1.0",
    "description": "Skincare Intelligence Platform",
}

# Open Beauty Facts API
OPEN_BEAUTY_FACTS_BASE = "https://world.openbeautyfacts.org/api/v2"

# Skin types
SKIN_TYPES = ["Normal", "Oily", "Dry", "Combination", "Sensitive"]

# Skin concerns
SKIN_CONCERNS = [
    "Acne / Breakouts",
    "Hyperpigmentation",
    "Anti-aging / Wrinkles",
    "Dryness / Dehydration",
    "Redness / Rosacea",
    "Large Pores",
    "Uneven Skin Tone",
    "Dark Circles",
    "Sensitivity / Irritation",
    "Oiliness / Shine",
    "Texture / Roughness",
    "Sun Damage",
]

# Product categories
PRODUCT_CATEGORIES = [
    "Cleanser",
    "Toner",
    "Serum",
    "Moisturiser",
    "Sunscreen",
    "Eye Cream",
    "Face Mask",
    "Exfoliant",
    "Face Oil",
    "Mist / Essence",
    "Spot Treatment",
    "Other",
]

# Skincare goals
SKIN_GOALS = [
    "Clear skin",
    "Brighter complexion",
    "Even skin tone",
    "Anti-ageing / smoother texture",
    "Stronger skin barrier",
    "Hydrated glow",
]

# Ingredient flags
INGREDIENT_FLAGS = {
    "fragrance_free": "🌿 Fragrance-free",
    "alcohol_free": "🚫 Alcohol-free",
    "paraben_free": "✅ Paraben-free",
    "cruelty_free": "🐰 Cruelty-free",
    "vegan": "🌱 Vegan",
    "reef_safe": "🐠 Reef-safe",
}
