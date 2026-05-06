import requests

def search_obf(query: str) -> list[dict]:
    url = "https://world.openbeautyfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "fields": "code,product_name,brands,ingredients_text",
        "page_size": 10,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json().get("products", [])

def parse_ingredients_text(text):
    cleaned_text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    ingredients = [i.strip().lower() for i in cleaned_text.split(",")]
    return ingredients

import json

with open("dermstore_data.json") as f:
    data = json.load(f)

bundles = [p for p in data if "\n" in p["ingredients"]]
singles = [p for p in data if "\n" not in p["ingredients"] and p["ingredients"]]
empty = [p for p in data if not p["ingredients"]]

print(f"singles: {len(singles)}")
print(f"bundles: {len(bundles)}")
print(f"empty: {len(empty)}")

# results = search_obf("cerave moisturising cream")
# ingredients_text = parse_ingredients_text(results[0]['ingredients_text'])
# print(ingredients_text)