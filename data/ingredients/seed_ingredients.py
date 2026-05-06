"""
seed_ingredients.py

Fetches ingredient knowledge via PubMed + LLM, merges curated conflict data,
and writes results to ingredients.json. No database writes happen here.

Review ingredients.json, edit anything you want, then import via Cloud Console.

Usage:
    python seed_ingredients.py
    python seed_ingredients.py --resume               # skip entries already in ingredients.json
    python seed_ingredients.py --only retinol "glycolic acid"
"""

import os
import re
import json
import time
import argparse
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from data.conflict_lookup import get_conflicts

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
PUBMED_API_KEY  = os.getenv("PUBMED_API_KEY", "")

LLM_MODEL = "anthropic/claude-sonnet-4-5"

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

OUTPUT_FILE            = Path("ingredients.json")
INTER_INGREDIENT_DELAY = 3   # seconds between ingredients

# ---------------------------------------------------------------------------
# Seed list — (inci_name, [aliases])
# ---------------------------------------------------------------------------

SEED_INGREDIENTS = [
    # Exfoliants
    ("glycolic acid",               ["glycolic acid", "AHA"]),
    ("lactic acid",                 ["lactic acid", "AHA"]),
    ("salicylic acid",              ["salicylic acid", "BHA"]),
    ("mandelic acid",               ["mandelic acid", "AHA"]),
    ("azelaic acid",                ["azelaic acid"]),
    ("citric acid",                 ["citric acid"]),

    # Retinoids
    ("retinol",                     ["retinol", "vitamin A"]),
    ("retinal",                     ["retinaldehyde", "retinal"]),
    ("retinyl palmitate",           ["retinyl palmitate"]),
    ("adapalene",                   ["adapalene", "differin"]),

    # Vitamin C family
    ("ascorbic acid",               ["vitamin C", "L-ascorbic acid", "ascorbic acid"]),
    ("sodium ascorbyl phosphate",   ["sodium ascorbyl phosphate", "SAP"]),
    ("ascorbyl glucoside",          ["ascorbyl glucoside"]),
    ("tetrahexyldecyl ascorbate",   ["THD ascorbate"]),

    # Niacinamide and B vitamins
    ("niacinamide",                 ["niacinamide", "vitamin B3", "nicotinamide"]),
    ("panthenol",                   ["panthenol", "vitamin B5", "pro-vitamin B5"]),
    ("pyridoxine",                  ["pyridoxine", "vitamin B6"]),

    # Peptides
    ("palmitoyl tripeptide-1",      ["palmitoyl tripeptide-1", "matrixyl"]),
    ("palmitoyl tetrapeptide-7",    ["palmitoyl tetrapeptide-7", "matrixyl 3000"]),
    ("acetyl hexapeptide-3",        ["argireline", "acetyl hexapeptide-8"]),
    ("copper tripeptide-1",         ["GHK-Cu", "copper peptide"]),

    # Humectants
    ("hyaluronic acid",             ["hyaluronic acid", "HA", "sodium hyaluronate"]),
    ("sodium hyaluronate",          ["sodium hyaluronate", "hyaluronic acid salt"]),
    ("glycerin",                    ["glycerin", "glycerol"]),
    ("propylene glycol",            ["propylene glycol"]),
    ("urea",                        ["urea", "carbamide"]),
    ("sorbitol",                    ["sorbitol"]),

    # Occlusives and emollients
    ("petrolatum",                  ["petrolatum", "petroleum jelly", "vaseline"]),
    ("dimethicone",                 ["dimethicone", "silicone"]),
    ("squalane",                    ["squalane"]),
    ("ceramide np",                 ["ceramide NP", "ceramide 3"]),
    ("ceramide ap",                 ["ceramide AP", "ceramide 6-II"]),
    ("ceramide eop",                ["ceramide EOP", "ceramide 1"]),
    ("shea butter",                 ["shea butter", "butyrospermum parkii"]),
    ("jojoba oil",                  ["jojoba oil", "simmondsia chinensis"]),
    ("rosehip oil",                 ["rosehip oil", "rosa canina"]),

    # Antioxidants
    ("tocopherol",                  ["vitamin E", "tocopherol", "alpha-tocopherol"]),
    ("tocopheryl acetate",          ["vitamin E acetate", "tocopheryl acetate"]),
    ("resveratrol",                 ["resveratrol"]),
    ("ferulic acid",                ["ferulic acid"]),
    ("coenzyme q10",                ["coQ10", "ubiquinone", "coenzyme Q10"]),

    # SPF actives
    ("zinc oxide",                  ["zinc oxide", "mineral SPF"]),
    ("titanium dioxide",            ["titanium dioxide", "mineral SPF"]),
    ("avobenzone",                  ["avobenzone", "butyl methoxydibenzoylmethane"]),
    ("octinoxate",                  ["octinoxate", "ethylhexyl methoxycinnamate"]),

    # Brightening
    ("kojic acid",                  ["kojic acid"]),
    ("alpha arbutin",               ["alpha arbutin"]),
    ("tranexamic acid",             ["tranexamic acid"]),
    ("licorice root extract",       ["licorice extract", "glycyrrhiza glabra"]),

    # Anti-inflammatory and soothing
    ("centella asiatica extract",   ["centella asiatica", "CICA", "gotu kola"]),
    ("bisabolol",                   ["bisabolol", "alpha-bisabolol"]),
    ("allantoin",                   ["allantoin"]),
    ("caffeine",                    ["caffeine"]),
    ("green tea extract",           ["green tea", "camellia sinensis"]),
    ("aloe vera",                   ["aloe vera", "aloe barbadensis"]),

    # Preservatives
    ("phenoxyethanol",              ["phenoxyethanol"]),
    ("ethylhexylglycerin",          ["ethylhexylglycerin"]),
    ("sodium benzoate",             ["sodium benzoate"]),
    ("potassium sorbate",           ["potassium sorbate"]),

    # pH adjusters and functional
    ("sodium hydroxide",            ["sodium hydroxide", "lye"]),
    ("triethanolamine",             ["triethanolamine", "TEA"]),
    ("carbomer",                    ["carbomer", "carbopol"]),

    # Acne-targeted
    ("benzoyl peroxide",            ["benzoyl peroxide", "BPO"]),
    ("zinc pca",                    ["zinc PCA"]),
    ("sulfur",                      ["sulfur", "sulphur"]),

    # Enzymes
    ("papain",                      ["papain", "papaya enzyme"]),
    ("bromelain",                   ["bromelain", "pineapple enzyme"]),
]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_inci(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


# ---------------------------------------------------------------------------
# JSON store — load once, write after every ingredient so crashes lose nothing
# ---------------------------------------------------------------------------

def load_existing() -> dict:
    if not OUTPUT_FILE.exists():
        return {}
    with open(OUTPUT_FILE) as f:
        entries = json.load(f)
    return {e["inci_name"]: e for e in entries}


def save_all(store: dict):
    entries = sorted(store.values(), key=lambda e: e["inci_name"])
    with open(OUTPUT_FILE, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 3) -> list[dict]:
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        r = requests.get(PUBMED_SEARCH, params=params, timeout=10)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
    except Exception as e:
        print(f"  PubMed search error: {e}")
        return []

    fetch_params = {
        "db":      "pubmed",
        "id":      ",".join(ids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if PUBMED_API_KEY:
        fetch_params["api_key"] = PUBMED_API_KEY

    try:
        r = requests.get(PUBMED_FETCH, params=fetch_params, timeout=15)
        r.raise_for_status()
        xml = r.text
    except Exception as e:
        print(f"  PubMed fetch error: {e}")
        return []

    titles    = re.findall(r"<ArticleTitle>(.*?)</ArticleTitle>",    xml, re.DOTALL)
    abstracts = re.findall(r"<AbstractText.*?>(.*?)</AbstractText>", xml, re.DOTALL)

    results = []
    for i, pmid in enumerate(ids):
        title    = re.sub(r"<.*?>", "", titles[i])    if i < len(titles)    else "No title"
        abstract = re.sub(r"<.*?>", "", abstracts[i]) if i < len(abstracts) else ""
        if not abstract:
            continue
        results.append({
            "title":    title.strip(),
            "abstract": abstract.strip()[:2000],
            "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return results


def is_relevant(source: dict, ingredient_name: str) -> bool:
    """
    Drop sources where the ingredient name does not appear in the abstract body.
    Catches old anomalous PubMed records and compound name false matches
    (e.g. polylactic-co-glycolic acid matching a search for glycolic acid).
    """
    abstract_lower = source.get("abstract", "").lower()
    name_lower     = ingredient_name.lower()

    if name_lower in abstract_lower:
        return True

    first_word = name_lower.split()[0]
    if len(first_word) > 4 and first_word in abstract_lower:
        return True

    return False


def fetch_all_sources(ingredient_name: str) -> list[dict]:
    """
    Two PubMed searches, relevance-filtered and deduplicated.
    1-second sleep between calls to stay within the 3 req/s rate limit.
    """
    general_query  = (
        f'"{ingredient_name}"[Title] '
        f"AND (skin OR cosmetic OR topical OR dermatology)"
    )
    conflict_query = (
        f'"{ingredient_name}"[Title] '
        f"AND (interaction OR combination OR incompatible OR avoid OR irritation)"
    )

    general_raw = search_pubmed(general_query, max_results=3)
    general     = [s for s in general_raw if is_relevant(s, ingredient_name)]
    print(f"  General: {len(general_raw)} fetched, {len(general_raw) - len(general)} dropped, {len(general)} kept.")

    time.sleep(1)

    conflict_raw = search_pubmed(conflict_query, max_results=2)
    conflict     = [s for s in conflict_raw if is_relevant(s, ingredient_name)]
    print(f"  Conflict: {len(conflict_raw)} fetched, {len(conflict_raw) - len(conflict)} dropped, {len(conflict)} kept.")

    seen     = {s["url"] for s in general}
    combined = general + [s for s in conflict if s["url"] not in seen]
    print(f"  Combined: {len(combined)} unique relevant abstract(s).")

    return combined


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(inci_name: str, aliases: list[str], sources: list[dict]) -> str:
    if sources:
        sources_block = ""
        for s in sources:
            sources_block += f"\n---\nTITLE: {s['title']}\nURL: {s['url']}\nCONTENT:\n{s['abstract']}\n"
    else:
        sources_block = "No external sources were retrieved."

    return f"""You are building a structured ingredient knowledge entry for a skincare advisor app.
The entry will be read by real users making decisions about their skincare routine.

INGREDIENT (INCI name): {inci_name}
ALSO KNOWN AS: {", ".join(aliases)}

RETRIEVED SOURCES:
{sources_block}

FIELD-BY-FIELD INSTRUCTIONS:

functions
  First, extract from the sources any functional roles this ingredient plays
  (e.g. exfoliant, humectant, antioxidant, sebostatic, skin brightening, keratolytic,
  emollient, occlusive, pH-adjuster, preservative, film-forming).
  Second, if functions is still empty after extraction, infer from the INCI name
  and category using established cosmetic chemistry knowledge.
  Mark inferred values with a trailing asterisk (e.g. "exfoliant*").
  If you cannot reasonably infer even from the name, return empty array.

conditions
  Extract from sources only. List specific skin conditions this ingredient is
  documented to address (e.g. acne vulgaris, rosacea, hyperpigmentation, eczema,
  melasma, actinic keratosis). Use clinical names where the source uses them.
  If not in sources, return empty array.

known_conflicts
  Extract from sources only. List INCI names (lowercase) of ingredients that
  interact negatively with this one. If not in sources, return empty array.

safety_notes
  Extract from sources only. Cover tolerability, photosensitivity, pregnancy
  safety, or barrier warnings. 1 to 3 plain English sentences.
  If nothing documented, return empty string.

suitable_skin_types
  First, extract any explicit skin type mentions from the sources.
  Second, if still empty, infer from the functions field using established
  cosmetic chemistry knowledge. Mark inferred values with a trailing asterisk.
  Only use these values: oily, dry, combination, sensitive, normal, acne-prone, mature.
  IMPORTANT: if this ingredient is well-tolerated across all skin types with no
  meaningful preference, return ["all"] instead of listing every type individually.
  If you cannot reasonably infer, return empty array.

avoid_skin_types
  Only populate when the ingredient is genuinely problematic for a specific skin
  type — not just suboptimal. Mark inferred values with a trailing asterisk.
  Use the same allowed values as suitable_skin_types.
  If nothing is documented or reasonably inferable, return empty array.

usage_guidance
  Write this for a real person managing their skincare routine. Be practical and direct.
  First, extract any specific guidance from the sources (concentration ranges, frequency, timing).
  Second, draw on established cosmetic chemistry knowledge to fill in practical detail
  that the sources do not cover — clinical abstracts rarely contain consumer-facing instructions.
  Cover: AM or PM or both, frequency, layering order, what to expect in the first few weeks,
  any concentration or pH considerations.
  Do NOT mention what the sources said or did not say. Do NOT hedge.
  Write as if you are advising the user directly.

sources
  Only include URLs that appear in RETRIEVED SOURCES above. Do not invent sources.

Return ONLY a valid JSON object matching this exact schema. No markdown, no explanation:

{{
  "functions": ["exfoliant*", "keratolytic*"],
  "suitable_skin_types": ["oily", "acne-prone"],
  "avoid_skin_types": ["sensitive*"],
  "conditions": ["acne vulgaris", "hyperpigmentation"],
  "known_conflicts": [],
  "safety_notes": "Well tolerated at concentrations up to 10%. Increases photosensitivity — always follow with SPF in the morning.",
  "usage_guidance": "Start 2 to 3 times a week in the evening and build up to daily as tolerated. Apply to clean dry skin before moisturiser. Always wear SPF the following morning. Expect some initial tingling — this settles within a few weeks.",
  "sources": [
    {{"title": "exact title from above", "url": "exact url from above"}}
  ]
}}""".strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, prompt: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$",          "", text)
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None



# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def compute_confidence(sources: list[dict], llm_result: dict) -> int:
    """
    3 = 2+ PubMed abstracts AND functions + usage_guidance + suitable_skin_types all populated
    2 = 1 PubMed abstract OR key fields partially populated
    1 = no sources and minimal output
    """
    pubmed_count      = len(sources)
    key_fields_filled = all([
        llm_result.get("functions"),
        llm_result.get("usage_guidance"),
        llm_result.get("suitable_skin_types"),
    ])
    if pubmed_count >= 2 and key_fields_filled:
        return 3
    if pubmed_count >= 1 or key_fields_filled:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Conflict merge
# ---------------------------------------------------------------------------

def merge_conflicts(inci_name: str, llm_result: dict) -> tuple[list[str], list[dict]]:
    """
    Merges LLM-extracted conflicts with the curated lookup table.
    Returns:
      known_conflicts        — deduplicated INCI name list (for DB storage)
      known_conflicts_detail — full conflict objects with level + reason (for JSON)
    """
    curated       = get_conflicts(inci_name)
    llm_names     = set(llm_result.get("known_conflicts", []))
    curated_names = {c["inci_name"] for c in curated}
    merged_names  = sorted(curated_names | llm_names)

    # LLM-only entries get a placeholder detail object
    curated_detail = list(curated)
    for name in sorted(llm_names - curated_names):
        curated_detail.append({
            "inci_name": name,
            "level":     "caution",
            "reason":    "Extracted by LLM from sources — not yet verified in curated lookup.",
        })

    return merged_names, curated_detail


# ---------------------------------------------------------------------------
# Single-ingredient pipeline
# ---------------------------------------------------------------------------

def enrich_one(client: OpenAI, inci_name: str, aliases: list[str]) -> dict | None:
    norm = normalise_inci(inci_name)

    print(f"  Fetching PubMed sources...")
    sources = fetch_all_sources(inci_name)

    print(f"  Calling LLM ({LLM_MODEL})...")
    prompt     = build_prompt(norm, aliases, sources)
    llm_result = call_llm(client, prompt)

    if llm_result is None:
        print(f"  LLM returned no usable result.")
        return None

    # Merge conflicts
    merged_conflicts, conflict_detail = merge_conflicts(norm, llm_result)
    llm_result["known_conflicts"]        = merged_conflicts
    llm_result["known_conflicts_detail"] = conflict_detail

    confidence = compute_confidence(sources, llm_result)
    print(f"  Confidence: {confidence}/3  |  conflicts: {len(merged_conflicts)}  |  functions: {len(llm_result.get('functions', []))}")

    return {
        "inci_name":               norm,
        "common_names":            aliases,
        "functions":               llm_result.get("functions", []),
        "suitable_skin_types":     llm_result.get("suitable_skin_types", []),
        "avoid_skin_types":        llm_result.get("avoid_skin_types", []),
        "conditions":              llm_result.get("conditions", []),
        "known_conflicts":         merged_conflicts,
        "known_conflicts_detail":  conflict_detail,
        "safety_notes":            llm_result.get("safety_notes", ""),
        "usage_guidance":          llm_result.get("usage_guidance", ""),
        "confidence":              confidence,
        "sources":                 llm_result.get("sources", []),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip ingredients already in ingredients.json")
    parser.add_argument("--only", nargs="+", metavar="INCI_NAME",
                        help="Run only these specific ingredients")
    args = parser.parse_args()

    print("=== Skincare Advisor: Ingredient Corpus Seeding ===")
    print(f"Model:  {LLM_MODEL}")
    print(f"Output: {OUTPUT_FILE}\n")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    client = get_client()
    store  = load_existing()
    print(f"Existing entries in {OUTPUT_FILE}: {len(store)}")

    seed_map = {normalise_inci(n): aliases for n, aliases in SEED_INGREDIENTS}

    if args.only:
        targets = [(normalise_inci(n), seed_map.get(normalise_inci(n), [n])) for n in args.only]
    else:
        targets = [(normalise_inci(n), aliases) for n, aliases in SEED_INGREDIENTS]

    if args.resume:
        before  = len(targets)
        targets = [(n, a) for n, a in targets if n not in store]
        print(f"Resuming: {before - len(targets)} already done, {len(targets)} remaining.\n")

    total     = len(targets)
    succeeded = 0
    failed    = []

    for i, (norm, aliases) in enumerate(targets, 1):
        print(f"\n({i}/{total}) [{norm}]")

        entry = enrich_one(client, norm, aliases)

        if entry is None:
            failed.append(norm)
        else:
            store[norm] = entry
            save_all(store)
            succeeded += 1
            print(f"  Saved to {OUTPUT_FILE}.")

        if i < total:
            time.sleep(INTER_INGREDIENT_DELAY)

    print(f"\n\n=== Done ===")
    print(f"Succeeded: {succeeded}/{total}")
    print(f"File:      {OUTPUT_FILE} ({len(store)} total entries)")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
        print(f"\nRe-run failed ones with:")
        print(f"  python seed_ingredients.py --only {' '.join(repr(n) for n in failed)}")
    else:
        print("All ingredients completed.")


if __name__ == "__main__":
    main()