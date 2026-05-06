"""
test_single_ingredient.py

Validates the full pipeline for one ingredient without writing anything anywhere.
Tests the updated schema with conditions, safety_notes, and improved usage_guidance.

Usage:
    python test_single_ingredient.py
    python test_single_ingredient.py "glycolic acid"
"""

import sys
import re
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI
import os
from conflict_lookup import get_conflicts

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
PUBMED_API_KEY  = os.getenv("PUBMED_API_KEY", "")

LLM_MODEL = "anthropic/claude-sonnet-4-5"

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_inci(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# PubMed — two searches: general + conflict-focused
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 3) -> list[dict]:
    """Generic PubMed search. Returns list of {title, abstract, url}."""
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
    Drop a source if the ingredient name does not appear in the abstract text.
    This catches two failure modes:
    - Old PubMed records where the abstract is unrelated despite a title match
    - Compound names that contain the ingredient as a substring (e.g.
      polylactic-co-glycolic acid matching a search for glycolic acid)
    Checks against the normalised ingredient name and its first word as a
    fallback for multi-word ingredients.
    """
    abstract_lower = source.get("abstract", "").lower()
    name_lower     = ingredient_name.lower()

    if name_lower in abstract_lower:
        return True

    # Fallback: first meaningful word (skips articles/prepositions)
    first_word = name_lower.split()[0]
    if len(first_word) > 4 and first_word in abstract_lower:
        return True

    return False


def fetch_all_sources(ingredient_name: str) -> list[dict]:
    """
    Runs two PubMed searches, filters for relevance, and deduplicates by URL.

    General search   — title-scoped to avoid false matches on compound names
    Conflict search  — title-scoped, targets interaction and combination papers
    Relevance filter — drops any abstract that does not mention the ingredient
                       in its body text, regardless of what the title says
    """
    general_query  = (
        f'"{ingredient_name}"[Title] '
        f"AND (skin OR cosmetic OR topical OR dermatology)"
    )
    conflict_query = (
        f'"{ingredient_name}"[Title] '
        f"AND (interaction OR combination OR incompatible OR avoid OR irritation)"
    )

    print(f"  General PubMed search...")
    general_raw  = search_pubmed(general_query,  max_results=3)
    general      = [s for s in general_raw if is_relevant(s, ingredient_name)]
    dropped      = len(general_raw) - len(general)
    print(f"  Got {len(general_raw)} abstract(s), {dropped} dropped as irrelevant, {len(general)} kept.")

    # Brief pause — PubMed rate limit is 3 req/s without a key
    time.sleep(1)

    print(f"  Conflict-focused PubMed search...")
    conflict_raw = search_pubmed(conflict_query, max_results=2)
    conflict     = [s for s in conflict_raw if is_relevant(s, ingredient_name)]
    dropped      = len(conflict_raw) - len(conflict)
    print(f"  Got {len(conflict_raw)} abstract(s), {dropped} dropped as irrelevant, {len(conflict)} kept.")

    # Deduplicate by URL, general results take priority
    seen     = {s["url"] for s in general}
    combined = general + [s for s in conflict if s["url"] not in seen]

    print(f"  Combined: {len(combined)} unique relevant abstract(s).")
    return combined


# ---------------------------------------------------------------------------
# Prompt — updated schema
# ---------------------------------------------------------------------------

def build_prompt(inci_name: str, aliases: list[str], sources: list[dict]) -> str:
    if sources:
        sources_block = ""
        for s in sources:
            sources_block += f"\n---\nTITLE: {s['title']}\nURL: {s['url']}\nCONTENT:\n{s['abstract']}\n"
    else:
        sources_block = "No external sources were retrieved."

    return f"""You are building a structured ingredient knowledge entry for a skincare advisor app. \
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
  Second, if functions is still empty after extraction, you may infer from the
ingredient INCI name and category using established cosmetic chemistry knowledge.
Mark inferred values with a trailing asterisk (e.g. "exfoliant*").
If you cannot reasonably infer even from the name, return empty array.

conditions
  Extract from sources only. List specific skin conditions this ingredient is \
documented to address (e.g. acne vulgaris, rosacea, hyperpigmentation, eczema, \
melasma, actinic keratosis). Use clinical names where the source uses them. \
If not in sources, return empty array.

known_conflicts
  Extract from sources only. List INCI names (lowercase) of ingredients that \
interact negatively with this one. If not in sources, return empty array.

safety_notes
  Extract from sources only. Cover tolerability, photosensitivity, pregnancy \
safety, or barrier warnings. 1 to 3 plain English sentences. \
If nothing documented, return empty string.

suitable_skin_types and avoid_skin_types
  First, extract any explicit skin type mentions from the sources.
  Second, if suitable_skin_types is still empty after extraction, you may infer \
from the functions field using established cosmetic chemistry knowledge \
(e.g. sebostatic and antimicrobial functions imply oily and acne-prone). \
Mark inferred values with a trailing asterisk so they can be reviewed \
(e.g. "oily*"). Only use these values: \
oily, dry, combination, sensitive, normal, acne-prone, mature. \
  IMPORTANT: if this ingredient is well-tolerated across all skin types with no \
meaningful preference, return ["all"] instead of listing every type individually. \
  avoid_skin_types: only populate when the ingredient is genuinely problematic \
for a specific skin type, not just suboptimal. Mark inferred values with *. \
If nothing is documented or reasonably inferable, return empty array.

usage_guidance
  Write this for a real person managing their skincare routine. Be practical and direct.
  First, extract any specific guidance from the sources (concentration ranges, \
frequency, timing).
  Second, you may draw on established cosmetic chemistry knowledge to fill in \
practical detail that the sources do not cover, because clinical abstracts rarely \
contain consumer-facing instructions.
  Cover: AM or PM or both, frequency (daily or a few times a week), \
layering order relative to other products, what to expect in the first few weeks, \
any concentration or pH considerations worth knowing.
  Do NOT include any sentence about what the sources did or did not say. \
Do NOT hedge. Write as if you are advising the user directly.

sources
  Only include URLs that appear in RETRIEVED SOURCES above. Do not invent sources.

Return ONLY a valid JSON object matching this exact schema. No markdown, no explanation:

{{
  "functions": ["exfoliant*", "keratolytic*"],
  "suitable_skin_types": ["oily", "acne-prone"],
  "avoid_skin_types": ["sensitive*"],
  "conditions": ["acne vulgaris", "hyperpigmentation"],
  "known_conflicts": ["retinol"],
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
        print(f"  Raw response (first 400 chars): {text[:400]}")
        return None
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Confidence — updated to check new fields
# ---------------------------------------------------------------------------

def compute_confidence(sources: list[dict], llm_result: dict) -> int:
    """
    3 = 2+ PubMed abstracts AND all three key fields populated
    2 = 1 PubMed abstract OR key fields partially populated
    1 = no sources and minimal output

    Key fields: functions, usage_guidance, suitable_skin_types.
    An entry missing functions is less useful to the chatbot regardless
    of how many sources were retrieved, so it cannot score 3.
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
# Test runner
# ---------------------------------------------------------------------------

def test(inci_name: str = "niacinamide", aliases: list[str] | None = None):
    if aliases is None:
        aliases = [inci_name]

    norm = normalise_inci(inci_name)
    print(f"\n=== Testing pipeline for: {norm} ===")
    print(f"Model: {LLM_MODEL}\n")

    client = get_client()

    # Step 1 — PubMed (two searches)
    print("--- Step 1: PubMed retrieval ---")
    sources = fetch_all_sources(inci_name)
    print()

    # Step 2 — LLM
    print("--- Step 2: LLM enrichment ---")
    prompt = build_prompt(norm, aliases, sources)
    print(f"Prompt length: {len(prompt)} chars")
    print("Calling LLM...")
    result = call_llm(client, prompt)

    if result is None:
        print("FAILED: LLM returned no usable result.")
        return

    print("\nLLM result:")
    print(json.dumps(result, indent=2))

    # Step 3 — Merge curated conflicts from lookup table
    print("\n--- Step 3: Conflict lookup merge ---")
    curated       = get_conflicts(norm)
    llm_names     = set(result.get("known_conflicts", []))
    curated_names = {c["inci_name"] for c in curated}
    llm_only      = llm_names - curated_names
    lookup_only   = curated_names - llm_names
    both          = llm_names & curated_names
    merged_names  = sorted(curated_names | llm_names)

    # Merge into result — curated is source of truth, LLM additions kept but flagged
    result["known_conflicts"]        = merged_names
    result["known_conflicts_detail"] = curated   # full objects with level + reason

    print(f"  Curated conflicts:           {sorted(curated_names) or 'none'}")
    print(f"  LLM-extracted:               {sorted(llm_names) or 'none'}")
    print(f"  In both:                     {sorted(both) or 'none'}")
    print(f"  Lookup only (added):         {sorted(lookup_only) or 'none'}")
    print(f"  LLM only (kept, unverified): {sorted(llm_only) or 'none'}")
    print(f"  Final known_conflicts:       {merged_names or 'none'}")

    if curated:
        print("\n  Conflict detail:")
        for c in curated:
            print(f"    [{c['level'].upper()}] vs {c['inci_name']}")
            print(f"      {c['reason'][:90]}...")

    # Step 4 — Confidence
    print("\n--- Step 4: Confidence ---")
    confidence = compute_confidence(sources, result)
    print(f"Confidence: {confidence}/3")

    # Step 5 — Field summary
    print("\n--- Step 5: Field summary ---")
    print("  (* = inferred from ingredient name or functions, not extracted from sources)")
    fields = [
        "functions", "suitable_skin_types", "avoid_skin_types",
        "conditions", "known_conflicts", "safety_notes", "usage_guidance",
    ]
    for f in fields:
        val = result.get(f)
        if isinstance(val, list):
            status = f"{len(val)} item(s): {val}" if val else "EMPTY"
        else:
            status = f"{len(val)} chars" if val else "EMPTY"
        print(f"  {f:<22} {status}")

    print("\n=== Test complete. No files written. ===")


if __name__ == "__main__":
    ingredient = sys.argv[1] if len(sys.argv) > 1 else "niacinamide"
    test(ingredient)