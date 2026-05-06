#!/usr/bin/env python3
"""
Boots.com skincare scraper (Crawlbase) — run once to populate Cloud SQL.

Usage:
    cd skincare_advisor
    python data/products/scrape_boots_crawlbase.py

Scrapes face/skincare categories, visits each product page for ingredients,
and saves to skincare_advisor.products.

Requires env vars:
    CRAWLBASE_TOKEN  — Crawlbase API token (JS token for JS-rendered pages)
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

CRAWLBASE_TOKEN = os.getenv("CRAWLBASE_TOKEN")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

BOOTS_BASE = "https://www.boots.com"
DELAY = 3  # seconds between Crawlbase calls

CATEGORIES = [
    ("Facial Skincare", "/beauty/skincare/facial-skincare"),
]

# Curated brand slug → display name for common Boots brands
BRAND_LOOKUP = {
    "no7": "No7",
    "cerave": "CeraVe",
    "la-roche-posay": "La Roche-Posay",
    "la roche posay": "La Roche-Posay",
    "laroche-posay": "La Roche-Posay",
    "neutrogena": "Neutrogena",
    "olay": "Olay",
    "loreal": "L'Oréal",
    "l-oreal": "L'Oréal",
    "garnier": "Garnier",
    "vichy": "Vichy",
    "avene": "Avène",
    "bioderma": "Bioderma",
    "eucerin": "Eucerin",
    "nivea": "NIVEA",
    "simple": "Simple",
    "clinique": "Clinique",
    "estee": "Estée Lauder",
    "clarins": "Clarins",
    "elemis": "Elemis",
    "the-ordinary": "The Ordinary",
    "the ordinary": "The Ordinary",
    "inkey": "The INKEY List",
    "paula": "Paula's Choice",
    "paulas": "Paula's Choice",
    "peter-thomas-roth": "Peter Thomas Roth",
    "drunk-elephant": "Drunk Elephant",
    "tatcha": "Tatcha",
    "kiehl": "Kiehl's",
    "kiehls": "Kiehl's",
    "origins": "Origins",
    "fresh": "Fresh",
    "liz-earle": "Liz Earle",
    "medik8": "Medik8",
    "alpha-h": "Alpha-H",
    "nip-fab": "Nip+Fab",
    "indeed-labs": "Indeed Labs",
    "byoma": "BYOMA",
    "isntree": "Isntree",
    "some-by-mi": "Some By Mi",
    "cosrx": "COSRX",
    "anua": "Anua",
    "torriden": "Torriden",
    "skin1004": "Skin1004",
    "lumene": "Lumene",
    "bamford": "Bamford",
    "boots": "Boots",
    "soap-glory": "Soap & Glory",
    "ted-baker": "Ted Baker",
    "sanctuary": "Sanctuary Spa",
    "radox": "Radox",
    "dove": "Dove",
    "aveeno": "Aveeno",
    "e45": "E45",
    "sudocrem": "Sudocrem",
    "vaseline": "Vaseline",
    "aquaphor": "Aquaphor",
    "la-mer": "La Mer",
    "shiseido": "Shiseido",
    "sk-ii": "SK-II",
    "sulwhasoo": "Sulwhasoo",
    "laneige": "Laneige",
    "innisfree": "Innisfree",
    "missha": "Missha",
    "nars": "NARS",
    "charlotte-tilbury": "Charlotte Tilbury",
}


# ── Crawlbase API ─────────────────────────────────────────────────────────────

def crawlbase_get(url: str, render: bool = True) -> requests.Response | None:
    """Fetch a URL via Crawlbase API.

    Terminates the process immediately on 401 (invalid/missing token).
    Returns None on other non-200 responses.
    """
    params = {
        "token": CRAWLBASE_TOKEN,
        "url": url,
        "country": "GB",
    }
    if render:
        params["ajax_wait"] = "true"
        params["page_wait"] = "5000"

    try:
        resp = requests.get(
            "https://api.crawlbase.com/",
            params=params,
            timeout=180,
        )
        if resp.status_code == 401:
            log.error("Crawlbase returned 401 Unauthorized — check CRAWLBASE_TOKEN. Terminating.")
            sys.exit(1)
        if resp.status_code == 200:
            return resp
        log.warning(f"Crawlbase returned {resp.status_code} for {url}: {resp.text[:200]}")
        return None
    except Exception as e:
        log.error(f"Crawlbase error for {url}: {e}")
        return None


# ── Product URL discovery ─────────────────────────────────────────────────────

_INCLUDE = re.compile(
    r"moistur|serum|cleanser|toner|spf|suncream|sunscreen|eye.cream|face.mask|"
    r"exfoliat|retinol|vitamin.c|niacinamide|hyaluronic|peptide|"
    r"salicylic|glycolic|azelaic|ceramide|aha|bha",
    re.I,
)

_EXCLUDE = re.compile(
    r"hair|shampoo|conditioner|detangling|scalp|heat.protect|curl|frizz|split.end|"
    r"kerastase|olaplex|wella|redken|tresemme|elvive|herbal.essence|pantene|"
    r"head.and.shoulders|lee.stafford|philip.kingsley|aussie|tigi|john.frieda|"
    r"vo5|alberto.balsam|briogeo|leave.in.|\bbody.|\bbody\b|hand.|foot.|\bleg.|"
    r"shower.gel|\bbath.|intimate|mascara|foundation|concealer|eyeshadow|eyeliner|"
    r"lipstick|lip.gloss|lip.balm|lip.cream|lip.serum|lip.treatment|liptreatment|"
    r"volumising.lip|\blip.|\bblush\b|bronzer|\bhighlighter\b|\bbrow.|\blash.|"
    r"self.tan|fake.tan|gradual.tan|tanning|bronze|tan.protect|festival/|"
    r"/sunprotection/|family.sun|piz.buin|ambre.solaire|after.sun|banana.boat|"
    r"sun.oil|hawaiian.tropic|p20.original|baby.|newborn|childs.farm|midwife|"
    r"pediatric|paediatric|eau.tendre|eau.de.parfum|eau.de.toilette|cologne|"
    r"aftershave|sauvage|bleu.de|bb.cream|cc.cream|makeup.serum|smoothie.makeup|"
    r"aveda|dior.sauvage|gillette|razor|blades.refill|capsules|tablets|"
    r"supplement|multivitamin|.device.|sennse",
    re.I,
)

SITEMAP_FILE = os.path.expanduser("~/Downloads/uk-product-sitemap.xml")


def get_urls_from_sitemap() -> list[str]:
    """Load skincare product URLs from local sitemap file."""
    if not os.path.exists(SITEMAP_FILE):
        log.warning(f"Sitemap file not found: {SITEMAP_FILE}")
        return []
    log.info(f"Loading sitemap from {SITEMAP_FILE}")
    with open(SITEMAP_FILE, encoding="utf-8") as f:
        content = f.read()
    urls = re.findall(r"<loc>(https://www\.boots\.com[^<]+)</loc>", content)
    skincare = [u for u in urls if _INCLUDE.search(u) and not _EXCLUDE.search(u)]
    log.info(f"Sitemap: {len(urls)} total URLs → {len(skincare)} skincare matches")
    return skincare


# ── Category page parsing ─────────────────────────────────────────────────────

def extract_product_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"-\d{7,}(?:\?|$)", href + "?"):
            full = urljoin(BOOTS_BASE, href).split("?")[0]
            if "boots.com" in full and "/beauty/" not in full.split(BOOTS_BASE)[-1][:8]:
                urls.add(full)
            elif "boots.com" not in full:
                pass
            else:
                urls.add(full)

    return list(urls)


def get_page_count(html: str) -> int:
    """Estimate number of pagination pages from the listing HTML."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ")

    m = re.search(r"of\s+(\d+)\s+product", text, re.IGNORECASE)
    if m:
        total_products = int(m.group(1))
        per_page = 24
        return min((total_products + per_page - 1) // per_page, 25)

    page_nums = [
        int(a.get_text(strip=True))
        for a in soup.find_all("a", href=True)
        if re.fullmatch(r"\d+", a.get_text(strip=True))
        and "pageNumber" in a["href"]
    ]
    return max(page_nums, default=1)


BOT_CHALLENGE_TITLES = re.compile(r"pardon our interruption|access denied|just a moment|are you a robot|checking your browser", re.I)


def _is_bot_challenge(html: str) -> bool:
    """Return True if Boots returned a bot-detection page instead of a product page."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("title")
    return bool(title and BOT_CHALLENGE_TITLES.search(title.get_text()))


# ── Product page parsing ──────────────────────────────────────────────────────

def parse_product_page(html: str, url: str, category: str = "") -> dict | None:
    if _is_bot_challenge(html):
        log.warning(f"  Bot challenge page detected, skipping: {url}")
        return None

    soup = BeautifulSoup(html, "lxml")

    product = _parse_jsonld(soup)

    if not product:
        product = _parse_selectors(soup)

    if not product or not product.get("product_name"):
        log.debug(f"Could not parse product at {url}")
        return None

    if not product.get("brand"):
        product["brand"] = _brand_from_url(url)

    if BUNDLE_KEYWORDS.search(product["product_name"]):
        log.debug(f"Skipping bundle/gift set: {product['product_name']}")
        return None

    boots_id = _extract_boots_id(url)
    if boots_id and not product.get("image_url"):
        product["image_url"] = (
            f"https://boots.scene7.com/is/image/Boots/{boots_id}"
            "?id=-Klmv1&fmt=jpg&wid=300&hei=300"
        )

    if not product.get("ingredients_raw"):
        product["ingredients_raw"] = _extract_ingredients(soup)

    product["quantity"] = product.get("quantity") or _extract_quantity(soup)
    product["categories"] = _extract_categories(product["product_name"])
    product["country"] = "United Kingdom"
    product["source"] = "boots_scrape"
    product["boots_url"] = url
    product["boots_id"] = _extract_boots_id(url)

    return product


def _parse_jsonld(soup: BeautifulSoup) -> dict | None:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = json.loads(script.string or "")
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                if item.get("@type") == "Product":
                    brand = item.get("brand", {})
                    if isinstance(brand, dict):
                        brand = brand.get("name", "")
                    image = item.get("image", "")
                    if isinstance(image, list):
                        image = image[0] if image else ""
                    return {
                        "product_name": item.get("name", "").strip(),
                        "brand": str(brand).strip(),
                        "image_url": image,
                        "ingredients_raw": "",
                    }
        except Exception:
            continue
    return None


def _parse_selectors(soup: BeautifulSoup) -> dict | None:
    name_el = (
        soup.find(attrs={"itemprop": "name"})
        or soup.find("h1")
    )
    name = name_el.get_text(strip=True) if name_el else ""

    if not name:
        title_el = soup.find("title")
        if title_el:
            name = re.sub(r"\s*[-|]\s*Boots.*$", "", title_el.get_text(strip=True), flags=re.I)

    if not name:
        return None

    brand = ""
    for selector in [
        lambda s: s.find(attrs={"itemprop": "brand"}),
        lambda s: s.find(class_=re.compile(r"product[_-]brand", re.I)),
        lambda s: s.find(class_=re.compile(r"\bbrand\b", re.I)),
        lambda s: s.find(attrs={"data-brand": True}),
    ]:
        el = selector(soup)
        if el:
            txt = el.get("data-brand") or el.get_text(strip=True)
            if txt and len(txt) < 80:
                brand = txt
                break

    img_el = (
        soup.find("img", attrs={"itemprop": "image"})
        or soup.find("img", id=re.compile(r"product.?image|main", re.I))
    )
    image_url = img_el.get("src", "") if img_el else ""

    return {"product_name": name, "brand": brand, "image_url": image_url, "ingredients_raw": ""}


_CATEGORY_RULES = [
    ("SPF / Sunscreen",  re.compile(r"\bspf\b|sun\s*cream|sun\s*screen|sun\s*block|uv\s*fluid|uv\s*cream", re.I)),
    ("Eye Cream",        re.compile(r"\beye\s*(cream|gel|serum|balm|treatment)\b", re.I)),
    ("Face Mask",        re.compile(r"\b(face\s*)?mask\b|\bpeel\b", re.I)),
    ("Exfoliant",        re.compile(r"\bexfoliant\b|\bexfoliat\w+\b|\bscrub\b|\bpeel\b", re.I)),
    ("Serum",            re.compile(r"\bserum\b|\bessence\b|\bampoule\b|\bbooster\b", re.I)),
    ("Toner",            re.compile(r"\btoner\b|\btonic\b|\bmist\b", re.I)),
    ("Cleanser",         re.compile(r"\bcleans\w+\b|\bface\s*wash\b|\bfoam\w*\b|\bmicellar\b|\bwipes?\b|\bmake.?up\s*remover\b", re.I)),
    ("Moisturiser",      re.compile(r"\bmoisturis\w+\b|\bcream\b|\blotion\b|\bfluid\b|\bgel.cream\b|\bemulsion\b|\bday\s*cream\b|\bnight\s*cream\b", re.I)),
    ("Face Oil",         re.compile(r"\bface\s*oil\b|\bfacial\s*oil\b|\bdry\s*oil\b", re.I)),
]


def _extract_categories(product_name: str) -> list[str]:
    for label, pattern in _CATEGORY_RULES:
        if pattern.search(product_name):
            return [label]
    return ["Skincare"]


def _normalise_ingredients(raw: str) -> str:
    """Convert bullet-separated (•) or comma-separated ingredient text to comma-separated."""
    if "•" in raw:
        parts = [p.strip() for p in raw.split("•") if p.strip()]
        return ", ".join(parts)
    return raw.strip()


def _extract_ingredients(soup: BeautifulSoup) -> str:
    heading = soup.find(id="product_ingredients")
    if heading:
        p = heading.find_next("p")
        if p:
            return _normalise_ingredients(p.get_text(" ", strip=True))[:4000]

    for el in list(soup.find_all(attrs={"id": re.compile(r"ingredient", re.I)})) + \
              list(soup.find_all(attrs={"class": re.compile(r"ingredient", re.I)})):
        txt = _normalise_ingredients(el.get_text(" ", strip=True))
        if len(txt) > 40:
            return txt[:4000]

    for heading in soup.find_all(string=re.compile(r"^\s*(?:full\s+)?ingredients?\s*$", re.I)):
        parent = heading.parent
        for candidate in [parent.find_next_sibling(), parent.find_next("p"), parent.find_next("div")]:
            if candidate:
                txt = _normalise_ingredients(candidate.get_text(" ", strip=True))
                if len(txt) > 40:
                    return txt[:4000]

    text = soup.get_text(" ", strip=True)
    stop = r"(?:How to use|Directions?|Warnings?|Cautions?|Hazards?|Made in|Country of origin|Delivery|Attachments)"
    for pattern in [
        rf"(?:Full\s+)?Ingredients?[:\s]+(.+?)(?:{stop}|$)",
        rf"INCI[:\s]+(.+?)(?:{stop}|$)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            raw = _normalise_ingredients(m.group(1))
            if len(raw) > 40:
                return raw[:4000]
    return ""


def _extract_quantity(soup: BeautifulSoup) -> str:
    text = soup.get_text(" ", strip=True)
    m = re.search(r"\b(\d+(?:\.\d+)?\s*(?:ml|g|oz|fl\.?\s*oz|L|kg))\b", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_ingredients(raw: str) -> list[dict]:
    if not raw:
        return []
    items = [i.strip() for i in re.split(r",(?![^(]*\))", raw) if i.strip()]
    return [{"name": n, "position": i + 1, "normalized": n.lower()} for i, n in enumerate(items)]


_PRODUCT_WORDS = {
    "cream", "gel", "serum", "lotion", "oil", "balm", "milk", "foam", "mist",
    "cleanser", "cleansing", "toner", "moisturiser", "moisturizer", "moisturising",
    "mask", "scrub", "peel", "wash", "spf", "sunscreen", "sunblock", "primer",
    "essence", "ampoule", "treatment", "fluid", "emulsion", "concentrate",
    "face", "facial", "skin", "body", "eye", "lip", "neck", "hand", "pore",
    "blemish", "acne", "anti", "sensitive", "daily", "gentle", "deep", "intense",
    "hydrating", "hydration", "soothing", "brightening", "firming", "lifting",
    "repair", "repairing", "protect", "protecting", "renew", "renewal",
    "with", "for", "and", "the", "ultra", "rich", "light", "plus", "advanced",
    "biodegradable", "foaming", "exfoliating", "refreshing", "purifying",
    "nourishing", "restoring", "revitalising", "revitalizing",
}


def _brand_from_url(url: str) -> str:
    slug = urlparse(url).path.strip("/").split("/")[-1]
    slug = re.sub(r"-\d{7,}$", "", slug)

    words = slug.split("-")
    for n in range(min(4, len(words)), 0, -1):
        key = "-".join(words[:n])
        if key in BRAND_LOOKUP:
            return BRAND_LOOKUP[key]

    brand_words = []
    for word in words:
        if word.lower() in _PRODUCT_WORDS and brand_words:
            break
        if len(brand_words) >= 3:
            break
        brand_words.append(word)
    return " ".join(w.title() for w in brand_words)


def _extract_boots_id(url: str) -> str:
    m = re.search(r"-(\d{7,})(?:\?|$)", url)
    return m.group(1) if m else ""


BUNDLE_KEYWORDS = re.compile(
    r"\b(edit|box|set|kit|bundle|collection|gift|duo|trio|starter pack|beauty bag|bag)\b", re.I
)


# ── Database ──────────────────────────────────────────────────────────────────

def make_conn():
    return psycopg2.connect(
        **DB_CONFIG,
        cursor_factory=psycopg2.extras.RealDictCursor,
        keepalives=1,
        keepalives_idle=60,
        keepalives_interval=10,
        keepalives_count=5,
    )


def ensure_conn(conn):
    """Return a live connection, reconnecting if the existing one dropped."""
    try:
        conn.cursor().execute("SELECT 1")
        return conn
    except Exception:
        log.warning("DB connection lost — reconnecting")
        try:
            conn.close()
        except Exception:
            pass
        return make_conn()


def already_scraped(boots_id: str, conn) -> bool:
    """Skip only if product exists AND already has ingredients."""
    if not boots_id:
        return False
    with conn.cursor() as cur:
        cur.execute(
            """SELECT 1 FROM skincare_advisor.products
               WHERE source = 'boots_scrape' AND obf_code = %s
               AND ingredients_raw IS NOT NULL AND ingredients_raw != ''""",
            (boots_id,),
        )
        return cur.fetchone() is not None


def save_product(product: dict, conn) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO skincare_advisor.products
                    (obf_code, product_name, brand, quantity, image_url,
                     ingredients_raw, categories,
                     country, last_synced_at, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (obf_code) DO UPDATE SET
                    ingredients_raw = EXCLUDED.ingredients_raw,
                    last_synced_at  = EXCLUDED.last_synced_at
                WHERE (skincare_advisor.products.ingredients_raw IS NULL
                    OR skincare_advisor.products.ingredients_raw = '')
                """,
                (
                    product.get("boots_id") or None,
                    product["product_name"],
                    product.get("brand", ""),
                    product.get("quantity", ""),
                    product.get("image_url", ""),
                    product.get("ingredients_raw", ""),
                    json.dumps(product.get("categories", [])),
                    product.get("country", "United Kingdom"),
                    datetime.now(timezone.utc),
                    "boots_scrape",
                ),
            )
        conn.commit()
        return True
    except Exception as e:
        log.error(f"DB error saving '{product.get('product_name')}': {e}")
        conn.rollback()
        return False


# ── Main scrape loop ──────────────────────────────────────────────────────────

def scrape_category(category_name: str, path: str, conn, limit: int | None = None) -> tuple[int, object]:
    base_url = BOOTS_BASE + path
    log.info(f"\n{'='*60}")
    log.info(f"Category: {category_name}  →  {base_url}")

    raw_urls = get_urls_from_sitemap()

    if not raw_urls:
        log.info("Sitemap failed — scraping category page via Crawlbase")
        resp = crawlbase_get(base_url, render=True)
        if not resp:
            log.error(f"Failed to fetch category page, skipping {category_name}")
            return 0, conn

        raw_urls = extract_product_urls(resp.text)
        log.info(f"Found {len(raw_urls)} product URLs from category page")

    # Deduplicate while preserving sitemap order
    seen = set()
    product_list = [u for u in raw_urls if not (u in seen or seen.add(u))]
    total = len(product_list)

    if limit:
        product_list = product_list[:limit]
    log.info(f"Total product URLs: {total}" + (f" (capped at {limit})" if limit else ""))

    saved = 0
    for i, url in enumerate(product_list, 1):
        conn = ensure_conn(conn)
        boots_id = _extract_boots_id(url)
        if already_scraped(boots_id, conn):
            log.info(f"  [{i}/{total}] SKIP (already in DB): {url}")
            continue

        log.info(f"  [{i}/{total}] Scraping: {url}")
        time.sleep(DELAY)

        resp = crawlbase_get(url, render=True)
        if not resp:
            continue

        product = parse_product_page(resp.text, url, category_name)
        if not product:
            log.warning(f"  Could not parse: {url}")
            continue

        conn = ensure_conn(conn)
        if save_product(product, conn):
            saved += 1
            log.info(f"  ✓ Saved: {product['product_name']} ({product.get('brand', '—')})")
        else:
            log.warning(f"  ✗ Failed to save: {product['product_name']}")

    log.info(f"Category '{category_name}' done — {saved} products saved")
    return saved, conn


def main(limit: int | None = None):
    log.info("Boots scraper (Crawlbase) starting")
    log.info(f"DB host: {DB_CONFIG['host']} / DB: {DB_CONFIG['database']}")

    conn = make_conn()
    log.info("DB connected")

    total = 0
    for category_name, path in CATEGORIES:
        saved, conn = scrape_category(category_name, path, conn, limit=limit)
        total += saved

    conn.close()
    log.info(f"\nAll done. Total products saved: {total}")


def test_one():
    """Scrape a single product and print the result. Does NOT save to DB."""
    log.info("TEST MODE — scraping one product from Facial Skincare category")

    category_name, path = CATEGORIES[0]
    base_url = BOOTS_BASE + path

    urls = get_urls_from_sitemap()

    if not urls:
        log.info(f"Fetching category page: {base_url}")
        resp = crawlbase_get(base_url, render=True)
        if resp:
            urls = extract_product_urls(resp.text)

    if not urls:
        log.error("No product URLs found — inspecting page:")
        soup = BeautifulSoup(resp.text, "lxml")
        all_hrefs = [a["href"] for a in soup.find_all("a", href=True)]
        log.info(f"Total <a href> tags found: {len(all_hrefs)}")
        if all_hrefs:
            log.info("Sample hrefs:")
            print("\n".join(all_hrefs[:40]))
        else:
            log.info("No anchor tags — page likely didn't render. HTML snippet:")
            print(resp.text[2000:5000])
        return

    url = urls[0]
    log.info(f"Scraping product: {url}")
    time.sleep(DELAY)

    resp = crawlbase_get(url, render=True)
    if not resp:
        log.error("Failed to fetch product page")
        return

    if "--debug-html" in sys.argv:
        out = "debug_product.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(resp.text)
        log.info(f"Full HTML saved to {out}")

    product = parse_product_page(resp.text, url, category_name)
    if not product:
        log.error("Could not parse product — check HTML snippet below")
        print(resp.text[:3000])
        return

    log.info("Parsed product:")
    print(json.dumps({
        "product_name": product.get("product_name"),
        "brand": product.get("brand"),
        "quantity": product.get("quantity"),
        "image_url": product.get("image_url"),
        "categories": product.get("categories"),
        "ingredients_raw": product.get("ingredients_raw", "")[:300],
        "boots_id": product.get("boots_id"),
    }, indent=2))

    if "--save" in sys.argv:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)
        ok = save_product(product, conn)
        conn.close()
        log.info(f"Saved to DB: {ok}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_one()
    else:
        limit = None
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            limit = int(sys.argv[idx + 1])
        main(limit=limit)
