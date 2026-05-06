from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET

import requests

_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 24 * 3600

_HEADERS = {"User-Agent": "SkinIQ/1.0 (skincare advisor app; contact loklokyuen.m@gmail.com)"}
_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_INGREDIENT_ALIASES = {
    "pdrn": ["polydeoxyribonucleotide", "polydeoxyribonucleotides", "pdrn"],
}
_OFF_TOPIC_PUBMED_TERMS = {
    "intern",
    "internship",
    "ostomy",
    "perioperative ostomy",
    "program director",
    "resident",
    "residency",
}
_SKIN_EVIDENCE_TERMS = {
    "acne",
    "aesthetic",
    "barrier",
    "cosmetic",
    "dermatology",
    "dermal",
    "epithelial",
    "facial",
    "photoaging",
    "pigmentation",
    "regenerative",
    "scar",
    "skin",
    "topical",
    "wound",
    "wrinkle",
}


def _shorten(text: str, limit: int = 360) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if len(sentence.strip()) > 35
    ]


def _evidence_sentence(query: str, abstract: str, title: str) -> str:
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 2
    }
    skin_terms = {
        "skin", "topical", "dermatology", "cosmetic", "photoaging",
        "inflammation", "barrier", "wrinkle", "acne", "pigmentation",
        "antioxidant", "irritation", "clinical",
    }
    candidates = _sentences(abstract) or _sentences(title)
    if not candidates:
        return ""

    def score(sentence: str) -> tuple[int, int, int]:
        lower = sentence.lower()
        query_score = sum(1 for token in query_terms if token in lower)
        skin_score = sum(1 for token in skin_terms if token in lower)
        human_score = 1 if any(word in lower for word in ("clinical", "patient", "trial", "human")) else 0
        return query_score, skin_score + human_score, -len(sentence)

    best = max(candidates, key=score)
    return _shorten(best, limit=260)


def _key_point(abstract: str, title: str) -> str:
    if abstract:
        return _shorten(abstract)
    return _shorten(f"PubMed record for: {title}. Open the source to inspect the abstract and publication details.")


def _cache_get(key: str) -> list[dict] | None:
    entry = _CACHE.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    _CACHE.pop(key, None)
    return None


def _cache_set(key: str, results: list[dict]) -> None:
    _CACHE[key] = (time.time(), results)


def _article_text(article: ET.Element, tag: str) -> str:
    node = article.find(f".//{tag}")
    if node is None:
        return ""
    return " ".join(part.strip() for part in node.itertext() if part.strip())


def _article_year(article: ET.Element) -> str:
    for path in (
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
    ):
        node = article.find(path)
        if node is not None and node.text:
            return node.text.strip()
    return ""


def _query_aliases(query: str) -> list[str]:
    clean = _sanitize_pubmed_term(query).lower()
    aliases = _INGREDIENT_ALIASES.get(clean, [clean])
    seen = set()
    return [alias for alias in aliases if alias and not (alias in seen or seen.add(alias))]


def _pubmed_search_term(query: str) -> str:
    aliases = _query_aliases(query)
    ingredient_clause = " OR ".join(f'"{alias}"[Title/Abstract]' for alias in aliases)
    skin_clause = (
        "skin OR dermatology OR dermal OR topical OR cosmetic OR wound OR scar "
        "OR photoaging OR wrinkle OR aesthetic"
    )
    exclusion_clause = (
        '"program director"[Title/Abstract] OR resident[Title/Abstract] OR '
        'residency[Title/Abstract] OR intern[Title/Abstract] OR ostomy[Title/Abstract]'
    )
    return f"({ingredient_clause}) AND ({skin_clause}) NOT ({exclusion_clause})"


def _is_relevant_article(query: str, title: str, abstract: str) -> bool:
    combined = f"{title} {abstract}".lower()
    aliases = _query_aliases(query)
    if not any(alias in combined for alias in aliases):
        return False
    if any(term in combined for term in _OFF_TOPIC_PUBMED_TERMS):
        return False
    return any(term in combined for term in _SKIN_EVIDENCE_TERMS)


def _parse_pubmed_articles(xml_text: str, limit: int, query: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    results = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _article_text(article, "PMID")
        title = html.unescape(_article_text(article, "ArticleTitle"))
        if not pmid or not title:
            continue

        journal = html.unescape(_article_text(article, "Title"))
        abstract = html.unescape(_article_text(article, "AbstractText"))
        if not _is_relevant_article(query, title, abstract):
            continue
        key_point = _evidence_sentence(query, abstract, title) or _key_point(abstract, title)
        results.append(
            {
                "title": title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source": journal,
                "year": _article_year(article),
                "snippet": key_point,
                "summary": key_point,
            }
        )
        if len(results) >= limit:
            break
    return results


def summarize(query: str, papers: list[dict]) -> dict:
    """Build a concise, non-LLM summary of retrieved PubMed records."""
    if not papers:
        return {"takeaway": "", "points": []}

    points = []
    for paper in papers[:3]:
        text = paper.get("summary") or paper.get("snippet") or paper.get("title") or ""
        if not text:
            continue
        year = str(paper.get("year") or "").strip()
        suffix = f" ({year})" if year else ""
        points.append(f"{_shorten(text, 180)}{suffix}")

    query_label = query.strip() or "this ingredient"
    if len(papers) == 1:
        takeaway = f"Found 1 PubMed result for {query_label}; treat it as limited evidence."
    else:
        takeaway = (
            f"Found {len(papers)} PubMed results for {query_label}. "
            "Use these as supporting evidence, not proof that it will suit every routine."
        )
    return {"takeaway": takeaway, "points": points}


def _sanitize_pubmed_term(query: str) -> str:
    """Strip parenthetical qualifiers from ingredient names for PubMed compatibility.

    e.g. "Salicylic acid (BHA)" -> "Salicylic acid"
    """
    return re.sub(r"\s*\([^)]*\)", "", query).strip()


def search(query: str, limit: int = 5) -> list[dict]:
    """Search PubMed for ingredient literature relevant to topical skincare use."""
    clean_query = query.strip()
    if not clean_query:
        return []

    key = f"{clean_query.lower()}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    pubmed_term = _sanitize_pubmed_term(clean_query)
    try:
        search_resp = requests.get(
            f"{_PUBMED_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": _pubmed_search_term(pubmed_term),
                "retmode": "json",
                "retmax": max(1, min(limit, 8)),
                "sort": "relevance",
            },
            headers=_HEADERS,
            timeout=8,
        )
        search_resp.raise_for_status()
        ids = (search_resp.json().get("esearchresult") or {}).get("idlist") or []
        if not ids:
            _cache_set(key, [])
            return []

        fetch_resp = requests.get(
            f"{_PUBMED_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            },
            headers=_HEADERS,
            timeout=8,
        )
        fetch_resp.raise_for_status()
        results = _parse_pubmed_articles(fetch_resp.text, limit, clean_query)
    except Exception:
        results = []

    _cache_set(key, results)
    return results
