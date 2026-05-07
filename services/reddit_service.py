from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

SKINCARE_SUBREDDITS = [
    "SkincareAddiction",
    "AsianBeauty",
    "tretinoin",
    "30PlusSkinCare",
]

_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 3600

_HEADERS = {"User-Agent": "SkinIQ/1.0 (skincare advisor app; https://github.com/loklokyuen)"}
_GENERIC_QUERY_TOKENS = {
    "about",
    "community",
    "cream",
    "day",
    "does",
    "feedback",
    "moisturiser",
    "moisturizer",
    "night",
    "people",
    "product",
    "question",
    "reddit",
    "review",
    "reviews",
    "routine",
    "say",
    "saying",
    "serum",
    "skin",
    "skincare",
    "what",
}
_PRODUCT_FILLER_TOKENS = {
    "brightening",
    "cleanser",
    "cream",
    "enzyme",
    "facial",
    "gel",
    "lotion",
    "mask",
    "moisturiser",
    "moisturizer",
    "pad",
    "pads",
    "serum",
    "toner",
    "treatment",
    "vitamin",
}
_POSITIVE_WORDS = {
    "amazing",
    "better",
    "calming",
    "clear",
    "cleared",
    "comfortable",
    "effective",
    "favorite",
    "gentle",
    "glowing",
    "good",
    "great",
    "helped",
    "hydrating",
    "like",
    "love",
    "nice",
    "recommend",
    "repurchase",
    "soft",
    "soothing",
    "worked",
}
_NEGATIVE_WORDS = {
    "acne",
    "bad",
    "breakout",
    "breakouts",
    "burn",
    "clogged",
    "dry",
    "expensive",
    "hate",
    "irritated",
    "irritating",
    "not",
    "pilling",
    "sticky",
    "worse",
}
_LOW_SIGNAL_PHRASES = {
    "currently using",
    "looking for",
    "thank you",
    "recommendations",
    "any recommendations",
}
_SKIN_AND_ME_RE = re.compile(r"\bskin\s*(?:\+|and)?\s*me\b", re.IGNORECASE)


def _shorten(text: str, limit: int = 300) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [token for token in tokens if len(token) > 2 and token not in _GENERIC_QUERY_TOKENS]


def _strict_query_tokens(query: str) -> list[str]:
    tokens = _query_tokens(query)
    if len(tokens) < 4:
        return tokens
    strict = [token for token in tokens if token not in _PRODUCT_FILLER_TOKENS]
    return strict or tokens


def _is_skin_and_me_query(query: str) -> bool:
    return bool(_SKIN_AND_ME_RE.search(query))


def _matches_query(query: str, text: str) -> bool:
    if _is_skin_and_me_query(query):
        return bool(_SKIN_AND_ME_RE.search(text))

    tokens = _query_tokens(query)
    if not tokens:
        return False
    lower = text.lower()
    if len(tokens) < 4:
        return any(token in lower for token in tokens)

    strict_tokens = _strict_query_tokens(query)
    brand_token = strict_tokens[0] if strict_tokens else tokens[0]
    brand_present = brand_token in lower
    strict_hits = sum(1 for token in strict_tokens if token in lower)
    broad_hits = sum(1 for token in tokens if token in lower)

    if len(strict_tokens) == 1:
        return brand_present and broad_hits >= 3
    return (brand_present and strict_hits >= 2) or strict_hits >= 3


def _sentence_candidates(*texts: str) -> list[str]:
    candidates = []
    for text in texts:
        if not text or text in {"[removed]", "[deleted]"}:
            continue
        cleaned = " ".join(text.split())
        candidates.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if len(sentence.strip()) > 20
        )
    return candidates


_NEGATION_RE = re.compile(
    r"\b(?:doesn'?t|don'?t|didn'?t|won'?t|isn'?t|aren'?t|wasn'?t|weren'?t|"
    r"can'?t|couldn'?t|hasn'?t|haven'?t|wouldn'?t|shouldn'?t|never|no longer)\b"
)


def _sentence_sentiment(sentence: str) -> str | None:
    lower = sentence.lower()
    # Negation (e.g. "doesn't agree", "won't work") flips any positive signals.
    has_negation = bool(_NEGATION_RE.search(lower))
    positive = sum(1 for word in _POSITIVE_WORDS if re.search(rf"\b{re.escape(word)}\b", lower))
    negative = sum(1 for word in _NEGATIVE_WORDS if re.search(rf"\b{re.escape(word)}\b", lower))
    if has_negation:
        negative += positive
        positive = 0
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return None


def _best_quote(query: str, title: str, selftext: str, comments: list[str]) -> str:
    tokens = _query_tokens(query)
    sentences = _sentence_candidates(selftext, *comments, title)
    if not sentences:
        return ""

    def score(sentence: str) -> tuple[int, int]:
        lower = sentence.lower()
        if not _matches_query(query, sentence):
            return (-1, -1)
        token_score = sum(1 for token in tokens if token in lower)
        sentiment_score = sum(1 for word in _POSITIVE_WORDS | _NEGATIVE_WORDS if word in lower)
        return token_score, sentiment_score

    best = max(sentences, key=score)
    if score(best)[0] < 0:
        return ""
    return _shorten(best, limit=260)


def _best_sentiment_quote(
    query: str,
    title: str,
    selftext: str,
    comments: list[str],
    sentiment: str,
) -> str:
    tokens = _query_tokens(query)
    sentences = _sentence_candidates(selftext, *comments, title)
    scored: list[tuple[tuple[int, int, int], str]] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(phrase in lower for phrase in _LOW_SIGNAL_PHRASES):
            continue
        if _sentence_sentiment(sentence) != sentiment:
            continue
        if not _matches_query(query, sentence):
            continue
        token_score = sum(1 for token in tokens if token in lower)
        if tokens and token_score == 0:
            continue
        sentiment_words = _POSITIVE_WORDS if sentiment == "positive" else _NEGATIVE_WORDS
        sentiment_score = sum(1 for word in sentiment_words if re.search(rf"\b{re.escape(word)}\b", lower))
        scored.append(((token_score, sentiment_score, len(sentence)), sentence))
    if not scored:
        return ""
    scored.sort(key=lambda item: item[0], reverse=True)
    return _shorten(scored[0][1], limit=240)


def _sentiment_label(text: str) -> str:
    lower = text.lower()
    positive = sum(1 for word in _POSITIVE_WORDS if word in lower)
    negative = sum(1 for word in _NEGATIVE_WORDS if word in lower)
    if positive > negative:
        return "mostly positive"
    if negative > positive:
        return "mostly critical"
    return "mixed or unclear"


def _post_summary(title: str, selftext: str, subreddit: str, num_comments: int) -> str:
    if selftext and selftext not in {"[removed]", "[deleted]"}:
        return _shorten(selftext)
    community = f"r/{subreddit}" if subreddit else "Reddit"
    comment_text = f" with {num_comments} comments" if num_comments else ""
    return _shorten(f"{community} discussion: {title}{comment_text}.")


def _relevance_score(query: str, post: dict) -> int:
    if _is_skin_and_me_query(query):
        combined = " ".join(
            str(post.get(key) or "")
            for key in ("title", "selftext", "comments_text", "summary", "quote")
        )
        return 1 if _SKIN_AND_ME_RE.search(combined) else 0

    tokens = _query_tokens(query)
    if not tokens:
        return 0
    combined = " ".join(
        str(post.get(key) or "")
        for key in ("title", "selftext", "comments_text", "summary", "quote")
    )
    if not _matches_query(query, combined):
        return 0
    combined_lower = combined.lower()
    return sum(1 for token in tokens if token in combined_lower)


def _direct_relevance_score(query: str, post: dict) -> int:
    if _is_skin_and_me_query(query):
        direct_text = " ".join(
            str(post.get(key) or "")
            for key in ("title", "selftext", "quote", "positive_quote", "negative_quote")
        )
        return 1 if _SKIN_AND_ME_RE.search(direct_text) else 0

    tokens = _query_tokens(query)
    if not tokens:
        return 0
    direct_text = " ".join(
        str(post.get(key) or "")
        for key in ("title", "selftext", "quote", "positive_quote", "negative_quote")
    ).lower()
    return sum(1 for token in tokens if token in direct_text)


def _cache_get(key: str) -> list[dict] | None:
    entry = _CACHE.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    _CACHE.pop(key, None)
    return None


def _cache_set(key: str, results: list[dict]) -> None:
    _CACHE[key] = (time.time(), results)


def _parse_posts(data: dict, limit: int) -> list[dict]:
    posts = []
    for child in (data.get("data") or {}).get("children") or []:
        post = child.get("data") or {}
        title = post.get("title") or ""
        permalink = post.get("permalink") or ""
        if not title or not permalink:
            continue
        selftext = post.get("selftext") or ""
        subreddit = post.get("subreddit") or ""
        num_comments = post.get("num_comments") or 0
        summary = _post_summary(title, selftext, subreddit, num_comments)
        posts.append({
            "title": title,
            "url": f"https://reddit.com{permalink}",
            "permalink": permalink,
            "score": post.get("score") or 0,
            "num_comments": num_comments,
            "snippet": summary,
            "summary": summary,
            "selftext": selftext,
            "subreddit": subreddit,
        })
        if len(posts) >= limit:
            break
    return posts


_MAX_COMMENT_DEPTH = 8


def _parse_comment_bodies(node: object, limit: int, bodies: list[str], depth: int = 0) -> None:
    if len(bodies) >= limit or depth > _MAX_COMMENT_DEPTH or not isinstance(node, dict):
        return
    kind = node.get("kind")
    data = node.get("data") or {}
    if kind == "t1":
        body = data.get("body") or ""
        if body and body not in {"[removed]", "[deleted]"}:
            bodies.append(body)
    replies = data.get("replies")
    if isinstance(replies, dict):
        for child in ((replies.get("data") or {}).get("children") or []):
            _parse_comment_bodies(child, limit, bodies, depth + 1)


def _fetch_comments(permalink: str, limit: int = 8) -> list[str]:
    try:
        resp = requests.get(
            f"https://www.reddit.com{permalink}.json",
            params={"limit": limit, "sort": "top"},
            headers=_HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    bodies: list[str] = []
    try:
        comments = ((payload[1].get("data") or {}).get("children") or [])
    except Exception:
        return []
    for child in comments:
        _parse_comment_bodies(child, limit, bodies)
        if len(bodies) >= limit:
            break
    return bodies


def _enrich_posts(query: str, posts: list[dict], limit: int) -> list[dict]:
    if not posts:
        return []

    # Fetch all comments in parallel; order is preserved by index.
    with ThreadPoolExecutor(max_workers=min(len(posts), 6)) as pool:
        comment_futures = [
            pool.submit(_fetch_comments, post.get("permalink") or "", 8)
            for post in posts
        ]
        post_comments = []
        for fut in comment_futures:
            try:
                post_comments.append(fut.result())
            except Exception:
                post_comments.append([])

    enriched = []
    for post, comments in zip(posts, post_comments):
        comments_text = " ".join(comments)
        positive_quote = _best_sentiment_quote(
            query,
            post.get("title") or "",
            post.get("selftext") or "",
            comments,
            "positive",
        )
        negative_quote = _best_sentiment_quote(
            query,
            post.get("title") or "",
            post.get("selftext") or "",
            comments,
            "negative",
        )
        quote = positive_quote or negative_quote or _best_quote(
            query,
            post.get("title") or "",
            post.get("selftext") or "",
            comments,
        )
        source_text = " ".join(
            part
            for part in [post.get("selftext") or "", comments_text, quote]
            if part
        )
        summary = _post_summary(
            post.get("title") or "",
            post.get("selftext") or "",
            post.get("subreddit") or "",
            post.get("num_comments") or 0,
        )
        enriched_post = {
            **post,
            "comments_text": comments_text,
            "quote": quote,
            "positive_quote": positive_quote,
            "negative_quote": negative_quote,
            "sentiment": _sentiment_label(source_text or summary),
            "summary": summary,
        }
        relevance = _relevance_score(query, enriched_post)
        direct_relevance = _direct_relevance_score(query, enriched_post)
        if relevance and direct_relevance:
            enriched_post["relevance_score"] = relevance
            enriched.append(enriched_post)

    enriched.sort(
        key=lambda item: item.get("score") or 0,
        reverse=True,
    )
    return enriched[: min(limit, 3)]


def _search_subreddit(subreddit: str, query: str, limit: int = 10, sort: str = "relevance") -> list[dict]:
    try:
        resp = requests.get(
            f"https://www.reddit.com/r/{subreddit}/search.json",
            params={"q": query, "restrict_sr": "true", "sort": sort, "limit": limit},
            headers=_HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        return _parse_posts(resp.json(), limit)
    except Exception:
        return []


def _search_wide(query: str, limit: int = 20, sort: str = "relevance") -> list[dict]:
    try:
        resp = requests.get(
            "https://www.reddit.com/search.json",
            params={"q": query, "sort": sort, "limit": limit, "type": "link"},
            headers=_HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        return _parse_posts(resp.json(), limit)
    except Exception:
        return []


def search(query: str, limit: int = 3) -> list[dict]:
    limit = min(max(1, limit), 3)
    key = f"{query.lower().strip()}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    results: list[dict] = []
    seen_urls: set[str] = set()

    # Only search known skincare subreddits — global Reddit search picks up
    # off-topic high-traffic posts whose comment sections happen to mention skincare words.
    search_tasks = [
        (subreddit, sort)
        for subreddit in SKINCARE_SUBREDDITS
        for sort in ("relevance", "top")
    ]

    with ThreadPoolExecutor(max_workers=len(search_tasks)) as pool:
        futures = [
            pool.submit(_search_subreddit, subreddit, query, 10, sort)
            for subreddit, sort in search_tasks
        ]
        futures.extend(
            pool.submit(_search_wide, query, 20, sort)
            for sort in ("relevance", "top")
        )

        for fut in futures:
            try:
                for post in fut.result():
                    if post["url"] not in seen_urls:
                        seen_urls.add(post["url"])
                        results.append(post)
            except Exception:
                pass

    relevant_results = [post for post in results if _relevance_score(query, post)]
    candidate_pool = relevant_results or results
    candidate_pool.sort(key=lambda post: post.get("score") or 0, reverse=True)
    candidate_pool = candidate_pool[:12]

    results = _enrich_posts(query, candidate_pool, limit)
    _cache_set(key, results)
    return results


def summarize(query: str, posts: list[dict]) -> dict:
    """Build a concise, non-LLM summary from the Reddit results we display."""
    if not posts:
        return {
            "takeaway": f"No clearly relevant Reddit discussions found for {query}.",
            "liked": [],
            "cautions": [],
        }

    liked = [
        post.get("positive_quote")
        for post in posts
        if post.get("positive_quote")
    ][:2]
    cautions = [
        post.get("negative_quote")
        for post in posts
        if post.get("negative_quote")
    ][:2]
    if liked and cautions:
        tone = "mixed"
    elif liked:
        tone = "mostly positive"
    elif cautions:
        tone = "mostly cautious"
    else:
        tone = "limited and mostly thread-level"

    takeaway = (
        f"Reddit signal for {query} is {tone} across {len(posts)} relevant "
        f"thread{'s' if len(posts) != 1 else ''}."
    )
    return {"takeaway": takeaway, "liked": liked, "cautions": cautions}
