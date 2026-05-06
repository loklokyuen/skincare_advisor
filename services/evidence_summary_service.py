from __future__ import annotations

import logging
import re
from functools import lru_cache

from config.openai import load_openai_config, openai_chat_kwargs
from services import literature_service, reddit_service

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_summary_llm(model: str, api_key: str, base_url: str | None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        **openai_chat_kwargs(
            model,
            api_key,
            base_url,
            temperature=0.2,
            timeout=20,
            max_completion_tokens=220,
            reasoning_effort="minimal",
            verbosity="low",
        )
    )


def _shorten(text: str, limit: int = 420) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _paper_sources(query: str, limit: int = 4) -> list[dict]:
    sources = []
    for index, paper in enumerate(literature_service.search(query, limit=limit), start=1):
        sources.append(
            {
                "id": f"L{index}",
                "kind": "literature",
                "title": paper.get("title") or "PubMed source",
                "url": paper.get("url") or "",
                "source": paper.get("source") or "PubMed",
                "year": paper.get("year") or "",
                "evidence": paper.get("summary") or paper.get("snippet") or "",
            }
        )
    return sources


def _community_sources(query: str, limit: int = 3) -> list[dict]:
    sources = []
    for index, post in enumerate(reddit_service.search(query, limit=limit), start=1):
        evidence = (
            post.get("positive_quote")
            or post.get("negative_quote")
            or post.get("quote")
            or post.get("summary")
            or ""
        )
        community = f"r/{post.get('subreddit')}" if post.get("subreddit") else "Reddit"
        sources.append(
            {
                "id": f"C{index}",
                "kind": "community",
                "title": post.get("title") or "Reddit discussion",
                "url": post.get("url") or "",
                "source": community,
                "score": post.get("score") or 0,
                "comments": post.get("num_comments") or 0,
                "sentiment": post.get("sentiment") or "",
                "evidence": evidence,
            }
        )
    return sources


def _looks_like_product_query(query: str) -> bool:
    lower = query.lower()
    tokens = re.findall(r"[a-z0-9]+", lower)
    if len(tokens) >= 5:
        return True
    return bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:ml|oz|%|percent)\b", lower)
        or re.search(
            r"\b(?:serum|moisturi[sz]er|cream|cleanser|toner|emulsion|mask|spf|sunscreen)\b",
            lower,
        )
    )


def _skip_literature_for_same_community_query(
    literature_query: str | None,
    community_query: str | None,
) -> bool:
    if not literature_query or not community_query:
        return False
    if literature_query.strip().lower() != community_query.strip().lower():
        return False
    return _looks_like_product_query(literature_query)


def _source_line(source: dict) -> str:
    details = []
    if source.get("kind") == "literature" and source.get("year"):
        details.append(str(source["year"]))
    if source.get("kind") == "community":
        if source.get("sentiment"):
            details.append(str(source["sentiment"]))
        if source.get("comments"):
            details.append(f"{source['comments']} comments")
    meta = f" ({', '.join(details)})" if details else ""
    return (
        f"{source['id']} | {source.get('kind')} | {source.get('title')} | "
        f"{source.get('url')} | {source.get('source')}{meta}: "
        f"{_shorten(source.get('evidence'), 360)}"
    )


def _fallback_paragraph(literature_query: str | None, community_query: str | None, sources: list[dict]) -> str:
    if not sources:
        return ""

    community_sources = [source for source in sources if source.get("kind") == "community"]
    literature_sources = [source for source in sources if source.get("kind") == "literature"]

    claims = []

    if literature_sources:
        raw_subject = literature_query or "these ingredients"
        ingredient_names = [s.strip() for s in raw_subject.split(",")] if "," in raw_subject else [raw_subject]
        subject_str = " and ".join(f"**{s}**" for s in ingredient_names[:3])
        citations = " ".join(
            f"[[{_shorten(src.get('title') or 'source', 70)}]]({src.get('url') or ''})"
            for src in literature_sources[:4]
            if src.get("url")
        )
        evidence_points = [
            _shorten(src.get("evidence") or "", 120)
            for src in literature_sources[:2]
            if src.get("evidence")
        ]
        evidence_text = f" The retrieved source notes: {' '.join(evidence_points)}" if evidence_points else ""
        tail = f" {citations}" if citations else ""
        claims.append(
            f"Clinical literature found for {subject_str} is supporting context, not proof it will work for everyone."
            f"{evidence_text}{tail}"
        )

    if community_sources:
        community_claims = []
        for source in community_sources[:3]:
            title = _shorten(source.get("title") or "source", 80)
            url = source.get("url") or ""
            if not title or not url:
                continue
            evidence = _shorten(source.get("evidence") or "", 110)
            evidence_text = f": {evidence}" if evidence else ""
            community_claims.append(f"[[{title}]]({url}){evidence_text}")
        if community_claims:
            claims.append(
                "Community experience varies, so treat Reddit as practical context rather than proof.\n\n"
                + "\n".join(f"- {claim}" for claim in community_claims)
            )

    if not claims:
        return ""
    return " ".join(claims)


def _summarize_with_llm(
    literature_query: str | None,
    community_query: str | None,
    sources: list[dict],
) -> str:
    api_key, model, base_url = load_openai_config(
        model_env="OPENAI_SUMMARY_MODEL",
        default_model="gpt-5-mini",
    )
    if not api_key or not sources:
        return ""

    source_text = "\n".join(_source_line(source) for source in sources)
    ingredient_list = literature_query or "none"
    prompt = f"""
Summarize skincare evidence sources in concise Markdown.

Ingredients covered: {ingredient_list}
Community query: {community_query or "none"}

Sources:
{source_text}

Requirements:
- Write 55-105 words in natural language for a skincare user.
- If citing multiple community sources, put each source on its own Markdown bullet line.
- ONLY discuss ingredients listed under "Ingredients covered". Do not mention any other ingredients that appear in the source material, even if prominently featured.
- Cover each ingredient from the list that has sources — name each one explicitly, never use "this ingredient" or "it" without naming it first.
- Explain what the sources mean in practical terms: likely benefits, common complaints, tolerance, who might like it, and what remains uncertain.
- Prefer concrete plain-English phrases like "people seem to like...", "the main caution is...", "this is not proof it will work for everyone".
- Cite each source inline using a markdown link where the display text is the source title in square brackets, e.g. [[Title]](url). Never write "study:", "PubMed", or "discussion:" as part of the citation.
- Write your own complete sentences. Never copy or paraphrase evidence text verbatim from Sources — summarise the meaning instead.
- Distinguish clinical literature from Reddit/community experience.
- Do not invent claims, certainty, study outcomes, or consensus not present in Sources.
""".strip()

    try:
        llm = _get_summary_llm(model, api_key, base_url)
        response = llm.invoke(prompt)
        return str(response.content or "").strip()
    except Exception as exc:
        log.warning("Evidence summary LLM failed: %s", exc)
        return ""


def summarize_evidence(
    *,
    literature_query: str | None = None,
    community_query: str | None = None,
) -> dict:
    """Fetch literature/community sources and return one cited paragraph.

    literature_query may be a comma-separated list of ingredient names (learn mode).
    Papers are fetched for each ingredient individually so the summary is specific.
    """
    sources: list[dict] = []
    if literature_query and not _skip_literature_for_same_community_query(
        literature_query,
        community_query,
    ):
        ingredient_names = [q.strip() for q in literature_query.split(",") if q.strip()]
        if len(ingredient_names) > 1:
            for name in ingredient_names[:3]:
                sources.extend(_paper_sources(name, limit=2))
        else:
            sources.extend(_paper_sources(literature_query))
    if community_query:
        sources.extend(_community_sources(community_query))

    paragraph = _summarize_with_llm(literature_query, community_query, sources)
    if not paragraph:
        paragraph = _fallback_paragraph(literature_query, community_query, sources)

    return {
        "paragraph": paragraph,
        "sources": sources,
        "literature_query": literature_query,
        "community_query": community_query,
    }
