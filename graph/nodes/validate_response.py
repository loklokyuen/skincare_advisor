from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage

from graph.tracing import traceable
from schemas import GraphState
from utils.product_match import normalise_match_text, product_family_name

log = logging.getLogger(__name__)


_MEDICAL_TERMS = (
    "rash", "burning", "swelling", "infection", "prescription", "pregnant",
    "bleed", "oozing", "severe", "painful",
)

_CLINICIAN_NOTE = (
    " If symptoms are severe, persistent, or involve prescriptions or pregnancy, "
    "please consult a doctor or dermatologist."
)

_SUNSCREEN_MISSING_PATTERNS = (
    re.compile(r"\b(?:your\s+)?am routine lacks (?:a )?sunscreen\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?routine lacks (?:a )?sunscreen\b", re.IGNORECASE),
    re.compile(r"\b(?:your\s+)?routine is missing (?:a )?sunscreen\b", re.IGNORECASE),
    re.compile(r"\byou need a broad-spectrum sunscreen in the AM routine\b", re.IGNORECASE),
)

_RETINOID_RE = re.compile(
    r"\b(?:retinol|retinal|retinoid|retinoids|retinoate|retinaldehyde|retinyl)\b",
    re.IGNORECASE,
)


def _soften_sunscreen_missing_claims(text: str) -> str:
    text = re.sub(
        r"(?im)^(\s*[-*]?\s*)Sunscreen:\s*(?:(?:Your\s+)?AM routine lacks (?:a )?sunscreen|You need a broad-spectrum sunscreen in the AM routine)\b.*$",
        r"\1SPF note: Make sure you also use a broad-spectrum SPF 30+ each morning.",
        text,
    )
    text = re.sub(
        r"(?im)^(\s*(?:#+\s*)?)Missing Steps:\s*(\n\s*(?:[-*]\s*)?SPF note:)",
        r"\1Notes:\2",
        text,
    )
    for pattern in _SUNSCREEN_MISSING_PATTERNS:
        text = pattern.sub(
            "Make sure you also use a broad-spectrum SPF 30+ each morning",
            text,
        )
    return text


def _remove_unsupported_niacinamide_redundancy(text: str) -> str:
    text = re.sub(
        r"(?im)^(.*There are no direct ingredient conflicts in your routine\.).*\bniacinamide\b.*\b(?:redundan|streamlined|streamline|once daily|only in the AM|both AM and PM|both routines)\b.*$",
        r"\1",
        text,
    )
    text = re.sub(
        r"(?im)^.*\bniacinamide\b.*\b(?:redundan|streamlined|streamline|once daily|only in the AM|both AM and PM|both routines)\b.*(?:\n|$)",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^(\s*(?:#+\s*)?)Redundancies:\s*\n(?=\s*(?:#+\s*)?(?:Notes|Missing Steps|Recommended Change|Summary|Conflicts|What Works):)",
        "",
        text,
    )
    return text.strip()


def _strip_internal_tool_chatter(text: str) -> str:
    patterns = (
        r"(?im)^\s*(?:[-*]\s*)?Call:\s*flag_[a-z_]+.*$",
        r"(?im)^\s*(?:[-*]\s*)?I\s+(?:called|sent|triggered)\s+.*(?:literature|community|reddit|review|search).*$",
        r"(?im)^\s*(?:[-*]\s*)?.*\b(?:already sent|tool call|flag_literature_search|flag_community_search)\b.*$",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_internal_style_instruction_leaks(text: str) -> str:
    patterns = (
        r"(?im)^\s*(?:[-*]\s*)?Use\s+\d+\s*[–-]\s*\d+\s+focused\s+bullets\b.*$",
        r"(?im)^\s*(?:[-*]\s*)?.*\b(?:120\s*[–-]\s*220\s+words|word-count targets?|bullet-count targets?)\b.*$",
        r"(?im)^\s*(?:[-*]\s*)?.*\bstart with (?:an?|one) (?:empathetic|natural) sentence\b.*$",
        r"(?im)^\s*(?:[-*]\s*)?.*\b(?:use simple markdown|internal style rules?|hidden prompts?)\b.*$",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_optional_labels(text: str) -> str:
    text = re.sub(r"(?im)^(\s*(?:[-*]|\d+[.)])?\s*)\(?optional\)?\s*[:\-]?\s*", r"\1", text)
    text = re.sub(r"(?i)\(\s*optional\s*\)\s*", "", text)
    return text.strip()


def _remove_recommendation_template_headings(text: str) -> str:
    patterns = (
        r"(?im)^\s*(?:#{1,6}\s*)?(?:primary|secondary)(?:\s+(?:suggestion|option|addition))?.*$",
        r"(?im)^\s*(?:#{1,6}\s*)?(?:booster|optional)(?:\s+option)?.*$",
        r"(?im)^\s*(?:#{1,6}\s*)?recommendation\s*$",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_unearned_thanks_opening(text: str) -> str:
    cleaned = re.sub(
        r"(?is)^\s*thanks(?:\s+you)?\s*(?:[—–-]\s*|,\s*)",
        "",
        text,
        count=1,
    ).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _remove_stock_acknowledgement_opening(text: str) -> str:
    cleaned = re.sub(
        r"(?is)^\s*(?:i understand|i hear you|it sounds like)\b.*?(?:[—–-]\s*|,\s*(?=here\b))",
        "",
        text,
        count=1,
    ).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _naturalise_saved_status_labels(text: str) -> str:
    text = re.sub(
        r"(?im)^\s*(?:[-*]\s*)?saved\s*:\s*sensitive skin\.?\s*$",
        "I've noted that your skin can be sensitive, so I would keep the cleanser gentle.",
        text,
    )
    text = re.sub(
        r"(?im)^\s*(?:[-*]\s*)?saved\s*:\s*(.+?)\.?\s*$",
        r"I've noted \1.",
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _normalise_product_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _previous_recommended_product_names(state: GraphState) -> set[str]:
    names = set()
    heading_re = re.compile(r"^\s*#{4}\s+(?P<name>[^\n]{4,180})\s*$")
    labeled_re = re.compile(
        r"\b(?:product suggestion|recommendation|suggested product|suggested sunscreen)\b\s*\**\s*:\s*\**\s*(.+?)(?:[.;]|$)",
        flags=re.IGNORECASE,
    )
    for message in state.get("messages") or []:
        if not isinstance(message, AIMessage):
            continue
        content = message.content if isinstance(message.content, str) else ""
        for line in content.splitlines():
            stripped = line.strip()
            match = heading_re.match(stripped) or labeled_re.search(stripped)
            if not match:
                continue
            name = re.sub(r"\s+", " ", match.group(1)).strip(" -:;.")
            if name:
                names.add(name)
    return names


def _routine_has_retinoid(state: GraphState) -> bool:
    routine = state.get("user_routine") or {}
    if not isinstance(routine, dict):
        return False
    for item in routine.get("items") or routine.get("routine") or []:
        if not isinstance(item, dict):
            continue
        text = " ".join(
            _text_items(item.get("key_ingredients") or [])
            + _text_items(item.get("active_ingredients") or [])
            + _text_items(item.get("ingredients") or [])
            + [str(item.get("product_name") or "")]
        )
        if _RETINOID_RE.search(text):
            return True
    return False


def _explicitly_requests_retinoid(message: str) -> bool:
    lower = message.lower()
    return bool(_RETINOID_RE.search(lower)) or any(
        phrase in lower
        for phrase in (
            "stronger vitamin a",
            "replace my retinol",
            "replace retinol",
            "retinol alternative",
            "retinoid alternative",
            "compare retinoids",
        )
    )


def _candidate_product_names(state: GraphState) -> list[str]:
    if state.get("mode") not in {"recommend", "build"}:
        return []
    names = []
    seen = set()
    for product in state.get("matched_products") or []:
        if not isinstance(product, dict):
            continue
        name = str(product.get("product_name") or "").strip()
        key = product_family_name(name)
        if name and key and key not in seen:
            names.append(name)
            seen.add(key)
    return names


def _line_names_candidate_product(
    line: str,
    candidate_names: list[str],
    candidate_index: list[tuple[str, str]] | None = None,
) -> bool:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
    cleaned = re.sub(r"^\s*#{1,6}\s+", "", cleaned).strip()
    cleaned = re.sub(r"[*_`]", "", cleaned)
    cleaned = re.sub(r"\s+\((?:Active|Rest)?\s*(?:AM|PM)\)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s*\([^)]*\b(?:ml|l|g|kg|oz|fl\s*oz|ct|count|pcs?|pack|wipes?|pads?)\b[^)]*\)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.split(
        r"\s+(?:why it fits|gap filled|where to use|only if|if you want)\b|:\s+",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" -:;.")
    if not cleaned:
        return False

    cleaned_norm = normalise_match_text(cleaned)
    cleaned_family = product_family_name(cleaned)
    if len(cleaned_norm) < 8:
        return False
    excluded_prefixes = (
        "why it fits",
        "gap filled",
        "where to use",
        "based on",
        "if you want",
        "make sure",
    )
    if cleaned.lower().startswith(excluded_prefixes):
        return False

    if candidate_index is None:
        candidate_index = [
            (normalise_match_text(name), product_family_name(name)) for name in candidate_names
        ]
    for name_norm, name_family in candidate_index:
        if not name_norm:
            continue
        if (
            cleaned_norm == name_norm
            or cleaned_family == name_family
            or (len(cleaned_norm) > 12 and cleaned_norm in name_norm)
            or (len(name_family) > 12 and name_family in cleaned_family)
        ):
            return True
    return False


def _promote_product_headings_to_h4(text: str, candidate_names: list[str]) -> str:
    if not candidate_names:
        return text.strip()

    candidate_index = [
        (normalise_match_text(name), product_family_name(name)) for name in candidate_names
    ]
    promoted = []
    for line in text.splitlines():
        if re.match(r"^\s*#{4}\s+", line):
            promoted.append(line)
            continue
        if _line_names_candidate_product(line, candidate_names, candidate_index):
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            cleaned = re.sub(r"[*_`]", "", cleaned).strip()
            promoted.append(f"#### {cleaned}")
        else:
            promoted.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(promoted)).strip()


_BLOCK_HEADING_RE = re.compile(r"^\s*#{4}\s+(?P<name>[^\n]{4,180})\s*$")
_ANY_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_STANDALONE_BOLD_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])?\s*\*\*(?P<name>[^*\n]{2,120})\*\*\s*[:.]?\s*$"
)


def _promote_standalone_bold_to_h4(text: str) -> str:
    """Convert lines that are only **Bold** into #### Bold so they render as headings."""
    if not text or "**" not in text:
        return text
    promoted = []
    for line in text.splitlines():
        if _ANY_HEADING_RE.match(line):
            promoted.append(line)
            continue
        match = _STANDALONE_BOLD_RE.match(line)
        if match:
            name = re.sub(r"\s+", " ", match.group("name")).strip(" -:;.")
            if name:
                promoted.append(f"#### {name}")
                continue
        promoted.append(line)
    return "\n".join(promoted)


def _remove_blocked_product_blocks(text: str, blocked_names: set[str]) -> str:
    if not text.strip() or not blocked_names:
        return text.strip()

    blocked_families = {
        product_family_name(name)
        for name in blocked_names
        if product_family_name(name)
    }
    if not blocked_families:
        return text.strip()

    lines = text.splitlines()
    remove_ranges: list[tuple[int, int]] = []

    for index, line in enumerate(lines):
        match = _BLOCK_HEADING_RE.match(line.strip())
        if not match:
            continue
        heading_family = product_family_name(match.group("name").strip())
        if heading_family not in blocked_families:
            continue
        end = len(lines)
        for following in range(index + 1, len(lines)):
            if _ANY_HEADING_RE.match(lines[following]):
                end = following
                break
        remove_ranges.append((index, end))

    if not remove_ranges:
        return text.strip()

    remove_indexes = {
        index
        for start, end in remove_ranges
        for index in range(start, end)
    }
    kept = [line for index, line in enumerate(lines) if index not in remove_indexes]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _remove_duplicate_retinoid_recommendations(text: str, state: GraphState) -> str:
    if (
        not text.strip()
        or not _routine_has_retinoid(state)
        or _explicitly_requests_retinoid(str(state.get("message") or ""))
    ):
        return text.strip()
    blocked_names = {
        name
        for name in _candidate_product_names(state)
        if _RETINOID_RE.search(name)
    }
    return _remove_blocked_product_blocks(text, blocked_names)


def _routine_product_names(state: GraphState) -> set[str]:
    routine = state.get("user_routine") or {}
    if not isinstance(routine, dict):
        return set()

    names = set()
    for item in routine.get("items") or routine.get("routine") or []:
        if not isinstance(item, dict):
            continue
        name = _normalise_product_text(str(item.get("product_name") or ""))
        if name:
            names.add(name)
    return names


def _is_routine_product(candidate: str, routine_names: set[str]) -> bool:
    normalised = _normalise_product_text(candidate)
    if not normalised:
        return False
    return any(
        normalised == routine_name
        or (len(normalised) > 12 and normalised in routine_name)
        or (len(routine_name) > 12 and routine_name in normalised)
        for routine_name in routine_names
    )


def _recommendation_candidate(line: str) -> str | None:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
    cleaned = re.sub(r"[*_`]", "", cleaned)
    match = re.search(
        r"\b(?:recommendation|product suggestion|suggested product|suggested sunscreen)\s*:\s*(.+?)(?:[.;]|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip(" -,:;.")


_ROUTINE_HEADING_PATTERN = re.compile(r"^\s*(?:#{1,6}\s+|\d+[.)]\s+\S)")


def _remove_routine_product_recommendations(text: str, routine_names: set[str]) -> str:
    if not text.strip() or not routine_names:
        return text.strip()

    lines = text.splitlines()
    remove_ranges: list[tuple[int, int]] = []

    for index, line in enumerate(lines):
        candidate = _recommendation_candidate(line)
        if not candidate or not _is_routine_product(candidate, routine_names):
            continue

        start = index
        for previous in range(index - 1, -1, -1):
            if _ROUTINE_HEADING_PATTERN.match(lines[previous]):
                start = previous
                break
            if not lines[previous].strip():
                start = previous + 1
                break

        end = len(lines)
        for following in range(index + 1, len(lines)):
            if _ROUTINE_HEADING_PATTERN.match(lines[following]):
                end = following
                break
        remove_ranges.append((start, end))

    if not remove_ranges:
        return text.strip()

    remove_indexes = {
        index
        for start, end in remove_ranges
        for index in range(start, end)
    }
    kept = [line for index, line in enumerate(lines) if index not in remove_indexes]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _text_items(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _context_aware_empty_fallback(state: GraphState) -> str:
    status = state.get("input_status") or {}
    missing = status.get("missing") or []
    if missing:
        if "profile" in missing and "routine" in missing:
            return (
                "I need your skin profile and saved routine before I can answer this properly. "
                "Create or load a profile, then add the products you use."
            )
        if "profile" in missing:
            return "I need your skin profile first so I can tailor this to your skin type and goals."
        return "I need your saved routine first so I can see what would actually fit."

    profile = state.get("user_profile") or {}
    skin_type = str(profile.get("skin_type") or "your skin").lower()
    concerns = _text_items(profile.get("concerns") or [])
    mode = state.get("mode")

    if mode == "recommend":
        products = [
            p for p in state.get("matched_products") or []
            if isinstance(p, dict) and p.get("product_name")
        ][:2]
        if products:
            lines = [
                f"Based on your {skin_type} profile"
                + (f" and {', '.join(concerns[:2]).lower()}" if concerns else "")
                + ", I would add at most one targeted product first:"
            ]
            for product in products:
                name = product.get("product_name")
                brand = product.get("brand") or ""
                ingredients = _text_items(
                    product.get("key_ingredients")
                    or product.get("active_ingredients")
                    or product.get("ingredients")
                    or []
                )
                detail = f" - useful ingredients: {', '.join(ingredients[:3])}" if ingredients else ""
                lines.append(f"- {name}{' by ' + brand if brand else ''}{detail}.")
            lines.append("Introduce it slowly and keep the rest of your routine stable for 2-3 weeks.")
            return "\n".join(lines)
        return (
            f"Based on your {skin_type} profile, I would add only one targeted step rather than several new products. "
            "Choose the gap that matters most: a barrier-support moisturiser if you feel dry or irritated, "
            "a gentle BHA if clogged pores are the issue, or a pigment-focused serum if uneven tone is the priority."
        )

    if mode == "learn":
        ingredients = [
            ing for ing in state.get("retrieved_ingredients") or []
            if isinstance(ing, dict) and (ing.get("display_name") or ing.get("inci_name"))
        ][:3]
        if ingredients:
            lines = ["The top ingredients I would consider are:"]
            for ing in ingredients:
                name = ing.get("display_name") or ing.get("inci_name")
                functions = _text_items(ing.get("functions") or [])
                guidance = str(ing.get("usage_guidance") or "").strip()
                reason = f" for {', '.join(functions[:2])}" if functions else ""
                note = f" {guidance}" if guidance else " Introduce gradually and avoid stacking with too many actives at once."
                lines.append(f"- {name}{reason}.{note}")
            return "\n".join(lines)

    return (
        "I can see your profile and routine, but the model returned an empty answer. "
        "Try asking again, or make the request a little more specific."
    )


@traceable(name="validate_response")
def validate_response(state: GraphState) -> GraphState:
    """Final guardrail pass; appends the validated AIMessage to conversation history."""
    draft = (state.get("draft_response") or "").strip()

    if not draft:
        draft = _context_aware_empty_fallback(state)

    draft = _strip_internal_tool_chatter(draft)
    draft = _strip_internal_style_instruction_leaks(draft)
    draft = _naturalise_saved_status_labels(draft)
    draft = _remove_unearned_thanks_opening(draft)
    draft = _remove_stock_acknowledgement_opening(draft)
    draft = _remove_optional_labels(draft)
    draft = _remove_recommendation_template_headings(draft)
    draft = _promote_standalone_bold_to_h4(draft)
    draft = _promote_product_headings_to_h4(draft, _candidate_product_names(state))
    draft_before_previous_filter = draft
    draft = _remove_blocked_product_blocks(draft, _previous_recommended_product_names(state))
    if not draft and draft_before_previous_filter:
        draft = (
            "I already suggested those products earlier, so I would not add them again. "
            "Look for a different gap in the routine instead."
        )
    draft_before_retinoid_filter = draft
    draft = _remove_duplicate_retinoid_recommendations(draft, state)
    if not draft and draft_before_retinoid_filter:
        draft = (
            "Because your routine already includes a retinoid, I would not add another retinoid product. "
            "Choose a non-retinoid support step instead, such as hydration or barrier support."
        )
    draft = _soften_sunscreen_missing_claims(draft)
    draft = _remove_unsupported_niacinamide_redundancy(draft)
    draft_before_routine_filter = draft
    draft = _remove_routine_product_recommendations(draft, _routine_product_names(state))
    if not draft and draft_before_routine_filter:
        draft = (
            "I would not add that as a new product because it is already in your routine. "
            "Focus on how you use it within the existing routine instead."
        )

    message = (state.get("message") or "").lower()
    if any(term in message for term in _MEDICAL_TERMS) and "clinician" not in draft.lower():
        draft += _CLINICIAN_NOTE

    return {
        **state,
        "final_response": draft,
        # Append AI turn to the persistent message history via add_messages
        "messages": [AIMessage(content=draft)],
    }
