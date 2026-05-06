from __future__ import annotations

import re
from typing import Any

from config.settings import SKIN_CONCERNS, SKIN_GOALS, SKIN_TYPES
from services.profile_service import get_user_profile, normalize_user_id, save_user_profile

def _as_clean_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    cleaned = []
    seen = set()
    for value in values:
        item = " ".join(str(value or "").strip().split())
        key = item.lower()
        if item and key not in seen:
            cleaned.append(item)
            seen.add(key)
    return cleaned


def _merge_list(existing: list[str] | None, additions: list[str]) -> list[str]:
    merged = _as_clean_list(existing)
    seen = {item.lower() for item in merged}
    for item in _as_clean_list(additions):
        if item.lower() not in seen:
            merged.append(item)
            seen.add(item.lower())
    return merged


def _match_known(value: str | None, allowed: list[str]) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.lower().split())
    for candidate in allowed:
        if normalized == " ".join(candidate.lower().split()):
            return candidate
    return None


def _match_many(values: list[str] | None, allowed: list[str]) -> list[str]:
    matched = []
    seen = set()
    for value in _as_clean_list(values):
        candidate = _match_known(value, allowed)
        if candidate and candidate not in seen:
            matched.append(candidate)
            seen.add(candidate)
    return matched


def _implies_sensitive_skin(*values: object) -> bool:
    text = " ".join(
        str(item)
        for value in values
        for item in (
            value if isinstance(value, list) else [value]
        )
        if item
    ).lower()
    return bool(
        re.search(
            r"\b(?:sensitive|sensitised|sensitized|easily irritated|irritation-prone)\b",
            text,
        )
    )


def save_user_key_facts_for_profile(
    current_profile: dict[str, Any],
    *,
    skin_type: str | None = None,
    concerns: list[str] | None = None,
    goals: list[str] | None = None,
    allergens: list[str] | None = None,
    avoid_ingredients: list[str] | None = None,
    preferences: list[str] | None = None,
) -> dict[str, Any]:
    """Persist explicit user-stated skincare facts into the saved profile."""
    user_id = normalize_user_id(current_profile.get("user_id"))
    if user_id == "default":
        return {
            "saved": False,
            "reason": "No saved profile is loaded, so chat facts were not persisted.",
            "profile": current_profile,
            "changed_fields": [],
        }

    saved_profile = get_user_profile(user_id) or {}
    profile = {**saved_profile, **current_profile, "user_id": user_id}
    changed_fields: list[str] = []

    matched_skin_type = _match_known(skin_type, SKIN_TYPES)
    current_skin_type = profile.get("skin_type")
    sensitive_as_concern = (
        matched_skin_type == "Sensitive"
        and current_skin_type
        and current_skin_type != "Sensitive"
    )
    if matched_skin_type and not sensitive_as_concern and current_skin_type != matched_skin_type:
        profile["skin_type"] = matched_skin_type
        changed_fields.append("skin_type")

    matched_concerns = _match_many(concerns, SKIN_CONCERNS)
    if _implies_sensitive_skin(skin_type, concerns, preferences):
        matched_concerns = _merge_list(matched_concerns, ["Sensitivity / Irritation"])
    if matched_concerns:
        merged = _merge_list(profile.get("concerns"), matched_concerns)
        if merged != profile.get("concerns", []):
            profile["concerns"] = merged
            changed_fields.append("concerns")

    matched_goals = _match_many(goals, SKIN_GOALS)
    if matched_goals:
        merged = _merge_list(profile.get("goals"), matched_goals)
        if merged != profile.get("goals", []):
            profile["goals"] = merged
            changed_fields.append("goals")

    changed_fields = sorted(set(changed_fields))
    if not changed_fields:
        return {
            "saved": False,
            "reason": "No new supported profile facts were found.",
            "profile": profile,
            "changed_fields": [],
        }

    saved = save_user_profile(user_id, profile)
    return {
        "saved": saved,
        "reason": "Profile updated." if saved else "Could not save the updated profile.",
        "profile": profile if saved else current_profile,
        "changed_fields": changed_fields if saved else [],
    }
