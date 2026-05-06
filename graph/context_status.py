from __future__ import annotations


PROFILE_REQUIRED_MODES = {"analyse", "build", "recommend"}
ROUTINE_REQUIRED_MODES = {"analyse"}
ROUTINE_DEPENDENCY_TERMS = (
    "routine",
    "current",
    "existing",
    "what i use",
    "already use",
    "add to",
)


def has_profile(profile: dict | None) -> bool:
    if not isinstance(profile, dict):
        return False
    return bool(profile.get("skin_type") or profile.get("concerns") or profile.get("goals"))


def routine_items(user_routine: dict | None) -> list[dict]:
    if not isinstance(user_routine, dict):
        return []
    items = user_routine.get("items") or user_routine.get("routine") or []
    return [item for item in items if isinstance(item, dict) and item.get("product_name")]


def has_routine(user_routine: dict | None) -> bool:
    return bool(routine_items(user_routine))


def mode_needs_routine(mode: str | None, message: str = "") -> bool:
    if mode in ROUTINE_REQUIRED_MODES:
        return True
    lower = (message or "").lower()
    return mode == "recommend" and any(term in lower for term in ROUTINE_DEPENDENCY_TERMS)


def missing_inputs(
    mode: str | None,
    profile: dict | None,
    user_routine: dict | None,
    message: str = "",
) -> list[str]:
    missing = []
    if mode in PROFILE_REQUIRED_MODES and not has_profile(profile):
        missing.append("profile")
    if mode_needs_routine(mode, message) and not has_routine(user_routine):
        missing.append("routine")
    return missing


def input_status(
    mode: str | None,
    profile: dict | None,
    user_routine: dict | None,
    message: str = "",
) -> dict:
    missing = missing_inputs(mode, profile, user_routine, message)
    return {
        "has_profile": has_profile(profile),
        "has_routine": has_routine(user_routine),
        "missing": missing,
        "ready": not missing,
    }
