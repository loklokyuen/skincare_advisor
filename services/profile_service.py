from __future__ import annotations

import logging
import re
from typing import Any

from psycopg2.extras import Json

from utils.db import run_mutation, run_query

log = logging.getLogger(__name__)

_TABLE_READY = False


def _ensure_table() -> bool:
    """Create the profile table on first use."""
    global _TABLE_READY
    if _TABLE_READY:
        return True

    schema_ok = run_mutation("CREATE SCHEMA IF NOT EXISTS skincare_advisor")
    if not schema_ok:
        return False

    table_ok = run_mutation(
        """
        CREATE TABLE IF NOT EXISTS skincare_advisor.user_profiles (
            user_id TEXT PRIMARY KEY,
            profile JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    _TABLE_READY = table_ok
    return table_ok


def normalize_user_id(raw_user_id: str | None) -> str:
    """Return a stable Cloud SQL key for profile lookup."""
    user_id = re.sub(r"\s+", "_", (raw_user_id or "").strip().lower())
    return user_id or "default"


def profile_exists(user_id: str) -> bool:
    """Return whether a profile ID already exists in Cloud SQL."""
    user_id = normalize_user_id(user_id)
    if not _ensure_table():
        return False

    rows = run_query(
        """
        SELECT 1
        FROM skincare_advisor.user_profiles
        WHERE user_id = %s
        LIMIT 1
        """,
        (user_id,),
    )
    return bool(rows)


def get_user_profile(user_id: str) -> dict[str, Any] | None:
    """Fetch a saved profile from Cloud SQL."""
    user_id = normalize_user_id(user_id)
    if not _ensure_table():
        return None

    rows = run_query(
        """
        SELECT user_id, profile, updated_at
        FROM skincare_advisor.user_profiles
        WHERE user_id = %s
        """,
        (user_id,),
    )
    if not rows:
        return None

    profile = dict(rows[0].get("profile") or {})
    profile["user_id"] = rows[0]["user_id"]
    updated_at = rows[0].get("updated_at")
    profile["profile_updated_at"] = updated_at.isoformat() if updated_at else None
    return profile


def save_user_profile(user_id: str, profile: dict[str, Any]) -> bool:
    """Upsert a profile into Cloud SQL."""
    user_id = normalize_user_id(user_id)
    if not _ensure_table():
        return False

    stored_profile = dict(profile)
    stored_profile["user_id"] = user_id
    stored_profile.pop("profile_updated_at", None)

    return run_mutation(
        """
        INSERT INTO skincare_advisor.user_profiles (user_id, profile)
        VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE
        SET profile = EXCLUDED.profile,
            updated_at = NOW()
        """,
        (user_id, Json(stored_profile)),
    )
