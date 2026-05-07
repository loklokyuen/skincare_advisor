from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from copy import deepcopy
import hashlib
import json
import logging
import re
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

_TTL_SECONDS = 30 * 60
_MAX_ENTRIES = 24
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="advisor-prewarm")
_lock = threading.Lock()
_cache: dict[tuple[str, str, str], dict] = {}
_timestamps: dict[tuple[str, str, str], float] = {}
_inflight: dict[tuple[str, str, str], Future] = {}

_GENERIC_RECOMMEND_RE = re.compile(
    r"\b(?:what should i add|add to my routine|targeted additions?|next step|"
    r"recommend(?:ation)?|suggest(?:ion)?)\b",
    re.IGNORECASE,
)
_GENERIC_INGREDIENT_RE = re.compile(
    r"\b(?:ingredients? should i try|what ingredients?|suggest.*ingredients?|"
    r"top .*ingredients?|ingredients? for my skin)\b",
    re.IGNORECASE,
)
_SPECIFIC_PRODUCT_RE = re.compile(
    r"\b(?:sunscreen|spf|cleanser|moisturi[sz]er|serum|toner|retinol|retinal|"
    r"vitamin c|azelaic|salicylic|glycolic|lactic|benzoyl|niacinamide|"
    r"the ordinary|cerave|la roche|paula'?s choice|skin \+ me)\b",
    re.IGNORECASE,
)


def _stable_json(value: object) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _context_signature(user_profile: dict | None, user_routine: dict | None) -> str:
    payload = {
        "profile": user_profile or {},
        "routine": user_routine or {},
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _normalise_prompt(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (prompt or "").lower()).strip()


def _cache_key(
    prompt: str,
    mode: str | None,
    user_profile: dict | None,
    user_routine: dict | None,
) -> tuple[str, str, str]:
    return (
        _context_signature(user_profile, user_routine),
        mode or "",
        _normalise_prompt(prompt),
    )


def _is_generic_similar(message: str, mode: str | None) -> bool:
    if mode == "recommend":
        return bool(_GENERIC_RECOMMEND_RE.search(message)) and not _SPECIFIC_PRODUCT_RE.search(message)
    if mode == "learn":
        return bool(_GENERIC_INGREDIENT_RE.search(message)) and not _SPECIFIC_PRODUCT_RE.search(message)
    return False


def _similarity(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if len(token) > 2}
    right_tokens = {token for token in right.split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _prune_locked(now: float) -> None:
    expired = [key for key, ts in _timestamps.items() if now - ts > _TTL_SECONDS]
    for key in expired:
        _cache.pop(key, None)
        _timestamps.pop(key, None)

    if len(_cache) <= _MAX_ENTRIES:
        return

    oldest = sorted(_timestamps.items(), key=lambda item: item[1])
    for key, _ts in oldest[: len(_cache) - _MAX_ENTRIES]:
        _cache.pop(key, None)
        _timestamps.pop(key, None)


def store_prepared_answer(
    prompt: str,
    mode: str | None,
    user_profile: dict | None,
    user_routine: dict | None,
    result: dict,
) -> None:
    key = _cache_key(prompt, mode, user_profile, user_routine)
    value = deepcopy(result)
    value["prepared_answer"] = True
    now = time.monotonic()
    with _lock:
        _cache[key] = value
        _timestamps[key] = now
        _prune_locked(now)


def _best_matching_entry_locked(
    entries: dict[tuple[str, str, str], object],
    key: tuple[str, str, str],
) -> object | None:
    match = _best_matching_entry_with_key_locked(entries, key)
    return match[1] if match else None


def _best_matching_entry_with_key_locked(
    entries: dict[tuple[str, str, str], object],
    key: tuple[str, str, str],
) -> tuple[tuple[str, str, str], object] | None:
    signature, mode_key, prompt_key = key
    best: tuple[float, tuple[str, str, str], object] | None = None
    for entry_key, value in entries.items():
        entry_signature, entry_mode, entry_prompt = entry_key
        if entry_signature != signature or entry_mode != mode_key:
            continue
        score = _similarity(prompt_key, entry_prompt)
        if score >= 0.12 and (best is None or score > best[0]):
            best = (score, entry_key, value)
    return (best[1], best[2]) if best else None


def _cache_result(key: tuple[str, str, str], result: dict) -> dict:
    value = deepcopy(result)
    value["prepared_answer"] = True
    now = time.monotonic()
    with _lock:
        _cache[key] = value
        _timestamps[key] = now
        _prune_locked(now)
    return deepcopy(value)


def get_prepared_answer(
    message: str,
    mode: str | None,
    user_profile: dict | None,
    user_routine: dict | None,
    wait_timeout: float = 0.0,
) -> dict | None:
    """Return prepared answer if available, popping it so later turns regenerate."""
    key = _cache_key(message, mode, user_profile, user_routine)
    now = time.monotonic()
    future: Future | None = None
    future_key: tuple[str, str, str] | None = None
    with _lock:
        _prune_locked(now)
        exact = _cache.get(key)
        if exact is not None:
            value = deepcopy(exact)
            _cache.pop(key, None)
            _timestamps.pop(key, None)
            return value

        if not _is_generic_similar(message, mode):
            return None

        match = _best_matching_entry_with_key_locked(_cache, key)
        if match is not None:
            matched_key, cached = match
            value = deepcopy(cached)
            _cache.pop(matched_key, None)
            _timestamps.pop(matched_key, None)
            return value

        exact_future = _inflight.get(key)
        if exact_future is not None:
            future = exact_future
            future_key = key
        elif wait_timeout > 0:
            inflight_match = _best_matching_entry_with_key_locked(_inflight, key)
            if inflight_match is not None:
                future_key, future = inflight_match

    if future is None or future_key is None:
        return None

    try:
        result = future.result(timeout=max(wait_timeout, 0.0))
    except TimeoutError:
        return None
    except Exception as exc:
        log.debug("Advisor prewarm result unavailable for %s: %s", future_key[1], exc, exc_info=True)
        return None
    # Result returned via _finish_prepare_job which already cached it; deliver
    # and pop so the next turn re-runs the graph instead of replaying.
    cached_value = _cache_result(future_key, result)
    with _lock:
        _cache.pop(future_key, None)
        _timestamps.pop(future_key, None)
    return cached_value


def prepare_advisor_answers(
    *,
    prompts: list[dict],
    user_profile: dict | None,
    user_routine: dict | None,
    thread_id: str,
    runner: Callable[..., dict] | None = None,
) -> None:
    """Start background answer generation for generic shortcut prompts.

    `runner` is injectable for tests. In production we lazy-import graph.graph to
    avoid importing the compiled graph while this module itself is imported.
    """
    profile = deepcopy(user_profile or {})
    routine = deepcopy(user_routine or {})

    for item in prompts:
        prompt = str(item.get("prompt") or "").strip()
        mode = str(item.get("mode") or "").strip() or None
        if mode not in {"recommend", "learn", "build", "analyse"} or not prompt:
            continue
        key = _cache_key(prompt, mode, profile, routine)
        with _lock:
            if key in _cache or key in _inflight:
                continue

            future = _executor.submit(
                _run_prepare_job,
                prompt,
                mode,
                profile,
                routine,
                f"{thread_id}:prewarm:{mode}",
                runner,
            )
            _inflight[key] = future
            future.add_done_callback(lambda _future, cache_key=key: _finish_prepare_job(cache_key, _future))


def _run_prepare_job(
    prompt: str,
    mode: str,
    user_profile: dict,
    user_routine: dict,
    thread_id: str,
    runner: Callable[..., dict] | None,
) -> dict:
    if runner is None:
        from graph.graph import run_advisor_with_progress

        runner = run_advisor_with_progress

    return runner(
        prompt,
        user_profile=user_profile,
        user_routine=user_routine,
        thread_id=thread_id,
        explicit_mode=mode,
        allow_prepared_answer=False,
    )


def _finish_prepare_job(key: tuple[str, str, str], future: Future) -> None:
    with _lock:
        _inflight.pop(key, None)

    try:
        result = future.result()
    except Exception as exc:
        log.debug("Advisor prewarm failed for %s: %s", key[1], exc, exc_info=True)
        return

    value = deepcopy(result)
    value["prepared_answer"] = True
    now = time.monotonic()
    with _lock:
        _cache[key] = value
        _timestamps[key] = now
        _prune_locked(now)
