"""Shared HTTP helper with timeout, retries and consistent error handling."""
from __future__ import annotations

import time

import httpx

from scip.core.config import get_settings
from scip.core.logging import get_logger

log = get_logger("http")


class ExternalAPIError(RuntimeError):
    """Raised when an external API is unreachable or returns an error (callers degrade gracefully)."""


def get_json(url: str, params: dict | None = None, retries: int = 2, timeout: float | None = None):
    s = get_settings()
    if not s.enable_external_apis:
        raise ExternalAPIError("External APIs disabled (ENABLE_EXTERNAL_APIS=false)")
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = httpx.get(url, params=params, timeout=timeout or s.http_timeout_seconds, follow_redirects=True)
            if r.status_code >= 500:
                raise ExternalAPIError(f"{url} returned HTTP {r.status_code}")
            if r.status_code >= 400:
                raise ExternalAPIError(f"{url} returned HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        except (httpx.HTTPError, ExternalAPIError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    log.warning("External API call failed: %s (%s)", url, last)
    raise ExternalAPIError(str(last))


def post_json(url: str, payload: dict, timeout: float | None = None) -> int:
    try:
        r = httpx.post(url, json=payload, timeout=timeout or get_settings().http_timeout_seconds)
        return r.status_code
    except httpx.HTTPError as exc:
        raise ExternalAPIError(str(exc)) from exc
