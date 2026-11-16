"""Thin HTTP client: the dashboard talks to the platform ONLY through the REST API."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("SCIP_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY")
TIMEOUT = float(os.environ.get("SCIP_API_TIMEOUT", "120"))


class APIError(RuntimeError):
    pass


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


def get(path: str, **params):
    try:
        r = httpx.get(f"{API_URL}{path}", params={k: v for k, v in params.items() if v is not None},
                      headers=_headers(), timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        raise APIError(f"Cannot reach the SCIP API at {API_URL} ({exc}). Is the backend running?") from exc
    if r.status_code >= 400:
        raise APIError(_detail(r))
    return r.json()


def post(path: str, json: dict | None = None, **params):
    try:
        r = httpx.post(f"{API_URL}{path}", json=json, params={k: v for k, v in params.items() if v is not None},
                       headers=_headers(), timeout=TIMEOUT * 3)
    except httpx.HTTPError as exc:
        raise APIError(f"Cannot reach the SCIP API at {API_URL} ({exc}).") from exc
    if r.status_code >= 400:
        raise APIError(_detail(r))
    return r.json()


def _detail(r) -> str:
    try:
        d = r.json().get("detail")
        return f"API {r.status_code}: {d if isinstance(d, str) else d}"
    except Exception:
        return f"API {r.status_code}: {r.text[:200]}"


@st.cache_data(ttl=30, show_spinner=False)
def cached(path: str, **params):
    return get(path, **params)


def df(path: str, **params) -> pd.DataFrame:
    return pd.DataFrame(cached(path, **params))


def refresh():
    cached.clear()
