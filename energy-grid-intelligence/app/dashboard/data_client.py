"""Data access for the dashboard.

If EGI_API_URL is set, results are fetched from the FastAPI backend (the same contract a React
frontend would use); otherwise they are read from the local result store. Both paths are cached.
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from app.config import API_URL
from app.utils import helpers


@st.cache_data(ttl=600, show_spinner=False)
def doc(module: str, name: str):
    if API_URL:
        try:
            r = requests.get(f"{API_URL}/api/v1/results/{module}/{name}/json", timeout=20)
            return r.json() if r.ok else None
        except requests.RequestException:
            st.toast("API unreachable - falling back to local results")
    return helpers.load_json(module, name)


@st.cache_data(ttl=600, show_spinner=False)
def table(module: str, name: str) -> pd.DataFrame | None:
    if API_URL:
        try:
            r = requests.get(f"{API_URL}/api/v1/results/{module}/{name}", params={"limit": 100_000}, timeout=60)
            if r.ok:
                df = pd.DataFrame(r.json()["rows"])
                for c in df.columns:
                    if df[c].dtype == object and df[c].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}T").all():
                        df[c] = pd.to_datetime(df[c])
                return df
            return None
        except requests.RequestException:
            st.toast("API unreachable - falling back to local results")
    return helpers.load_table(module, name)


@st.cache_data(ttl=600, show_spinner=False)
def processed(name: str) -> pd.DataFrame | None:
    from app.config import PROCESSED_DIR

    p = PROCESSED_DIR / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None
