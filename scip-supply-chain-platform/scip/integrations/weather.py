"""Open-Meteo integration (https://open-meteo.com) for weather demand factors.

* History: archive API (daily max/min temperature, precipitation) cached in `weather_daily`.
* Forward-looking features: the 16-day forecast API when the planning date is "today"; for dates beyond the
  forecast range (or for historical as-of dates) the same calendar day of the previous year is used as a
  climatology proxy — never actual future observations (that would leak information).
If the API is unreachable, weather features are simply omitted and the run records that fact.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from scip.core.config import get_settings
from scip.core.db import read_sql, write_frame
from scip.core.logging import get_logger
from scip.integrations.http import ExternalAPIError, get_json

log = get_logger("weather")

LOCATIONS = {"london": (51.5072, -0.1276), "sao_paulo": (-23.5505, -46.6333)}
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum"


def _parse(data: dict, key: str, source: str) -> pd.DataFrame:
    d = data.get("daily") or {}
    if not d.get("time"):
        return pd.DataFrame()
    return pd.DataFrame({"location_key": key, "date": pd.to_datetime(d["time"]).date,
                         "temp_max": d.get("temperature_2m_max"), "temp_min": d.get("temperature_2m_min"),
                         "precipitation": d.get("precipitation_sum"), "source": source})


def fetch_history(key: str, start: date, end: date) -> pd.DataFrame:
    lat, lng = LOCATIONS[key]
    data = get_json(get_settings().open_meteo_archive_url, params={
        "latitude": lat, "longitude": lng, "start_date": str(start), "end_date": str(end),
        "daily": DAILY_VARS, "timezone": "UTC"})
    return _parse(data, key, "open-meteo-archive")


def fetch_forecast(key: str, days: int = 16) -> pd.DataFrame:
    lat, lng = LOCATIONS[key]
    data = get_json(get_settings().open_meteo_url, params={
        "latitude": lat, "longitude": lng, "daily": DAILY_VARS, "forecast_days": min(days, 16), "timezone": "UTC"})
    return _parse(data, key, "open-meteo-forecast")


def sync_history(key: str, start: date, end: date, engine=None) -> dict:
    """Ensure cached history covers [start, end]. Returns a status dict (never raises on API failure)."""
    cached = read_sql("SELECT COUNT(*) AS n FROM weather_daily WHERE location_key=:k AND date BETWEEN :a AND :b",
                      {"k": key, "a": start, "b": end}, engine=engine)["n"].iat[0]
    expected = (end - start).days + 1
    if cached >= expected:
        return {"status": "cached", "rows": int(cached)}
    try:
        df = fetch_history(key, start, end)
    except ExternalAPIError as exc:
        return {"status": "unavailable", "error": str(exc), "rows": int(cached)}
    n = write_frame(df, "weather_daily", engine=engine,
                    replace_where=f"location_key='{key}' AND date BETWEEN '{start}' AND '{end}'")
    log.info("Weather %s: stored %d days", key, n)
    return {"status": "fetched", "rows": n}


def weather_features(key: str, dates: pd.DatetimeIndex, as_of: pd.Timestamp, engine=None) -> pd.DataFrame | None:
    """Daily weather features for `dates`. Dates <= as_of use observations; later dates use the previous
    year's same calendar day (climatology proxy). Returns None if no weather is cached."""
    w = read_sql("SELECT date, temp_max, precipitation FROM weather_daily WHERE location_key=:k ORDER BY date",
                 {"k": key}, engine=engine, parse_dates=["date"])
    if w.empty:
        return None
    w = w.set_index("date")
    out = pd.DataFrame(index=dates)
    obs = w.reindex(dates)
    proxy = w.reindex(dates - pd.DateOffset(years=1))
    proxy.index = dates
    future = dates > as_of
    out["temp_max"] = obs["temp_max"].where(~future, proxy["temp_max"])
    out["precipitation"] = obs["precipitation"].where(~future, proxy["precipitation"])
    if out.isna().mean().max() > 0.2:
        return None
    return out.ffill().bfill()
