"""Feature engineering. Lag features are horizon-safe: for an H-step-ahead forecast every lag is >= H,
so no feature ever uses information after the forecast origin (direct multi-horizon strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import SETTINGS


def holiday_flags(index: pd.DatetimeIndex, country: str) -> np.ndarray:
    import holidays

    years = range(index.min().year, index.max().year + 1)
    code = {"GB_GBN": "GB"}.get(country, country)
    try:
        hol = holidays.country_holidays(code, years=years)
    except (KeyError, NotImplementedError):
        return np.zeros(len(index), dtype=int)
    return np.array([d in hol for d in index.date], dtype=int)


def calendar_features(index: pd.DatetimeIndex, country: str = "FR", fourier_k: int = 3) -> pd.DataFrame:
    f = pd.DataFrame(index=index)
    f["hour"] = index.hour
    f["dayofweek"] = index.dayofweek
    f["month"] = index.month
    f["dayofyear"] = index.dayofyear
    f["is_weekend"] = (index.dayofweek >= 5).astype(int)
    f["is_holiday"] = holiday_flags(index, country)
    f["is_offday"] = ((f["is_weekend"] + f["is_holiday"]) > 0).astype(int)
    doy = 2 * np.pi * index.dayofyear / 365.25
    hod = 2 * np.pi * index.hour / 24
    for k in range(1, fourier_k + 1):
        f[f"year_sin{k}"], f[f"year_cos{k}"] = np.sin(k * doy), np.cos(k * doy)
    for k in range(1, 3):
        f[f"day_sin{k}"], f[f"day_cos{k}"] = np.sin(k * hod), np.cos(k * hod)
    return f


def safe_lags(horizon: int, season: int = 24, max_lag: int = 336) -> list[int]:
    if season == 1:  # daily series: the most recent admissible week + same day last year
        return sorted({horizon + i for i in range(7)} | {364} if horizon <= 364 else {horizon + i for i in range(7)})
    lags = {horizon, horizon + 1, horizon + 2}
    lags |= {m for m in range(season, max_lag + 1, season) if m >= horizon}
    lags |= {l for l in (7 * season, 14 * season) if l >= horizon}
    return sorted(l for l in lags if l <= max(max_lag, horizon + 2))


def lag_features(y: pd.Series, horizon: int, season: int = 24) -> pd.DataFrame:
    f = pd.DataFrame(index=y.index)
    for l in safe_lags(horizon, season, max_lag=max(14 * season, horizon + 2)):
        f[f"lag_{l}"] = y.shift(l)
    base = y.shift(horizon)
    for w in ((season, 7 * season) if season > 1 else (7, 28)):
        f[f"roll_mean_{w}"] = base.rolling(w, min_periods=w // 2).mean()
        f[f"roll_std_{w}"] = base.rolling(w, min_periods=w // 2).std()
    return f


def degree_days(temp_c: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"hdd": np.clip(SETTINGS.hdd_base_c - temp_c, 0, None),
                         "cdd": np.clip(temp_c - SETTINGS.cdd_base_c, 0, None)}, index=temp_c.index)


def supervised_frame(y: pd.Series, horizon: int, country: str, season: int = 24,
                     exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Target + calendar + horizon-safe lags (+ optional exogenous drivers known at target time)."""
    X = pd.concat([calendar_features(y.index, country), lag_features(y, horizon, season)], axis=1)
    if season == 1:  # daily series: hour features are constant
        X = X.drop(columns=[c for c in X.columns if c.startswith("day_") or c == "hour"])
    if exog is not None:
        X = X.join(exog, how="left")
    X["y"] = y
    return X
