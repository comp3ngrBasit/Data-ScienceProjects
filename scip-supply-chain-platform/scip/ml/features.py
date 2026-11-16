"""Leakage-safe feature engineering for demand forecasting.

`features_at(Y, t, exog, cats)` builds the features for predicting day t from Y[:, :t] only (strict past) plus
exogenous variables known in advance for day t (calendar, holidays, M5 events, weather proxy). The same
function is used for training and for recursive multi-step forecasting.
"""
from __future__ import annotations

import warnings

import holidays
import numpy as np
import pandas as pd

LAG_FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28", "rmean_7", "rmean_28", "rmean_56", "rstd_28",
                "nonzero_rate_28", "trend_ratio", "days_since_sale"]
MIN_HISTORY = 28


def exogenous_features(dates: pd.DatetimeIndex, events: pd.DataFrame | None = None,
                       weather: pd.DataFrame | None = None) -> pd.DataFrame:
    uk = holidays.country_holidays("GB", years=sorted(set(dates.year)))
    ex = pd.DataFrame(index=dates)
    ex["dow"] = dates.weekday
    ex["is_weekend"] = (dates.weekday >= 5).astype(int)
    # NOTE: month / day-of-year features are deliberately excluded: with ~12 months of history the
    # model would memorise calendar dates instead of learning seasonality (verified on the hold-out).
    ex["uk_holiday"] = [int(d in uk) for d in dates.date]
    xmas = pd.to_datetime([f"{y}-12-25" for y in dates.year])
    ex["days_to_xmas"] = np.clip((xmas - dates).days, -30, 120)
    if events is not None:
        ex = ex.join(events[["event_any", "event_religious", "event_national", "event_cultural"]])
    if weather is not None:
        ex = ex.join(weather[["temp_max", "precipitation"]])
    return ex.fillna(0)


def _wmean(Y, a, b):
    a = max(a, 0)
    if b <= a:
        return np.full(Y.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(Y[:, a:b], axis=1)


def features_at(Y: np.ndarray, t: int, exog: np.ndarray, cats: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    lag = lambda k: Y[:, t - k] if t - k >= 0 else np.full(n, np.nan)
    rm7, rm28, rm56 = _wmean(Y, t - 7, t), _wmean(Y, t - 28, t), _wmean(Y, t - 56, t)
    win = Y[:, max(0, t - 28):t]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rstd = np.nanstd(win, axis=1)
        nz = np.nanmean(win > 0, axis=1)
    past = Y[:, max(0, t - 90):t]
    last_sale = np.where((past > 0).any(axis=1), past.shape[1] - 1 - np.argmax((past > 0)[:, ::-1], axis=1), 90)
    lagf = np.column_stack([lag(1), lag(7), lag(14), lag(28), rm7, rm28, rm56, rstd, nz,
                            rm7 / (rm28 + 0.05), last_sale])
    ex = np.repeat(exog[t][None, :], n, axis=0)
    return np.column_stack([lagf, ex, cats.astype(float)])


def build_training(Yn: np.ndarray, cutoff: int, exog: np.ndarray, cats: np.ndarray, start: int = MIN_HISTORY):
    Xs, ys = [], []
    for t in range(start, cutoff):
        y = Yn[:, t]
        ok = ~np.isnan(y) & ~np.isnan(Yn[:, t - MIN_HISTORY])
        if ok.any():
            Xs.append(features_at(Yn, t, exog, cats)[ok])
            ys.append(y[ok])
    X = np.nan_to_num(np.vstack(Xs), nan=0.0)
    return X, np.concatenate(ys)


def feature_names(exog_cols: list[str]) -> list[str]:
    return LAG_FEATURES + list(exog_cols) + ["category"]


# --------------------------------------------------------------------------- direct multi-horizon (weekly)
WEEKS = 13


def _week_exog(exog: np.ndarray, start: int, end: int) -> np.ndarray:
    return exog[start:end].mean(axis=0)


def direct_feature_names(exog_cols: list[str]) -> list[str]:
    return LAG_FEATURES + [f"week_{c}" for c in exog_cols] + ["category", "horizon_week"]


def _direct_row_block(Yn, origin, k, exog, cats):
    state = features_at(Yn, origin, exog, cats)[:, :len(LAG_FEATURES)]
    wk = np.repeat(_week_exog(exog, origin + 7 * (k - 1), origin + 7 * k)[None, :], Yn.shape[0], axis=0)
    return np.column_stack([state, wk, cats.astype(float), np.full(Yn.shape[0], k, float)])


def build_direct_training(Yn: np.ndarray, cutoff: int, exog: np.ndarray, cats: np.ndarray, step: int = 7):
    """Samples (origin o, horizon week k): features = SKU state at o (past only) + calendar of target week k;
    target = mean daily demand in days [o+7(k-1), o+7k). Only targets fully before `cutoff` are used."""
    Xs, ys = [], []
    for o in range(MIN_HISTORY, cutoff - 6, step):
        for k in range(1, WEEKS + 1):
            end = o + 7 * k
            if end > cutoff:
                break
            tgt = Yn[:, end - 7:end]
            ok = ~np.isnan(tgt).any(axis=1) & ~np.isnan(Yn[:, o - MIN_HISTORY])
            if not ok.any():
                continue
            Xs.append(_direct_row_block(Yn, o, k, exog, cats)[ok])
            ys.append(tgt[ok].mean(axis=1))
    return np.nan_to_num(np.vstack(Xs), nan=0.0), np.concatenate(ys)


def direct_features(Yn: np.ndarray, origin: int, horizon: int, exog: np.ndarray, cats: np.ndarray) -> np.ndarray:
    """Feature rows ordered SKU-major then week, for weeks covering `horizon` days after origin."""
    n_weeks = int(np.ceil(horizon / 7))
    blocks = [_direct_row_block(Yn, origin, k, exog, cats) for k in range(1, n_weeks + 1)]
    return np.nan_to_num(np.stack(blocks, axis=1).reshape(Yn.shape[0] * n_weeks, -1), nan=0.0)


def dow_profile(Y: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """(n_sku, 7) share of weekly demand by weekday (falls back to the portfolio profile)."""
    dow = dates.weekday.values
    tot = np.nan_to_num(Y)
    port = np.array([tot[:, dow == d].sum() for d in range(7)], float)
    port = port / port.sum() if port.sum() > 0 else np.full(7, 1 / 7)
    prof = np.array([[tot[i, dow == d].sum() for d in range(7)] for i in range(Y.shape[0])], float)
    sums = prof.sum(axis=1, keepdims=True)
    prof = np.where(sums > 0, prof / np.where(sums > 0, sums, 1), port[None, :])
    return 0.7 * prof + 0.3 * port[None, :]          # shrink towards the portfolio pattern


def weekly_to_daily(weekly_mean: np.ndarray, profile: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    """Spread weekly mean daily demand over days using weekday shares (7 * share = weekday multiplier)."""
    out = np.zeros((weekly_mean.shape[0], len(days)))
    for j, d in enumerate(days):
        out[:, j] = weekly_mean[:, j // 7] * 7 * profile[:, d.weekday()]
    return out
