"""Forecast evaluation metrics and residual diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from app.utils.helpers import safe_mape


def point_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[m], yhat[m]
    err = y - yhat
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y - y.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE": safe_mape(y, yhat), "WAPE": float(np.sum(np.abs(err)) / np.sum(np.abs(y)) * 100),
            "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), "Bias": float(np.mean(-err)),
            "n": int(m.sum())}


def interval_metrics(y, lo, hi) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    if not m.any():
        return {"PI_coverage": float("nan"), "PI_width": float("nan")}
    return {"PI_coverage": float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])) * 100),
            "PI_width": float(np.mean(hi[m] - lo[m]))}


def residual_diagnostics(resid: pd.Series, lags: int = 24) -> dict:
    r = pd.Series(resid).dropna()
    lb = acorr_ljungbox(r, lags=[min(lags, len(r) // 5)], return_df=True)
    jb = stats.jarque_bera(r)
    return {"mean": float(r.mean()), "std": float(r.std()), "skew": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)), "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
            "ljung_box_p": float(lb["lb_pvalue"].iloc[0]), "durbin_watson": float(durbin_watson(r)),
            "jarque_bera_p": float(jb.pvalue),
            "interpretation": ("residuals show significant autocorrelation (structure left unmodelled)"
                               if lb["lb_pvalue"].iloc[0] < 0.05 else "no significant residual autocorrelation")}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of ``actual`` vs ``expected`` (quantile bins of expected)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
