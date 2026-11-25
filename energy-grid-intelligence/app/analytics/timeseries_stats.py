"""Time-series statistical analysis: trend, STL, ACF/PACF, ADF/KPSS, differencing, change points, rolling stats."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from app.data import preprocessing as pp
from app.utils.helpers import save_json, save_table


def stationarity(s: pd.Series, name: str) -> dict:
    s = s.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when p is outside its lookup table; ADF API notice
        adf = adfuller(s, autolag="AIC")
        k = kpss(s, regression="c", nlags="auto")
    adf_stat = adf[0] < adf[4]["5%"]
    kpss_stat = k[1] > 0.05
    verdict = ("stationary" if adf_stat and kpss_stat else
               "non-stationary" if not adf_stat and not kpss_stat else
               "conflicting: ADF rejects a unit root but KPSS rejects level stationarity "
               "(deterministic trend or structural change likely)" if adf_stat else
               "conflicting: neither test rejects (low power)")
    return {"series": name, "adf_stat": adf[0], "adf_p": adf[1], "adf_lags": adf[2], "adf_crit_5": adf[4]["5%"],
            "kpss_stat": k[0], "kpss_p": k[1], "kpss_p_note": "KPSS p-values are bounded to [0.01, 0.10] by lookup table",
            "verdict": verdict}


def trend_test(s: pd.Series) -> dict:
    """Linear trend + Kendall tau (Mann-Kendall) test.

    Note: p-values assume independent observations; serial correlation inflates significance, so they are
    reported as descriptive evidence, not confirmatory tests."""
    s = s.dropna()
    t = np.arange(len(s))
    lr = stats.linregress(t, s.values)
    tau = stats.kendalltau(t, s.values)
    return {"slope_per_step": lr.slope, "slope_p": lr.pvalue, "r2": lr.rvalue ** 2,
            "kendall_tau": tau.statistic, "kendall_p": tau.pvalue}


def change_points(s: pd.Series, penalty: float | None = None, min_size: int = 8) -> pd.DataFrame:
    import ruptures as rpt

    x = s.dropna()
    signal = ((x - x.mean()) / x.std()).values.reshape(-1, 1)
    pen = penalty or 3 * np.log(len(signal))  # BIC-like penalty
    bkps = rpt.Pelt(model="l2", min_size=min_size).fit(signal).predict(pen=pen)
    rows, start = [], 0
    for end in bkps:
        seg = x.iloc[start:end]
        rows.append({"start": seg.index[0], "end": seg.index[-1], "mean": seg.mean(), "std": seg.std(), "n": len(seg)})
        start = end
    return pd.DataFrame(rows)


def analyse(s: pd.Series, name: str, period: int, nlags: int) -> dict:
    s = s.asfreq(s.index.freq or pd.infer_freq(s.index)).interpolate(limit=3).dropna()
    stl = STL(s, period=period, robust=True).fit()
    comp = pd.DataFrame({"observed": s, "trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid})
    var_r = np.var(stl.resid)
    strength_trend = max(0, 1 - var_r / np.var(stl.trend + stl.resid))
    strength_season = max(0, 1 - var_r / np.var(stl.seasonal + stl.resid))
    ac = acf(s, nlags=nlags, fft=True)
    pac = pacf(s, nlags=min(nlags, len(s) // 4), method="ywm")
    conf = 1.96 / np.sqrt(len(s))
    corr = pd.DataFrame({"lag": np.arange(len(ac)), "acf": ac})
    corr["pacf"] = pd.Series(pac).reindex(corr.index)
    corr["conf"] = conf
    diffs = {"level": stationarity(s, name), "diff_1": stationarity(s.diff(), f"{name} (1st diff)"),
             f"seasonal_diff_{period}": stationarity(s.diff(period), f"{name} (seasonal diff {period})")}
    rolling = pd.DataFrame({"value": s, "roll_mean": s.rolling(period * 4).mean(), "roll_std": s.rolling(period * 4).std()})
    return {"stl": comp, "acf": corr, "rolling": rolling,
            "summary": {"series": name, "stl_period": period, "trend_strength": strength_trend,
                        "seasonal_strength": strength_season, "stationarity": diffs, "trend": trend_test(s),
                        "significant_acf_lags": [int(l) for l in np.where(np.abs(ac[1:]) > conf)[0][:10] + 1]}}


def run() -> dict:
    hd = pp.load_household_daily()["energy_kwh"].asfreq("D")
    hh = pp.load_household_hourly()["energy_kwh"].asfreq("h")
    gd = pp.grid_daily("DE")["load_mwh"].asfreq("D")
    results = {}
    for key, s, period, nlags in [("household_daily", hd, 7, 60), ("grid_daily", gd, 7, 60),
                                  ("household_hourly", hh.iloc[-24 * 120:], 24, 24 * 8)]:
        r = analyse(s, key, period, nlags)
        save_table(r["stl"], "timeseries", f"{key}_stl")
        save_table(r["acf"], "timeseries", f"{key}_acf")
        save_table(r["rolling"], "timeseries", f"{key}_rolling")
        results[key] = r["summary"]
    # annual STL on weekly-aggregated series to expose yearly seasonality and long-term trend
    for key, s in [("household_weekly", hd.resample("W").sum()), ("grid_weekly", gd.resample("W").sum())]:
        s = s.iloc[1:-1]
        stl = STL(s, period=52, robust=True).fit()
        save_table(pd.DataFrame({"observed": s, "trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid}),
                   "timeseries", f"{key}_stl")
        results[key] = {"trend_on_deseasonalised": trend_test(s - stl.seasonal),
                        "trend_change_total_pct": float((stl.trend.iloc[-1] / stl.trend.iloc[0] - 1) * 100)}
    for key, s in [("household_daily", hd), ("grid_daily", gd)]:
        cp = change_points(s.resample("W").mean().iloc[1:-1])
        save_table(cp, "timeseries", f"{key}_changepoints")
        results[key]["n_change_points"] = int(len(cp) - 1)
    save_json(results, "timeseries", "summary")
    return results
