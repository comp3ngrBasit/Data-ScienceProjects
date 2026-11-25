"""Module 2 - Peak load analysis (household and grid)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

from app.data import preprocessing as pp
from app.data.feature_engineering import holiday_flags
from app.utils.helpers import save_json, save_table

SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def peak_window(hourly: pd.Series, width: int = 4) -> tuple[int, int]:
    """Data-driven peak window: the ``width`` consecutive hours with the highest mean load (circular)."""
    prof = hourly.groupby(hourly.index.hour).mean().reindex(range(24)).to_numpy()
    scores = [prof[[(h + i) % 24 for i in range(width)]].mean() for h in range(24)]
    start = int(np.argmax(scores))
    return start, (start + width) % 24


def in_window(hours: np.ndarray, start: int, end: int) -> np.ndarray:
    return (hours >= start) & (hours < end) if start < end else (hours >= start) | (hours < end)


def high_demand_events(hourly: pd.Series, q: float = 0.99) -> tuple[pd.DataFrame, float]:
    thr = float(hourly.quantile(q))
    above = hourly > thr
    grp = (above != above.shift()).cumsum()
    rows = []
    for _, seg in hourly[above].groupby(grp[above]):
        rows.append({"start": seg.index[0], "end": seg.index[-1], "duration_h": len(seg), "max": seg.max(),
                     "mean": seg.mean(), "energy_above_threshold": float((seg - thr).sum()),
                     "weekday": seg.index[0].day_name(), "hour": seg.index[0].hour, "season": SEASON[seg.index[0].month]})
    ev = pd.DataFrame(rows)
    if len(ev):
        ev = ev.sort_values("max", ascending=False).reset_index(drop=True)
    return ev, thr


def extreme_value(daily_peak: pd.Series, q: float = 0.95, run: int = 3) -> dict:
    """Peaks-over-threshold with a Generalised Pareto fit on declustered daily peaks; return levels."""
    s = daily_peak.dropna()
    u = float(s.quantile(q))
    exc = s[s > u]
    # runs declustering: exceedances closer than ``run`` days belong to one cluster -> keep the cluster max
    clusters, cur = [], [exc.index[0]] if len(exc) else []
    for t_prev, t in zip(exc.index[:-1], exc.index[1:]):
        if (t - t_prev).days <= run:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    if cur:
        clusters.append(cur)
    peaks = np.array([exc.loc[c].max() for c in clusters]) - u
    if len(peaks) < 10:
        return {"status": "insufficient exceedances", "n_clusters": len(peaks)}
    xi, _, sigma = stats.genpareto.fit(peaks, floc=0)
    ks = stats.kstest(peaks, "genpareto", args=(xi, 0, sigma))
    years = len(s) / 365.25
    lam = len(peaks) / years  # cluster rate per year
    rl = {}
    for T in (1, 2, 5, 10):
        m = lam * T
        rl[f"{T}y"] = u + (sigma / xi * (m ** xi - 1) if abs(xi) > 1e-6 else sigma * np.log(m))
    # bootstrap CI for the 10-year return level
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(300):
        b = rng.choice(peaks, len(peaks), replace=True)
        try:
            bx, _, bs = stats.genpareto.fit(b, floc=0)
            m = lam * 10
            boots.append(u + (bs / bx * (m ** bx - 1) if abs(bx) > 1e-6 else bs * np.log(m)))
        except Exception:  # noqa: BLE001
            continue
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"threshold": u, "threshold_quantile": q, "n_exceedances": int(len(exc)), "n_clusters": int(len(peaks)),
            "shape_xi": float(xi), "scale_sigma": float(sigma), "ks_p": float(ks.pvalue), "years_observed": years,
            "return_levels": rl, "return_level_10y_ci95": [float(lo), float(hi)], "observed_max": float(s.max()),
            "caveat": "Daily peaks are seasonal and serially dependent; declustering reduces but does not remove "
                      "dependence. Return levels assume a stationary climate/demand regime."}


def analyse(hourly: pd.Series, name: str, country: str, peak_def: tuple[int, int] | None, weekdays_only: bool) -> dict:
    df = pd.DataFrame({"v": hourly})
    df["date"] = df.index.normalize()
    daily = df.groupby("date")["v"].agg(peak="max", mean="mean", total="sum", minimum="min")
    daily["peak_hour"] = df.groupby("date")["v"].idxmax().dt.hour
    daily["par"] = daily["peak"] / daily["mean"]
    daily["load_factor"] = daily["mean"] / daily["peak"]
    daily["is_weekend"] = daily.index.dayofweek >= 5
    daily["season"] = daily.index.month.map(SEASON)
    daily["peak_ma7"] = daily["peak"].rolling(7).mean()
    daily["peak_ma30"] = daily["peak"].rolling(30).mean()
    daily["peak_rollstd30"] = daily["peak"].rolling(30).std()
    weekly = df["v"].resample("W").agg(["max", "mean"]).rename(columns={"max": "peak"})
    weekly["par"] = weekly["peak"] / weekly["mean"]
    monthly = df["v"].resample("MS").agg(["max", "mean", "idxmax"]).rename(columns={"max": "peak", "idxmax": "peak_time"})
    monthly["par"] = monthly["peak"] / monthly["mean"]
    stl = STL(daily["peak"].asfreq("D").interpolate(), period=7, robust=True).fit()
    decomp = pd.DataFrame({"observed": stl.observed, "trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid})

    if peak_def is None:
        start, end = peak_window(hourly)
        definition = f"data-driven: {start:02d}:00-{end:02d}:00 (4 h window with highest mean load)"
    else:
        start, end = peak_def
        definition = f"{start:02d}:00-{end:02d}:00" + (" Mon-Fri (EEX peak-load definition)" if weekdays_only else "")
    hrs = df.index.hour.to_numpy()
    mask = in_window(hrs, start, end)
    if weekdays_only:
        mask &= (df.index.dayofweek < 5) & (holiday_flags(df.index, country) == 0)
    peak_energy, off_energy = df.loc[mask, "v"].sum(), df.loc[~mask, "v"].sum()
    events, thr = high_demand_events(hourly)
    pct = hourly.quantile([0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    dist = np.histogram(hourly, bins=60)
    summary = {
        "series": name,
        "max": {"value": float(hourly.max()), "timestamp": str(hourly.idxmax())},
        "overall_peak_to_average": float(hourly.max() / hourly.mean()),
        "mean_daily_par": float(daily["par"].mean()), "mean_daily_load_factor": float(daily["load_factor"].mean()),
        "percentiles": {f"p{round(k * 100, 1)}": float(v) for k, v in pct.items()},
        "peak_definition": definition,
        "peak_share_of_energy": float(peak_energy / (peak_energy + off_energy)),
        "peak_share_of_hours": float(mask.mean()),
        "peak_intensity_ratio": float((peak_energy / mask.sum()) / (off_energy / (~mask).sum())),
        "most_common_daily_peak_hour": int(daily["peak_hour"].mode().iloc[0]),
        "weekday_mean_daily_peak": float(daily.loc[~daily["is_weekend"], "peak"].mean()),
        "weekend_mean_daily_peak": float(daily.loc[daily["is_weekend"], "peak"].mean()),
        "seasonal_mean_daily_peak": daily.groupby("season")["peak"].mean().to_dict(),
        "high_demand_threshold_p99": thr, "n_high_demand_events": int(len(events)),
        "extreme_value": extreme_value(daily["peak"]),
    }
    save_table(daily, "peak", f"{name}_daily")
    save_table(weekly, "peak", f"{name}_weekly")
    save_table(monthly.assign(peak_time=monthly["peak_time"].astype(str)), "peak", f"{name}_monthly")
    save_table(decomp, "peak", f"{name}_daily_peak_stl")
    save_table(events, "peak", f"{name}_events")
    save_table(daily.groupby(["peak_hour", "is_weekend"]).size().rename("days").reset_index(), "peak", f"{name}_peak_hour_dist")
    save_table(pd.DataFrame({"bin_left": dist[1][:-1], "bin_right": dist[1][1:], "count": dist[0]}), "peak", f"{name}_distribution")
    return summary


def run() -> dict:
    hh = pp.load_household_hourly()["energy_kwh"]
    g = pp.grid_local_hourly("DE")["load_mw"]
    res = {"household": analyse(hh, "household", "FR", None, False),
           "grid": analyse(g, "grid", "DE", (8, 20), True)}
    save_json(res, "peak", "summary")
    return res
