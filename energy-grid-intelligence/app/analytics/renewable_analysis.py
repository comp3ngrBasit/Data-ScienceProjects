"""Module 5 - Renewable energy contribution analysis (OPSD, Germany in depth + European comparison).

Definitions (every denominator stated):
  * VRE (variable renewable energy) = wind onshore + wind offshore + solar PV actual generation (MW).
    OPSD time series do not contain hydro, biomass or conventional generation, so "renewable" here
    means VRE only and the complement is *residual load* (load - VRE), i.e. what dispatchable plants,
    other renewables, storage and net imports must cover. It is NOT "non-renewable generation".
  * VRE penetration (%) = sum(VRE energy) / sum(load energy) over the same hours.
  * Solar (wind) share of VRE (%) = solar (wind) energy / VRE energy.
  * Demand-to-VRE ratio = load energy / VRE energy.
  * Capacity factor = generation / installed capacity (OPSD capacity series).
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.seasonal import STL

from app.data import ingestion
from app.data import preprocessing as pp
from app.utils.helpers import save_json, save_table

SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def low_vre_periods(share: pd.Series, thr: float = 0.10, min_hours: int = 24) -> pd.DataFrame:
    """Sustained low-VRE ('shortage') periods: VRE covers < thr of load for >= min_hours consecutive hours."""
    low = share < thr
    grp = (low != low.shift()).cumsum()
    rows = []
    for _, seg in share[low].groupby(grp[low]):
        if len(seg) >= min_hours:
            rows.append({"start": seg.index[0], "end": seg.index[-1], "duration_h": len(seg), "mean_vre_share": seg.mean()})
    return pd.DataFrame(rows)


def run(country: str = "DE") -> dict:
    g = pp.grid_local_hourly(country)
    g["vre_share"] = g["vre_mw"] / g["load_mw"]
    g["season"] = g.index.month.map(SEASON)
    tot = g[["load_mw", "solar_mw", "wind_mw", "wind_onshore_mw", "wind_offshore_mw", "vre_mw"]].sum() / 1e6  # TWh
    kpis = {"period": [str(g.index.min()), str(g.index.max())],
            "total_load_twh": tot["load_mw"], "total_vre_twh": tot["vre_mw"], "total_solar_twh": tot["solar_mw"],
            "total_wind_twh": tot["wind_mw"], "vre_penetration_pct": tot["vre_mw"] / tot["load_mw"] * 100,
            "solar_share_of_vre_pct": tot["solar_mw"] / tot["vre_mw"] * 100,
            "wind_share_of_vre_pct": tot["wind_mw"] / tot["vre_mw"] * 100,
            "solar_share_of_load_pct": tot["solar_mw"] / tot["load_mw"] * 100,
            "wind_share_of_load_pct": tot["wind_mw"] / tot["load_mw"] * 100,
            "demand_to_vre_ratio": tot["load_mw"] / tot["vre_mw"],
            "hourly_vre_share_cv": float(g["vre_share"].std() / g["vre_share"].mean()),
            "vre_hourly_cv": float(g["vre_mw"].std() / g["vre_mw"].mean()),
            "max_hourly_vre_share_pct": float(g["vre_share"].max() * 100),
            "max_hourly_vre_share_time": str(g["vre_share"].idxmax()),
            "hours_vre_exceeds_load": int((g["vre_mw"] >= g["load_mw"]).sum()),
            "hours_vre_share_above_50pct": int((g["vre_share"] >= 0.5).sum()),
            "definitions": __doc__.split("Definitions (every denominator stated):")[1].strip()}
    annual = g.groupby(g.index.year)[["load_mw", "solar_mw", "wind_mw", "vre_mw"]].sum() / 1e6
    annual.columns = ["load_twh", "solar_twh", "wind_twh", "vre_twh"]
    annual["vre_penetration_pct"] = annual["vre_twh"] / annual["load_twh"] * 100
    annual["hours"] = g.groupby(g.index.year).size()
    annual["complete_year"] = annual["hours"] >= 8760
    if "solar_capacity_mw" in g and "wind_capacity_mw" in g:
        cap = g.groupby(g.index.year)[["solar_capacity_mw", "wind_capacity_mw"]].mean()
        annual["solar_capacity_factor_pct"] = g.groupby(g.index.year)["solar_mw"].mean() / cap["solar_capacity_mw"] * 100
        annual["wind_capacity_factor_pct"] = g.groupby(g.index.year)["wind_mw"].mean() / cap["wind_capacity_mw"] * 100
    monthly = g[["load_mw", "solar_mw", "wind_mw", "vre_mw"]].resample("MS").sum() / 1e3  # GWh
    monthly.columns = ["load_gwh", "solar_gwh", "wind_gwh", "vre_gwh"]
    monthly["vre_penetration_pct"] = monthly["vre_gwh"] / monthly["load_gwh"] * 100
    seasonal = g.groupby("season")[["load_mw", "solar_mw", "wind_mw", "vre_share"]].mean()
    diurnal = g.groupby([g["season"], g.index.hour])[["solar_mw", "wind_mw", "load_mw"]].mean().reset_index(names=["season", "hour"])
    daily = g[["load_mw", "solar_mw", "wind_mw", "vre_mw", "residual_load_mw"]].resample("D").sum() / 1e3
    daily.columns = ["load_gwh", "solar_gwh", "wind_gwh", "vre_gwh", "residual_load_gwh"]
    daily["vre_share"] = daily["vre_gwh"] / daily["load_gwh"]
    daily["vre_roll30"] = daily["vre_gwh"].rolling(30, center=True).mean()
    daily["load_roll30"] = daily["load_gwh"].rolling(30, center=True).mean()
    stl = STL(daily["vre_gwh"].asfreq("D").interpolate().resample("W").mean().iloc[1:-1], period=52, robust=True).fit()
    vre_stl = pd.DataFrame({"observed": stl.observed, "trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid})
    # variability: hourly ramps
    ramps = g[["solar_mw", "wind_mw", "vre_mw", "load_mw"]].diff()
    ramp_stats = {c: {"p1": float(ramps[c].quantile(0.01)), "p99": float(ramps[c].quantile(0.99)),
                      "max_up": float(ramps[c].max()), "max_down": float(ramps[c].min()), "std": float(ramps[c].std())}
                  for c in ramps}
    # correlation & regression: daily load vs VRE (does renewable output track demand?)
    corr = {}
    for c in ["solar_gwh", "wind_gwh", "vre_gwh"]:
        r, p = stats.pearsonr(daily["load_gwh"], daily[c])
        rho, ps = stats.spearmanr(daily["load_gwh"], daily[c])
        corr[c] = {"pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": ps}
    X = sm.add_constant(pd.DataFrame({"wind_gwh": daily["wind_gwh"], "solar_gwh": daily["solar_gwh"],
                                      "is_weekend": (daily.index.dayofweek >= 5).astype(float)}))
    ols = sm.OLS(daily["residual_load_gwh"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    reg = {"model": "residual_load_gwh ~ wind_gwh + solar_gwh + is_weekend (OLS, Newey-West SE, 7 lags)",
           "r2": ols.rsquared, "coefficients": {k: {"coef": ols.params[k], "se": ols.bse[k], "p": ols.pvalues[k],
                                                     "ci95": ols.conf_int().loc[k].tolist()} for k in ols.params.index}}
    surplus = g.loc[g["vre_share"] >= 0.6, ["load_mw", "vre_mw", "vre_share"]].copy()
    shortages = low_vre_periods(g["vre_share"])
    kpis.update({"n_low_vre_periods_24h": int(len(shortages)),
                 "longest_low_vre_period_h": int(shortages["duration_h"].max()) if len(shortages) else 0,
                 "n_high_vre_hours_60pct": int(len(surplus)),
                 "low_vre_definition": "VRE < 10 % of load for >= 24 consecutive hours",
                 "high_vre_definition": "VRE >= 60 % of load in an hour"})
    # European comparison: complete calendar years where load, solar and wind are all reported
    cols = ingestion.opsd_columns()
    countries = sorted({c.split("_load_actual")[0] for c in cols if c.endswith("_load_actual_entsoe_transparency")})
    countries = [c for c in countries if "_" not in c or c == "GB_GBN"]
    plan = {}
    for c in countries:
        need = [f"{c}_load_actual_entsoe_transparency", f"{c}_solar_generation_actual"]
        parts = [w for w in (f"{c}_wind_onshore_generation_actual", f"{c}_wind_offshore_generation_actual") if w in cols]
        wind = parts or [w for w in (f"{c}_wind_generation_actual",) if w in cols]
        if all(n in cols for n in need) and wind:
            plan[c] = (need, wind)
    allcols = sorted({x for need, wind in plan.values() for x in need + wind})
    df = ingestion.load_opsd_raw(usecols=["utc_timestamp", *allcols]).loc["2016-01-01":"2019-12-31"]
    rows = []
    for c, (need, wind) in plan.items():
        if df[need[0]].notna().mean() < 0.95 or df[need[1]].notna().mean() < 0.95 or df[wind[0]].notna().mean() < 0.95:
            continue  # incomplete reporting would bias penetration downwards
        ok = df[need + wind].notna().all(axis=1)  # identical hours in numerator and denominator
        load, solar, wnd = df.loc[ok, need[0]].sum(), df.loc[ok, need[1]].sum(), df.loc[ok, wind].sum(axis=1).sum()
        rows.append({"country": c, "load_twh": load / 1e6, "hours_used": int(ok.sum()),
                     "solar_share_of_load_pct": solar / load * 100, "wind_share_of_load_pct": wnd / load * 100,
                     "vre_penetration_pct": (solar + wnd) / load * 100})
    europe = pd.DataFrame(rows).sort_values("vre_penetration_pct", ascending=False)
    for name, tab in [("annual", annual), ("monthly", monthly), ("seasonal", seasonal), ("diurnal", diurnal),
                      ("daily", daily), ("vre_stl_weekly", vre_stl), ("low_vre_periods", shortages),
                      ("high_vre_hours", surplus), ("europe_2016_2019", europe)]:
        save_table(tab, "renewable", name)
    sample = g[["load_mw", "solar_mw", "wind_mw", "vre_mw", "vre_share"]].iloc[::3]
    save_table(sample, "renewable", "hourly_sample")
    res = {"country": country, "kpis": kpis, "ramps_mw_per_h": ramp_stats, "correlation_daily_load_vs_generation": corr,
           "residual_load_regression": reg, "n_countries_compared": int(len(europe))}
    save_json(res, "renewable", "summary")
    return res
