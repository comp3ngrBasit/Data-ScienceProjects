"""Exploratory consumption analytics (household and grid)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import preprocessing as pp
from app.data.feature_engineering import holiday_flags
from app.utils.helpers import save_json, save_table


def describe(s: pd.Series) -> dict:
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"count": int(s.count()), "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
            "cv": s.std() / s.mean(), "skew": s.skew(), "kurtosis": s.kurt(),
            **{f"p{int(k * 100)}": v for k, v in q.items()}}


def profiles(hourly: pd.Series, country: str) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame({"v": hourly})
    idx = hourly.index
    df["hour"], df["dow"], df["month"], df["year"] = idx.hour, idx.dayofweek, idx.month, idx.year
    df["daytype"] = np.where(holiday_flags(idx, country) == 1, "Holiday",
                             np.where(idx.dayofweek >= 5, "Weekend", "Weekday"))
    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["month"].map(season_map)
    return {
        "hourly_by_daytype": df.groupby(["hour", "daytype"])["v"].agg(["mean", "std", "median"]).reset_index(),
        "hourly_by_season": df.groupby(["hour", "season"])["v"].mean().unstack().reset_index(),
        "heatmap_dow_hour": df.groupby(["dow", "hour"])["v"].mean().unstack(),
        "heatmap_month_hour": df.groupby(["month", "hour"])["v"].mean().unstack(),
        "monthly_by_year": df.groupby(["year", "month"])["v"].sum().unstack(0),
    }


def run() -> dict:
    hh = pp.load_household_hourly()
    hd = pp.load_household_daily()
    out = {"household": {}, "grid": {}}
    out["household"]["hourly_kwh"] = describe(hh["energy_kwh"])
    out["household"]["daily_kwh"] = describe(hd["energy_kwh"])
    out["household"]["voltage_v"] = describe(hh["voltage_v"])
    out["household"]["total_kwh"] = float(hh["energy_kwh"].sum())
    sub_cols = ["sub_kitchen_kwh", "sub_laundry_kwh", "sub_water_heater_ac_kwh", "sub_other_kwh"]
    sub_tot = hh[sub_cols].sum()
    out["household"]["submeter_share"] = (sub_tot / sub_tot.sum()).to_dict()
    for name, tab in profiles(hh["energy_kwh"], "FR").items():
        save_table(tab, "consumption", f"household_{name}")
    sub_hour = hh.groupby(hh.index.hour)[sub_cols].mean()
    save_table(sub_hour, "consumption", "household_submeter_by_hour")
    save_table(hd, "consumption", "household_daily")
    hh_w = hh[["energy_kwh"]].resample("W").sum()
    save_table(hh_w, "consumption", "household_weekly")

    g = pp.grid_local_hourly("DE")
    gd = pp.grid_daily("DE")
    out["grid"]["hourly_load_mw"] = describe(g["load_mw"])
    out["grid"]["daily_load_mwh"] = describe(gd["load_mwh"])
    out["grid"]["total_load_twh"] = float(g["load_mw"].sum() / 1e6)
    for name, tab in profiles(g["load_mw"], "DE").items():
        save_table(tab, "consumption", f"grid_{name}")
    save_table(gd, "consumption", "grid_daily")
    save_json(out, "consumption", "summary")
    return out
