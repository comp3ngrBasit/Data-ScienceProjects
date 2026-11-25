"""Reusable preprocessing pipelines: resampling, imputation, timezone handling, persistence."""
from __future__ import annotations

import pandas as pd

from app.config import COUNTRY_TZ, PROCESSED_DIR, SETTINGS
from app.data import cleaning, ingestion, validation
from app.utils.helpers import save_json
from app.utils.logging import get_logger

log = get_logger("preprocessing")

HH_HOURLY = PROCESSED_DIR / "household_hourly.parquet"
HH_DAILY = PROCESSED_DIR / "household_daily.parquet"
HH_MINUTE = PROCESSED_DIR / "household_minute.parquet"


def household_to_hourly(minute: pd.DataFrame, min_coverage: float = SETTINGS.min_hour_coverage) -> pd.DataFrame:
    """Aggregate minute data to hourly energy (kWh) with coverage-aware scaling.

    An hour with >= ``min_coverage`` valid minutes is scaled to a full hour (mean power x 1 h);
    hours below the threshold become missing and are imputed afterwards (flagged).
    """
    valid = minute["Global_active_power"].notna()
    g = minute.resample("h")
    coverage = valid.resample("h").mean()
    out = pd.DataFrame(index=coverage.index)
    out["energy_kwh"] = g["Global_active_power"].mean()  # mean kW over an hour = kWh in that hour
    out["reactive_kvarh"] = g["Global_reactive_power"].mean()
    out["voltage_v"] = g["Voltage"].mean()
    out["intensity_a"] = g["Global_intensity"].mean()
    out["peak_kw"] = g["Global_active_power"].max()
    for i, name in [(1, "kitchen"), (2, "laundry"), (3, "water_heater_ac")]:
        out[f"sub_{name}_kwh"] = g[f"Sub_metering_{i}"].mean() * 60 / 1000
    out["sub_other_kwh"] = g["Sub_metering_other"].mean() * 60 / 1000
    out["coverage"] = coverage
    low = coverage < min_coverage
    out.loc[low, out.columns.difference(["coverage"])] = float("nan")
    out["imputed"] = False
    for col in out.columns.difference(["coverage", "imputed"]):
        filled, flag = cleaning.fill_gaps(out[col], period=168)
        out[col] = filled
        if col == "energy_kwh":
            out["imputed"] = flag
    out.index.name = "timestamp"
    return out


def build_household(force: bool = False) -> dict:
    if HH_HOURLY.exists() and HH_DAILY.exists() and not force:
        log.info("Household processed data cached")
        return {"cached": True}
    raw = ingestion.load_household_raw()
    report = validation.validate_household(raw)
    minute, clog = cleaning.clean_household(raw)
    del raw
    hourly = household_to_hourly(minute)
    # trim partial first/last day so daily aggregates are complete
    first = hourly.index.normalize()[0] + pd.Timedelta(days=1)
    last = hourly.index.normalize()[-1]
    hourly = hourly.loc[first: last - pd.Timedelta(hours=1)]
    daily = hourly.resample("D").agg({"energy_kwh": "sum", "peak_kw": "max", "voltage_v": "mean",
                                      "sub_kitchen_kwh": "sum", "sub_laundry_kwh": "sum",
                                      "sub_water_heater_ac_kwh": "sum", "sub_other_kwh": "sum", "imputed": "mean"})
    daily = daily.rename(columns={"imputed": "imputed_share"})
    minute.loc[first: last - pd.Timedelta(minutes=1), ["Global_active_power", "Voltage", "Global_intensity"]].to_parquet(HH_MINUTE)
    hourly.to_parquet(HH_HOURLY)
    daily.to_parquet(HH_DAILY)
    summary = {"quality_report": report.to_dict(), "cleaning_log": clog,
               "hourly_rows": len(hourly), "daily_rows": len(daily),
               "hours_imputed": int(hourly["imputed"].sum()),
               "hours_imputed_share": float(hourly["imputed"].mean()),
               "period": [str(hourly.index.min()), str(hourly.index.max())],
               "timezone_note": ("UCI timestamps carry no zone; the minute sequence is continuous across DST change dates "
                                  "(no skipped or repeated hour), consistent with a fixed-offset logger clock. "
                                  "Timestamps are kept tz-naive and treated as local time.")}
    save_json(summary, "data_quality", "household")
    log.info("Household: %d hourly rows, %.2f%% imputed", len(hourly), 100 * hourly["imputed"].mean())
    return summary


def grid_columns(country: str) -> dict[str, str]:
    c = country
    return {"load_mw": f"{c}_load_actual_entsoe_transparency",
            "tso_forecast_mw": f"{c}_load_forecast_entsoe_transparency",
            "solar_mw": f"{c}_solar_generation_actual",
            "wind_onshore_mw": f"{c}_wind_onshore_generation_actual",
            "wind_offshore_mw": f"{c}_wind_offshore_generation_actual",
            "solar_capacity_mw": f"{c}_solar_capacity",
            "wind_capacity_mw": f"{c}_wind_capacity"}


def grid_path(country: str):
    return PROCESSED_DIR / f"grid_{country}_hourly.parquet"


def build_grid(country: str, force: bool = False) -> dict:
    out_path = grid_path(country)
    if out_path.exists() and not force:
        return {"cached": True}
    available = set(ingestion.opsd_columns())
    mapping = {k: v for k, v in grid_columns(country).items() if v in available}
    if "load_mw" not in mapping:
        raise validation.ValidationError(f"OPSD has no load series for {country}")
    raw = ingestion.load_opsd_raw(usecols=["utc_timestamp", *mapping.values()])
    raw = raw.rename(columns={v: k for k, v in mapping.items()})
    report = validation.validate_grid(raw, list(mapping.keys()), country)
    # restrict to the span where load is reported
    lv = raw["load_mw"].dropna()
    df = raw.loc[lv.index.min(): lv.index.max()].copy()
    df = df[~df.index.duplicated()].asfreq("h")
    flags = {}
    for col in [c for c in ["load_mw", "tso_forecast_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"] if c in df]:
        df[col], flag = cleaning.fill_gaps(df[col], period=168)
        flags[col] = int(flag.sum())
        if col == "load_mw":
            df["imputed"] = flag
    for col in ["solar_capacity_mw", "wind_capacity_mw"]:
        if col in df:
            df[col] = df[col].ffill().bfill()
    wind_parts = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
    df["wind_mw"] = df[wind_parts].sum(axis=1) if wind_parts else float("nan")
    if "solar_mw" not in df:
        df["solar_mw"] = 0.0
    df["vre_mw"] = df["wind_mw"] + df["solar_mw"]
    df["residual_load_mw"] = df["load_mw"] - df["vre_mw"]
    tz = COUNTRY_TZ.get(country, "UTC")
    df["local_time"] = df.index.tz_convert(tz)
    df.index.name = "utc_timestamp"
    df.to_parquet(out_path)
    summary = {"quality_report": report.to_dict(), "imputed_hours_by_column": flags, "timezone": tz,
               "rows": len(df), "period_utc": [str(df.index.min()), str(df.index.max())],
               "columns_used": mapping}
    save_json(summary, "data_quality", f"grid_{country}")
    return summary


def load_household_hourly() -> pd.DataFrame:
    return pd.read_parquet(HH_HOURLY)


def load_household_daily() -> pd.DataFrame:
    return pd.read_parquet(HH_DAILY)


def load_grid(country: str) -> pd.DataFrame:
    return pd.read_parquet(grid_path(country))


def grid_local_hourly(country: str) -> pd.DataFrame:
    """Grid data indexed by local wall-clock time (for calendar analyses). DST duplicates are averaged."""
    g = load_grid(country)
    num = g.select_dtypes("number")
    num.index = g["local_time"].dt.tz_localize(None)
    num = num.groupby(level=0).mean()
    return num.asfreq("h").interpolate(limit=2)


def grid_daily(country: str) -> pd.DataFrame:
    """Daily energy (MWh) in local days; days are summed from UTC-hourly data mapped to local dates."""
    g = load_grid(country)
    num = g.select_dtypes("number").drop(columns=["imputed"], errors="ignore")
    num["date"] = g["local_time"].dt.date
    d = num.groupby("date").agg({c: "sum" for c in num.columns if c.endswith("_mw") and "capacity" not in c})
    d.columns = [c.replace("_mw", "_mwh") for c in d.columns]
    d["peak_load_mw"] = num.groupby("date")["load_mw"].max()
    d["hours"] = num.groupby("date").size()
    d.index = pd.to_datetime(d.index)
    return d[d["hours"].between(23, 25)]
