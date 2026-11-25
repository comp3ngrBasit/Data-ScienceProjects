"""Cleaning: duplicates, range violations, physical inconsistencies, regular time grids."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from app.data.validation import HOUSEHOLD_SCHEMA
from app.utils.logging import get_logger

log = get_logger("cleaning")


def clean_household(df: pl.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a minute-level pandas frame on a complete minute grid plus a cleaning log.

    Timestamps in the UCI file have no zone and are continuous across DST dates, so they are kept
    tz-naive (fixed-offset local time); no shifting is applied.
    """
    log_ = {}
    n0 = df.height
    df = df.unique(subset="timestamp", keep="first", maintain_order=True).sort("timestamp")
    log_["duplicates_removed"] = n0 - df.height
    for c, rng in HOUSEHOLD_SCHEMA.items():
        if isinstance(rng, tuple):
            bad = (pl.col(c) < rng[0]) | (pl.col(c) > rng[1])
            log_[f"out_of_range_set_null__{c}"] = int(df.select(bad.sum()).item() or 0)
            df = df.with_columns(pl.when(bad).then(None).otherwise(pl.col(c)).alias(c))
    pdf = df.to_pandas().set_index("timestamp")
    full = pd.date_range(pdf.index.min(), pdf.index.max(), freq="min")
    log_["minutes_added_to_complete_grid"] = int(len(full) - len(pdf))
    pdf = pdf.reindex(full)
    pdf.index.name = "timestamp"
    # unmetered remainder (Wh per minute) = whole-house energy minus the three sub-meters
    pdf["Sub_metering_other"] = (pdf["Global_active_power"] * 1000 / 60
                                 - pdf[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1, min_count=3))
    neg = pdf["Sub_metering_other"] < 0
    log_["negative_remainder_clipped"] = int(neg.sum())
    pdf.loc[neg, "Sub_metering_other"] = 0.0
    log_["missing_minutes_final"] = int(pdf["Global_active_power"].isna().sum())
    return pdf.astype("float32"), log_


def fill_gaps(s: pd.Series, period: int, short_gap: int = 3, lookback_periods: int = 4) -> tuple[pd.Series, pd.Series]:
    """Impute a regular series.

    Gaps of <= ``short_gap`` steps are linearly interpolated. Longer gaps use the median of the same
    seasonal position (``period`` steps apart) over the previous ``lookback_periods`` periods, falling
    back to following periods only if no past value exists. Returns (filled, imputed_flag).
    """
    s = s.astype(float)
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(grp).transform("sum")
    short = isna & (run_len <= short_gap)
    filled = s.copy()
    interp = s.interpolate(method="linear", limit_area="inside")
    filled[short] = interp[short]
    long_idx = np.where(filled.isna().to_numpy())[0]
    vals = filled.to_numpy().copy()
    for i in long_idx:
        past = [vals[i - k * period] for k in range(1, lookback_periods + 1) if i - k * period >= 0]
        past = [v for v in past if not np.isnan(v)]
        if not past:
            past = [s.iat[i + k * period] for k in range(1, lookback_periods + 1) if i + k * period < len(s)]
            past = [v for v in past if not np.isnan(v)]
        if past:
            vals[i] = float(np.median(past))
    filled = pd.Series(vals, index=s.index, name=s.name)
    return filled, isna
