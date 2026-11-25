"""Schema and data-quality validation. Produces machine-readable quality reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import polars as pl

HOUSEHOLD_SCHEMA = {
    "timestamp": "datetime",
    "Global_active_power": (0.0, 15.0),  # kW; main breaker is ~ 10 kVA but allow headroom
    "Global_reactive_power": (0.0, 5.0),
    "Voltage": (200.0, 260.0),  # V; outside -> suspicious
    "Global_intensity": (0.0, 60.0),  # A
    "Sub_metering_1": (0.0, 100.0),  # Wh per minute
    "Sub_metering_2": (0.0, 100.0),
    "Sub_metering_3": (0.0, 100.0),
}


class ValidationError(ValueError):
    pass


@dataclass
class QualityReport:
    dataset: str
    n_rows: int
    start: str
    end: str
    expected_rows: int
    duplicate_timestamps: int
    missing_by_column: dict = field(default_factory=dict)
    out_of_range_by_column: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _gap_summary(is_missing: np.ndarray, step_label: str) -> dict:
    """Run-length summary of consecutive missing observations."""
    if not is_missing.any():
        return {"n_gaps": 0, "longest": 0, "unit": step_label, "length_distribution": {}}
    m = is_missing.astype(int)
    diff = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    lengths = ends - starts
    bins = [1, 2, 6, 61, 1441, 10**9]
    labels = ["1", "2-5", "6-60", "61-1440", ">1440"]
    counts = pd.cut(pd.Series(lengths), bins=bins, right=False, labels=labels).value_counts().sort_index()
    return {"n_gaps": int(len(lengths)), "longest": int(lengths.max()), "unit": step_label,
            "length_distribution": {str(k): int(v) for k, v in counts.items()}}


def validate_household(df: pl.DataFrame) -> QualityReport:
    for col in HOUSEHOLD_SCHEMA:
        if col not in df.columns:
            raise ValidationError(f"Missing required column {col}")
    if df["timestamp"].null_count() > 0:
        raise ValidationError(f"{df['timestamp'].null_count()} unparseable timestamps")
    ts = df["timestamp"]
    start, end = ts.min(), ts.max()
    expected = int((end - start).total_seconds() // 60) + 1
    dup = int(df.height - ts.n_unique())
    missing = {c: int(df[c].null_count()) for c in df.columns if c != "timestamp"}
    oor = {}
    for c, rng in HOUSEHOLD_SCHEMA.items():
        if isinstance(rng, tuple):
            s = df[c]
            oor[c] = int(((s < rng[0]) | (s > rng[1])).sum())
    # physical consistency: sub-meters (Wh/min) cannot exceed whole-house energy (kW*1000/60 Wh/min)
    sub = df.select((pl.col("Sub_metering_1") + pl.col("Sub_metering_2") + pl.col("Sub_metering_3")
                     - pl.col("Global_active_power") * 1000 / 60).alias("excess"))["excess"]
    inconsistent = int((sub > 1.0).sum())  # 1 Wh tolerance for rounding of integer sub-meter readings
    # P <= V * I: intensity is reported in 0.2 A steps, so compare against the upper edge of the
    # rounding interval and allow a further 5 % metering tolerance
    pf = df.select((pl.col("Global_active_power") * 1000
                    / (pl.col("Voltage") * (pl.col("Global_intensity") + 0.2))).alias("pf"))["pf"]
    pf_violation = int((pf > 1.05).sum())
    rep = QualityReport(
        dataset="household", n_rows=df.height, start=str(start), end=str(end), expected_rows=expected,
        duplicate_timestamps=dup, missing_by_column=missing, out_of_range_by_column=oor,
        gaps=_gap_summary(df.sort("timestamp")["Global_active_power"].is_null().to_numpy(), "minutes"),
    )
    rep.checks = [
        {"check": "row count equals minute span (no silently dropped minutes)", "passed": df.height == expected,
         "detail": f"{df.height:,} rows vs {expected:,} expected"},
        {"check": "no duplicate timestamps", "passed": dup == 0, "detail": f"{dup} duplicates"},
        {"check": "sub-meters <= whole-house energy", "passed": inconsistent == 0,
         "detail": f"{inconsistent} minutes where sub-meter sum exceeds total by > 1 Wh"},
        {"check": "P <= V*I (0.2 A rounding, 5 % tolerance)", "passed": pf_violation == 0,
         "detail": f"{pf_violation} minutes violate P <= V*I"},
        {"check": "missing values share", "passed": missing["Global_active_power"] / df.height < 0.05,
         "detail": f"{missing['Global_active_power'] / df.height:.3%} of minutes missing"},
    ]
    return rep


def validate_grid(df: pd.DataFrame, columns: list[str], country: str) -> QualityReport:
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"OPSD file lacks columns {missing_cols}")
    idx = df.index
    expected = int((idx.max() - idx.min()) / pd.Timedelta(hours=1)) + 1
    dup = int(idx.duplicated().sum())
    missing = {c: int(df[c].isna().sum()) for c in columns}
    oor = {c: int((df[c] < 0).sum()) for c in columns}
    load_col = next((c for c in columns if "load_actual" in c), columns[0])
    rep = QualityReport(dataset=f"grid_{country}", n_rows=len(df), start=str(idx.min()), end=str(idx.max()),
                        expected_rows=expected, duplicate_timestamps=dup, missing_by_column=missing,
                        out_of_range_by_column=oor, gaps=_gap_summary(df[load_col].isna().to_numpy(), "hours"))
    rep.checks = [
        {"check": "hourly index complete", "passed": len(df) == expected, "detail": f"{len(df)} vs {expected}"},
        {"check": "no duplicate timestamps", "passed": dup == 0, "detail": f"{dup} duplicates"},
        {"check": "no negative generation/load", "passed": sum(oor.values()) == 0, "detail": str(oor)},
        {"check": "load missing share < 2 %", "passed": missing[load_col] / len(df) < 0.02,
         "detail": f"{missing[load_col] / len(df):.3%}"},
    ]
    return rep
