"""M5 Forecasting (real) — https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

* calendar.csv (provided): daily calendar 2011-01-29 … 2016-06-19 with named events (Sporting, Cultural,
  National, Religious) and SNAP days. Used as event features for demand forecasting.
* sales_train_evaluation.csv / sell_prices.csv (optional, competition login required): if present, one store
  can be loaded as an alternative demand source (not required by the platform).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scip.core.logging import get_logger

log = get_logger("ingest.m5")
SOURCE_ID = "m5"


def load_calendar(folder: Path) -> pd.DataFrame | None:
    hits = list(Path(folder).rglob("calendar*.csv"))
    if not hits:
        return None
    cal = pd.read_csv(hits[0], parse_dates=["date"])
    ev = pd.DataFrame({"date": cal["date"]})
    ev["event_any"] = cal["event_name_1"].notna().astype(int)
    for t in ["National", "Religious", "Cultural", "Sporting"]:
        ev[f"event_{t.lower()}"] = ((cal["event_type_1"] == t) | (cal["event_type_2"] == t)).astype(int)
    ev["event_name"] = cal["event_name_1"]
    return ev


def event_features(folder: Path, dates: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Event flags for `dates` (0 outside the calendar's range); None if the calendar is missing."""
    ev = load_calendar(folder)
    if ev is None:
        return None
    ev = ev.set_index("date").reindex(dates)
    covered = ev["event_any"].notna().mean()
    out = ev.drop(columns="event_name").fillna(0).astype(int)
    out.attrs["coverage"] = float(covered)
    return out


def load_store_sales(folder: Path, store_id: str = "CA_1", max_items: int = 200) -> pd.DataFrame | None:
    folder = Path(folder)
    sales = next(iter(folder.rglob("sales_train_evaluation.csv")), None) or next(iter(folder.rglob("sales_train_validation.csv")), None)
    cal = load_calendar(folder)
    if sales is None or cal is None:
        return None
    df = pd.read_csv(sales)
    df = df[df["store_id"] == store_id].head(max_items)
    long = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name="d", value_name="quantity")
    calendar = pd.read_csv(next(iter(folder.rglob("calendar*.csv"))), usecols=["d", "date"], parse_dates=["date"])
    return long.merge(calendar, on="d")[["date", "item_id", "cat_id", "quantity"]]
