"""DataCo Smart Supply Chain (real) —
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Raw: 180,519 order-item lines (2015–2018), 53 columns, latin-1 encoded.
Transformations:
  * personal data columns (email, password, names, street) are dropped at ingestion;
  * aggregated to ONE shipment per Order Id (quantity / sales summed; shipping attributes are per order);
  * promised date = order date + scheduled shipping days; delivered date = order date + real shipping days;
  * label `is_late` = Late_delivery_risk (1 = delivered after schedule); cancelled shipments keep is_late NULL.
`Days for shipping (real)`, `Delivery Status` and `Order Status` describe the outcome and are therefore
never used as model features (leakage) — they are stored only for tracking/analytics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scip.core.logging import get_logger

log = get_logger("ingest.dataco")
SOURCE_ID = "dataco"
PII_COLUMNS = ["Customer Email", "Customer Password", "Customer Fname", "Customer Lname", "Customer Street",
               "Product Image", "Product Description"]


def find_file(folder: Path) -> Path | None:
    folder = Path(folder)
    for p in folder.rglob("*.csv"):
        if p.name.lower().startswith("datacosupplychaindataset"):
            return p
    for p in folder.rglob("*.zip"):
        if "dataco" in p.name.lower():
            return p
    return None


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", compression="infer")
    return df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])


def build(path: Path) -> dict:
    raw = read_raw(path)
    stats = {"raw_rows": len(raw)}
    raw["order_ts"] = pd.to_datetime(raw["order date (DateOrders)"], format="%m/%d/%Y %H:%M", errors="coerce")
    raw["ship_ts"] = pd.to_datetime(raw["shipping date (DateOrders)"], format="%m/%d/%Y %H:%M", errors="coerce")
    bad = raw["order_ts"].isna()
    stats["unparseable_dates_removed"] = int(bad.sum())
    raw = raw[~bad]
    g = raw.groupby("Order Id")
    first = g.first()
    agg = g.agg(quantity=("Order Item Quantity", "sum"), value=("Sales", "sum"),
                profit=("Order Profit Per Order", "first"), items=("Order Item Id", "count"))
    order_date = first["order_ts"].dt.normalize()
    sched = first["Days for shipment (scheduled)"].astype(float)
    real = first["Days for shipping (real)"].astype(float)
    cancelled = first["Delivery Status"].eq("Shipping canceled")
    ship = pd.DataFrame({
        "shipment_id": "DC-" + first.index.astype(str),
        "source_id": SOURCE_ID,
        "order_date": order_date.dt.date,
        "ship_date": first["ship_ts"].dt.normalize().dt.date,
        "promised_date": (order_date + pd.to_timedelta(sched, unit="D")).dt.date,
        "delivered_date": np.where(cancelled, None, (order_date + pd.to_timedelta(real, unit="D")).dt.date),
        "ship_mode": first["Shipping Mode"],
        "scheduled_days": sched,
        "actual_days": np.where(cancelled, np.nan, real),
        "status": np.where(cancelled, "cancelled", "delivered"),
        "is_late": np.where(cancelled, None, first["Late_delivery_risk"].astype(bool)),
        "supplier_id": None,
        "origin_state": None,
        "dest_state": first["Order State"],
        "dest_country": first["Order Country"],
        "market": first["Market"],
        "region": first["Order Region"],
        "category": first["Category Name"],
        "quantity": agg["quantity"].astype(float),
        "value": agg["value"].round(2),
        "freight_value": None,
        "distance_km": None,
        "dest_lat": first["Latitude"],
        "dest_lng": first["Longitude"],
        "review_score": None,
        "customer_segment": first["Customer Segment"],
        "department": first["Department Name"],
        "order_type": first["Type"],
        "items": agg["items"].astype(int),
        "discount_rate": g["Order Item Discount Rate"].mean().round(4),
        "data_origin": "real:dataco",
    }).reset_index(drop=True)
    stats.update(shipments=len(ship), late_rate=float(pd.Series(ship["is_late"]).dropna().astype(float).mean()),
                 cancelled=int(cancelled.sum()), start=str(order_date.min().date()), end=str(order_date.max().date()))
    log.info("DataCo: %s", stats)
    return {"shipments": ship, "stats": stats}
