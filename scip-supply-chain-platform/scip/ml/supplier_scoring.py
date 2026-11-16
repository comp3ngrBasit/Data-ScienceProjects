"""Supplier performance scoring.

Vendors (purchase-order history, simulated & calibrated on real DataCo delays):
  on-time rate = received POs with received <= promised / received POs
  avg delay    = mean days late over late POs
  fill rate    = received qty / ordered qty
  lead-time CV = std / mean of actual lead time
  predicted risk = mean model-predicted late probability of the vendor's open POs (delivery-risk model)
Marketplace sellers (Olist, real — when order/seller files are loaded): on-time rate and delay from shipments,
review score from customer reviews.

Score (0-100) = 100 x (0.40 on-time + 0.20 fill rate + 0.15 (1 - min(CV, 1)) + 0.15 (1 - predicted risk)
                       + 0.10 quality)  where quality = review score / 5 (sellers) or fill rate (vendors).
Tier: Strategic >= 80, Preferred >= 65, Watch >= 50, At risk < 50.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from scip.core.db import get_engine, read_sql, write_frame
from scip.core.logging import get_logger

log = get_logger("ml.supplier_scoring")
WEIGHTS = {"on_time": 0.40, "fill_rate": 0.20, "consistency": 0.15, "predicted": 0.15, "quality": 0.10}


def tier(score: float) -> str:
    return "Strategic" if score >= 80 else "Preferred" if score >= 65 else "Watch" if score >= 50 else "At risk"


def score_suppliers(engine=None, persist: bool = True) -> pd.DataFrame:
    engine = engine or get_engine()
    po = read_sql("SELECT supplier_id, quantity, qty_received, order_date, promised_date, received_date, status, "
                  "predicted_late_risk FROM purchase_orders", engine=engine,
                  parse_dates=["order_date", "promised_date", "received_date"])
    rows = []
    for sid, g in po.groupby("supplier_id"):
        rec = g[g["received_date"].notna()]
        if rec.empty:
            continue
        lead = (rec["received_date"] - rec["order_date"]).dt.days
        delay = (rec["received_date"] - rec["promised_date"]).dt.days
        on_time = float((delay <= 0).mean())
        fill = float(rec["qty_received"].sum() / max(rec["quantity"].sum(), 1))
        cv = float(lead.std() / lead.mean()) if lead.mean() > 0 else 0.0
        pred = g.loc[g["received_date"].isna(), "predicted_late_risk"].dropna()
        pred_v = float(pred.mean()) if len(pred) else 1 - on_time
        rows.append(dict(supplier_id=sid, shipments=len(rec), on_time_rate=on_time,
                         avg_delay_days=float(delay[delay > 0].mean()) if (delay > 0).any() else 0.0,
                         predicted_late_risk=pred_v, review_score=None, lead_time_cv=cv, fill=fill,
                         quality=fill, data_origin="simulated"))
    sh = read_sql("SELECT supplier_id, is_late, actual_days, scheduled_days, review_score FROM shipments "
                  "WHERE supplier_id IS NOT NULL AND is_late IS NOT NULL", engine=engine)
    for sid, g in sh.groupby("supplier_id"):
        if len(g) < 5:
            continue
        delay = g["actual_days"] - g["scheduled_days"]
        on_time = float(1 - g["is_late"].astype(float).mean())
        rv = g["review_score"].dropna()
        rows.append(dict(supplier_id=sid, shipments=len(g), on_time_rate=on_time,
                         avg_delay_days=float(delay[delay > 0].mean()) if (delay > 0).any() else 0.0,
                         predicted_late_risk=1 - on_time, review_score=float(rv.mean()) if len(rv) else None,
                         lead_time_cv=float(g["actual_days"].std() / g["actual_days"].mean()) if g["actual_days"].mean() else 0.0,
                         fill=1.0, quality=float(rv.mean() / 5) if len(rv) else on_time, data_origin="real:olist"))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    w = WEIGHTS
    df["score"] = (100 * (w["on_time"] * df["on_time_rate"] + w["fill_rate"] * df["fill"].clip(upper=1)
                          + w["consistency"] * (1 - df["lead_time_cv"].clip(0, 1))
                          + w["predicted"] * (1 - df["predicted_late_risk"].clip(0, 1))
                          + w["quality"] * df["quality"].clip(0, 1))).round(1)
    df["tier"] = df["score"].map(tier)
    df["scored_at"] = datetime.now()
    if persist:
        write_frame(df.drop(columns=["fill", "quality"]), "supplier_scores", engine)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def latest_scores(engine=None) -> pd.DataFrame:
    return read_sql("SELECT sc.*, s.name, s.country_code, s.supplier_type FROM supplier_scores sc "
                    "JOIN suppliers s USING(supplier_id) WHERE sc.scored_at=(SELECT MAX(scored_at) FROM supplier_scores) "
                    "ORDER BY sc.score DESC", engine=engine)
