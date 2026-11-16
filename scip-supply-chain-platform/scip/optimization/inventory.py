"""Inventory optimization: safety stock, reorder point, EOQ, min/max and replenishment quantities.

d   = champion-forecast mean daily demand over the next 30 days (fallback: last 56 days of history)
sd  = champion daily forecast RMSE on the hold-out (the uncertainty safety stock must absorb);
      fallback: std of daily demand
L   = primary supplier lead time: mean of actual PO lead times when >= 3 receipts, else quoted
sL  = std of actual lead times (else catalog lead-time std)
SS  = z(service level by ABC class) * sqrt(L*sd^2 + d^2*sL^2)
ROP = d*L + SS ; EOQ = sqrt(2*D*S/H) ; order qty = max(EOQ, MOQ) ; Max = ROP + order qty
Recommended qty = max(Max - (on hand + on order), MOQ) when on hand + on order <= ROP, else 0.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from scip.core.config import get_settings
from scip.core.db import get_engine, read_sql, write_frame
from scip.ingestion.pipeline import as_of_date


def safety_stock(d, sd, L, sL, service_level):
    d, sd, L, sL = (np.asarray(x, float) for x in (d, sd, L, sL))
    if (d < 0).any() or (sd < 0).any() or (L < 0).any() or (sL < 0).any():
        raise ValueError("inputs must be non-negative")
    return norm.ppf(np.asarray(service_level, float)) * np.sqrt(L * sd ** 2 + d ** 2 * sL ** 2)


def eoq(annual_demand, ordering_cost, unit_holding_cost):
    D, H = np.asarray(annual_demand, float), np.asarray(unit_holding_cost, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.sqrt(2 * D * ordering_cost / H)
    return np.where((D > 0) & (H > 0), q, 0.0)


def abc_classes(revenue: pd.Series) -> pd.Series:
    r = revenue.fillna(0).sort_values(ascending=False)
    share_before = (r.cumsum() - r) / max(r.sum(), 1e-9)
    return pd.Series(np.where(share_before < 0.8, "A", np.where(share_before < 0.95, "B", "C")), index=r.index)


def compute_policy(engine=None, persist: bool = True) -> pd.DataFrame:
    engine = engine or get_engine()
    s = get_settings()
    as_of = as_of_date(engine)
    prod = read_sql("SELECT sku, name, category, unit_cost, unit_price FROM products WHERE is_planning_sku", engine=engine)
    hist = read_sql("SELECT s.sku, s.date, SUM(s.quantity) q FROM sales_daily s JOIN products p USING(sku) "
                    "WHERE p.is_planning_sku AND s.date > :d GROUP BY s.sku, s.date",
                    {"d": (as_of - pd.Timedelta(days=365)).date()}, engine=engine, parse_dates=["date"])
    wide = hist.pivot_table(index="date", columns="sku", values="q", aggfunc="sum") \
        .reindex(pd.date_range(as_of - pd.Timedelta(days=364), as_of), fill_value=0).fillna(0)
    last56 = wide.iloc[-56:]
    df = prod.set_index("sku")
    df["hist_rate"] = last56.mean().reindex(df.index).fillna(0)
    df["hist_std"] = last56.std().reindex(df.index).fillna(0)
    df["revenue_365"] = (wide.sum() * df["unit_price"]).reindex(df.index).fillna(0)

    fc = read_sql("SELECT f.sku, AVG(f.yhat) rate FROM forecasts f JOIN model_runs r USING(run_id) "
                  "WHERE r.is_champion AND r.model_type='forecast' AND f.date <= :d GROUP BY f.sku",
                  {"d": (as_of + pd.Timedelta(days=30)).date()}, engine=engine).set_index("sku")["rate"]
    err = read_sql("SELECT m.sku, m.rmse FROM forecast_metrics m JOIN model_runs r USING(run_id) "
                   "WHERE r.is_champion AND r.model_type='forecast'", engine=engine).set_index("sku")["rmse"]
    use_fc = df.index.isin(fc.index)
    df["avg_daily_demand"] = np.where(use_fc, fc.reindex(df.index), df["hist_rate"])
    df["demand_std"] = np.where(use_fc & df.index.isin(err.index), err.reindex(df.index), df["hist_std"])
    df["demand_basis"] = np.where(use_fc, "champion forecast (30 d)", "history (56 d)")

    sp = read_sql("SELECT supplier_id, sku, unit_cost AS vendor_cost, moq, lead_time_days, lead_time_std FROM supplier_products",
                  engine=engine)
    po = read_sql("SELECT sku, supplier_id, order_date, received_date, quantity, status FROM purchase_orders",
                  engine=engine, parse_dates=["order_date", "received_date"])
    primary = po.sort_values("order_date").groupby("sku")["supplier_id"].last()
    cheapest = sp.sort_values("vendor_cost").groupby("sku")["supplier_id"].first()
    df["supplier_id"] = primary.reindex(df.index).fillna(cheapest.reindex(df.index))
    sp_idx = sp.set_index(["supplier_id", "sku"])
    keys = list(zip(df["supplier_id"], df.index))
    df["moq"] = [sp_idx["moq"].get(k, 1) for k in keys]
    df["quoted_lt"] = [sp_idx["lead_time_days"].get(k, 30.0) for k in keys]
    df["quoted_lt_std"] = [sp_idx["lead_time_std"].get(k, 5.0) for k in keys]
    rec = po[po["received_date"].notna()].assign(lt=lambda x: (x["received_date"] - x["order_date"]).dt.days)
    lt_stats = rec.groupby(["sku", "supplier_id"])["lt"].agg(["mean", "std", "count"])
    act = [lt_stats.loc[k[::-1]] if k[::-1] in lt_stats.index else None for k in keys]
    df["lead_time_days"] = [a["mean"] if a is not None and a["count"] >= 3 else q for a, q in zip(act, df["quoted_lt"])]
    df["lead_time_std"] = [a["std"] if a is not None and a["count"] >= 3 and pd.notna(a["std"]) else q
                           for a, q in zip(act, df["quoted_lt_std"])]

    inv = read_sql("SELECT sku, SUM(on_hand) on_hand FROM inventory_current GROUP BY sku", engine=engine).set_index("sku")["on_hand"]
    df["on_hand"] = inv.reindex(df.index).fillna(0)
    open_po = po[po["status"].isin(["sent", "approved"])].groupby("sku")["quantity"].sum()
    df["on_order"] = open_po.reindex(df.index).fillna(0)

    df["abc_class"] = abc_classes(df["revenue_365"]).reindex(df.index)
    sl = {"A": s.service_level_a, "B": s.service_level_b, "C": s.service_level_c}
    df["service_level"] = df["abc_class"].map(sl)
    d, sd, L, sL = df["avg_daily_demand"], df["demand_std"], df["lead_time_days"], df["lead_time_std"]
    df["safety_stock"] = np.ceil(safety_stock(d, sd, L, sL, df["service_level"]))
    df["reorder_point"] = np.ceil(d * L + df["safety_stock"])
    df["eoq"] = np.ceil(eoq(365 * d, s.ordering_cost, df["unit_cost"] * s.holding_cost_rate))
    df["order_qty"] = np.maximum(df["eoq"], df["moq"])
    df["max_level"] = df["reorder_point"] + df["order_qty"]
    ip = df["on_hand"] + df["on_order"]
    df["recommended_qty"] = np.where(ip <= df["reorder_point"], np.ceil(np.maximum(df["max_level"] - ip, df["moq"])), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["days_cover"] = np.where(d > 0, df["on_hand"] / d, np.nan)
    df["stockout_date"] = [as_of + pd.Timedelta(days=int(x)) if pd.notna(x) and x < 3650 else pd.NaT for x in df["days_cover"]]
    df["inventory_value"] = df["on_hand"] * df["unit_cost"]
    df["status"] = np.select([(df["on_hand"] <= 0) | (ip < d * L), ip <= df["reorder_point"],
                              df["on_hand"] > df["max_level"]], ["Critical", "Reorder", "Overstock"], "Healthy")
    out = df.reset_index().rename(columns={"index": "sku"})
    out["computed_at"] = datetime.now()
    if persist:
        cols = ["computed_at", "sku", "abc_class", "service_level", "avg_daily_demand", "demand_std", "lead_time_days",
                "lead_time_std", "safety_stock", "reorder_point", "eoq", "max_level", "on_hand", "on_order",
                "recommended_qty", "days_cover", "stockout_date", "inventory_value", "status", "demand_basis"]
        p = out[cols].copy()
        p["stockout_date"] = p["stockout_date"].dt.date
        write_frame(p, "inventory_policy", engine)
    return out


def latest_policy(engine=None) -> pd.DataFrame:
    return read_sql("SELECT ip.*, p.name, p.category, p.unit_cost, p.unit_price FROM inventory_policy ip "
                    "JOIN products p USING(sku) WHERE ip.computed_at=(SELECT MAX(computed_at) FROM inventory_policy)",
                    engine=engine, parse_dates=["stockout_date"])
