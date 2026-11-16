"""Control-tower read models: KPIs, inventory risk heatmap, delivery tracking, model overview."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scip.core.db import get_engine, read_sql
from scip.ingestion.pipeline import as_of_date
from scip.ml.forecasting import champion_forecast
from scip.ml.supplier_scoring import latest_scores
from scip.optimization.inventory import latest_policy


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def kpis(engine=None) -> dict:
    engine = engine or get_engine()
    as_of = as_of_date(engine)
    pol = latest_policy(engine)
    fc = champion_forecast(engine)
    out = {"as_of_date": as_of.date().isoformat()}
    out["total_inventory_value"] = float(pol["inventory_value"].sum()) if not pol.empty else 0.0
    out["planning_skus"] = int(len(pol))
    if not fc.empty:
        price = read_sql("SELECT sku, unit_price FROM products", engine=engine).set_index("sku")["unit_price"]
        start = fc["date"].min()
        for h in (7, 30, 90):
            sub = fc[fc["date"] < start + pd.Timedelta(days=h)]
            out[f"predicted_demand_{h}d_units"] = float(sub["yhat"].sum())
            out[f"predicted_demand_{h}d_value"] = float((sub["yhat"] * sub["sku"].map(price)).sum())
        out["forecast_model"] = fc["model"].iat[0]
    st = pol["status"].value_counts().to_dict() if not pol.empty else {}
    out.update(sku_critical=int(st.get("Critical", 0)), sku_reorder=int(st.get("Reorder", 0)),
               sku_overstock=int(st.get("Overstock", 0)), sku_healthy=int(st.get("Healthy", 0)))
    so = read_sql("SELECT risk_band, COUNT(*) n FROM stockout_predictions WHERE scored_at=(SELECT MAX(scored_at) "
                  "FROM stockout_predictions) GROUP BY risk_band", engine=engine)
    out["stockout_risk_high"] = int(so.set_index("risk_band")["n"].get("High", 0)) if not so.empty else 0
    al = read_sql("SELECT severity, COUNT(*) n FROM alerts WHERE status='open' GROUP BY severity", engine=engine)
    al = al.set_index("severity")["n"] if not al.empty else pd.Series(dtype=int)
    out.update(alerts_critical=int(al.get("Critical", 0)), alerts_warning=int(al.get("Warning", 0)))
    sc = latest_scores(engine)
    if not sc.empty:
        w = sc["shipments"].clip(lower=1)
        out["supplier_on_time_rate"] = float(np.average(sc["on_time_rate"], weights=w))
        out["supplier_avg_score"] = float(sc["score"].mean())
        out["suppliers_at_risk"] = int(sc["tier"].isin(["At risk", "Watch"]).sum())
    po = read_sql("SELECT status, COUNT(*) n, SUM(total_value) v FROM purchase_orders GROUP BY status", engine=engine)
    po = po.set_index("status") if not po.empty else pd.DataFrame(columns=["n", "v"])
    out["po_counts"] = {k: int(v) for k, v in po["n"].items()}
    out["po_pending_approval"] = int(po["n"].get("pending_approval", 0))
    out["po_open_value"] = float(po.loc[po.index.isin(["sent", "approved", "pending_approval"]), "v"].sum())
    late = read_sql("SELECT COUNT(*) n FROM purchase_orders WHERE status='sent' AND promised_date < :d",
                    {"d": as_of.date()}, engine=engine)["n"].iat[0]
    out["inbound_late"] = int(late)
    return {k: _clean(v) for k, v in out.items()}


def risk_heatmap(engine=None, weeks: int = 12) -> pd.DataFrame:
    """Share of SKUs per category projected to fall below safety stock (0.5) or to zero (1.0) in each week,
    projecting on-hand forward with the champion forecast and open PO receipts on their promised dates."""
    engine = engine or get_engine()
    pol = latest_policy(engine)
    fc = champion_forecast(engine)
    if pol.empty or fc.empty:
        return pd.DataFrame()
    as_of = as_of_date(engine)
    po = read_sql("SELECT sku, promised_date, quantity FROM purchase_orders WHERE status IN ('sent','approved')",
                  engine=engine, parse_dates=["promised_date"])
    days = pd.date_range(as_of + pd.Timedelta(days=1), periods=weeks * 7)
    dem = fc.pivot_table(index="date", columns="sku", values="yhat").reindex(days).fillna(0)
    arr = po.assign(promised_date=po["promised_date"].clip(lower=days[0])).pivot_table(
        index="promised_date", columns="sku", values="quantity", aggfunc="sum").reindex(days).fillna(0)
    rows = []
    for _, p in pol.iterrows():
        sku = p["sku"]
        d = dem[sku].values if sku in dem else np.zeros(len(days))
        a = arr[sku].values if sku in arr else np.zeros(len(days))
        level = p["on_hand"] + np.cumsum(a - d)
        for w in range(weeks):
            seg = level[w * 7:(w + 1) * 7]
            risk = 1.0 if (seg <= 0).any() else 0.5 if (seg < p["safety_stock"]).any() else 0.0
            rows.append(dict(category=p["category"], week=w + 1, week_start=days[w * 7].date().isoformat(), sku=sku, risk=risk))
    r = pd.DataFrame(rows)
    return r.groupby(["category", "week", "week_start"], as_index=False).agg(risk=("risk", "mean"), skus=("sku", "nunique"))


def delivery_tracking(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    as_of = as_of_date(engine)
    po = read_sql("SELECT p.po_id, p.sku, pr.name AS product, s.name AS supplier, p.quantity, p.total_value, p.order_date, "
                  "p.promised_date, p.received_date, p.status, p.predicted_late_risk FROM purchase_orders p "
                  "JOIN suppliers s USING(supplier_id) JOIN products pr USING(sku) "
                  "WHERE p.status IN ('sent','approved') OR p.received_date > :d ORDER BY p.promised_date",
                  {"d": (as_of - pd.Timedelta(days=14)).date()}, engine=engine, parse_dates=["order_date", "promised_date", "received_date"])
    if po.empty:
        return po
    late = po["received_date"].isna() & (po["promised_date"] < as_of)
    risky = po["received_date"].isna() & (po["predicted_late_risk"].fillna(0) >= 0.7)
    po["tracking_status"] = np.select([po["received_date"].notna(), late, risky],
                                      ["Delivered", "Overdue", "At risk"], "On track")
    po["days_to_promised"] = (po["promised_date"] - as_of).dt.days
    return po


def shipment_performance(engine=None) -> pd.DataFrame:
    return read_sql("SELECT source_id, ship_mode, market, COUNT(*) shipments, AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END) late_rate, "
                    "AVG(scheduled_days) avg_scheduled_days, AVG(actual_days) avg_actual_days FROM shipments "
                    "WHERE is_late IS NOT NULL GROUP BY source_id, ship_mode, market ORDER BY source_id, ship_mode, market",
                    engine=engine)


def model_overview(engine=None) -> pd.DataFrame:
    return read_sql("SELECT run_id, model_type, model_name, trained_at, metrics, is_champion, data_window, notes "
                    "FROM model_runs ORDER BY model_type, run_id DESC", engine=engine)
