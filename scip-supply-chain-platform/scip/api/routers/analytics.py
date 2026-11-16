from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from scip.api.deps import records
from scip.core.db import read_sql
from scip.ml.forecasting import champion_forecast, horizon_summary
from scip.ml.supplier_scoring import latest_scores
from scip.optimization.inventory import latest_policy
from scip.services import control_tower as ct

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.get("/dashboard/kpis")
def dashboard_kpis():
    return ct.kpis()


@router.get("/dashboard/risk-heatmap")
def heatmap(weeks: int = Query(12, ge=1, le=13)):
    return records(ct.risk_heatmap(weeks=weeks))


@router.get("/dashboard/delivery-tracking")
def tracking():
    return records(ct.delivery_tracking())


@router.get("/forecasts/summary")
def forecast_summary():
    hs = horizon_summary()
    if hs.empty:
        return []
    names = read_sql("SELECT sku, name, category FROM products")
    return records(hs.merge(names, on="sku", how="left"))


@router.get("/forecasts/models")
def forecast_models():
    df = read_sql("SELECT run_id, model_name, metrics, is_champion, trained_at, data_window, notes FROM model_runs "
                  "WHERE model_type='forecast' AND run_id IN (SELECT MAX(run_id) FROM model_runs WHERE model_type='forecast' "
                  "GROUP BY model_name) ORDER BY model_name")
    df["metrics"] = df["metrics"].map(lambda m: m if isinstance(m, dict) else json.loads(m))
    return records(df)


@router.get("/forecasts/portfolio")
def forecast_portfolio(category: str | None = None):
    fc = champion_forecast()
    hist = read_sql("SELECT s.date, s.sku, SUM(s.quantity) q FROM sales_daily s JOIN products p USING(sku) "
                    "WHERE p.is_planning_sku GROUP BY s.date, s.sku", parse_dates=["date"])
    if category:
        skus = read_sql("SELECT sku FROM products WHERE category=:c", {"c": category})["sku"]
        fc, hist = fc[fc["sku"].isin(skus)], hist[hist["sku"].isin(skus)]
    # Weeks end on the as-of weekday so the last history week and every forecast week are complete;
    # incomplete buckets at either end are dropped. Days without sales (e.g. Saturdays in UCI) count as zero.
    from scip.ingestion.pipeline import as_of_date
    as_of = as_of_date()
    rule = f"W-{as_of.day_name()[:3].upper()}"
    hd = hist.groupby("date")["q"].sum()
    hd = hd.reindex(pd.date_range(hd.index.min(), as_of), fill_value=0.0)
    h = hd.resample(rule).agg(["sum", "count"])
    h = h[h["count"] == 7]["sum"].rename("actual")
    h.index.name = "date"
    fd = fc.groupby("date")["yhat"].sum()
    n = fd.resample(rule).count()
    f = fd.resample(rule).sum()[n == 7].rename("yhat").to_frame()
    # 80% band for a portfolio week: 1.28 * sqrt(7 * sum of per-SKU daily RMSE^2)
    # (independent errors across SKUs and days; summing per-SKU bands would assume perfect correlation)
    rmse = read_sql("SELECT m.sku, m.rmse FROM forecast_metrics m JOIN model_runs r USING(run_id) "
                    "WHERE r.is_champion AND r.model_type='forecast'").set_index("sku")["rmse"]
    rmse = rmse[rmse.index.isin(fc["sku"].unique())]
    half = 1.28 * float(np.sqrt(7 * (rmse ** 2).sum()))
    f["yhat_lower"] = (f["yhat"] - half).clip(lower=0)
    f["yhat_upper"] = f["yhat"] + half
    return {"history": records(h.reset_index()), "forecast": records(f.reset_index())}


@router.get("/forecasts/{sku}")
def forecast_sku(sku: str, history_days: int = 180):
    fc = read_sql("SELECT f.date, f.model, f.yhat, f.yhat_lower, f.yhat_upper, r.is_champion FROM forecasts f "
                  "JOIN model_runs r USING(run_id) WHERE f.sku=:s AND r.run_id IN (SELECT MAX(run_id) FROM model_runs "
                  "WHERE model_type='forecast' GROUP BY model_name)", {"s": sku}, parse_dates=["date"])
    if fc.empty:
        raise HTTPException(404, f"No forecast for SKU {sku}")
    hist = read_sql("SELECT date, SUM(quantity) quantity FROM sales_daily WHERE sku=:s GROUP BY date ORDER BY date",
                    {"s": sku}, parse_dates=["date"])
    hist = hist[hist["date"] > hist["date"].max() - pd.Timedelta(days=history_days)]
    met = read_sql("SELECT m.model, m.mae, m.rmse, m.mape, m.wape FROM forecast_metrics m WHERE m.sku=:s AND m.run_id IN "
                   "(SELECT MAX(run_id) FROM model_runs WHERE model_type='forecast' GROUP BY model_name)", {"s": sku})
    prod = read_sql("SELECT sku, name, category, unit_price FROM products WHERE sku=:s", {"s": sku})
    return {"product": records(prod)[0], "history": records(hist), "forecast": records(fc), "metrics": records(met)}


@router.get("/inventory/policy")
def inventory_policy(status: str | None = None):
    df = latest_policy()
    if status:
        df = df[df["status"] == status]
    so = read_sql("SELECT sku, probability AS stockout_probability, risk_band FROM stockout_predictions "
                  "WHERE scored_at=(SELECT MAX(scored_at) FROM stockout_predictions)")
    return records(df.merge(so, on="sku", how="left").sort_values(["status", "days_cover"]))


@router.post("/inventory/recompute")
def inventory_recompute():
    from scip.optimization.inventory import compute_policy
    df = compute_policy()
    return {"skus": len(df), "status_counts": df["status"].value_counts().to_dict()}


@router.get("/stockout/predictions")
def stockout_predictions():
    return records(read_sql("SELECT sp.*, p.name, p.category FROM stockout_predictions sp JOIN products p USING(sku) "
                            "WHERE scored_at=(SELECT MAX(scored_at) FROM stockout_predictions) ORDER BY probability DESC"))


@router.get("/suppliers/scores")
def supplier_scores():
    return records(latest_scores())


@router.get("/suppliers/{supplier_id}")
def supplier_detail(supplier_id: str):
    sc = latest_scores()
    row = sc[sc["supplier_id"] == supplier_id]
    if row.empty:
        raise HTTPException(404, "Supplier not found or not scored")
    pos = read_sql("SELECT po_id, sku, quantity, total_value, order_date, promised_date, received_date, status, "
                   "predicted_late_risk FROM purchase_orders WHERE supplier_id=:s ORDER BY order_date DESC LIMIT 200",
                   {"s": supplier_id})
    cat = read_sql("SELECT sku, unit_cost, moq, lead_time_days, lead_time_std, capacity_per_month FROM supplier_products "
                   "WHERE supplier_id=:s", {"s": supplier_id})
    return {"score": records(row)[0], "purchase_orders": records(pos), "catalog": records(cat)}


@router.get("/models")
def models():
    df = ct.model_overview()
    df["metrics"] = df["metrics"].map(lambda m: m if isinstance(m, dict) else json.loads(m))
    return records(df)


@router.get("/shipments/performance")
def shipments_performance():
    return records(ct.shipment_performance())


@router.get("/shipments")
def shipments(source: str | None = None, limit: int = Query(200, le=5000)):
    q = "SELECT * FROM shipments"
    p = {}
    if source:
        q += " WHERE source_id=:s"
        p["s"] = source
    q += " ORDER BY order_date DESC LIMIT :l"
    p["l"] = limit
    return records(read_sql(q, p))
