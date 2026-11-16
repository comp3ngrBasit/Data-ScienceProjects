"""Stockout prediction: probability that a planning SKU runs out of stock within the next 14 days.

Training data: daily inventory positions (SIMULATED — produced by the replenishment simulation driven by REAL
UCI demand) joined with demand statistics and the inbound PO pipeline known on each day.
Label: on-hand reaches zero on any day in (t, t+14].
Features (all known at t): on hand, days of cover (on hand / 28-day mean demand), 7/28-day demand mean,
28-day demand std and CV, share of selling days, open PO quantity, days until the next promised receipt,
quoted lead time, and whether stock is already zero.
Split: chronological (last 90 days = test). Models: Logistic Regression, LightGBM. Metrics: F1, ROC-AUC, PR-AUC.
"""
from __future__ import annotations

from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scip.core.config import get_settings
from scip.core.db import get_engine, read_sql, session_scope, write_frame
from scip.core.logging import get_logger
from scip.core.models import ModelRun
from scip.ml import metrics as M

log = get_logger("ml.stockout")
HORIZON = 14
FEATURES = ["on_hand", "days_cover", "rmean_7", "rmean_28", "rstd_28", "cv_28", "sell_rate_28", "on_order",
            "days_to_receipt", "lead_time", "is_zero"]
MODEL_PATH = "stockout.joblib"


def build_dataset(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    inv = read_sql("SELECT date, sku, on_hand FROM inventory_snapshots", engine=engine, parse_dates=["date"])
    dem = read_sql("SELECT s.date, s.sku, SUM(s.quantity) q FROM sales_daily s JOIN products p USING(sku) "
                   "WHERE p.is_planning_sku GROUP BY s.date, s.sku", engine=engine, parse_dates=["date"])
    po = read_sql("SELECT sku, order_date, promised_date, received_date, quantity FROM purchase_orders "
                  "WHERE created_by='history'", engine=engine, parse_dates=["order_date", "promised_date", "received_date"])
    lt = read_sql("SELECT sku, MIN(lead_time_days) lt FROM supplier_products GROUP BY sku", engine=engine).set_index("sku")["lt"]
    frames = []
    for sku, g in inv.groupby("sku"):
        g = g.sort_values("date").set_index("date")
        d = dem[dem["sku"] == sku].set_index("date")["q"].reindex(g.index, fill_value=0.0)
        f = pd.DataFrame(index=g.index)
        f["on_hand"] = g["on_hand"]
        f["rmean_7"] = d.rolling(7, min_periods=1).mean()
        f["rmean_28"] = d.rolling(28, min_periods=7).mean()
        f["rstd_28"] = d.rolling(28, min_periods=7).std()
        f["cv_28"] = f["rstd_28"] / (f["rmean_28"] + 0.1)
        f["sell_rate_28"] = (d > 0).rolling(28, min_periods=7).mean()
        f["days_cover"] = np.clip(f["on_hand"] / (f["rmean_28"] + 0.1), 0, 365)
        p = po[po["sku"] == sku]
        on_order, to_receipt = [], []
        for t in g.index:
            open_ = p[(p["order_date"] <= t) & ((p["received_date"].isna()) | (p["received_date"] > t))]
            on_order.append(open_["quantity"].sum())
            nxt = (open_["promised_date"] - t).dt.days
            to_receipt.append(float(max(nxt.min(), 0)) if len(nxt) else 60.0)
        f["on_order"] = on_order
        f["days_to_receipt"] = to_receipt
        f["lead_time"] = float(lt.get(sku, 30))
        f["is_zero"] = (f["on_hand"] <= 0).astype(int)
        zero = (g["on_hand"] <= 0).astype(int)
        future_min = zero[::-1].rolling(HORIZON, min_periods=1).max()[::-1].shift(-1)
        f["label"] = future_min
        f["sku"] = sku
        frames.append(f.iloc[28:])
    df = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    return df


def train(engine=None, persist: bool = True) -> dict:
    engine = engine or get_engine()
    df = build_dataset(engine)
    labelled = df.dropna(subset=["label"])
    labelled = labelled[labelled["date"] <= labelled["date"].max() - pd.Timedelta(days=HORIZON)]
    cut = labelled["date"].max() - pd.Timedelta(days=90)
    tr, te = labelled[labelled["date"] <= cut], labelled[labelled["date"] > cut]
    X, y = tr[FEATURES].fillna(0), tr["label"].astype(int)
    vcut = tr["date"].max() - pd.Timedelta(days=45)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                       subsample=0.8, subsample_freq=1, random_state=get_settings().random_seed, verbose=-1),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(X[tr["date"] <= vcut], y[tr["date"] <= vcut])
        thr = M.best_f1_threshold(y[tr["date"] > vcut], m.predict_proba(X[tr["date"] > vcut])[:, 1])
        m.fit(X, y)
        results[name] = M.classification_report(te["label"].astype(int), m.predict_proba(te[FEATURES].fillna(0))[:, 1], thr)
        fitted[name] = m
        log.info("Stockout %s: F1 %.3f AUC %.3f", name, results[name]["f1"], results[name]["roc_auc"])
    champion = max(results, key=lambda k: results[k]["roc_auc"])
    path = get_settings().models_dir / MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[champion], "threshold": results[champion]["threshold"], "name": champion}, path)
    if persist:
        with session_scope(engine) as s:
            s.query(ModelRun).filter(ModelRun.model_type == "stockout").update({"is_champion": False})
            for name, rep in results.items():
                s.add(ModelRun(model_type="stockout", model_name=name, metrics=rep, is_champion=name == champion,
                               params={"features": FEATURES, "horizon_days": HORIZON, "split": "last 90 days test"},
                               data_window=f"{labelled['date'].min().date()}..{labelled['date'].max().date()}",
                               notes="Trained on simulated inventory positions driven by real UCI demand."))
    latest = df[df["date"] == df["date"].max()]
    proba = fitted[champion].predict_proba(latest[FEATURES].fillna(0))[:, 1]
    scored = pd.DataFrame({"scored_at": datetime.now(), "sku": latest["sku"].values,
                           "probability": proba.round(4),
                           "risk_band": np.select([proba >= 0.6, proba >= 0.3], ["High", "Medium"], "Low")})
    if persist:
        write_frame(scored, "stockout_predictions", engine)
    return {"champion": champion, "results": results, "train_rows": len(tr), "test_rows": len(te),
            "positive_rate": float(labelled["label"].mean()), "scored": len(scored)}
