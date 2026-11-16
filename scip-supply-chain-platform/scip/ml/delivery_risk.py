"""Supplier / shipment delivery-risk classification.

Training data: REAL shipments with observed outcome (DataCo `Late_delivery_risk`; Olist delivered vs
estimated date when the full Olist data is loaded). Cancelled shipments are excluded.
Features are only those known when the order is placed: shipping mode, scheduled days, market, region,
category, department, customer segment, payment type, quantity, item count, value, discount, order weekday
and month, destination coordinates. Outcome columns (actual days, delivery status) are never used.
Split: chronological — the most recent 20% of orders form the test set; the decision threshold (Youden's J) is chosen on
the last 15% of the training period. Models: Logistic Regression (baseline), LightGBM, XGBoost.

The champion is applied to open purchase orders (mapped to the same feature space: vendor shipping mode,
quoted lead time, market by vendor country …) to predict supplier delays. This is a transfer from customer
shipments to inbound vendor shipments and is flagged as such.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from scip.core.config import get_settings
from scip.core.db import get_engine, read_sql, session_scope
from scip.core.logging import get_logger
from scip.core.models import ModelRun
from scip.ml import metrics as M

log = get_logger("ml.delivery_risk")
CAT = ["ship_mode", "market", "region", "category", "department", "customer_segment", "order_type"]
NUM = ["scheduled_days", "quantity", "items", "value", "discount_rate", "dow", "month", "dest_lat", "dest_lng"]
MODEL_PATH = "delivery_risk.joblib"


def load_training(engine=None) -> pd.DataFrame:
    df = read_sql("SELECT * FROM shipments WHERE is_late IS NOT NULL AND status <> 'cancelled'",
                  engine=engine, parse_dates=["order_date"])
    if df.empty:
        raise ValueError("No labelled shipments — load DataCo or full Olist data first.")
    df["dow"] = df["order_date"].dt.weekday
    df["month"] = df["order_date"].dt.month
    for c in CAT:
        df[c] = df[c].fillna("unknown").astype(str)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["y"] = df["is_late"].astype(int)
    return df.sort_values("order_date").reset_index(drop=True)


def _models(seed: int):
    num_pipe = Pipeline([("imp", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    logit = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
                                                  ("num", num_pipe, NUM)])),
                      ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    ordinal = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
                                 ("num", "passthrough", NUM)])
    lgbm = Pipeline([("pre", ordinal), ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                                                   min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                                   colsample_bytree=0.8, random_state=seed, verbose=-1))])
    xgbm = Pipeline([("pre", ordinal), ("clf", xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                                                                  subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                                                  n_jobs=-1, verbosity=0, eval_metric="logloss"))])
    return {"Logistic Regression": logit, "LightGBM": lgbm, "XGBoost": xgbm}


def train(engine=None, persist: bool = True) -> dict:
    engine = engine or get_engine()
    s = get_settings()
    df = load_training(engine)
    n = len(df)
    test_start, val_start = int(n * 0.8), int(n * 0.8 * 0.85)
    tr, va, te = df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]
    results, fitted = {}, {}
    for name, model in _models(s.random_seed).items():
        model.fit(tr[CAT + NUM], tr["y"])
        thr = M.youden_threshold(va["y"], model.predict_proba(va[CAT + NUM])[:, 1])
        model.fit(pd.concat([tr, va])[CAT + NUM], pd.concat([tr, va])["y"])
        rep = M.classification_report(te["y"], model.predict_proba(te[CAT + NUM])[:, 1], thr)
        rep["baseline_always_late_f1"] = M.classification_report(te["y"], np.ones(len(te)), 0.5)["f1"]
        results[name] = rep
        fitted[name] = model
        log.info("%s: F1 %.3f ROC-AUC %.3f", name, rep["f1"], rep["roc_auc"])
    champion = max(results, key=lambda k: results[k]["roc_auc"])
    final = fitted[champion]
    path = s.models_dir / MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "threshold": results[champion]["threshold"], "name": champion}, path)
    window = f"{df['order_date'].min().date()}..{df['order_date'].max().date()}"
    if persist:
        with session_scope(engine) as ses:
            ses.query(ModelRun).filter(ModelRun.model_type == "delivery_risk").update({"is_champion": False})
            for name, rep in results.items():
                ses.add(ModelRun(model_type="delivery_risk", model_name=name,
                                 metrics={k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in rep.items()},
                                 params={"features": CAT + NUM, "split": "chronological 80/20",
                                         "sources": sorted(df["source_id"].unique().tolist())},
                                 data_window=window, is_champion=(name == champion),
                                 notes=f"Test set: {len(te):,} most recent shipments; late rate {te['y'].mean():.1%}."))
    return {"champion": champion, "results": results, "test_size": len(te), "train_size": len(tr) + len(va),
            "late_rate": float(df["y"].mean()), "window": window}


def load_model() -> dict | None:
    path = get_settings().models_dir / MODEL_PATH
    return joblib.load(path) if path.exists() else None


MODE_SCHEDULED_DAYS = {"Same Day": 0.0, "First Class": 1.0, "Second Class": 2.0, "Standard Class": 4.0}
VENDOR_MARKET = {"GBR": ("Europe", "Northern Europe"), "NLD": ("Europe", "Western Europe"), "DEU": ("Europe", "Western Europe"),
                 "TUR": ("Europe", "Eastern Europe"), "IND": ("Pacific Asia", "South Asia"),
                 "CHN": ("Pacific Asia", "Eastern Asia"), "PAK": ("Pacific Asia", "South Asia")}


def score_purchase_orders(engine=None) -> int:
    """Predict late-delivery probability for open purchase orders and store it on the PO."""
    engine = engine or get_engine()
    bundle = load_model()
    if bundle is None:
        return 0
    from scip.ingestion.simulate import VENDORS
    mode = {v[0]: v[5] for v in VENDORS}
    po = read_sql("SELECT p.po_id, p.supplier_id, p.quantity, p.total_value, p.order_date, p.promised_date, "
                  "s.country_code, pr.category FROM purchase_orders p JOIN suppliers s USING(supplier_id) "
                  "JOIN products pr USING(sku) WHERE p.status IN ('sent','approved','pending_approval','draft')",
                  engine=engine, parse_dates=["order_date", "promised_date"])
    if po.empty:
        return 0
    X = pd.DataFrame({
        "ship_mode": po["supplier_id"].map(mode).fillna("Standard Class"),
        "market": po["country_code"].map(lambda c: VENDOR_MARKET.get(c, ("Europe", ""))[0]),
        "region": po["country_code"].map(lambda c: VENDOR_MARKET.get(c, ("", "Western Europe"))[1]),
        "category": po["category"].fillna("unknown"), "department": "unknown", "customer_segment": "Corporate",
        "order_type": "TRANSFER",
        # DataCo's scheduled days are a property of the shipping mode (Same Day 0, First 1, Second 2, Standard 4);
        # use the mode's value rather than the vendor's quoted lead time, which is outside the training range.
        "scheduled_days": po["supplier_id"].map(mode).map(MODE_SCHEDULED_DAYS).fillna(4.0).astype(float),
        "quantity": po["quantity"], "items": 1, "value": po["total_value"], "discount_rate": 0.0,
        "dow": po["order_date"].dt.weekday, "month": po["order_date"].dt.month, "dest_lat": np.nan, "dest_lng": np.nan})
    proba = bundle["model"].predict_proba(X[CAT + NUM])[:, 1]
    from sqlalchemy import text
    with engine.begin() as conn:
        for pid, p in zip(po["po_id"], proba):
            conn.execute(text("UPDATE purchase_orders SET predicted_late_risk=:p WHERE po_id=:i"), {"p": float(p), "i": pid})
    return len(po)
