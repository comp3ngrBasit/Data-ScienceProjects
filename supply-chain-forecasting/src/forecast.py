"""Demand forecasting.

A single global gradient-boosting model is trained across all SKU/warehouse
series. Features use only information available HORIZON days before the target
date (lags >= HORIZON), so every prediction is a genuine HORIZON-day-ahead
forecast. Quantile models give an 80% prediction interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

HORIZON = 28
FEATURES = [
    "lag_28", "lag_35", "lag_42", "lag_364", "roll_mean_28", "roll_mean_91",
    "roll_std_28", "dow", "month", "weekofyear", "dayofyear",
    "promo", "price", "series_id",
]
FEATURE_LABELS = {
    "lag_28": "Sales 4 weeks ago", "lag_35": "Sales 5 weeks ago", "lag_42": "Sales 6 weeks ago",
    "lag_364": "Sales same day last year", "roll_mean_28": "28-day average",
    "roll_mean_91": "91-day average", "roll_std_28": "28-day volatility",
    "dow": "Day of week", "month": "Month", "weekofyear": "Week of year",
    "dayofyear": "Day of year", "promo": "Promotion flag", "price": "Price",
    "series_id": "SKU / warehouse",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku", "warehouse", "date"]).copy()
    df["series_id"] = df.groupby(["sku", "warehouse"]).ngroup()
    g = df.groupby("series_id")["units"]
    for lag in (28, 35, 42, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(HORIZON)
    df["roll_mean_28"] = shifted.groupby(df["series_id"]).transform(lambda s: s.rolling(28).mean())
    df["roll_mean_91"] = shifted.groupby(df["series_id"]).transform(lambda s: s.rolling(91).mean())
    df["roll_std_28"] = shifted.groupby(df["series_id"]).transform(lambda s: s.rolling(28).std())
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    return df


def _model(**kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        categorical_features=[FEATURES.index("series_id")], random_state=0, **kw)


def wape(y, yhat) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def fit_and_backtest(df: pd.DataFrame, test_days: int = 91) -> dict:
    """Train on history, evaluate on the final `test_days`, then refit on all data."""
    feat = build_features(df).dropna(subset=FEATURES)
    cutoff = feat["date"].max() - pd.Timedelta(days=test_days)
    train, test = feat[feat["date"] <= cutoff], feat[feat["date"] > cutoff].copy()

    point = _model(loss="poisson").fit(train[FEATURES], train["units"])
    lo = _model(loss="quantile", quantile=0.1).fit(train[FEATURES], train["units"])
    hi = _model(loss="quantile", quantile=0.9).fit(train[FEATURES], train["units"])

    test["forecast"] = point.predict(test[FEATURES])
    test["lo"] = lo.predict(test[FEATURES])
    test["hi"] = hi.predict(test[FEATURES])
    # baselines a planner would otherwise use
    test["naive_lastyear"] = test["lag_364"]
    test["naive_ma28"] = test["roll_mean_28"]

    metrics = {
        "model_wape": wape(test["units"], test["forecast"]),
        "lastyear_wape": wape(test["units"], test["naive_lastyear"]),
        "ma28_wape": wape(test["units"], test["naive_ma28"]),
        "coverage_80": float(((test["units"] >= test["lo"]) & (test["units"] <= test["hi"])).mean()),
        "bias": float((test["forecast"].sum() - test["units"].sum()) / test["units"].sum()),
    }
    per_sku = (test.groupby("sku").apply(lambda d: pd.Series({
        "model_wape": wape(d["units"], d["forecast"]),
        "ma28_wape": wape(d["units"], d["naive_ma28"]),
        "lastyear_wape": wape(d["units"], d["naive_lastyear"]),
    }), include_groups=False).reset_index())

    sample = test.sample(min(len(test), 3000), random_state=0)
    imp = permutation_importance(point, sample[FEATURES], sample["units"], n_repeats=5,
                                 random_state=0, scoring="neg_mean_absolute_error")
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))
    importance["label"] = importance["feature"].map(FEATURE_LABELS)

    # residual spread per series feeds safety stock
    test["resid"] = test["units"] - test["forecast"]
    resid_std = test.groupby(["sku", "warehouse"])["resid"].std().rename("resid_std").reset_index()

    final = _model(loss="poisson").fit(feat[FEATURES], feat["units"])
    final_lo = _model(loss="quantile", quantile=0.1).fit(feat[FEATURES], feat["units"])
    final_hi = _model(loss="quantile", quantile=0.9).fit(feat[FEATURES], feat["units"])
    return {"test": test, "metrics": metrics, "per_sku": per_sku, "importance": importance,
            "resid_std": resid_std, "models": (final, final_lo, final_hi)}


def forecast_future(df: pd.DataFrame, models, horizon: int = HORIZON) -> pd.DataFrame:
    """Forecast the next `horizon` days. Lags >= horizon are all known today."""
    last = df["date"].max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    keys = df[["sku", "category", "warehouse", "unit_cost", "lead_time_days"]].drop_duplicates()
    latest_price = df[df["date"] == last][["sku", "warehouse", "price"]]
    fut = keys.merge(pd.DataFrame({"date": future_dates}), how="cross").merge(latest_price, on=["sku", "warehouse"])
    fut["units"] = np.nan
    fut["promo"] = 0  # no promotions planned in the forecast window
    full = build_features(pd.concat([df, fut], ignore_index=True))
    out = full[full["date"] > last].copy()
    final, lo, hi = models
    out["forecast"] = final.predict(out[FEATURES])
    out["lo"] = lo.predict(out[FEATURES])
    out["hi"] = hi.predict(out[FEATURES])
    return out[["date", "sku", "category", "warehouse", "forecast", "lo", "hi", "unit_cost", "lead_time_days"]]
