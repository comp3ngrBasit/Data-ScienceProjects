"""Explainable ML for the forecasting models: SHAP, permutation importance, partial dependence,
feature correlation and an auto-generated explainability report (from computed values only)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from app.config import MODELS_DIR, RANDOM_SEED
from app.models.forecasting.engine import series_specs, split_points
from app.data.feature_engineering import supervised_frame
from app.utils.helpers import load_json, save_json, save_table
from app.utils.logging import get_logger

log = get_logger("explain")

FEATURE_GROUPS = {"lag_": "recent-history lags", "roll_": "rolling statistics", "year_": "annual seasonality",
                  "day_": "daily cycle", "hour": "hour of day", "dayofweek": "day of week", "is_": "calendar flags",
                  "month": "month", "dayofyear": "annual seasonality"}


def group_of(f: str) -> str:
    return next((g for p, g in FEATURE_GROUPS.items() if f.startswith(p)), "other")


def explain(key: str, spec_key: str, horizon: int) -> dict | None:
    path = MODELS_DIR / "forecasting" / f"{key}_XGBoost.joblib"
    if not path.exists():
        log.warning("No XGBoost artifact for %s", key)
        return None
    import shap
    from sklearn.inspection import partial_dependence, permutation_importance

    bundle = joblib.load(path)
    model, feats = bundle["model"], bundle["features"]
    spec = series_specs()[spec_key]
    frame = supervised_frame(spec.y, horizon, spec.country, season=spec.season if spec.season > 1 else 1).dropna()
    _, val_end = split_points(spec.y)
    test = frame.loc[val_end:].iloc[1:]
    rng = np.random.default_rng(RANDOM_SEED)
    sample = test.iloc[np.sort(rng.choice(len(test), min(2000, len(test)), replace=False))]
    sv = shap.TreeExplainer(model).shap_values(sample[feats])
    shap_df = pd.DataFrame(sv, columns=feats, index=sample.index)
    imp = shap_df.abs().mean().sort_values(ascending=False)
    shap_imp = pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values, "share_pct": imp.values / imp.sum() * 100,
                             "group": [group_of(f) for f in imp.index]})
    perm = permutation_importance(model, sample[feats], sample["y"], n_repeats=10, random_state=RANDOM_SEED,
                                  scoring="neg_mean_absolute_error")
    perm_df = pd.DataFrame({"feature": feats, "mae_increase": perm.importances_mean, "std": perm.importances_std}) \
        .sort_values("mae_increase", ascending=False)
    top = list(imp.index[:4])
    pdps = []
    for f in top:
        r = partial_dependence(model, sample[feats].astype(float), [f], grid_resolution=30, kind="average")
        pdps.append(pd.DataFrame({"feature": f, "value": r["grid_values"][0], "pdp": r["average"][0]}))
    corr = sample[list(imp.index[:12]) + ["y"]].corr(method="spearman")
    beeswarm = shap_df[top[:6]].stack().rename("shap").reset_index()
    beeswarm.columns = ["timestamp", "feature", "shap"]
    beeswarm["feature_value"] = [sample.at[t, f] for t, f in zip(beeswarm["timestamp"], beeswarm["feature"])]
    groups = shap_imp.groupby("group")["share_pct"].sum().sort_values(ascending=False)
    for name, tab in [("shap_importance", shap_imp), ("permutation_importance", perm_df), ("partial_dependence", pd.concat(pdps)),
                      ("feature_correlation", corr.reset_index(names="feature")), ("shap_beeswarm", beeswarm),
                      ("group_importance", groups.rename("share_pct").reset_index())]:
        save_table(tab, "explainability", f"{key}_{name}")
    fsum = load_json("forecasting", f"{key}_summary") or {}
    report = [f"Model: XGBoost direct {horizon}-step forecaster for {fsum.get('label', key)}, explained on {len(sample)} test observations.",
              "Top features by mean |SHAP|: " + ", ".join(f"{r.feature} ({r.share_pct:.1f} %)" for r in shap_imp.head(5).itertuples()) + ".",
              "Feature groups: " + ", ".join(f"{g} {v:.1f} %" for g, v in groups.items()) + ".",
              f"Permutation importance agrees on the leading driver: {perm_df.iloc[0]['feature']} "
              f"(+{perm_df.iloc[0]['mae_increase']:.3g} MAE when shuffled)." if perm_df.iloc[0]["feature"] == shap_imp.iloc[0]["feature"]
              else f"Permutation importance ranks {perm_df.iloc[0]['feature']} first, SHAP ranks {shap_imp.iloc[0]['feature']} first; "
                   "rankings differ because correlated lag features share credit differently under the two methods.",
              "SHAP values explain the model, not the physical system: they show what the model relies on, not causes."]
    out = {"key": key, "n_explained": int(len(sample)), "top_features": shap_imp.head(10).to_dict("records"),
           "group_importance": groups.to_dict(), "report": report}
    save_json(out, "explainability", f"{key}_summary")
    return out


def run() -> dict:
    res = {}
    for key, spec_key, h in [("household_hourly_h24", "household_hourly", 24), ("grid_hourly_h24", "grid_hourly", 24),
                             ("household_daily_h30", "household_daily", 30)]:
        r = explain(key, spec_key, h)
        if r:
            res[key] = r
    save_json({"available": list(res)}, "explainability", "index")
    return res
