"""Module 6 - Weather sensitivity modelling.

Unit of analysis: daily national load (GWh) vs population-weighted daily weather. Correlations and
coefficients describe associations; they are not causal estimates (calendar, economic activity and
daylight co-vary with weather).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from app.config import COUNTRY_TZ, RANDOM_SEED, SETTINGS
from app.data import ingestion
from app.data import preprocessing as pp
from app.data.feature_engineering import degree_days, holiday_flags
from app.models.evaluation.metrics import point_metrics
from app.utils.helpers import save_json, save_table
from app.utils.logging import get_logger

log = get_logger("weather")
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def build_daily(country: str = "DE") -> tuple[pd.DataFrame, dict]:
    g = pp.load_grid(country)
    w, meta = ingestion.acquire_grid_weather(country, str(g.index.min().date()), str(g.index.max().date()))
    if w is None:
        raise ingestion.DataUnavailableError(meta.get("reason", "weather unavailable"))
    tz = COUNTRY_TZ.get(country, "UTC")
    w = w.tz_convert(tz)
    agg = {c: ("sum" if c == "precipitation" else "mean") for c in w.columns if c != "weather_code"}
    wd = w.resample("D").agg(agg)
    wd = wd[w["temperature"].resample("D").count() >= 22]  # drop partial local days (UTC/local edge)
    wd["temp_max"] = w["temperature"].resample("D").max()
    wd["temp_min"] = w["temperature"].resample("D").min()
    wd.index = wd.index.tz_localize(None)
    load = pp.grid_daily(country)["load_mwh"] / 1e3
    df = wd.join(load.rename("load_gwh"), how="inner").dropna()
    df = df.join(degree_days(df["temperature"]))
    df["dayofweek"] = df.index.dayofweek
    df["is_holiday"] = holiday_flags(df.index, country)
    df["is_offday"] = ((df["dayofweek"] >= 5) | (df["is_holiday"] == 1)).astype(int)
    df["season"] = df.index.month.map(SEASON)
    # Christmas/New-year week has idiosyncratic demand; flagged so models can absorb it
    df["xmas_period"] = (((df.index.month == 12) & (df.index.day >= 22)) | ((df.index.month == 1) & (df.index.day <= 2))).astype(int)
    meta["n_days"] = int(len(df))
    meta["period"] = [str(df.index.min().date()), str(df.index.max().date())]
    return df, meta


def design(df: pd.DataFrame, weather_vars: list[str]) -> pd.DataFrame:
    X = df[weather_vars].copy()
    for d in range(1, 7):
        X[f"dow_{d}"] = (df["dayofweek"] == d).astype(float)
    X["is_holiday"] = df["is_holiday"].astype(float)
    X["xmas_period"] = df["xmas_period"].astype(float)
    return X


def correlations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    wd = df[df["is_offday"] == 0]  # working days only: removes the weekly calendar confounder
    for c in cols:
        for label, d in (("all days", df), ("working days", wd)):
            r, p = stats.pearsonr(d[c], d["load_gwh"])
            z, se = np.arctanh(r), 1 / np.sqrt(len(d) - 3)
            rho, ps = stats.spearmanr(d[c], d["load_gwh"])
            rows.append({"variable": c, "sample": label, "n": len(d), "pearson_r": r, "pearson_p": p,
                         "pearson_ci95_low": np.tanh(z - 1.96 * se), "pearson_ci95_high": np.tanh(z + 1.96 * se),
                         "spearman_rho": rho, "spearman_p": ps})
    return pd.DataFrame(rows)


def household_sensitivity() -> dict:
    """Household daily kWh vs local weather (Open-Meteo, Sceaux). Runs only when that weather can be fetched."""
    w, meta = ingestion.acquire_household_weather()
    if w is None:
        return {"status": "unavailable", **meta}
    w = w.tz_convert("Europe/Paris")
    w.index = w.index.tz_localize(None)  # household timestamps are tz-naive local time
    agg = {c: ("sum" if c == "precipitation" else "mean") for c in w.columns if c != "weather_code"}
    wd = w.resample("D").agg(agg)
    hd = pp.load_household_daily()
    df = wd.join(hd[["energy_kwh", "imputed_share"]], how="inner")
    df = df[df["imputed_share"] < 0.25].dropna()
    df = df.join(degree_days(df["temperature"]))
    df["dayofweek"], df["is_holiday"] = df.index.dayofweek, holiday_flags(df.index, "FR")
    df["xmas_period"] = 0
    vars_ = [c for c in ["temperature", "humidity", "wind_speed", "solar_radiation", "precipitation", "hdd", "cdd"] if c in df]
    corr = correlations(df.assign(load_gwh=df["energy_kwh"], is_offday=((df["dayofweek"] >= 5) | (df["is_holiday"] == 1)).astype(int)), vars_)
    core = ["hdd", "cdd"] + [c for c in ["humidity", "wind_speed", "solar_radiation", "precipitation"] if c in df]
    ols = sm.OLS(df["energy_kwh"], sm.add_constant(design(df, core))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    cal = sm.OLS(df["energy_kwh"], sm.add_constant(design(df, []))).fit()
    coef = pd.DataFrame({"coef": ols.params, "se_hac": ols.bse, "p_value": ols.pvalues, "ci95_low": ols.conf_int()[0],
                         "ci95_high": ols.conf_int()[1]}).reset_index(names="term")
    save_table(df, "weather", "household_daily")
    save_table(corr.rename(columns=lambda c: c), "weather", "household_correlations")
    save_table(coef, "weather", "household_ols_coefficients")
    return {"status": "ok", **meta, "n_days": int(len(df)), "r2": ols.rsquared, "calendar_only_r2": cal.rsquared,
            "hdd_kwh_per_degree_day": float(ols.params["hdd"]), "hdd_p": float(ols.pvalues["hdd"])}


def run(country: str = "DE") -> dict:
    df, meta = build_daily(country)
    weather_avail = [c for c in ["temperature", "humidity", "wind_speed", "solar_radiation", "precipitation"] if c in df]
    core = ["hdd", "cdd"] + [c for c in weather_avail if c != "temperature"]
    corr = correlations(df, weather_avail + ["hdd", "cdd"])
    split = df.index[int(len(df) * 0.67)]
    tr, te = df.loc[:split], df.loc[split:].iloc[1:]
    results, preds = {}, pd.DataFrame({"actual": te["load_gwh"]})

    # 1) multiple linear regression with HAC standard errors (daily residuals are autocorrelated)
    Xtr, Xte = sm.add_constant(design(tr, core)), sm.add_constant(design(te, core))
    ols = sm.OLS(tr["load_gwh"], Xtr).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    preds["MultipleLinear"] = ols.predict(Xte)
    coef = pd.DataFrame({"coef": ols.params, "se_hac": ols.bse, "p_value": ols.pvalues,
                         "ci95_low": ols.conf_int()[0], "ci95_high": ols.conf_int()[1]}).reset_index(names="term")
    calendar_only = sm.OLS(tr["load_gwh"], sm.add_constant(design(tr, []))).fit()
    results["MultipleLinear"] = {"r2_train": ols.rsquared, "adj_r2_train": ols.rsquared_adj,
                                 "calendar_only_r2_train": calendar_only.rsquared,
                                 "weather_incremental_r2": ols.rsquared - calendar_only.rsquared}
    # 2) polynomial regression on temperature (cubic) + other weather
    def poly(d):
        X = design(d, [c for c in core if c not in ("hdd", "cdd")])
        t = (d["temperature"] - 10) / 10
        X["t1"], X["t2"], X["t3"] = t, t ** 2, t ** 3
        return sm.add_constant(X)
    pol = sm.OLS(tr["load_gwh"], poly(tr)).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    preds["Polynomial"] = pol.predict(poly(te))
    results["Polynomial"] = {"r2_train": pol.rsquared, "degree": 3}
    # 3) GAM: penalised B-spline smooths of weather + parametric calendar terms (statsmodels GLMGam);
    #    smoothing penalties selected by generalised cross-validation on the training block
    from statsmodels.gam.api import BSplines, GLMGam

    smooth_cols = ["temperature"] + [c for c in weather_avail if c != "temperature"]
    bs = BSplines(tr[smooth_cols], df=[10] + [6] * (len(smooth_cols) - 1), degree=[3] * len(smooth_cols))
    lin_tr = sm.add_constant(design(tr, []))
    best_gcv, alpha_opt, n = np.inf, None, len(tr)
    for a in np.logspace(-1, 4, 11):  # GCV = n * RSS / (n - edf)^2, common penalty weight for all smooths
        fit = GLMGam(tr["load_gwh"], exog=lin_tr, smoother=bs, alpha=[a] * len(smooth_cols)).fit()
        gcv = n * np.sum(fit.resid_response ** 2) / (n - fit.edf.sum()) ** 2
        if gcv < best_gcv:
            best_gcv, alpha_opt, gam = gcv, a, fit
    te_clip = te[smooth_cols].clip(tr[smooth_cols].min(), tr[smooth_cols].max(), axis=1)  # splines cannot extrapolate
    preds["GAM"] = gam.predict(exog=sm.add_constant(design(te, []), has_constant="add"), exog_smooth=te_clip)
    grid_t = np.linspace(tr["temperature"].min(), tr["temperature"].max(), 80)
    grid_frame = pd.DataFrame({c: np.full(80, tr[c].median()) for c in smooth_cols})
    grid_frame["temperature"] = grid_t
    eff = gam.partial_values(0, include_constant=False)  # (values, se) at training points
    order = np.argsort(tr["temperature"].to_numpy())
    gam_curve = pd.DataFrame({"temperature": tr["temperature"].to_numpy()[order], "effect_gwh": eff[0][order],
                              "ci_low": (eff[0] - 1.96 * eff[1])[order], "ci_high": (eff[0] + 1.96 * eff[1])[order]})
    results["GAM"] = {"pseudo_r2": float(1 - gam.deviance / gam.null_deviance), "edf_total": float(gam.edf.sum()),
                      "smooth_terms": smooth_cols, "penalty_weight": float(alpha_opt),
                      "smooth_selection": "generalised cross-validation over a log grid", "gcv": float(best_gcv)}
    # 4-5) tree ensembles on raw features
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import partial_dependence, permutation_importance
    from xgboost import XGBRegressor

    tree_cols = weather_avail + ["dayofweek", "is_holiday", "xmas_period", "temp_max", "temp_min"]
    tree_cols = [c for c in dict.fromkeys(tree_cols) if c in df]
    rf = RandomForestRegressor(400, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1).fit(tr[tree_cols], tr["load_gwh"])
    xgb = XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                       random_state=RANDOM_SEED, n_jobs=2).fit(tr[tree_cols], tr["load_gwh"])
    preds["RandomForest"], preds["XGBoost"] = rf.predict(te[tree_cols]), xgb.predict(te[tree_cols])
    perm = permutation_importance(xgb, te[tree_cols], te["load_gwh"], n_repeats=30, random_state=RANDOM_SEED)
    perm_df = pd.DataFrame({"feature": tree_cols, "importance_mean": perm.importances_mean,
                            "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    import shap

    sv = shap.TreeExplainer(xgb).shap_values(te[tree_cols])
    shap_df = pd.DataFrame(sv, columns=tree_cols, index=te.index)
    shap_imp = shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap_gwh").rename_axis("feature").reset_index()
    pdps = []
    for feat in [c for c in ["temperature", "wind_speed", "solar_radiation", "humidity"] if c in tree_cols]:
        pd_res = partial_dependence(xgb, df[tree_cols], [feat], grid_resolution=40, kind="average")
        pdps.append(pd.DataFrame({"feature": feat, "value": pd_res["grid_values"][0], "pdp_gwh": pd_res["average"][0]}))
    pdp_df = pd.concat(pdps)
    # 6) Bayesian regression -> see app.models.bayesian.regression (posterior predictive merged below if present)
    metrics = pd.DataFrame([{"model": m, **point_metrics(preds["actual"], preds[m])} for m in preds.columns if m != "actual"])
    try:
        from app.models.bayesian.regression import run as bayes_run

        b = bayes_run(df, split, core)
        preds["BayesianRegression"] = b["posterior_mean_test"]
        metrics = pd.concat([metrics, pd.DataFrame([{"model": "BayesianRegression",
                                                     **point_metrics(preds["actual"], preds["BayesianRegression"]),
                                                     "PI_coverage": b["coverage_90"]}])], ignore_index=True)
        results["BayesianRegression"] = {k: v for k, v in b.items() if k != "posterior_mean_test"}
    except Exception as exc:  # noqa: BLE001
        log.exception("Bayesian regression failed")
        results["BayesianRegression"] = {"status": "failed", "error": str(exc)}
    metrics = metrics.sort_values("RMSE").reset_index(drop=True)
    # seasonal sensitivity: working-day load vs temperature slope by season (GWh per degC)
    seas = []
    wd = df[df["is_offday"] == 0]
    for sname, d in wd.groupby("season"):
        lr = stats.linregress(d["temperature"], d["load_gwh"])
        seas.append({"season": sname, "slope_gwh_per_c": lr.slope, "slope_ci95_low": lr.slope - 1.96 * lr.stderr,
                     "slope_ci95_high": lr.slope + 1.96 * lr.stderr, "p_value": lr.pvalue, "r2": lr.rvalue ** 2, "n": len(d)})
    seas = pd.DataFrame(seas)
    # weather-related variability: residual std of calendar-only vs calendar+weather models on all data
    cal_all = sm.OLS(df["load_gwh"], sm.add_constant(design(df, []))).fit()
    full_all = sm.OLS(df["load_gwh"], sm.add_constant(design(df, core))).fit()
    hdd_coef = coef.set_index("term").loc["hdd"]
    temp_bins = df.assign(temp_bin=pd.cut(df["temperature"], bins=np.arange(-15, 35, 2.5))).groupby(
        ["temp_bin", "is_offday"], observed=True)["load_gwh"].agg(["mean", "count"]).reset_index()
    temp_bins["temp_bin"] = temp_bins["temp_bin"].astype(str)
    for name, tab in [("daily", df), ("correlations", corr), ("ols_coefficients", coef), ("gam_temperature_curve", gam_curve),
                      ("model_metrics", metrics), ("test_predictions", preds), ("permutation_importance", perm_df),
                      ("shap_importance", shap_imp), ("shap_values", shap_df.join(te[tree_cols], rsuffix="_value")),
                      ("partial_dependence", pdp_df), ("seasonal_sensitivity", seas), ("temperature_bins", temp_bins)]:
        save_table(tab, "weather", name)
    try:
        household = household_sensitivity()
    except Exception as exc:  # noqa: BLE001
        log.exception("Household weather sensitivity failed")
        household = {"status": "failed", "reason": str(exc)}
    summary = {"country": country, "household": household, "weather_source": meta, "variables_available": weather_avail,
               "variables_missing": [v for v in ["humidity", "precipitation"] if v not in weather_avail],
               "train_period": [str(tr.index.min().date()), str(tr.index.max().date())],
               "test_period": [str(te.index.min().date()), str(te.index.max().date())],
               "hdd_base_c": SETTINGS.hdd_base_c, "cdd_base_c": SETTINGS.cdd_base_c,
               "hdd_effect_gwh_per_degree_day": {"coef": hdd_coef["coef"], "ci95": [hdd_coef["ci95_low"], hdd_coef["ci95_high"]],
                                                  "p_value": hdd_coef["p_value"]},
               "cdd_days_in_sample": int((df["cdd"] > 0).sum()),
               "weather_explained_variance": {"calendar_only_r2": cal_all.rsquared, "calendar_plus_weather_r2": full_all.rsquared,
                                              "residual_std_calendar_only_gwh": float(np.std(cal_all.resid)),
                                              "residual_std_with_weather_gwh": float(np.std(full_all.resid))},
               "best_model_test": metrics.iloc[0]["model"], "models": results,
               "causality_note": "Associations only: weather co-varies with daylight, season and economic activity."}
    save_json(summary, "weather", "summary")
    return summary
