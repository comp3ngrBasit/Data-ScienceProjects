"""Walk-forward forecasting experiment runner.

Design (per series x horizon):
  * chronological split: train | validation | test (no shuffling)
  * validation block: hyper-parameter selection, early stopping, conformal calibration
  * test block: rolling-origin walk-forward; at each origin models see only data <= origin
  * pooled metrics across all folds + per-fold spread; best model chosen by pooled WAPE
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from app.config import MLFLOW_TRACKING_URI, MODELS_DIR, SETTINGS
from app.data import preprocessing as pp
from app.models.evaluation import metrics as M
from app.models.forecasting.models import SeriesSpec, build
from app.utils.helpers import save_json, save_table, set_seeds
from app.utils.logging import get_logger

log = get_logger("forecast.engine")


@dataclass
class Experiment:
    spec: SeriesSpec
    horizon: int
    n_folds: int
    label: str
    models: tuple[str, ...]


def series_specs() -> dict[str, SeriesSpec]:
    hh = pp.load_household_hourly()["energy_kwh"].asfreq("h")
    hd = pp.load_household_daily()["energy_kwh"].asfreq("D")
    g = pp.grid_local_hourly("DE")
    gl = g["load_mw"].asfreq("h")
    return {
        "household_hourly": SeriesSpec("household_hourly", hh, 24, 168, "FR", "kWh"),
        "household_daily": SeriesSpec("household_daily", hd, 1, 7, "FR", "kWh"),
        "grid_hourly": SeriesSpec("grid_hourly", gl, 24, 168, "DE", "MW"),
    }


def experiments() -> list[Experiment]:
    fc = SETTINGS.forecast
    s = series_specs()
    allm = fc.models
    return [
        Experiment(s["household_hourly"], 24, fc.n_folds_24h, "Household - next 24 h (hourly)", allm),
        Experiment(s["household_hourly"], 168, fc.n_folds_7d, "Household - next 7 days (hourly)", allm),
        Experiment(s["household_daily"], 30, fc.n_folds_30d, "Household - next 30 days (daily)", allm),
        Experiment(s["grid_hourly"], 24, fc.n_folds_24h, "Germany grid - next 24 h (hourly)", allm),
        Experiment(s["grid_hourly"], 168, fc.n_folds_7d, "Germany grid - next 7 days (hourly)", allm),
    ]


def split_points(y: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    n = len(y)
    fc = SETTINGS.forecast
    train_end = y.index[int(n * (1 - fc.test_fraction - fc.val_fraction)) - 1]
    val_end = y.index[int(n * (1 - fc.test_fraction)) - 1]
    return train_end, val_end


def origins(y: pd.Series, val_end: pd.Timestamp, horizon: int, n_folds: int) -> list[pd.Timestamp]:
    pos0, last = y.index.get_loc(val_end), len(y) - 1 - horizon
    # align hourly origins to midnight so that "next 24 h" means the next calendar day
    cand = np.linspace(pos0, last, n_folds).astype(int)
    if y.index.freqstr == "h":
        def snap(p: int) -> int:  # move to the closest preceding 23:00 so forecasts start at midnight
            back = p - (y.index[p].hour - 23) % 24
            return back if back >= pos0 else back + 24
        cand = [snap(int(p)) for p in cand]
    return [y.index[p] for p in sorted(set(int(c) for c in cand)) if p <= last]


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("energy-grid-intelligence-forecasting")
        return mlflow
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow unavailable: %s", exc)
        return None


def tso_benchmark(exp: Experiment, orig: list[pd.Timestamp]) -> pd.DataFrame | None:
    """ENTSO-E day-ahead TSO load forecast as an external benchmark (grid, 24 h only)."""
    if exp.spec.key != "grid_hourly" or exp.horizon != 24:
        return None
    tso = pp.grid_local_hourly("DE")["tso_forecast_mw"]
    rows = []
    for i, o in enumerate(orig):
        idx = pd.date_range(o + pd.Timedelta(hours=1), periods=24, freq="h")
        rows.append(pd.DataFrame({"fold": i, "origin": o, "timestamp": idx, "step": np.arange(1, 25),
                                  "actual": exp.spec.y.reindex(idx).to_numpy(), "pred": tso.reindex(idx).to_numpy(),
                                  "lower": np.nan, "upper": np.nan, "model": "TSO day-ahead (ENTSO-E)"}))
    return pd.concat(rows)


def metrics_table(P: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per model across all walk-forward origins, plus fold spread and skill vs seasonal naive."""
    rows = []
    for name, d in P.groupby("model", sort=False):
        pooled = {**M.point_metrics(d["actual"], d["pred"]), **M.interval_metrics(d["actual"], d["lower"], d["upper"])}
        per_fold = d.groupby("fold").apply(lambda g: M.point_metrics(g["actual"], g["pred"])["WAPE"], include_groups=False)
        rows.append({"model": name, **pooled, "WAPE_fold_std": float(per_fold.std()), "n_folds": int(d["fold"].nunique())})
    met = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
    base = met.loc[met["model"] == "SeasonalNaive", "WAPE"]
    met["skill_vs_naive_pct"] = (1 - met["WAPE"] / base.iloc[0]) * 100 if len(base) else np.nan
    return met


def rescore(key: str) -> pd.DataFrame:
    """Re-derive the metrics table from stored predictions (no retraining), applying the same post-processing."""
    from app.utils.helpers import load_table

    P = load_table("forecasting", f"{key}_predictions")
    if (P["actual"].dropna() >= 0).all():
        P["lower"] = P["lower"].clip(lower=0)
    save_table(P, "forecasting", f"{key}_predictions")
    met = metrics_table(P)
    save_table(met, "forecasting", f"{key}_metrics")
    return met


def run_experiment(exp: Experiment, mlf=None) -> dict:
    set_seeds()
    y = exp.spec.y
    train_end, val_end = split_points(y)
    orig = origins(y, val_end, exp.horizon, exp.n_folds)
    key = f"{exp.spec.key}_h{exp.horizon}"
    log.info("%s: train<=%s val<=%s, %d folds", key, train_end, val_end, len(orig))
    preds, info = [], {}
    for name in exp.models:
        t0 = time.time()
        try:
            model = build(name, exp.horizon, SETTINGS.forecast.interval_alpha).fit(exp.spec, train_end, val_end)
            for i, o in enumerate(orig):
                f = model.predict(exp.spec, o)
                idx = pd.date_range(o + y.index.freq, periods=exp.horizon, freq=y.index.freq)
                preds.append(pd.DataFrame({"fold": i, "origin": o, "timestamp": idx, "step": np.arange(1, exp.horizon + 1),
                                           "actual": y.reindex(idx).to_numpy(), "pred": f.mean, "lower": f.lower,
                                           "upper": f.upper, "model": name}))
            info[name] = {"params": getattr(model, "params", {}), "search": getattr(model, "search", None),
                          "history": getattr(model, "history", None), "seconds": round(time.time() - t0, 1),
                          "status": "ok"}
            if name in ("XGBoost", "RandomForest", "LinearRegression"):
                path = MODELS_DIR / "forecasting" / f"{key}_{name}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": model.model, "features": model.features, "horizon": exp.horizon,
                             "conformal_q": model.q, "trained_until": str(val_end)}, path, compress=3)
                info[name]["artifact"] = str(path.relative_to(MODELS_DIR.parent))
            if name == "LSTM":
                import torch
                path = MODELS_DIR / "forecasting" / f"{key}_LSTM.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.net.state_dict(), path)
            log.info("  %-16s done in %.1fs", name, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - one failing model must not kill the comparison
            log.exception("  %s failed", name)
            info[name] = {"status": "failed", "error": str(exc)}
    bench = tso_benchmark(exp, orig)
    if bench is not None:
        preds.append(bench)
    P = pd.concat(preds, ignore_index=True)
    if (y.dropna() >= 0).all():  # consumption cannot be negative: truncate symmetric intervals at 0
        P["lower"] = P["lower"].clip(lower=0)
    met = metrics_table(P)
    candidates = met[~met["model"].str.startswith("TSO")]
    best = candidates.iloc[0]["model"]
    bd = P[P["model"] == best].copy()
    bd["resid"] = bd["actual"] - bd["pred"]
    diag = M.residual_diagnostics(bd.sort_values("timestamp")["resid"], lags=min(24, exp.horizon))
    step_err = P.assign(ae=(P["actual"] - P["pred"]).abs()).groupby(["model", "step"])["ae"].mean().reset_index()
    fold_err = P.groupby(["model", "fold", "origin"]).apply(
        lambda g: pd.Series({"WAPE": M.point_metrics(g["actual"], g["pred"])["WAPE"]}), include_groups=False).reset_index()
    # drift: target distribution in test vs training, plus fold-level error trend for the best model
    psi_target = M.psi(y.loc[:train_end].to_numpy(), y.loc[val_end:].to_numpy())
    bf = fold_err[fold_err["model"] == best]
    drift = {"psi_target_test_vs_train": psi_target,
             "psi_interpretation": "stable" if psi_target < 0.1 else "moderate shift" if psi_target < 0.25 else "significant shift",
             "best_model_wape_trend_per_fold": float(np.polyfit(np.arange(len(bf)), bf["WAPE"], 1)[0]) if len(bf) > 2 else None}
    save_table(P, "forecasting", f"{key}_predictions")
    save_table(met, "forecasting", f"{key}_metrics")
    save_table(step_err, "forecasting", f"{key}_error_by_step")
    save_table(fold_err, "forecasting", f"{key}_error_by_fold")
    summary = {"key": key, "label": exp.label, "unit": exp.spec.unit, "horizon": exp.horizon,
               "train_end": str(train_end), "val_end": str(val_end), "test_end": str(y.index[-1]),
               "origins": [str(o) for o in orig], "best_model": best, "selection_metric": "pooled WAPE",
               "interval_level": 1 - SETTINGS.forecast.interval_alpha, "model_info": info,
               "residual_diagnostics_best": diag, "drift": drift}
    save_json(summary, "forecasting", f"{key}_summary")
    if mlf:
        for r in met.to_dict("records"):
            try:
                with mlf.start_run(run_name=f"{key}:{r['model']}"):
                    mlf.log_params({"series": exp.spec.key, "horizon": exp.horizon, "model": r["model"],
                                    **{k: str(v)[:250] for k, v in (info.get(r["model"], {}).get("params") or {}).items()}})
                    mlf.log_metrics({k: float(r[k]) for k in ("MAE", "RMSE", "WAPE", "R2", "PI_coverage")
                                     if pd.notna(r.get(k))})
            except Exception as exc:  # noqa: BLE001
                log.warning("MLflow logging failed: %s", exc)
    return summary


def run(only: list[str] | None = None) -> list[dict]:
    mlf = _mlflow()
    out = []
    for exp in experiments():
        key = f"{exp.spec.key}_h{exp.horizon}"
        if only and key not in only:
            continue
        out.append(run_experiment(exp, mlf))
    index = [{"key": s["key"], "label": s["label"], "best_model": s["best_model"]} for s in out]
    if not only:
        save_json(index, "forecasting", "index")
    return out


def forward_forecast(spec_key: str, horizon: int) -> dict:
    """Forecast beyond the last observation with the best walk-forward model for this series/horizon.

    The model is re-estimated with its validation block moved to the end of the data, so the forecast uses
    every observation available. Datasets are historical, so the forecast period lies in the past relative
    to today; there are no actuals to compare it with.
    """
    from app.utils.helpers import load_json

    key = f"{spec_key}_h{horizon}"
    summ = load_json("forecasting", f"{key}_summary")
    if not summ:
        raise FileNotFoundError(f"run the forecasting stage first ({key})")
    spec = series_specs()[spec_key]
    y = spec.y
    val_len = len(y.loc[summ["train_end"]:summ["val_end"]]) - 1
    train_end, val_end = y.index[-val_len - 1], y.index[-1]
    best = summ["best_model"]
    model = build(best, horizon, SETTINGS.forecast.interval_alpha).fit(spec, train_end, val_end)
    f = model.predict(spec, val_end)
    idx = pd.date_range(val_end + y.index.freq, periods=horizon, freq=y.index.freq)
    lower = np.clip(f.lower, 0, None) if (y.dropna() >= 0).all() else f.lower
    out = pd.DataFrame({"timestamp": idx, "pred": f.mean, "lower": lower, "upper": f.upper})
    save_table(out, "forecasting", f"{key}_forward")
    res = {"key": key, "model": best, "issued_at_data_end": str(val_end), "unit": spec.unit,
           "total_next_period": float(out["pred"].sum()), "peak_next_period": float(out["pred"].max()),
           "peak_time": str(out.loc[out["pred"].idxmax(), "timestamp"]),
           "interval_level": 1 - SETTINGS.forecast.interval_alpha}
    save_json(res, "forecasting", f"{key}_forward")
    return res
