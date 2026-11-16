"""Demand forecasting engine — horizons 7 / 30 / 60 / 90 days.

Models: Moving Average (baseline), Holt-Winters exponential smoothing, Prophet (UK holidays),
LightGBM and XGBoost (global DIRECT multi-horizon models over all planning SKUs: one model predicts mean daily
demand for each of the next 13 weeks from the state at the forecast origin — no recursive feedback, so errors
cannot compound; weekly predictions are spread over days with each SKU's historical weekday profile).

Protocol: the last `validation_days` (91 = the full 13-week horizon) are held out; every model is trained on
data before the cutoff and forecasts the whole hold-out window from the cutoff without seeing it. Metrics: MAE, RMSE, MAPE (non-zero days), WAPE, weekly WAPE. The champion is the model with
the best average rank over MAE, RMSE, MAPE, WAPE and weekly WAPE; all models are then refitted on full history and forecast 90 days ahead.
Results are written to model_runs, forecast_metrics and forecasts.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from scip.core.config import get_settings
from scip.core.db import get_engine, read_sql, write_frame
from scip.core.logging import get_logger
from scip.ingestion import m5
from scip.ingestion.pipeline import as_of_date, raw_dir
from scip.integrations import weather
from scip.ml import metrics as M
from scip.ml.features import (MIN_HISTORY, build_direct_training, direct_feature_names, direct_features, dow_profile,
                               exogenous_features, weekly_to_daily)

log = get_logger("ml.forecasting")
MA, HW, PROPHET, LGBM, XGB = "Moving Average", "Holt-Winters", "Prophet", "LightGBM", "XGBoost"
ALL_MODELS = (MA, HW, PROPHET, LGBM, XGB)


@dataclass
class ForecastConfig:
    horizon: int = 90
    validation_days: int = 91     # 13 weeks: validates the full 90-day horizon
    models: tuple[str, ...] = ALL_MODELS
    ma_window: int = 28
    random_state: int = 42
    skus: list[str] | None = None


@dataclass
class ForecastResult:
    champion: str
    overall: dict[str, dict]
    sku_metrics: pd.DataFrame
    forecasts: pd.DataFrame          # sku, date, model, yhat, yhat_lower, yhat_upper
    validation: pd.DataFrame         # sku, date, model, actual, predicted
    feature_importance: pd.DataFrame
    exogenous_used: list[str]
    notes: list[str] = field(default_factory=list)
    run_ids: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


# --------------------------------------------------------------------------- statistical models
def ma_forecast(h: np.ndarray, horizon: int, window: int) -> np.ndarray:
    h = h[~np.isnan(h)]
    return np.full(horizon, float(h[-window:].mean()) if h.size else 0.0)


def hw_forecast(h: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    h = h[~np.isnan(h)]
    if h.size < 28 or h.sum() == 0:
        return ma_forecast(h, horizon, 28)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(h, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                       initialization_method="estimated").fit()
            fc = np.asarray(fit.forecast(horizon), float)
            if not np.isfinite(fc).all():
                raise ValueError
        except Exception:
            fc = ma_forecast(h, horizon, 28)
    return np.clip(fc, 0, None)


def prophet_forecast(h: np.ndarray, dates: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    mask = ~np.isnan(h)
    df = pd.DataFrame({"ds": dates[mask], "y": h[mask]})
    if len(df) < 28 or df["y"].sum() == 0:
        return ma_forecast(h, horizon, 28)
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                seasonality_mode="additive", changepoint_prior_scale=0.05)
    m.add_country_holidays("UK")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df)
    fut = pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon)})
    return np.clip(m.predict(fut)["yhat"].values, 0, None)


# --------------------------------------------------------------------------- ML models
def _ml_model(name: str, seed: int):
    if name == LGBM:
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                 objective="tweedie", tweedie_variance_power=1.3, random_state=seed, verbose=-1)
    return xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                            colsample_bytree=0.8, objective="reg:squarederror", random_state=seed, n_jobs=-1,
                            verbosity=0)


def _recursive(model, Yn: np.ndarray, start: int, horizon: int, exog: np.ndarray, cats: np.ndarray) -> np.ndarray:
    Y = np.full((Yn.shape[0], start + horizon), np.nan)
    Y[:, :start] = Yn[:, :start]
    for h in range(horizon):
        t = start + h
        X = np.nan_to_num(features_at(Y, t, exog, cats), nan=0.0)
        Y[:, t] = np.clip(model.predict(X), 0, None)
    return Y[:, start:]


def _scale(Y: np.ndarray, end: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanmean(Y[:, :end], axis=1)
    return np.where(np.isnan(s) | (s < 0.1), 0.1, s)


# --------------------------------------------------------------------------- engine
class DemandForecaster:
    def __init__(self, cfg: ForecastConfig | None = None, engine=None):
        self.cfg = cfg or ForecastConfig()
        self.engine = engine or get_engine()

    def load_matrix(self) -> tuple[pd.DataFrame, pd.Series]:
        q = ("SELECT s.date, s.sku, SUM(s.quantity) AS quantity FROM sales_daily s JOIN products p ON p.sku=s.sku "
             "WHERE p.is_planning_sku GROUP BY s.date, s.sku")
        d = read_sql(q, engine=self.engine, parse_dates=["date"])
        if self.cfg.skus:
            d = d[d["sku"].isin(self.cfg.skus)]
        if d.empty:
            raise ValueError("No planning-SKU demand found. Run ingestion first.")
        end = as_of_date(self.engine)
        wide = d.pivot_table(index="date", columns="sku", values="quantity", aggfunc="sum")
        wide = wide.reindex(pd.date_range(wide.index.min(), end, freq="D"))
        first = d.groupby("sku")["date"].min()
        for c in wide.columns:
            m = wide.index >= first[c]
            wide.loc[m, c] = wide.loc[m, c].fillna(0.0)
        cats = read_sql("SELECT sku, category FROM products", engine=self.engine).set_index("sku")["category"]
        return wide, cats.reindex(wide.columns).fillna("Other")

    def _fit_predict(self, Y, dates_all, cutoff, horizon, exog, cats, models, progress):
        preds: dict[str, np.ndarray] = {}
        n = Y.shape[0]
        hist_dates = dates_all[:cutoff]
        if MA in models:
            preds[MA] = np.vstack([ma_forecast(Y[i, :cutoff], horizon, self.cfg.ma_window) for i in range(n)])
        if HW in models:
            preds[HW] = np.vstack([hw_forecast(Y[i, :cutoff], horizon) for i in range(n)])
        if PROPHET in models:
            out = []
            for i in range(n):
                out.append(prophet_forecast(Y[i, :cutoff], hist_dates, horizon))
                if progress and i % 25 == 0:
                    progress(f"Prophet {i + 1}/{n}")
            preds[PROPHET] = np.vstack(out)
        fi = None
        ml = [m for m in models if m in (LGBM, XGB)]
        if ml:
            sc = _scale(Y, cutoff)
            Yn = Y[:, :cutoff] / sc[:, None]
            X, y = build_direct_training(Yn, cutoff, exog, cats)
            Xf = direct_features(Yn, cutoff, horizon, exog, cats)
            profile = dow_profile(Y[:, :cutoff], dates_all[:cutoff])
            for name in ml:
                if progress:
                    progress(f"Training {name} (direct multi-horizon) on {len(y):,} SKU-week samples")
                model = _ml_model(name, self.cfg.random_state).fit(X, y)
                if name == LGBM:
                    fi = model.feature_importances_
                weekly = np.clip(model.predict(Xf), 0, None).reshape(Y.shape[0], -1)   # mean daily demand per week
                preds[name] = weekly_to_daily(weekly, profile, dates_all[cutoff:cutoff + horizon]) * sc[:, None]
        return preds, fi

    def run(self, progress=None, persist: bool = True) -> ForecastResult:
        t0 = time.time()
        cfg, notes = self.cfg, []
        wide, cat_series = self.load_matrix()
        skus = list(wide.columns)
        Y = wide.T.values.astype(float)
        n, T = Y.shape
        V, H = cfg.validation_days, cfg.horizon
        if T < V + MIN_HISTORY + 28:
            raise ValueError(f"Not enough history ({T} days) for a {V}-day validation window.")
        dates_all = pd.date_range(wide.index[0], periods=T + H, freq="D")
        as_of = wide.index[-1]

        events = m5.event_features(raw_dir("m5"), dates_all)
        wx = weather.weather_features("london", dates_all, as_of, engine=self.engine)
        exog_df = exogenous_features(dates_all, events, wx)
        used = ["calendar", "uk_holidays"] + (["m5_events"] if events is not None else []) + \
            (["weather_open_meteo"] if wx is not None else [])
        if events is not None:
            notes.append(f"M5 calendar events used as features (covers {events.attrs.get('coverage', 0):.0%} of dates).")
        if wx is None:
            notes.append("Weather features not used: no Open-Meteo data cached (API unreachable or not synced).")
        exog = exog_df.values.astype(float)
        cats = pd.Categorical(cat_series).codes.astype(float)

        # ---- validation
        cutoff = T - V
        val, _ = self._fit_predict(Y, dates_all, cutoff, V, exog, cats, cfg.models, progress)
        actual = Y[:, cutoff:]
        overall, rows, vrows = {}, [], []
        vdates = wide.index[cutoff:]
        for name, P in val.items():
            overall[name] = M.regression_report(actual.ravel(), P.ravel())
            for i, sku in enumerate(skus):
                rows.append(dict(sku=sku, model=name, **{k: v for k, v in M.regression_report(actual[i], P[i]).items()
                                                        if k in ("mae", "rmse", "mape", "wape")}))
                vrows.append(pd.DataFrame({"sku": sku, "date": vdates, "model": name, "actual": actual[i], "predicted": P[i]}))
        sku_metrics = pd.DataFrame(rows)
        # Champion = best average rank across MAE, RMSE, MAPE, WAPE and weekly WAPE. No single metric is
        # sufficient for lumpy wholesale demand (daily WAPE rewards under-forecasting, weekly WAPE rewards
        # the highest flat level during a seasonal ramp), so every metric gets an equal vote.
        rank_metrics = ["mae", "rmse", "mape", "wape", "weekly_wape"]
        ranks = pd.DataFrame(overall).T[rank_metrics].rank(axis=0, method="min")
        champion = ranks.mean(axis=1).idxmin()
        best = {m: pd.DataFrame(overall).T[m].idxmin() for m in rank_metrics}
        notes.append("Best model per metric: " + ", ".join(f"{m} → {b}" for m, b in best.items())
                     + f". Champion by average rank: {champion}.")
        base = overall.get(MA, {}).get("wape")
        if base:
            notes.append(f"Champion {champion}: daily WAPE {overall[champion]['wape']:.1%} (baseline {base:.1%}), weekly WAPE "
                         f"{overall[champion]['weekly_wape']:.1%} (baseline {overall[MA]['weekly_wape']:.1%}), RMSE "
                         f"{overall[champion]['rmse']:.1f} (baseline {overall[MA]['rmse']:.1f}).")

        # ---- final fit + forecast
        final, fi = self._fit_predict(Y, dates_all, T, H, exog, cats, cfg.models, progress)
        fut = pd.date_range(as_of + pd.Timedelta(days=1), periods=H, freq="D")
        frows = []
        for name, P in final.items():
            sd = sku_metrics[sku_metrics["model"] == name].set_index("sku")["rmse"].reindex(skus).fillna(0).values
            for i, sku in enumerate(skus):
                frows.append(pd.DataFrame({"sku": sku, "date": fut, "model": name, "yhat": P[i],
                                           "yhat_lower": np.clip(P[i] - 1.28 * sd[i], 0, None),
                                           "yhat_upper": P[i] + 1.28 * sd[i]}))
        forecasts = pd.concat(frows, ignore_index=True)
        fi_df = pd.DataFrame()
        if fi is not None:
            fi_df = pd.DataFrame({"feature": direct_feature_names(list(exog_df.columns)), "importance": fi}) \
                .sort_values("importance", ascending=False)
        res = ForecastResult(champion, overall, sku_metrics, forecasts, pd.concat(vrows, ignore_index=True),
                             fi_df, used, notes, seconds=time.time() - t0)
        if persist:
            self.persist(res, wide.index[0], as_of)
        log.info("Forecast run done in %.0fs; champion %s", res.seconds, champion)
        return res

    def persist(self, res: ForecastResult, start, end) -> None:
        from scip.core.db import session_scope
        from scip.core.models import ModelRun
        from sqlalchemy import update
        with session_scope(self.engine) as s:
            s.execute(update(ModelRun).where(ModelRun.model_type == "forecast").values(is_champion=False))
            for name, met in res.overall.items():
                run = ModelRun(model_type="forecast", model_name=name, metrics={k: _f(v) for k, v in met.items()},
                               params={"horizon": self.cfg.horizon, "validation_days": self.cfg.validation_days,
                                       "exogenous": res.exogenous_used},
                               data_window=f"{start.date()}..{end.date()}", is_champion=(name == res.champion),
                               notes=" ".join(res.notes))
                s.add(run)
                s.flush()
                res.run_ids[name] = run.run_id
        fm = res.sku_metrics.assign(run_id=res.sku_metrics["model"].map(res.run_ids))
        write_frame(fm.replace([np.inf, -np.inf], np.nan), "forecast_metrics", self.engine)
        fc = res.forecasts.assign(run_id=res.forecasts["model"].map(res.run_ids))
        fc["date"] = fc["date"].dt.date
        write_frame(fc, "forecasts", self.engine)


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def champion_forecast(engine=None) -> pd.DataFrame:
    """Daily forecast of the current champion model (sku, date, yhat, yhat_lower, yhat_upper, model)."""
    return read_sql("SELECT f.sku, f.date, f.yhat, f.yhat_lower, f.yhat_upper, f.model FROM forecasts f "
                    "JOIN model_runs r ON r.run_id=f.run_id WHERE r.is_champion AND r.model_type='forecast'",
                    engine=engine, parse_dates=["date"])


def horizon_summary(engine=None, horizons=(7, 30, 60, 90)) -> pd.DataFrame:
    """Horizon totals per SKU with an approximate 80% interval: total +/- 1.28 * daily RMSE * sqrt(h)
    (champion hold-out RMSE; assumes independent daily errors). Summing clipped daily bands would be wrong."""
    fc = champion_forecast(engine)
    if fc.empty:
        return pd.DataFrame()
    rmse = read_sql("SELECT m.sku, m.rmse FROM forecast_metrics m JOIN model_runs r USING(run_id) "
                    "WHERE r.is_champion AND r.model_type='forecast'", engine=engine).set_index("sku")["rmse"]
    start = fc["date"].min()
    out = fc[["sku"]].drop_duplicates().set_index("sku")
    sd = rmse.reindex(out.index).fillna(0)
    for h in horizons:
        tot = fc[fc["date"] < start + pd.Timedelta(days=h)].groupby("sku")["yhat"].sum().reindex(out.index)
        half = 1.28 * sd * np.sqrt(h)
        out[f"forecast_{h}d"] = tot
        out[f"lower_{h}d"] = (tot - half).clip(lower=0)
        out[f"upper_{h}d"] = tot + half
    out["model"] = fc["model"].iat[0]
    return out.reset_index()
