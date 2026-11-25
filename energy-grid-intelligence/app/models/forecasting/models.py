"""Forecasting models behind one interface.

Every model is fitted with data strictly before the first test origin. At each walk-forward origin it
receives the history up to (and including) that origin only, and returns mean + lower/upper bounds
for the next ``horizon`` steps.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import RANDOM_SEED
from app.data.feature_engineering import calendar_features, supervised_frame
from app.utils.logging import get_logger

log = get_logger("forecast.models")


@dataclass
class SeriesSpec:
    key: str
    y: pd.Series  # regular, gap-free series
    season: int  # 24 for hourly, 1 for daily (daily lags in days)
    weekly: int  # steps per week (168 or 7)
    country: str
    unit: str


@dataclass
class Forecast:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class BaseForecaster:
    horizon: int
    alpha: float = 0.10
    name: str = "base"
    params: dict = field(default_factory=dict)

    def fit(self, spec: SeriesSpec, train_end: pd.Timestamp, val_end: pd.Timestamp) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, spec: SeriesSpec, origin: pd.Timestamp) -> Forecast:
        raise NotImplementedError


def _future_index(spec: SeriesSpec, origin: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(origin + spec.y.index.freq, periods=h, freq=spec.y.index.freq)


# ------------------------------------------------------------------ baseline
class SeasonalNaive(BaseForecaster):
    """Repeat the last observed week. Interval from empirical residual quantiles on the validation period."""

    def fit(self, spec, train_end, val_end):
        self.name = "SeasonalNaive"
        y = spec.y
        val = y.loc[train_end:val_end].iloc[1:]
        errs = []
        for o in val.index[:: max(1, self.horizon)][:-1]:
            f = self._point(spec, o)
            idx = _future_index(spec, o, self.horizon)
            errs.append(y.reindex(idx).to_numpy() - f)
        self.q = np.nanquantile(np.abs(np.concatenate(errs)), 1 - self.alpha) if errs else 0.0
        return self

    def _point(self, spec, origin):
        hist = spec.y.loc[:origin].to_numpy()
        w = spec.weekly
        return np.array([hist[-w + (k % w)] for k in range(self.horizon)])

    def predict(self, spec, origin):
        m = self._point(spec, origin)
        return Forecast(m, m - self.q, m + self.q)


# ------------------------------------------------------------------ ML (direct, horizon-safe lags)
class MLForecaster(BaseForecaster):
    """Direct multi-horizon regressor on calendar + lags >= horizon; split-conformal intervals."""

    kind: str = "linear"

    def _make(self, params):
        if self.kind == "linear":
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(), LinearRegression())
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **params)
        if self.kind == "xgb":
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=RANDOM_SEED, n_jobs=2, tree_method="hist", **params)
        raise ValueError(self.kind)

    def _grid(self):
        if self.kind == "rf":
            return [{"n_estimators": 150, "max_depth": d, "min_samples_leaf": l, "max_features": 0.5}
                    for d in (10, 16) for l in (3, 10)]
        if self.kind == "xgb":
            return [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.8, "colsample_bytree": 0.8}
                    for n, d, lr in ((300, 4, 0.05), (500, 6, 0.03), (300, 8, 0.05), (800, 4, 0.02))]
        return [{}]

    def fit(self, spec, train_end, val_end):
        self.name = {"linear": "LinearRegression", "rf": "RandomForest", "xgb": "XGBoost"}[self.kind]
        # extend the index by one horizon so features for steps beyond the last observation exist
        # (all lags are >= horizon, so these rows only use observed history)
        ext_index = spec.y.index.append(pd.date_range(spec.y.index[-1] + spec.y.index.freq, periods=self.horizon,
                                                      freq=spec.y.index.freq))
        y_ext = spec.y.reindex(ext_index)
        self.frame = supervised_frame(y_ext, self.horizon, spec.country, season=spec.season if spec.season > 1 else 1)
        self.features = [c for c in self.frame.columns if c != "y"]
        f = self.frame.dropna()
        tr, va = f.loc[:train_end], f.loc[train_end:val_end].iloc[1:]
        if va.empty:
            raise ValueError("validation block is empty")
        best, best_score = None, np.inf
        self.search = []
        for p in self._grid():  # hyper-parameter selection on the chronological validation block
            m = self._make(p).fit(tr[self.features], tr["y"])
            score = float(np.sqrt(np.mean((m.predict(va[self.features]) - va["y"]) ** 2)))
            self.search.append({"params": p, "val_rmse": score})
            if score < best_score:
                best, best_score, model_val = p, score, m
        self.params = best
        resid = va["y"].to_numpy() - model_val.predict(va[self.features])
        self.q = float(np.quantile(np.abs(resid), 1 - self.alpha))
        trv = f.loc[:val_end]
        self.model = self._make(best).fit(trv[self.features], trv["y"])
        return self

    def predict(self, spec, origin):
        idx = _future_index(spec, origin, self.horizon)
        X = self.frame.reindex(idx)[self.features]
        m = self.model.predict(X)
        return Forecast(m, m - self.q, m + self.q)


class LinearForecaster(MLForecaster):
    kind = "linear"


class RFForecaster(MLForecaster):
    kind = "rf"


class XGBForecaster(MLForecaster):
    kind = "xgb"


# ------------------------------------------------------------------ SARIMA / SARIMAX
class SarimaForecaster(BaseForecaster):
    """Order chosen by AIC on the pre-test window; parameters frozen, state re-filtered at each origin."""

    exog: bool = False

    def _exog(self, spec, index):
        cal = calendar_features(index, spec.country, fourier_k=2)
        cols = ["is_offday"]
        if spec.season > 1:  # hourly: weekly Fourier terms give SARIMAX the weekly cycle that s=24 misses
            how = 2 * np.pi * (index.dayofweek * 24 + index.hour) / 168
            for k in (1, 2):
                cal[f"week_sin{k}"], cal[f"week_cos{k}"] = np.sin(k * how), np.cos(k * how)
                cols += [f"week_sin{k}", f"week_cos{k}"]
        else:
            cols += ["year_sin1", "year_cos1", "year_sin2", "year_cos2"]
        return cal[cols].astype(float)

    def fit(self, spec, train_end, val_end):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.name = "SARIMAX" if self.exog else "SARIMA"
        s = spec.season if spec.season > 1 else spec.weekly
        self.window = 24 * 42 if spec.season > 1 else 3 * 365
        y = spec.y.loc[:val_end].iloc[-self.window:]
        cands = ([(1, 0, 1), (2, 0, 1), (1, 0, 2)], [(1, 1, 1), (0, 1, 1)])
        best = (np.inf, None)
        self.search = []
        for order in cands[0]:
            for so in cands[1]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod = SARIMAX(y, order=order, seasonal_order=(*so, s),
                                      exog=self._exog(spec, y.index) if self.exog else None,
                                      enforce_stationarity=False, enforce_invertibility=False)
                        res = mod.fit(disp=False, maxiter=60)
                    self.search.append({"order": order, "seasonal_order": (*so, s), "aic": float(res.aic)})
                    if res.aic < best[0]:
                        best = (res.aic, res, order, so)
                except Exception as exc:  # noqa: BLE001
                    log.warning("SARIMA %s%s failed: %s", order, so, exc)
        _, self.res, order, so = best
        self.params = {"order": order, "seasonal_order": (*so, s), "aic": float(best[0])}
        return self

    def predict(self, spec, origin):
        y = spec.y.loc[:origin].iloc[-self.window:]
        idx = _future_index(spec, origin, self.horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.res.apply(y, exog=self._exog(spec, y.index) if self.exog else None)
            fc = res.get_forecast(self.horizon, exog=self._exog(spec, idx) if self.exog else None)
        ci = fc.conf_int(alpha=self.alpha).to_numpy()
        return Forecast(np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1])


class SarimaxForecaster(SarimaForecaster):
    exog = True


# ------------------------------------------------------------------ Prophet
class ProphetForecaster(BaseForecaster):
    """Refitted at each origin on a trailing window (Prophet has no state-update mechanism)."""

    def fit(self, spec, train_end, val_end):
        self.name = "Prophet"
        self.window = 24 * 730 if spec.season > 1 else 4 * 365
        best = (np.inf, 0.05)
        self.search = []
        val_origin = spec.y.loc[:val_end].index[-self.horizon - 1]
        for cps in (0.01, 0.1):
            self.cps = cps
            f = self.predict(spec, val_origin)
            idx = _future_index(spec, val_origin, self.horizon)
            rmse = float(np.sqrt(np.mean((spec.y.reindex(idx).to_numpy() - f.mean) ** 2)))
            self.search.append({"changepoint_prior_scale": cps, "val_rmse": rmse})
            if rmse < best[0]:
                best = (rmse, cps)
        self.cps = best[1]
        self.params = {"changepoint_prior_scale": self.cps, "window": self.window}
        return self

    def predict(self, spec, origin):
        from prophet import Prophet

        y = spec.y.loc[:origin].iloc[-self.window:]
        df = pd.DataFrame({"ds": y.index, "y": y.values})
        m = Prophet(interval_width=1 - self.alpha, changepoint_prior_scale=getattr(self, "cps", 0.05),
                    daily_seasonality=spec.season > 1, weekly_seasonality=True, yearly_seasonality=True,
                    uncertainty_samples=300)
        m.add_country_holidays(country_name=spec.country)
        m.fit(df)
        fut = pd.DataFrame({"ds": _future_index(spec, origin, self.horizon)})
        p = m.predict(fut)
        return Forecast(p["yhat"].to_numpy(), p["yhat_lower"].to_numpy(), p["yhat_upper"].to_numpy())


def build(name: str, horizon: int, alpha: float) -> BaseForecaster:
    from app.models.forecasting.lstm import LSTMForecaster

    registry = {"SeasonalNaive": SeasonalNaive, "LinearRegression": LinearForecaster, "RandomForest": RFForecaster,
                "XGBoost": XGBForecaster, "SARIMA": SarimaForecaster, "SARIMAX": SarimaxForecaster,
                "Prophet": ProphetForecaster, "LSTM": LSTMForecaster}
    return registry[name](horizon=horizon, alpha=alpha)
