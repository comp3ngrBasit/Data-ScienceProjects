"""Unit tests on small synthetic fixtures (test-only data; never used by the application)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from app.analytics.peak_analysis import extreme_value, high_demand_events, in_window, peak_window
from app.data.cleaning import fill_gaps
from app.data.feature_engineering import lag_features, safe_lags, supervised_frame
from app.data.validation import ValidationError, validate_household
from app.models.anomaly_detection.detector import baseline, severity
from app.models.evaluation.metrics import interval_metrics, point_metrics, psi
from app.utils.helpers import safe_mape


@pytest.fixture
def hourly_series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=24 * 7 * 30, freq="h")
    daily = 1 + 0.5 * np.sin(2 * np.pi * idx.hour / 24)
    weekly = 1 + 0.2 * (idx.dayofweek >= 5)
    return pd.Series(daily * weekly + rng.normal(0, 0.05, len(idx)), index=idx, name="y")


def _minute_frame(n=120):
    ts = pl.datetime_range(pd.Timestamp("2007-01-01"), pd.Timestamp("2007-01-01") + pd.Timedelta(minutes=n - 1), "1m", eager=True)
    return pl.DataFrame({"timestamp": ts, "Global_active_power": [1.2] * n, "Global_reactive_power": [0.1] * n,
                         "Voltage": [240.0] * n, "Global_intensity": [5.0] * n, "Sub_metering_1": [0.0] * n,
                         "Sub_metering_2": [1.0] * n, "Sub_metering_3": [10.0] * n})


# ------------------------------------------------------------------ validation / cleaning
def test_validation_passes_on_consistent_data():
    rep = validate_household(_minute_frame())
    assert rep.n_rows == 120 and rep.duplicate_timestamps == 0
    assert all(c["passed"] for c in rep.checks)


def test_validation_flags_inconsistent_submeters():
    df = _minute_frame().with_columns(pl.lit(50.0).alias("Sub_metering_3"))  # 51 Wh/min > 20 Wh/min total
    rep = validate_household(df)
    assert not next(c for c in rep.checks if "sub-meters" in c["check"])["passed"]


def test_validation_rejects_missing_column():
    with pytest.raises(ValidationError):
        validate_household(_minute_frame().drop("Voltage"))


def test_fill_gaps_short_interpolates_long_uses_past_season():
    idx = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
    s = pd.Series(np.tile(np.arange(24.0), 10), index=idx)
    s.iloc[100:102] = np.nan  # short gap
    s.iloc[200:210] = np.nan  # long gap
    filled, flag = fill_gaps(s, period=24)
    assert flag.sum() == 12 and filled.notna().all()
    assert filled.iloc[205] == pytest.approx(205 % 24)  # same hour of earlier days


# ------------------------------------------------------------------ leakage
@pytest.mark.parametrize("h", [24, 168])
def test_safe_lags_never_below_horizon(h):
    assert min(safe_lags(h)) >= h
    assert min(safe_lags(30, season=1)) >= 30


@pytest.mark.parametrize("h", [24, 168])
def test_features_do_not_use_future_values(hourly_series, h):
    """Changing every value after the origin must not change any feature for targets in (origin, origin+h]."""
    y = hourly_series
    origin = y.index[3000]
    f1 = lag_features(y, h)
    y2 = y.copy()
    y2.loc[y2.index > origin] += 1000.0
    f2 = lag_features(y2, h)
    window = (f1.index > origin) & (f1.index <= origin + pd.Timedelta(hours=h))
    pd.testing.assert_frame_equal(f1[window], f2[window])


def test_supervised_frame_has_target_and_calendar(hourly_series):
    f = supervised_frame(hourly_series, 24, "FR")
    assert "y" in f and "is_holiday" in f and "lag_24" in f and "lag_23" not in f


# ------------------------------------------------------------------ metrics
def test_point_metrics_known_values():
    m = point_metrics([1, 2, 3, 4], [1, 2, 3, 5])
    assert m["MAE"] == pytest.approx(0.25) and m["RMSE"] == pytest.approx(0.5) and m["WAPE"] == pytest.approx(10.0)


def test_safe_mape_ignores_zero_actuals():
    assert safe_mape(np.array([0.0, 2.0]), np.array([5.0, 1.0])) == pytest.approx(50.0)


def test_interval_coverage():
    assert interval_metrics([1, 2, 3, 4], [0, 0, 0, 5], [2, 2, 4, 6])["PI_coverage"] == pytest.approx(75.0)


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(1).normal(size=5000)
    assert psi(x, x) == pytest.approx(0, abs=1e-9)
    assert psi(x, x + 2) > 0.25


# ------------------------------------------------------------------ forecasting models
def test_seasonal_naive_and_linear_forecasters(hourly_series):
    from app.models.forecasting.engine import origins
    from app.models.forecasting.models import LinearForecaster, SeasonalNaive, SeriesSpec

    spec = SeriesSpec("t", hourly_series.asfreq("h"), 24, 168, "FR", "kWh")
    tr, va = spec.y.index[3000], spec.y.index[4000]
    for cls in (SeasonalNaive, LinearForecaster):
        m = cls(horizon=24).fit(spec, tr, va)
        o = origins(spec.y, va, 24, 3)
        assert all(spec.y.index.get_loc(x) + 24 < len(spec.y) for x in o)
        assert all(x.hour == 23 for x in o)  # forecasts start at midnight
        f = m.predict(spec, o[0])
        actual = spec.y.loc[o[0]:].iloc[1:25].to_numpy()
        assert len(f.mean) == 24 and np.all(f.lower <= f.upper)
        assert point_metrics(actual, f.mean)["WAPE"] < 15


# ------------------------------------------------------------------ peak
def test_peak_window_finds_evening_peak():
    idx = pd.date_range("2020-01-01", periods=24 * 20, freq="h")
    s = pd.Series(np.where(np.isin(idx.hour, [19, 20, 21, 22]), 5.0, 1.0), index=idx)
    assert peak_window(s) == (19, 23)
    assert in_window(np.array([23, 1, 22]), 22, 2).tolist() == [True, True, True]


def test_high_demand_events_group_consecutive_hours():
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    s = pd.Series(1.0, index=idx)
    s.iloc[50:53] = 10.0
    ev, thr = high_demand_events(s, q=0.98)
    assert len(ev) == 1 and ev.iloc[0]["duration_h"] == 3


def test_extreme_value_return_levels_increase():
    rng = np.random.default_rng(3)
    s = pd.Series(rng.gumbel(100, 5, 3000), index=pd.date_range("2010-01-01", periods=3000, freq="D"))
    ev = extreme_value(s)
    rl = ev["return_levels"]
    assert rl["1y"] < rl["10y"] and ev["n_clusters"] >= 10


# ------------------------------------------------------------------ anomaly
def test_anomaly_baseline_is_causal(hourly_series):
    b1 = baseline(hourly_series)
    y2 = hourly_series.copy()
    y2.iloc[-1] = 999
    b2 = baseline(y2)
    pd.testing.assert_series_equal(b1, b2)  # last value never influences any baseline value


def test_severity_bands():
    assert [severity(z) for z in (2, 4.5, -6.5, 9)] == ["low", "medium", "high", "critical"]


def test_household_weather_path_runs_when_weather_available(monkeypatch, tmp_path):
    """Exercise the Open-Meteo code path with a synthetic weather fixture (the real API may be unreachable in CI)."""
    from app.analytics import weather_analysis as wa
    from app.config import PROCESSED_DIR

    if not (PROCESSED_DIR / "household_daily.parquet").exists():
        pytest.skip("processed household data missing")
    idx = pd.date_range("2006-12-16", "2010-11-27", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    w = pd.DataFrame({"temperature": 12 + 8 * np.sin(2 * np.pi * (idx.dayofyear - 110) / 365) + rng.normal(0, 2, len(idx)),
                      "humidity": rng.uniform(40, 95, len(idx)), "wind_speed": rng.gamma(2, 5, len(idx)),
                      "solar_radiation": np.clip(rng.normal(150, 100, len(idx)), 0, None), "precipitation": rng.exponential(0.1, len(idx)),
                      "weather_code": 0}, index=idx)
    monkeypatch.setattr(wa.ingestion, "acquire_household_weather", lambda: (w, {"source": "synthetic test fixture"}))
    monkeypatch.setattr(wa, "save_table", lambda *a, **k: None)
    res = wa.household_sensitivity()
    assert res["status"] == "ok" and res["n_days"] > 1000 and "hdd_kwh_per_degree_day" in res
