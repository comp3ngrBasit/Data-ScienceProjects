"""Integration tests on the real pipeline outputs, the API and the dashboard (skipped if the pipeline has not run)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.config import PROCESSED_DIR, RESULTS_DIR
from app.utils.helpers import load_json, load_table

pytestmark = pytest.mark.skipif(not (RESULTS_DIR / "overview" / "kpis.json").exists(),
                                reason="run `python -m app.pipeline run` first")


def test_processed_data_is_regular_and_complete():
    hh = pd.read_parquet(PROCESSED_DIR / "household_hourly.parquet")
    assert hh.index.is_monotonic_increasing and hh.index.is_unique
    assert (hh.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert hh["energy_kwh"].notna().all() and (hh["energy_kwh"] >= 0).all()
    g = pd.read_parquet(PROCESSED_DIR / "grid_DE_hourly.parquet")
    assert g["load_mw"].notna().all() and g.index.tz is not None


def test_uci_documented_facts():
    q = load_json("data_quality", "household")["quality_report"]
    assert q["n_rows"] == 2_075_259  # documented UCI row count
    assert q["missing_by_column"]["Global_active_power"] == 25_979  # documented missing minutes


def test_kpis_match_underlying_data():
    k = load_json("overview", "kpis")
    hh = pd.read_parquet(PROCESSED_DIR / "household_hourly.parquet")
    assert k["household_total_kwh"] == pytest.approx(hh["energy_kwh"].sum(), rel=1e-6)
    assert k["household_max_hourly_kwh"] == pytest.approx(hh["energy_kwh"].max(), rel=1e-6)
    g = pd.read_parquet(PROCESSED_DIR / "grid_DE_hourly.parquet")
    assert k["grid_max_load_mw"] == pytest.approx(g["load_mw"].max(), rel=1e-6)
    r = load_json("renewable", "summary")["kpis"]
    assert r["vre_penetration_pct"] == pytest.approx(r["total_vre_twh"] / r["total_load_twh"] * 100)


def test_forecast_experiments_are_chronological_and_complete():
    for key in ("household_hourly_h24", "household_hourly_h168", "household_daily_h30", "grid_hourly_h24", "grid_hourly_h168"):
        s = load_json("forecasting", f"{key}_summary")
        p = load_table("forecasting", f"{key}_predictions")
        assert s["train_end"] < s["val_end"]
        assert (pd.to_datetime(p["origin"]) >= pd.Timestamp(s["val_end"])).all()  # all test origins after validation
        assert (pd.to_datetime(p["timestamp"]) > pd.to_datetime(p["origin"])).all()
        models = set(p["model"])
        assert {"SeasonalNaive", "LinearRegression", "RandomForest", "XGBoost", "SARIMA", "SARIMAX", "Prophet", "LSTM"} <= models
        m = load_table("forecasting", f"{key}_metrics")
        assert s["best_model"] == m[~m["model"].str.startswith("TSO")].sort_values("WAPE").iloc[0]["model"]


def test_anomalies_have_required_fields():
    c = load_table("anomaly", "consensus")
    for col in ("timestamp", "observed_kwh", "expected_kwh", "deviation_pct", "severity", "methods", "evidence"):
        assert col in c.columns and c[col].notna().all()


def test_hypothesis_tests_are_holm_corrected():
    t = load_table("statistics", "hypothesis_tests").dropna(subset=["p_value"])
    assert (t["p_value_holm"] >= t["p_value"] - 1e-12).all()


def test_bayesian_converged():
    b = load_json("weather", "summary")["models"]["BayesianRegression"]
    assert b["max_r_hat"] < 1.05 and b["divergences"] == 0


def test_api_endpoints():
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    assert c.get("/api/v1/health").json()["results_available"]
    assert "household_total_kwh" in c.get("/api/v1/kpis").json()["kpis"]
    r = c.get("/api/v1/results/forecasting/grid_hourly_h24_metrics").json()
    assert r["total_rows"] >= 8 and "WAPE" in r["columns"]
    assert c.get("/api/v1/results/../etc/passwd").status_code in (400, 404)
    assert c.get("/api/v1/forecasts/household_hourly_h24").status_code == 200
    assert len(c.get("/api/v1/anomalies", params={"limit": 5}).json()) <= 5
    assert c.post("/api/v1/pipeline/run").status_code == 403  # protected


@pytest.mark.parametrize("page", ["overview", "consumption", "forecasting", "peak_load", "segmentation", "anomalies", "renewable",
                                  "weather", "statistics", "model_performance", "data_explorer"])
def test_dashboard_page_renders(page):
    from streamlit.testing.v1 import AppTest

    script = f"""
import sys; sys.path.insert(0, {str(RESULTS_DIR.parents[1])!r})
from app.dashboard.theme import inject_css
from app.dashboard import {page}
inject_css()
{page}.render()
"""
    at = AppTest.from_string(script, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
