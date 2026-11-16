"""Unit tests: metrics, formulas, features, ingestion cleaning, integrations (mocked), optimisation primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from scip.ml import metrics as M


# ------------------------------------------------------------------ metrics
def test_regression_metrics():
    a, f = np.array([10, 0, 5, 5]), np.array([8, 2, 5, 9])
    assert M.mae(a, f) == pytest.approx(2.0)
    assert M.rmse(a, f) == pytest.approx(np.sqrt(24 / 4))
    assert M.wape(a, f) == pytest.approx(8 / 20)
    assert M.mape(a, f) == pytest.approx(np.mean([0.2, 0.0, 0.8]))   # zero-actual day excluded
    assert np.isnan(M.wape([0, 0], [1, 1]))


def test_weekly_wape_cancels_within_week():
    a = np.ones(14)
    f = np.tile([0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2)
    assert M.wape(a, f) == pytest.approx(2 / 7)
    assert M.weekly_wape(a, f) == pytest.approx(0.0)


def test_classification_report_and_thresholds():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.8, 0.7, 0.35, 0.2])
    r = M.classification_report(y, p, 0.5)
    assert r["precision"] == 1.0 and r["recall"] == pytest.approx(2 / 3)
    assert r["roc_auc"] == pytest.approx(8 / 9)
    assert 0.3 <= M.youden_threshold(y, p) <= 0.8
    # majority-positive data: F1-optimal threshold degenerates low; Youden does not
    y2 = np.r_[np.ones(70), np.zeros(30)].astype(int)
    p2 = np.r_[np.linspace(0.3, 0.9, 70), np.linspace(0.1, 0.6, 30)]
    assert M.youden_threshold(y2, p2) > M.best_f1_threshold(y2, p2)


# ------------------------------------------------------------------ inventory formulas
def test_safety_stock_and_eoq():
    from scip.optimization.inventory import abc_classes, eoq, safety_stock
    ss = safety_stock(20, 5, 16, 4, 0.98)
    assert float(ss) == pytest.approx(norm.ppf(0.98) * np.sqrt(16 * 25 + 400 * 16))
    assert float(safety_stock(10, 4, 9, 0, 0.95)) == pytest.approx(norm.ppf(0.95) * 4 * 3)
    with pytest.raises(ValueError):
        safety_stock(-1, 1, 1, 1, 0.9)
    assert float(eoq(1000, 50, 2.5)) == pytest.approx(np.sqrt(2 * 1000 * 50 / 2.5))
    assert float(eoq(0, 50, 2.5)) == 0
    cls = abc_classes(pd.Series({"a": 700, "b": 150, "c": 100, "d": 50}))
    assert (cls["a"], cls["c"], cls["d"]) == ("A", "B", "C")


# ------------------------------------------------------------------ features (leakage)
def test_features_use_only_the_past():
    from scip.ml.features import exogenous_features, features_at
    dates = pd.date_range("2011-01-01", periods=120)
    ex = exogenous_features(dates).values.astype(float)
    Y = np.arange(240, dtype=float).reshape(2, 120)
    x1 = features_at(Y, 60, ex, np.array([0, 1]))
    Y2 = Y.copy()
    Y2[:, 60:] = -999
    np.testing.assert_allclose(np.nan_to_num(x1), np.nan_to_num(features_at(Y2, 60, ex, np.array([0, 1]))))


def test_direct_training_targets_and_daily_spread():
    from scip.ml.features import (WEEKS, build_direct_training, direct_features, dow_profile, exogenous_features,
                                  weekly_to_daily)
    dates = pd.date_range("2011-01-03", periods=240)
    ex = exogenous_features(dates).values.astype(float)
    Y = np.tile(np.array([1, 2, 3, 4, 5, 6, 0], float), 20)[None, :].repeat(3, axis=0)
    X, y = build_direct_training(Y, 120, ex, np.zeros(3))
    assert np.allclose(y, 3.0)                       # weekly mean of the repeating pattern
    Xf = direct_features(Y, 120, 90, ex, np.zeros(3))
    assert Xf.shape[0] == 3 * WEEKS and Xf[0, -1] == 1 and Xf[WEEKS - 1, -1] == WEEKS
    prof = dow_profile(Y, dates[:140])
    daily = weekly_to_daily(np.full((3, 2), 3.0), prof, pd.date_range("2011-05-23", periods=14))
    assert daily[:, :7].sum(axis=1) == pytest.approx(21.0)


# ------------------------------------------------------------------ ingestion cleaning
def test_uci_cleaning_rules():
    from scip.ingestion.uci import clean, derive_category
    raw = pd.DataFrame({
        "InvoiceNo": ["1", "C2", "3", "4", "5", "5"], "StockCode": ["85123A", "85123A", "POST", "22423", "22423", "22423"],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER", "x", "POSTAGE", None, "REGENCY CAKESTAND", "REGENCY CAKESTAND"],
        "Quantity": [6, -6, 1, 2, -1, 4], "InvoiceDate": pd.to_datetime(["2011-01-04"] * 6),
        "UnitPrice": [2.55, 2.55, 18.0, 12.75, 12.75, 12.75], "CustomerID": [1, 1, 2, 3, 3, 3],
        "Country": ["United Kingdom", "United Kingdom", "France", "EIRE", "EIRE", "EIRE"]})
    df, st = clean(raw)
    assert st["cancellations_removed"] == 1 and st["non_product_codes_removed"] == 1 and st["non_positive_removed"] == 1
    assert set(df["StockCode"]) == {"85123A", "22423"}
    assert df.loc[df["StockCode"] == "22423", "Description"].iloc[0] == "REGENCY CAKESTAND"   # filled from mode
    assert set(df["location_id"]) == {"CUST-GBR", "CUST-IRL"}
    assert derive_category("WHITE HANGING HEART T-LIGHT HOLDER") in {"Lighting & Candles", "Home Decor"}
    assert derive_category("JUMBO BAG RED RETROSPOT") == "Bags & Storage"


def test_dataco_build_on_real_schema(tmp_path):
    from scip.ingestion.dataco import build
    cols = {"Type": "DEBIT", "Days for shipping (real)": 5, "Days for shipment (scheduled)": 4, "Delivery Status": "Late delivery",
            "Late_delivery_risk": 1, "Category Name": "Cleats", "Customer Segment": "Consumer", "Department Name": "Footwear",
            "Latitude": 18.2, "Longitude": -66.3, "Market": "LATAM", "Order Country": "Mexico", "Order Region": "Central America",
            "order date (DateOrders)": "1/31/2018 22:56", "Order Id": 1, "Order Item Id": 1, "Order Item Discount Rate": 0.1,
            "Order Item Quantity": 2, "Sales": 100.0, "Order Profit Per Order": 10.0, "Order State": "Jalisco",
            "shipping date (DateOrders)": "2/5/2018 22:56", "Shipping Mode": "Standard Class", "Customer Email": "XXXX",
            "Customer Password": "XXXX", "Customer Fname": "A", "Customer Lname": "B", "Customer Street": "S"}
    rows = [dict(cols), {**cols, "Order Item Id": 2, "Order Item Quantity": 3, "Sales": 50.0},
            {**cols, "Order Id": 2, "Order Item Id": 3, "Delivery Status": "Shipping canceled", "Late_delivery_risk": 0}]
    p = tmp_path / "DataCoSupplyChainDataset.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="latin-1")
    out = build(p)
    s = out["shipments"].set_index("shipment_id")
    assert len(s) == 2 and s.at["DC-1", "quantity"] == 5 and s.at["DC-1", "value"] == 150
    assert bool(s.at["DC-1", "is_late"]) and s.at["DC-1", "promised_date"] == pd.Timestamp("2018-02-04").date()
    assert s.at["DC-2", "status"] == "cancelled" and s.at["DC-2", "is_late"] is None
    assert not any("Email" in c or "Password" in c for c in out["shipments"].columns)


def test_olist_customers_only_and_full(tmp_path):
    from scip.ingestion.olist import build
    pd.DataFrame({"customer_id": ["c1", "c2"], "customer_unique_id": ["u1", "u2"], "customer_zip_code_prefix": ["01001", "20000"],
                  "customer_city": ["sao paulo", "rio"], "customer_state": ["SP", "RJ"]}).to_csv(tmp_path / "olist_customers_dataset.csv", index=False)
    out = build(tmp_path)
    assert "shipments" not in out and len(out["locations"]) == 2
    pd.DataFrame({"order_id": ["o1"], "customer_id": ["c1"], "order_status": ["delivered"],
                  "order_purchase_timestamp": ["2018-01-01 10:00"], "order_approved_at": ["2018-01-01"],
                  "order_delivered_carrier_date": ["2018-01-03"], "order_delivered_customer_date": ["2018-01-12"],
                  "order_estimated_delivery_date": ["2018-01-10"]}).to_csv(tmp_path / "olist_orders_dataset.csv", index=False)
    pd.DataFrame({"order_id": ["o1", "o1"], "order_item_id": [1, 2], "product_id": ["p1", "p1"], "seller_id": ["s123456789012345", "s123456789012345"],
                  "shipping_limit_date": ["2018-01-02"] * 2, "price": [10.0, 10.0], "freight_value": [2.0, 2.0]}).to_csv(tmp_path / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame({"seller_id": ["s123456789012345"], "seller_zip_code_prefix": ["13000"], "seller_city": ["campinas"],
                  "seller_state": ["SP"]}).to_csv(tmp_path / "olist_sellers_dataset.csv", index=False)
    out = build(tmp_path)
    s = out["shipments"].iloc[0]
    assert s["quantity"] == 2 and s["value"] == 20 and bool(s["is_late"]) and s["supplier_id"].startswith("OLIST-")
    assert len(out["suppliers"]) == 1


def test_m5_calendar_events(tmp_path):
    from scip.ingestion.m5 import event_features
    pd.DataFrame({"date": ["2011-12-24", "2011-12-25"], "event_name_1": [None, "Christmas"], "event_type_1": [None, "National"],
                  "event_type_2": [None, None]}).to_csv(tmp_path / "calendar.csv", index=False)
    ev = event_features(tmp_path, pd.date_range("2011-12-23", periods=4))
    assert ev["event_national"].tolist() == [0, 0, 1, 0]


# ------------------------------------------------------------------ integrations (mocked network)
def test_countries_fallback_and_parse(monkeypatch):
    from scip.integrations import countries
    from scip.integrations.http import ExternalAPIError

    def boom(*a, **k):
        raise ExternalAPIError("offline")
    monkeypatch.setattr(countries, "get_json", boom)
    df = countries.fetch_countries(["GBR", "BRA"])
    assert set(df["source"]) == {"static_fallback"} and len(df) == 2
    monkeypatch.setattr(countries, "get_json", lambda *a, **k: [
        {"cca3": "GBR", "name": {"common": "United Kingdom"}, "region": "Europe", "subregion": "Northern Europe",
         "latlng": [54, -2], "currencies": {"GBP": {}}}])
    df = countries.fetch_countries(["GBR", "BRA"]).set_index("country_code")
    assert df.at["GBR", "source"] == "restcountries" and df.at["BRA", "source"] == "static_fallback"


def test_weather_parse_and_no_future_leakage(monkeypatch):
    from scip.integrations import weather
    data = {"daily": {"time": ["2011-01-01", "2011-01-02"], "temperature_2m_max": [5.0, 6.0],
                      "temperature_2m_min": [1.0, 2.0], "precipitation_sum": [0.0, 3.2]}}
    df = weather._parse(data, "london", "open-meteo-archive")
    assert list(df["temp_max"]) == [5.0, 6.0] and df["source"].iat[0] == "open-meteo-archive"
    hist = pd.DataFrame({"date": pd.date_range("2010-01-01", "2011-12-31"), "temp_max": 10.0, "precipitation": 1.0})
    hist.loc[hist["date"] >= "2011-06-01", "temp_max"] = 99.0       # "future" observations must not be used
    hist.loc[(hist["date"] >= "2010-06-01") & (hist["date"] < "2011-01-01"), "temp_max"] = 20.0
    monkeypatch.setattr(weather, "read_sql", lambda *a, **k: hist)
    f = weather.weather_features("london", pd.date_range("2011-05-30", periods=5), pd.Timestamp("2011-05-31"))
    assert f["temp_max"].iloc[:2].tolist() == [10.0, 10.0]
    assert f["temp_max"].iloc[2:].tolist() == [20.0, 20.0, 20.0]     # previous-year proxy, not 99


def test_n8n_and_llm_degrade_gracefully(monkeypatch):
    from scip.core.config import get_settings
    from scip.integrations import llm, n8n
    assert n8n.publish("test", {"x": 1})["delivered"] is False                # no URL configured
    monkeypatch.setattr(get_settings(), "n8n_webhook_url", "http://example.invalid/hook")
    sent = {}
    monkeypatch.setattr(n8n, "post_json", lambda url, body: sent.update(url=url, body=body) or 200)
    r = n8n.publish("po.approval_requested", {"po_id": "PO-1"})
    assert r["delivered"] and sent["body"]["event"] == "po.approval_requested" and sent["body"]["po_id"] == "PO-1"
    text, provider = llm.generate("sys", "hello")
    assert text is None and provider == "none"


def test_mock_supplier_portal_is_deterministic():
    from datetime import date
    from scip.integrations.supplier_portal import MockSupplierPortal
    p = MockSupplierPortal()
    a = p.send_purchase_order("PO-1", "V-1", "SKU", 10, date(2011, 12, 9), 12)
    assert a == p.send_purchase_order("PO-1", "V-1", "SKU", 10, date(2011, 12, 9), 12)
    assert a["confirmed_delivery_date"] == "2011-12-21" and a["mock"] is True


# ------------------------------------------------------------------ optimisation primitives
def test_vrp_respects_capacity():
    from scip.optimization.logistics import solve_vrp
    stops = pd.DataFrame({"name": list("ABCDEF"), "lat": [-23, -22, -20, -25, -27, -19], "lng": [-46, -43, -40, -49, -48, -44],
                          "demand": [4, 4, 4, 4, 4, 4]})
    r = solve_vrp(stops, (-23.55, -46.63), vehicles=3, capacity=10, time_limit=2)
    assert sum(len(x["stops"]) for x in r["routes"]) == 6
    assert all(x["load"] <= 10 for x in r["routes"]) and r["total_distance_km"] > 0


def test_approval_levels():
    from scip.services.procurement_workflow import approval_level
    assert approval_level(100) == "auto" and approval_level(10_000) == "manager" and approval_level(100_000) == "director"
