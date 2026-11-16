"""Integration tests: full platform built into the PostgreSQL test database from the real datasets
(20 planning SKUs). Covers ingestion lineage, every ML model, optimisation, procurement workflow,
automation, Copilot, n8n webhook delivery and the REST API."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def q(sql, **p):
    from scip.core.db import read_sql
    return read_sql(sql, p)


# ------------------------------------------------------------------ data
def test_ingestion_lineage(platform):
    from scip.core.db import table_counts
    c = table_counts()
    assert c["products"] > 3000 and c["sales_daily"] > 250_000 and c["shipments"] > 60_000
    assert q("SELECT COUNT(*) n FROM products WHERE is_planning_sku")["n"].iat[0] == 20
    origins = set(q("SELECT DISTINCT data_origin FROM sales_daily")["data_origin"]) | \
        set(q("SELECT DISTINCT data_origin FROM purchase_orders")["data_origin"])
    assert "real:uci" in origins and "simulated" in origins
    assert set(q("SELECT DISTINCT data_origin FROM shipments")["data_origin"]) <= {"real:dataco", "real:olist"}
    src = q("SELECT * FROM data_sources").set_index("source_id")
    assert src.at["uci_online_retail", "origin"] == "real" and src.at["simulated", "origin"] == "simulated"
    # cleaning: no cancellations / negative quantities survive
    assert q("SELECT MIN(quantity) m FROM sales_daily")["m"].iat[0] > 0
    assert len(q("SELECT * FROM locations WHERE location_id LIKE 'BR-CUST-%'")) == 27


def test_no_pii_from_dataco(platform):
    from scip.core.models import Shipment
    cols = [c.name for c in Shipment.__table__.columns]
    assert not any(k in " ".join(cols).lower() for k in ("email", "password", "fname", "lname", "street"))


# ------------------------------------------------------------------ ML
def test_forecasting_models_and_horizons(platform):
    runs = q("SELECT model_name, metrics, is_champion FROM model_runs WHERE model_type='forecast'")
    assert set(runs["model_name"]) == {"Moving Average", "Holt-Winters", "Prophet", "LightGBM", "XGBoost"}
    assert runs["is_champion"].sum() == 1
    for m in runs["metrics"]:
        m = m if isinstance(m, dict) else json.loads(m)
        assert {"mae", "rmse", "mape", "wape", "weekly_wape"} <= set(m) and m["rmse"] >= m["mae"]
    from scip.ml.forecasting import horizon_summary
    hs = horizon_summary()
    assert len(hs) == 20
    assert (hs["forecast_7d"] <= hs["forecast_30d"]).all() and (hs["forecast_60d"] <= hs["forecast_90d"]).all()
    assert (hs["lower_30d"] <= hs["forecast_30d"]).all() and (hs["forecast_30d"] <= hs["upper_30d"]).all()
    fc = q("SELECT f.date FROM forecasts f JOIN model_runs r USING(run_id) WHERE r.is_champion AND r.model_type='forecast'",)
    asof = pd.Timestamp(q("SELECT value FROM app_metadata WHERE key='as_of_date'")["value"].iat[0])
    assert pd.to_datetime(fc["date"]).min() == asof + pd.Timedelta(days=1)


def test_classifiers_have_f1_and_auc(platform):
    for mtype in ("delivery_risk", "stockout"):
        runs = q("SELECT model_name, metrics, is_champion FROM model_runs WHERE model_type=:t", t=mtype)
        assert runs["is_champion"].sum() == 1 and len(runs) >= 2
        for m in runs["metrics"]:
            m = m if isinstance(m, dict) else json.loads(m)
            assert 0 <= m["f1"] <= 1 and 0.5 < m["roc_auc"] <= 1
    dr = platform["delivery_risk"]["results"]
    assert max(r["roc_auc"] for r in dr.values()) > 0.65          # better than chance on real DataCo
    assert q("SELECT COUNT(*) n FROM purchase_orders WHERE status='sent' AND predicted_late_risk IS NULL")["n"].iat[0] == 0


def test_supplier_scores(platform):
    sc = q("SELECT * FROM supplier_scores")
    assert not sc.empty and sc["score"].between(0, 100).all()
    assert set(sc["tier"]) <= {"Strategic", "Preferred", "Watch", "At risk"}


# ------------------------------------------------------------------ optimisation
def test_inventory_policy_consistency(platform):
    from scip.optimization.inventory import latest_policy
    p = latest_policy()
    assert len(p) == 20
    assert (p["reorder_point"] >= p["safety_stock"]).all() and (p["max_level"] >= p["reorder_point"]).all()
    need = (p["on_hand"] + p["on_order"]) <= p["reorder_point"]
    assert (p.loc[~need, "recommended_qty"] == 0).all()


def test_cp_sat_allocation_respects_moq_and_budget(platform):
    from scip.optimization.procurement import Requirement, optimize_allocation
    sku = q("SELECT sku FROM supplier_products GROUP BY sku HAVING COUNT(*) >= 2 LIMIT 1")["sku"].iat[0]
    r = optimize_allocation([Requirement(sku, 500)])
    assert r["status"] == "OPTIMAL" and sum(a["quantity"] for a in r["allocations"]) >= 500
    moq = q("SELECT supplier_id, moq FROM supplier_products WHERE sku=:s", s=sku).set_index("supplier_id")["moq"]
    assert all(a["quantity"] >= moq[a["supplier_id"]] for a in r["allocations"])
    tight = optimize_allocation([Requirement(sku, 500)], budget=10.0)
    assert tight["shortfalls"].get(sku, 0) > 0                      # budget binds -> shortfall reported


def test_logistics_plans(platform):
    from scip.optimization.logistics import warehouse_allocation
    for net in ("EU", "BR"):
        r = warehouse_allocation(net, persist=False)
        assert r["objective_cost"] <= r["baseline_single_dc_cost"] + 1e-6
        dem = sum(f["units"] for f in r["flows"])
        assert dem > 0 and all(u <= 1.0001 for u in r["utilisation"].values())
    assert platform["routing"]["vehicles_used"] >= 1


# ------------------------------------------------------------------ workflow + automation
def test_procurement_workflow_lifecycle(platform):
    from scip.services import procurement_workflow as wf
    sp = q("SELECT supplier_id, sku, moq, unit_cost FROM supplier_products ORDER BY unit_cost DESC LIMIT 1").iloc[0]
    qty = max(int(sp["moq"]), int(np.ceil(3000 / sp["unit_cost"])))           # > auto-approval limit
    po = wf.create_po(sp["sku"], sp["supplier_id"], qty, created_by="test")
    assert po["status"] == "draft" and po["approval_level"] in ("manager", "director")
    po = wf.submit_po(po["po_id"])
    assert po["status"] == "pending_approval"
    with pytest.raises(wf.WorkflowError):
        wf.send_po(po["po_id"])                                               # cannot send before approval
    po = wf.decide_po(po["po_id"], True, "test.manager", "ok")
    sent = wf.send_po(po["po_id"])
    assert sent["status"] == "sent" and sent["acknowledgement"]["mock"] is True
    hist = wf.approval_history(po["po_id"])
    assert hist["action"].tolist() == ["created", "submitted", "approved", "sent"]
    small = wf.submit_po(wf.create_po(sp["sku"], sp["supplier_id"], int(sp["moq"]), created_by="test")["po_id"])
    if small["total_value"] <= 2500:
        assert small["status"] == "approved" and small["approved_by"].startswith("system")
    with pytest.raises(wf.WorkflowError):
        wf.create_po(sp["sku"], sp["supplier_id"], max(int(sp["moq"]) - 1, 0) or -1, created_by="test")


def test_automation_cycle_and_n8n_webhook(platform, monkeypatch):
    received = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from scip.core.config import get_settings
    monkeypatch.setattr(get_settings(), "n8n_webhook_url", f"http://127.0.0.1:{srv.server_port}/webhook/scip-events")
    from scip.automation.engine import run_cycle
    r = run_cycle(trigger="test")
    srv.shutdown()
    assert r["status"] == "ok" and r["rag_documents"] > 20
    events = [e["event"] for e in received]
    assert "automation.cycle_completed" in events
    log = q("SELECT * FROM automation_log WHERE action='cycle_completed' ORDER BY id DESC LIMIT 1").iloc[0]
    assert bool(log["notified"]) is True
    assert q("SELECT COUNT(*) n FROM alerts WHERE status='open'")["n"].iat[0] > 0
    assert q("SELECT COUNT(*) n FROM reports")["n"].iat[0] >= 1


def test_copilot_retrieval(platform):
    from scip.copilot.rag import ask
    sku = q("SELECT sku FROM products WHERE is_planning_sku LIMIT 1")["sku"].iat[0]
    r = ask(f"What is the status of SKU {sku}?")
    assert r["sources"][0]["doc_id"] == f"sku:{sku}" and r["provider"] == "retrieval-only"
    r = ask("Which suppliers are the riskiest?")
    assert any(s["doc_id"] == "summary:suppliers" for s in r["sources"][:2])
    assert q("SELECT COUNT(*) n FROM copilot_messages")["n"].iat[0] >= 2


# ------------------------------------------------------------------ REST API
@pytest.fixture(scope="module")
def client(platform):
    from fastapi.testclient import TestClient
    from scip.api.main import app
    with TestClient(app) as c:
        yield c


def test_api_read_endpoints(client):
    sku = q("SELECT sku FROM products WHERE is_planning_sku LIMIT 1")["sku"].iat[0]
    for path in ["/health", "/api/v1/status", "/api/v1/dashboard/kpis", "/api/v1/dashboard/risk-heatmap",
                 "/api/v1/dashboard/delivery-tracking", "/api/v1/forecasts/summary", "/api/v1/forecasts/models",
                 "/api/v1/forecasts/portfolio", f"/api/v1/forecasts/{sku}", "/api/v1/inventory/policy",
                 "/api/v1/stockout/predictions", "/api/v1/suppliers/scores", "/api/v1/models", "/api/v1/shipments/performance",
                 "/api/v1/procurement/purchase-orders", "/api/v1/alerts", "/api/v1/automation/log", "/api/v1/reports",
                 "/api/v1/data/sources", "/api/v1/logistics/plans"]:
        r = client.get(path)
        assert r.status_code == 200, (path, r.text[:200])
    k = client.get("/api/v1/dashboard/kpis").json()
    assert k["planning_skus"] == 20 and k["total_inventory_value"] > 0
    assert client.get("/api/v1/forecasts/NOPE").status_code == 404


def test_api_procurement_flow_and_validation(client):
    sp = q("SELECT supplier_id, sku, moq, unit_cost FROM supplier_products ORDER BY unit_cost DESC LIMIT 1").iloc[0]
    qty = max(int(sp["moq"]), int(np.ceil(3000 / sp["unit_cost"])))
    r = client.post("/api/v1/procurement/purchase-orders", json={"sku": sp["sku"], "supplier_id": sp["supplier_id"], "quantity": qty})
    assert r.status_code == 201 and r.json()["status"] == "pending_approval"
    pid = r.json()["po_id"]
    assert client.post(f"/api/v1/procurement/purchase-orders/{pid}/approve", json={"actor": ""}).status_code == 422
    assert client.post(f"/api/v1/procurement/purchase-orders/{pid}/reject", json={"actor": "qa"}).json()["status"] == "rejected"
    assert client.post(f"/api/v1/procurement/purchase-orders/{pid}/approve", json={"actor": "qa"}).status_code == 409
    assert client.post("/api/v1/procurement/purchase-orders", json={"sku": sp["sku"], "supplier_id": sp["supplier_id"],
                                                                     "quantity": -5}).status_code == 422
    opt = client.post("/api/v1/procurement/optimize", json={"requirements": [{"sku": sp["sku"], "quantity": qty}]})
    assert opt.status_code == 200 and opt.json()["status"] == "OPTIMAL"
    ans = client.post("/api/v1/copilot/ask", json={"question": "Which products are about to stock out?"})
    assert ans.status_code == 200 and ans.json()["sources"]


def test_api_key_protects_mutations(client, monkeypatch):
    from scip.core.config import get_settings
    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.post("/api/v1/reports/daily").status_code == 401
    assert client.post("/api/v1/reports/daily", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/api/v1/dashboard/kpis").status_code == 200        # reads stay open


def test_dashboard_views_render_against_live_api(platform):
    """Starts the API on a free port (test database) and renders every Control Tower view through it."""
    import importlib
    import os
    import socket
    import time

    import uvicorn
    from streamlit.testing.v1 import AppTest
    from scip.api.main import app
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    os.environ["SCIP_API_URL"] = f"http://127.0.0.1:{port}"
    import dashboard.api_client as api_client
    importlib.reload(api_client)
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=300)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        for view in ["🛰️ Control Tower", "📈 Demand Forecasting", "📦 Inventory & Stockout", "🏭 Supplier Intelligence",
                     "🧾 Procurement", "🚚 Logistics", "🤖 AI Copilot", "⚙️ Automation", "🗄️ Data & Models"]:
            at.radio(key="nav").set_value(view).run()
            assert not at.exception, (view, [e.value for e in at.exception])
            assert not at.error, (view, [e.value for e in at.error])
    finally:
        server.should_exit = True
