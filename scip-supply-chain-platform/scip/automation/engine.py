"""Automation engine — the autonomous operations loop.

`run_cycle()` (triggered by the n8n cron workflow, the API or the dashboard):
  1. recompute the inventory policy from the latest champion forecast;
  2. score open POs for delivery risk and refresh supplier scores;
  3. raise alerts (stockout, low stock / reorder, overstock, overdue inbound POs, high-risk POs, at-risk suppliers);
  4. build procurement requirements (recommended qty > 0 and no PO already in the pipeline), allocate them across
     suppliers with the CP-SAT optimizer, store recommendations, create draft POs and submit them into the
     approval workflow (small POs auto-approve; others request manager/director approval via n8n);
  5. generate the daily operations report;
  6. rebuild the Copilot knowledge index;
  7. write every step to automation_log and publish a cycle summary to n8n.
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from scip.core.db import get_engine, read_sql, session_scope, write_frame
from scip.core.logging import get_logger
from scip.core.models import AutomationLog
from scip.copilot.rag import build_index
from scip.ingestion.pipeline import as_of_date
from scip.integrations import n8n
from scip.ml import delivery_risk
from scip.ml.supplier_scoring import latest_scores, score_suppliers
from scip.optimization.inventory import compute_policy
from scip.optimization.procurement import Requirement, optimize_allocation
from scip.services import procurement_workflow as wf
from scip.services.reports import daily_operations_report

log = get_logger("automation")


def _log(engine, job, action, status, details, notified=False):
    with session_scope(engine) as s:
        s.add(AutomationLog(job=job, action=action, status=status, details=details, notified=notified))


def raise_alerts(engine, policy: pd.DataFrame) -> pd.DataFrame:
    as_of = as_of_date(engine)
    rows = []
    for _, r in policy.iterrows():
        if r["status"] == "Critical":
            msg = ("Out of stock" if r["on_hand"] <= 0 else
                   f"Stock {r['on_hand']:.0f} + on order {r['on_order']:.0f} below lead-time demand "
                   f"({r['avg_daily_demand']:.1f}/day x {r['lead_time_days']:.0f} d)")
            rows.append(("Critical", "low_stock", r["sku"], f"{r['name']}: {msg}. Recommended order {r['recommended_qty']:.0f}."))
        elif r["status"] == "Reorder":
            rows.append(("Warning", "reorder_point", r["sku"],
                         f"{r['name']}: inventory position at/below reorder point {r['reorder_point']:.0f}; order {r['recommended_qty']:.0f}."))
        elif r["status"] == "Overstock":
            rows.append(("Info", "overstock", r["sku"],
                         f"{r['name']}: on hand {r['on_hand']:.0f} exceeds max level {r['max_level']:.0f}."))
    so = read_sql("SELECT sku, probability FROM stockout_predictions WHERE scored_at=(SELECT MAX(scored_at) FROM "
                  "stockout_predictions) AND risk_band='High'", engine=engine)
    crit = set(policy.loc[policy["status"] == "Critical", "sku"])
    for r in so.itertuples():
        if r.sku not in crit:
            rows.append(("Warning", "stockout_prediction", r.sku, f"ML stockout probability {r.probability:.0%} within 14 days."))
    po = read_sql("SELECT po_id, sku, supplier_id, promised_date, predicted_late_risk FROM purchase_orders WHERE status='sent'",
                  engine=engine, parse_dates=["promised_date"])
    for r in po.itertuples():
        if r.promised_date < as_of:
            rows.append(("Critical", "overdue_po", r.po_id,
                         f"PO {r.po_id} ({r.sku}) from {r.supplier_id} is {(as_of - r.promised_date).days} day(s) overdue."))
        elif (r.predicted_late_risk or 0) >= 0.85 and (r.promised_date - as_of).days <= 21:
            rows.append(("Warning", "delivery_risk", r.po_id,
                         f"PO {r.po_id} ({r.sku}) predicted late-delivery risk {r.predicted_late_risk:.0%}."))
    sc = latest_scores(engine)
    for r in sc[sc["tier"] == "At risk"].itertuples():
        rows.append(("Warning", "supplier_risk", r.supplier_id, f"{r.name} scored {r.score:.0f}/100 (on-time {r.on_time_rate:.0%})."))
    with engine.begin() as conn:   # supersede previous automatic alerts
        conn.execute(text("UPDATE alerts SET status='superseded' WHERE status='open'"))
    df = pd.DataFrame(rows, columns=["severity", "alert_type", "entity", "message"])
    df["status"] = "open"
    df["created_at"] = datetime.now().replace(microsecond=0)
    write_frame(df, "alerts", engine)
    return df


def plan_procurement(engine, policy: pd.DataFrame, auto_submit: bool = True, max_skus: int = 40) -> dict:
    in_pipeline = set(read_sql("SELECT DISTINCT sku FROM purchase_orders WHERE status IN "
                               "('draft','pending_approval','approved','sent')", engine=engine)["sku"])
    need = policy[(policy["recommended_qty"] > 0) & ~policy["sku"].isin(in_pipeline)]
    need = need.sort_values(["status", "days_cover"]).head(max_skus)
    if need.empty:
        return {"requirements": 0, "pos": []}
    reqs = [Requirement(r.sku, int(r.recommended_qty)) for r in need.itertuples()]
    res = optimize_allocation(reqs, engine)
    created = []
    pol = need.set_index("sku")
    for a in res["allocations"]:
        p = pol.loc[a["sku"]]
        rationale = (f"Inventory position {p['on_hand'] + p['on_order']:.0f} <= ROP {p['reorder_point']:.0f} "
                     f"(d={p['avg_daily_demand']:.1f}/day, L={p['lead_time_days']:.0f} d, SS={p['safety_stock']:.0f}); "
                     f"CP-SAT allocated {a['quantity']} units to {a['supplier_name']} (unit cost {a['unit_cost']:.2f}, "
                     f"supplier-level late risk {a['predicted_late_risk']:.0%}, lead {a['lead_time_days']:.0f} d).")
        with session_scope(engine) as s:
            from scip.core.models import ProcurementRecommendation
            rec = ProcurementRecommendation(sku=a["sku"], supplier_id=a["supplier_id"], quantity=a["quantity"],
                                            unit_cost=a["unit_cost"], total_value=a["total_value"],
                                            score_details={k: a[k] for k in ("predicted_late_risk", "lead_time_days", "tier")},
                                            rationale=rationale)
            s.add(rec)
            s.flush()
            rec_id = rec.id
        if auto_submit:
            po = wf.create_po(a["sku"], a["supplier_id"], a["quantity"], created_by="automation", rationale=rationale,
                              recommendation_id=rec_id, engine=engine)
            po = wf.submit_po(po["po_id"], engine=engine)
            if po["status"] == "approved":
                po = wf.send_po(po["po_id"], engine=engine)
            created.append({k: po[k] for k in ("po_id", "sku", "supplier_id", "quantity", "total_value", "status", "approval_level")})
    return {"requirements": len(reqs), "solver_status": res["status"], "shortfalls": res["shortfalls"], "pos": created}


def run_cycle(engine=None, trigger: str = "manual", auto_submit: bool = True) -> dict:
    engine = engine or get_engine()
    t0 = time.time()
    summary: dict = {"trigger": trigger, "started_at": datetime.now().isoformat(timespec="seconds")}
    try:
        policy = compute_policy(engine)
        policy = policy.merge(read_sql("SELECT sku, name FROM products", engine=engine), on="sku", how="left",
                              suffixes=("", "_p"))
        summary["policy_skus"] = len(policy)
        summary["po_scored"] = delivery_risk.score_purchase_orders(engine)
        score_suppliers(engine)
        alerts = raise_alerts(engine, policy)
        summary["alerts"] = alerts["severity"].value_counts().to_dict()
        _log(engine, "automation", "alerts_raised", "ok", summary["alerts"])
        proc = plan_procurement(engine, policy, auto_submit)
        summary["procurement"] = {"requirements": proc["requirements"], "pos_created": len(proc["pos"]),
                                  "pending_approval": sum(p["status"] == "pending_approval" for p in proc["pos"]),
                                  "auto_sent": sum(p["status"] == "sent" for p in proc["pos"])}
        _log(engine, "automation", "procurement_planned", "ok", summary["procurement"])
        rep = daily_operations_report(engine)
        summary["report_id"] = rep["report_id"]
        summary["rag_documents"] = build_index(engine)
        summary["status"] = "ok"
    except Exception as exc:
        log.exception("Automation cycle failed")
        summary.update(status="error", error=str(exc))
    summary["seconds"] = round(time.time() - t0, 1)
    critical = read_sql("SELECT alert_type, entity, message FROM alerts WHERE status='open' AND severity='Critical' "
                        "LIMIT 20", engine=engine).to_dict("records")
    res = n8n.publish("automation.cycle_completed", {"summary": summary, "critical_alerts": critical})
    _log(engine, "automation", "cycle_completed", summary["status"], {**summary, "n8n": res}, bool(res.get("delivered")))
    return summary
