from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from scip.api.deps import clean, records, require_api_key
from scip.core.db import read_sql
from scip.optimization import logistics
from scip.optimization.procurement import Requirement, optimize_allocation, recommend_suppliers
from scip.services import procurement_workflow as wf

router = APIRouter(prefix="/api/v1", tags=["operations"])


class RequirementIn(BaseModel):
    sku: str
    quantity: int = Field(gt=0)
    max_lead_days: float | None = None


class OptimizeIn(BaseModel):
    requirements: list[RequirementIn]
    budget: float | None = Field(default=None, gt=0)
    risk_weight: float = Field(default=0.5, ge=0, le=5)


class POIn(BaseModel):
    sku: str
    supplier_id: str
    quantity: float = Field(gt=0)
    rationale: str = ""
    submit: bool = True


class DecisionIn(BaseModel):
    actor: str = Field(min_length=1)
    comment: str = ""


class VRPIn(BaseModel):
    hub: str = "BR-HUB-SP"
    vehicles: int = Field(default=5, ge=1, le=20)


def _wf(fn, *a, **k):
    try:
        return clean(fn(*a, **k))
    except wf.WorkflowError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ---------------------------------------------------------------- procurement
@router.get("/procurement/recommend")
def recommend(sku: str, quantity: int = Query(gt=0)):
    df = recommend_suppliers(sku, quantity)
    if df.empty:
        raise HTTPException(404, f"No suppliers found for SKU {sku}")
    return records(df)


@router.post("/procurement/optimize")
def optimize(body: OptimizeIn):
    if not body.requirements:
        raise HTTPException(422, "At least one requirement is needed")
    res = optimize_allocation([Requirement(r.sku, r.quantity, r.max_lead_days) for r in body.requirements],
                              budget=body.budget, risk_weight=body.risk_weight)
    return clean(res)


@router.get("/procurement/recommendations")
def recommendations(limit: int = 100):
    return records(read_sql("SELECT r.*, s.name AS supplier_name, p.name AS product_name FROM procurement_recommendations r "
                            "JOIN suppliers s USING(supplier_id) JOIN products p USING(sku) ORDER BY r.id DESC LIMIT :l", {"l": limit}))


@router.get("/procurement/purchase-orders")
def list_pos(status: str | None = None, limit: int = Query(300, le=5000)):
    return records(wf.list_pos(status, limit))


@router.post("/procurement/purchase-orders", dependencies=[Depends(require_api_key)], status_code=201)
def create_po(body: POIn):
    po = _wf(wf.create_po, body.sku, body.supplier_id, body.quantity, "user", body.rationale)
    if body.submit:
        po = _wf(wf.submit_po, po["po_id"], "user")
    return po


@router.post("/procurement/purchase-orders/{po_id}/submit", dependencies=[Depends(require_api_key)])
def submit(po_id: str):
    return _wf(wf.submit_po, po_id)


@router.post("/procurement/purchase-orders/{po_id}/approve", dependencies=[Depends(require_api_key)])
def approve(po_id: str, body: DecisionIn):
    return _wf(wf.decide_po, po_id, True, body.actor, body.comment)


@router.post("/procurement/purchase-orders/{po_id}/reject", dependencies=[Depends(require_api_key)])
def reject(po_id: str, body: DecisionIn):
    return _wf(wf.decide_po, po_id, False, body.actor, body.comment)


@router.post("/procurement/purchase-orders/{po_id}/send", dependencies=[Depends(require_api_key)])
def send(po_id: str):
    return _wf(wf.send_po, po_id)


@router.get("/procurement/purchase-orders/{po_id}/history")
def history(po_id: str):
    return records(wf.approval_history(po_id))


# ---------------------------------------------------------------- logistics
@router.post("/logistics/allocation")
def allocation(network: str = Query("EU", pattern="^(EU|BR)$")):
    try:
        return clean(logistics.warehouse_allocation(network))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.post("/logistics/routing")
def routing(body: VRPIn):
    try:
        return clean(logistics.brazil_linehaul(hub=body.hub, vehicles=body.vehicles))
    except (ValueError, RuntimeError, IndexError) as exc:
        raise HTTPException(422, str(exc))


@router.get("/logistics/plans")
def plans(plan_type: str | None = None, limit: int = 10):
    q = "SELECT * FROM logistics_plans" + (" WHERE plan_type=:t" if plan_type else "") + " ORDER BY plan_id DESC LIMIT :l"
    return records(read_sql(q, {"t": plan_type, "l": limit} if plan_type else {"l": limit}))


# ---------------------------------------------------------------- automation, alerts, reports
@router.post("/automation/run", dependencies=[Depends(require_api_key)])
def automation_run(trigger: str = "api", auto_submit: bool = True):
    from scip.automation.engine import run_cycle
    return clean(run_cycle(trigger=trigger, auto_submit=auto_submit))


@router.get("/automation/log")
def automation_log(limit: int = Query(200, le=2000)):
    return records(read_sql("SELECT * FROM automation_log ORDER BY id DESC LIMIT :l", {"l": limit}))


@router.get("/alerts")
def alerts(status: str = "open", severity: str | None = None):
    q = "SELECT * FROM alerts WHERE status=:s" + (" AND severity=:v" if severity else "") + \
        " ORDER BY CASE severity WHEN 'Critical' THEN 0 WHEN 'Warning' THEN 1 ELSE 2 END, alert_id DESC"
    return records(read_sql(q, {"s": status, "v": severity}))


@router.post("/alerts/{alert_id}/acknowledge", dependencies=[Depends(require_api_key)])
def ack(alert_id: int):
    from sqlalchemy import text
    from scip.core.db import get_engine
    with get_engine().begin() as conn:
        n = conn.execute(text("UPDATE alerts SET status='acknowledged' WHERE alert_id=:a AND status='open'"), {"a": alert_id}).rowcount
    if not n:
        raise HTTPException(404, "Open alert not found")
    return {"alert_id": alert_id, "status": "acknowledged"}


@router.get("/reports")
def reports():
    return records(read_sql("SELECT report_id, created_at, report_type, title FROM reports ORDER BY report_id DESC"))


@router.post("/reports/daily", dependencies=[Depends(require_api_key)])
def make_report():
    from scip.services.reports import daily_operations_report
    return daily_operations_report()


@router.get("/reports/{report_id}")
def report(report_id: int):
    df = read_sql("SELECT * FROM reports WHERE report_id=:r", {"r": report_id})
    if df.empty:
        raise HTTPException(404, "Report not found")
    return records(df)[0]
