"""Purchase-order lifecycle and approval workflow.

draft -> pending_approval -> approved -> sent -> received          (or rejected / cancelled)
Approval level by PO value: auto (<= AUTO_APPROVE_LIMIT), manager (<= MANAGER_APPROVE_LIMIT), director (above).
Auto-level POs are approved by the system immediately. Every transition is recorded in po_approvals and
automation_log, and published to n8n (approval requests, decisions).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import func, select

from scip.core.config import get_settings
from scip.core.db import get_engine, read_sql, session_scope
from scip.core.models import AutomationLog, POApproval, ProcurementRecommendation, PurchaseOrder, SupplierProduct
from scip.ingestion.pipeline import as_of_date
from scip.integrations import n8n
from scip.integrations.supplier_portal import get_portal


class WorkflowError(ValueError):
    pass


def approval_level(value: float) -> str:
    s = get_settings()
    return "auto" if value <= s.auto_approve_limit else "manager" if value <= s.manager_approve_limit else "director"


def _log(s, job, action, status, details, notify_event: str | None = None):
    res = n8n.publish(notify_event, details) if notify_event else {"delivered": False}
    s.add(AutomationLog(job=job, action=action, status=status, details={**details, "n8n": res},
                        notified=bool(res.get("delivered"))))


def _next_po_id(s) -> str:
    n = s.execute(select(func.count()).select_from(PurchaseOrder).where(PurchaseOrder.po_id.like("PO-A%"))).scalar_one()
    return f"PO-A{n + 1:06d}"


def create_po(sku: str, supplier_id: str, quantity: float, created_by: str = "user", rationale: str = "",
              recommendation_id: int | None = None, engine=None) -> dict:
    engine = engine or get_engine()
    if quantity <= 0:
        raise WorkflowError("Quantity must be positive")
    with session_scope(engine) as s:
        sp = s.get(SupplierProduct, (supplier_id, sku))
        if sp is None:
            raise WorkflowError(f"Supplier {supplier_id} does not supply SKU {sku}")
        if quantity < sp.moq:
            raise WorkflowError(f"Quantity {quantity} is below the supplier MOQ of {sp.moq}")
        today = as_of_date(engine).date()
        value = round(quantity * sp.unit_cost, 2)
        po = PurchaseOrder(po_id=_next_po_id(s), supplier_id=supplier_id, sku=sku, quantity=float(quantity),
                           unit_cost=sp.unit_cost, total_value=value, order_date=today,
                           promised_date=(pd.Timestamp(today) + pd.Timedelta(days=round(sp.lead_time_days))).date(),
                           status="draft", approval_level=approval_level(value), created_by=created_by,
                           rationale=rationale, data_origin="simulated" if created_by == "history" else "platform")
        s.add(po)
        s.flush()
        s.add(POApproval(po_id=po.po_id, action="created", actor=created_by, comment=rationale[:500]))
        if recommendation_id:
            rec = s.get(ProcurementRecommendation, recommendation_id)
            if rec:
                rec.po_id = po.po_id
        _log(s, "procurement", "po_created", "ok", {"po_id": po.po_id, "sku": sku, "value": value})
        return _po_dict(po)


def submit_po(po_id: str, actor: str = "system", engine=None) -> dict:
    with session_scope(engine or get_engine()) as s:
        po = _get(s, po_id, {"draft"})
        if po.approval_level == "auto":
            po.status = "approved"
            po.approved_by, po.approved_at = "system:auto-approval", datetime.now()
            s.add(POApproval(po_id=po_id, action="auto_approved", actor="system",
                             comment=f"Value {po.total_value:,.2f} <= auto-approval limit"))
            _log(s, "procurement", "po_auto_approved", "ok", {"po_id": po_id, "value": po.total_value})
        else:
            po.status = "pending_approval"
            s.add(POApproval(po_id=po_id, action="submitted", actor=actor, comment=f"Requires {po.approval_level} approval"))
            _log(s, "procurement", "approval_requested", "ok",
                 {"po_id": po_id, "sku": po.sku, "supplier_id": po.supplier_id, "value": po.total_value,
                  "approval_level": po.approval_level,
                  "approve_url": f"{get_settings().api_base_url}/api/v1/procurement/purchase-orders/{po_id}/approve",
                  "reject_url": f"{get_settings().api_base_url}/api/v1/procurement/purchase-orders/{po_id}/reject"},
                 notify_event="po.approval_requested")
        return _po_dict(po)


def decide_po(po_id: str, approve: bool, actor: str, comment: str = "", engine=None) -> dict:
    if not actor:
        raise WorkflowError("An approver name is required")
    with session_scope(engine or get_engine()) as s:
        po = _get(s, po_id, {"pending_approval"})
        po.status = "approved" if approve else "rejected"
        po.approved_by, po.approved_at = actor, datetime.now()
        s.add(POApproval(po_id=po_id, action="approved" if approve else "rejected", actor=actor, comment=comment))
        _log(s, "procurement", "po_approved" if approve else "po_rejected", "ok",
             {"po_id": po_id, "actor": actor, "comment": comment, "value": po.total_value},
             notify_event="po.decision")
        return _po_dict(po)


def send_po(po_id: str, engine=None) -> dict:
    with session_scope(engine or get_engine()) as s:
        po = _get(s, po_id, {"approved"})
        sp = s.get(SupplierProduct, (po.supplier_id, po.sku))
        ack = get_portal().send_purchase_order(po.po_id, po.supplier_id, po.sku, po.quantity, po.order_date,
                                               sp.lead_time_days if sp else 30)
        po.status = "sent"
        po.promised_date = pd.Timestamp(ack["confirmed_delivery_date"]).date()
        s.add(POApproval(po_id=po_id, action="sent", actor="system", comment=f"{ack['portal']} {ack['confirmation_number']}"))
        _log(s, "procurement", "po_sent", "ok", {"po_id": po_id, "ack": ack}, notify_event="po.sent")
        return {**_po_dict(po), "acknowledgement": ack}


def _get(s, po_id, allowed: set[str]) -> PurchaseOrder:
    po = s.get(PurchaseOrder, po_id)
    if po is None:
        raise WorkflowError(f"Purchase order {po_id} not found")
    if po.status not in allowed:
        raise WorkflowError(f"PO {po_id} is '{po.status}'; expected one of {sorted(allowed)}")
    return po


def _po_dict(po: PurchaseOrder) -> dict:
    return {c: (v.isoformat() if hasattr(v, "isoformat") else v)
            for c, v in ((c.name, getattr(po, c.name)) for c in PurchaseOrder.__table__.columns)}


def list_pos(status: str | None = None, limit: int = 500, engine=None) -> pd.DataFrame:
    q = ("SELECT p.*, s.name AS supplier_name, pr.name AS product_name FROM purchase_orders p "
         "JOIN suppliers s USING(supplier_id) JOIN products pr USING(sku)")
    params = {}
    if status:
        q += " WHERE p.status = :st"
        params["st"] = status
    q += " ORDER BY p.order_date DESC, p.po_id DESC LIMIT :lim"
    params["lim"] = limit
    return read_sql(q, params, engine=engine)


def approval_history(po_id: str, engine=None) -> pd.DataFrame:
    return read_sql("SELECT * FROM po_approvals WHERE po_id=:p ORDER BY at, id", {"p": po_id}, engine=engine)
