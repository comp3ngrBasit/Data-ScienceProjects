from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import api_client as api
from dashboard import ui


def _act(fn, *a, **k):
    try:
        r = fn(*a, **k)
        api.refresh()
        return r
    except api.APIError as e:
        st.error(str(e))
        return None


def render():
    ui.ribbon("Procurement Automation", "Supplier recommendation · CP-SAT order allocation · PO approval workflow")
    ui.origin_note("Approval levels by PO value: auto ≤ $2,500 · manager ≤ $25,000 · director above. Approval requests and "
                   "decisions are published to n8n. Supplier portal is a <b>mock</b> integration.")
    pos = api.df("/api/v1/procurement/purchase-orders", limit=2000)
    platform = pos[pos["created_by"] != "history"] if not pos.empty else pos
    cnt = platform["status"].value_counts() if not platform.empty else pd.Series(dtype=int)
    ui.kpi_row([ui.ikpi("📝", ui.num(cnt.get("pending_approval", 0)), "Pending approval"),
                ui.ikpi("✅", ui.num(cnt.get("approved", 0)), "Approved, not sent"),
                ui.ikpi("📤", ui.num(cnt.get("sent", 0)), "Sent by platform"),
                ui.ikpi("❌", ui.num(cnt.get("rejected", 0)), "Rejected"),
                ui.ikpi("💰", ui.money(platform.loc[platform["status"].isin(["pending_approval"]), "total_value"].sum() if not platform.empty else 0),
                        "Value awaiting approval")])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Approval queue", "Recommend & optimise", "All purchase orders", "Recommendations log"])
    with t1:
        q = platform[platform["status"].isin(["pending_approval", "approved", "draft"])] if not platform.empty else platform
        if q.empty:
            st.success("Nothing awaiting action.")
        for r in q.itertuples():
            with ui.panel(f"po_{r.po_id}"):
                c1, c2 = st.columns([3, 1.2])
                c1.markdown(f"**{r.po_id}** · {ui.badge(r.status.replace('_', ' ').title())} · {r.approval_level} approval  \n"
                            f"{r.quantity:,.0f} × **{r.sku}** {r.product_name} from {r.supplier_name}  \n"
                            f"Value **{ui.money(r.total_value)}** · promised {str(r.promised_date)[:10]} · "
                            f"predicted late risk {ui.pct(r.predicted_late_risk) if pd.notna(r.predicted_late_risk) else 'n/a'}",
                            unsafe_allow_html=True)
                if r.rationale:
                    c1.caption(r.rationale)
                actor = c2.text_input("Approver", value="ops.manager", key=f"act_{r.po_id}")
                b1, b2, b3 = c2.columns(3)
                if r.status == "pending_approval":
                    if b1.button("Approve", key=f"ap_{r.po_id}", type="primary"):
                        _act(api.post, f"/api/v1/procurement/purchase-orders/{r.po_id}/approve", {"actor": actor, "comment": "Approved in Control Tower"})
                        st.rerun()
                    if b2.button("Reject", key=f"rj_{r.po_id}"):
                        _act(api.post, f"/api/v1/procurement/purchase-orders/{r.po_id}/reject", {"actor": actor, "comment": "Rejected in Control Tower"})
                        st.rerun()
                if r.status == "approved" and b1.button("Send", key=f"sd_{r.po_id}", type="primary"):
                    _act(api.post, f"/api/v1/procurement/purchase-orders/{r.po_id}/send")
                    st.rerun()
                if r.status == "draft" and b1.button("Submit", key=f"sb_{r.po_id}"):
                    _act(api.post, f"/api/v1/procurement/purchase-orders/{r.po_id}/submit")
                    st.rerun()
    with t2:
        pol = api.df("/api/v1/inventory/policy")
        need = pol[pol["recommended_qty"] > 0]
        c1, c2 = st.columns([1, 1])
        sku = c1.selectbox("SKU", pol["sku"], index=int(pol.index.get_indexer([need.index[0]])[0]) if not need.empty else 0,
                           format_func=lambda s: f"{s} · {pol.set_index('sku').at[s, 'name']}", key="pr_sku")
        default_q = int(pol.set_index("sku").at[sku, "recommended_qty"]) or 100
        qty = c2.number_input("Quantity", 1, 1_000_000, default_q, key="pr_qty")
        try:
            rec = pd.DataFrame(api.get("/api/v1/procurement/recommend", sku=sku, quantity=int(qty)))
            ui.table(rec[["rank", "supplier_name", "tier", "score", "unit_cost", "moq", "lead_time_days", "risk", "order_qty",
                          "purchase_cost", "risk_cost", "transit_capital_cost", "total_expected_cost"]],
                     column_config={"risk": st.column_config.NumberColumn("Predicted late risk", format="percent")})
            choice = st.selectbox("Create PO with supplier", rec["supplier_id"], format_func=lambda s: rec.set_index("supplier_id").at[s, "supplier_name"], key="pr_sup")
            if st.button("Create & submit purchase order", type="primary", key="pr_create"):
                r = _act(api.post, "/api/v1/procurement/purchase-orders",
                         {"sku": sku, "supplier_id": choice, "quantity": int(max(qty, rec.set_index('supplier_id').at[choice, 'moq'])),
                          "rationale": "Created from Control Tower recommendation", "submit": True})
                if r:
                    st.success(f"{r['po_id']} created — status {r['status']} ({r['approval_level']} approval).")
        except api.APIError as e:
            st.error(str(e))
        st.divider()
        st.markdown("**Multi-SKU allocation (OR-Tools CP-SAT)** — all SKUs currently below reorder point")
        budget = st.number_input("Budget cap ($, 0 = none)", 0, 10_000_000, 0, 1000, key="pr_budget")
        if st.button("Optimise allocation", key="pr_opt") and not need.empty:
            body = {"requirements": [{"sku": r.sku, "quantity": int(r.recommended_qty)} for r in need.itertuples()],
                    "budget": budget or None}
            res = _act(api.post, "/api/v1/procurement/optimize", body)
            if res:
                st.write(f"Solver status **{res['status']}** · objective {ui.money(res['objective'])} · {res['solve_seconds']:.2f}s")
                ui.table(pd.DataFrame(res["allocations"]))
                if res["shortfalls"]:
                    st.warning(f"Unfilled (budget/capacity): {res['shortfalls']}")
    with t3:
        st_f = st.multiselect("Status", sorted(pos["status"].unique()), default=None, key="po_st")
        t = pos[pos["status"].isin(st_f)] if st_f else pos
        ui.table(t[["po_id", "status", "approval_level", "supplier_name", "sku", "quantity", "total_value", "order_date",
                    "promised_date", "received_date", "predicted_late_risk", "created_by", "data_origin"]], height=440)
        pid = st.text_input("Approval history for PO", key="po_hist")
        if pid:
            ui.table(api.df(f"/api/v1/procurement/purchase-orders/{pid}/history"))
    with t4:
        ui.table(api.df("/api/v1/procurement/recommendations"), height=420)
