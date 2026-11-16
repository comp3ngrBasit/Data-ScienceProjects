"""Supply Chain Control Tower — the executive overview."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard import ui


def render():
    k = api.cached("/api/v1/dashboard/kpis")
    ui.ribbon("Supply Chain Control Tower", f"Data as of {k['as_of_date']} · forecast model {k.get('forecast_model', 'n/a')}")
    ui.origin_note("Demand = <b>real</b> UCI Online Retail · shipments = <b>real</b> DataCo · Brazil customers = <b>real</b> Olist · "
                   "inventory, vendors &amp; purchase orders = <b>simulated</b> (labelled throughout).")
    ui.kpi_row([
        ui.ikpi("📦", ui.money(k["total_inventory_value"]), "Total inventory value", f"{k['planning_skus']} planning SKUs"),
        ui.ikpi("📈", ui.num(k.get("predicted_demand_30d_units")), "Predicted demand 30d (units)",
                f"{ui.money(k.get('predicted_demand_30d_value'))} · 7d {ui.num(k.get('predicted_demand_7d_units'))}"),
        ui.ikpi("⛔", ui.num(k["alerts_critical"]), "Critical alerts", f"{k['stockout_risk_high']} high ML stockout risk"),
        ui.ikpi("🏭", ui.pct(k.get("supplier_on_time_rate")), "Supplier on-time", f"avg score {ui.num(k.get('supplier_avg_score'), 1)}"),
        ui.ikpi("🧾", ui.num(k["po_pending_approval"]), "POs pending approval", f"open value {ui.money(k['po_open_value'])}"),
        ui.ikpi("🚚", ui.num(k["inbound_late"]), "Overdue inbound POs", f"{k['sku_critical']} critical SKUs"),
    ])
    st.write("")
    c1, c2 = st.columns([1.55, 1])
    with c1, ui.panel("ct_fc"):
        ui.title("Demand: actual vs forecast (weekly, planning SKUs)")
        p = api.cached("/api/v1/forecasts/portfolio")
        h, f = pd.DataFrame(p["history"]), pd.DataFrame(p["forecast"])
        fig = go.Figure()
        if not h.empty:
            fig.add_scatter(x=pd.to_datetime(h["date"]), y=h["actual"], name="Actual (real UCI)", line=dict(color="#898781", width=2))
        if not f.empty:
            x = pd.to_datetime(f["date"])
            fig.add_scatter(x=list(x) + list(x[::-1]), y=list(f["yhat_upper"]) + list(f["yhat_lower"][::-1]), fill="toself",
                            fillcolor="rgba(47,158,138,.15)", line=dict(width=0), name="80% band", hoverinfo="skip")
            fig.add_scatter(x=x, y=f["yhat"], name=f"Forecast ({k.get('forecast_model')})", line=dict(color=ui.BRAND, width=3))
        fig.update_layout(height=330, hovermode="x unified", yaxis_title="Units / week")
        ui.chart(fig, "ct_fc")
        st.caption("Complete weeks only (weeks end on the data as-of weekday). Band = 80% interval from hold-out RMSE, "
                   "assuming independent errors across SKUs and days — it likely understates portfolio uncertainty.")
    with c2, ui.panel("ct_sup"):
        ui.title("Supplier performance")
        sc = api.df("/api/v1/suppliers/scores")
        if not sc.empty:
            sc = sc.sort_values("score")
            fig = go.Figure(go.Bar(y=sc["name"].str.replace(" (simulated)", "", regex=False), x=sc["score"], orientation="h",
                                   marker_color=[ui.STATUS.get(t, ui.BRAND) for t in sc["tier"]],
                                   text=[f"{s:.0f} · {o:.0%}" for s, o in zip(sc["score"], sc["on_time_rate"])], textposition="outside",
                                   customdata=sc[["tier", "predicted_late_risk"]].values,
                                   hovertemplate="%{y}<br>score %{x:.1f} (%{customdata[0]})<br>predicted late risk %{customdata[1]:.0%}<extra></extra>"))
            fig.update_layout(height=330, xaxis_range=[0, 112], xaxis_title="Score (label: score · on-time %)")
            ui.chart(fig, "ct_sup")

    c1, c2 = st.columns([1.3, 1])
    with c1, ui.panel("ct_heat"):
        ui.title("Inventory risk heatmap — share of SKUs projected below safety stock / out of stock")
        hm = api.df("/api/v1/dashboard/risk-heatmap")
        if not hm.empty:
            pv = hm.pivot(index="category", columns="week_start", values="risk")
            fig = go.Figure(go.Heatmap(z=pv.values, x=[f"W{i + 1}<br>{d[5:]}" for i, d in enumerate(pv.columns)], y=pv.index,
                                       colorscale=[[0, "#eef7f4"], [0.35, "#fde3a7"], [0.7, "#f08a5d"], [1, "#b22b2b"]],
                                       zmin=0, zmax=1, colorbar=dict(title="risk", tickformat=".0%"),
                                       hovertemplate="%{y} · %{x}<br>risk index %{z:.0%}<extra></extra>"))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            ui.chart(fig, "ct_heat")
            st.caption("Projection: on hand + open PO receipts − champion daily forecast. 1.0 = projected stockout, 0.5 = below safety stock.")
    with c2, ui.panel("ct_track"):
        ui.title("Delivery tracking (inbound purchase orders)")
        tr = api.df("/api/v1/dashboard/delivery-tracking")
        if not tr.empty:
            cnt = tr["tracking_status"].value_counts().reindex(["Overdue", "At risk", "On track", "Delivered"]).fillna(0)
            fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[ui.STATUS[s] for s in cnt.index],
                                   text=cnt.astype(int), textposition="outside"))
            fig.update_layout(height=200, yaxis_range=[0, cnt.max() * 1.25 + 1], margin=dict(t=10))
            ui.chart(fig, "ct_track")
            open_ = tr[tr["tracking_status"].isin(["Overdue", "At risk"])].sort_values("days_to_promised")
            ui.table(open_[["po_id", "supplier", "sku", "promised_date", "days_to_promised", "predicted_late_risk", "tracking_status"]]
                     .assign(promised_date=lambda d: d["promised_date"].str[:10]), height=170,
                     column_config={"predicted_late_risk": st.column_config.ProgressColumn("Late risk", min_value=0, max_value=1, format="percent")})

    c1, c2 = st.columns([1.1, 1])
    with c1, ui.panel("ct_recs"):
        ui.title("🤖 AI-generated recommendations")
        recs = api.df("/api/v1/procurement/recommendations", limit=8)
        alerts = api.df("/api/v1/alerts", severity="Critical")
        for a in alerts.head(4).itertuples():
            st.markdown(f"<div class='rec'>{ui.badge('Critical')} <span class='t'>{a.alert_type.replace('_', ' ').title()} · {a.entity}</span>"
                        f"<div class='w'>{a.message}</div></div>", unsafe_allow_html=True)
        for r in recs.head(4).itertuples():
            st.markdown(f"<div class='rec'>{ui.badge('Info')} <span class='t'>Order {r.quantity:,.0f} × {r.sku} from "
                        f"{r.supplier_name.replace(' (simulated)', '')}</span><div class='w'>{r.rationale}</div></div>",
                        unsafe_allow_html=True)
        if recs.empty and alerts.empty:
            st.success("No actions required.")
    with c2, ui.panel("ct_log"):
        ui.title("⚙️ Automation activity log")
        lg = api.df("/api/v1/automation/log", limit=40)
        if not lg.empty:
            lg["at"] = lg["at"].str[:19].str.replace("T", " ")
            lg["details"] = lg["details"].astype(str).str[:140]
            ui.table(lg[["at", "job", "action", "status", "notified", "details"]], height=360)
