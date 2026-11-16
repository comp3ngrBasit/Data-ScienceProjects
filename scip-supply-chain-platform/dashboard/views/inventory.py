from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard import ui


def render():
    ui.ribbon("Inventory Optimization & Stockout Prediction", "Safety stock · reorder points · EOQ · ML stockout risk (14 days)")
    ui.origin_note("Demand rates come from the champion forecast on <b>real</b> UCI demand; on-hand stock, lead times and POs are "
                   "<b>simulated</b>. Formulas: SS = z·√(L·σd² + d²·σL²), ROP = d·L + SS, EOQ = √(2DS/H).")
    pol = api.df("/api/v1/inventory/policy")
    if pol.empty:
        st.info("No inventory policy yet.")
        return
    st_counts = pol["status"].value_counts()
    ui.kpi_row([ui.ikpi("⛔", ui.num(st_counts.get("Critical", 0)), "Critical", "stock below lead-time demand"),
                ui.ikpi("🛒", ui.num(st_counts.get("Reorder", 0)), "At reorder point"),
                ui.ikpi("📦", ui.num(st_counts.get("Overstock", 0)), "Overstocked", "above max level"),
                ui.ikpi("✅", ui.num(st_counts.get("Healthy", 0)), "Healthy"),
                ui.ikpi("🔮", ui.num((pol["risk_band"] == "High").sum()), "High ML stockout risk", "probability ≥ 60%"),
                ui.ikpi("💰", ui.money((pol["recommended_qty"] * pol["unit_cost"]).sum()), "Recommended orders", "at unit cost")])
    st.write("")
    c1, c2 = st.columns([1, 1.2])
    with c1, ui.panel("inv_scatter"):
        ui.title("Stockout probability vs days of cover")
        p = pol.dropna(subset=["stockout_probability"])
        fig = go.Figure(go.Scatter(x=p["days_cover"].clip(upper=200), y=p["stockout_probability"], mode="markers",
                                   marker=dict(size=9, color=[ui.STATUS.get(s, "#888") for s in p["status"]], line=dict(color="#fff", width=1)),
                                   text=p["sku"] + " " + p["name"], hovertemplate="%{text}<br>cover %{x:.0f} d<br>P(stockout) %{y:.0%}<extra></extra>"))
        fig.update_layout(height=360, xaxis_title="Days of cover (capped 200)", yaxis_title="P(stockout ≤14 d)", yaxis_tickformat=".0%")
        ui.chart(fig, "inv_scatter")
        st.caption("Colour = policy status (red critical, amber reorder, blue overstock, green healthy).")
    with c2, ui.panel("inv_models"):
        ui.title("Stockout classifier performance (time-based test set)")
        mdl = api.df("/api/v1/models")
        so = mdl[mdl["model_type"] == "stockout"].drop_duplicates("model_name")
        if not so.empty:
            mm = pd.json_normalize(so["metrics"]).assign(model=so["model_name"].values, champion=so["is_champion"].map({True: "🏆", False: ""}).values)
            ui.table(mm[["model", "champion", "f1", "roc_auc", "pr_auc", "precision", "recall", "threshold", "positive_rate", "n"]])
            st.caption(so["notes"].iat[0])
    with ui.panel("inv_tbl"):
        ui.title("Inventory policy by SKU")
        status = st.multiselect("Status", ["Critical", "Reorder", "Overstock", "Healthy"], default=["Critical", "Reorder"], key="inv_st")
        t = pol[pol["status"].isin(status)] if status else pol
        ui.table(t[["status", "sku", "name", "abc_class", "service_level", "avg_daily_demand", "demand_std", "lead_time_days",
                    "safety_stock", "reorder_point", "max_level", "on_hand", "on_order", "recommended_qty", "days_cover",
                    "stockout_probability", "demand_basis"]], height=420,
                 column_config={"service_level": st.column_config.NumberColumn(format="percent"),
                                "stockout_probability": st.column_config.ProgressColumn("P(stockout)", min_value=0, max_value=1, format="percent"),
                                **{c: st.column_config.NumberColumn(format="%.1f") for c in ["avg_daily_demand", "demand_std", "lead_time_days", "days_cover"]}})
        st.download_button("⬇ Inventory policy (CSV)", t.to_csv(index=False).encode(), "inventory_policy.csv", key="inv_dl")
