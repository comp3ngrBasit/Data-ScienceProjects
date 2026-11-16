from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard import ui


def render():
    ui.ribbon("Supplier Intelligence", "Delivery-risk prediction · performance scoring · shipping analytics")
    ui.origin_note("Delivery-risk models are trained on <b>real</b> DataCo shipments (late-delivery labels). Vendor POs are "
                   "<b>simulated</b>, with lateness calibrated to real DataCo delays by shipping mode.")
    mdl = api.df("/api/v1/models")
    dr = mdl[mdl["model_type"] == "delivery_risk"].drop_duplicates("model_name")
    c1, c2 = st.columns([1.3, 1])
    with c1, ui.panel("sup_models"):
        ui.title("Delivery-risk classifiers — most recent 20% of orders (chronological test set)")
        if not dr.empty:
            mm = pd.json_normalize(dr["metrics"]).assign(model=dr["model_name"].values,
                                                         champion=dr["is_champion"].map({True: "🏆", False: ""}).values)
            ui.table(mm[["model", "champion", "roc_auc", "pr_auc", "f1", "precision", "recall", "threshold",
                         "baseline_always_late_f1", "n"]])
            st.caption("Threshold chosen by Youden's J on a validation slice. 'Always late' F1 shown as the naive baseline — "
                       "F1 alone is misleading when 57% of shipments are late, so ROC-AUC selects the champion. "
                       "Leakage columns (actual shipping days, delivery status) are excluded.")
    with c2, ui.panel("sup_modes"):
        ui.title("Real late-delivery rate by shipping mode (DataCo)")
        perf = api.df("/api/v1/shipments/performance")
        if not perf.empty:
            g = perf.groupby("ship_mode").apply(lambda x: pd.Series({"late": (x["late_rate"] * x["shipments"]).sum() / x["shipments"].sum(),
                                                                     "n": x["shipments"].sum()}), include_groups=False).sort_values("late")
            fig = go.Figure(go.Bar(x=g["late"], y=g.index, orientation="h", marker_color=ui.BRAND,
                                   text=[f"{v:.0%} of {int(n):,}" for v, n in zip(g["late"], g["n"])], textposition="outside"))
            fig.update_layout(height=250, xaxis_tickformat=".0%", xaxis_range=[0, 1.25])
            ui.chart(fig, "sup_modes")
    sc = api.df("/api/v1/suppliers/scores")
    with ui.panel("sup_scores"):
        ui.title("Supplier scorecard")
        ui.table(sc[["supplier_id", "name", "country_code", "tier", "score", "shipments", "on_time_rate", "avg_delay_days",
                     "predicted_late_risk", "lead_time_cv", "data_origin"]],
                 column_config={"score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                                "on_time_rate": st.column_config.NumberColumn("On-time", format="percent"),
                                "predicted_late_risk": st.column_config.NumberColumn("Predicted late risk", format="percent")})
        st.caption("Score = 100 × (0.40 on-time + 0.20 fill rate + 0.15 lead-time consistency + 0.15 (1 − predicted risk) + 0.10 quality).")
    with ui.panel("sup_detail"):
        sid = st.selectbox("Supplier detail", sc["supplier_id"], format_func=lambda s: sc.set_index("supplier_id").at[s, "name"], key="sup_sel")
        d = api.cached(f"/api/v1/suppliers/{sid}")
        po = pd.DataFrame(d["purchase_orders"])
        c1, c2 = st.columns([1.4, 1])
        with c1:
            if not po.empty:
                rec = po.dropna(subset=["received_date"]).copy()
                rec["delay"] = (pd.to_datetime(rec["received_date"]) - pd.to_datetime(rec["promised_date"])).dt.days
                fig = go.Figure(go.Histogram(x=rec["delay"], marker_color=ui.BRAND, xbins=dict(size=1)))
                fig.update_layout(title="Delivery delay vs promised date (days; ≤0 = on time)", height=280, xaxis_title="days late")
                ui.chart(fig, "sup_delay")
        with c2:
            ui.table(pd.DataFrame(d["catalog"]), height=280)
