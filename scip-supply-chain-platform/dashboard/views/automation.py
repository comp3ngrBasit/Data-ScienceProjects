from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard import api_client as api
from dashboard import ui


def render():
    ui.ribbon("Automation Engine", "Alerts · procurement actions · reports · notifications (n8n)")
    st_ = api.cached("/api/v1/status")["integrations"]
    ui.origin_note(f"n8n webhook: <b>{'configured' if st_['n8n_webhook_configured'] else 'not configured — events are logged only'}</b>. "
                   "Schedule <i>Run automation cycle</i> with the n8n cron workflow (n8n/workflows).")
    c1, c2 = st.columns([1, 3])
    auto = c1.toggle("Auto-create & submit POs", value=True, key="au_auto")
    if c1.button("▶ Run automation cycle", type="primary", key="au_run"):
        with st.spinner("Running cycle…"):
            try:
                r = api.post("/api/v1/automation/run", trigger="dashboard", auto_submit=auto)
                api.refresh()
                c2.success(f"Cycle {r['status']} in {r['seconds']}s — alerts {r.get('alerts')}, procurement {r.get('procurement')}, "
                           f"report #{r.get('report_id')}, {r.get('rag_documents')} Copilot documents indexed.")
            except api.APIError as e:
                c2.error(str(e))
    t1, t2, t3 = st.tabs(["Alerts", "Activity log", "Reports"])
    with t1:
        a = api.df("/api/v1/alerts")
        if a.empty:
            st.success("No open alerts.")
        else:
            cnt = a["severity"].value_counts()
            ui.kpi_row([ui.ikpi("⛔", ui.num(cnt.get("Critical", 0)), "Critical"), ui.ikpi("⚠️", ui.num(cnt.get("Warning", 0)), "Warning"),
                        ui.ikpi("ℹ️", ui.num(cnt.get("Info", 0)), "Info")])
            sev = st.multiselect("Severity", ["Critical", "Warning", "Info"], default=["Critical", "Warning"], key="au_sev")
            typ = st.multiselect("Type", sorted(a["alert_type"].unique()), key="au_typ")
            t = a[a["severity"].isin(sev)] if sev else a
            t = t[t["alert_type"].isin(typ)] if typ else t
            ui.table(t[["alert_id", "severity", "alert_type", "entity", "message", "created_at"]], height=420)
            aid = st.number_input("Acknowledge alert id", 0, step=1, key="au_ack")
            if st.button("Acknowledge", key="au_ack_btn") and aid:
                try:
                    api.post(f"/api/v1/alerts/{int(aid)}/acknowledge")
                    api.refresh()
                    st.success(f"Alert {int(aid)} acknowledged.")
                except api.APIError as e:
                    st.error(str(e))
    with t2:
        lg = api.df("/api/v1/automation/log", limit=500)
        if not lg.empty:
            lg["details"] = lg["details"].astype(str)
            ui.table(lg, height=480)
    with t3:
        reps = api.df("/api/v1/reports")
        if st.button("Generate daily operations report now", key="au_rep"):
            try:
                api.post("/api/v1/reports/daily")
                api.refresh()
                st.rerun()
            except api.APIError as e:
                st.error(str(e))
        if reps.empty:
            st.info("No reports yet.")
        else:
            rid = st.selectbox("Report", reps["report_id"], format_func=lambda i: f"#{i} · {reps.set_index('report_id').at[i, 'title']}", key="au_rsel")
            r = api.cached(f"/api/v1/reports/{int(rid)}")
            st.download_button("⬇ Download report (HTML)", r["html"].encode(), f"report_{rid}.html", "text/html", key="au_rdl")
            components.html(r["html"], height=700, scrolling=True)
