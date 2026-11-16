from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import api_client as api
from dashboard import ui


def render():
    ui.ribbon("Data & Models", "Sources · lineage · model registry · background jobs")
    status = api.cached("/api/v1/status")
    src = api.df("/api/v1/data/sources")
    with ui.panel("dm_src"):
        ui.title("Data sources (real vs simulated)")
        ui.table(src, column_config={"url": st.column_config.LinkColumn("URL")})
    c1, c2 = st.columns(2)
    with c1, ui.panel("dm_counts"):
        ui.title("Database tables (PostgreSQL)")
        ui.table(pd.DataFrame(list(status["table_counts"].items()), columns=["table", "rows"]).sort_values("rows", ascending=False), height=380)
    with c2, ui.panel("dm_status"):
        ui.title("Platform status")
        st.json({k: status[k] for k in ("as_of_date", "ingested_at", "bootstrapped_at", "llm", "integrations")})
    with ui.panel("dm_models"):
        ui.title("Model registry")
        m = api.df("/api/v1/models")
        if not m.empty:
            m["metrics"] = m["metrics"].map(lambda d: ", ".join(f"{k}={v:.3f}" for k, v in d.items()
                                                                if isinstance(v, (int, float)) and k in
                                                                ("mae", "rmse", "mape", "wape", "weekly_wape", "f1", "roc_auc", "pr_auc")))
            ui.table(m[["run_id", "model_type", "model_name", "is_champion", "metrics", "data_window", "trained_at"]], height=360)
    with ui.panel("dm_jobs"):
        ui.title("Background jobs")
        c1, c2, c3 = st.columns(3)
        if c1.button("Re-train classifiers", key="dm_rt"):
            st.info(api.post("/api/v1/models/retrain"))
        if c2.button("Re-train forecasts", key="dm_fc"):
            st.info(api.post("/api/v1/forecasts/run"))
        if c3.button("Full rebuild (ingest → train → automate)", key="dm_bs"):
            st.warning(api.post("/api/v1/pipeline/bootstrap"))
        jobs = api.get("/api/v1/jobs")
        if jobs:
            ui.table(pd.DataFrame(jobs)[["job_id", "name", "status", "started_at", "error"]])
