"""Streamlit entry point.   streamlit run app/dashboard/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Energy Grid Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

from app.config import REPORTS_DIR  # noqa: E402
from app.dashboard import (anomalies, consumption, data_explorer, forecasting, model_performance, overview,  # noqa: E402
                           peak_load, renewable, segmentation, statistics, weather)
from app.dashboard import data_client as dc  # noqa: E402
from app.dashboard.theme import inject_css  # noqa: E402

inject_css()


def _safe(render):
    def page():
        try:
            render()
        except Exception as exc:  # noqa: BLE001 - show a friendly error state instead of a stack trace
            st.error(f"This page could not be rendered: {exc}")
            st.caption("Check that the pipeline has completed (`python -m app.pipeline run`) and see logs/egi.log.")
    page.__name__ = render.__module__.split(".")[-1]
    return page


PAGES = [
    (overview.render, "Executive Overview", ":material/dashboard:"),
    (consumption.render, "Consumption Analytics", ":material/bolt:"),
    (forecasting.render, "Demand Forecasting", ":material/trending_up:"),
    (peak_load.render, "Peak Load Intelligence", ":material/show_chart:"),
    (segmentation.render, "Energy Segmentation", ":material/scatter_plot:"),
    (anomalies.render, "Anomaly Detection", ":material/warning:"),
    (renewable.render, "Renewable Energy Analytics", ":material/eco:"),
    (weather.render, "Weather Intelligence", ":material/thermostat:"),
    (statistics.render, "Statistical Research", ":material/functions:"),
    (model_performance.render, "Model Performance", ":material/speed:"),
    (data_explorer.render, "Data Explorer", ":material/table_view:"),
]
nav = st.navigation([st.Page(_safe(fn), title=t, icon=i, url_path=t.lower().replace(" ", "-"), default=(n == 0))
                     for n, (fn, t, i) in enumerate(PAGES)])

with st.sidebar:
    st.markdown("### ⚡ Energy Grid Intelligence")
    st.caption("UCI household · OPSD German grid")
    k = dc.doc("overview", "kpis")
    if k:
        st.caption(f"Results generated {k['generated_at'][:16].replace('T', ' ')} UTC")
    pdf = REPORTS_DIR / "energy_grid_intelligence_report.pdf"
    if pdf.exists():
        st.download_button("Download PDF report", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf", width="stretch")
    zp = REPORTS_DIR / "energy_grid_intelligence_tables.zip"
    if zp.exists():
        st.download_button("Download CSV tables", zp.read_bytes(), file_name=zp.name, mime="application/zip", width="stretch")

nav.run()
