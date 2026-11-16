"""SCIP Control Tower (Streamlit). Run: streamlit run dashboard/app.py  (needs the API: uvicorn scip.api.main:app)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="SCIP Control Tower", page_icon="🛰️", layout="wide")

from dashboard import api_client as api  # noqa: E402
from dashboard import ui  # noqa: E402
from dashboard.views import (automation, control_tower, copilot, data_models, forecasting, inventory,  # noqa: E402
                             logistics, procurement, suppliers)

PAGES = {
    "🛰️ Control Tower": control_tower.render, "📈 Demand Forecasting": forecasting.render,
    "📦 Inventory & Stockout": inventory.render, "🏭 Supplier Intelligence": suppliers.render,
    "🧾 Procurement": procurement.render, "🚚 Logistics": logistics.render, "🤖 AI Copilot": copilot.render,
    "⚙️ Automation": automation.render, "🗄️ Data & Models": data_models.render,
}


def main():
    ui.setup()
    with st.sidebar:
        st.markdown("<div class='brand'>🛰️ SCIP</div><div class='tag'>Supply Chain Intelligence &amp; Autonomous Operations</div>",
                    unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        if st.button("↻ Refresh data", width="stretch"):
            api.refresh()
        st.caption(f"API: {api.API_URL}")
    try:
        status = api.cached("/api/v1/status")
    except api.APIError as e:
        st.error(str(e))
        st.stop()
    if not status["table_counts"].get("products"):
        ui.ribbon("SCIP Control Tower")
        st.warning("The database is empty. Run `python scripts/bootstrap.py` (or the button below) to ingest data and train models.")
        if st.button("Run full bootstrap now"):
            st.info(api.post("/api/v1/pipeline/bootstrap"))
        st.stop()
    try:
        PAGES[page]()
    except api.APIError as e:
        st.error(str(e))


main()
