from __future__ import annotations

import streamlit as st

from dashboard import api_client as api
from dashboard import ui

EXAMPLES = ["How much inventory do we have and what is the 30-day demand forecast?", "Which products are about to stock out?",
            "Which suppliers are the riskiest?", "How accurate is the forecasting model?", "What is the status of SKU 22423?",
            "Which purchase orders are waiting for approval?"]


def render():
    ui.ribbon("AI Supply Chain Copilot", "Retrieval-augmented answers grounded in the platform database")
    status = api.cached("/api/v1/status")["llm"]
    if status["active"]:
        ui.origin_note(f"LLM: <b>{status['active']}</b> ({status['model']}). Answers cite the database records they use.")
    else:
        ui.origin_note("No LLM key configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY). The Copilot runs in "
                       "<b>retrieval-only</b> mode and shows the most relevant database records.")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex_{i}", width="stretch"):
            st.session_state["pending_q"] = ex
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.caption("Sources: " + " · ".join(f"[{s['doc_id']}]" for s in m["sources"]))
    q = st.chat_input("Ask about inventory, demand, suppliers, purchase orders, logistics…") or st.session_state.pop("pending_q", None)
    if q:
        st.session_state["chat"].append({"role": "user", "content": q})
        try:
            r = api.post("/api/v1/copilot/ask", {"question": q})
            st.session_state["chat"].append({"role": "assistant", "content": r["answer"].replace("$", "\\$"),
                                             "sources": r["sources"], "provider": r["provider"]})
        except api.APIError as e:
            st.session_state["chat"].append({"role": "assistant", "content": f"⚠️ {e}"})
        st.rerun()
