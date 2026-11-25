"""Page 11 - Data Explorer: processed datasets, data quality, provenance and every result table."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import OPSD_SOURCE, OPSD_WEATHER_SOURCE, RESULTS_DIR, UCI_SOURCE
from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, SERIES, callout, header, missing, style

DATASETS = {"household_hourly": "Household - hourly (kWh)", "household_daily": "Household - daily (kWh)", "grid_DE_hourly": "German grid - hourly (MW, UTC)"}


def render() -> None:
    header("Data Explorer", "Processed datasets, quality reports, provenance and all analytical result tables")
    t1, t2, t3 = st.tabs(["Processed data", "Data quality & provenance", "Result tables"])
    with t1:
        name = st.selectbox("Dataset", list(DATASETS), format_func=DATASETS.get)
        df = dc.processed(name)
        if df is None:
            missing("Processed data")
            return
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]
        c1, c2 = st.columns([1, 3])
        cols = c1.multiselect("Variables", num, default=num[:1])
        lo, hi = df.index.min().date(), df.index.max().date()
        rng = c1.date_input("Period", value=(max(lo, (pd.Timestamp(hi) - pd.Timedelta(days=90)).date()), hi), min_value=lo, max_value=hi)
        res = c1.selectbox("Resample", ["none", "D", "W", "MS"], index=0 if "daily" in name else 1)
        a, b = (str(rng[0]), str(rng[1])) if isinstance(rng, tuple) and len(rng) == 2 else (str(lo), str(hi))
        idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
        view = df.set_axis(idx).loc[a:b, cols] if cols else df.iloc[:0]
        if res != "none" and len(view):
            view = view.resample(res).mean()
        fig = go.Figure()
        for i, c in enumerate(cols[:6]):
            fig.add_scatter(x=view.index, y=view[c], name=c, line=dict(color=SERIES[i], width=1.5))
        fig.update_layout(hovermode="x unified")
        c2.plotly_chart(style(fig, 420, "Time series"), width="stretch")
        st.markdown("##### Descriptive statistics (selected period)")
        st.dataframe(view.describe().T.style.format(precision=3), width="stretch")
        st.download_button("Download selection (CSV)", view.to_csv().encode(), file_name=f"{name}_{a}_{b}.csv", mime="text/csv")
        if len(cols) >= 2:
            fig = go.Figure(go.Scatter(x=view[cols[0]], y=view[cols[1]], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.4)))
            st.plotly_chart(style(fig, 360, f"{cols[1]} vs {cols[0]}", xaxis_title=cols[0], yaxis_title=cols[1]), width="stretch")
    with t2:
        for src in (UCI_SOURCE, OPSD_SOURCE, OPSD_WEATHER_SOURCE):
            callout(f"<b>{src.name}</b><br>{src.citation}<br>Official URL: {src.urls[0]}"
                    + (f"<br>Integrity: SHA-256 {src.sha256[:16]}… verified on download" if src.sha256 else ""))
        for key, label in (("household", "Household"), ("grid_DE", "German grid")):
            q = dc.doc("data_quality", key)
            if not q:
                continue
            st.markdown(f"#### {label}")
            qr = q["quality_report"]
            st.dataframe(pd.DataFrame(qr["checks"]).assign(passed=lambda d: d["passed"].map({True: "✓ pass", False: "✗ fail"})),
                         width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            c1.markdown("Missing values by column")
            c1.dataframe(pd.Series(qr["missing_by_column"], name="missing").to_frame(), width="stretch")
            c2.markdown(f"Gaps: {qr['gaps'].get('n_gaps')} runs, longest {qr['gaps'].get('longest')} {qr['gaps'].get('unit')}")
            c2.dataframe(pd.Series(qr["gaps"].get("length_distribution", {}), name="runs").to_frame("runs"), width="stretch")
            extra = {k: v for k, v in q.items() if k != "quality_report"}
            with st.expander("Processing log"):
                st.json(extra)
    with t3:
        mods = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())
        c1, c2 = st.columns(2)
        mod = c1.selectbox("Module", mods)
        names = sorted(p.stem for p in (RESULTS_DIR / mod).glob("*.parquet"))
        if not names:
            st.info("No tables in this module")
            return
        nm = c2.selectbox("Table", names)
        t = dc.table(mod, nm)
        st.caption(f"{len(t):,} rows × {t.shape[1]} columns")
        st.dataframe(t.head(5000), width="stretch")
        st.download_button("Download table (CSV)", t.to_csv().encode(), file_name=f"{mod}__{nm}.csv", mime="text/csv")
