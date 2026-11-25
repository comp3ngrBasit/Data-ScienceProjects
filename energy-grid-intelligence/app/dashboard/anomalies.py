"""Page 6 - Anomaly Detection."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, RED, SERIES, SEVERITY, TEXT, callout, fmt, header, kpi_row, missing, style


def render() -> None:
    header("Anomaly Detection", "Six detectors against a causal seasonal baseline; consensus requires two or more detectors")
    s = dc.doc("anomaly", "summary")
    cons, scored = dc.table("anomaly", "consensus"), dc.table("anomaly", "scored_series")
    if not s or cons is None or scored is None:
        missing("Anomaly detection")
        return
    callout(f"<b>Interpretation.</b> {s['disclaimer']}", RED)
    sev = s["consensus_by_severity"]
    kpi_row([("Consensus anomalies", fmt(s["n_consensus_anomalies"], 0), "hours", f"of {s['hours_scored']:,} hours scored", RED),
             ("Critical / high", f"{sev.get('critical', 0)} / {sev.get('high', 0)}", "", s["severity_rule"][:60] + "…", SEVERITY["high"]),
             ("Daily anomalies", fmt(s["n_daily_anomalies"], 0), "days", "daily total & volatility detectors", SERIES[1]),
             ("Data-quality findings", fmt(s["n_data_quality_findings"], 0), "", "outages, voltage, sub-meter inconsistency", SERIES[3])])
    c0, c1, c2 = st.columns([2, 1, 1])
    lo, hi = scored.index.min().date(), scored.index.max().date()
    rng = c0.date_input("Period", value=(pd.Timestamp(hi) - pd.Timedelta(days=60), hi), min_value=lo, max_value=hi)
    sevs = c1.multiselect("Severity", ["critical", "high", "medium", "low"], default=["critical", "high", "medium", "low"])
    types = c2.multiselect("Type", sorted(cons["type"].unique()), default=sorted(cons["type"].unique()))
    a, b = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1)) if isinstance(rng, tuple) and len(rng) == 2 else (pd.Timestamp(lo), pd.Timestamp(hi))
    sc = scored.loc[a:b]
    cf = cons[(cons["timestamp"] >= a) & (cons["timestamp"] < b) & cons["severity"].isin(sevs) & cons["type"].isin(types)]
    fig = go.Figure()
    fig.add_scatter(x=sc.index, y=sc["expected"], name="Expected (4-week same-hour median)", line=dict(color=SERIES[1], width=1.2, dash="dot"))
    fig.add_scatter(x=sc.index, y=sc["observed"], name="Observed", line=dict(color=BLUE, width=1.2),
                    hovertemplate="%{x|%d %b %H:%M}: %{y:.2f} kWh<extra></extra>")
    for sv in ["low", "medium", "high", "critical"]:
        d = cf[cf["severity"] == sv]
        if len(d):
            fig.add_scatter(x=d["timestamp"], y=d["observed_kwh"], mode="markers", name=sv.title(),
                            marker=dict(size=10, color=SEVERITY[sv], line=dict(width=2, color="#151F32"), symbol="diamond"),
                            customdata=d[["expected_kwh", "deviation_pct", "methods"]],
                            hovertemplate="%{x|%d %b %Y %H:%M}<br>observed %{y:.2f} kWh<br>expected %{customdata[0]:.2f} kWh"
                                          "<br>deviation %{customdata[1]:+.0f} %<br>%{customdata[2]}<extra>" + sv + "</extra>")
    st.plotly_chart(style(fig, 420, "Anomaly timeline (kWh per hour)"), width="stretch")
    st.markdown(f"#### Consensus anomalies in view ({len(cf)})")
    cols = ["timestamp", "observed_kwh", "expected_kwh", "deviation_pct", "severity", "severity_score", "type", "methods",
            "driver_submeter", "evidence"]
    st.dataframe(cf.sort_values("severity_score", ascending=False)[cols].style.format(
        {"observed_kwh": "{:.2f}", "expected_kwh": "{:.2f}", "deviation_pct": "{:+.0f} %", "severity_score": "{:.1f}"}),
        width="stretch", hide_index=True, height=320)
    c3, c4 = st.columns(2)
    fm = pd.Series(s["flags_by_method"]).sort_values()
    fig = go.Figure(go.Bar(x=fm.values, y=fm.index, orientation="h", marker_color=BLUE, text=fm.values, textposition="outside"))
    c3.plotly_chart(style(fig, 330, "Hours flagged by each detector"), width="stretch")
    ag = dc.table("anomaly", "method_agreement")
    if ag is not None:
        m = ag.set_index("method")
        fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale=[[0, "#0B1A33"], [1, "#3B82F6"]], zmin=0, zmax=1,
                                   text=m.round(2).values, texttemplate="%{text}", textfont=dict(color=TEXT, size=10),
                                   hovertemplate="%{y} vs %{x}: Jaccard %{z:.2f}<extra></extra>"))
        c4.plotly_chart(style(fig, 330, "Detector agreement (Jaccard index of flagged hours)"), width="stretch")
    c5, c6 = st.columns(2)
    drv = pd.Series(s.get("consensus_by_driver", {})).sort_values()
    if len(drv):
        fig = go.Figure(go.Bar(x=drv.values, y=drv.index, orientation="h", marker_color=SERIES[2], text=drv.values, textposition="outside"))
        c5.plotly_chart(style(fig, 280, "Sub-meter with the largest deviation at each anomaly"), width="stretch")
    rep = dc.table("anomaly", "repeated_patterns")
    if rep is not None and len(rep):
        c6.markdown("##### Repeated abnormal patterns")
        c6.dataframe(rep.style.format({"ratio_to_uniform": "{:.2f}"}), width="stretch", hide_index=True, height=250)
    t1, t2 = st.tabs(["Daily anomalies", "Potential meter / data-quality issues"])
    dd = dc.table("anomaly", "daily_events")
    t1.dataframe(dd, width="stretch", hide_index=True)
    dq = dc.table("anomaly", "data_quality")
    t2.dataframe(dq, width="stretch", hide_index=True)
    t2.caption("Data-quality findings are candidates for verification with the meter operator, not confirmed faults.")
