"""Page 1 - Executive Overview."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import AMBER, BLUE, CYAN, GREEN, RED, SERIES, SEVERITY, TEXT2, callout, fmt, header, kpi_row, missing, style


def render() -> None:
    header("Executive Overview", "Headline indicators computed from the UCI household and OPSD German grid pipelines")
    k = dc.doc("overview", "kpis")
    if not k:
        missing("The KPI summary")
        return
    st.markdown("##### Household · Sceaux, France (UCI, 2006-2010)")
    kpi_row([
        ("Total consumption", fmt(k.get("household_total_kwh") / 1e3, 1), "MWh", "whole recording period", BLUE),
        ("Average daily consumption", fmt(k.get("household_avg_daily_kwh"), 1), "kWh/day", f"daily CV {fmt(k.get('household_daily_cv'), 2)}", BLUE),
        ("Maximum hourly demand", fmt(k.get("household_max_hourly_kwh"), 2), "kWh/h", str(k.get("household_max_time", ""))[:16], AMBER),
        ("Detected anomalies", fmt(k.get("consensus_anomalies"), 0), "hours",
         " · ".join(f"{s}: {n}" for s, n in (k.get("anomalies_by_severity") or {}).items()), RED),
    ])
    st.markdown("##### German transmission grid (OPSD / ENTSO-E, 2015-2020)")
    kpi_row([
        ("Total electricity load", fmt(k.get("grid_total_twh"), 0), "TWh", f"avg {fmt(k.get('grid_avg_daily_gwh'), 0)} GWh/day", CYAN),
        ("Maximum recorded demand", fmt(k.get("grid_max_load_mw") / 1e3, 1), "GW", str(k.get("grid_max_time", ""))[:16], AMBER),
        ("Renewable (wind + solar) share", fmt(k.get("vre_penetration_pct"), 1), "%", "VRE energy / load energy, same hours", GREEN),
        ("Consumption variability", fmt(k.get("grid_daily_cv") * 100, 1), "% CV", "coefficient of variation of daily load", CYAN),
    ])
    st.markdown("##### Forecasting")
    kpi_row([
        ("Household 24 h - best model", str(k.get("household_hourly_h24_best_model", "n/a")), "",
         f"WAPE {fmt(k.get('household_hourly_h24_best_wape'), 1)} % · {fmt(k.get('household_hourly_h24_skill_vs_naive_pct'), 1)} % better than seasonal naive", BLUE),
        ("Grid 24 h - best model", str(k.get("grid_hourly_h24_best_model", "n/a")), "",
         f"WAPE {fmt(k.get('grid_hourly_h24_best_wape'), 2)} % · 90 % PI coverage {fmt(k.get('grid_hourly_h24_pi_coverage'), 1)} %", BLUE),
        ("Forecasted demand - household", fmt(k.get("household_hourly_h24_forward_total"), 1), "kWh",
         f"next 24 h after {str(k.get('household_hourly_h24_forward_from', ''))[:10]} (end of data)", SERIES[1]),
        ("Forecasted demand - grid", fmt((k.get("grid_hourly_h24_forward_total") or 0) / 1e3, 0), "GWh",
         f"next 24 h after {str(k.get('grid_hourly_h24_forward_from', ''))[:10]}; peak {fmt((k.get('grid_hourly_h24_forward_peak') or 0) / 1e3, 1)} GW", SERIES[1]),
    ])
    c1, c2 = st.columns(2)
    w = dc.table("consumption", "household_weekly")
    if w is not None:
        w = w.iloc[1:-1]
        fig = go.Figure(go.Scatter(x=w.index, y=w["energy_kwh"], line=dict(color=BLUE, width=2), name="Weekly kWh",
                                   hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} kWh<extra></extra>"))
        c1.plotly_chart(style(fig, 320, "Household weekly consumption (kWh)"), width="stretch")
    g = dc.table("consumption", "grid_daily")
    if g is not None:
        fig = go.Figure()
        fig.add_scatter(x=g.index, y=g["load_mwh"] / 1e3, line=dict(color=SERIES[2], width=1), name="Daily load", opacity=0.6,
                        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} GWh<extra></extra>")
        fig.add_scatter(x=g.index, y=(g["load_mwh"] / 1e3).rolling(30, center=True).mean(), line=dict(color=BLUE, width=2),
                        name="30-day mean", hovertemplate="%{y:,.0f} GWh<extra></extra>")
        c2.plotly_chart(style(fig, 320, "German daily load (GWh)"), width="stretch")
    c3, c4 = st.columns(2)
    a = dc.doc("anomaly", "summary")
    if a:
        sev = a["consensus_by_severity"]
        order = [s for s in ("critical", "high", "medium", "low") if s in sev]
        fig = go.Figure(go.Bar(x=[sev[s] for s in order], y=order, orientation="h", marker_color=[SEVERITY[s] for s in order],
                               text=[sev[s] for s in order], textposition="outside", hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        c3.plotly_chart(style(fig, 300, "Consensus anomalies by severity"), width="stretch")
    r = dc.table("renewable", "annual")
    if r is not None:
        fig = go.Figure()
        x = r.index.astype(str)
        fig.add_bar(x=x, y=r["wind_twh"], name="Wind", marker_color=SERIES[2], hovertemplate="%{x}: %{y:,.1f} TWh<extra>Wind</extra>")
        fig.add_bar(x=x, y=r["solar_twh"], name="Solar", marker_color=SERIES[1], hovertemplate="%{x}: %{y:,.1f} TWh<extra>Solar</extra>")
        fig.update_layout(barmode="stack", bargap=0.35)
        for xi, tot, pen in zip(x, r["vre_twh"], r["vre_penetration_pct"]):
            fig.add_annotation(x=xi, y=tot, text=f"{pen:.1f} %", showarrow=False, yshift=10, font=dict(color=TEXT2, size=11))
        c4.plotly_chart(style(fig, 300, "German wind + solar generation (TWh); label = share of load"), width="stretch")
    callout("<b>Scope.</b> The household and the German grid are different geographies, periods and measurement scales; "
            "they are analysed independently and never combined. 2015 and 2020 are partial years for the grid.")
