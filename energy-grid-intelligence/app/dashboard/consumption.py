"""Page 2 - Consumption Analytics."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, SEQ, SERIES, callout, fmt, header, kpi_row, missing, style

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def render() -> None:
    header("Consumption Analytics", "Hourly, daily, weekly and seasonal patterns; sub-metering and voltage")
    scope = st.radio("Dataset", ["Household (kWh)", "German grid (MW)"], horizontal=True, label_visibility="collapsed")
    pre = "household" if scope.startswith("House") else "grid"
    unit = "kWh" if pre == "household" else "MW"
    s = dc.doc("consumption", "summary")
    if not s:
        missing("Consumption analytics")
        return
    d = s[pre]
    stats = d["daily_kwh"] if pre == "household" else d["daily_load_mwh"]
    h = d["hourly_kwh"] if pre == "household" else d["hourly_load_mw"]
    kpi_row([("Mean hourly", fmt(h["mean"], 2), unit, f"median {fmt(h['p50'], 2)}", BLUE),
             ("Hourly P99", fmt(h["p99"], 2), unit, f"max {fmt(h['max'], 2)}", SERIES[1]),
             ("Mean daily", fmt(stats["mean"] / (1 if pre == "household" else 1e3), 1), "kWh" if pre == "household" else "GWh",
              f"CV {fmt(stats['cv'], 2)} · skew {fmt(stats['skew'], 2)}", SERIES[2]),
             ("Daily range", f"{fmt(stats['min'] / (1 if pre == 'household' else 1e3), 1)} - {fmt(stats['max'] / (1 if pre == 'household' else 1e3), 1)}",
              "kWh" if pre == "household" else "GWh", "min - max day", SERIES[3])])
    c1, c2 = st.columns(2)
    prof = dc.table("consumption", f"{pre}_hourly_by_daytype")
    if prof is not None:
        fig = go.Figure()
        for i, dt in enumerate(["Weekday", "Weekend", "Holiday"]):
            p = prof[prof["daytype"] == dt]
            fig.add_scatter(x=p["hour"], y=p["mean"], name=dt, line=dict(color=SERIES[i], width=2),
                            hovertemplate=f"{dt} %{{x}}:00 - %{{y:,.2f}} {unit}<extra></extra>")
        c1.plotly_chart(style(fig, 360, f"Hourly load curve by day type ({unit})", xaxis_title="Hour of day"), width="stretch")
    seas = dc.table("consumption", f"{pre}_hourly_by_season")
    if seas is not None:
        fig = go.Figure()
        for i, sname in enumerate(["Winter", "Spring", "Summer", "Autumn"]):
            fig.add_scatter(x=seas["hour"], y=seas[sname], name=sname, line=dict(color=SERIES[i], width=2),
                            hovertemplate=f"{sname} %{{x}}:00 - %{{y:,.2f}} {unit}<extra></extra>")
        c2.plotly_chart(style(fig, 360, f"Daily profile by season ({unit})", xaxis_title="Hour of day"), width="stretch")
    c3, c4 = st.columns(2)
    hm = dc.table("consumption", f"{pre}_heatmap_dow_hour")
    if hm is not None:
        fig = go.Figure(go.Heatmap(z=hm.values, x=[int(c) for c in hm.columns], y=DOW, colorscale=SEQ,
                                   colorbar=dict(title=unit), hovertemplate="%{y} %{x}:00<br>%{z:,.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        c3.plotly_chart(style(fig, 330, "Weekly demand heatmap (mean by weekday x hour)"), width="stretch")
    hm2 = dc.table("consumption", f"{pre}_heatmap_month_hour")
    if hm2 is not None:
        fig = go.Figure(go.Heatmap(z=hm2.values, x=[int(c) for c in hm2.columns], y=[MONTHS[m - 1] for m in hm2.index],
                                   colorscale=SEQ, colorbar=dict(title=unit), hovertemplate="%{y} %{x}:00<br>%{z:,.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        c4.plotly_chart(style(fig, 330, "Hourly consumption heatmap (month x hour)"), width="stretch")
    mby = dc.table("consumption", f"{pre}_monthly_by_year")
    if mby is not None:
        fig = go.Figure()
        for i, y in enumerate(mby.columns):
            fig.add_scatter(x=[MONTHS[m - 1] for m in mby.index], y=mby[y] / (1 if pre == "household" else 1e6), name=str(y),
                            mode="lines+markers", line=dict(color=SERIES[i] if i < len(SERIES) else "#64748B", width=2),
                            marker=dict(size=8), hovertemplate=f"{y} %{{x}}: %{{y:,.1f}}<extra></extra>")
        st.plotly_chart(style(fig, 360, f"Monthly consumption by year ({'kWh' if pre == 'household' else 'TWh'}; partial years at the edges)"),
                        width="stretch")
    if pre == "household":
        st.markdown("#### Sub-metering")
        share = d["submeter_share"]
        labels = {"sub_kitchen_kwh": "Kitchen", "sub_laundry_kwh": "Laundry", "sub_water_heater_ac_kwh": "Water heater / AC",
                  "sub_other_kwh": "Unmetered remainder"}
        c5, c6 = st.columns([1, 2])
        fig = go.Figure(go.Bar(y=[labels[k] for k in share], x=[v * 100 for v in share.values()], orientation="h",
                               marker_color=SERIES[:4], text=[f"{v * 100:.1f} %" for v in share.values()], textposition="outside",
                               hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
        c5.plotly_chart(style(fig, 330, "Share of total energy"), width="stretch")
        sub = dc.table("consumption", "household_submeter_by_hour")
        if sub is not None:
            fig = go.Figure()
            for i, (col, lab) in enumerate(labels.items()):
                fig.add_scatter(x=sub.index, y=sub[col], name=lab, stackgroup="one", line=dict(width=0.5, color=SERIES[i]),
                                hovertemplate=f"{lab} %{{x}}:00 - %{{y:.3f}} kWh<extra></extra>")
            c6.plotly_chart(style(fig, 330, "Mean hourly energy by sub-meter (kWh, stacked)", xaxis_title="Hour"), width="stretch")
        v = dc.processed("household_hourly")
        if v is not None:
            vv = v["voltage_v"].resample("D").mean()
            fig = go.Figure(go.Scatter(x=vv.index, y=vv, line=dict(color=SERIES[3], width=1.5), name="Voltage",
                                       hovertemplate="%{x|%d %b %Y}: %{y:.1f} V<extra></extra>"))
            fig.add_hrect(y0=207, y1=253, fillcolor="#22C55E", opacity=0.05, line_width=0)
            st.plotly_chart(style(fig, 300, "Daily mean voltage (V); band = EN 50160 limits 207-253 V"), width="stretch")
            callout(f"Voltage mean {fmt(d['voltage_v']['mean'], 1)} V (P1 {fmt(d['voltage_v']['p1'], 1)}, P99 {fmt(d['voltage_v']['p99'], 1)}). "
                    "Global intensity tracks active power closely (P ≈ V·I·pf); both are summarised in the Data Explorer.")
