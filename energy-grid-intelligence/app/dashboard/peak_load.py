"""Page 4 - Peak Load Intelligence."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import AMBER, BLUE, SERIES, TEXT2, callout, fmt, header, kpi_row, missing, style

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]


def render() -> None:
    header("Peak Load Intelligence", "Daily, weekly and monthly peaks, peak-to-average ratios, high-demand events and extreme values")
    scope = st.radio("Dataset", ["Household", "German grid"], horizontal=True, label_visibility="collapsed")
    pre = "household" if scope == "Household" else "grid"
    unit, sc = ("kWh/h", 1) if pre == "household" else ("GW", 1e3)
    s = (dc.doc("peak", "summary") or {}).get(pre)
    daily = dc.table("peak", f"{pre}_daily")
    if not s or daily is None:
        missing("Peak analysis")
        return
    ev = s["extreme_value"]
    kpi_row([("Maximum demand", fmt(s["max"]["value"] / sc, 2), unit, s["max"]["timestamp"][:16], AMBER),
             ("Mean daily peak-to-average", fmt(s["mean_daily_par"], 2), "x", f"load factor {fmt(s['mean_daily_load_factor'], 2)}", BLUE),
             ("Peak-window energy share", fmt(s["peak_share_of_energy"] * 100, 1), "%",
              f"{fmt(s['peak_share_of_hours'] * 100, 1)} % of hours · {s['peak_definition']}", SERIES[2]),
             ("10-year return level", fmt(ev.get("return_levels", {}).get("10y", float("nan")) / sc, 2), unit,
              f"95 % CI {fmt(ev.get('return_level_10y_ci95', [0, 0])[0] / sc, 2)}-{fmt(ev.get('return_level_10y_ci95', [0, 0])[1] / sc, 2)}", SERIES[3])])
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily["peak"] / sc, name="Daily peak", line=dict(color="#33415C", width=1),
                    hovertemplate="%{x|%d %b %Y}: %{y:,.2f}<extra></extra>")
    fig.add_scatter(x=daily.index, y=daily["peak_ma7"] / sc, name="7-day moving average", line=dict(color=BLUE, width=1.6))
    fig.add_scatter(x=daily.index, y=daily["peak_ma30"] / sc, name="30-day moving average", line=dict(color=SERIES[1], width=2))
    fig.add_hline(y=s["high_demand_threshold_p99"] / sc, line=dict(color=AMBER, dash="dot", width=1),
                  annotation_text="P99 hourly threshold", annotation_font_color=TEXT2)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 380, f"Daily peak demand with moving averages ({unit})"), width="stretch")
    c1, c2 = st.columns(2)
    mon = dc.table("peak", f"{pre}_monthly")
    if mon is not None:
        fig = go.Figure(go.Bar(x=mon.index, y=mon["peak"] / sc, marker_color=BLUE, customdata=mon["peak_time"],
                               hovertemplate="%{x|%b %Y}: %{y:,.2f}<br>at %{customdata}<extra></extra>"))
        c1.plotly_chart(style(fig, 340, f"Monthly peak demand ({unit})"), width="stretch")
    wk = dc.table("peak", f"{pre}_weekly")
    if wk is not None:
        fig = go.Figure(go.Scatter(x=wk.index, y=wk["par"], line=dict(color=SERIES[2], width=1.8),
                                   hovertemplate="week of %{x|%d %b %Y}: %{y:.2f}<extra></extra>"))
        c2.plotly_chart(style(fig, 340, "Weekly peak-to-average ratio"), width="stretch")
    c3, c4 = st.columns(2)
    fig = go.Figure()
    for i, (lab, flag) in enumerate([("Weekday", False), ("Weekend", True)]):
        fig.add_box(y=daily.loc[daily["is_weekend"] == flag, "peak"] / sc, name=lab, marker_color=SERIES[i], boxmean=True)
    for j, sname in enumerate(SEASONS):
        fig.add_box(y=daily.loc[daily["season"] == sname, "peak"] / sc, name=sname, marker_color=SERIES[2 + j % 4], boxmean=True)
    c3.plotly_chart(style(fig, 360, f"Daily peak distribution: weekday/weekend and season ({unit})", showlegend=False), width="stretch")
    ph = dc.table("peak", f"{pre}_peak_hour_dist")
    if ph is not None:
        fig = go.Figure()
        for i, (lab, flag) in enumerate([("Weekday", False), ("Weekend", True)]):
            d = ph[ph["is_weekend"] == flag]
            fig.add_bar(x=d["peak_hour"], y=d["days"], name=lab, marker_color=SERIES[i], hovertemplate=f"{lab} %{{x}}:00 - %{{y}} days<extra></extra>")
        fig.update_layout(barmode="group", bargap=0.2)
        c4.plotly_chart(style(fig, 360, "Hour at which the daily peak occurs", xaxis_title="Hour"), width="stretch")
    c5, c6 = st.columns(2)
    dist = dc.table("peak", f"{pre}_distribution")
    if dist is not None:
        fig = go.Figure(go.Bar(x=(dist["bin_left"] + dist["bin_right"]) / 2 / sc, y=dist["count"], marker_color=BLUE,
                               hovertemplate="%{x:,.2f}: %{y} hours<extra></extra>"))
        for name, v in s["percentiles"].items():
            if name in ("p50.0", "p95.0", "p99.0"):
                fig.add_vline(x=v / sc, line=dict(color=AMBER, dash="dot", width=1), annotation_text=name.replace(".0", "").upper(),
                              annotation_font_color=TEXT2)
        fig.update_layout(bargap=0.05)
        c5.plotly_chart(style(fig, 340, f"Hourly load distribution ({unit})"), width="stretch")
    stl = dc.table("peak", f"{pre}_daily_peak_stl")
    if stl is not None:
        fig = go.Figure()
        fig.add_scatter(x=stl.index, y=stl["trend"] / sc, name="Trend", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=stl.index[-60:], y=stl["seasonal"].iloc[-60:] / sc, name="Weekly seasonal (last 60 days)",
                        line=dict(color=SERIES[1], width=1.5))
        c6.plotly_chart(style(fig, 340, "Seasonal decomposition of daily peaks (STL, period 7)"), width="stretch")
    st.markdown("#### High-demand events (hours above the P99 threshold, grouped into consecutive runs)")
    evs = dc.table("peak", f"{pre}_events")
    if evs is not None and len(evs):
        st.dataframe(evs.head(200), width="stretch", hide_index=True)
    if ev.get("shape_xi") is not None:
        callout(f"<b>Extreme-value analysis.</b> Generalised Pareto fit to {ev['n_clusters']} declustered daily-peak exceedances above "
                f"{fmt(ev['threshold'] / sc, 2)} {unit} (P95): shape ξ = {ev['shape_xi']:.2f} "
                f"({'bounded upper tail' if ev['shape_xi'] < 0 else 'heavy tail'}), KS p = {ev['ks_p']:.2f}. Return levels - "
                + ", ".join(f"{k}: {fmt(v / sc, 2)}" for k, v in ev["return_levels"].items()) + f" {unit}. {ev['caveat']}")
