"""Page 7 - Renewable Energy Analytics."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, GREEN, SERIES, TEXT, callout, fmt, header, kpi_row, missing, style

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]


def render() -> None:
    header("Renewable Energy Analytics", "Wind and solar contribution to German load, variability, surplus and shortage periods")
    s = dc.doc("renewable", "summary")
    if not s:
        missing("Renewable analysis")
        return
    k = s["kpis"]
    kpi_row([("Total wind + solar", fmt(k["total_vre_twh"], 0), "TWh", f"{k['period'][0][:10]} to {k['period'][1][:10]}", GREEN),
             ("VRE penetration", fmt(k["vre_penetration_pct"], 1), "%", "VRE energy / load energy", GREEN),
             ("Wind share of VRE", fmt(k["wind_share_of_vre_pct"], 1), "%", f"{fmt(k['wind_share_of_load_pct'], 1)} % of load", SERIES[2]),
             ("Solar share of VRE", fmt(k["solar_share_of_vre_pct"], 1), "%", f"{fmt(k['solar_share_of_load_pct'], 1)} % of load", SERIES[1])])
    kpi_row([("Demand-to-VRE ratio", fmt(k["demand_to_vre_ratio"], 2), "x", "load energy / VRE energy", BLUE),
             ("VRE variability", fmt(k["vre_hourly_cv"] * 100, 1), "% CV", "hourly VRE output", SERIES[3]),
             ("Max hourly VRE share", fmt(k["max_hourly_vre_share_pct"], 1), "%", k["max_hourly_vre_share_time"][:16], SERIES[4]),
             ("Low-VRE periods", str(k["n_low_vre_periods_24h"]), "", f"longest {k['longest_low_vre_period_h']} h · {k['low_vre_definition']}", "#F59E0B")])
    callout("<b>Definitions.</b> VRE = wind onshore + offshore + solar PV (OPSD actual generation). Penetration uses ENTSO-E actual "
            "load over identical hours as the denominator. OPSD has no hydro, biomass or conventional series, so the remainder is shown "
            "as <b>residual load</b> (load − VRE: covered by dispatchable plants, other renewables, storage and net imports), not as "
            "non-renewable generation.")
    daily = dc.table("renewable", "daily")
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily["load_roll30"], name="Load (30-day mean)", line=dict(color=TEXT, width=2))
    fig.add_scatter(x=daily.index, y=daily["wind_gwh"].rolling(30, center=True).mean(), name="Wind (30-day mean)",
                    stackgroup="vre", line=dict(color=SERIES[2], width=0.5))
    fig.add_scatter(x=daily.index, y=daily["solar_gwh"].rolling(30, center=True).mean(), name="Solar (30-day mean)",
                    stackgroup="vre", line=dict(color=SERIES[1], width=0.5))
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 400, "Demand vs renewable generation (GWh/day, 30-day rolling means)"), width="stretch")
    c1, c2 = st.columns(2)
    mon = dc.table("renewable", "monthly")
    fig = go.Figure(go.Bar(x=mon.index, y=mon["vre_penetration_pct"], marker_color=GREEN,
                           hovertemplate="%{x|%b %Y}: %{y:.1f} % of load<extra></extra>"))
    c1.plotly_chart(style(fig, 340, "Monthly VRE penetration (% of load)"), width="stretch")
    di = dc.table("renewable", "diurnal")
    sname = c2.selectbox("Season", SEASONS, index=2)
    d = di[di["season"] == sname]
    fig = go.Figure()
    fig.add_scatter(x=d["hour"], y=d["load_mw"] / 1e3, name="Load", line=dict(color=TEXT, width=2))
    fig.add_scatter(x=d["hour"], y=d["wind_mw"] / 1e3, name="Wind", line=dict(color=SERIES[2], width=2))
    fig.add_scatter(x=d["hour"], y=d["solar_mw"] / 1e3, name="Solar", line=dict(color=SERIES[1], width=2))
    fig.update_layout(hovermode="x unified")
    c2.plotly_chart(style(fig, 300, f"Mean diurnal profile, {sname} (GW)", xaxis_title="Hour (local)"), width="stretch")
    c3, c4 = st.columns(2)
    stl = dc.table("renewable", "vre_stl_weekly")
    fig = go.Figure()
    fig.add_scatter(x=stl.index, y=stl["observed"], name="Weekly mean VRE", line=dict(color="#33415C", width=1))
    fig.add_scatter(x=stl.index, y=stl["trend"], name="Trend", line=dict(color=GREEN, width=2.2))
    c3.plotly_chart(style(fig, 330, "VRE time-series decomposition: trend (GWh/day)"), width="stretch")
    fig = go.Figure(go.Scatter(x=stl.index, y=stl["seasonal"], line=dict(color=SERIES[3], width=1.8), name="Seasonal"))
    c4.plotly_chart(style(fig, 330, "VRE annual seasonal component (GWh/day)"), width="stretch")
    c5, c6 = st.columns(2)
    corr = s["correlation_daily_load_vs_generation"]
    names = {"solar_gwh": "Solar", "wind_gwh": "Wind", "vre_gwh": "Wind + solar"}
    fig = go.Figure(go.Bar(x=[names[c] for c in corr], y=[v["pearson_r"] for v in corr.values()], marker_color=SERIES[:3],
                           text=[f"r = {v['pearson_r']:.2f}<br>ρ = {v['spearman_rho']:.2f}" for v in corr.values()], textposition="outside",
                           hovertemplate="%{x}: r = %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[-0.6, 0.6], zeroline=True)
    c5.plotly_chart(style(fig, 330, "Correlation of daily load with daily generation"), width="stretch")
    reg = s["residual_load_regression"]
    with c6:
        st.markdown("##### Residual-load regression")
        st.caption(reg["model"] + f" · R² = {reg['r2']:.3f}")
        st.dataframe([{"term": t, "coef": round(v["coef"], 3), "SE (HAC)": round(v["se"], 3), "p": f"{v['p']:.2g}",
                       "95 % CI": f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"} for t, v in reg["coefficients"].items()],
                     width="stretch", hide_index=True)
        st.caption("A wind coefficient near −1 means each extra GWh of wind displaces ~1 GWh of residual load on the same day.")
    ann = dc.table("renewable", "annual")
    st.markdown("#### Annual summary")
    st.dataframe(ann.style.format(precision=2), width="stretch")
    c7, c8 = st.columns(2)
    eu = dc.table("renewable", "europe_2016_2019")
    if eu is not None:
        eu = eu.sort_values("vre_penetration_pct")
        fig = go.Figure()
        fig.add_bar(y=eu["country"], x=eu["wind_share_of_load_pct"], name="Wind", orientation="h", marker_color=SERIES[2])
        fig.add_bar(y=eu["country"], x=eu["solar_share_of_load_pct"], name="Solar", orientation="h", marker_color=SERIES[1])
        fig.update_layout(barmode="stack", bargap=0.25)
        c7.plotly_chart(style(fig, 460, "Regional comparison 2016-2019: VRE % of load"), width="stretch")
    low = dc.table("renewable", "low_vre_periods")
    with c8:
        st.markdown("##### Renewable shortage periods (VRE < 10 % of load ≥ 24 h)")
        st.dataframe(low.sort_values("duration_h", ascending=False) if low is not None and len(low) else low,
                     width="stretch", hide_index=True, height=200)
        st.markdown(f"##### High-VRE hours (≥ 60 % of load): {k['n_high_vre_hours_60pct']:,}")
        st.caption(f"Hours with VRE ≥ load (absolute surplus): {k['hours_vre_exceeds_load']}. Ramp P1/P99 of hourly VRE change: "
                   f"{fmt(s['ramps_mw_per_h']['vre_mw']['p1'] / 1e3, 1)} / {fmt(s['ramps_mw_per_h']['vre_mw']['p99'] / 1e3, 1)} GW/h.")
