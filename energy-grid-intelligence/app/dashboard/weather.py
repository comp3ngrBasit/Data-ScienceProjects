"""Page 8 - Weather Intelligence."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, SERIES, TEXT, callout, fmt, header, kpi_row, missing, style


def render() -> None:
    header("Weather Intelligence", "Daily German load versus population-weighted weather: correlation, regression, GAM, ML and SHAP")
    s = dc.doc("weather", "summary")
    if not s:
        missing("Weather sensitivity")
        return
    src = s["weather_source"]
    callout(f"<b>Data.</b> {src.get('source')} · {src.get('n_days')} days ({src['period'][0]} to {src['period'][1]}). "
            f"Variables: {', '.join(s['variables_available'])}. "
            + (f"<b>Not available:</b> {', '.join(s['variables_missing'])} ({src.get('note', '')}). " if s["variables_missing"] else "")
            + f"<b>{s['causality_note']}</b>")
    e, wv = s["hdd_effect_gwh_per_degree_day"], s["weather_explained_variance"]
    kpi_row([("Heating sensitivity", fmt(e["coef"], 2), "GWh / HDD", f"95 % CI {fmt(e['ci95'][0], 2)}-{fmt(e['ci95'][1], 2)} · HDD base {s['hdd_base_c']} °C", BLUE),
             ("Variance explained by weather", fmt((wv["calendar_plus_weather_r2"] - wv["calendar_only_r2"]) * 100, 1), "pp R²",
              f"calendar-only R² {fmt(wv['calendar_only_r2'], 3)} → {fmt(wv['calendar_plus_weather_r2'], 3)}", SERIES[1]),
             ("Weather-related variability", fmt(wv["residual_std_calendar_only_gwh"] - wv["residual_std_with_weather_gwh"], 1), "GWh",
              "reduction in residual std when weather is added", SERIES[2]),
             ("Best test model", s["best_model_test"], "", f"test {s['test_period'][0]} to {s['test_period'][1]}", SERIES[3])])
    daily = dc.table("weather", "daily")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for i, (lab, flag) in enumerate([("Working days", 0), ("Weekends & holidays", 1)]):
        d = daily[daily["is_offday"] == flag]
        fig.add_scatter(x=d["temperature"], y=d["load_gwh"], mode="markers", name=lab,
                        marker=dict(size=8, color=SERIES[i], opacity=0.7, symbol=["circle", "diamond"][i], line=dict(width=1, color="#151F32")),
                        customdata=d.index.strftime("%d %b %Y"), hovertemplate="%{customdata}<br>%{x:.1f} °C · %{y:,.0f} GWh<extra></extra>")
    c1.plotly_chart(style(fig, 400, "Temperature vs daily load", xaxis_title="Daily mean temperature (°C)", yaxis_title="GWh"),
                    width="stretch")
    gam = dc.table("weather", "gam_temperature_curve")
    fig = go.Figure()
    fig.add_scatter(x=list(gam["temperature"]) + list(gam["temperature"][::-1]), y=list(gam["ci_high"]) + list(gam["ci_low"][::-1]),
                    fill="toself", fillcolor="rgba(59,130,246,0.18)", line=dict(width=0), name="95 % band", hoverinfo="skip")
    fig.add_scatter(x=gam["temperature"], y=gam["effect_gwh"], line=dict(color=BLUE, width=2.4), name="GAM smooth",
                    hovertemplate="%{x:.1f} °C: %{y:+.1f} GWh<extra></extra>")
    c2.plotly_chart(style(fig, 400, "Nonlinear temperature effect (GAM partial effect, GWh)", xaxis_title="°C"), width="stretch")
    c3, c4 = st.columns(2)
    corr = dc.table("weather", "correlations")
    piv = corr.pivot(index="variable", columns="sample", values="pearson_r")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, zmin=-1, zmax=1, zmid=0,
                               colorscale=[[0, "#3B82F6"], [0.5, "#383838"], [1, "#EF4444"]], text=piv.round(2).values,
                               texttemplate="%{text}", textfont=dict(color=TEXT), hovertemplate="%{y} (%{x}): r = %{z:.3f}<extra></extra>"))
    c3.plotly_chart(style(fig, 330, "Pearson correlation with daily load"), width="stretch")
    with c4:
        st.markdown("##### Pearson & Spearman with 95 % CI")
        st.dataframe(corr.style.format({c: "{:.3f}" for c in corr.columns if corr[c].dtype.kind == "f"}), width="stretch",
                     hide_index=True, height=300)
    seas = dc.table("weather", "seasonal_sensitivity")
    fig = go.Figure(go.Bar(x=seas["season"], y=seas["slope_gwh_per_c"], marker_color=SERIES[:4],
                           error_y=dict(type="data", symmetric=False, array=seas["slope_ci95_high"] - seas["slope_gwh_per_c"],
                                        arrayminus=seas["slope_gwh_per_c"] - seas["slope_ci95_low"], color="#94A3B8"),
                           hovertemplate="%{x}: %{y:.2f} GWh/°C<extra></extra>"))
    c5, c6 = st.columns(2)
    c5.plotly_chart(style(fig, 330, "Seasonal weather sensitivity: working-day load slope (GWh per °C, 95 % CI)"), width="stretch")
    m = dc.table("weather", "model_metrics")
    with c6:
        st.markdown("##### Model comparison (chronological hold-out)")
        st.dataframe(m.style.format({c: "{:.3f}" for c in m.columns if m[c].dtype.kind == "f"}), width="stretch", hide_index=True)
    st.markdown("#### Multiple linear regression (Newey-West HAC standard errors)")
    st.dataframe(dc.table("weather", "ols_coefficients").style.format(precision=3), width="stretch", hide_index=True)
    c7, c8 = st.columns(2)
    sh = dc.table("weather", "shap_importance").sort_values("mean_abs_shap_gwh")
    fig = go.Figure(go.Bar(x=sh["mean_abs_shap_gwh"], y=sh["feature"], orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.1f} GWh<extra></extra>"))
    c7.plotly_chart(style(fig, 360, "SHAP feature importance (XGBoost, mean |SHAP|, GWh)"), width="stretch")
    pdp = dc.table("weather", "partial_dependence")
    fig = go.Figure()
    for i, f in enumerate(pdp["feature"].unique()):
        d = pdp[pdp["feature"] == f]
        fig.add_scatter(x=(d["value"] - d["value"].min()) / (d["value"].max() - d["value"].min()), y=d["pdp_gwh"], name=f,
                        line=dict(color=SERIES[i], width=2), customdata=d["value"], hovertemplate=f"{f} = %{{customdata:.1f}}: %{{y:,.0f}} GWh<extra></extra>")
    c8.plotly_chart(style(fig, 360, "Partial dependence (XGBoost); x scaled to each feature's range", xaxis_title="Relative feature value"),
                    width="stretch")
    hw = s.get("household", {})
    st.markdown("#### Household weather sensitivity (Sceaux, France)")
    if hw.get("status") == "ok":
        kpi_row([("Heating sensitivity", fmt(hw["hdd_kwh_per_degree_day"], 2), "kWh / HDD", f"p = {hw['hdd_p']:.3g}", BLUE),
                 ("R² with weather", fmt(hw["r2"], 3), "", f"calendar only {fmt(hw['calendar_only_r2'], 3)}", SERIES[1]),
                 ("Days", str(hw["n_days"]), "", hw.get("source", ""), SERIES[2])])
        hc = dc.table("weather", "household_correlations")
        if hc is not None:
            st.dataframe(hc.style.format(precision=3), width="stretch", hide_index=True)
        hcoef = dc.table("weather", "household_ols_coefficients")
        if hcoef is not None:
            st.dataframe(hcoef.style.format(precision=3), width="stretch", hide_index=True)
    else:
        callout(f"Not computed in this build: {hw.get('reason', 'Open-Meteo data for the household location was unavailable')}. "
                "It runs automatically when the Open-Meteo archive API is reachable.", "#F59E0B")
    tp = dc.table("weather", "test_predictions")
    fig = go.Figure()
    fig.add_scatter(x=tp.index, y=tp["actual"], name="Actual", line=dict(color=TEXT, width=2))
    for i, c in enumerate([c for c in tp.columns if c != "actual"]):
        fig.add_scatter(x=tp.index, y=tp[c], name=c, line=dict(color=SERIES[i % 6], width=1.4), visible=True if c == s["best_model_test"] else "legendonly")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 360, "Hold-out period: actual vs weather-model predictions (GWh/day)"), width="stretch")
