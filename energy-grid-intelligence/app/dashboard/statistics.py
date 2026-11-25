"""Page 9 - Statistical Research: time-series statistics, Bayesian modelling, hypothesis tests."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, SERIES, TEXT, TEXT2, callout, fmt, header, kpi_row, missing, style


def _timeseries() -> None:
    ts = dc.doc("timeseries", "summary")
    if not ts:
        missing("Time-series statistics")
        return
    key = st.selectbox("Series", ["household_daily", "grid_daily", "household_hourly"],
                       format_func={"household_daily": "Household daily kWh", "grid_daily": "German grid daily MWh",
                                    "household_hourly": "Household hourly kWh (last 120 days)"}.get)
    s = ts[key]
    lv = s["stationarity"]["level"]
    kpi_row([("Trend strength (STL)", fmt(s["trend_strength"], 2), "", f"STL period {s['stl_period']}", BLUE),
             ("Seasonal strength (STL)", fmt(s["seasonal_strength"], 2), "", "0 = none, 1 = dominant", SERIES[1]),
             ("ADF p-value (level)", f"{lv['adf_p']:.3g}", "", f"statistic {lv['adf_stat']:.2f} vs 5 % crit {lv['adf_crit_5']:.2f}", SERIES[2]),
             ("KPSS p-value (level)", f"{lv['kpss_p']:.3g}", "", "bounded to [0.01, 0.10]", SERIES[3])])
    st.dataframe([{"series": v["series"], "ADF stat": round(v["adf_stat"], 3), "ADF p": f"{v['adf_p']:.3g}", "KPSS stat": round(v["kpss_stat"], 3),
                   "KPSS p": f"{v['kpss_p']:.3g}", "verdict": v["verdict"]} for v in s["stationarity"].values()],
                 width="stretch", hide_index=True)
    stl = dc.table("timeseries", f"{key}_stl")
    sp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    for i, comp in enumerate(["observed", "trend", "seasonal", "resid"]):
        sp.add_scatter(x=stl.index, y=stl[comp], line=dict(color=[TEXT2, BLUE, SERIES[1], "#64748B"][i], width=1.2), row=i + 1, col=1, showlegend=False)
    st.plotly_chart(style(sp, 560, "Seasonal decomposition (STL, robust)"), width="stretch")
    c1, c2 = st.columns(2)
    acf = dc.table("timeseries", f"{key}_acf")
    for col, name, c in (("acf", "Autocorrelation (ACF)", c1), ("pacf", "Partial autocorrelation (PACF)", c2)):
        d = acf.dropna(subset=[col]).iloc[1:]
        fig = go.Figure(go.Bar(x=d["lag"], y=d[col], marker_color=BLUE, hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"))
        fig.add_hrect(y0=-d["conf"].iloc[0], y1=d["conf"].iloc[0], fillcolor="#94A3B8", opacity=0.12, line_width=0)
        c.plotly_chart(style(fig, 300, name + " with 95 % band", xaxis_title="Lag"), width="stretch")
    if key != "household_hourly":
        roll = dc.table("timeseries", f"{key}_rolling")
        cp = dc.table("timeseries", f"{key}_changepoints")
        fig = go.Figure()
        fig.add_scatter(x=roll.index, y=roll["value"], name="Value", line=dict(color="#33415C", width=1))
        fig.add_scatter(x=roll.index, y=roll["roll_mean"], name="Rolling mean (28 d)", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=roll.index, y=roll["roll_std"], name="Rolling std (28 d)", line=dict(color=SERIES[1], width=1.5))
        for _, r in cp.iloc[1:].iterrows():
            fig.add_vline(x=r["start"], line=dict(color="#F59E0B", dash="dash", width=1.4))
        st.plotly_chart(style(fig, 360, f"Rolling statistics and PELT change points ({len(cp) - 1} detected, dashed)"), width="stretch")
        st.dataframe(cp, width="stretch", hide_index=True)
        tr = s["trend"]
        callout(f"Linear trend slope {tr['slope_per_step']:.4g} per day (R² {tr['r2']:.3f}); Kendall τ = {tr['kendall_tau']:.3f} "
                f"(p = {tr['kendall_p']:.3g}). Trend p-values assume independence and are descriptive given serial correlation.")


def _bayesian() -> None:
    w = dc.doc("weather", "summary") or {}
    b = w.get("models", {}).get("BayesianRegression", {})
    summ = dc.table("bayesian", "posterior_summary")
    if not b or summ is None:
        missing("Bayesian regression")
        return
    kpi_row([("Max R-hat", fmt(b["max_r_hat"], 3), "", "< 1.01 indicates convergence", BLUE),
             ("Min bulk ESS", fmt(b["min_ess_bulk"], 0), "", f"{b['chains']} chains × {b['draws']} draws", SERIES[1]),
             ("Divergences", str(b["divergences"]), "", "NUTS sampler", SERIES[2]),
             ("90 % interval coverage", fmt(b["coverage_90"], 1), "%", f"50 % interval: {fmt(b['coverage_50'], 1)} %", SERIES[3])])
    with st.expander("Model specification"):
        st.code(b["spec"], language="text")
    coef = dc.table("bayesian", "coefficients")
    c1, c2 = st.columns(2)
    d = coef[~coef["feature"].str.startswith("dow_")].iloc[::-1]
    fig = go.Figure(go.Scatter(x=d["mean_gwh_per_unit"], y=d["feature"], mode="markers", marker=dict(size=10, color=BLUE),
                               error_x=dict(type="data", symmetric=False, array=d["ci95_high"] - d["mean_gwh_per_unit"],
                                            arrayminus=d["mean_gwh_per_unit"] - d["ci95_low"], color=TEXT2),
                               hovertemplate="%{y}: %{x:.2f} GWh per unit<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#64748B", width=1))
    c1.plotly_chart(style(fig, 360, "Posterior means with 95 % credible intervals (GWh per unit)"), width="stretch")
    samples = dc.table("bayesian", "prior_posterior_samples")
    feat = c2.selectbox("Parameter: prior vs posterior", list(samples["feature"].unique()))
    sm = samples[samples["feature"] == feat]
    fig = go.Figure()
    post = sm[sm["distribution"] == "posterior"]["value"]
    lo, hi = post.quantile(0.001), post.quantile(0.999)
    span = hi - lo
    for i, dist in enumerate(["prior", "posterior"]):
        v = sm[sm["distribution"] == dist]["value"]
        v = v[(v > lo - 4 * span) & (v < hi + 4 * span)]
        fig.add_histogram(x=v, name=dist.title(), histnorm="probability density", nbinsx=80, marker_color=[SERIES[1], BLUE][i],
                          opacity=0.7 if i else 0.45)
    fig.update_layout(barmode="overlay")
    c2.plotly_chart(style(fig, 300, f"{feat}: prior vs posterior (prior clipped to the posterior's neighbourhood)"), width="stretch")
    fc = dc.table("bayesian", "posterior_predictive_test")
    fig = go.Figure()
    fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc["hi90"]) + list(fc["lo90"][::-1]), fill="toself",
                    fillcolor="rgba(59,130,246,0.14)", line=dict(width=0), name="90 % credible", hoverinfo="skip")
    fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc["hi50"]) + list(fc["lo50"][::-1]), fill="toself",
                    fillcolor="rgba(59,130,246,0.28)", line=dict(width=0), name="50 % credible", hoverinfo="skip")
    fig.add_scatter(x=fc.index, y=fc["actual"], name="Actual", line=dict(color=TEXT, width=1.6))
    fig.add_scatter(x=fc.index, y=fc["mean"], name="Posterior mean", line=dict(color=BLUE, width=1.6, dash="dot"))
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 380, "Probabilistic demand forecast on the hold-out block (GWh/day)"), width="stretch")
    st.dataframe(summ.style.format(precision=3), width="stretch", hide_index=True)


def _tests() -> None:
    t = dc.table("statistics", "hypothesis_tests")
    s = dc.doc("statistics", "summary")
    if t is None:
        missing("Hypothesis tests")
        return
    kpi_row([("Tests", str(s["n_tests"]), "", s["correction"], BLUE),
             ("Significant after correction", str(s["n_significant"]), "", "Holm-adjusted p < 0.05", SERIES[1])])
    callout("Each test reports the classical statistic, an effect size, and a 95 % CI for the mean difference from a moving-block "
            "bootstrap (weeks or months), which respects serial dependence that classical tests ignore.")
    fig = go.Figure()
    tt = t.dropna(subset=["diff_pct"]).iloc[::-1]
    fig.add_bar(x=tt["effect_size_cohens_d"], y=tt["hypothesis"].str.split(":").str[0], orientation="h",
                marker_color=[BLUE if s_ else "#475569" for s_ in tt["significant_holm_0.05"]],
                customdata=tt[["hypothesis", "p_value_holm"]], hovertemplate="%{customdata[0]}<br>d = %{x:.2f}<br>Holm p = %{customdata[1]:.3g}<extra></extra>")
    fig.add_vline(x=0, line=dict(color="#64748B", width=1))
    st.plotly_chart(style(fig, 320, "Effect sizes (Cohen's d); grey = not significant after Holm"), width="stretch")
    for r in t.to_dict("records"):
        with st.expander(r["hypothesis"]):
            if r.get("p_value") is None or (isinstance(r.get("p_value"), float) and np.isnan(r["p_value"])):
                st.write(r.get("interpretation"))
                continue
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Mean - {r['group_a']}", f"{r['mean_a']:,.3g} {r['unit']}")
            c2.metric(f"Mean - {r['group_b']}", f"{r['mean_b']:,.3g} {r['unit']}", f"{r['diff_pct']:+.1f} % (A vs B)", delta_color="off")
            c3.metric("Holm-adjusted p", f"{r['p_value_holm']:.3g}")
            st.dataframe([{"test": r["primary_test"], "statistic": f"{r['statistic']:,.4g}", "raw p": f"{r['p_value']:.3g}",
                           "Welch t": f"{r['welch_t']:.3f}", "Welch p": f"{r['welch_p']:.3g}", "Cohen's d": f"{r['effect_size_cohens_d']:.3f}",
                           "rank-biserial": f"{r['effect_size_rank_biserial']:.3f}",
                           "95 % CI of difference": f"[{r['ci95_diff_low']:,.4g}, {r['ci95_diff_high']:,.4g}] {r['unit']}",
                           "n": f"{r['n_a']} / {r['n_b']}"}], width="stretch", hide_index=True)
            st.markdown(f"**Interpretation.** {r['interpretation']}")
            st.caption(f"CI method: {r['ci_method']}. {r.get('context') or ''}")


def render() -> None:
    header("Statistical Research", "Time-series diagnostics, Bayesian regression and hypothesis testing")
    t1, t2, t3 = st.tabs(["Time-series analysis", "Bayesian modelling", "Hypothesis testing"])
    with t1:
        _timeseries()
    with t2:
        _bayesian()
    with t3:
        _tests()
