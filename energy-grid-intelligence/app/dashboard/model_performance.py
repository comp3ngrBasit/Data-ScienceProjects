"""Page 10 - Model Performance: cross-experiment comparison, diagnostics, drift, tuning, explainability."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.forecasting import KEYS
from app.dashboard.theme import BLUE, SERIES, TEXT, callout, fmt, header, kpi_row, missing, style


def render() -> None:
    header("Model Performance", "Model comparison across experiments, residual diagnostics, drift monitoring and explainability")
    rows = []
    for key, label in KEYS.items():
        m, s = dc.table("forecasting", f"{key}_metrics"), dc.doc("forecasting", f"{key}_summary")
        if m is not None and s:
            for r in m.to_dict("records"):
                rows.append({"experiment": label, "key": key, **r, "best": r["model"] == s["best_model"]})
    if not rows:
        missing("Forecasting results")
        return
    allm = pd.DataFrame(rows)
    st.markdown("#### Skill versus seasonal-naive baseline (% WAPE reduction)")
    piv = allm.pivot_table(index="model", columns="experiment", values="skill_vs_naive_pct")
    piv = piv.reindex(columns=[v for v in KEYS.values() if v in piv.columns])
    fig = go.Figure(go.Heatmap(z=piv.values, x=[c.replace(" · ", "<br>") for c in piv.columns], y=piv.index, zmid=0,
                               colorscale=[[0, "#EF4444"], [0.5, "#383838"], [1, "#3B82F6"]], text=piv.round(1).values,
                               texttemplate="%{text}", textfont=dict(color=TEXT, size=11),
                               hovertemplate="%{y} · %{x}<br>skill %{z:.1f} %<extra></extra>"))
    st.plotly_chart(style(fig, 420, "Diverging: blue = better than baseline, red = worse"), width="stretch")
    best = allm[allm["best"]][["experiment", "model", "MAE", "RMSE", "WAPE", "R2", "PI_coverage", "skill_vs_naive_pct"]]
    st.dataframe(best.style.format(precision=3), width="stretch", hide_index=True)
    key = st.selectbox("Inspect experiment", list(KEYS), format_func=KEYS.get)
    s = dc.doc("forecasting", f"{key}_summary")
    p = dc.table("forecasting", f"{key}_predictions")
    if not s or p is None:
        missing("This experiment")
        return
    rd, dr = s["residual_diagnostics_best"], s["drift"]
    kpi_row([("Ljung-Box p (best model)", f"{rd['ljung_box_p']:.3g}", "", rd["interpretation"], BLUE),
             ("Durbin-Watson", fmt(rd["durbin_watson"], 2), "", f"residual mean {fmt(rd['mean'], 3)} · skew {fmt(rd['skew'], 2)}", SERIES[1]),
             ("Target drift (PSI)", fmt(dr["psi_target_test_vs_train"], 3), "", dr["psi_interpretation"], SERIES[2]),
             ("WAPE trend per origin", fmt(dr.get("best_model_wape_trend_per_fold"), 2) if dr.get("best_model_wape_trend_per_fold") is not None else "n/a",
              "pp", "positive = degrading over the test period", SERIES[3])])
    c1, c2 = st.columns(2)
    bd = p[p["model"] == s["best_model"]].sort_values("timestamp")
    res = bd["actual"] - bd["pred"]
    fig = go.Figure(go.Histogram(x=res, nbinsx=60, marker_color=BLUE, hovertemplate="%{x:.2f}: %{y}<extra></extra>"))
    c1.plotly_chart(style(fig, 320, f"Residual distribution - {s['best_model']}"), width="stretch")
    fig = go.Figure(go.Scatter(x=bd["pred"], y=res, mode="markers", marker=dict(size=8, color=BLUE, opacity=0.5),
                               hovertemplate="pred %{x:.2f}, residual %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#64748B"))
    c2.plotly_chart(style(fig, 320, "Residuals vs predictions", xaxis_title="Prediction", yaxis_title="Actual − predicted"), width="stretch")
    fe = dc.table("forecasting", f"{key}_error_by_fold")
    if fe is not None:
        fig = go.Figure()
        for i, mod in enumerate(fe["model"].unique()):
            d = fe[fe["model"] == mod]
            fig.add_scatter(x=d["origin"], y=d["WAPE"], name=mod, mode="lines+markers", marker=dict(size=8),
                            line=dict(color=SERIES[i % 6] if i < 6 else "#64748B", width=2 if mod == s["best_model"] else 1),
                            visible=True if mod in (s["best_model"], "SeasonalNaive") else "legendonly")
        st.plotly_chart(style(fig, 360, "Performance drift: WAPE at each walk-forward origin (%)"), width="stretch")
    st.markdown("#### Hyper-parameter optimisation and training details")
    info = s["model_info"]
    for mod, v in info.items():
        with st.expander(f"{mod} - {v.get('status')} · {v.get('seconds', '?')} s"):
            st.json({"selected": v.get("params"), "artifact": v.get("artifact")})
            if v.get("search"):
                st.dataframe(pd.DataFrame(v["search"]).astype(str), width="stretch", hide_index=True)
            if v.get("history"):
                h = pd.DataFrame(v["history"])
                fig = go.Figure()
                fig.add_scatter(x=h["epoch"], y=h["train_mse"], name="train MSE", line=dict(color=BLUE))
                fig.add_scatter(x=h["epoch"], y=h["val_mse"], name="validation MSE", line=dict(color=SERIES[1]))
                st.plotly_chart(style(fig, 260, "LSTM learning curve (early stopping on validation)"), width="stretch")
            if v.get("error"):
                st.error(v["error"])
    st.markdown("#### Explainability")
    ekey = st.selectbox("Model", ["household_hourly_h24", "grid_hourly_h24", "household_daily_h30"], format_func=KEYS.get)
    es = dc.doc("explainability", f"{ekey}_summary")
    if not es:
        missing("Explainability for this model")
        return
    for line in es["report"]:
        st.markdown(f"- {line}")
    c3, c4 = st.columns(2)
    imp = dc.table("explainability", f"{ekey}_shap_importance").head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.4g}<extra></extra>"))
    c3.plotly_chart(style(fig, 420, "SHAP importance (mean |SHAP|, XGBoost)"), width="stretch")
    pim = dc.table("explainability", f"{ekey}_permutation_importance").head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=pim["mae_increase"], y=pim["feature"], orientation="h", marker_color=SERIES[1],
                           error_x=dict(type="data", array=pim["std"], color="#94A3B8"), hovertemplate="%{y}: +%{x:.4g} MAE<extra></extra>"))
    c4.plotly_chart(style(fig, 420, "Permutation importance (MAE increase when shuffled)"), width="stretch")
    c5, c6 = st.columns(2)
    bee = dc.table("explainability", f"{ekey}_shap_beeswarm")
    fig = go.Figure()
    for f in bee["feature"].unique()[::-1]:
        d = bee[bee["feature"] == f]
        v = d["feature_value"]
        fig.add_scatter(x=d["shap"], y=[f] * len(d), mode="markers", showlegend=False,
                        marker=dict(size=6, color=(v - v.min()) / (v.max() - v.min() + 1e-12), colorscale=[[0, "#1A4C8C"], [1, "#F59E0B"]], opacity=0.6),
                        hovertemplate=f"{f}<br>SHAP %{{x:.3g}}<extra></extra>")
    fig.update_yaxes(type="category")
    c5.plotly_chart(style(fig, 380, "SHAP values per observation (colour: low → high feature value)"), width="stretch")
    pdp = dc.table("explainability", f"{ekey}_partial_dependence")
    feat = c6.selectbox("Partial dependence feature", list(pdp["feature"].unique()))
    d = pdp[pdp["feature"] == feat]
    fig = go.Figure(go.Scatter(x=d["value"], y=d["pdp"], line=dict(color=BLUE, width=2.4), mode="lines+markers"))
    c6.plotly_chart(style(fig, 330, f"Partial dependence: {feat}"), width="stretch")
    corr = dc.table("explainability", f"{ekey}_feature_correlation").set_index("feature")
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, zmid=0,
                               colorscale=[[0, "#3B82F6"], [0.5, "#383838"], [1, "#EF4444"]], hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    st.plotly_chart(style(fig, 460, "Feature correlation (Spearman) - top features and target"), width="stretch")
    callout("SHAP and permutation importance describe what the model relies on, not physical causes; correlated lag features share credit.")
