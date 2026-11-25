"""Page 3 - Demand Forecasting."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, SERIES, TEXT, callout, fmt, header, kpi_row, missing, style

KEYS = {"household_hourly_h24": "Household · next 24 h (hourly)", "household_hourly_h168": "Household · next 7 days (hourly)",
        "household_daily_h30": "Household · next 30 days (daily)", "grid_hourly_h24": "German grid · next 24 h (hourly)",
        "grid_hourly_h168": "German grid · next 7 days (hourly)"}


def render() -> None:
    header("Demand Forecasting", "Walk-forward evaluation of eight models plus external benchmark; 90 % prediction intervals")
    key = st.selectbox("Experiment", list(KEYS), format_func=KEYS.get)
    s, m, p = dc.doc("forecasting", f"{key}_summary"), dc.table("forecasting", f"{key}_metrics"), dc.table("forecasting", f"{key}_predictions")
    if not s or m is None or p is None:
        missing("This forecasting experiment")
        return
    unit = s["unit"]
    best = m[m["model"] == s["best_model"]].iloc[0]
    kpi_row([("Best model (pooled WAPE)", s["best_model"], "", f"selected on test walk-forward, {len(s['origins'])} origins", BLUE),
             ("WAPE", fmt(best["WAPE"], 2), "%", f"MAE {fmt(best['MAE'], 3)} {unit}", SERIES[1]),
             ("Skill vs seasonal naive", fmt(best["skill_vs_naive_pct"], 1), "%", "reduction in WAPE", SERIES[2]),
             ("90 % interval coverage", fmt(best["PI_coverage"], 1), "%", f"mean width {fmt(best['PI_width'], 2)} {unit}", SERIES[3])])
    models = [x for x in m["model"] if x in set(p["model"])]
    c1, c2 = st.columns([3, 1])
    shown = c2.multiselect("Models", models, default=[s["best_model"], "SeasonalNaive"] if s["best_model"] != "SeasonalNaive" else [s["best_model"]])
    fold = c2.select_slider("Walk-forward origin", options=sorted(p["fold"].unique()), value=int(p["fold"].max()),
                            format_func=lambda i: str(s["origins"][i])[:16] if i < len(s["origins"]) else str(i))
    show_pi = c2.toggle("Show 90 % interval (first model)", True)
    d = p[p["fold"] == fold]
    fig = go.Figure()
    a = d[d["model"] == d["model"].iloc[0]]
    fig.add_scatter(x=a["timestamp"], y=a["actual"], name="Actual", line=dict(color=TEXT, width=2),
                    hovertemplate=f"%{{x|%d %b %H:%M}}<br>Actual %{{y:,.2f}} {unit}<extra></extra>")
    order = [x for x in models if x in shown]
    for i, mod in enumerate(order):
        dm = d[d["model"] == mod]
        col = SERIES[models.index(mod) % len(SERIES)]
        if show_pi and i == 0 and dm["lower"].notna().any():
            fig.add_scatter(x=list(dm["timestamp"]) + list(dm["timestamp"][::-1]), y=list(dm["upper"]) + list(dm["lower"][::-1]),
                            fill="toself", fillcolor="rgba(59,130,246,0.15)", line=dict(width=0), name=f"{mod} 90 % PI", hoverinfo="skip")
        fig.add_scatter(x=dm["timestamp"], y=dm["pred"], name=mod, line=dict(color=col, width=2, dash="solid" if i == 0 else "dot"),
                        hovertemplate=f"{mod} %{{y:,.2f}} {unit}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    c1.plotly_chart(style(fig, 430, f"Actual vs predicted ({unit})"), width="stretch")
    st.markdown("#### Model comparison (pooled over all walk-forward origins)")
    cols = ["model", "MAE", "RMSE", "MAPE", "WAPE", "R2", "Bias", "PI_coverage", "PI_width", "WAPE_fold_std", "skill_vs_naive_pct"]
    st.dataframe(m[cols].style.format({c: "{:,.3f}" for c in cols[1:]}).highlight_min(subset=["MAE", "RMSE", "WAPE"], color="#1E3A5F"),
                 width="stretch", hide_index=True)
    c3, c4 = st.columns(2)
    fig = go.Figure(go.Bar(x=m["WAPE"], y=m["model"], orientation="h",
                           marker_color=[BLUE if x == s["best_model"] else "#33415C" for x in m["model"]],
                           text=m["WAPE"].map(lambda v: f"{v:.2f} %"), textposition="outside", hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    c3.plotly_chart(style(fig, 360, "WAPE by model (lower is better)"), width="stretch")
    es = dc.table("forecasting", f"{key}_error_by_step")
    if es is not None:
        fig = go.Figure()
        for i, mod in enumerate([x for x in models if x in shown] or models[:3]):
            e = es[es["model"] == mod]
            fig.add_scatter(x=e["step"], y=e["ae"], name=mod, line=dict(color=SERIES[models.index(mod) % len(SERIES)], width=2),
                            hovertemplate=f"{mod} step %{{x}}: %{{y:,.3f}}<extra></extra>")
        c4.plotly_chart(style(fig, 360, f"Mean absolute error by lead time ({unit})", xaxis_title="Steps ahead"), width="stretch")
    fw = dc.table("forecasting", f"{key}_forward")
    fws = dc.doc("forecasting", f"{key}_forward")
    if fw is not None and fws:
        st.markdown("#### Forecast beyond the end of the data")
        fig = go.Figure()
        fig.add_scatter(x=list(fw["timestamp"]) + list(fw["timestamp"][::-1]), y=list(fw["upper"]) + list(fw["lower"][::-1]), fill="toself",
                        fillcolor="rgba(201,120,10,0.18)", line=dict(width=0), name="90 % interval", hoverinfo="skip")
        fig.add_scatter(x=fw["timestamp"], y=fw["pred"], name=fws["model"], line=dict(color=SERIES[1], width=2),
                        hovertemplate=f"%{{x|%d %b %H:%M}} %{{y:,.2f}} {unit}<extra></extra>")
        st.plotly_chart(style(fig, 320, f"{fws['model']} forecast issued at {fws['issued_at_data_end'][:16]} (no actuals exist for this period)"),
                        width="stretch")
    callout(f"<b>Validation design.</b> Train ≤ {s['train_end'][:10]}, validation ≤ {s['val_end'][:10]}, test to {s['test_end'][:10]}. "
            "Lag features are horizon-safe (every lag ≥ horizon), so no future information enters a prediction. "
            "MAPE excludes zero actuals and is inflated by small hourly values; WAPE is the selection metric."
            + (" <b>TSO day-ahead (ENTSO-E)</b> is the transmission operators' published forecast, an external benchmark. It is issued "
               "around midday of the previous day, so it uses less recent data than the midnight-origin models here, and the "
               "comparison covers only the walk-forward days shown: treat the ranking as indicative, not conclusive."
               if "TSO day-ahead (ENTSO-E)" in set(m["model"]) else ""))
    st.caption("Residual diagnostics, drift and hyper-parameter searches are on the Model Performance page.")
