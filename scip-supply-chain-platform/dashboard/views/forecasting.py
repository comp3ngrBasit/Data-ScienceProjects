from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard import ui

COLORS = {"Moving Average": ui.SERIES[3], "Holt-Winters": ui.SERIES[4], "Prophet": ui.SERIES[6],
          "LightGBM": ui.SERIES[0], "XGBoost": ui.BRAND}


def render():
    ui.ribbon("Demand Forecasting", "7 / 30 / 60 / 90-day horizons · MA · Holt-Winters · Prophet · LightGBM · XGBoost")
    ui.origin_note("Trained on <b>real</b> UCI Online Retail daily demand (Dec 2010 – Dec 2011) with UK holidays and real M5 "
                   "calendar events. Time-based hold-out of the last 13 weeks (the full 90-day horizon). LightGBM/XGBoost are direct "
                   "multi-horizon models (no recursive feedback); champion = best average rank over MAE, RMSE, MAPE, WAPE and weekly WAPE.")
    models = api.df("/api/v1/forecasts/models")
    if models.empty:
        st.info("No forecast yet — run the bootstrap from Data & Models.")
        return
    m = pd.json_normalize(models["metrics"]).set_index(models["model_name"])
    m["champion"] = models.set_index("model_name")["is_champion"].map({True: "🏆", False: ""})
    c1, c2 = st.columns([1.3, 1])
    with c1, ui.panel("fc_tbl"):
        ui.title("Model comparison on hold-out (portfolio)")
        ui.table(m.reset_index()[["model_name", "champion", "mae", "rmse", "mape", "wape", "weekly_wape"]], column_config={
            "mae": st.column_config.NumberColumn("MAE", format="%.2f"), "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
            "mape": st.column_config.NumberColumn("MAPE (non-zero days)", format="percent"),
            "wape": st.column_config.NumberColumn("WAPE (daily)", format="percent"),
            "weekly_wape": st.column_config.NumberColumn("WAPE (weekly) — selection", format="percent")})
        st.caption(models["notes"].iat[0] if "notes" in models and models["notes"].iat[0] else "")
    with c2, ui.panel("fc_bar"):
        o = m.sort_values("weekly_wape")
        fig = go.Figure(go.Bar(x=o["weekly_wape"], y=o.index, orientation="h", text=[f"{v:.1%}" for v in o["weekly_wape"]],
                               textposition="outside", marker_color=[ui.BRAND if c else ui.BRAND_LIGHT for c in o["champion"]]))
        fig.update_layout(title="Weekly WAPE (lower is better) · dark = champion", height=260, xaxis_tickformat=".0%", yaxis_autorange="reversed",
                          xaxis_range=[0, o["weekly_wape"].max() * 1.25])
        ui.chart(fig, "fc_bar")
    if st.button("▶ Re-train forecasting models (background job)", key="fc_run"):
        try:
            j = api.post("/api/v1/forecasts/run")
            st.success(f"Job {j['job_id']} started — check Data & Models › Jobs.")
        except api.APIError as e:
            st.error(str(e))

    summ = api.df("/api/v1/forecasts/summary")
    with ui.panel("fc_sku"):
        ui.title("Product-level forecast")
        skus = summ.sort_values("forecast_30d", ascending=False)
        sku = st.selectbox("SKU", skus["sku"], format_func=lambda s: f"{s} · {skus.set_index('sku').at[s, 'name']}", key="fc_sku")
        d = api.cached(f"/api/v1/forecasts/{sku}")
        row = skus.set_index("sku").loc[sku]
        ui.kpi_row([ui.ikpi("📅", ui.num(row[f"forecast_{h}d"]), f"Next {h} days",
                            f"80% range {ui.num(row[f'lower_{h}d'])}–{ui.num(row[f'upper_{h}d'])}") for h in (7, 30, 60, 90)])
        show_all = st.toggle("Show all models", key="fc_all")
        h, f = pd.DataFrame(d["history"]), pd.DataFrame(d["forecast"])
        fig = go.Figure()
        fig.add_scatter(x=pd.to_datetime(h["date"]), y=h["quantity"], name="Actual", line=dict(color="#898781", width=1.4))
        for name, g in f.groupby("model"):
            champ = bool(g["is_champion"].iat[0])
            if not (champ or show_all):
                continue
            x = pd.to_datetime(g["date"])
            if champ:
                fig.add_scatter(x=list(x) + list(x[::-1]), y=list(g["yhat_upper"]) + list(g["yhat_lower"][::-1]), fill="toself",
                                fillcolor="rgba(47,158,138,.13)", line=dict(width=0), hoverinfo="skip", name="80% band")
            fig.add_scatter(x=x, y=g["yhat"], name=name + (" 🏆" if champ else ""), line=dict(color=COLORS.get(name), width=3 if champ else 1.6))
        fig.update_layout(height=380, hovermode="x unified", yaxis_title="Units / day")
        ui.chart(fig, "fc_sku")
        ui.table(pd.DataFrame(d["metrics"]), column_config={"mape": st.column_config.NumberColumn(format="percent"),
                                                              "wape": st.column_config.NumberColumn(format="percent")})
    with ui.panel("fc_all_tbl"):
        ui.title("Forecast table — all planning SKUs")
        ui.table(summ[["sku", "name", "category", "forecast_7d", "forecast_30d", "forecast_60d", "forecast_90d", "model"]], height=380,
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in ["forecast_7d", "forecast_30d", "forecast_60d", "forecast_90d"]})
        st.download_button("⬇ Download forecasts (CSV)", summ.to_csv(index=False).encode(), "forecasts.csv", key="fc_dl")
