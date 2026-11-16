from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard import ui


def _flow_map(res: dict, scope: str) -> go.Figure:
    fl = pd.DataFrame(res["flows"])
    fig = go.Figure()
    mx = fl["units"].max()
    for r in fl.itertuples():
        fig.add_trace(go.Scattergeo(lon=[r.wh_lng, r.dest_lng], lat=[r.wh_lat, r.dest_lat], mode="lines",
                                    line=dict(width=1 + 6 * r.units / mx, color="rgba(47,158,138,.55)"), hoverinfo="text",
                                    text=f"{r.warehouse_name} → {r.destination_name}: {r.units:,.0f} units, {r.distance_km:,.0f} km",
                                    showlegend=False))
    wh = fl.drop_duplicates("warehouse")
    dest = fl.drop_duplicates("destination")
    fig.add_trace(go.Scattergeo(lon=dest["dest_lng"], lat=dest["dest_lat"], mode="markers", name="Demand points",
                                marker=dict(size=7, color=ui.SERIES[0]), text=dest["destination_name"], hoverinfo="text"))
    fig.add_trace(go.Scattergeo(lon=wh["wh_lng"], lat=wh["wh_lat"], mode="markers+text", name="Warehouses",
                                marker=dict(size=15, color=ui.SERIES[1], symbol="square"), text=wh["warehouse"], textposition="top center"))
    geo = dict(scope=scope, showcountries=True, countrycolor="#cfd8d5", landcolor="#f3f7f5", showland=True)
    if scope == "europe":
        geo.update(lonaxis_range=[-25, 45], lataxis_range=[30, 70])
    fig.update_layout(geo=geo, height=470, margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0, x=0))
    return fig


@st.cache_data(ttl=300, show_spinner="Solving with OR-Tools…")
def _solve(path: str, body: dict | None = None, **params):
    return api.post(path, body, **params)


def render():
    ui.ribbon("Logistics Optimization", "Warehouse allocation (OR-Tools LP) · delivery routing (OR-Tools VRP)")
    ui.origin_note("EU demand = <b>real</b> UCI customer-country demand (last 90 days); Brazil demand = <b>real</b> Olist customer "
                   "counts per state. Warehouses, capacities and transport rates are <b>simulated</b> assumptions.")
    t1, t2 = st.tabs(["Warehouse allocation", "Delivery routing"])
    with t1:
        net = st.radio("Network", ["EU", "BR"], horizontal=True, format_func=lambda n: {"EU": "Europe (UCI)", "BR": "Brazil (Olist)"}[n], key="lg_net")
        try:
            res = _solve("/api/v1/logistics/allocation", network=net)
        except api.APIError as e:
            st.error(str(e))
            return
        ui.kpi_row([ui.ikpi("💶", ui.money(res["objective_cost"]), "Optimised transport cost"),
                    ui.ikpi("🏭", ui.money(res["baseline_single_dc_cost"]), "Baseline: single largest DC"),
                    ui.ikpi("📉", ui.pct(res["saving_pct"]), "Saving vs baseline"),
                    ui.ikpi("🔁", str(len(res["flows"])), "Active lanes")])
        c1, c2 = st.columns([1.6, 1])
        with c1, ui.panel("lg_map"):
            ui.chart(_flow_map(res, "europe" if net == "EU" else "south america"), f"lg_map_{net}")
        with c2, ui.panel("lg_util"):
            u = pd.Series(res["utilisation"]).sort_values()
            fig = go.Figure(go.Bar(x=u.values, y=u.index, orientation="h", marker_color=ui.BRAND, text=[f"{v:.0%}" for v in u.values],
                                   textposition="outside"))
            fig.update_layout(title="Warehouse capacity utilisation", height=230, xaxis_tickformat=".0%", xaxis_range=[0, 1.25])
            ui.chart(fig, f"lg_util_{net}")
            ui.table(pd.DataFrame(res["flows"])[["warehouse", "destination_name", "units", "distance_km", "cost"]], height=220)
        st.caption(f"Demand basis: {res['demand_basis']}. Cost = units × (distance × 0.0004 + 0.05 handling).")
    with t2:
        v = st.slider("Vehicles", 2, 10, 5, key="lg_veh")
        try:
            res = _solve("/api/v1/logistics/routing", {"hub": "BR-HUB-SP", "vehicles": v})
        except api.APIError as e:
            st.error(str(e))
            return
        ui.kpi_row([ui.ikpi("🚛", str(res["vehicles_used"]), "Vehicles used", f"capacity {res['capacity']} pallets each"),
                    ui.ikpi("🛣️", f"{res['total_distance_km']:,.0f} km", "Total distance"),
                    ui.ikpi("📍", str(sum(len(r['stops']) for r in res['routes'])), "Stops (state capitals)")])
        fig = go.Figure()
        for i, r in enumerate(res["routes"]):
            lat = [res["hub_lat"]] + [s["lat"] for s in r["stops"]] + [res["hub_lat"]]
            lng = [res["hub_lng"]] + [s["lng"] for s in r["stops"]] + [res["hub_lng"]]
            fig.add_trace(go.Scattergeo(lat=lat, lon=lng, mode="lines+markers", name=f"Vehicle {r['vehicle']} · {r['distance_km']:,.0f} km",
                                        line=dict(width=2.5, color=ui.SERIES[i % len(ui.SERIES)]), marker=dict(size=7),
                                        text=["Hub"] + [s["stop"] for s in r["stops"]] + ["Hub"], hoverinfo="text"))
        fig.add_trace(go.Scattergeo(lat=[res["hub_lat"]], lon=[res["hub_lng"]], mode="markers", name="Hub São Paulo",
                                    marker=dict(size=16, symbol="square", color="#0b0b0b")))
        fig.update_layout(geo=dict(scope="south america", showcountries=True, landcolor="#f3f7f5", showland=True),
                          height=520, margin=dict(l=0, r=0, t=0, b=0))
        with ui.panel("lg_vrp"):
            ui.chart(fig, f"lg_vrp_{v}")
            st.caption(res["demand_basis"])
