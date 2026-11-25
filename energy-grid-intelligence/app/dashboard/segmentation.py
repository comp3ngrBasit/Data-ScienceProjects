"""Page 5 - Energy Segmentation."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard import data_client as dc
from app.dashboard.theme import BLUE, SERIES, callout, fmt, header, kpi_row, missing, style

SYMBOLS = ["circle", "diamond", "square", "triangle-up", "x", "star", "cross", "hexagon"]


def render() -> None:
    header("Energy Segmentation", "Unsupervised discovery of consumption behaviours: household day-profiles and national grids")
    s = dc.doc("segmentation", "summary")
    if not s:
        missing("Segmentation")
        return
    tab1, tab2 = st.tabs(["Household day-profiles", "European grids (regional)"])
    with tab1:
        h = s["household"]
        days, prof = dc.table("segmentation", "household_days"), dc.table("segmentation", "household_profiles")
        comp = dc.table("segmentation", "household_algorithm_comparison")
        km = comp[comp["algorithm"] == "KMeans"].iloc[0]
        kpi_row([("Observations", fmt(h["n_days"], 0), "days", h["unit_of_analysis"], BLUE),
                 ("Clusters (K-Means)", str(h["chosen_k"]), "", f"GMM by BIC: {h['gmm_k_by_bic']}", SERIES[1]),
                 ("Silhouette", fmt(km["silhouette"], 3), "", f"Davies-Bouldin {fmt(km['davies_bouldin'], 2)} · CH {fmt(km['calinski_harabasz'], 0)}", SERIES[2]),
                 ("Bootstrap stability", fmt(h["stability"]["mean_ari"], 2), "ARI", f"± {fmt(h['stability']['std_ari'], 2)} over {h['stability']['n_bootstrap']} resamples", SERIES[3])])
        callout("The UCI data cover a single household, so households cannot be clustered. Each <b>day</b> is treated as a consumption "
                "profile; clusters are named only from their measured centroid statistics (|z| > 0.6) and a binomial test for "
                "weekend over-representation.")
        names = list(prof.sort_values("cluster")["name"])
        c1, c2 = st.columns([3, 2])
        fig = go.Figure()
        for i, n in enumerate(names):
            d = days[days["cluster_name"] == n]
            fig.add_scatter(x=d["pc1"], y=d["pc2"], mode="markers", name=f"{n} ({len(d)})",
                            marker=dict(size=8, color=SERIES[i % 6], symbol=SYMBOLS[i % 8], opacity=0.75, line=dict(width=1, color="#151F32")),
                            customdata=d[["date", "daily_kwh"]].astype(str),
                            hovertemplate="%{customdata[0]}<br>%{customdata[1]} kWh<extra>" + n + "</extra>")
        ev = h["pca_explained_variance"]
        c1.plotly_chart(style(fig, 460, "Day clusters in PCA space", xaxis_title=f"PC1 ({ev[0] * 100:.0f} %)",
                              yaxis_title=f"PC2 ({ev[1] * 100:.0f} %)", legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
                              margin=dict(l=48, r=18, t=48, b=120)), width="stretch")
        curves = dc.table("segmentation", "household_cluster_curves")
        fig = go.Figure()
        for i, n in enumerate(names):
            if n in curves:
                fig.add_scatter(x=curves["hour"], y=curves[n], name=n, line=dict(color=SERIES[i % 6], width=2),
                                hovertemplate=f"{n}<br>%{{x}}:00 - %{{y:.2f}} kWh<extra></extra>")
        c2.plotly_chart(style(fig, 460, "Mean 24-h load curve per cluster (kWh)", showlegend=False, xaxis_title="Hour"),
                        width="stretch")
        st.markdown("#### Cluster profiles (actual statistics)")
        cols = ["name", "n_days", "daily_kwh", "max_kwh", "min_kwh", "variability_cv", "night_share", "daytime_share", "peak_hour_kwh",
                "load_factor", "weekend_day_share", "months_most_common"]
        st.dataframe(prof[cols].style.format({c: "{:.3f}" for c in cols[2:-1]}), width="stretch", hide_index=True)
        c3, c4 = st.columns(2)
        st.markdown("")
        c3.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns if comp[c].dtype.kind == "f"}), width="stretch", hide_index=True)
        sel = dc.table("segmentation", "household_k_selection")
        fig = go.Figure()
        fig.add_scatter(x=sel["k"], y=sel["kmeans_silhouette"], mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                        name="Silhouette", hovertemplate="k=%{x}: %{y:.3f}<extra></extra>")
        c4.plotly_chart(style(fig, 280, "Model selection: K-Means silhouette by k", xaxis_title="k"), width="stretch")
        load = dc.table("segmentation", "household_pca_loadings")
        fig = go.Figure(go.Heatmap(z=load[["PC1", "PC2", "PC3"]].values, x=["PC1", "PC2", "PC3"], y=load["feature"],
                                   colorscale=[[0, "#3B82F6"], [0.5, "#383838"], [1, "#EF4444"]], zmid=0,
                                   hovertemplate="%{y} on %{x}: %{z:.2f}<extra></extra>"))
        st.plotly_chart(style(fig, 330, "PCA loadings (diverging: blue negative, red positive)"), width="stretch")
    with tab2:
        g = s["grid"]
        gc = dc.table("segmentation", "grid_countries")
        kpi_row([("Countries", str(g["n_countries"]), "", g["unit_of_analysis"], BLUE),
                 ("Clusters (Ward)", str(g["chosen_k"]), "", "silhouette-selected", SERIES[1]),
                 ("K-Means agreement", fmt(g["kmeans_vs_hierarchical_ari"], 2), "ARI", "vs hierarchical labels", SERIES[2])])
        c1, c2 = st.columns(2)
        fig = go.Figure()
        for i, c in enumerate(sorted(gc["cluster"].unique())):
            d = gc[gc["cluster"] == c]
            fig.add_scatter(x=d["pc1"], y=d["pc2"], mode="markers+text", text=d["country"], textposition="top center",
                            marker=dict(size=11, color=SERIES[i % 6], symbol=SYMBOLS[i], line=dict(width=1, color="#151F32")),
                            name=f"Cluster {c}", hovertemplate="%{text}<extra></extra>")
        c1.plotly_chart(style(fig, 440, "National grids by load-shape features (PCA)"), width="stretch")
        fig = go.Figure(go.Scatter(x=gc["winter_summer_ratio"], y=gc["weekend_ratio"], mode="markers+text", text=gc["country"],
                                   textposition="top center", marker=dict(size=10, color=BLUE),
                                   hovertemplate="%{text}<br>winter/summer %{x:.2f}<br>weekend/weekday %{y:.2f}<extra></extra>"))
        c2.plotly_chart(style(fig, 440, "Seasonality vs weekly pattern", xaxis_title="Winter / summer load ratio",
                              yaxis_title="Weekend / weekday load ratio"), width="stretch")
        st.dataframe(gc.drop(columns=["pc1", "pc2"]).style.format(precision=3), width="stretch", hide_index=True)
        st.caption("Dendrogram leaf order (Ward linkage): " + " · ".join(g["dendrogram_order"]))
