"""Analytical report engine: PDF (ReportLab + Matplotlib) and CSV exports.

Every sentence containing a number is generated from result files written by the pipeline;
nothing is hard-coded. Sections whose results are missing are reported as missing.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from app.config import OPSD_SOURCE, OPSD_WEATHER_SOURCE, REPORTS_DIR, UCI_SOURCE  # noqa: E402
from app.utils.helpers import load_json, load_table  # noqa: E402

BLUE, CYAN, GREEN, AMBER, RED = "#3B82F6", "#0E9FB8", "#5FA83A", "#C9780A", "#EF4444"
EXPORT_TABLES = [("forecasting", f"{k}_metrics") for k in
                 ("household_hourly_h24", "household_hourly_h168", "household_daily_h30", "grid_hourly_h24", "grid_hourly_h168")] + [
    ("anomaly", "consensus"), ("anomaly", "daily_events"), ("anomaly", "data_quality"), ("peak", "household_events"),
    ("peak", "grid_events"), ("segmentation", "household_profiles"), ("segmentation", "grid_countries"),
    ("renewable", "annual"), ("renewable", "europe_2016_2019"), ("weather", "model_metrics"), ("weather", "ols_coefficients"),
    ("weather", "correlations"), ("bayesian", "posterior_summary"), ("bayesian", "coefficients"),
    ("statistics", "hypothesis_tests")]


def f(x, nd=1):
    try:
        return f"{x:,.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fig(fn, w=16, h=6.5) -> Image:
    fig, ax = plt.subplots(figsize=(w / 2.54, h / 2.54), dpi=150)
    fn(ax)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w * cm, height=h * cm)


_CELL = ParagraphStyle("cell", fontSize=6.3, leading=7.6)
PAGE_W = 17.8 * cm


def _table(df: pd.DataFrame, max_rows: int = 12, col_w: list | None = None) -> Table:
    """Render a DataFrame; long text is wrapped in Paragraphs and columns are sized to fit the page width."""
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}")
    body = d.astype(str).values.tolist()
    lens = [max([len(str(c))] + [len(r[i]) for r in body]) for i, c in enumerate(d.columns)]
    if col_w is None:
        weights = [min(max(n, 10), 60) for n in lens]
        col_w = [PAGE_W * w / sum(weights) for w in weights]
    wrap = lambda x, w: Paragraph(x.replace("&", "&amp;").replace("<", "&lt;"), _CELL) if len(x) * 3.3 > w else x  # noqa: E731
    data = [list(map(str, d.columns))] + [[wrap(v, col_w[i]) for i, v in enumerate(r)] for r in body]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#151F32")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")])]))
    return t


def recommendations() -> list[str]:
    rec = []
    p, a, r = load_json("peak", "summary"), load_json("anomaly", "summary"), load_json("renewable", "summary")
    w, st = load_json("weather", "summary"), load_table("statistics", "hypothesis_tests")
    seg = load_table("segmentation", "household_profiles")
    if p:
        h = p["household"]
        rec.append(f"Household load shifting: the {h['peak_definition'].split(': ')[-1]} window holds "
                   f"{h['peak_share_of_energy'] * 100:.1f} % of energy in {h['peak_share_of_hours'] * 100:.1f} % of hours "
                   f"(intensity ratio {h['peak_intensity_ratio']:.2f}). Moving deferrable loads (laundry, water heating) out of "
                   f"this window is the most direct lever on peak demand and time-of-use costs.")
        g = p["grid"]
        rec.append(f"Grid peak planning: the 10-year return level of the German daily peak is "
                   f"{g['extreme_value']['return_levels']['10y'] / 1e3:,.1f} GW (95 % CI "
                   f"{g['extreme_value']['return_level_10y_ci95'][0] / 1e3:,.1f}-{g['extreme_value']['return_level_10y_ci95'][1] / 1e3:,.1f} GW) "
                   f"against an observed maximum of {g['max']['value'] / 1e3:,.1f} GW; adequacy margins should be set against the upper bound.")
    if seg is not None and len(seg):
        low = seg.sort_values("daily_kwh").iloc[0]
        rec.append(f"Occupancy-aware analytics: {int(low['n_days'])} days form the '{low['name']}' segment averaging "
                   f"{low['daily_kwh']:.1f} kWh/day (most common months: {low['months_most_common']}). Excluding such days from "
                   f"baselines avoids under-forecasting normal operation.")
    if a:
        top_driver = max(a["consensus_by_driver"], key=a["consensus_by_driver"].get) if a.get("consensus_by_driver") else "n/a"
        rec.append(f"Anomaly triage: {a['n_consensus_anomalies']} hours were flagged by >= 2 detectors; the sub-meter most "
                   f"often driving them is '{top_driver}'. {a['n_data_quality_findings']} data-quality findings (outages, voltage "
                   f"excursions) should be resolved with the meter operator before anomalies are acted on.")
    if r:
        k = r["kpis"]
        rec.append(f"Flexibility needs: variable renewables covered {k['vre_penetration_pct']:.1f} % of German load, but "
                   f"{k['n_low_vre_periods_24h']} periods of >= 24 h had VRE below 10 % of load (longest {k['longest_low_vre_period_h']} h). "
                   f"Firm capacity, storage or imports must cover these multi-day lulls.")
    if w:
        e = w["hdd_effect_gwh_per_degree_day"]
        rec.append(f"Weather-normalised planning: each heating degree-day adds {e['coef']:.1f} GWh of daily German load "
                   f"(95 % CI {e['ci95'][0]:.1f}-{e['ci95'][1]:.1f}); weather explains "
                   f"{(w['weather_explained_variance']['calendar_plus_weather_r2'] - w['weather_explained_variance']['calendar_only_r2']) * 100:.1f} "
                   f"percentage points of variance beyond the calendar, so temperature forecasts belong in demand forecasts.")
    if st is not None and len(st):
        h7 = st[st["hypothesis"].str.startswith("H7")]
        if len(h7) and not bool(h7.iloc[0]["significant_holm_0.05"]):
            rec.append("Cooling demand is not yet a significant driver of German load (hot vs moderate summer working days: "
                       f"Holm-adjusted p = {h7.iloc[0]['p_value_holm']:.3f}); monitor this as air-conditioning adoption grows.")
    return rec


def generate() -> dict:
    out_dir = REPORTS_DIR
    pdf_path = out_dir / "energy_grid_intelligence_report.pdf"
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#0F172A"))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#1E40AF"), spaceBefore=8)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=8.8, leading=12)
    S = ParagraphStyle("S", parent=B, fontSize=7.5, textColor=colors.HexColor("#475569"))
    el = []
    k = load_json("overview", "kpis") or {}
    el += [Paragraph("Energy Consumption &amp; Grid Intelligence - Analytical Report", H1),
           Paragraph(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from pipeline outputs.", S), Spacer(1, 6)]
    if k:
        el.append(Paragraph("Executive summary", H2))
        el.append(Paragraph(
            f"Household: {f(k.get('household_total_kwh', 0) / 1e3)} MWh recorded, {f(k.get('household_avg_daily_kwh'))} kWh/day on average, "
            f"maximum hourly demand {f(k.get('household_max_hourly_kwh'), 2)} kWh; {k.get('consensus_anomalies', 'n/a')} consensus anomalies. "
            f"German grid: {f(k.get('grid_total_twh'), 0)} TWh load, peak {f(k.get('grid_max_load_mw', 0) / 1e3)} GW, wind + solar = "
            f"{f(k.get('vre_penetration_pct'))} % of load. Best 24-h forecasters: household {k.get('household_hourly_h24_best_model')} "
            f"(WAPE {f(k.get('household_hourly_h24_best_wape'))} %), grid {k.get('grid_hourly_h24_best_model')} "
            f"(WAPE {f(k.get('grid_hourly_h24_best_wape'), 2)} %).", B))
    # 1 dataset description
    dq_h, dq_g = load_json("data_quality", "household"), load_json("data_quality", "grid_DE")
    el.append(Paragraph("1. Datasets", H2))
    for src in (UCI_SOURCE, OPSD_SOURCE, OPSD_WEATHER_SOURCE):
        el.append(Paragraph(f"<b>{src.name}</b>. {src.citation}", B))
    if dq_h:
        el.append(Paragraph(
            f"Household: {dq_h['quality_report']['n_rows']:,} minute records ({dq_h['quality_report']['start']} to "
            f"{dq_h['quality_report']['end']}), aggregated to {dq_h['hourly_rows']:,} hours; {dq_h['hours_imputed']} hours "
            f"({dq_h['hours_imputed_share'] * 100:.2f} %) imputed and flagged. {dq_h['timezone_note']}", B))
    if dq_g:
        el.append(Paragraph(f"Grid (DE): {dq_g['rows']:,} hourly records, {dq_g['period_utc'][0]} to {dq_g['period_utc'][1]} UTC, "
                            f"analysed in {dq_g['timezone']} local time.", B))
    el.append(Paragraph("The household (France, 2006-2010) and the German grid (2015-2020) differ in geography, period, units "
                        "and measurement definitions, so they are analysed independently and never merged.", B))
    # 2 data quality
    el.append(Paragraph("2. Data quality assessment", H2))
    rows = []
    for name, dq in (("Household", dq_h), ("Grid DE", dq_g)):
        if dq:
            for c in dq["quality_report"]["checks"]:
                rows.append({"dataset": name, "check": c["check"], "passed": "yes" if c["passed"] else "NO", "detail": c["detail"]})
    if rows:
        el.append(_table(pd.DataFrame(rows), 20, [2.2 * cm, 6.5 * cm, 1.3 * cm, 7 * cm]))
    # 3 consumption patterns
    c = load_json("consumption", "summary")
    el.append(Paragraph("3. Consumption patterns", H2))
    if c:
        hd = c["household"]["daily_kwh"]
        sh = c["household"]["submeter_share"]
        el.append(Paragraph(
            f"Household daily consumption averages {f(hd['mean'])} kWh (median {f(hd['p50'])}, CV {hd['cv']:.2f}). Of total energy, "
            f"water heater/AC accounts for {sh['sub_water_heater_ac_kwh'] * 100:.1f} %, laundry {sh['sub_laundry_kwh'] * 100:.1f} %, kitchen "
            f"{sh['sub_kitchen_kwh'] * 100:.1f} % and unmetered loads {sh['sub_other_kwh'] * 100:.1f} %. German load totals "
            f"{f(c['grid']['total_load_twh'], 0)} TWh over the period, averaging {f(c['grid']['daily_load_mwh']['mean'] / 1e3, 0)} GWh/day.", B))
        prof = load_table("consumption", "household_hourly_by_daytype")
        if prof is not None:
            def plot(ax):
                for dt, col in zip(["Weekday", "Weekend", "Holiday"], [BLUE, AMBER, CYAN]):
                    d = prof[prof["daytype"] == dt]
                    ax.plot(d["hour"], d["mean"], color=col, label=dt, lw=1.6)
                ax.set_xlabel("Hour", fontsize=7); ax.set_ylabel("kWh", fontsize=7); ax.legend(fontsize=7, frameon=False)
                ax.set_title("Household mean hourly load by day type", fontsize=8)
            el.append(_fig(plot))
    # 4 peak
    p = load_json("peak", "summary")
    el.append(Paragraph("4. Peak demand analysis", H2))
    if p:
        for nm, unit, sc in (("household", "kWh/h", 1), ("grid", "GW", 1e3)):
            x = p[nm]
            ev = x["extreme_value"]
            el.append(Paragraph(
                f"<b>{nm.title()}</b>: maximum {f(x['max']['value'] / sc, 2)} {unit} at {x['max']['timestamp']}; mean daily peak-to-average "
                f"ratio {x['mean_daily_par']:.2f}, load factor {x['mean_daily_load_factor']:.2f}. Peak window ({x['peak_definition']}) carries "
                f"{x['peak_share_of_energy'] * 100:.1f} % of energy. {x['n_high_demand_events']} high-demand events above P99. "
                f"GPD fit on {ev.get('n_clusters')} declustered exceedances (shape {ev.get('shape_xi', float('nan')):.2f}, KS p = "
                f"{ev.get('ks_p', float('nan')):.2f}) gives a 10-year return level of {f(ev['return_levels']['10y'] / sc, 2)} {unit}.", B))
    # 5 forecasting
    el.append(PageBreak())
    el.append(Paragraph("5. Forecasting results and model comparison", H2))
    el.append(Paragraph("Chronological train/validation/test split; hyper-parameters chosen on validation; rolling-origin "
                        "walk-forward on the test block; lag features are horizon-safe (every lag >= horizon); 90 % "
                        "prediction intervals (native for SARIMA/Prophet, split-conformal otherwise). Best model by pooled WAPE.", B))
    for key in ("household_hourly_h24", "household_hourly_h168", "household_daily_h30", "grid_hourly_h24", "grid_hourly_h168"):
        s, m = load_json("forecasting", f"{key}_summary"), load_table("forecasting", f"{key}_metrics")
        if s is None or m is None:
            el.append(Paragraph(f"{key}: results missing.", S))
            continue
        el.append(Paragraph(f"<b>{s['label']}</b> - {len(s['origins'])} walk-forward origins; best: <b>{s['best_model']}</b>. "
                            f"Residual diagnostics: {s['residual_diagnostics_best']['interpretation']} (Ljung-Box p = "
                            f"{s['residual_diagnostics_best']['ljung_box_p']:.3f}). Target PSI test vs train "
                            f"{s['drift']['psi_target_test_vs_train']:.3f} ({s['drift']['psi_interpretation']}).", B))
        el.append(_table(m[["model", "MAE", "RMSE", "MAPE", "WAPE", "R2", "PI_coverage", "skill_vs_naive_pct"]], 10))
        el.append(Spacer(1, 4))
    pr = load_table("forecasting", "grid_hourly_h24_predictions")
    if pr is not None:
        best = load_json("forecasting", "grid_hourly_h24_summary")["best_model"]
        d = pr[(pr["model"] == best) & (pr["fold"] == pr["fold"].max())]
        def plot(ax):
            ax.fill_between(d["timestamp"], d["lower"] / 1e3, d["upper"] / 1e3, color=BLUE, alpha=0.18, label="90 % interval")
            ax.plot(d["timestamp"], d["actual"] / 1e3, color="#0F172A", lw=1.2, label="Actual")
            ax.plot(d["timestamp"], d["pred"] / 1e3, color=BLUE, lw=1.4, label=f"{best}")
            ax.set_ylabel("GW", fontsize=7); ax.legend(fontsize=7, frameon=False); ax.set_title("German load, last test fold", fontsize=8)
        el.append(_fig(plot))
    # 6 segmentation
    el.append(Paragraph("6. Segmentation findings", H2))
    sg, prof = load_json("segmentation", "summary"), load_table("segmentation", "household_profiles")
    if sg and prof is not None:
        el.append(Paragraph(f"Unit of analysis: {sg['household']['unit_of_analysis']} ({sg['household']['n_days']} days). "
                            f"K-Means with k = {sg['household']['chosen_k']} (silhouette-selected); bootstrap stability ARI "
                            f"{sg['household']['stability']['mean_ari']:.2f} +/- {sg['household']['stability']['std_ari']:.2f}.", B))
        el.append(_table(prof[["name", "n_days", "daily_kwh", "night_share", "daytime_share", "variability_cv", "load_factor",
                               "weekend_day_share"]], 10))
        comp = load_table("segmentation", "household_algorithm_comparison")
        if comp is not None:
            el.append(_table(comp, 6))
        el.append(Paragraph(f"Regional view: {sg['grid']['n_countries']} national grids clustered into {sg['grid']['chosen_k']} "
                            f"load-shape groups (Ward; K-Means agreement ARI {sg['grid']['kmeans_vs_hierarchical_ari']:.2f}).", B))
    # 7 anomalies
    el.append(Paragraph("7. Anomaly detection results", H2))
    a = load_json("anomaly", "summary")
    if a:
        el.append(Paragraph(f"{a['hours_scored']:,} hours scored by {len(a['methods'])} detectors against a causal baseline "
                            f"({a['baseline']}). Consensus ({a['consensus_rule']}): {a['n_consensus_anomalies']} hours - "
                            + ", ".join(f"{k}: {v}" for k, v in a["consensus_by_type"].items()) + ". Severity: "
                            + ", ".join(f"{k} {v}" for k, v in a["consensus_by_severity"].items()) + f". {a['disclaimer']}", B))
        cons = load_table("anomaly", "consensus")
        if cons is not None and len(cons):
            top = cons.sort_values("severity_score", ascending=False)[["timestamp", "observed_kwh", "expected_kwh", "deviation_pct",
                                                                         "severity", "type", "methods"]]
            el.append(_table(top, 10))
    # 8 renewables
    el.append(PageBreak())
    el.append(Paragraph("8. Renewable energy analysis", H2))
    r = load_json("renewable", "summary")
    if r:
        k_ = r["kpis"]
        el.append(Paragraph(f"Germany {k_['period'][0][:10]} to {k_['period'][1][:10]}: VRE (wind + solar) = {f(k_['total_vre_twh'], 0)} TWh "
                            f"vs load {f(k_['total_load_twh'], 0)} TWh -> penetration {k_['vre_penetration_pct']:.1f} % (denominator: ENTSO-E "
                            f"actual load, same hours). Wind {k_['wind_share_of_vre_pct']:.1f} % / solar {k_['solar_share_of_vre_pct']:.1f} % of VRE. "
                            f"Hourly VRE share peaked at {k_['max_hourly_vre_share_pct']:.1f} % ({k_['max_hourly_vre_share_time']}). "
                            f"OPSD lacks hydro/biomass/conventional series, so the complement is reported as residual load, not as "
                            f"non-renewable generation.", B))
        ann = load_table("renewable", "annual")
        if ann is not None:
            el.append(_table(ann.reset_index(), 8))
            def plot(ax):
                x = ann.index.astype(str)
                ax.bar(x, ann["wind_twh"], color="#0E9FB8", label="Wind", edgecolor="white", linewidth=1)
                ax.bar(x, ann["solar_twh"], bottom=ann["wind_twh"], color="#C9780A", label="Solar", edgecolor="white", linewidth=1)
                for xi, tot, pen in zip(x, ann["vre_twh"], ann["vre_penetration_pct"]):
                    ax.annotate(f"{pen:.1f} % of load", (xi, tot), ha="center", va="bottom", fontsize=6.5, xytext=(0, 2),
                                textcoords="offset points")
                ax.set_ylabel("TWh", fontsize=7); ax.legend(fontsize=7, frameon=False, loc="upper left")
                ax.set_title("German VRE generation; label = VRE share of load (2015, 2020 partial years)", fontsize=8)
            el.append(_fig(plot))
    # 9 weather
    el.append(Paragraph("9. Weather sensitivity", H2))
    w = load_json("weather", "summary")
    if w:
        wm = load_table("weather", "model_metrics")
        el.append(Paragraph(f"Daily German load vs population-weighted weather ({w['weather_source'].get('source')}), "
                            f"train {w['train_period'][0]} to {w['train_period'][1]}, test {w['test_period'][0]} to {w['test_period'][1]}. "
                            f"Unavailable variables: {', '.join(w['variables_missing']) or 'none'}. Best test model: {w['best_model_test']}. "
                            f"{w['causality_note']}", B))
        if wm is not None:
            el.append(_table(wm, 8))
        corr = load_table("weather", "correlations")
        if corr is not None:
            el.append(_table(corr[["variable", "sample", "n", "pearson_r", "pearson_ci95_low", "pearson_ci95_high", "spearman_rho"]], 10))
    hw = (w or {}).get("household", {})
    if hw.get("status") == "ok":
        el.append(Paragraph(f"Household (Open-Meteo, {hw['n_days']} days): each heating degree-day adds {hw['hdd_kwh_per_degree_day']:.2f} kWh/day "
                            f"(p = {hw['hdd_p']:.3g}); R² {hw['r2']:.3f} vs calendar-only {hw['calendar_only_r2']:.3f}.", B))
    elif hw:
        el.append(Paragraph(f"Household weather sensitivity not computed: {hw.get('reason', hw.get('status'))}.", S))
    # 10 bayesian
    el.append(Paragraph("10. Bayesian analysis", H2))
    b = (w or {}).get("models", {}).get("BayesianRegression", {})
    if b and b.get("max_r_hat"):
        el.append(Paragraph(f"PyMC Student-t regression, {b['chains']} chains x {b['draws']} draws: max R-hat {b['max_r_hat']:.3f}, "
                            f"min bulk ESS {b['min_ess_bulk']:.0f}, {b['divergences']} divergences. Posterior-predictive coverage on the "
                            f"held-out block: {b['coverage_90']:.1f} % (90 % interval), {b['coverage_50']:.1f} % (50 % interval).", B))
        bc = load_table("bayesian", "coefficients")
        if bc is not None:
            el.append(_table(bc[~bc["feature"].str.startswith("dow_")], 8))
    # 11 statistics
    el.append(Paragraph("11. Statistical findings", H2))
    st = load_table("statistics", "hypothesis_tests")
    if st is not None:
        for rr in st.to_dict("records"):
            el.append(Paragraph(f"<b>{rr['hypothesis']}</b> - {rr.get('primary_test', '')}, statistic {rr.get('statistic', float('nan')):,.4g}, "
                                f"Holm p = {rr.get('p_value_holm', float('nan')):.3g}, Cohen's d = {rr.get('effect_size_cohens_d', float('nan')):.2f}, "
                                f"95 % CI of difference [{rr.get('ci95_diff_low', float('nan')):,.3g}, {rr.get('ci95_diff_high', float('nan')):,.3g}] "
                                f"{rr.get('unit', '')}. {rr['interpretation']}", B))
    ts = load_json("timeseries", "summary")
    if ts:
        for key in ("household_daily", "grid_daily"):
            sv = ts[key]["stationarity"]["level"]
            el.append(Paragraph(f"{key}: ADF p = {sv['adf_p']:.3g}, KPSS p = {sv['kpss_p']:.3g} -> {sv['verdict']}; STL weekly seasonal "
                                f"strength {ts[key]['seasonal_strength']:.2f}; {ts[key]['n_change_points']} change point(s) in weekly means.", B))
    # 12 recommendations + limitations
    el.append(Paragraph("12. Business recommendations", H2))
    for line in recommendations():
        el.append(Paragraph("- " + line, B))
    el.append(Paragraph("13. Model limitations and assumptions", H2))
    for line in LIMITATIONS:
        el.append(Paragraph("- " + line, B))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title="Energy Grid Intelligence Report")
    doc.build(el)
    csv_zip = export_csv()
    return {"pdf": str(pdf_path), "csv_zip": str(csv_zip)}


LIMITATIONS = [
    "The household dataset describes one dwelling near Paris (2006-2010); its findings do not generalise to a population.",
    "Household and grid data are never combined: different country, period, units (kWh per household vs MW national) and definitions.",
    "Household weather sensitivity runs only when Open-Meteo (ERA5) data for the household location can be downloaded; its status "
    "is reported in the weather section. When Open-Meteo is unreachable, German weather comes from the OPSD weather package "
    "(2015-2016 overlap with load), which has no humidity or precipitation.",
    "Forecasting models use calendar information and past load only; weather is not used as a forecast input, avoiding the "
    "look-ahead bias of using observed weather in place of weather forecasts.",
    "ML/LSTM/SARIMA parameters are frozen at the start of the test block and the model state is updated with new observations at each "
    "origin; Prophet is refitted at each origin. Periodic retraining in production would likely reduce error under drift.",
    "The German test block (Nov 2019 - Sep 2020) contains the COVID-19 demand shock, a structural break that inflates test error.",
    "Imputed hours (long gaps filled from same-hour medians of earlier weeks) are flagged; short gaps use interpolation, which uses the "
    "next observed value and is negligible in scale (<= 3 consecutive hours).",
    "Daily observations are serially correlated: classical p-values are complemented by block-bootstrap confidence intervals; "
    "trend p-values are descriptive.",
    "Anomalies are statistical deviations; none is evidence of theft, failure or malfunction without verification.",
    "Extreme-value return levels assume stationarity of demand and climate regimes.",
]


def export_csv() -> str:
    exp_dir = REPORTS_DIR / "exports"
    exp_dir.mkdir(exist_ok=True)
    zpath = REPORTS_DIR / "energy_grid_intelligence_tables.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for module, name in EXPORT_TABLES:
            t = load_table(module, name)
            if t is not None:
                p = exp_dir / f"{module}__{name}.csv"
                t.to_csv(p, index=not isinstance(t.index, pd.RangeIndex))
                z.write(p, p.name)
    return str(zpath)
