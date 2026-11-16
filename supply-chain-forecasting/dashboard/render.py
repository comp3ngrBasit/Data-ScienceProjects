"""Render dashboard screens from outputs/ into screenshots/*.png.

Charts are drawn with matplotlib as inline SVG, laid out in HTML, and captured
with a headless browser at 1600x1200 (4:3, Upwork portfolio friendly).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT, SHOTS = ROOT / "outputs", ROOT / "screenshots"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
STATUS = {"Healthy": ("#0ca30c", "●"), "Reorder": ("#fab219", "▲"),
          "Critical": ("#d03b3b", "■"), "Overstock": ("#ec835a", "◆")}

plt.rcParams.update({
    "font.family": "Inter", "font.size": 11, "text.color": INK2, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "none", "axes.facecolor": "none", "svg.fonttype": "none",
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.axisbelow": True,
})
kfmt = FuncFormatter(lambda v, _: f"{v/1000:.1f}K" if abs(v) >= 1000 else f"{v:.0f}")


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def page(title: str, active: str, body: str, summary: dict) -> str:
    nav = "".join(f'<div class="nav {"on" if n == active else ""}">{n}</div>'
                  for n in ["Overview", "Demand forecast", "Inventory plan", "Model insights"])
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:1600px;height:1200px;background:#f3f2ee;font-family:Inter,sans-serif;color:{INK};display:flex}}
.side{{width:250px;background:#16233a;color:#c9d3e3;padding:32px 22px}}
.logo{{color:#fff;font-weight:700;font-size:19px;line-height:1.25;margin-bottom:6px}}
.sub{{font-size:12px;color:#8fa0bb;margin-bottom:36px}}
.nav{{padding:11px 14px;border-radius:8px;font-size:15px;margin-bottom:4px}}
.nav.on{{background:#2a78d6;color:#fff;font-weight:600}}
.meta{{position:absolute;bottom:30px;left:22px;width:206px;font-size:12px;color:#8fa0bb;line-height:1.6}}
main{{flex:1;padding:34px 40px;display:flex;flex-direction:column;gap:22px}}
h1{{font-size:28px;font-weight:700;letter-spacing:-.01em}}
.hdr{{display:flex;justify-content:space-between;align-items:flex-end}}
.hdr p{{color:{INK2};font-size:15px;margin-top:6px}}
.chip{{background:#fff;border:1px solid rgba(11,11,11,.1);border-radius:8px;padding:8px 14px;font-size:13px;color:{INK2}}}
.card{{background:{SURFACE};border:1px solid rgba(11,11,11,.08);border-radius:14px;padding:22px 24px}}
.card h2{{font-size:17px;font-weight:600;margin-bottom:4px}}
.card .cap{{font-size:13px;color:{INK2};margin-bottom:12px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}}
.kpi .lab{{font-size:13px;color:{INK2};font-weight:500}}
.kpi .val{{font-size:36px;font-weight:700;margin:8px 0 4px;letter-spacing:-.02em}}
.kpi .d{{font-size:13px;color:{INK2}}} .good{{color:#006300;font-weight:600}}
.row{{display:grid;gap:18px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;font-weight:600;color:{INK2};font-size:12px;text-transform:uppercase;letter-spacing:.04em;padding:9px 10px;border-bottom:1px solid {AXIS}}}
td{{padding:10px 10px;border-bottom:1px solid {GRID};font-variant-numeric:tabular-nums}}
td.n,th.n{{text-align:right}}
.st{{display:inline-flex;align-items:center;gap:7px;font-weight:600;font-size:13px;padding:4px 10px;border-radius:20px;background:#f3f2ee;color:{INK}}}
.legend{{display:flex;gap:20px;font-size:13px;color:{INK2};margin-bottom:6px}}
.sw{{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}}
svg{{width:100%;height:auto;display:block}}
.bar{{position:relative;height:12px;background:#ecebe6;border-radius:6px;width:220px}}
.bar i{{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:{S1}}}
.bar b{{position:absolute;top:-4px;bottom:-4px;width:2px;background:{INK}}}
</style></head><body>
<div class="side"><div class="logo">AI Supply Chain<br>Forecasting</div><div class="sub">Demand &amp; inventory intelligence</div>{nav}
<div class="meta">{summary['n_skus']} SKUs · {summary['n_warehouses']} warehouses<br>History {summary['history_start']} → {summary['history_end']}<br>Model: gradient boosting, {summary['horizon_days']}-day horizon</div></div>
<main><div class="hdr"><div><h1>{title}</h1></div><div class="chip">Data through {summary['history_end']}</div></div>{body}</main></body></html>"""


def overview(sales, test, future, inv, s) -> str:
    m = s["metrics"]
    tot = sales.groupby("date")["units"].sum()
    tot = tot[tot.index > tot.index.max() - pd.Timedelta(days=150)]
    bt = test.groupby("date")["forecast"].sum()
    fut = future.groupby("date")[["forecast"]].sum()
    fig, ax = plt.subplots(figsize=(12.6, 3.7))
    ax.plot(tot.index, tot.values, color=S1, lw=2, label="Actual")
    ax.plot(bt.index, bt.values, color=S2, lw=2, label="Forecast (backtest)")
    ax.plot(fut.index, fut["forecast"], color=S2, lw=2, ls=(0, (4, 3)), label="Forecast (next 28 days)")
    ax.axvline(tot.index.max(), color=AXIS, lw=1)
    ax.text(tot.index.max(), ax.get_ylim()[1], "  Today", va="top", color=INK2, fontsize=11)
    ax.yaxis.set_major_formatter(kfmt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylim(0, None)
    chart = svg(fig)

    reorder = int(s["status_counts"].get("Reorder", 0) + s["status_counts"].get("Critical", 0))
    over = int(s["status_counts"].get("Overstock", 0))
    improve = 1 - m["model_wape"] / m["ma28_wape"]
    sim = {r["policy"]: r for r in s["simulation"]}
    stock_cut = 1 - sim["AI forecast"]["avg_stock_value"] / sim["Manual min/max"]["avg_stock_value"]
    kpis = f"""<div class="kpis">
<div class="card kpi"><div class="lab">Forecast error (WAPE)</div><div class="val">{m['model_wape']:.1%}</div><div class="d"><span class="good">{improve:.0%} lower</span> than 28-day average</div></div>
<div class="card kpi"><div class="lab">Forecast demand, next 28 days</div><div class="val">{s['forecast_units_28d']/1000:.1f}K</div><div class="d">units across all SKUs</div></div>
<div class="card kpi"><div class="lab">Items to reorder now</div><div class="val">{reorder} <span style="font-size:20px;color:{INK2};font-weight:500">of {len(inv)}</span></div><div class="d">{over} overstocked · ${s['excess_value']/1000:.1f}K excess</div></div>
<div class="card kpi"><div class="lab">Inventory held vs manual rule</div><div class="val">−{stock_cut:.0%}</div><div class="d">at {sim['AI forecast']['fill_rate']:.1%} fill rate (backtest)</div></div></div>"""
    legend = f"""<div class="legend"><span><i class="sw" style="background:{S1}"></i>Actual</span><span><i class="sw" style="background:{S2}"></i>Forecast (backtest)</span><span><i class="sw" style="background:repeating-linear-gradient(90deg,{S2} 0 5px,transparent 5px 8px)"></i>Forecast (next 28 days)</span></div>"""
    order = {"Critical": 0, "Reorder": 1, "Overstock": 2}
    alerts = inv[inv["status"] != "Healthy"].assign(o=lambda d: d["status"].map(order)).sort_values(["o", "days_of_cover"]).head(5)
    rows = "".join(f"<tr><td><b>{r.sku}</b> · {r.category}</td><td>{r.warehouse}</td><td>{badge(r.status)}</td><td class='n'>{r.days_of_cover:.0f} days</td><td class='n'>{int(r.suggested_order):,}</td></tr>" for r in alerts.itertuples())
    return kpis + f"""<div class="card"><h2>Total daily demand: actual vs forecast</h2><div class="cap">All SKUs and warehouses. The model is scored on the last 91 days, which it never saw during training.</div>{legend}{chart}</div>
<div class="card"><h2>Stock alerts</h2><div class="cap">Items outside their target range, from the inventory plan</div><table><tr><th>Item</th><th>Warehouse</th><th>Status</th><th class="n">Days of cover</th><th class="n">Suggested order</th></tr>{rows}</table></div>"""


def badge(status: str) -> str:
    c, icon = STATUS[status]
    return f'<span class="st"><span style="color:{c}">{icon}</span>{status}</span>'


def forecast_page(test, per_sku, s) -> str:
    d = test[test["warehouse"] == "Karachi DC"]
    skus = sorted(d["sku"].unique())
    cats = d.groupby("sku")["category"].first()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), sharex=True)
    for ax, sku in zip(axes.flat, skus):
        x = d[d["sku"] == sku].sort_values("date")
        ax.fill_between(x["date"], x["lo"], x["hi"], color=S2, alpha=0.15, lw=0)
        ax.plot(x["date"], x["units"], color=S1, lw=1.6)
        ax.plot(x["date"], x["forecast"], color=S2, lw=2)
        w = per_sku.set_index("sku").loc[sku, "model_wape"]
        ax.set_title(f"{sku} · {cats[sku]}", loc="left", fontsize=12.5, color=INK, fontweight=600, pad=8)
        ax.text(1, 1.02, f"WAPE {w:.1%}", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=INK2)
        ax.set_ylim(0, None)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    chart = svg(fig)

    p = per_sku.copy()
    p["gain"] = 1 - p["model_wape"] / p["ma28_wape"]
    rows = "".join(f"<tr><td><b>{r.sku}</b></td><td>{cats[r.sku]}</td><td class='n'><b>{r.model_wape:.1%}</b></td><td class='n'>{r.ma28_wape:.1%}</td><td class='n'>{r.lastyear_wape:.1%}</td><td class='n good'>{r.gain:.0%}</td></tr>" for r in p.itertuples())
    m = s["metrics"]
    rows += f"<tr><td><b>All items</b></td><td></td><td class='n'><b>{m['model_wape']:.1%}</b></td><td class='n'>{m['ma28_wape']:.1%}</td><td class='n'>{m['lastyear_wape']:.1%}</td><td class='n good'>{1-m['model_wape']/m['ma28_wape']:.0%}</td></tr>"
    legend = f"""<div class="legend"><span><i class="sw" style="background:{S1}"></i>Actual units</span><span><i class="sw" style="background:{S2}"></i>28-day-ahead forecast</span><span><i class="sw" style="background:{S2};opacity:.25;height:10px"></i>80% prediction range</span></div>"""
    return f"""<div class="card"><h2>Forecast vs actual by SKU · Karachi DC, backtest period</h2><div class="cap">Each forecast was made 28 days in advance, using only data available at that time</div>{legend}{chart}</div>
<div class="card"><h2>Forecast error by SKU (WAPE, lower is better)</h2><div class="cap">Compared with two common planning baselines over the same 91-day period</div>
<table><tr><th>SKU</th><th>Category</th><th class="n">AI model</th><th class="n">28-day average</th><th class="n">Same day last year</th><th class="n">Error reduction vs 28-day avg</th></tr>{rows}</table></div>"""


def inventory_page(inv, s) -> str:
    inv = inv.sort_values(["sku", "warehouse"])
    rows = ""
    for r in inv.itertuples():
        scale = max(r.on_hand, r.max_stock) * 1.05
        rows += (f"<tr><td><b>{r.sku}</b><br><span style='color:{INK2};font-size:12px'>{r.category}</span></td><td>{r.warehouse}</td>"
                 f"<td class='n'>{r.daily_forecast:,.0f}</td><td class='n'>{r.lead_time_days}d</td><td class='n'>{r.safety_stock:,.0f}</td>"
                 f"<td class='n'>{r.reorder_point:,.0f}</td><td class='n'><b>{r.on_hand:,.0f}</b></td>"
                 f"<td><div class='bar'><i style='width:{r.on_hand/scale*100:.1f}%'></i><b style='left:{r.reorder_point/scale*100:.1f}%'></b></div></td>"
                 f"<td>{badge(r.status)}</td><td class='n'>{int(r.suggested_order):,}</td></tr>")
    return f"""<style>td{{padding:15px 10px}}</style><div class="card" style="flex:1"><h2>Replenishment plan</h2><div class="cap">Safety stock targets a {s['service_level']:.0%} service level using each item's forecast error and supplier lead time. Order size uses the economic order quantity.</div>
<div class="legend"><span><i class="sw" style="background:{S1};height:10px"></i>On hand</span><span><i class="sw" style="background:{INK};width:2px;height:14px"></i>Reorder point</span></div>
<table><tr><th>Item</th><th>Warehouse</th><th class="n">Daily forecast</th><th class="n">Lead time</th><th class="n">Safety stock</th><th class="n">Reorder point</th><th class="n">On hand</th><th>Stock vs reorder point</th><th>Status</th><th class="n">Order qty</th></tr>{rows}</table></div>"""


def insights_page(imp, s) -> str:
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.barh(top["label"], top["importance"], color=S1, height=0.6)
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2, labelsize=11.5)
    ax.set_xlabel("Increase in error when feature is shuffled (units)", fontsize=10.5)
    chart1 = svg(fig)

    m = s["metrics"]
    labels = ["Same day\nlast year", "28-day\naverage", "AI model"]
    vals = [m["lastyear_wape"], m["ma28_wape"], m["model_wape"]]
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    bars = ax.bar(labels, vals, color=[AXIS, AXIS, S1], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.1%}", ha="center", color=INK, fontsize=13, fontweight=600)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="x", colors=INK2, labelsize=11.5)
    chart2 = svg(fig)

    sim = {r["policy"]: r for r in s["simulation"]}
    a, b = sim["AI forecast"], sim["Manual min/max"]
    simrows = "".join(f"<tr><td>{lab}</td><td class='n'><b>{fa}</b></td><td class='n'>{fb}</td></tr>" for lab, fa, fb in [
        ("Average inventory value", f"${a['avg_stock_value']/1000:.1f}K", f"${b['avg_stock_value']/1000:.1f}K"),
        ("Fill rate (demand served)", f"{a['fill_rate']:.1%}", f"{b['fill_rate']:.1%}"),
        ("Stockout days (12 items × 91 days)", f"{a['stockout_days']:.0f}", f"{b['stockout_days']:.0f}"),
    ])
    return f"""<div class="row" style="grid-template-columns:1.1fr 1fr">
<div class="card"><h2>What drives the forecast</h2><div class="cap">Permutation importance on the backtest period, top 10 features</div>{chart1}</div>
<div class="card"><h2>Forecast error vs baselines</h2><div class="cap">WAPE over 91 held-out days, all items (lower is better)</div>{chart2}</div></div>
<div class="row" style="grid-template-columns:1.1fr 1fr">
<div class="card"><h2>Policy simulation on the backtest period</h2><div class="cap">Day-by-day replay with real lead times: forecast-driven reorder points vs a manual min/max rule</div>
<table><tr><th>Metric</th><th class="n">AI policy</th><th class="n">Manual min/max</th></tr>{simrows}</table></div>
<div class="card"><h2>Model card</h2><div class="cap">How the forecast is produced</div><table>
<tr><td>Algorithm</td><td>Gradient boosting (Poisson loss), one global model</td></tr>
<tr><td>Uncertainty</td><td>Quantile models, 80% range ({m['coverage_80']:.0%} realised coverage)</td></tr>
<tr><td>Horizon</td><td>{s['horizon_days']} days ahead, no future data leakage</td></tr>
<tr><td>Forecast bias</td><td>{m['bias']:+.1%} (total forecast vs actual)</td></tr>
<tr><td>Training data</td><td>{s['n_rows']:,} daily SKU × warehouse records</td></tr></table></div></div>"""


def build_app(bodies: dict, s: dict) -> str:
    """One interactive page: sidebar switches between the four views (hash routed)."""
    views = [("overview", "Overview", "Demand & Inventory Overview"),
             ("forecast", "Demand forecast", "Demand Forecast Accuracy"),
             ("inventory", "Inventory plan", "Inventory Optimization"),
             ("insights", "Model insights", "Model Insights")]
    shell = page("__TITLE__", "", "__BODY__", s)
    head, rest = shell.split("<body>", 1)
    head = head.replace("<head>", '<head><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Supply Chain Forecasting</title>'
                        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">')
    head = head.replace("body{width:1600px;height:1200px;", "body{min-width:1280px;min-height:100vh;")
    head = head.replace(".side{", ".side{position:sticky;top:0;height:100vh;flex:none;")
    head = head.replace("</style>", ".nav{cursor:pointer;user-select:none}.nav:hover:not(.on){background:rgba(255,255,255,.07)}.view{display:none;flex-direction:column;gap:22px}td b{white-space:nowrap}th{white-space:nowrap}.bar{width:min(220px,13vw)}.view.on{display:flex}</style>")
    nav = "".join(f'<div class="nav" data-v="{k}">{lab}</div>' for k, lab, _ in views)
    side_start = rest.index('<div class="nav')
    side_end = rest.index('<div class="meta">')
    rest = rest[:side_start] + nav + rest[side_end:]
    main_start = rest.index("<main>")
    sections = "".join(
        f'<section class="view" id="v-{k}"><div class="hdr"><div><h1>{title}</h1></div><div class="chip">Data through {s["history_end"]}</div></div>{bodies[k]}</section>'
        for k, _, title in views)
    script = """<script>
const show=()=>{let v=(location.hash||'#overview').slice(1);if(!document.getElementById('v-'+v))v='overview';
document.querySelectorAll('.view').forEach(e=>e.classList.toggle('on',e.id==='v-'+v));
document.querySelectorAll('.nav').forEach(e=>e.classList.toggle('on',e.dataset.v===v));window.scrollTo(0,0)};
document.querySelectorAll('.nav').forEach(e=>e.onclick=()=>{location.hash=e.dataset.v});
addEventListener('hashchange',show);show();</script>"""
    return head + "<body>" + rest[:main_start] + "<main>" + sections + "</main>" + script + "</body></html>"


def main() -> None:
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(exist_ok=True)
    s = json.loads((OUT / "summary.json").read_text())
    sales = pd.read_csv(OUT / "sales_history.csv", parse_dates=["date"])
    test = pd.read_csv(OUT / "backtest.csv", parse_dates=["date"])
    future = pd.read_csv(OUT / "forecast_next_28_days.csv", parse_dates=["date"])
    inv = pd.read_csv(OUT / "inventory_plan.csv")
    per_sku = pd.read_csv(OUT / "backtest_by_sku.csv")
    imp = pd.read_csv(OUT / "feature_importance.csv")

    bodies = {"overview": overview(sales, test, future, inv, s), "forecast": forecast_page(test, per_sku, s),
              "inventory": inventory_page(inv, s), "insights": insights_page(imp, s)}
    pages = {
        "01_overview": page("Demand & Inventory Overview", "Overview", bodies["overview"], s),
        "02_demand_forecast": page("Demand Forecast Accuracy", "Demand forecast", bodies["forecast"], s),
        "03_inventory_plan": page("Inventory Optimization", "Inventory plan", bodies["inventory"], s),
        "04_model_insights": page("Model Insights", "Model insights", bodies["insights"], s),
    }
    (ROOT / "dashboard" / "index.html").write_text(build_app(bodies, s))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1600, "height": 1200})
        for name, html in pages.items():
            (SHOTS / f"{name}.html").write_text(html)
            pg.goto((SHOTS / f"{name}.html").as_uri())
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(SHOTS / f"{name}.png"))
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
