"""Automated reports (HTML) stored in the `reports` table and the reports/ folder."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from scip.core.config import get_settings
from scip.core.db import get_engine, read_sql, session_scope
from scip.core.models import Report
from scip.integrations.llm import generate
from scip.services import control_tower as ct


def _table(df: pd.DataFrame, n: int = 15) -> str:
    if df is None or df.empty:
        return "<p class='muted'>None.</p>"
    return df.head(n).to_html(index=False, border=0, classes="tbl", na_rep="", float_format=lambda x: f"{x:,.2f}")


def daily_operations_report(engine=None) -> dict:
    engine = engine or get_engine()
    k = ct.kpis(engine)
    alerts = read_sql("SELECT severity, alert_type, entity, message FROM alerts WHERE status='open' "
                      "ORDER BY CASE severity WHEN 'Critical' THEN 0 WHEN 'Warning' THEN 1 ELSE 2 END, alert_id DESC",
                      engine=engine)
    pending = read_sql("SELECT po_id, sku, supplier_id, quantity, total_value, approval_level FROM purchase_orders "
                       "WHERE status='pending_approval' ORDER BY total_value DESC", engine=engine)
    recs = read_sql("SELECT sku, supplier_id, quantity, total_value, rationale FROM procurement_recommendations "
                    "ORDER BY id DESC LIMIT 15", engine=engine)
    summary_prompt = (f"KPIs: {k}. Open alerts (first 15): {alerts.head(15).to_dict('records')}. "
                      f"POs pending approval: {pending.head(10).to_dict('records')}. Write a 4-sentence executive summary "
                      "for the operations manager; mention that inventory/vendor data is simulated.")
    narrative, provider = generate("You write concise, factual supply chain briefings from the data given.", summary_prompt, 400)
    if not narrative:
        narrative = (f"As of {k['as_of_date']}, inventory is valued at {k['total_inventory_value']:,.0f} across "
                     f"{k['planning_skus']} planning SKUs. {k['sku_critical']} SKUs are critical and {k['sku_reorder']} need "
                     f"reordering; {k['alerts_critical']} critical alerts are open. Demand for the next 30 days is forecast at "
                     f"{(k.get('predicted_demand_30d_units') or 0):,.0f} units by the {k.get('forecast_model')} model. "
                     f"{k['po_pending_approval']} purchase orders await approval. (Rule-based summary — no LLM configured.)")
        provider = "template"
    kp = "".join(f"<div class='kpi'><div class='l'>{html.escape(l)}</div><div class='v'>{html.escape(v)}</div></div>" for l, v in [
        ("Inventory value", f"{k['total_inventory_value']:,.0f}"),
        ("Forecast demand 30d", f"{(k.get('predicted_demand_30d_units') or 0):,.0f} units"),
        ("Critical SKUs", str(k["sku_critical"])), ("High stockout risk", str(k["stockout_risk_high"])),
        ("Supplier on-time", f"{(k.get('supplier_on_time_rate') or 0):.1%}"),
        ("POs pending approval", str(k["po_pending_approval"]))])
    title = f"Daily Operations Report — {k['as_of_date']}"
    body = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1050px;margin:24px auto;color:#16352f}}
h1{{color:#1f6f61}} h2{{color:#1f6f61;border-bottom:2px solid #cde7df}} .muted{{color:#6b8a84}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .kpi{{border:1px solid #cde7df;border-left:5px solid #2f9e8a;border-radius:8px;padding:8px}}
.l{{font-size:12px;color:#3d6b62;text-transform:uppercase}} .v{{font-size:20px;font-weight:700}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px}} .tbl th{{background:#1f6f61;color:#fff;text-align:left;padding:5px}} .tbl td{{border-bottom:1px solid #e3eeea;padding:5px}}
</style></head><body><h1>{html.escape(title)}</h1><p class='muted'>Generated {datetime.now():%Y-%m-%d %H:%M} · summary by {provider}</p>
<p>{html.escape(narrative)}</p><div class='grid'>{kp}</div>
<h2>Open alerts</h2>{_table(alerts, 25)}<h2>Purchase orders awaiting approval</h2>{_table(pending)}
<h2>Latest procurement recommendations</h2>{_table(recs)}
<p class='muted'>Demand: real UCI Online Retail. Shipments: real DataCo. Inventory, vendors and POs: simulated.</p></body></html>"""
    with session_scope(engine) as s:
        r = Report(report_type="daily_operations", title=title, html=body)
        s.add(r)
        s.flush()
        rid = r.report_id
    path = get_settings().reports_dir
    path.mkdir(parents=True, exist_ok=True)
    (path / f"daily_operations_{datetime.now():%Y%m%d_%H%M%S}.html").write_text(body, encoding="utf-8")
    return {"report_id": rid, "title": title, "summary_provider": provider}
