"""AI Supply Chain Copilot — retrieval-augmented generation over the platform database.

Index: `rag_documents` holds one text document per SKU, supplier, KPI snapshot, model card, alert digest,
PO pipeline, logistics plan and data-source note (rebuilt by `build_index`, run by the automation cycle).
Retrieval: TF-IDF (word 1-2 grams) cosine similarity — free, local, no embedding API required — with SKU /
supplier id boosting for exact identifiers in the question.
Generation: Claude or OpenAI with the retrieved documents as the only context and an instruction to cite
document ids. Without an LLM key the Copilot returns a retrieval-only answer built from the top documents.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from scip.core.db import get_engine, read_sql, session_scope, write_frame
from scip.core.logging import get_logger
from scip.core.models import CopilotMessage
from scip.integrations.llm import generate
from scip.services import control_tower as ct

log = get_logger("copilot")
SYSTEM = ("You are the SCIP Supply Chain Copilot. Answer ONLY from the provided context documents, which come "
          "from the company's supply chain database. Be concise and specific, use numbers from the context, and cite "
          "the document ids you used in square brackets, e.g. [sku:22423]. If the context does not contain the "
          "answer, say so. Point out when figures come from simulated data.")


def _fmt(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.{nd}f}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)


def build_documents(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    docs = []
    k = ct.kpis(engine)
    docs.append(("kpi:snapshot", "kpi", None, f"KPI snapshot as of {k['as_of_date']}",
                 "Supply chain KPI snapshot as of {as_of}. Total inventory value {inv}. Planning SKUs {n}. "
                 "Predicted demand next 7 days {d7} units, next 30 days {d30} units (value {v30}), next 90 days {d90} units, "
                 "forecast model {m}. SKUs critical {c}, needing reorder {r}, overstocked {o}, healthy {h}. "
                 "High stockout-risk SKUs {so}. Open alerts: {ac} critical, {aw} warning. Supplier on-time rate {ot}, "
                 "average supplier score {ss}, suppliers on watch/at risk {sr}. Purchase orders pending approval {pp}; "
                 "open PO value {pv}; overdue inbound POs {late}.".format(
                     as_of=k["as_of_date"], inv=_fmt(k.get("total_inventory_value"), 0), n=k.get("planning_skus"),
                     d7=_fmt(k.get("predicted_demand_7d_units"), 0), d30=_fmt(k.get("predicted_demand_30d_units"), 0),
                     v30=_fmt(k.get("predicted_demand_30d_value"), 0), d90=_fmt(k.get("predicted_demand_90d_units"), 0),
                     m=k.get("forecast_model"), c=k.get("sku_critical"), r=k.get("sku_reorder"), o=k.get("sku_overstock"),
                     h=k.get("sku_healthy"), so=k.get("stockout_risk_high"), ac=k.get("alerts_critical"),
                     aw=k.get("alerts_warning"), ot=_fmt((k.get("supplier_on_time_rate") or 0) * 100) + "%",
                     ss=_fmt(k.get("supplier_avg_score")), sr=k.get("suppliers_at_risk"), pp=k.get("po_pending_approval"),
                     pv=_fmt(k.get("po_open_value"), 0), late=k.get("inbound_late"))))
    pol = read_sql("SELECT ip.*, p.name, p.category, p.unit_price FROM inventory_policy ip JOIN products p USING(sku) "
                   "WHERE computed_at=(SELECT MAX(computed_at) FROM inventory_policy)", engine=engine)
    from scip.ml.forecasting import horizon_summary
    hs = horizon_summary(engine).set_index("sku") if True else None
    so = read_sql("SELECT sku, probability, risk_band FROM stockout_predictions WHERE scored_at=(SELECT MAX(scored_at) "
                  "FROM stockout_predictions)", engine=engine).set_index("sku")
    sup = read_sql("SELECT p.sku, p.order_date, s.name FROM purchase_orders p JOIN suppliers s USING(supplier_id)",
                   engine=engine).sort_values("order_date").groupby("sku").last()
    for _, r in pol.iterrows():
        f = hs.loc[r["sku"]] if r["sku"] in hs.index else {}
        p = so.loc[r["sku"]] if r["sku"] in so.index else {}
        docs.append((f"sku:{r['sku']}", "sku", r["sku"], f"{r['sku']} {r['name']}",
                     f"Product SKU {r['sku']} '{r['name']}' in category {r['category']} (ABC class {r['abc_class']}). "
                     f"Status {r['status']}. On hand {_fmt(r['on_hand'], 0)} units, on order {_fmt(r['on_order'], 0)}, "
                     f"inventory value {_fmt(r['inventory_value'], 0)}. Average daily demand {_fmt(r['avg_daily_demand'])} "
                     f"({r['demand_basis']}). Safety stock {_fmt(r['safety_stock'], 0)}, reorder point {_fmt(r['reorder_point'], 0)}, "
                     f"max level {_fmt(r['max_level'], 0)}, recommended order {_fmt(r['recommended_qty'], 0)} units. "
                     f"Days of cover {_fmt(r['days_cover'])}; projected stockout date {r['stockout_date'] or 'n/a'}. "
                     f"Lead time {_fmt(r['lead_time_days'])} days. Forecast next 7 days {_fmt(f.get('forecast_7d'), 0)}, "
                     f"30 days {_fmt(f.get('forecast_30d'), 0)}, 60 days {_fmt(f.get('forecast_60d'), 0)}, "
                     f"90 days {_fmt(f.get('forecast_90d'), 0)} units. Stockout probability (14 days) "
                     f"{_fmt((p.get('probability') or 0) * 100)}% ({p.get('risk_band', 'n/a')}). "
                     f"Current supplier: {sup.loc[r['sku'], 'name'] if r['sku'] in sup.index else 'n/a'}. "
                     "Demand is real (UCI); inventory and supplier data are simulated."))
    from scip.ml.supplier_scoring import latest_scores
    sc = latest_scores(engine)
    for _, r in sc.iterrows():
        docs.append((f"supplier:{r['supplier_id']}", "supplier", r["supplier_id"], r["name"],
                     f"Supplier {r['supplier_id']} {r['name']} ({r['country_code']}, {r['supplier_type']}). Score {r['score']:.1f}/100, "
                     f"tier {r['tier']}. On-time delivery {r['on_time_rate']:.1%} over {r['shipments']} deliveries; "
                     f"average delay when late {r['avg_delay_days']:.1f} days; lead-time CV {r['lead_time_cv']:.2f}; "
                     f"predicted late-delivery risk {r['predicted_late_risk']:.1%}. Data origin: {r['data_origin']}."))
    if not sc.empty:
        docs.append(("summary:suppliers", "supplier_summary", None, "Supplier ranking (all suppliers)",
                     "Supplier ranking from best to worst score: " + " | ".join(
                         f"{r.name} ({r.supplier_id}) score {r.score:.1f}, tier {r.tier}, on-time {r.on_time_rate:.0%}, "
                         f"predicted late risk {r.predicted_late_risk:.0%}" for r in sc.itertuples())
                     + ". Riskiest suppliers are those with the lowest scores (Watch / At risk tiers)."))
    if not pol.empty:
        crit = pol[pol["status"].isin(["Critical", "Reorder"])].sort_values("days_cover")
        over = pol[pol["status"] == "Overstock"].sort_values("inventory_value", ascending=False)
        docs.append(("summary:inventory", "inventory_summary", None, "Inventory risk summary (critical, reorder, overstock SKUs)",
                     f"SKUs critical or at reorder point ({len(crit)}): " + "; ".join(
                         f"{r.sku} {r.name} status {r.status}, on hand {r.on_hand:.0f}, days cover {_fmt(r.days_cover)}, "
                         f"recommended order {r.recommended_qty:.0f}" for r in crit.head(30).itertuples())
                     + f". Overstocked SKUs ({len(over)}), largest by value: " + "; ".join(
                         f"{r.sku} {r.name} value {r.inventory_value:,.0f}" for r in over.head(10).itertuples()) + "."))
    if hs is not None and not hs.empty:
        top = hs.sort_values("forecast_30d", ascending=False).head(15)
        names = pol.set_index("sku")["name"] if not pol.empty else pd.Series(dtype=str)
        docs.append(("summary:forecast", "forecast_summary", None, "Demand forecast summary (top SKUs)",
                     f"Champion model {hs['model'].iat[0]}. Total forecast next 7 days {hs['forecast_7d'].sum():,.0f}, "
                     f"30 days {hs['forecast_30d'].sum():,.0f}, 60 days {hs['forecast_60d'].sum():,.0f}, 90 days "
                     f"{hs['forecast_90d'].sum():,.0f} units. Highest 30-day demand: " + "; ".join(
                         f"{sku} {names.get(sku, '')} {r.forecast_30d:,.0f} units" for sku, r in top.iterrows()) + "."))
    runs = ct.model_overview(engine)
    for mtype, g in runs.groupby("model_type"):
        lines = []
        for _, r in g.drop_duplicates("model_name").iterrows():
            m = r["metrics"] if isinstance(r["metrics"], dict) else json.loads(r["metrics"] or "{}")
            keys = [x for x in ("mae", "rmse", "mape", "wape", "weekly_wape", "f1", "roc_auc", "precision", "recall") if x in m]
            lines.append(f"{r['model_name']}{' (champion)' if r['is_champion'] else ''}: " +
                         ", ".join(f"{x} {m[x]:.3f}" for x in keys if m[x] is not None))
        docs.append((f"model:{mtype}", "model", mtype, f"Model performance — {mtype}",
                     f"Model performance for {mtype} (data window {g['data_window'].iat[0]}): " + "; ".join(lines) + "."))
    al = read_sql("SELECT severity, alert_type, entity, message FROM alerts WHERE status='open' ORDER BY alert_id DESC LIMIT 60",
                  engine=engine)
    if not al.empty:
        docs.append(("alerts:open", "alerts", None, "Open alerts",
                     "Open operational alerts: " + " | ".join(f"{a.severity} {a.alert_type} {a.entity}: {a.message}"
                                                            for a in al.itertuples())))
    po = read_sql("SELECT p.po_id, p.sku, s.name, p.quantity, p.total_value, p.status, p.approval_level, p.promised_date, "
                  "p.predicted_late_risk FROM purchase_orders p JOIN suppliers s USING(supplier_id) "
                  "WHERE p.status IN ('draft','pending_approval','approved','sent') ORDER BY p.order_date DESC LIMIT 80", engine=engine)
    if not po.empty:
        docs.append(("po:pipeline", "po", None, "Open purchase orders",
                     "Open purchase orders: " + " | ".join(
                         f"{r.po_id} SKU {r.sku} from {r.name}: {r.quantity:.0f} units, value {r.total_value:,.0f}, status {r.status}, "
                         f"approval {r.approval_level}, promised {r.promised_date}, predicted late risk "
                         f"{(r.predicted_late_risk or 0):.0%}" for r in po.itertuples())))
    plans = read_sql("SELECT plan_id, plan_type, objective_value, details FROM logistics_plans ORDER BY plan_id DESC LIMIT 6", engine=engine)
    for _, r in plans.drop_duplicates("plan_type").iterrows():
        d = r["details"] if isinstance(r["details"], dict) else json.loads(r["details"])
        if r["plan_type"].startswith("allocation"):
            text = (f"Warehouse allocation plan {r['plan_id']} for network {d.get('network')}: optimal cost {d.get('objective_cost')}, "
                    f"single-DC baseline {d.get('baseline_single_dc_cost')}, saving {(d.get('saving_pct') or 0):.1%}. "
                    f"Utilisation {d.get('utilisation')}. Demand basis: {d.get('demand_basis')}.")
        else:
            text = (f"Routing plan {r['plan_id']} from {d.get('hub_name')}: {d.get('vehicles_used')} vehicles, total distance "
                    f"{d.get('total_distance_km')} km. Routes: " + "; ".join(
                        f"vehicle {x['vehicle']} {x['distance_km']} km via " + ", ".join(s['stop'] for s in x['stops'])
                        for x in d.get("routes", [])))
        docs.append((f"logistics:{r['plan_type']}", "logistics", r["plan_type"], f"Logistics plan {r['plan_type']}", text))
    ds = read_sql("SELECT * FROM data_sources", engine=engine)
    docs.append(("data:sources", "data", None, "Data sources",
                 "Data sources: " + " | ".join(f"{r.name} ({r.origin}) rows {r.row_count}; {r.notes or ''}" for r in ds.itertuples())
                 + ". Demand history is real UCI Online Retail data (Dec 2010 - Dec 2011); shipment delivery data is real "
                   "DataCo; Brazil customer geography is real Olist; inventory, vendors and purchase orders are simulated."))
    perf = ct.shipment_performance(engine)
    if not perf.empty:
        g = perf.groupby("ship_mode").apply(lambda x: np.average(x["late_rate"], weights=x["shipments"]), include_groups=False)
        docs.append(("shipping:modes", "shipping", None, "Shipping mode performance (DataCo)",
                     "Real DataCo late-delivery rate by shipping mode: " + ", ".join(f"{m} {v:.1%}" for m, v in g.items())))
    df = pd.DataFrame(docs, columns=["doc_id", "doc_type", "entity", "title", "content"])
    return df


def build_index(engine=None) -> int:
    engine = engine or get_engine()
    df = build_documents(engine)
    df["updated_at"] = datetime.now().replace(microsecond=0)
    write_frame(df, "rag_documents", engine, truncate=True)
    _Index.cache = None
    return len(df)


class _Index:
    cache = None


def _index(engine):
    if _Index.cache is None:
        docs = read_sql("SELECT doc_id, doc_type, entity, title, content FROM rag_documents", engine=engine)
        if docs.empty:
            return None
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english",
                              token_pattern=r"(?u)\b[\w\-\.]+\b")
        mat = vec.fit_transform(docs["title"] + " " + docs["content"])
        _Index.cache = (docs, vec, mat)
    return _Index.cache


def retrieve(question: str, k: int = 6, engine=None) -> pd.DataFrame:
    idx = _index(engine or get_engine())
    if idx is None:
        return pd.DataFrame()
    docs, vec, mat = idx
    sims = linear_kernel(vec.transform([question]), mat).ravel()
    q = question.upper()
    for i, e in enumerate(docs["entity"].fillna("")):
        if e and re.search(rf"\b{re.escape(e.upper().replace('_', ' '))}", q.replace("-", " ")):
            sims[i] += 1.0
    q_l = question.lower()
    boosts = {"kpi": ["kpi", "overall", "summary", "total", "inventory value", "how are we", "how much inventory",
                      "demand forecast", "next 30", "next 7", "next 90"],
              "supplier_summary": ["supplier", "vendor", "risk"], "inventory_summary": ["stockout", "reorder", "critical",
                                                                                         "overstock", "low stock", "which sku", "which products"],
              "forecast_summary": ["forecast", "demand", "best sell", "top product"],
              "model": ["accuracy", "model", "mape", "wape", "f1", "auc", "forecast performance"],
              "alerts": ["alert", "urgent", "critical"], "po": ["purchase order", " po", "approval", "pending"],
              "logistics": ["route", "warehouse", "allocation", "logistic"], "data": ["data source", "real", "simulated"],
              "shipping": ["shipping mode", "carrier", "late deliver"]}
    for i, t in enumerate(docs["doc_type"]):
        if any(w in q_l for w in boosts.get(t, [])):
            sims[i] += 0.3
    top = np.argsort(-sims)[:k]
    out = docs.iloc[top].copy()
    out["score"] = sims[top]
    return out[out["score"] > 0.01]


def ask(question: str, engine=None, k: int = 6) -> dict:
    engine = engine or get_engine()
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty")
    hits = retrieve(question, k, engine)
    if hits.empty:
        answer, provider = "I couldn't find anything relevant in the supply chain database for that question.", "retrieval"
    else:
        context = "\n\n".join(f"[{r.doc_id}] {r.title}\n{r.content}" for r in hits.itertuples())
        text, provider = generate(SYSTEM, f"Context documents:\n{context}\n\nQuestion: {question}")
        if text:
            answer = text
        else:
            answer = ("LLM not configured — showing the most relevant records from the database:\n\n" +
                      "\n\n".join(f"**[{r.doc_id}] {r.title}** — {r.content[:700]}" for r in hits.head(3).itertuples()))
            provider = "retrieval-only"
    sources = [dict(doc_id=r.doc_id, title=r.title, score=round(float(r.score), 3)) for r in hits.itertuples()] if not hits.empty else []
    with session_scope(engine) as s:
        s.add(CopilotMessage(question=question, answer=answer, sources=sources, provider=provider))
    return {"question": question, "answer": answer, "sources": sources, "provider": provider}
