# SCIP — Supply Chain Intelligence & Autonomous Operations Platform
## Solution architecture, data model, dataset plan and roadmap

### 1. Architecture overview

```
                    ┌─────────────────────────── Data sources ───────────────────────────┐
                    │ REAL: UCI Online Retail · Olist (Brazil) · DataCo Smart SC · M5 (opt)│
                    │ APIs: Open-Meteo (weather) · REST Countries (geo) · Claude / OpenAI  │
                    │ SIMULATED (labelled): inventory, supplier catalog, POs, warehouses   │
                    └───────────────┬─────────────────────────────────────────────────────┘
                                    │  scip.ingestion  (extract → validate → clean → load)
                                    ▼
┌──────────────────────────── PostgreSQL (single source of truth, SQLAlchemy ORM) ────────────────────────────┐
│ reference: data_sources, countries, locations, products, suppliers, supplier_products                       │
│ facts:     sales_daily, shipments, inventory_snapshots, inventory_current, purchase_orders, po_approvals,   │
│            weather_daily                                                                                    │
│ analytics: model_runs, forecasts, forecast_metrics, stockout_predictions, supplier_scores, inventory_policy,│
│            procurement_recommendations, logistics_plans, alerts, automation_log, reports, rag_documents,    │
│            copilot_messages, app_metadata                                                                  │
└───────▲──────────────▲──────────────▲───────────────▲───────────────▲────────────────▲───────────────────┘
        │              │              │               │               │                │
  scip.ml        scip.optimization  scip.services  scip.copilot   scip.automation  scip.integrations
  forecasting    inventory policy   KPIs, control  RAG: retrieve  rules engine,    weather, countries,
  stockout clf   procurement MIP    tower views    + LLM answer   approvals,       LLM providers,
  delivery-risk  (OR-Tools CP-SAT)                 with sources   reports, notify  n8n webhooks,
  supplier score logistics: LP + VRP                                               mock supplier API
        └──────────────┴──────────────┴───────┬───────┴───────────────┴────────────────┘
                                              ▼
                              FastAPI REST backend (scip.api, /api/v1/*, OpenAPI docs)
                                    ▲                        ▲
                     Streamlit Control Tower           n8n workflows (cron, webhooks,
                     (dashboard/, calls the API)        approval e-mails / Slack)
```

**Principles.** Every module reads and writes the same PostgreSQL schema; the dashboard and n8n talk only to
the REST API; configuration comes from environment variables (`.env`), never hard-coded keys; each
record carries `data_origin` (`real:uci`, `real:olist`, `real:dataco`, `real:m5`, `simulated`) so real and
simulated data are never confused; every external API has a timeout, retry, cache and an explicit
"unavailable" state rather than invented values.

### 2. Folder structure

```
scip_platform/
├── docker-compose.yml · Dockerfile · .env.example · requirements.txt · README.md
├── docs/ARCHITECTURE.md            (this file) · docs/DATA_DICTIONARY.md
├── scip/
│   ├── core/          config (pydantic-settings), logging, db engine/session, ORM models
│   ├── ingestion/     uci.py, olist.py, dataco.py, m5.py, kaggle.py (kagglehub download),
│   │                  simulate.py (labelled simulation), pipeline.py (orchestration), samples.py
│   ├── integrations/  weather.py (Open-Meteo), countries.py (REST Countries), llm.py (Claude/OpenAI),
│   │                  n8n.py (webhooks), supplier_portal.py (mock supplier API)
│   ├── ml/            features.py, metrics.py, forecasting.py, stockout.py, delivery_risk.py, supplier_scoring.py
│   ├── optimization/  inventory.py, procurement.py (CP-SAT allocation), logistics.py (LP allocation + VRP)
│   ├── services/      kpis.py, control_tower.py, procurement_workflow.py, reports.py
│   ├── copilot/       rag.py (index + retrieve + answer)
│   ├── automation/    engine.py (jobs, rules, notifications, activity log)
│   └── api/           main.py + routers/ (health, data, forecasts, inventory, suppliers, procurement,
│                      logistics, copilot, automation, dashboard)
├── dashboard/         Streamlit Control Tower (app.py, api_client.py, pages/)
├── n8n/workflows/     importable n8n workflow JSON
├── scripts/           bootstrap.py (ingest → train → optimize → index → automate), download_datasets.py
├── tests/             pytest suite (unit, integration with PostgreSQL/SQLite, API, workflow)
├── data/raw/          downloaded datasets (not committed) · reports/ generated reports
```

### 3. Database schema (PostgreSQL)

| Table | Key | Purpose / key columns |
|---|---|---|
| data_sources | source_id | name, url, origin (real/simulated), row_count, loaded_at, notes |
| countries | country_code | name, region, subregion, lat, lng, currency, source (restcountries / static_fallback) |
| locations | location_id | type (warehouse / customer_region / supplier_site), city, state, country_code, lat, lng, capacity_units |
| products | sku | name, category, unit_cost, unit_price, is_planning_sku, data_origin |
| suppliers | supplier_id | name, country_code, city, state, lat, lng, data_origin |
| supplier_products | (supplier_id, sku) | unit_cost, moq, lead_time_days, lead_time_std, capacity_per_month |
| sales_daily | (date, sku, location_id) | quantity, revenue, data_origin |
| shipments | shipment_id | source, order_date, ship_mode, scheduled_days, actual_days, promised_date, delivered_date, status, is_late, supplier_id, origin/dest, distance_km, category, quantity, value |
| inventory_snapshots | (date, sku, location_id) | on_hand (simulated, driven by real demand) |
| inventory_current | (sku, location_id) | on_hand, on_order, as_of |
| purchase_orders | po_id | supplier_id, sku, qty, unit_cost, total_value, order_date, promised_date, received_date, status (draft → pending_approval → approved → sent → received / rejected / cancelled), approval_level, created_by |
| po_approvals | id | po_id, action, actor, comment, at |
| weather_daily | (location_key, date) | temp_max, temp_min, precipitation, source |
| model_runs | run_id | model_type, model_name, metrics (JSON), params (JSON), trained_at, is_champion |
| forecasts | (run_id, sku, date) | yhat, yhat_lower, yhat_upper, model |
| forecast_metrics | (run_id, sku, model) | mae, rmse, mape, wape |
| stockout_predictions | (scored_at, sku) | probability (14 days), risk_band |
| supplier_scores | (scored_at, supplier_id) | on_time_rate, avg_delay, predicted_risk, review_score, score, tier |
| inventory_policy | (computed_at, sku) | safety_stock, reorder_point, eoq, max_level, recommended_qty, days_cover, stockout_date, status |
| procurement_recommendations | id | sku, supplier_id, qty, cost, rationale, po_id |
| logistics_plans | plan_id | plan_type (allocation / routing), objective_value, details (JSON) |
| alerts | alert_id | severity, alert_type, entity, message, status |
| automation_log | id | job, action, status, details (JSON), notified |
| reports | report_id | report_type, title, html |
| rag_documents | doc_id | doc_type, entity, title, content |
| copilot_messages | id | question, answer, sources, provider |

### 4. Dataset acquisition and integration plan

| Dataset | Acquisition | Used for | Integration |
|---|---|---|---|
| UCI Online Retail (real) | `data/raw/uci/Online Retail.xlsx` (provided) | Demand history, products, prices, customer countries | Remove cancellations (InvoiceNo "C…"), non-product codes (POST, M, DOT, BANK CHARGES…), non-positive qty/price; aggregate to daily SKU × country; planning SKUs = top N by revenue |
| Olist Brazilian E-Commerce (real) | `kagglehub.dataset_download("olistbr/brazilian-ecommerce")` | Delivery tracking, late-delivery labels, seller (supplier) performance, review scores, customer geography for routing | orders + items + sellers + customers + reviews + products (+ geolocation) joined to `shipments`; sellers → `suppliers` |
| DataCo Smart Supply Chain (real) | `kagglehub.dataset_download("shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")` | Delivery-risk classification, shipping-mode performance, market / region logistics | `DataCoSupplyChainDataset.csv` (latin-1) → `shipments`; label `Late_delivery_risk`; leakage columns (actual days, delivery status) excluded from features |
| M5 Forecasting (real, optional) | Kaggle competition (login + rules acceptance) | Alternative demand source | Subset (one store) melted to daily SKU demand |
| Open-Meteo (API) | `archive-api.open-meteo.com` (history), `api.open-meteo.com` (16-day forecast) | Weather demand factors | Cached in `weather_daily`; used as features only when present |
| REST Countries (API) | `restcountries.com/v3.1/alpha` | Country region, coordinates, currency | Cached in `countries`; static fallback for the countries in the data |
| Claude / OpenAI (API) | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Copilot answers, report narratives | Optional; retrieval-only fallback without keys |
| Simulated (labelled) | `scip.ingestion.simulate` (seeded) | Inventory positions, supplier catalog for UCI SKUs, PO history, warehouses | Lead-time / delay distributions calibrated on real shipment data when available |

### 5. Machine learning

| Problem | Data | Models | Metrics | Validation |
|---|---|---|---|---|
| Demand forecasting 7/30/60/90 d | UCI daily demand (+ weather, UK holidays) | Moving average (baseline), Holt-Winters, Prophet, LightGBM, XGBoost (global, recursive) | MAE, RMSE, MAPE (non-zero days), WAPE | Time-based hold-out (last 56 days), recursive multi-step, no leakage |
| Stockout prediction (14 days) | Simulated inventory trajectories driven by real demand | Logistic regression, LightGBM | F1, ROC-AUC, precision, recall | Time-based split |
| Delivery-risk classification | DataCo + Olist shipments (real labels) | Logistic regression, LightGBM, XGBoost | F1, ROC-AUC, PR-AUC | Time-based split; leakage columns removed |
| Supplier performance scoring | Shipments + POs (+ Olist reviews) | Weighted scorecard + model-predicted risk | Rank stability, documented weights | — |

### 6. Optimization
* **Inventory policy**: z·√(L·σd² + d²·σL²) safety stock, ROP, EOQ, min/max, using forecast demand.
* **Procurement allocation** (OR-Tools CP-SAT): split required quantity across candidate suppliers minimising
  cost + lead-time + risk penalty, subject to MOQ, capacity and budget.
* **Warehouse allocation** (OR-Tools linear solver): transportation model assigning regional demand to warehouses
  under capacity, minimising distance-weighted cost.
* **Delivery routing** (OR-Tools routing): capacitated VRP for a day's deliveries from a hub.

### 7. Automation and approvals
Rules engine (scheduled via n8n cron or API): low-stock and stockout-risk alerts → procurement recommendation →
draft PO → approval level by value (auto ≤ $2,500, manager ≤ $25,000, director above) → notifications posted to
n8n webhooks (e-mail / Slack) → approve / reject via API → PO sent to (mock) supplier portal → activity log.

### 8. Implementation roadmap
1. Architecture & schema (this document) — core config, logging, DB.
2. Ingestion: UCI (real), Olist & DataCo loaders, M5 loader, simulation; data dictionary.
3. Integrations: Open-Meteo, REST Countries (cached, fault-tolerant), LLM providers, n8n client.
4. ML: forecasting suite, stockout classifier, delivery-risk classifier, supplier scoring; model registry.
5. Optimization: inventory policy, procurement CP-SAT, warehouse LP, VRP.
6. Services: KPIs, control tower, procurement workflow, reports.
7. Copilot (RAG) and automation engine.
8. FastAPI backend.
9. Streamlit Control Tower over the API.
10. Docker, n8n workflows, tests, README; end-to-end verification.
