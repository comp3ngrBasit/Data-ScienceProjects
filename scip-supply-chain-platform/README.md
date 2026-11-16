# SCIP — AI-Powered Supply Chain Intelligence & Autonomous Operations Platform

One integrated application that goes from **raw data ingestion → predictions → optimised decisions → automated
business actions**: demand forecasting, inventory optimisation, supplier intelligence, procurement automation with
approvals, logistics optimisation (OR-Tools), a RAG AI Copilot, an automation engine with n8n workflows, a FastAPI
backend and a Streamlit Control Tower — all on one PostgreSQL database.

> **Real vs simulated data.** Demand history is **real** (UCI Online Retail), shipment/delivery outcomes are **real**
> (DataCo), Brazil customer geography is **real** (Olist), event calendar is **real** (M5). What public data does not
> contain — inventory positions, vendors, vendor catalog, purchase orders, warehouses — is **simulated**, seeded,
> calibrated on the real data and labelled `data_origin='simulated'` everywhere.

Architecture, schema, dataset plan and roadmap: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
Lineage and assumptions: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)

---

## 1. Quick start

### Option A — Docker (recommended on Windows: Docker Desktop)
```bash
cp .env.example .env                      # add ANTHROPIC_API_KEY (optional) etc.
python scripts/prepare_data.py --uci "<path>/online+retail.zip" --dataco "<path>/DataCoSupplyChainDataset.csv.zip" \
    --olist "<path>/olist zip or folder" --m5 "<path>/calendar.csv"
docker compose up --build
```
The API container bootstraps the platform on first start (≈ 2–3 minutes), then:
* Control Tower: http://localhost:8501
* REST API + OpenAPI docs: http://localhost:8000/docs
* n8n: http://localhost:5678 → import the three files in `n8n/workflows/`, add SMTP credentials to the e-mail nodes, activate.

### Option B — local Python 3.11+ and PostgreSQL 16
```bash
python -m venv .venv && .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
createdb scip                                           # or create it in pgAdmin
cp .env.example .env                                    # set DATABASE_URL
python scripts/prepare_data.py --uci ... --dataco ... --olist ... --m5 ...   # or: python scripts/download_datasets.py
python scripts/bootstrap.py                             # ingest -> train -> optimise -> automate (~2 min)
uvicorn scip.api.main:app --port 8000                   # terminal 1
streamlit run dashboard/app.py                          # terminal 2 (SCIP_API_URL defaults to localhost:8000)
```
Windows shortcut: `powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1`.

### Tests
```bash
pytest -q          # 31 tests: 16 unit + 15 integration (integration needs PostgreSQL + data/raw files)
```
Integration tests build the full platform into a separate `scip_test` database (20 SKUs) and never touch production data.

---

## 2. What each module does

| Module | Implementation | Key output |
|---|---|---|
| **Data ingestion** (`scip/ingestion`) | UCI, DataCo, Olist, M5 loaders with cleaning rules; seeded simulation calibrated on real shipments; REST Countries + Open-Meteo with cache and fallback | Unified PostgreSQL schema with lineage |
| **Demand forecasting** (`scip/ml/forecasting.py`) | Moving Average, Holt-Winters, Prophet (UK holidays), LightGBM, XGBoost (global *direct* multi-horizon models); 13-week time-based hold-out | 7/30/60/90-day forecasts with 80% ranges, MAE/RMSE/MAPE/WAPE |
| **Stockout prediction** (`scip/ml/stockout.py`) | Logistic Regression vs LightGBM on daily inventory state | P(stockout within 14 days) per SKU, F1/ROC-AUC |
| **Supplier delivery risk** (`scip/ml/delivery_risk.py`) | Logistic Regression, LightGBM, XGBoost on real DataCo outcomes, chronological split, leakage columns excluded | Late-delivery probability on every open PO |
| **Supplier scoring** (`scip/ml/supplier_scoring.py`) | Weighted scorecard incl. model-predicted risk | Score 0–100, tier |
| **Inventory optimisation** (`scip/optimization/inventory.py`) | SS = z·√(L·σd² + d²·σL²), ROP, EOQ, min/max, ABC service levels; demand from champion forecast | Policy + recommended quantities |
| **Procurement automation** (`scip/optimization/procurement.py`, `scip/services/procurement_workflow.py`) | Supplier ranking by expected total cost; OR-Tools CP-SAT allocation (MOQ, capacity, budget, risk); PO lifecycle draft → approval (auto/manager/director) → sent (mock supplier portal) | POs, approval history, notifications |
| **Logistics optimisation** (`scip/optimization/logistics.py`) | OR-Tools LP warehouse allocation (EU from real UCI demand, Brazil from real Olist customers); OR-Tools capacitated VRP | Flows, savings vs single-DC baseline, routes |
| **AI Copilot** (`scip/copilot/rag.py`) | RAG: 170+ documents generated from the DB, TF-IDF retrieval, Claude (default) or OpenAI generation with citations; retrieval-only fallback without a key | Grounded answers with sources |
| **Automation engine** (`scip/automation/engine.py`) | Policy refresh → risk scoring → alerts → CP-SAT procurement → draft/submit POs → daily report → Copilot re-index → n8n events | Alerts, POs, reports, activity log |
| **REST API** (`scip/api`) | FastAPI, 40+ endpoints, background jobs, optional `X-API-Key` on mutations, request logging, JSON-safe errors | `/docs` |
| **Control Tower** (`dashboard/`) | Streamlit, talks only to the API: KPIs, forecast charts, inventory risk heatmap, delivery tracking, supplier performance, PO approval queue, logistics maps, Copilot chat, automation log, reports, model registry | http://localhost:8501 |
| **n8n** (`n8n/workflows`) | Daily cron → `/automation/run` → e-mail on critical alerts; event router (approval requests, decisions, cycle summaries); e-mail approve/reject links → API | Workflow automation |

## 3. Results on the real data (this build)

**Demand forecasting** — 150 UCI planning SKUs, hold-out = last 13 weeks (Sep 9 – Dec 9 2011), daily units:

| Model | MAE | RMSE | MAPE* | WAPE | Weekly WAPE |
|---|---|---|---|---|---|
| Moving Average (baseline) | 32.6 | 99.7 | 185% | 91.7% | **58.4%** |
| Holt-Winters | 31.4 | 98.7 | 247% | 88.3% | 60.5% |
| Prophet | 32.4 | 107.9 | 266% | 91.2% | 63.4% |
| LightGBM | **29.3** | 99.7 | **163%** | **82.4%** | 64.6% |
| **XGBoost (champion)** | 29.7 | **97.7** | 205% | 83.7% | 60.2% |

\*MAPE over days with non-zero sales; UCI is lumpy B2B wholesale demand, so percentage errors are large.
The champion is chosen by average rank over all five metrics. Honest reading: the ML models clearly win on
day-level error, but **no model beats the moving average on weekly totals** over this hold-out, which spans the
autumn ramp-up that a single year of history cannot teach. With more history (or M5 data) the yearly-seasonality
features become learnable.

**Delivery-risk classification** — real DataCo, most recent 12,580 orders as test set (57% late):
Logistic Regression (champion) ROC-AUC 0.746, PR-AUC 0.837, precision 0.89, recall 0.54, F1 0.67;
LightGBM ROC-AUC 0.741; XGBoost 0.740. The naive "always late" rule has F1 0.73 but no discrimination, so ROC-AUC
selects the champion.

**Stockout prediction (14 days)** — LightGBM ROC-AUC 0.955, PR-AUC 0.869, F1 0.77 (precision 0.87, recall 0.69)
on the last 90 days. Trained on simulated inventory positions driven by real demand.

**Logistics** — EU warehouse allocation saves 9.9% vs serving everything from the UK DC; Brazil hub allocation
saves 25.8%; the 5-vehicle Brazil line-haul plan covers all 26 non-hub states.

## 4. Configuration (`.env`)
All settings are environment variables (see `.env.example`); no keys are hard-coded.
`DATABASE_URL`, `LLM_PROVIDER` (anthropic | openai | none), `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`N8N_WEBHOOK_URL`, `API_KEY` (protects mutating endpoints), `AUTO_APPROVE_LIMIT`, `MANAGER_APPROVE_LIMIT`,
`PLANNING_SKU_COUNT`, `OLIST_DIR` / `DATACO_DIR` / `M5_DIR`, `ENABLE_EXTERNAL_APIS`.

## 5. Integrations — what is live and what is mocked
| Integration | Status |
|---|---|
| PostgreSQL, FastAPI, Streamlit, OR-Tools, scikit-learn, XGBoost, LightGBM, Prophet | Implemented and tested |
| Open-Meteo, REST Countries | Implemented with retries, caching and labelled fallback; in the build sandbox these hosts were blocked, so the REST Countries static fallback was used and weather features were skipped — on a normal network they are fetched automatically |
| Claude / OpenAI | Implemented; activates when a key is set. Without a key the Copilot answers retrieval-only |
| n8n | Platform → n8n webhook events tested end-to-end against a live HTTP receiver; workflow JSON included. The n8n server itself could not be installed in the build sandbox, so import the workflows and set SMTP credentials in your n8n |
| Supplier portal | **Mock** (no supplier credentials exist); replace `MockSupplierPortal` with a real connector |

## 6. Limitations
* One year of UCI history limits yearly seasonality learning; forecasts post-Christmas are driven by state and calendar features, not by a prior-year pattern.
* Delivery-risk models transfer from customer shipments (DataCo) to inbound vendor POs; treat PO-level risk as indicative.
* The full Olist dataset (orders, items, sellers, reviews) enables real seller performance scoring; with only the customers file the supplier scorecard uses simulated vendor POs.
* Plotly map outlines load from the Plotly CDN; on an offline machine the maps render without country outlines.
