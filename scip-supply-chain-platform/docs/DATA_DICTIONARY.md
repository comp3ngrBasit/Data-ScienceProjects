# Data dictionary and lineage

Every business row carries `data_origin`: `real:uci`, `real:dataco`, `real:olist`, `real:m5` or `simulated`.

## Raw sources → tables

| Source (origin) | Raw file(s) | Rows (this build) | Loaded into | Key transformations |
|---|---|---|---|---|
| UCI Online Retail (real) | `Online Retail.xlsx` | 541,909 → 522,502 clean | `products` (3,791), `sales_daily` (302,585), `locations` (customer countries) | Removed 9,288 cancellations (InvoiceNo "C…"), 2,517 non-positive qty/price, 2,379 non-product codes (POST, DOT, M, fees, samples), 5,223 exact duplicates; daily SKU × country aggregation; top 150 SKUs by revenue with ≥ 240 days of life and ≥ 100 selling days = planning SKUs |
| DataCo Smart Supply Chain (real) | `DataCoSupplyChainDataset.csv` (latin-1) | 180,519 lines → 65,752 shipments | `shipments` | PII columns dropped on read (email, password, names, street); one shipment per Order Id; promised = order + scheduled days; delivered = order + real days; `is_late` = Late_delivery_risk; 2,855 cancelled → `is_late` NULL |
| Olist (real) | `olist_customers_dataset.csv` (others optional) | 99,441 customers | `locations` (27 `BR-CUST-*` state demand points) | Unique customers per state as demand weight; state-capital coordinates. With orders/items/sellers files: shipments, sellers as suppliers, review scores |
| M5 calendar (real) | `calendar.csv` | 1,969 days | forecasting features (not stored) | Event flags (National, Religious, Cultural, Sporting) |
| REST Countries (API) | — | 40 countries | `countries` | `source` = `restcountries` or `static_fallback` when the API is unreachable |
| Open-Meteo (API) | — | 0 here (API blocked in build environment) | `weather_daily` | Used as features only when present; future dates use previous-year values (no leakage) |
| Simulation (simulated) | `scip/ingestion/simulate.py`, seed 42 | 56,100 snapshots, 1,672 historical POs | `suppliers` (10 vendors), `supplier_products`, `locations` (6 warehouses), `inventory_*`, `purchase_orders` | Replenishment policy run against REAL UCI demand; vendor lateness calibrated on REAL DataCo late rates by ship mode |

## Assumptions (all labelled in the UI)
* UCI has no cost data: `unit_cost` = 55% of median selling price.
* UCI has no categories: derived from description keywords (`scip/ingestion/uci.py`).
* Vendors, catalog prices, MOQs, capacities, warehouse locations/capacities and transport rates are simulated.
* DataCo's First Class mode is late on 100% of orders in the source data; vendors using that mode inherit high predicted risk.
* Brazil route demand: customers per state / 100 = pallets.

## Key tables
See `docs/ARCHITECTURE.md` §3 and `scip/core/models.py` (authoritative column definitions).
