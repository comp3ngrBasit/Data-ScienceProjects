# ⚡ Energy Consumption & Grid Intelligence Analytics Platform

An end-to-end data-science platform that analyses **real electricity data**: one French household
(UCI, minute resolution, 2006–2010) and the **German transmission grid** (Open Power System Data / ENTSO-E,
hourly, 2015–2020). The project covers forecasting, peak-load intelligence, segmentation, anomaly detection,
renewable integration, weather sensitivity, Bayesian modelling and hypothesis testing. Results are served by
a **FastAPI** backend and an 11-page dark-mode **Streamlit** dashboard, and exported as PDF/CSV reports.

Every number in the dashboard, API and report is read from files written by the pipeline. Nothing is hard-coded.

---

## Headline results (from the current pipeline run)

| Area | Result |
|---|---|
| Household | 37.6 MWh over 1,440 days; 26.1 kWh/day mean (CV 0.38); max hourly demand 6.56 kWh (2008-11-23 18:00) |
| German grid | 2,797 TWh load; peak 77.5 GW (2017-12-13 17:00); wind + solar = **29.0 %** of load (2019: 33.8 %) |
| 24 h forecast, grid | **XGBoost WAPE 1.90 %**, 49 % better than seasonal naive; the TSOs' published day-ahead forecast scores 2.58 % on the same days (12 days only, and the TSO forecast is issued earlier, so treat this as indicative) |
| 24 h forecast, household | LSTM WAPE 47.7 % (22 % better than naive). A single home's hourly load is highly stochastic |
| 30-day forecast, household | Random Forest WAPE 18.6 %, 35 % better than naive |
| Anomalies | 171 consensus hours (≥ 2 of 6 detectors): 16 critical, 35 high; 19 data-quality findings |
| Segmentation | 6 day-profile clusters (bootstrap stability ARI 0.84), e.g. a 24-day low-use cluster concentrated in August |
| Weather | +4.89 GWh/day per heating degree-day (95 % CI 3.31–6.47); weather adds 16 pp of R² beyond calendar effects |
| Bayesian | PyMC Student-t regression, R̂ ≤ 1.004, 0 divergences, 89 % coverage of the 90 % predictive interval |
| Hypothesis tests | 6 of 7 significant after Holm correction; hot vs moderate summer days **not** significant in Germany (p = 0.11) |

---

## Data sources and provenance

| Dataset | Use | Notes |
|---|---|---|
| [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | household modules | 2,075,259 minutes; 25,979 missing (both match the documentation) |
| [Open Power System Data, time series 60 min, 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06) | grid, renewables, regional | DE load, TSO forecast, wind on/offshore, solar, capacities; 31 countries for regional views |
| [Open-Meteo archive (ERA5)](https://open-meteo.com/) | weather (primary) | household point + population-weighted German points |
| [OPSD weather package 2019-04-09](https://data.open-power-system-data.org/weather_data/2019-04-09) | weather (fallback) | DE temperature, wind, radiation 2013–2016 |

**Acquisition and integrity.** `app/data/ingestion.py` downloads from the official URL first. Mirrors are
accepted only when their **SHA-256 matches the official release** (hashes in `app/config.py`).

**Build environment note.** The environment used to build this release could not reach the UCI, OPSD or
Open-Meteo hosts. The UCI and OPSD files were obtained from GitHub mirrors and verified by SHA-256 against the
official releases. Weather came from the OPSD weather package (via a public extract), which has **no humidity
or precipitation**. Because of that, household weather sensitivity is not computed in this build. On a machine
with normal internet access, the pipeline uses Open-Meteo automatically: it adds humidity and precipitation,
and the household weather analysis runs.

> The household (France, 2006–2010) and the grid (Germany, 2015–2020) differ in geography, period, units and
> definitions. They are **analysed independently and never merged**.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: small CPU wheel
pip install -r requirements.txt

python -m app.pipeline run            # download, validate, model, report (cached; ~30 min first time)
streamlit run app/dashboard/streamlit_app.py     # http://localhost:8501
uvicorn app.main:app --port 8000                 # http://localhost:8000/docs
pytest -q                                        # 39 unit + integration tests
```

Useful variants:

```bash
python -m app.pipeline status
python -m app.pipeline run --stage forecast --force     # retrain one stage
EGI_FAST=1 python -m app.pipeline run --force           # quick smoke run (fewer folds / MCMC draws)
```

**Docker** (PostgreSQL + API + dashboard):

```bash
cp .env.example .env
docker compose run --rm pipeline          # build data, models, reports into mounted volumes
docker compose up -d --build db api dashboard
```

With `DATABASE_URL` set, every result table is also mirrored into PostgreSQL. Parquet remains the primary store.

---

## Architecture

```
app/
  config.py                 settings, sources + SHA-256, env vars
  pipeline.py               orchestrator / CLI, stage caching by input fingerprint
  main.py                   FastAPI app
  reporting.py              PDF (ReportLab) + CSV export
  data/                     ingestion · validation · cleaning · preprocessing · feature_engineering
  analytics/                consumption · timeseries_stats · peak_analysis · renewable_analysis
                            weather_analysis · statistical_tests · explainability
  models/forecasting/       8 models behind one interface + walk-forward engine + LSTM
  models/clustering/        segmentation (K-Means, Ward, GMM, DBSCAN, PCA)
  models/anomaly_detection/ 6 detectors + consensus + attribution
  models/bayesian/          PyMC regression
  models/evaluation/        metrics, residual diagnostics, PSI drift
  api/routes, api/schemas   REST contract (ready for a React frontend)
  dashboard/                11 Streamlit pages + theme + data client (API or local)
data/{raw,processed,external}   models/   reports/{results,exports}   tests/   configs/
```

Pipeline stages: `data → eda → forecast → forward → peak → segmentation → anomaly → renewable → weather
→ statistics → explain → kpis → report`. Each stage writes Parquet/JSON to `reports/results/<module>/`. The API
and dashboard only read these files and never retrain on request.

---

## Scientific methodology

**Data engineering.** Minute data are read with Polars, validated (row count, duplicates, ranges, sub-meter
consistency, P ≤ V·I with 0.2 A rounding), and aggregated to hours with coverage-aware scaling (≥ 75 % valid
minutes). Short gaps (≤ 3 h) are interpolated. Longer gaps use same-hour medians of the previous four weeks.
All imputed hours are flagged (1.26 %) and excluded from anomaly scoring and segmentation. Grid data use UTC
storage and are converted to Europe/Berlin for calendar analyses.

**Forecasting (Module 1).** Models: Seasonal naive (baseline), Linear Regression, Random Forest, XGBoost,
SARIMA, SARIMAX (calendar + weekly Fourier exogenous terms), Prophet and a 2-layer LSTM. Horizons: 24 h and
7 days (hourly) and 30 days (daily).
* Chronological split 75 / 10 / 15 %. No shuffling.
* **Horizon-safe direct features:** every lag is ≥ the horizon, so no feature uses data after the forecast
  origin. This is unit-tested by perturbing future values.
* Hyper-parameters are chosen on the validation block: grid search for RF/XGB, AIC for SARIMA, changepoint
  prior for Prophet, early stopping for the LSTM.
* Rolling-origin **walk-forward** test (12 / 6 / 6 origins). Metrics are pooled across origins: MAE, RMSE,
  MAPE (zeros excluded), WAPE, R², bias, 90 % interval coverage and width, and skill vs naive.
* Intervals are native for SARIMA/Prophet and split-conformal otherwise, truncated at 0 for consumption.
* Residual diagnostics (Ljung-Box, Durbin-Watson, Jarque-Bera); drift monitoring (target PSI, per-origin error trend).
* The ENTSO-E TSO day-ahead forecast is an external benchmark. MLflow logs every model run.

**Peak load (2).** Daily, weekly and monthly peaks; peak-to-average ratio and load factor; data-driven
household peak window (19–23 h) vs EEX grid peak (08–20 Mon–Fri); P99 high-demand events; STL of daily peaks;
**GPD peaks-over-threshold** with declustering and bootstrapped return levels.

**Segmentation (3).** The data cover one household, so the unit is the **household-day**. Eight profile
features; K-Means (k chosen by silhouette), Ward, GMM (BIC) and DBSCAN (k-distance ε); silhouette,
Davies-Bouldin and Calinski-Harabasz scores plus bootstrap ARI stability. Clusters are named from centroid
z-scores and a binomial weekend test. A regional view clusters 30 European grids by load shape.

**Anomalies (4).** The causal baseline is the same-hour-of-week median over the previous four weeks.
Detectors: rolling z-score, IQR fence, weekly-STL residual (each with a 0.5 kWh practical-significance floor),
Isolation Forest, LOF and One-Class SVM. Consensus requires at least two detectors. Each anomaly carries its
timestamp, observed and expected values, deviation %, severity, methods, evidence and driving sub-meter.
Daily-total and volatility detectors and meter data-quality checks (outages, EN 50160 voltage band) run
alongside. Anomalies are **never** presented as proof of theft or failure.

**Renewables (5).** VRE = wind + solar. Penetration is VRE energy divided by load energy over identical hours.
OPSD has no hydro, biomass or conventional series, so the remainder is reported as **residual load**, not
"non-renewable generation". The module also covers capacity factors, ramps, STL, load–generation correlation,
a HAC regression of residual load, low-VRE periods (< 10 % for ≥ 24 h), and a 2016–2019 European comparison.

**Weather (6).** Daily load vs population-weighted weather with HDD/CDD (15.5 / 22 °C). Pearson and Spearman
correlations with Fisher CIs, on all days and on working days only. Models: OLS with Newey-West SEs,
polynomial, **GAM** (penalised B-splines, GCV), Random Forest, XGBoost (SHAP, PDP, permutation importance)
and Bayesian regression. Seasonal slopes and the R² added by weather are also reported.
Associations, not causal effects.

**Statistics.** STL strength, ACF/PACF, ADF + KPSS at level and after differencing, PELT change points,
rolling statistics, Mann-Kendall trend. Seven hypothesis tests report the statistic, p-value, Holm-adjusted p,
Cohen's d, rank-biserial r and a **moving-block-bootstrap CI** robust to serial correlation.

**Bayesian.** PyMC Student-t regression with weakly informative priors on standardised predictors. Reports
prior-vs-posterior distributions, R̂ and ESS, coefficient credible intervals, and posterior-predictive
forecasts with 50 % and 90 % coverage.

**Reproducibility.** Fixed seeds (`RANDOM_SEED = 42`), deterministic PyTorch, versioned settings, stage
fingerprints, and serialized models (`models/forecasting/*.joblib|.pt`).

---

## Limitations

* One household: its findings describe that home, not a population.
* Forecast models use only calendar features and past load. Observed weather is not used as an input, which
  would leak information that a weather forecast could not provide.
* ML/LSTM/SARIMA parameters are frozen at the start of the test block, with state updated at each origin.
  Prophet is refitted at each origin. Scheduled retraining would help under drift.
* The German test block (Nov 2019 – Sep 2020) contains the **COVID-19 demand shock**.
* Weather for 2015–2016 only (OPSD fallback); humidity and precipitation are unavailable without Open-Meteo.
* Short-gap interpolation uses the next observed value (≤ 3 h; negligible scale).
* Trend p-values are descriptive. Return levels assume stationarity.

## API

| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | stage status |
| `GET /api/v1/kpis` | executive KPIs |
| `GET /api/v1/modules` | available result tables/documents |
| `GET /api/v1/results/{module}/{name}` | paginated table (`offset`, `limit`, `columns`) |
| `GET /api/v1/results/{module}/{name}/json` | result document |
| `GET /api/v1/forecasts/{key}` | forecast summary + metrics |
| `GET /api/v1/anomalies?severity=` | consensus anomalies |
| `GET /api/v1/reports/pdf`, `/reports/csv` | report downloads |
| `POST /api/v1/pipeline/run` | trigger stages (requires `X-API-Key` = `EGI_API_KEY`) |

## Citation

Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI ML Repository.
Open Power System Data (2020). *Data Package Time series*, version 2020-10-06. Open Power System Data (2019).
*Data Package Weather Data*, version 2019-04-09.
