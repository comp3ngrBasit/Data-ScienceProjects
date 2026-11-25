"""Central configuration. Every tunable lives here or in environment variables.

Environment variables (all optional):
    EGI_DATA_DIR       root data directory (default: <project>/data)
    EGI_MODELS_DIR     serialized models (default: <project>/models)
    EGI_REPORTS_DIR    analytical outputs and reports (default: <project>/reports)
    EGI_GRID_COUNTRY   ISO code of the grid analysed in depth (default: DE)
    EGI_API_URL        if set, the dashboard reads results through the FastAPI backend
    DATABASE_URL       SQLAlchemy URL (e.g. postgresql+psycopg://...) to mirror tables into
    MLFLOW_TRACKING_URI MLflow tracking store (default: sqlite:///<project>/mlflow.db)
    EGI_FAST           "1" shrinks walk-forward folds / MCMC draws for quick smoke runs
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(env: str, default: Path) -> Path:
    p = Path(os.getenv(env, str(default)))
    p.mkdir(parents=True, exist_ok=True)
    return p


DATA_DIR = _path("EGI_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = _path("EGI_RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR = _path("EGI_PROCESSED_DIR", DATA_DIR / "processed")
EXTERNAL_DIR = _path("EGI_EXTERNAL_DIR", DATA_DIR / "external")
MODELS_DIR = _path("EGI_MODELS_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = _path("EGI_REPORTS_DIR", PROJECT_ROOT / "reports")
RESULTS_DIR = _path("EGI_RESULTS_DIR", REPORTS_DIR / "results")
LOG_DIR = _path("EGI_LOG_DIR", PROJECT_ROOT / "logs")

RANDOM_SEED = 42
FAST_MODE = os.getenv("EGI_FAST", "0") == "1"
GRID_COUNTRY = os.getenv("EGI_GRID_COUNTRY", "DE")
API_URL = os.getenv("EGI_API_URL", "").rstrip("/")
DATABASE_URL = os.getenv("DATABASE_URL", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")


@dataclass(frozen=True)
class Source:
    """A downloadable dataset: official URL first, then integrity-checked mirrors."""

    name: str
    filename: str
    urls: tuple[str, ...]
    sha256: str | None = None
    citation: str = ""


UCI_SOURCE = Source(
    name="UCI Individual Household Electric Power Consumption",
    filename="household_power_consumption.zip",
    urls=(
        "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        # GitHub LFS mirror; accepted only if the SHA-256 below matches the official archive
        "https://media.githubusercontent.com/media/2417-droid/Minor_Project/HEAD/household-load-forecasting/data/raw/household_power_consumption.zip",
    ),
    sha256="9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff",
    citation="Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption. UCI ML Repository. https://doi.org/10.24432/C58K54",
)

OPSD_SOURCE = Source(
    name="Open Power System Data - Time series (60 min, single index), version 2020-10-06",
    filename="time_series_60min_singleindex.csv",
    urls=(
        "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv",
        "https://media.githubusercontent.com/media/rohith0110/Applied_Timeseries_OPSD_Analysis/HEAD/data/time_series_60min_singleindex.csv",
    ),
    sha256="6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8",
    citation="Open Power System Data (2020). Data Package Time series, version 2020-10-06. https://doi.org/10.25832/time_series/2020-10-06",
)

# Germany-level weather extracted from the OPSD weather package (2019-04-09; MERRA-2 via
# renewables.ninja, population-weighted country averages), columns DE_temperature,
# DE_windspeed_10m, DE_radiation_direct_horizontal, DE_radiation_diffuse_horizontal for 2013-2016.
OPSD_WEATHER_SOURCE = Source(
    name="OPSD Weather Data package 2019-04-09 (Germany extract 2013-2016)",
    filename="opsd_weather_germany_2013_2016.csv",
    urls=("https://raw.githubusercontent.com/Samreay/DSGoPipeline/HEAD/1_starting_point/germany.csv",),
    sha256=None,
    citation="Open Power System Data (2019). Data Package Weather Data, version 2019-04-09. https://doi.org/10.25832/weather_data/2019-04-09",
)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "precipitation",
    "weather_code",
)

# Household is in Sceaux (Hauts-de-Seine, France), 7 km from Paris (UCI dataset description).
HOUSEHOLD_LOCATION = {"name": "Sceaux, FR", "latitude": 48.778, "longitude": 2.290, "timezone": "Europe/Paris"}

# Population weights (millions, approx. city populations) used to build a population-weighted
# national weather series from Open-Meteo point data, mirroring the OPSD weather methodology.
GRID_WEATHER_POINTS = {
    "DE": [
        ("Berlin", 52.52, 13.40, 3.6),
        ("Hamburg", 53.55, 9.99, 1.8),
        ("Munich", 48.14, 11.58, 1.5),
        ("Cologne", 50.94, 6.96, 1.1),
        ("Frankfurt", 50.11, 8.68, 0.75),
        ("Stuttgart", 48.78, 9.18, 0.63),
        ("Duesseldorf", 51.23, 6.78, 0.62),
        ("Leipzig", 51.34, 12.37, 0.6),
        ("Dortmund", 51.51, 7.47, 0.59),
        ("Hanover", 52.38, 9.73, 0.54),
    ],
}

COUNTRY_TZ = {"DE": "Europe/Berlin", "FR": "Europe/Paris", "AT": "Europe/Vienna", "BE": "Europe/Brussels",
              "NL": "Europe/Amsterdam", "ES": "Europe/Madrid", "IT": "Europe/Rome", "DK": "Europe/Copenhagen",
              "GB_GBN": "Europe/London", "PL": "Europe/Warsaw", "CZ": "Europe/Prague", "PT": "Europe/Lisbon"}


@dataclass
class ForecastConfig:
    """Walk-forward evaluation design (all horizons in steps of the series frequency)."""

    n_folds_24h: int = 4 if FAST_MODE else 12
    n_folds_7d: int = 2 if FAST_MODE else 6
    n_folds_30d: int = 2 if FAST_MODE else 6
    fold_spacing_hours: int = 168 * 2
    test_fraction: float = 0.15
    val_fraction: float = 0.10
    sarima_train_hours: int = 24 * 42  # SARIMA refit on a trailing window to keep it tractable
    lstm_epochs: int = 3 if FAST_MODE else 12
    interval_alpha: float = 0.10  # 90 % prediction intervals
    models: tuple[str, ...] = ("SeasonalNaive", "LinearRegression", "RandomForest", "XGBoost",
                               "SARIMA", "SARIMAX", "Prophet", "LSTM")


@dataclass
class BayesConfig:
    draws: int = 300 if FAST_MODE else 1000
    tune: int = 300 if FAST_MODE else 1000
    chains: int = 2
    target_accept: float = 0.9


@dataclass
class Settings:
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    bayes: BayesConfig = field(default_factory=BayesConfig)
    hdd_base_c: float = 15.5  # Eurostat-style heating threshold
    cdd_base_c: float = 22.0
    peak_hours: tuple[int, int] = (17, 21)  # inclusive start, exclusive end, local time
    night_hours: tuple[int, int] = (0, 6)
    min_hour_coverage: float = 0.75  # minimum share of valid minutes for an hourly value


SETTINGS = Settings()
