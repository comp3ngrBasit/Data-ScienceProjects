"""Application settings, loaded from environment variables / .env (never hard-coded secrets)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SCIP — Supply Chain Intelligence Platform"
    environment: str = "development"
    log_level: str = "INFO"

    # Database: PostgreSQL in deployment; SQLite accepted for quick local tests
    database_url: str = "postgresql+psycopg2://scip:scip@localhost:5432/scip"

    # Data
    data_dir: Path = BASE_DIR / "data"
    reports_dir: Path = BASE_DIR / "reports"
    models_dir: Path = BASE_DIR / "data" / "models"
    random_seed: int = 42
    planning_sku_count: int = 150          # top-N UCI SKUs by revenue used for planning
    olist_dir: Path | None = None           # folder with Olist CSVs (kagglehub path)
    dataco_dir: Path | None = None          # folder with DataCo CSV (kagglehub path)
    m5_dir: Path | None = None
    use_samples_when_missing: bool = True   # labelled SIMULATED samples when Olist/DataCo files are absent

    # External APIs
    open_meteo_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    restcountries_url: str = "https://restcountries.com/v3.1"
    http_timeout_seconds: float = 10.0
    enable_external_apis: bool = True

    # LLM (Copilot)
    llm_provider: str = "anthropic"         # anthropic | openai | none
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Automation / n8n
    n8n_webhook_url: str | None = None      # e.g. http://n8n:5678/webhook/scip-events
    api_base_url: str = "http://localhost:8000"
    auto_approve_limit: float = 2_500.0
    manager_approve_limit: float = 25_000.0

    # Inventory policy
    service_level_a: float = 0.98
    service_level_b: float = 0.95
    service_level_c: float = 0.90
    holding_cost_rate: float = 0.25
    ordering_cost: float = 50.0

    # Forecasting
    forecast_horizons: tuple[int, ...] = (7, 30, 60, 90)
    validation_days: int = 91
    api_key: str | None = Field(default=None, description="Optional key required on mutating API calls")


@lru_cache
def get_settings() -> Settings:
    return Settings()
