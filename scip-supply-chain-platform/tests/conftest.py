"""Test configuration.

Unit tests run anywhere. Integration tests (marked `integration`) need PostgreSQL and the raw datasets in
data/raw/; they build the full platform into a separate test database (TEST_DATABASE_URL, default
postgresql+psycopg2://postgres@localhost:5432/scip_test) with 20 planning SKUs, external APIs disabled and
model files written to a temporary folder — production data and models are never touched.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="scip_test_"))
TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql+psycopg2://postgres@localhost:5432/scip_test")
os.environ.update({
    "DATABASE_URL": TEST_DB, "PLANNING_SKU_COUNT": "20", "ENABLE_EXTERNAL_APIS": "false", "LLM_PROVIDER": "none",
    "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", "N8N_WEBHOOK_URL": "", "API_KEY": "",
    "MODELS_DIR": str(_TMP / "models"), "REPORTS_DIR": str(_TMP / "reports"),
})


def _postgres_available() -> bool:
    try:
        import psycopg2
        admin = TEST_DB.rsplit("/", 1)[0].replace("postgresql+psycopg2", "postgresql") + "/postgres"
        conn = psycopg2.connect(admin)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname='scip_test'")
            if not cur.fetchone():
                cur.execute("CREATE DATABASE scip_test")
        conn.close()
        return True
    except Exception:
        return False


def _raw_data_available() -> bool:
    raw = ROOT / "data" / "raw"
    return any((raw / "uci").glob("*.xlsx")) and any((raw / "dataco").glob("*.csv"))


INTEGRATION_READY = _postgres_available() and _raw_data_available()


def pytest_collection_modifyitems(config, items):
    if INTEGRATION_READY:
        return
    skip = pytest.mark.skip(reason="integration tests need PostgreSQL + raw datasets in data/raw")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def platform():
    """Bootstrap the whole platform once into the test database."""
    from scip.core.db import get_engine
    from scip.services.bootstrap import bootstrap
    out = bootstrap(get_engine(), reset=True)
    return out
