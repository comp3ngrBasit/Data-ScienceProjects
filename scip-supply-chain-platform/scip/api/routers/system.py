from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from scip.api.deps import Jobs, records, require_api_key
from scip.core.config import get_settings
from scip.core.db import get_meta, read_sql, table_counts
from scip.integrations.llm import provider_status

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    try:
        read_sql("SELECT 1 AS ok")
        db = "ok"
    except Exception as exc:
        db = f"error: {exc}"
    return {"status": "ok" if db == "ok" else "degraded", "database": db}


@router.get("/api/v1/status")
def status():
    s = get_settings()
    return {"app": s.app_name, "environment": s.environment, "as_of_date": get_meta("as_of_date"),
            "ingested_at": get_meta("ingested_at"), "bootstrapped_at": get_meta("bootstrapped_at"),
            "table_counts": table_counts(), "llm": provider_status(),
            "integrations": {"n8n_webhook_configured": bool(s.n8n_webhook_url), "external_apis_enabled": s.enable_external_apis,
                             "supplier_portal": "mock"}}


@router.get("/api/v1/data/sources")
def data_sources():
    return records(read_sql("SELECT * FROM data_sources ORDER BY origin, source_id"))


@router.post("/api/v1/pipeline/bootstrap", dependencies=[Depends(require_api_key)])
def run_bootstrap(reset: bool = True):
    from scip.services.bootstrap import bootstrap
    return Jobs.start("bootstrap", bootstrap, reset=reset)


@router.post("/api/v1/forecasts/run", dependencies=[Depends(require_api_key)])
def run_forecast():
    from scip.ml.forecasting import DemandForecaster

    def _run():
        r = DemandForecaster().run()
        return {"champion": r.champion, "overall": r.overall, "notes": r.notes, "seconds": r.seconds}
    return Jobs.start("forecast", _run)


@router.post("/api/v1/models/retrain", dependencies=[Depends(require_api_key)])
def retrain_classifiers():
    from scip.ml import delivery_risk, stockout, supplier_scoring

    def _run():
        dr = delivery_risk.train()
        delivery_risk.score_purchase_orders()
        so = stockout.train()
        supplier_scoring.score_suppliers()
        return {"delivery_risk": dr["results"], "stockout": so["results"]}
    return Jobs.start("retrain", _run)


@router.get("/api/v1/jobs")
def jobs():
    return Jobs.all()


@router.get("/api/v1/jobs/{job_id}")
def job(job_id: str):
    j = Jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j
