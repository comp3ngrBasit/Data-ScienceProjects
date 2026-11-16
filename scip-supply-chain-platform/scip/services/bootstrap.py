"""End-to-end platform build: ingest -> train -> optimize -> automate -> index. Used by scripts/bootstrap.py,
the API (/api/v1/pipeline/bootstrap) and the Docker entrypoint."""
from __future__ import annotations

import time

from scip.core.db import get_engine, get_meta, init_db, set_meta
from scip.core.logging import get_logger

log = get_logger("bootstrap")


def is_bootstrapped(engine=None) -> bool:
    try:
        return get_meta("bootstrapped_at", None, engine) is not None
    except Exception:
        return False


def bootstrap(engine=None, reset: bool = True, progress=None) -> dict:
    from scip.automation.engine import run_cycle
    from scip.ingestion.pipeline import run_ingestion
    from scip.ml import delivery_risk, stockout, supplier_scoring
    from scip.ml.forecasting import DemandForecaster
    from scip.optimization import logistics

    engine = engine or get_engine()
    say = progress or (lambda m: log.info(m))
    t0 = time.time()
    out = {}
    init_db(engine)
    say("1/7 Ingesting datasets")
    out["ingestion"] = run_ingestion(engine, reset=reset)
    say("2/7 Training demand forecasting models")
    fr = DemandForecaster(engine=engine).run()
    out["forecast"] = {"champion": fr.champion, "overall": fr.overall, "notes": fr.notes}
    say("3/7 Training delivery-risk classifiers")
    dr = delivery_risk.train(engine)
    out["delivery_risk"] = {"champion": dr["champion"], "results": dr["results"]}
    delivery_risk.score_purchase_orders(engine)
    say("4/7 Training stockout classifiers")
    so = stockout.train(engine)
    out["stockout"] = {"champion": so["champion"], "results": so["results"]}
    say("5/7 Scoring suppliers")
    supplier_scoring.score_suppliers(engine)
    say("6/7 Logistics optimisation")
    out["logistics"] = {n: {k: v for k, v in logistics.warehouse_allocation(n, engine).items() if k != "flows"}
                        for n in ("EU", "BR")}
    out["routing"] = {k: v for k, v in logistics.brazil_linehaul(engine).items() if k != "routes"}
    say("7/7 Automation cycle (alerts, procurement, report, Copilot index)")
    out["automation"] = run_cycle(engine, trigger="bootstrap")
    set_meta("bootstrapped_at", time.strftime("%Y-%m-%dT%H:%M:%S"), engine)
    out["seconds"] = round(time.time() - t0, 1)
    say(f"Bootstrap complete in {out['seconds']}s")
    return out
