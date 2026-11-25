"""Pipeline orchestrator / CLI.

    python -m app.pipeline run                 # all stages, cached
    python -m app.pipeline run --stage forecast --force
    python -m app.pipeline status

Stages are skipped when their input fingerprint (processed data + code version of the stage + settings)
is unchanged, so expensive models are not retrained unnecessarily.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from datetime import datetime, timezone

from app.config import PROCESSED_DIR, RESULTS_DIR, SETTINGS
from app.utils.helpers import load_json, save_json, set_seeds
from app.utils.logging import get_logger

log = get_logger("pipeline")
MANIFEST = RESULTS_DIR / "_manifest.json"


def _stage_data():
    from app.data import preprocessing as pp

    return {"household": pp.build_household(), "grid": pp.build_grid("DE")}


def _stage_eda():
    from app.analytics import consumption, timeseries_stats

    return {"consumption": bool(consumption.run()), "timeseries": bool(timeseries_stats.run())}


def _stage_forecast():
    from app.models.forecasting import engine

    return {"experiments": [s["key"] for s in engine.run()]}


def _stage_forward():
    from app.models.forecasting import engine

    return {f"{spec}_h{h}": engine.forward_forecast(spec, h)["model"]
            for spec, h in (("household_hourly", 24), ("grid_hourly", 24))}


def _stage_peak():
    from app.analytics import peak_analysis

    return {"done": bool(peak_analysis.run())}


def _stage_segmentation():
    from app.models.clustering import segmentation

    return {"done": bool(segmentation.run())}


def _stage_anomaly():
    from app.models.anomaly_detection import detector

    return {"n": detector.run()["n_consensus_anomalies"]}


def _stage_renewable():
    from app.analytics import renewable_analysis

    return {"done": bool(renewable_analysis.run())}


def _stage_weather():
    from app.analytics import weather_analysis

    return {"best": weather_analysis.run()["best_model_test"]}


def _stage_statistics():
    from app.analytics import statistical_tests

    return statistical_tests.run()


def _stage_explain():
    from app.analytics import explainability

    return {"keys": list(explainability.run())}


def _stage_kpis():
    return compile_kpis()


def _stage_report():
    from app.reporting import generate

    return generate()


STAGES = {"data": _stage_data, "eda": _stage_eda, "forecast": _stage_forecast, "forward": _stage_forward,
          "peak": _stage_peak, "segmentation": _stage_segmentation, "anomaly": _stage_anomaly,
          "renewable": _stage_renewable, "weather": _stage_weather, "statistics": _stage_statistics,
          "explain": _stage_explain, "kpis": _stage_kpis, "report": _stage_report}
DEPENDS = {"forward": ["forecast"], "explain": ["forecast"], "kpis": list(STAGES)[:-2], "report": ["kpis"]}


def compile_kpis() -> dict:
    """Executive KPIs, every value read from a computed result file."""
    c = load_json("consumption", "summary")
    p = load_json("peak", "summary")
    r = load_json("renewable", "summary")
    a = load_json("anomaly", "summary")
    k = {"generated_at": datetime.now(timezone.utc).isoformat()}
    if c:
        k["household_total_kwh"] = c["household"]["total_kwh"]
        k["household_avg_daily_kwh"] = c["household"]["daily_kwh"]["mean"]
        k["household_daily_cv"] = c["household"]["daily_kwh"]["cv"]
        k["grid_total_twh"] = c["grid"]["total_load_twh"]
        k["grid_avg_daily_gwh"] = c["grid"]["daily_load_mwh"]["mean"] / 1e3
        k["grid_daily_cv"] = c["grid"]["daily_load_mwh"]["cv"]
    if p:
        k["household_max_hourly_kwh"] = p["household"]["max"]["value"]
        k["household_max_time"] = p["household"]["max"]["timestamp"]
        k["grid_max_load_mw"] = p["grid"]["max"]["value"]
        k["grid_max_time"] = p["grid"]["max"]["timestamp"]
    if r:
        k["vre_penetration_pct"] = r["kpis"]["vre_penetration_pct"]
        k["solar_share_of_vre_pct"] = r["kpis"]["solar_share_of_vre_pct"]
        k["wind_share_of_vre_pct"] = r["kpis"]["wind_share_of_vre_pct"]
    if a:
        k["consensus_anomalies"] = a["n_consensus_anomalies"]
        k["anomalies_by_severity"] = a["consensus_by_severity"]
    for key in ("household_hourly_h24", "grid_hourly_h24"):
        s = load_json("forecasting", f"{key}_summary")
        if s:
            from app.utils.helpers import load_table

            m = load_table("forecasting", f"{key}_metrics").set_index("model")
            best = s["best_model"]
            k[f"{key}_best_model"] = best
            k[f"{key}_best_wape"] = m.loc[best, "WAPE"]
            k[f"{key}_best_mae"] = m.loc[best, "MAE"]
            k[f"{key}_skill_vs_naive_pct"] = m.loc[best, "skill_vs_naive_pct"]
            k[f"{key}_pi_coverage"] = m.loc[best, "PI_coverage"]
        f = load_json("forecasting", f"{key}_forward")
        if f:
            k[f"{key}_forward_total"] = f["total_next_period"]
            k[f"{key}_forward_peak"] = f["peak_next_period"]
            k[f"{key}_forward_from"] = f["issued_at_data_end"]
    save_json(k, "overview", "kpis")
    return k


def _fingerprint(stage: str) -> str:
    h = hashlib.sha256()
    h.update(inspect.getsource(STAGES[stage]).encode())
    h.update(repr(SETTINGS).encode())
    for f in sorted(PROCESSED_DIR.glob("*.parquet")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def run(stages: list[str] | None = None, force: bool = False) -> dict:
    set_seeds()
    man = _manifest()
    todo = stages or list(STAGES)
    for st in todo:
        fp = _fingerprint(st) if st != "data" else "raw"
        upstream_newer = any(man.get(d, {}).get("finished", "") > man.get(st, {}).get("finished", "~")
                             for d in DEPENDS.get(st, []))
        if not force and man.get(st, {}).get("fingerprint") == fp and man[st].get("status") == "ok" and not upstream_newer:
            log.info("[%s] cached - skipping", st)
            continue
        log.info("[%s] running", st)
        t0 = time.time()
        try:
            out = STAGES[st]()
            man[st] = {"status": "ok", "fingerprint": fp, "seconds": round(time.time() - t0, 1),
                       "finished": datetime.now(timezone.utc).isoformat(), "output": out}
        except Exception as exc:  # noqa: BLE001 - record and continue with independent stages
            log.exception("[%s] failed", st)
            man[st] = {"status": "failed", "error": str(exc), "finished": datetime.now(timezone.utc).isoformat()}
        MANIFEST.write_text(json.dumps(man, indent=2, default=str))
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description="Energy Grid Intelligence pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", action="append", choices=list(STAGES), help="stage(s) to run (default: all)")
    r.add_argument("--force", action="store_true", help="ignore cache")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "run":
        man = run(a.stage, a.force)
        failed = [k for k, v in man.items() if v.get("status") != "ok"]
        print(json.dumps({k: {"status": v["status"], "seconds": v.get("seconds")} for k, v in man.items()}, indent=1))
        raise SystemExit(1 if failed else 0)
    print(json.dumps({k: {x: v.get(x) for x in ("status", "seconds", "finished")} for k, v in _manifest().items()}, indent=1))


if __name__ == "__main__":
    main()
