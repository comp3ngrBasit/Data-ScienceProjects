"""Ingestion orchestration: raw files / APIs -> cleaned frames -> unified PostgreSQL schema."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from scip.core.config import get_settings
from scip.core.db import get_engine, read_table, reset_db, set_meta, write_frame
from scip.core.logging import get_logger
from scip.ingestion import dataco, olist, simulate, uci
from scip.integrations import weather
from scip.integrations.countries import fetch_countries

log = get_logger("ingest.pipeline")

SOURCES = {
    "uci_online_retail": ("UCI Online Retail", "https://archive.ics.uci.edu/dataset/352/online+retail", "real"),
    "olist": ("Olist Brazilian E-Commerce", "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce", "real"),
    "dataco": ("DataCo Smart Supply Chain",
               "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis", "real"),
    "m5": ("M5 Forecasting (calendar)", "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data", "real"),
    "simulated": ("SCIP simulation (labelled)", None, "simulated"),
}


def raw_dir(name: str) -> Path:
    s = get_settings()
    override = {"olist": s.olist_dir, "dataco": s.dataco_dir, "m5": s.m5_dir}.get(name)
    return Path(override) if override else s.data_dir / "raw" / name


def run_ingestion(engine=None, reset: bool = True, fetch_weather: bool = True) -> dict:
    engine = engine or get_engine()
    s = get_settings()
    report: dict = {"started_at": datetime.now().isoformat(timespec="seconds")}
    if reset:
        reset_db(engine)
    write_frame(pd.DataFrame([dict(source_id=k, name=v[0], url=v[1], origin=v[2]) for k, v in SOURCES.items()]),
                "data_sources", engine)

    # ---- UCI (real demand)
    uci_dir = raw_dir("uci")
    uci_file = next(iter(sorted(uci_dir.glob("*.xlsx"))), None) or next(iter(uci_dir.glob("*.zip")), None)
    if uci_file is None:
        raise FileNotFoundError(f"UCI Online Retail file not found in {uci_dir}")
    u = uci.build(uci_file, s.planning_sku_count)
    report["uci"] = u["stats"]

    # ---- DataCo (real shipments)
    dc_file = dataco.find_file(raw_dir("dataco"))
    shipments = []
    if dc_file:
        d = dataco.build(dc_file)
        shipments.append(d["shipments"])
        report["dataco"] = d["stats"]
    else:
        report["dataco"] = {"status": "not found"}

    # ---- Olist (real)
    o = olist.build(raw_dir("olist"))
    report["olist"] = o["stats"]
    if "shipments" in o:
        shipments.append(o["shipments"])
    ship_df = pd.concat(shipments, ignore_index=True) if shipments else pd.DataFrame()

    # ---- simulation calibrated on real shipments
    calib = simulate.calibrate(ship_df)
    report["calibration"] = {m: {"late_rate": round(c["late_rate"], 3), "source": c["source"]} for m, c in calib.items()}
    ref = simulate.build_reference(u["products"], calib, s.random_seed)
    ops = simulate.simulate_operations(u["sales_daily"], u["products"], ref["supplier_products"], calib, s.random_seed)

    # ---- countries (REST Countries with fallback)
    codes = set(u["locations"]["country_code"].dropna()) | set(ref["suppliers"]["country_code"]) | \
        set(ref["locations"]["country_code"]) | {"BRA"}
    countries = fetch_countries(sorted(codes))
    report["countries"] = countries["source"].value_counts().to_dict()
    cmap = countries.set_index("country_code")
    loc = u["locations"].copy()
    loc["lat"] = loc["country_code"].map(cmap["lat"])
    loc["lng"] = loc["country_code"].map(cmap["lng"])

    # ---- load (parents before children)
    write_frame(countries, "countries", engine)
    locations = pd.concat([loc, ref["locations"], o.get("locations", pd.DataFrame())], ignore_index=True)
    write_frame(locations, "locations", engine)
    write_frame(u["products"], "products", engine)
    suppliers = pd.concat([ref["suppliers"], o.get("suppliers", pd.DataFrame())], ignore_index=True)
    write_frame(suppliers, "suppliers", engine)
    write_frame(ref["supplier_products"], "supplier_products", engine)
    write_frame(u["sales_daily"], "sales_daily", engine)
    if not ship_df.empty:
        write_frame(ship_df, "shipments", engine)
    write_frame(ops["inventory_snapshots"], "inventory_snapshots", engine)
    write_frame(ops["inventory_current"], "inventory_current", engine)
    write_frame(ops["purchase_orders"], "purchase_orders", engine)

    counts = {"uci_online_retail": len(u["sales_daily"]), "dataco": int((ship_df.get("source_id") == "dataco").sum()) if not ship_df.empty else 0,
              "olist": int(o["stats"].get("customers", 0)) + int(o["stats"].get("shipments", 0)),
              "m5": int((raw_dir("m5") / "calendar.csv").exists()),
              "simulated": len(ops["inventory_snapshots"]) + len(ops["purchase_orders"])}
    from sqlalchemy import text
    with engine.begin() as conn:
        for sid, n in counts.items():
            conn.execute(text("UPDATE data_sources SET row_count=:n, loaded_at=:t WHERE source_id=:s"),
                         {"n": n, "t": datetime.now(), "s": sid})
        conn.execute(text("UPDATE data_sources SET notes=:n WHERE source_id='olist'"),
                     {"n": o["stats"].get("status")})

    as_of = pd.Timestamp(u["stats"]["end"])
    set_meta("as_of_date", as_of.date().isoformat(), engine)
    set_meta("ingested_at", datetime.now().isoformat(timespec="seconds"), engine)

    if fetch_weather:
        report["weather"] = weather.sync_history("london", pd.Timestamp(u["stats"]["start"]).date(), as_of.date(), engine)
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    log.info("Ingestion complete")
    return report


def as_of_date(engine=None) -> pd.Timestamp:
    from scip.core.db import get_meta
    v = get_meta("as_of_date", None, engine)
    return pd.Timestamp(v) if v else pd.Timestamp.today().normalize()
