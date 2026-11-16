"""SIMULATED data (clearly labelled data_origin='simulated') for what the public datasets do not contain:

* vendor master and vendor catalog (cost, MOQ, lead time, capacity) for the UCI planning SKUs;
* warehouse network (UK/EU distribution centres, Brazil hubs);
* daily inventory positions and purchase-order history, produced by running a replenishment policy
  against the REAL UCI daily demand.

Calibration: each vendor's lateness and delay distribution are taken from the REAL DataCo shipments
(late rate and delay-days distribution of the matching shipping mode) when they are loaded, so simulated
supplier behaviour mirrors observed logistics performance. Everything is seeded and reproducible.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scip.core.logging import get_logger

log = get_logger("ingest.simulate")

VENDORS = [
    # id, name, country, lat, lng, mode, base lead days, tier, categories (None = all)
    ("V-UK-01", "Albion Giftware Ltd", "GBR", 52.48, -1.89, "First Class", 6, "good", None),
    ("V-UK-02", "Thames Home & Gift Co", "GBR", 51.51, -0.13, "Second Class", 8, "average", None),
    ("V-UK-03", "Pennine Paper Goods", "GBR", 53.80, -1.55, "Second Class", 9, "average", ["Stationery & Cards", "Christmas & Seasonal", "Other Gifts"]),
    ("V-NL-01", "Lowlands Trading BV", "NLD", 51.92, 4.48, "Standard Class", 12, "good", None),
    ("V-DE-01", "Rheinland Deko GmbH", "DEU", 50.94, 6.96, "Standard Class", 14, "average", ["Home Decor", "Lighting & Candles", "Christmas & Seasonal"]),
    ("V-TR-01", "Anatolia Textiles & Crafts", "TUR", 41.01, 28.98, "Standard Class", 21, "average", ["Bags & Storage", "Home Decor", "Fashion & Accessories"]),
    ("V-IN-01", "Jaipur Handicraft Exports", "IND", 26.91, 75.79, "Standard Class", 35, "poor", ["Bags & Storage", "Fashion & Accessories", "Home Decor", "Other Gifts"]),
    ("V-CN-01", "Ningbo Homeware Manufacturing", "CHN", 29.87, 121.54, "Standard Class", 42, "average", None),
    ("V-CN-02", "Yiwu Festive Products", "CHN", 29.31, 120.08, "Standard Class", 45, "poor", ["Christmas & Seasonal", "Toys & Party", "Lighting & Candles", "Other Gifts"]),
    ("V-PK-01", "Sialkot Kitchen & Craft", "PAK", 32.49, 74.53, "Standard Class", 38, "good", ["Kitchen & Dining", "Garden & Outdoor", "Other Gifts"]),
]
TIER_MULT = {"good": 0.45, "average": 0.85, "poor": 1.25}
COST_FACTOR = {"GBR": 1.10, "NLD": 1.04, "DEU": 1.06, "TUR": 0.95, "IND": 0.86, "CHN": 0.84, "PAK": 0.88}

WAREHOUSES = [
    ("DC-UK-MAIN", "UK National DC (Midlands)", "Birmingham", None, "GBR", 52.48, -1.89, 400_000),
    ("DC-NL-EU", "EU Gateway DC (Rotterdam)", "Rotterdam", None, "NLD", 51.92, 4.48, 120_000),
    ("DC-FR-SOUTH", "South Europe DC (Lyon)", "Lyon", None, "FRA", 45.76, 4.84, 60_000),
    ("BR-HUB-SP", "Brazil Hub São Paulo", "São Paulo", "SP", "BRA", -23.55, -46.63, 60_000),
    ("BR-HUB-PE", "Brazil Hub Recife", "Recife", "PE", "BRA", -8.05, -34.88, 20_000),
    ("BR-HUB-PR", "Brazil Hub Curitiba", "Curitiba", "PR", "BRA", -25.43, -49.27, 25_000),
]
PLANNING_DC = "DC-UK-MAIN"


def calibrate(shipments: pd.DataFrame | None) -> dict[str, dict]:
    """Late rate and positive-delay samples per ship mode from REAL shipments (DataCo)."""
    defaults = {"First Class": (0.95, [1]), "Second Class": (0.77, [1, 2, 2, 3]),
                "Standard Class": (0.38, [1, 2]), "Same Day": (0.46, [1])}
    out = {}
    if shipments is not None and not shipments.empty and "ship_mode" in shipments:
        s = shipments.dropna(subset=["is_late", "actual_days", "scheduled_days"])
        s = s[s["source_id"] == "dataco"]
        for mode, g in s.groupby("ship_mode"):
            delay = (g["actual_days"] - g["scheduled_days"])
            out[mode] = {"late_rate": float(g["is_late"].astype(float).mean()),
                         "delays": delay[delay > 0].astype(int).tolist() or [1], "source": "real:dataco"}
    for mode, (rate, delays) in defaults.items():
        out.setdefault(mode, {"late_rate": rate, "delays": delays, "source": "default"})
    return out


def build_reference(products: pd.DataFrame, calib: dict, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    vendors = pd.DataFrame([dict(supplier_id=v[0], name=v[1] + " (simulated)", country_code=v[2], city=None,
                                 state=None, lat=v[3], lng=v[4], supplier_type="vendor", data_origin="simulated",
                                 source_id="simulated") for v in VENDORS])
    wh = pd.DataFrame([dict(location_id=w[0], location_type="warehouse", name=w[1] + " (simulated)", city=w[2],
                            state=w[3], country_code=w[4], lat=w[5], lng=w[6], capacity_units=float(w[7]),
                            data_origin="simulated") for w in WAREHOUSES])
    rows = []
    plan = products[products["is_planning_sku"]]
    for _, p in plan.iterrows():
        cands = [v for v in VENDORS if v[8] is None or p["category"] in v[8]]
        k = min(len(cands), int(rng.integers(2, 4)))
        idx = rng.choice(len(cands), size=k, replace=False)
        for i in idx:
            v = cands[i]
            mode_late = calib[v[5]]["late_rate"]
            rows.append(dict(
                supplier_id=v[0], sku=p["sku"],
                unit_cost=round(float(p["unit_cost"]) * COST_FACTOR[v[2]] * float(rng.uniform(0.95, 1.05)), 3),
                moq=int(rng.choice([12, 24, 48, 96, 144])) if v[2] != "GBR" else int(rng.choice([6, 12, 24])),
                lead_time_days=float(v[6]),
                lead_time_std=round(float(np.std(calib[v[5]]["delays"])) * TIER_MULT[v[7]] * max(1.0, v[6] / 10) + 0.5, 2),
                capacity_per_month=float(rng.choice([2000, 5000, 10000, 20000])),
                data_origin="simulated"))
    sp = pd.DataFrame(rows)
    return {"suppliers": vendors, "locations": wh, "supplier_products": sp}


def vendor_behaviour(calib: dict) -> dict[str, dict]:
    out = {}
    for v in VENDORS:
        c = calib[v[5]]
        out[v[0]] = {"late_prob": min(0.95, c["late_rate"] * TIER_MULT[v[7]]), "delays": c["delays"],
                     "scale": max(1.0, v[6] / 8), "tier": v[7], "short_ship": {"good": .02, "average": .06, "poor": .12}[v[7]]}
    return out


def simulate_operations(sales: pd.DataFrame, products: pd.DataFrame, supplier_products: pd.DataFrame,
                        calib: dict, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Run a (deliberately imperfect) replenishment policy over REAL daily demand of planning SKUs."""
    rng = np.random.default_rng(seed + 1)
    beh = vendor_behaviour(calib)
    plan = products[products["is_planning_sku"]].set_index("sku")
    daily = sales[sales["sku"].isin(plan.index)].groupby(["sku", "date"])["quantity"].sum()
    start, end = sales["date"].min(), sales["date"].max()
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    snaps, pos, current = [], [], []
    po_seq = 0
    for sku in plan.index:
        d = daily.get(sku, pd.Series(dtype=float))
        d.index = pd.to_datetime(d.index)
        demand = d.reindex(dates, fill_value=0.0).values
        cands = supplier_products[supplier_products["sku"] == sku]
        if cands.empty:
            continue
        cands = cands.assign(landed=cands["unit_cost"] * (1 + cands["lead_time_days"] / 365 * 0.25))
        vend = cands.sort_values("landed").iloc[0]
        L = float(vend["lead_time_days"])
        b = beh[vend["supplier_id"]]
        bias = float(rng.choice([rng.uniform(0.5, 0.75), rng.uniform(0.9, 1.3), rng.uniform(1.6, 2.4)], p=[0.2, 0.6, 0.2]))
        rate = max(demand[:28].mean(), 0.5)
        sd = max(demand[:28].std(), 1.0)
        on_hand = math.ceil(rate * (L + 21) * bias)
        open_pos: list[dict] = []
        hist = np.zeros(n)
        for t in range(n):
            for po in open_pos:
                if po["arrive"] == t:
                    on_hand += po["qty_received"]
            open_pos = [p for p in open_pos if p["arrive"] > t]
            sold = min(demand[t], on_hand)
            on_hand -= sold
            hist[t] = on_hand
            if t >= 28 and t % 7 == 0:
                window = demand[max(0, t - 42):t]
                rate, sd = max(window.mean(), 0.3), max(window.std(), 0.8)
            ip = on_hand + sum(p["quantity"] for p in open_pos)
            rop = bias * (rate * L + 1.65 * sd * math.sqrt(L))
            if ip <= rop:
                qty = max(int(vend["moq"]), int(math.ceil(rate * 28 * bias)))
                late = rng.random() < b["late_prob"]
                delay = int(round(float(rng.choice(b["delays"])) * b["scale"])) if late else -int(rng.integers(0, 2))
                arrive = t + max(1, int(round(L)) + delay)
                qr = qty if rng.random() > b["short_ship"] else int(qty * rng.uniform(0.7, 0.95))
                po_seq += 1
                po = dict(po_id=f"PO-H{po_seq:06d}", supplier_id=vend["supplier_id"], sku=sku, quantity=float(qty),
                          unit_cost=float(vend["unit_cost"]), order_idx=t, promised_idx=t + int(round(L)),
                          arrive=arrive, qty_received=float(qr))
                open_pos.append(po)
                pos.append(po)
        snaps.append(pd.DataFrame({"date": dates.date, "sku": sku, "location_id": PLANNING_DC, "on_hand": hist,
                                   "data_origin": "simulated"}))
        current.append(dict(sku=sku, location_id=PLANNING_DC, on_hand=float(on_hand), as_of=end.date(),
                            data_origin="simulated"))
    po = pd.DataFrame(pos)
    if not po.empty:
        d0 = dates[0]
        po["order_date"] = (d0 + pd.to_timedelta(po["order_idx"], unit="D")).dt.date
        po["promised_date"] = (d0 + pd.to_timedelta(po["promised_idx"], unit="D")).dt.date
        received = po["arrive"] <= n - 1
        po["received_date"] = (d0 + pd.to_timedelta(po["arrive"], unit="D")).dt.date.where(received, None)
        po["qty_received"] = po["qty_received"].where(received, None)
        po["status"] = np.where(received, "received", "sent")
        po["total_value"] = (po["quantity"] * po["unit_cost"]).round(2)
        po["approval_level"] = "historical"
        po["created_by"] = "history"
        po["data_origin"] = "simulated"
        po = po.drop(columns=["order_idx", "promised_idx", "arrive"])
    log.info("Simulated operations: %d SKUs, %d POs", len(current), len(po))
    return {"inventory_snapshots": pd.concat(snaps, ignore_index=True) if snaps else pd.DataFrame(),
            "inventory_current": pd.DataFrame(current), "purchase_orders": po}
