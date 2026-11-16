"""Olist Brazilian E-Commerce (real) — https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Works with whatever subset of the 9 Olist CSVs is present:
  * customers only  -> customer demand points per state (logistics: warehouse allocation & routing);
  * + orders, order_items, sellers (+ reviews, products, category translation, geolocation)
    -> shipments with real promised vs delivered dates, sellers as suppliers, review scores.
State centroid coordinates are public reference values (IBGE state capitals) used when the
geolocation file is absent.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from scip.core.logging import get_logger

log = get_logger("ingest.olist")
SOURCE_ID = "olist"

# State capital coordinates (lat, lng)
BR_STATES = {
    "AC": ("Rio Branco", -9.97, -67.81), "AL": ("Maceió", -9.67, -35.74), "AM": ("Manaus", -3.12, -60.02),
    "AP": ("Macapá", 0.03, -51.07), "BA": ("Salvador", -12.97, -38.50), "CE": ("Fortaleza", -3.73, -38.52),
    "DF": ("Brasília", -15.79, -47.88), "ES": ("Vitória", -20.32, -40.34), "GO": ("Goiânia", -16.68, -49.25),
    "MA": ("São Luís", -2.53, -44.30), "MG": ("Belo Horizonte", -19.92, -43.94), "MS": ("Campo Grande", -20.47, -54.62),
    "MT": ("Cuiabá", -15.60, -56.10), "PA": ("Belém", -1.46, -48.49), "PB": ("João Pessoa", -7.12, -34.86),
    "PE": ("Recife", -8.05, -34.88), "PI": ("Teresina", -5.09, -42.80), "PR": ("Curitiba", -25.43, -49.27),
    "RJ": ("Rio de Janeiro", -22.91, -43.17), "RN": ("Natal", -5.79, -35.21), "RO": ("Porto Velho", -8.76, -63.90),
    "RR": ("Boa Vista", 2.82, -60.67), "RS": ("Porto Alegre", -30.03, -51.23), "SC": ("Florianópolis", -27.60, -48.55),
    "SE": ("Aracaju", -10.91, -37.07), "SP": ("São Paulo", -23.55, -46.63), "TO": ("Palmas", -10.18, -48.33),
}

FILES = {
    "customers": "olist_customers_dataset.csv", "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv", "sellers": "olist_sellers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv", "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv", "geolocation": "olist_geolocation_dataset.csv",
}


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def available_files(folder: Path) -> dict[str, Path]:
    folder = Path(folder)
    found = {}
    for key, name in FILES.items():
        hits = list(folder.rglob(name))
        if hits:
            found[key] = hits[0]
    return found


def build(folder: Path) -> dict:
    files = available_files(folder)
    stats = {"files_found": sorted(files)}
    out: dict = {"stats": stats}
    if "customers" not in files:
        stats["status"] = "no Olist files found"
        return out
    cust = pd.read_csv(files["customers"], dtype={"customer_zip_code_prefix": str})
    stats["customers"] = len(cust)
    by_state = cust.groupby("customer_state").agg(customers=("customer_unique_id", "nunique")).reset_index()
    out["locations"] = pd.DataFrame({
        "location_id": "BR-CUST-" + by_state["customer_state"],
        "location_type": "customer_region",
        "name": "Olist customers — " + by_state["customer_state"],
        "city": by_state["customer_state"].map(lambda s: BR_STATES.get(s, ("", 0, 0))[0]),
        "state": by_state["customer_state"], "country_code": "BRA",
        "lat": by_state["customer_state"].map(lambda s: BR_STATES.get(s, (None, None, None))[1]),
        "lng": by_state["customer_state"].map(lambda s: BR_STATES.get(s, (None, None, None))[2]),
        "capacity_units": by_state["customers"].astype(float),     # customer count used as demand weight
        "data_origin": "real:olist"})
    out["customers"] = cust

    needed = {"orders", "items", "sellers"}
    if not needed <= set(files):
        stats["status"] = "customers only — order/seller files missing, shipments not built"
        log.info("Olist: %s", stats)
        return out

    orders = pd.read_csv(files["orders"], parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date",
                                                       "order_delivered_customer_date", "order_estimated_delivery_date"])
    items = pd.read_csv(files["items"])
    sellers = pd.read_csv(files["sellers"], dtype={"seller_zip_code_prefix": str})
    prods = pd.read_csv(files["products"]) if "products" in files else None
    it = items.groupby(["order_id", "seller_id"]).agg(quantity=("order_item_id", "count"), value=("price", "sum"),
                                                      freight=("freight_value", "sum"),
                                                      product_id=("product_id", "first")).reset_index()
    df = it.merge(orders, on="order_id", how="inner").merge(cust, on="customer_id", how="left") \
        .merge(sellers, on="seller_id", how="left")
    if prods is not None:
        df = df.merge(prods[["product_id", "product_category_name"]], on="product_id", how="left")
        if "translation" in files:
            tr = pd.read_csv(files["translation"])
            df = df.merge(tr, on="product_category_name", how="left")
            df["category"] = df["product_category_name_english"].fillna(df["product_category_name"])
        else:
            df["category"] = df["product_category_name"]
    else:
        df["category"] = None
    if "reviews" in files:
        rv = pd.read_csv(files["reviews"]).groupby("order_id")["review_score"].mean()
        df["review_score"] = df["order_id"].map(rv)
    else:
        df["review_score"] = np.nan
    # coordinates: zip-prefix geolocation if available, else state capitals
    if "geolocation" in files:
        geo = pd.read_csv(files["geolocation"], dtype={"geolocation_zip_code_prefix": str}) \
            .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
        c_lat = df["customer_zip_code_prefix"].map(geo["geolocation_lat"])
        c_lng = df["customer_zip_code_prefix"].map(geo["geolocation_lng"])
        s_lat = df["seller_zip_code_prefix"].map(geo["geolocation_lat"])
        s_lng = df["seller_zip_code_prefix"].map(geo["geolocation_lng"])
    else:
        c_lat = c_lng = s_lat = s_lng = pd.Series(np.nan, index=df.index)
    st_lat = lambda s: s.map(lambda x: BR_STATES.get(x, (None, np.nan, np.nan))[1])
    st_lng = lambda s: s.map(lambda x: BR_STATES.get(x, (None, np.nan, np.nan))[2])
    c_lat, c_lng = c_lat.fillna(st_lat(df["customer_state"])), c_lng.fillna(st_lng(df["customer_state"]))
    s_lat, s_lng = s_lat.fillna(st_lat(df["seller_state"])), s_lng.fillna(st_lng(df["seller_state"]))

    purchase = df["order_purchase_timestamp"].dt.normalize()
    delivered = df["order_delivered_customer_date"].dt.normalize()
    est = df["order_estimated_delivery_date"].dt.normalize()
    status = np.select([df["order_status"].eq("delivered") & delivered.notna(),
                        df["order_status"].isin(["canceled", "unavailable"])],
                       ["delivered", "cancelled"], default="in_transit")
    ship = pd.DataFrame({
        "shipment_id": "OL-" + df["order_id"].str[:16] + "-" + df["seller_id"].str[:6],
        "source_id": SOURCE_ID, "order_date": purchase.dt.date,
        "ship_date": df["order_delivered_carrier_date"].dt.normalize().dt.date,
        "promised_date": est.dt.date, "delivered_date": delivered.dt.date, "ship_mode": "Marketplace",
        "scheduled_days": (est - purchase).dt.days.astype(float),
        "actual_days": (delivered - purchase).dt.days.astype(float),
        "status": status,
        "is_late": np.where(status == "delivered", delivered > est, None),
        "supplier_id": "OLIST-" + df["seller_id"].str[:12], "origin_state": df["seller_state"],
        "dest_state": df["customer_state"], "dest_country": "Brazil", "market": "LATAM", "region": "Brazil",
        "category": df["category"], "quantity": df["quantity"].astype(float), "value": df["value"].round(2),
        "freight_value": df["freight"].round(2),
        "distance_km": haversine_km(s_lat, s_lng, c_lat, c_lng).round(1),
        "dest_lat": c_lat, "dest_lng": c_lng, "review_score": df["review_score"], "data_origin": "real:olist"})
    ship = ship.drop_duplicates("shipment_id")
    sel = sellers.copy()
    out["shipments"] = ship
    out["suppliers"] = pd.DataFrame({
        "supplier_id": "OLIST-" + sel["seller_id"].str[:12],
        "name": "Olist seller " + sel["seller_id"].str[:6] + " (" + sel["seller_city"].str.title() + "/" + sel["seller_state"] + ")",
        "country_code": "BRA", "city": sel["seller_city"].str.title(), "state": sel["seller_state"],
        "lat": st_lat(sel["seller_state"]), "lng": st_lng(sel["seller_state"]),
        "supplier_type": "marketplace_seller", "data_origin": "real:olist", "source_id": SOURCE_ID}).drop_duplicates("supplier_id")
    stats.update(status="full", shipments=len(ship), sellers=len(out["suppliers"]),
                 late_rate=float(pd.Series(ship["is_late"]).dropna().astype(float).mean()))
    log.info("Olist: %s", stats)
    return out
