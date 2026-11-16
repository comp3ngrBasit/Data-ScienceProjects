"""UCI Online Retail (real) — https://archive.ics.uci.edu/dataset/352/online+retail

Raw: 541,909 invoice lines (Dec 2010 – Dec 2011) of a UK online gift wholesaler.
Cleaning rules (all documented in docs/DATA_DICTIONARY.md):
  * drop cancellations (InvoiceNo starting with 'C') and non-positive quantity / price;
  * drop non-product stock codes (postage, fees, manual adjustments, samples, gift vouchers);
  * fill missing descriptions with the SKU's most frequent description;
  * aggregate to daily demand per SKU x customer country.
Derived (labelled as assumptions): category from description keywords; unit cost = 55% of median price
(the dataset contains no cost data).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scip.core.logging import get_logger
from scip.integrations.countries import UCI_COUNTRY_TO_ISO3

log = get_logger("ingest.uci")

SOURCE_ID = "uci_online_retail"
NON_PRODUCT_CODES = {"POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK", "D", "S", "PADS", "B", "C2", "m"}
PRODUCT_CODE = re.compile(r"^\d{5}[A-Za-z]{0,2}$")
ASSUMED_COST_RATIO = 0.55

CATEGORY_RULES = [
    ("Christmas & Seasonal", r"CHRISTMAS|XMAS|ADVENT|SANTA|REINDEER|SNOWFLAKE|EASTER"),
    ("Bags & Storage", r"\bBAG\b|BAGS|STORAGE|BOX|TIN|BASKET|HAMPER|CASE"),
    ("Lighting & Candles", r"CANDLE|LIGHT|LANTERN|T-LIGHT|TEALIGHT|LAMP|NIGHT LIGHT"),
    ("Kitchen & Dining", r"MUG|CUP|PLATE|BOWL|JUG|TEAPOT|CAKE|BAKING|LUNCH|NAPKIN|KITCHEN|SPOON|TRAY|CUTLERY|DOILY|COASTER|EGG"),
    ("Stationery & Cards", r"CARD|PAPER|NOTEBOOK|PENCIL|PEN\b|STICKER|WRAP|TAPE|GIFT TAG|ENVELOPE|BOOK"),
    ("Home Decor", r"HEART|SIGN|FRAME|CLOCK|MIRROR|CUSHION|DOORMAT|WREATH|BUNTING|GARLAND|DECORATION|ORNAMENT|HANGING|HOOK|VASE|DOORSTOP"),
    ("Toys & Party", r"TOY|PARTY|BALLOON|GAME|PUZZLE|DOLL|SPACEBOY|DINOSAUR|SKIPPING|BUBBLES|PLAYHOUSE"),
    ("Garden & Outdoor", r"GARDEN|PLANT|FLOWER POT|WATERING|BIRD|PICNIC|PARASOL|UMBRELLA"),
    ("Fashion & Accessories", r"NECKLACE|BRACELET|EARRING|RING\b|PURSE|HAIR|SCARF|JEWEL|BROOCH|CHARM"),
]


def derive_category(description: str) -> str:
    d = (description or "").upper()
    for cat, pattern in CATEGORY_RULES:
        if re.search(pattern, d):
            return cat
    return "Other Gifts"


def read_raw(path: Path) -> pd.DataFrame:
    path = Path(path)
    pkl = path.with_name("online_retail.pkl")
    if pkl.exists():
        return pd.read_pickle(pkl)
    if path.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
            with z.open(name) as fh:
                df = pd.read_excel(fh)
    else:
        df = pd.read_excel(path)
    df.to_pickle(pkl)
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"raw_rows": len(raw)}
    df = raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    cancelled = df["InvoiceNo"].str.startswith("C")
    stats["cancellations_removed"] = int(cancelled.sum())
    df = df[~cancelled]
    bad_qty = (df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)
    stats["non_positive_removed"] = int(bad_qty.sum())
    df = df[~bad_qty]
    non_product = df["StockCode"].isin(NON_PRODUCT_CODES) | ~df["StockCode"].str.match(PRODUCT_CODE)
    stats["non_product_codes_removed"] = int(non_product.sum())
    df = df[~non_product]
    mode_desc = (df.dropna(subset=["Description"]).groupby("StockCode")["Description"]
                 .agg(lambda s: s.value_counts().index[0]))
    stats["missing_descriptions_filled"] = int(df["Description"].isna().sum())
    df["Description"] = df["StockCode"].map(mode_desc).fillna(df["Description"]).fillna("UNKNOWN")
    dupes = df.duplicated()
    stats["exact_duplicates_removed"] = int(dupes.sum())
    df = df[~dupes]
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df["country_code"] = df["Country"].map(UCI_COUNTRY_TO_ISO3)
    df["location_id"] = np.where(df["country_code"].notna(), "CUST-" + df["country_code"].fillna(""), "CUST-OTHER")
    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    stats["clean_rows"] = len(df)
    return df, stats


def build(path: Path, planning_sku_count: int = 150) -> dict[str, pd.DataFrame]:
    raw = read_raw(path)
    df, stats = clean(raw)
    log.info("UCI cleaned: %s", stats)

    prices = df.groupby("StockCode").agg(name=("Description", lambda s: s.value_counts().index[0]),
                                         unit_price=("UnitPrice", "median"), revenue=("revenue", "sum"),
                                         first=("date", "min"), active_days=("date", "nunique"))
    prices["category"] = prices["name"].map(derive_category)
    prices["unit_cost"] = (prices["unit_price"] * ASSUMED_COST_RATIO).round(3)
    end = df["date"].max()
    eligible = prices[(prices["first"] <= end - pd.Timedelta(days=240)) & (prices["active_days"] >= 100)]
    planning = set(eligible.sort_values("revenue", ascending=False).head(planning_sku_count).index)
    products = pd.DataFrame({
        "sku": prices.index, "name": prices["name"].str.strip().str.title().values,
        "category": prices["category"].values, "unit_cost": prices["unit_cost"].values,
        "unit_price": prices["unit_price"].round(3).values,
        "is_planning_sku": prices.index.isin(planning), "data_origin": "real:uci", "source_id": SOURCE_ID})

    sales = (df.groupby(["date", "StockCode", "location_id"], as_index=False)
             .agg(quantity=("Quantity", "sum"), revenue=("revenue", "sum"))
             .rename(columns={"StockCode": "sku"}))
    sales["data_origin"] = "real:uci"
    sales["revenue"] = sales["revenue"].round(2)

    countries = (df.dropna(subset=["country_code"]).groupby(["location_id", "country_code", "Country"])
                 .size().reset_index()[["location_id", "country_code", "Country"]])
    locations = pd.DataFrame({"location_id": countries["location_id"], "location_type": "customer_region",
                              "name": "Customers — " + countries["Country"], "city": None, "state": None,
                              "country_code": countries["country_code"], "lat": None, "lng": None,
                              "capacity_units": None, "data_origin": "real:uci"})
    locations = pd.concat([locations, pd.DataFrame([{
        "location_id": "CUST-OTHER", "location_type": "customer_region", "name": "Customers — Unspecified / EC",
        "data_origin": "real:uci"}])], ignore_index=True)
    stats.update(products=len(products), planning_skus=len(planning), sales_rows=len(sales),
                 start=str(df["date"].min().date()), end=str(end.date()))
    return {"products": products, "sales_daily": sales, "locations": locations, "stats": stats}
