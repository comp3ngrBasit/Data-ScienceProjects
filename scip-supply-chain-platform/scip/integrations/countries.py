"""REST Countries integration (https://restcountries.com) with a cached table and a small static
fallback for the countries that appear in the datasets (used only when the API is unreachable,
and marked source='static_fallback')."""
from __future__ import annotations

import pandas as pd

from scip.core.config import get_settings
from scip.core.logging import get_logger
from scip.integrations.http import ExternalAPIError, get_json

log = get_logger("countries")

# UCI Online Retail country names -> ISO alpha-3 (None = not a single country)
UCI_COUNTRY_TO_ISO3 = {
    "United Kingdom": "GBR", "France": "FRA", "Australia": "AUS", "Netherlands": "NLD", "Germany": "DEU",
    "Norway": "NOR", "EIRE": "IRL", "Switzerland": "CHE", "Spain": "ESP", "Poland": "POL", "Portugal": "PRT",
    "Italy": "ITA", "Belgium": "BEL", "Lithuania": "LTU", "Japan": "JPN", "Iceland": "ISL",
    "Channel Islands": "JEY", "Denmark": "DNK", "Cyprus": "CYP", "Sweden": "SWE", "Austria": "AUT",
    "Israel": "ISR", "Finland": "FIN", "Bahrain": "BHR", "Greece": "GRC", "Hong Kong": "HKG", "Singapore": "SGP",
    "Lebanon": "LBN", "United Arab Emirates": "ARE", "Saudi Arabia": "SAU", "Czech Republic": "CZE",
    "Canada": "CAN", "Unspecified": None, "Brazil": "BRA", "USA": "USA", "European Community": None,
    "Malta": "MLT", "RSA": "ZAF",
}

# Static fallback: name, region, subregion, lat, lng, currency (public reference facts)
STATIC = {
    "GBR": ("United Kingdom", "Europe", "Northern Europe", 54.0, -2.0, "GBP"),
    "FRA": ("France", "Europe", "Western Europe", 46.0, 2.0, "EUR"),
    "AUS": ("Australia", "Oceania", "Australia and New Zealand", -27.0, 133.0, "AUD"),
    "NLD": ("Netherlands", "Europe", "Western Europe", 52.5, 5.75, "EUR"),
    "DEU": ("Germany", "Europe", "Western Europe", 51.0, 9.0, "EUR"),
    "NOR": ("Norway", "Europe", "Northern Europe", 62.0, 10.0, "NOK"),
    "IRL": ("Ireland", "Europe", "Northern Europe", 53.0, -8.0, "EUR"),
    "CHE": ("Switzerland", "Europe", "Western Europe", 47.0, 8.0, "CHF"),
    "ESP": ("Spain", "Europe", "Southern Europe", 40.0, -4.0, "EUR"),
    "POL": ("Poland", "Europe", "Central Europe", 52.0, 20.0, "PLN"),
    "PRT": ("Portugal", "Europe", "Southern Europe", 39.5, -8.0, "EUR"),
    "ITA": ("Italy", "Europe", "Southern Europe", 42.83, 12.83, "EUR"),
    "BEL": ("Belgium", "Europe", "Western Europe", 50.83, 4.0, "EUR"),
    "LTU": ("Lithuania", "Europe", "Northern Europe", 56.0, 24.0, "EUR"),
    "JPN": ("Japan", "Asia", "Eastern Asia", 36.0, 138.0, "JPY"),
    "ISL": ("Iceland", "Europe", "Northern Europe", 65.0, -18.0, "ISK"),
    "JEY": ("Jersey", "Europe", "Northern Europe", 49.25, -2.17, "GBP"),
    "DNK": ("Denmark", "Europe", "Northern Europe", 56.0, 10.0, "DKK"),
    "CYP": ("Cyprus", "Europe", "Southern Europe", 35.0, 33.0, "EUR"),
    "SWE": ("Sweden", "Europe", "Northern Europe", 62.0, 15.0, "SEK"),
    "AUT": ("Austria", "Europe", "Central Europe", 47.33, 13.33, "EUR"),
    "ISR": ("Israel", "Asia", "Western Asia", 31.47, 35.13, "ILS"),
    "FIN": ("Finland", "Europe", "Northern Europe", 64.0, 26.0, "EUR"),
    "BHR": ("Bahrain", "Asia", "Western Asia", 26.0, 50.55, "BHD"),
    "GRC": ("Greece", "Europe", "Southern Europe", 39.0, 22.0, "EUR"),
    "HKG": ("Hong Kong", "Asia", "Eastern Asia", 22.25, 114.17, "HKD"),
    "SGP": ("Singapore", "Asia", "South-Eastern Asia", 1.37, 103.8, "SGD"),
    "LBN": ("Lebanon", "Asia", "Western Asia", 33.83, 35.83, "LBP"),
    "ARE": ("United Arab Emirates", "Asia", "Western Asia", 24.0, 54.0, "AED"),
    "SAU": ("Saudi Arabia", "Asia", "Western Asia", 25.0, 45.0, "SAR"),
    "CZE": ("Czechia", "Europe", "Central Europe", 49.75, 15.5, "CZK"),
    "CAN": ("Canada", "Americas", "North America", 60.0, -95.0, "CAD"),
    "BRA": ("Brazil", "Americas", "South America", -10.0, -55.0, "BRL"),
    "USA": ("United States", "Americas", "North America", 38.0, -97.0, "USD"),
    "MLT": ("Malta", "Europe", "Southern Europe", 35.83, 14.58, "EUR"),
    "ZAF": ("South Africa", "Africa", "Southern Africa", -29.0, 24.0, "ZAR"),
    "CHN": ("China", "Asia", "Eastern Asia", 35.0, 105.0, "CNY"),
    "IND": ("India", "Asia", "Southern Asia", 20.0, 77.0, "INR"),
    "TUR": ("Türkiye", "Asia", "Western Asia", 39.0, 35.0, "TRY"),
    "VNM": ("Vietnam", "Asia", "South-Eastern Asia", 16.17, 107.83, "VND"),
    "PAK": ("Pakistan", "Asia", "Southern Asia", 30.0, 70.0, "PKR"),
    "MEX": ("Mexico", "Americas", "North America", 23.0, -102.0, "MXN"),
}


def fetch_countries(codes: list[str]) -> pd.DataFrame:
    """Fetch country facts from REST Countries; falls back to the static table per country."""
    codes = sorted({c for c in codes if c})
    rows, missing = [], set(codes)
    s = get_settings()
    try:
        data = get_json(f"{s.restcountries_url}/alpha", params={
            "codes": ",".join(codes), "fields": "name,cca3,region,subregion,latlng,currencies"})
        for c in data if isinstance(data, list) else []:
            latlng = c.get("latlng") or [None, None]
            cur = next(iter((c.get("currencies") or {}).keys()), None)
            rows.append(dict(country_code=c["cca3"], name=c["name"]["common"], region=c.get("region"),
                             subregion=c.get("subregion"), lat=latlng[0], lng=latlng[1], currency=cur,
                             source="restcountries"))
            missing.discard(c["cca3"])
        log.info("REST Countries: fetched %d countries", len(rows))
    except ExternalAPIError as exc:
        log.warning("REST Countries unavailable (%s); using static fallback", exc)
    for code in sorted(missing):
        if code in STATIC:
            n, r, sr, lat, lng, cur = STATIC[code]
            rows.append(dict(country_code=code, name=n, region=r, subregion=sr, lat=lat, lng=lng, currency=cur,
                             source="static_fallback"))
    return pd.DataFrame(rows, columns=["country_code", "name", "region", "subregion", "lat", "lng", "currency", "source"])
