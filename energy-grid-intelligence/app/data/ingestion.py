"""Dataset acquisition and raw ingestion.

Each source is fetched from its official URL first; mirrors are only accepted when the file's
SHA-256 equals the hash of the official release. Weather comes from the Open-Meteo archive API
when reachable, otherwise from the OPSD weather data package (Germany only).
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests

from app.config import (EXTERNAL_DIR, GRID_WEATHER_POINTS, HOUSEHOLD_LOCATION, OPEN_METEO_ARCHIVE,
                        OPEN_METEO_HOURLY, OPSD_SOURCE, OPSD_WEATHER_SOURCE, RAW_DIR, UCI_SOURCE, Source)
from app.utils.helpers import sha256_file
from app.utils.logging import get_logger

log = get_logger("ingestion")


class DataUnavailableError(RuntimeError):
    """Raised when no source for a dataset could be fetched or validated."""


def download(source: Source, dest_dir: Path = RAW_DIR, timeout: int = 600) -> Path:
    dest = dest_dir / source.filename
    if dest.exists() and (source.sha256 is None or sha256_file(dest) == source.sha256):
        log.info("%s already present and verified", dest.name)
        return dest
    errors = []
    for url in source.urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            log.info("Downloading %s from %s", source.name, url)
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if source.sha256 and sha256_file(tmp) != source.sha256:
                raise ValueError("SHA-256 mismatch (file differs from the official release)")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
            log.warning("Source failed: %s -> %s", url, exc)
            tmp.unlink(missing_ok=True)
    raise DataUnavailableError(f"Could not acquire {source.name}:\n" + "\n".join(errors))


def load_household_raw(path: Path | None = None) -> pl.DataFrame:
    """Read the 2M-row minute file with Polars ('?' marks missing measurements)."""
    path = path or download(UCI_SOURCE)
    if path.suffix == ".zip":
        txt = RAW_DIR / "household_power_consumption.txt"
        if not txt.exists():
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    raise DataUnavailableError(f"Corrupted member in archive: {bad}")
                z.extract("household_power_consumption.txt", RAW_DIR)
        path = txt
    num = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
           "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = pl.read_csv(path, separator=";", null_values=["?", ""], infer_schema_length=0)
    missing_cols = set(["Date", "Time", *num]) - set(df.columns)
    if missing_cols:
        raise DataUnavailableError(f"Household file missing columns: {missing_cols}")
    df = df.with_columns(
        pl.concat_str([pl.col("Date"), pl.col("Time")], separator=" ")
        .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False).alias("timestamp"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in num],
    ).drop(["Date", "Time"])
    return df


def load_opsd_raw(path: Path | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    path = path or download(OPSD_SOURCE)
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "utc_timestamp" not in df.columns:
        raise DataUnavailableError("OPSD file has no utc_timestamp column")
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    return df.set_index("utc_timestamp").sort_index()


def opsd_columns(path: Path | None = None) -> list[str]:
    path = path or download(OPSD_SOURCE)
    return list(pd.read_csv(path, nrows=0).columns)


# ---------------------------------------------------------------- weather
def _open_meteo(lat: float, lon: float, start: str, end: str, timeout: int = 60) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(OPEN_METEO_HOURLY), "timezone": "UTC"}
    for attempt in range(3):
        try:
            r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=timeout)
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df.set_index("time")
        except Exception as exc:  # noqa: BLE001
            log.warning("Open-Meteo attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    raise DataUnavailableError("Open-Meteo archive API unreachable")


def _rename_open_meteo(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"temperature_2m": "temperature", "relative_humidity_2m": "humidity",
                              "wind_speed_10m": "wind_speed", "shortwave_radiation": "solar_radiation",
                              "precipitation": "precipitation", "weather_code": "weather_code"})


def acquire_household_weather() -> tuple[pd.DataFrame | None, dict]:
    """Hourly weather at the household location for the UCI period (Open-Meteo only)."""
    cache = EXTERNAL_DIR / "weather_household_openmeteo.parquet"
    meta = {"source": "Open-Meteo archive (ERA5)", "location": HOUSEHOLD_LOCATION["name"]}
    if cache.exists():
        return pd.read_parquet(cache), meta
    try:
        df = _rename_open_meteo(_open_meteo(HOUSEHOLD_LOCATION["latitude"], HOUSEHOLD_LOCATION["longitude"],
                                            "2006-12-16", "2010-11-26"))
        df.to_parquet(cache)
        return df, meta
    except DataUnavailableError as exc:
        meta.update(status="unavailable", reason=str(exc))
        log.warning("Household weather unavailable: %s", exc)
        return None, meta


def acquire_grid_weather(country: str, start: str, end: str) -> tuple[pd.DataFrame | None, dict]:
    """Population-weighted national hourly weather.

    1) Open-Meteo, weighted over GRID_WEATHER_POINTS (temperature, humidity, wind, radiation, precipitation)
    2) fallback: OPSD weather package extract (temperature, wind speed, direct+diffuse radiation; DE 2013-2016)
    """
    cache = EXTERNAL_DIR / f"weather_{country}_openmeteo.parquet"
    if cache.exists():
        return pd.read_parquet(cache), {"source": "Open-Meteo archive (ERA5), population-weighted",
                                         "variables": list(pd.read_parquet(cache).columns)}
    points = GRID_WEATHER_POINTS.get(country)
    if points:
        try:
            frames, weights = [], []
            for _name, lat, lon, w in points:
                frames.append(_rename_open_meteo(_open_meteo(lat, lon, start, end)).drop(columns="weather_code"))
                weights.append(w)
            wts = np.array(weights) / np.sum(weights)
            df = sum(f * wt for f, wt in zip(frames, wts))
            df.to_parquet(cache)
            return df, {"source": "Open-Meteo archive (ERA5), population-weighted", "points": [p[0] for p in points]}
        except DataUnavailableError as exc:
            log.warning("Open-Meteo unavailable for %s (%s); trying OPSD weather package", country, exc)
    if country != "DE":
        return None, {"status": "unavailable", "reason": "No reachable weather source for this country"}
    try:
        path = EXTERNAL_DIR / OPSD_WEATHER_SOURCE.filename
        if not path.exists():
            path = download(OPSD_WEATHER_SOURCE, EXTERNAL_DIR)
        w = pd.read_csv(path, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
        w.index = pd.to_datetime(w.index, utc=True)
        df = pd.DataFrame({
            "temperature": w["temperature"],
            "wind_speed": w["windspeed"],
            # OPSD gives direct + diffuse horizontal irradiance (W/m2); their sum is global horizontal irradiance
            "solar_radiation": w["rad_horizontal"] + w["rad_diffuse"],
        })
        return df, {"source": OPSD_WEATHER_SOURCE.name, "citation": OPSD_WEATHER_SOURCE.citation,
                    "variables": list(df.columns),
                    "missing_variables": ["humidity", "precipitation", "weather_code"],
                    "note": "Open-Meteo was unreachable; humidity and precipitation are not in the OPSD package."}
    except Exception as exc:  # noqa: BLE001
        return None, {"status": "unavailable", "reason": str(exc)}
