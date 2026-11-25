"""Small shared helpers: artifact IO, hashing, JSON serialisation, seeding, optional DB mirror."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATABASE_URL, RANDOM_SEED, RESULTS_DIR
from app.utils.logging import get_logger

log = get_logger("helpers")


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return _to_jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def module_dir(module: str) -> Path:
    d = RESULTS_DIR / module
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj: Any, module: str, name: str) -> Path:
    p = module_dir(module) / f"{name}.json"
    p.write_text(json.dumps(_to_jsonable(obj), indent=2))
    return p


def load_json(module: str, name: str) -> Any:
    p = RESULTS_DIR / module / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_table(df: pd.DataFrame, module: str, name: str, mirror_db: bool = True) -> Path:
    p = module_dir(module) / f"{name}.parquet"
    out = df.copy()
    # parquet needs string column names
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(p)
    if mirror_db and DATABASE_URL:
        write_db(out, f"{module}__{name}")
    return p


def load_table(module: str, name: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / module / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def write_db(df: pd.DataFrame, table: str) -> None:
    """Mirror a result table into PostgreSQL (or any SQLAlchemy DB). Failures are logged, not raised."""
    try:
        from sqlalchemy import create_engine

        eng = create_engine(DATABASE_URL)
        d = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
        d.to_sql(table[:63], eng, if_exists="replace", index=False, chunksize=5000)
    except Exception as exc:  # noqa: BLE001 - DB is an optional mirror
        log.warning("DB mirror of %s failed: %s", table, exc)


def safe_mape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-6) -> float:
    """MAPE over observations with |y| > eps (zero/near-zero actuals are excluded, not divided by)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.abs(y) > eps
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.any() else float("nan")
