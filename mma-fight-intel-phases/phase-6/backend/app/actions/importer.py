"""Import annotations from CSV / XLSX files (e.g. BoxingVI's per-video annotation sheets).

Accepted columns (case-insensitive, first match wins):
* start: ``start_frame`` | ``start`` | ``start_time`` | ``start_s``
* end:   ``end_frame``   | ``end``   | ``end_time``   | ``end_s``
* label: ``punch_class`` | ``class`` | ``label`` | ``action`` | ``category``
* optional: ``track_id``, ``fighter`` (fighter name), ``confidence``, ``reaction``
If no header matches, the first three columns are used as (start, end, label).
``units`` decides whether start/end are frames (converted with the video fps) or seconds.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

from app.actions.taxonomy import BOXINGVI_MAP_ORTHODOX, BOXINGVI_MAP_SOUTHPAW, normalise_key

START = ["start_frame", "start", "start_time", "start_s", "begin"]
END = ["end_frame", "end", "end_time", "end_s", "stop"]
LABEL = ["punch_class", "class", "label", "action", "category", "punch"]


@dataclass
class ParsedRow:
    start: float
    end: float
    category: str
    raw_label: str
    track_id: int | None = None
    fighter_name: str | None = None
    confidence: float = 1.0
    reaction: str | None = None


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content), sep=None, engine="python")


def _pick(cols: dict[str, str], options: list[str]) -> str | None:
    for o in options:
        if o in cols:
            return cols[o]
    return None


def parse_rows(df: pd.DataFrame, units: str, fps: float, label_map: str, custom_map: dict[str, str] | None) -> tuple[list[ParsedRow], list[str]]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    s_col, e_col, l_col = _pick(cols, START), _pick(cols, END), _pick(cols, LABEL)
    if not (s_col and e_col and l_col):
        if df.shape[1] < 3:
            raise ValueError("Need at least three columns: start, end, label")
        s_col, e_col, l_col = df.columns[:3]
    mapping = {"boxingvi_orthodox": BOXINGVI_MAP_ORTHODOX, "boxingvi_southpaw": BOXINGVI_MAP_SOUTHPAW}.get(label_map, {})
    mapping = {**mapping, **{k.lower(): v for k, v in (custom_map or {}).items()}}
    rows, errors = [], []
    for i, r in df.iterrows():
        try:
            raw = str(r[l_col]).strip()
            key = mapping.get(raw.lower().replace("_", " ")) or mapping.get(raw.lower()) or normalise_key(raw)
            if key is None:
                errors.append(f"row {i + 2}: unknown action label '{raw}'")
                continue
            a, b = float(r[s_col]), float(r[e_col])
            if units == "frames":
                a, b = a / fps, (b + 1) / fps
            if b <= a:
                errors.append(f"row {i + 2}: end is not after start")
                continue
            pr = ParsedRow(round(a, 4), round(b, 4), key, raw)
            if "track_id" in cols and pd.notna(r[cols["track_id"]]):
                pr.track_id = int(r[cols["track_id"]])
            if "fighter" in cols and pd.notna(r[cols["fighter"]]):
                pr.fighter_name = str(r[cols["fighter"]]).strip()
            if "confidence" in cols and pd.notna(r[cols["confidence"]]):
                pr.confidence = max(0.0, min(1.0, float(r[cols["confidence"]])))
            if "reaction" in cols and pd.notna(r[cols["reaction"]]):
                pr.reaction = str(r[cols["reaction"]]).strip()
            rows.append(pr)
        except (ValueError, TypeError) as exc:
            errors.append(f"row {i + 2}: {exc}")
    return rows, errors
