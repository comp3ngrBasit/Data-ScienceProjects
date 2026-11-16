"""Sample data generator.

Creates daily sales history for a set of SKUs across warehouses, with trend,
weekly and yearly seasonality, price changes, promotions and noise. The data is
synthetic so the project can run end to end without any private data; swap
`load_sales()` for your own source to use real history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKUS = {
    # sku: (category, base daily units, yearly amplitude, trend per year, unit cost, lead time days)
    "SKU-1001": ("Beverages", 120, 0.35, 0.18, 1.80, 7),
    "SKU-1002": ("Snacks", 85, 0.15, 0.10, 2.40, 10),
    "SKU-1003": ("Household", 40, 0.10, 0.05, 6.50, 14),
    "SKU-1004": ("Personal Care", 55, 0.08, 0.22, 4.20, 12),
    "SKU-1005": ("Frozen", 65, 0.40, -0.04, 3.10, 5),
    "SKU-1006": ("Dairy", 150, 0.05, 0.06, 1.20, 3),
}
WAREHOUSES = {"Karachi DC": 1.0, "Lahore DC": 0.8}


def make_sales(start: str = "2024-01-01", end: str = "2026-09-30", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates)) / 365.25
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    weekly = np.array([0.92, 0.90, 0.95, 1.00, 1.12, 1.20, 1.05])[dow]

    frames = []
    for sku, (cat, base, amp, trend, cost, lead) in SKUS.items():
        for wh, scale in WAREHOUSES.items():
            phase = rng.uniform(0, 2 * np.pi)
            yearly = 1 + amp * np.sin(2 * np.pi * doy / 365.25 + phase)
            level = base * scale * (1 + trend * t)
            promo = (rng.random(len(dates)) < 0.06).astype(int)
            # promotions run for a few days once started
            promo = np.convolve(promo, np.ones(3), mode="same").clip(0, 1).astype(int)
            price = cost * 1.6 * (1 - 0.15 * promo) * (1 + 0.03 * np.floor(t))
            lift = 1 + 0.45 * promo
            mean = level * yearly * weekly * lift
            units = rng.negative_binomial(n=20, p=20 / (20 + mean))
            frames.append(pd.DataFrame({
                "date": dates, "sku": sku, "category": cat, "warehouse": wh,
                "units": units, "price": price.round(2), "promo": promo,
                "unit_cost": cost, "lead_time_days": lead,
            }))
    return pd.concat(frames, ignore_index=True)


def load_sales() -> pd.DataFrame:
    return make_sales()
