"""Inventory optimization.

Turns demand forecasts into replenishment decisions per SKU and warehouse:
safety stock, reorder point, economic order quantity and stock status.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

SERVICE_LEVEL = 0.95      # target probability of no stockout during lead time
ORDER_COST = 45.0         # fixed cost per purchase order
HOLDING_RATE = 0.25       # annual holding cost as a share of unit cost


def plan(future: pd.DataFrame, resid_std: pd.DataFrame, on_hand: pd.DataFrame,
         service_level: float = SERVICE_LEVEL) -> pd.DataFrame:
    z = norm.ppf(service_level)
    g = future.groupby(["sku", "category", "warehouse"])
    daily = g["forecast"].mean().rename("daily_forecast")
    meta = g[["unit_cost", "lead_time_days"]].first()
    p = pd.concat([daily, meta], axis=1).reset_index()
    p = p.merge(resid_std, on=["sku", "warehouse"]).merge(on_hand, on=["sku", "warehouse"])

    lt = p["lead_time_days"]
    p["lead_time_demand"] = p["daily_forecast"] * lt
    p["safety_stock"] = z * p["resid_std"] * np.sqrt(lt)
    p["reorder_point"] = p["lead_time_demand"] + p["safety_stock"]
    annual = p["daily_forecast"] * 365
    holding = HOLDING_RATE * p["unit_cost"]
    p["order_qty"] = np.sqrt(2 * annual * ORDER_COST / holding)
    p["days_of_cover"] = p["on_hand"] / p["daily_forecast"]
    p["max_stock"] = p["reorder_point"] + p["order_qty"]

    def status(r):
        if r.on_hand < r.safety_stock:
            return "Critical"
        if r.on_hand < r.reorder_point:
            return "Reorder"
        if r.on_hand > 1.5 * r.max_stock:
            return "Overstock"
        return "Healthy"

    p["status"] = p.apply(status, axis=1)
    p["suggested_order"] = np.where(p["status"].isin(["Critical", "Reorder"]),
                                    np.ceil(p["max_stock"] - p["on_hand"]), 0)
    p["excess_value"] = np.where(p["status"] == "Overstock",
                                 (p["on_hand"] - p["max_stock"]) * p["unit_cost"], 0)
    for c in ["daily_forecast", "lead_time_demand", "safety_stock", "reorder_point", "order_qty", "max_stock"]:
        p[c] = p[c].round(0)
    return p


def simulate_policy(test: pd.DataFrame, resid_std: pd.DataFrame, service_level: float = SERVICE_LEVEL) -> pd.DataFrame:
    """Replay the backtest period day by day and compare two reorder policies:
    the forecast-driven policy above, and a fixed min/max rule based on the
    average of the previous 28 days (a common manual approach)."""
    z = norm.ppf(service_level)
    rows = []
    for (sku, wh), d in test.sort_values("date").groupby(["sku", "warehouse"]):
        lt = int(d["lead_time_days"].iloc[0])
        sd = float(resid_std.query("sku == @sku and warehouse == @wh")["resid_std"].iloc[0])
        demand = d["units"].to_numpy()
        for name, rate in (("AI forecast", d["forecast"].to_numpy()), ("Manual min/max", d["naive_ma28"].to_numpy())):
            if name == "AI forecast":
                ss = z * sd * np.sqrt(lt)
            else:
                ss = 0.5 * rate[0] * lt  # rule of thumb: half a lead time of cover
            stock = rate[0] * lt + ss + rate[0] * 14
            pipeline = []  # (arrival_day, qty)
            stockout_days = lost = held = 0.0
            for i, dem in enumerate(demand):
                stock += sum(q for a, q in pipeline if a == i)
                pipeline = [(a, q) for a, q in pipeline if a > i]
                sold = min(stock, dem)
                if dem > stock:
                    stockout_days += 1
                    lost += dem - stock
                stock -= sold
                held += stock
                position = stock + sum(q for _, q in pipeline)
                rop = rate[i] * lt + ss
                if position < rop:
                    pipeline.append((i + lt, rate[i] * 14 + rop - position))
            rows.append({"sku": sku, "warehouse": wh, "policy": name,
                         "stockout_days": stockout_days, "lost_units": lost,
                         "avg_stock": held / len(demand), "demand": demand.sum(),
                         "unit_cost": d["unit_cost"].iloc[0]})
    return pd.DataFrame(rows)
