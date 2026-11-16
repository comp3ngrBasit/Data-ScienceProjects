"""End-to-end pipeline: data -> forecast -> backtest -> inventory plan -> outputs/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_sales
from src.forecast import HORIZON, fit_and_backtest, forecast_future
from src.inventory import SERVICE_LEVEL, plan, simulate_policy

OUT = Path(__file__).parent / "outputs"


def current_stock(sales: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Sample on-hand snapshot (replace with your WMS/ERP stock levels)."""
    rng = np.random.default_rng(seed)
    recent = sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=28)]
    avg = recent.groupby(["sku", "warehouse"])["units"].mean().reset_index()
    avg["on_hand"] = (avg["units"] * rng.choice([2, 6, 11, 16, 24, 60], size=len(avg))).round(0)
    return avg[["sku", "warehouse", "on_hand"]]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sales = load_sales()
    print(f"Loaded {len(sales):,} rows, {sales['sku'].nunique()} SKUs, {sales['warehouse'].nunique()} warehouses")

    res = fit_and_backtest(sales)
    m = res["metrics"]
    print(f"Backtest WAPE  model {m['model_wape']:.1%} | 28-day avg {m['ma28_wape']:.1%} | last year {m['lastyear_wape']:.1%}")

    future = forecast_future(sales, res["models"])
    inv = plan(future, res["resid_std"], current_stock(sales))
    sim = simulate_policy(res["test"], res["resid_std"])

    sim_sum = sim.groupby("policy").agg(stockout_days=("stockout_days", "sum"), lost_units=("lost_units", "sum"),
                                         demand=("demand", "sum"))
    sim["stock_value"] = sim["avg_stock"] * sim["unit_cost"]
    sim_sum["avg_stock_value"] = sim.groupby("policy")["stock_value"].sum()
    sim_sum["fill_rate"] = 1 - sim_sum["lost_units"] / sim_sum["demand"]

    sales.to_csv(OUT / "sales_history.csv", index=False)
    res["test"].to_csv(OUT / "backtest.csv", index=False)
    res["per_sku"].to_csv(OUT / "backtest_by_sku.csv", index=False)
    res["importance"].to_csv(OUT / "feature_importance.csv", index=False)
    future.to_csv(OUT / "forecast_next_28_days.csv", index=False)
    inv.to_csv(OUT / "inventory_plan.csv", index=False)
    sim.to_csv(OUT / "policy_simulation.csv", index=False)

    summary = {
        "horizon_days": HORIZON, "service_level": SERVICE_LEVEL,
        "history_start": str(sales["date"].min().date()), "history_end": str(sales["date"].max().date()),
        "n_skus": int(sales["sku"].nunique()), "n_warehouses": int(sales["warehouse"].nunique()),
        "n_rows": int(len(sales)), "metrics": m,
        "forecast_units_28d": float(future["forecast"].sum()),
        "status_counts": inv["status"].value_counts().to_dict(),
        "excess_value": float(inv["excess_value"].sum()),
        "simulation": sim_sum.reset_index().to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
