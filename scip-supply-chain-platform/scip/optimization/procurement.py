"""Procurement optimization — supplier recommendation and order allocation with OR-Tools CP-SAT.

For each SKU requirement (quantity, latest acceptable lead time) the solver chooses how much to order from
each candidate supplier:

  minimise  sum_s  [ unit_cost_s * x_s                                (purchase cost)
                   + risk_weight * risk_s * unit_cost_s * x_s           (expected cost of late delivery)
                   + lead_weight * lead_s * unit_cost_s * x_s / 365    (capital tied up in transit)
                   + ordering_cost * y_s ]                             (fixed cost per PO)
            + shortfall_penalty * shortfall
  s.t.      sum_s x_s + shortfall = required
            moq_s * y_s <= x_s <= capacity_s * y_s     (y_s binary: supplier used)
            lead_s <= max_lead  (suppliers that cannot deliver in time are excluded)
            optional total budget across all requirements

risk_s = model-predicted late probability for the supplier (delivery-risk model / supplier score).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from scip.core.config import get_settings
from scip.core.db import get_engine, read_sql
from scip.ml.supplier_scoring import latest_scores

SCALE = 100  # CP-SAT needs integers: costs in cents


@dataclass
class Requirement:
    sku: str
    quantity: int
    max_lead_days: float | None = None


def candidate_table(engine=None, skus: list[str] | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    sp = read_sql("SELECT sp.*, s.name AS supplier_name, s.country_code FROM supplier_products sp "
                  "JOIN suppliers s USING(supplier_id)", engine=engine)
    if skus is not None:
        sp = sp[sp["sku"].isin(skus)]
    sc = latest_scores(engine)
    if not sc.empty:
        sp = sp.merge(sc[["supplier_id", "score", "tier", "predicted_late_risk", "on_time_rate"]], on="supplier_id", how="left")
    else:
        sp["score"], sp["tier"], sp["predicted_late_risk"], sp["on_time_rate"] = np.nan, "Unscored", np.nan, np.nan
    sp["risk"] = sp["predicted_late_risk"].fillna(1 - sp["on_time_rate"].fillna(0.7)).clip(0, 1)
    return sp


def recommend_suppliers(sku: str, quantity: int, engine=None) -> pd.DataFrame:
    """Rank candidate suppliers for one SKU by total expected cost (transparent single-supplier view)."""
    s = get_settings()
    c = candidate_table(engine, [sku]).copy()
    if c.empty:
        return c
    q = max(quantity, 1)
    c["order_qty"] = np.maximum(q, c["moq"])
    c["purchase_cost"] = c["order_qty"] * c["unit_cost"]
    c["risk_cost"] = 0.5 * c["risk"] * c["purchase_cost"]
    c["transit_capital_cost"] = s.holding_cost_rate * c["lead_time_days"] / 365 * c["purchase_cost"]
    c["total_expected_cost"] = c["purchase_cost"] + c["risk_cost"] + c["transit_capital_cost"] + s.ordering_cost
    c = c.sort_values("total_expected_cost").reset_index(drop=True)
    c["rank"] = np.arange(1, len(c) + 1)
    return c


def optimize_allocation(reqs: list[Requirement], engine=None, budget: float | None = None,
                        risk_weight: float = 0.5, time_limit: float = 10.0) -> dict:
    s = get_settings()
    cands = candidate_table(engine, [r.sku for r in reqs])
    model = cp_model.CpModel()
    xs, ys, shorts, rows = {}, {}, {}, []
    obj = []
    for r in reqs:
        c = cands[cands["sku"] == r.sku]
        if r.max_lead_days is not None:
            c = c[c["lead_time_days"] <= r.max_lead_days]
        shorts[r.sku] = model.NewIntVar(0, r.quantity, f"short_{r.sku}")
        terms = []
        for _, row in c.iterrows():
            key = (r.sku, row["supplier_id"])
            cap = int(min(row["capacity_per_month"] or r.quantity * 2, max(r.quantity * 2, row["moq"])))
            x = model.NewIntVar(0, cap, f"x_{r.sku}_{row['supplier_id']}")
            y = model.NewBoolVar(f"y_{r.sku}_{row['supplier_id']}")
            model.Add(x >= int(row["moq"]) * y)
            model.Add(x <= cap * y)
            unit = row["unit_cost"] * (1 + risk_weight * row["risk"] + s.holding_cost_rate * row["lead_time_days"] / 365)
            obj.append(int(round(unit * SCALE)) * x + int(round(s.ordering_cost * SCALE)) * y)
            xs[key], ys[key] = x, y
            terms.append(x)
            rows.append(dict(key=key, unit_cost=row["unit_cost"], risk=row["risk"], lead=row["lead_time_days"],
                             name=row["supplier_name"], tier=row["tier"]))
        model.Add(sum(terms) + shorts[r.sku] >= r.quantity)
        max_unit = float(c["unit_cost"].max()) if not c.empty else 100.0
        obj.append(int(round(max_unit * 10 * SCALE)) * shorts[r.sku])
    if budget is not None:
        info = {rw["key"]: rw for rw in rows}
        model.Add(sum(int(round(info[k]["unit_cost"] * SCALE)) * v for k, v in xs.items()) <= int(budget * SCALE))
    model.Minimize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 4
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    allocations = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        info = {rw["key"]: rw for rw in rows}
        for k, v in xs.items():
            q = solver.Value(v)
            if q > 0:
                i = info[k]
                allocations.append(dict(sku=k[0], supplier_id=k[1], supplier_name=i["name"], quantity=q,
                                        unit_cost=i["unit_cost"], total_value=round(q * i["unit_cost"], 2),
                                        predicted_late_risk=round(i["risk"], 3), lead_time_days=i["lead"], tier=i["tier"]))
    shortfalls = {sku: solver.Value(v) for sku, v in shorts.items()} if allocations or status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else {}
    return {"status": status_name, "objective": solver.ObjectiveValue() / SCALE if allocations else None,
            "allocations": allocations, "shortfalls": {k: v for k, v in shortfalls.items() if v > 0},
            "solve_seconds": solver.WallTime()}
