"""Logistics optimization with OR-Tools.

1. Warehouse allocation (linear program, GLOP): assign regional demand to distribution centres,
   minimising transport cost (distance_km x cost per unit-km + handling per unit) subject to DC capacity.
     * network "EU": REAL UCI customer-country demand (last 90 days, planning SKUs) -> simulated EU DCs.
     * network "BR": REAL Olist customer counts per state (demand weight) -> simulated Brazil hubs.
   Baseline for comparison: everything shipped from the largest DC.
2. Delivery routing (capacitated VRP, OR-Tools routing): vehicles from a hub visit demand points
   (default: Brazil state capitals weighted by REAL Olist customer counts) minimising total distance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.linear_solver import pywraplp

from scip.core.db import get_engine, read_sql, session_scope
from scip.core.models import LogisticsPlan
from scip.ingestion.olist import haversine_km
from scip.ingestion.pipeline import as_of_date

COST_PER_UNIT_KM = 0.0004
HANDLING_PER_UNIT = 0.05


def _network(network: str, engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    loc = read_sql("SELECT * FROM locations", engine=engine)
    if network == "EU":
        wh = loc[(loc["location_type"] == "warehouse") & (loc["country_code"] != "BRA")]
        as_of = as_of_date(engine)
        dem = read_sql("SELECT s.location_id, SUM(s.quantity) units FROM sales_daily s JOIN products p USING(sku) "
                       "WHERE p.is_planning_sku AND s.date > :d GROUP BY s.location_id",
                       {"d": (as_of - pd.Timedelta(days=90)).date()}, engine=engine)
        dem = dem.merge(loc, on="location_id").dropna(subset=["lat", "lng"])
        scale = 90 / 30   # DC capacities are monthly
        wh = wh.assign(capacity=wh["capacity_units"] * scale)
    elif network == "BR":
        wh = loc[(loc["location_type"] == "warehouse") & (loc["country_code"] == "BRA")]
        dem = loc[loc["location_id"].str.startswith("BR-CUST-")].copy()
        if dem.empty:
            raise ValueError("No Olist customer data loaded for the Brazil network.")
        dem["units"] = dem["capacity_units"]          # customer count as demand weight
        wh = wh.assign(capacity=wh["capacity_units"])
    else:
        raise ValueError("network must be 'EU' or 'BR'")
    return wh.reset_index(drop=True), dem.reset_index(drop=True)


def warehouse_allocation(network: str = "EU", engine=None, persist: bool = True) -> dict:
    engine = engine or get_engine()
    wh, dem = _network(network, engine)
    dist = haversine_km(wh["lat"].values[:, None], wh["lng"].values[:, None], dem["lat"].values[None, :], dem["lng"].values[None, :])
    cost = dist * COST_PER_UNIT_KM + HANDLING_PER_UNIT
    total_demand, total_cap = dem["units"].sum(), wh["capacity"].sum()
    if total_cap < total_demand:
        wh["capacity"] *= total_demand / total_cap * 1.05     # documented: capacities scaled to keep feasible
    solver = pywraplp.Solver.CreateSolver("GLOP")
    x = [[solver.NumVar(0, solver.infinity(), f"x{i}_{j}") for j in range(len(dem))] for i in range(len(wh))]
    for j in range(len(dem)):
        solver.Add(sum(x[i][j] for i in range(len(wh))) == float(dem.loc[j, "units"]))
    for i in range(len(wh)):
        solver.Add(sum(x[i][j] for j in range(len(dem))) <= float(wh.loc[i, "capacity"]))
    solver.Minimize(sum(cost[i, j] * x[i][j] for i in range(len(wh)) for j in range(len(dem))))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Allocation LP not optimal")
    flows = []
    for i in range(len(wh)):
        for j in range(len(dem)):
            v = x[i][j].solution_value()
            if v > 1e-6:
                flows.append(dict(warehouse=wh.loc[i, "location_id"], warehouse_name=wh.loc[i, "name"],
                                  destination=dem.loc[j, "location_id"], destination_name=dem.loc[j, "name"],
                                  units=round(v, 1), distance_km=round(float(dist[i, j]), 1),
                                  cost=round(v * cost[i, j], 2),
                                  dest_lat=float(dem.loc[j, "lat"]), dest_lng=float(dem.loc[j, "lng"]),
                                  wh_lat=float(wh.loc[i, "lat"]), wh_lng=float(wh.loc[i, "lng"])))
    big = int(wh["capacity"].idxmax())
    baseline = float((dem["units"].values * cost[big]).sum())
    optimal = solver.Objective().Value()
    util = pd.DataFrame(flows).groupby("warehouse")["units"].sum() if flows else pd.Series(dtype=float)
    result = {"network": network, "objective_cost": round(optimal, 2), "baseline_single_dc_cost": round(baseline, 2),
              "saving_pct": round(1 - optimal / baseline, 4) if baseline else None, "flows": flows,
              "utilisation": {w: round(float(util.get(w, 0) / wh.set_index("location_id").loc[w, "capacity"]), 3)
                              for w in wh["location_id"]},
              "demand_basis": "UCI last 90 days (units)" if network == "EU" else "Olist customers per state (weight)"}
    if persist:
        with session_scope(engine) as s:
            plan = LogisticsPlan(plan_type=f"allocation_{network}", objective_value=float(optimal), status="optimal", details=result)
            s.add(plan)
            s.flush()
            result["plan_id"] = plan.plan_id
    return result


def solve_vrp(stops: pd.DataFrame, depot: tuple[float, float], vehicles: int = 4, capacity: float | None = None,
              time_limit: int = 5) -> dict:
    """Capacitated VRP. stops: columns name, lat, lng, demand. Distances: haversine km (integer metres)."""
    pts = np.vstack([[depot[0], depot[1]], stops[["lat", "lng"]].values])
    D = haversine_km(pts[:, 0][:, None], pts[:, 1][:, None], pts[:, 0][None, :], pts[:, 1][None, :])
    dm = np.round(D * 1000).astype(int)
    demand = [0] + [int(np.ceil(v)) for v in stops["demand"]]
    capacity = capacity or int(np.ceil(sum(demand) / vehicles * 1.25))
    mgr = pywrapcp.RoutingIndexManager(len(pts), vehicles, 0)
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitCallback(lambda i, j: int(dm[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    dcb = routing.RegisterUnaryTransitCallback(lambda i: demand[mgr.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(dcb, 0, [int(capacity)] * vehicles, True, "Capacity")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("No feasible route found (check vehicle capacity)")
    routes, total = [], 0.0
    for v in range(vehicles):
        idx, seq, load, dist = routing.Start(v), [], 0, 0
        while not routing.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node > 0:
                r = stops.iloc[node - 1]
                seq.append(dict(stop=r["name"], lat=float(r["lat"]), lng=float(r["lng"]), demand=demand[node]))
                load += demand[node]
            nxt = sol.Value(routing.NextVar(idx))
            dist += dm[mgr.IndexToNode(idx), mgr.IndexToNode(nxt)]
            idx = nxt
        if seq:
            routes.append(dict(vehicle=v + 1, stops=seq, load=int(load), distance_km=round(float(dist) / 1000, 1)))
            total += float(dist) / 1000
    return {"vehicles_used": len(routes), "capacity": int(capacity), "total_distance_km": round(total, 1), "routes": routes}


def brazil_linehaul(engine=None, hub: str = "BR-HUB-SP", vehicles: int = 5, persist: bool = True) -> dict:
    engine = engine or get_engine()
    loc = read_sql("SELECT * FROM locations", engine=engine)
    h = loc[loc["location_id"] == hub].iloc[0]
    stops = loc[loc["location_id"].str.startswith("BR-CUST-")].copy()
    if stops.empty:
        raise ValueError("No Olist customer locations loaded.")
    stops["demand"] = np.ceil(stops["capacity_units"] / 100)      # pallets ~ per 100 customers (assumption)
    stops = stops[stops["state"] != h["state"]]                    # hub state served locally
    stops["name"] = stops["city"] + " (" + stops["state"] + ")"
    res = solve_vrp(stops[["name", "lat", "lng", "demand"]].reset_index(drop=True), (h["lat"], h["lng"]), vehicles)
    res.update(hub=hub, hub_name=h["name"], hub_lat=float(h["lat"]), hub_lng=float(h["lng"]),
               demand_basis="Olist customers per state / 100 = pallets (assumption)")
    if persist:
        with session_scope(engine) as s:
            plan = LogisticsPlan(plan_type="routing_BR", objective_value=float(res["total_distance_km"]), status="solved", details=res)
            s.add(plan)
            s.flush()
            res["plan_id"] = plan.plan_id
    return res
