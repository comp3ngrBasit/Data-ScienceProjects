"""Module 4 - Consumption anomaly detection engine.

Expected consumption is a *causal* seasonal baseline: the median of the same hour-of-week over the
previous four weeks (uses no future data). Six detectors run on the household hourly series; an hour is
a *consensus anomaly* when at least two independent detectors flag it. Data-quality findings come from
minute-level checks. Anomalies are statistical deviations only: they are never presented as proof of
theft, equipment failure or meter malfunction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import STL

from app.config import RANDOM_SEED
from app.data import preprocessing as pp
from app.utils.helpers import load_json, save_json, save_table

DISCLAIMER = ("Anomalies are statistical deviations from expected consumption. They do not by themselves prove "
              "electricity theft, equipment failure or meter malfunction; each requires on-site or metadata verification.")
CONTAMINATION = 0.005
MIN_ABS_DEVIATION_KWH = 0.5  # practical-significance floor for the statistical detectors


def baseline(y: pd.Series, weeks: int = 4, min_weeks: int = 3) -> pd.Series:
    lagged = pd.concat([y.shift(168 * k) for k in range(1, weeks + 1)], axis=1)
    return lagged.median(axis=1, skipna=True).where(lagged.notna().sum(axis=1) >= min_weeks)


def severity(z: float) -> str:
    a = abs(z)
    return "critical" if a >= 8 else "high" if a >= 6 else "medium" if a >= 4 else "low"


def anomaly_type(ts: pd.Timestamp, obs: float, exp: float) -> str:
    if obs >= exp:
        return "unusual overnight usage" if ts.hour < 5 else "consumption spike"
    return "sudden consumption drop"


def detect_hourly(hh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = hh["energy_kwh"]
    exp = baseline(y)
    resid = y - exp
    # robust rolling scale (MAD over previous 28 days, causal)
    mad = resid.abs().shift(1).rolling(24 * 28, min_periods=24 * 7).median() * 1.4826
    rz = resid / mad.replace(0, np.nan)
    roll_mu = resid.shift(1).rolling(24 * 28, min_periods=24 * 7).mean()
    roll_sd = resid.shift(1).rolling(24 * 28, min_periods=24 * 7).std()
    z = (resid - roll_mu) / roll_sd
    frame = pd.DataFrame({"observed": y, "expected": exp, "resid": resid, "robust_z": rz, "z": z,
                          "imputed": hh["imputed"]}).dropna(subset=["expected", "robust_z", "z"])
    frame = frame[~frame["imputed"]]  # never flag values we imputed ourselves
    flags = pd.DataFrame(index=frame.index)
    evidence = {}
    # 1) classical z-score on baseline residuals (rolling 28-day mean/std)
    big = frame["resid"].abs() >= MIN_ABS_DEVIATION_KWH
    flags["Z-score"] = (frame["z"].abs() > 4) & big
    evidence["Z-score"] = frame["z"].map(lambda v: f"residual z = {v:.2f} (rolling 28-day mean/std), threshold |z| > 4")
    # 2) IQR fence on residuals, per hour of day
    hr = frame.index.hour
    q1 = frame.groupby(hr)["resid"].transform(lambda s: s.quantile(0.25))
    q3 = frame.groupby(hr)["resid"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    flags["IQR"] = ((frame["resid"] < q1 - 3 * iqr) | (frame["resid"] > q3 + 3 * iqr)) & big
    evidence["IQR"] = (frame["resid"] - q3).where(frame["resid"] > q3, frame["resid"] - q1) / iqr
    evidence["IQR"] = evidence["IQR"].map(lambda v: f"residual {v:+.2f} IQR beyond the hour-of-day quartile (fence: 3 IQR)")
    # 3) seasonal residual: robust STL with weekly period on log1p(load) (household load is multiplicative).
    #    Robust STL fits the bulk tightly, so the MAD understates spread; residuals are scaled by their std.
    stl = STL(np.log1p(y.loc[frame.index.min():frame.index.max()].interpolate()), period=168, robust=True).fit()
    sr = stl.resid.reindex(frame.index)
    sr_z = (sr - sr.mean()) / sr.std()
    frame["stl_z"] = sr_z
    flags["Seasonal residual (STL)"] = (sr_z.abs() > 4) & big
    evidence["Seasonal residual (STL)"] = sr_z.map(lambda v: f"weekly-STL residual z = {v:.2f} (log scale), threshold |z| > 4")
    # 4-6) multivariate ML detectors on context features
    X = pd.DataFrame({"observed": frame["observed"], "expected": frame["expected"], "resid": frame["resid"],
                      "hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24),
                      "diff1": y.diff().reindex(frame.index), "roll_std_24": y.rolling(24).std().reindex(frame.index)}).fillna(0)
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=RANDOM_SEED).fit(Xs)
    flags["Isolation Forest"] = iso.predict(Xs) == -1
    s_iso = -iso.score_samples(Xs)
    evidence["Isolation Forest"] = pd.Series(s_iso, index=frame.index).map(
        lambda v: f"isolation score {v:.3f} (top {CONTAMINATION:.1%} of hours)")
    lof = LocalOutlierFactor(n_neighbors=50, contamination=CONTAMINATION)
    flags["Local Outlier Factor"] = lof.fit_predict(Xs) == -1
    evidence["Local Outlier Factor"] = pd.Series(-lof.negative_outlier_factor_, index=frame.index).map(
        lambda v: f"LOF = {v:.2f} (density ratio to 50 neighbours)")
    rng = np.random.default_rng(RANDOM_SEED)
    sub = rng.choice(len(Xs), min(8000, len(Xs)), replace=False)
    oc = OneClassSVM(nu=CONTAMINATION, gamma="scale").fit(Xs[sub])
    dec = oc.decision_function(Xs)
    flags["One-Class SVM"] = dec < np.quantile(dec, CONTAMINATION)
    evidence["One-Class SVM"] = pd.Series(dec, index=frame.index).map(lambda v: f"SVM decision value {v:.3f} (< {CONTAMINATION:.1%} quantile)")
    flags["n_methods"] = flags.sum(axis=1)
    # severity score = largest standardised deviation from the causal baseline
    frame["severity_score"] = frame[["robust_z", "z"]].abs().max(axis=1)
    rows = []
    for m in [c for c in flags.columns if c != "n_methods"]:
        idx = flags.index[flags[m]]
        for t in idx:
            o, e, r = frame.at[t, "observed"], frame.at[t, "expected"], frame.at[t, "robust_z"]
            sc = frame.at[t, "severity_score"]
            rows.append({"timestamp": t, "observed_kwh": o, "expected_kwh": e,
                         "deviation_pct": (o - e) / e * 100 if e > 0 else np.nan, "robust_z": r,
                         "severity_score": sc, "severity": severity(sc), "method": m, "type": anomaly_type(t, o, e),
                         "evidence": evidence[m][t], "n_methods_agreeing": int(flags.at[t, "n_methods"])})
    events = pd.DataFrame(rows).sort_values(["timestamp", "method"]).reset_index(drop=True)
    scored = frame.join(flags)
    return events, scored


def detect_daily(hh: pd.DataFrame) -> pd.DataFrame:
    d = hh["energy_kwh"].resample("D").sum()
    hourly_sd = hh["energy_kwh"].resample("D").std()
    imputed = hh["imputed"].resample("D").mean()
    exp = d.shift(1).rolling(28, min_periods=14).median()
    mad = (d - exp).abs().shift(1).rolling(56, min_periods=28).median() * 1.4826
    z = (d - exp) / mad
    vol_exp = hourly_sd.shift(1).rolling(28, min_periods=14).median()
    vol_mad = (hourly_sd - vol_exp).abs().shift(1).rolling(56, min_periods=28).median() * 1.4826
    vz = (hourly_sd - vol_exp) / vol_mad
    rows = []
    for t in d.index:
        if imputed[t] > 0.25 or np.isnan(z[t]):
            continue
        if abs(z[t]) > 4:
            rows.append({"date": t, "observed_kwh": d[t], "expected_kwh": exp[t], "deviation_pct": (d[t] / exp[t] - 1) * 100,
                         "robust_z": z[t], "severity": severity(z[t]), "method": "Daily robust z (28-day median)",
                         "type": "unexpected change in daily consumption",
                         "evidence": f"daily total z = {z[t]:.2f} vs trailing 28-day median {exp[t]:.1f} kWh"})
        if not np.isnan(vz[t]) and vz[t] > 4:
            rows.append({"date": t, "observed_kwh": hourly_sd[t], "expected_kwh": vol_exp[t],
                         "deviation_pct": (hourly_sd[t] / vol_exp[t] - 1) * 100, "robust_z": vz[t], "severity": severity(vz[t]),
                         "method": "Volatility robust z", "type": "abnormal consumption volatility",
                         "evidence": f"intra-day hourly std z = {vz[t]:.2f} (trailing 28-day median {vol_exp[t]:.2f} kWh)"})
    return pd.DataFrame(rows)


def data_quality_findings() -> pd.DataFrame:
    m = pd.read_parquet(pp.HH_MINUTE)
    rows = []
    miss = m["Global_active_power"].isna()
    grp = (miss != miss.shift()).cumsum()
    for _, seg in m[miss].groupby(grp[miss]):
        if len(seg) >= 60:
            rows.append({"start": seg.index[0], "end": seg.index[-1], "duration_min": len(seg),
                         "issue": "missing meter readings", "evidence": f"{len(seg)} consecutive minutes without data",
                         "severity": "high" if len(seg) >= 1440 else "medium"})
    v = m["Voltage"]
    out = (v < 207) | (v > 253)  # EN 50160: 230 V +/- 10 %
    day_counts = out.resample("D").sum()
    for t, n in day_counts[day_counts > 0].items():
        rows.append({"start": t, "end": t + pd.Timedelta(hours=23, minutes=59), "duration_min": int(n),
                     "issue": "voltage outside EN 50160 band (207-253 V)",
                     "evidence": f"{int(n)} minutes; min {v.loc[t:t + pd.Timedelta(days=1)].min():.1f} V, "
                                 f"max {v.loc[t:t + pd.Timedelta(days=1)].max():.1f} V", "severity": "medium"})
    q = load_json("data_quality", "household") or {}
    for c in q.get("quality_report", {}).get("checks", []):
        if not c["passed"] and "sub-meters" in c["check"]:
            rows.append({"start": pd.NaT, "end": pd.NaT, "duration_min": None, "issue": "sub-meter sum exceeds whole-house energy",
                         "evidence": c["detail"], "severity": "low"})
    return pd.DataFrame(rows)


def repeated_patterns(events: pd.DataFrame) -> pd.DataFrame:
    c = events[events["n_methods_agreeing"] >= 2].drop_duplicates("timestamp")
    if c.empty:
        return pd.DataFrame()
    c = c.assign(hour=c["timestamp"].dt.hour, weekday=c["timestamp"].dt.day_name(), month=c["timestamp"].dt.to_period("M").astype(str))
    rep = c.groupby(["hour", "type"]).agg(occurrences=("timestamp", "size"), months=("month", "nunique"),
                                            first=("timestamp", "min"), last=("timestamp", "max")).reset_index()
    expected = len(c) / 24
    rep["ratio_to_uniform"] = rep["occurrences"] / expected
    return rep[rep["occurrences"] >= 5].sort_values("occurrences", ascending=False)


SUBMETERS = {"sub_kitchen_kwh": "kitchen", "sub_laundry_kwh": "laundry",
             "sub_water_heater_ac_kwh": "water heater / AC", "sub_other_kwh": "unmetered remainder"}


def attribute(hh: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
    """Which sub-meter deviates most from its own causal baseline at each anomalous hour (supporting evidence)."""
    dev = pd.DataFrame({name: hh[col] - baseline(hh[col]) for col, name in SUBMETERS.items()}).reindex(ts)
    top = dev.abs().idxmax(axis=1)
    return pd.DataFrame({"driver_submeter": top.to_numpy(),
                         "driver_excess_kwh": [dev.at[t, c] if isinstance(c, str) else np.nan for t, c in zip(ts, top)]},
                        index=ts.index)


def run() -> dict:
    hh = pp.load_household_hourly()
    events, scored = detect_hourly(hh)
    daily = detect_daily(hh)
    dq = data_quality_findings()
    rep = repeated_patterns(events)
    consensus = (events[events["n_methods_agreeing"] >= 2].sort_values("n_methods_agreeing", ascending=False)
                 .drop_duplicates("timestamp").sort_values("timestamp"))
    consensus = consensus.assign(methods=consensus["timestamp"].map(events.groupby("timestamp")["method"].apply(", ".join)))
    consensus = consensus.join(attribute(hh, consensus["timestamp"]))
    consensus["evidence"] = (consensus["methods"] + " flagged; severity score " + consensus["severity_score"].round(1).astype(str)
                             + "; largest sub-meter deviation: " + consensus["driver_submeter"].astype(str) + " ("
                             + consensus["driver_excess_kwh"].round(2).map("{:+}".format) + " kWh vs its baseline)")
    agreement = scored[[c for c in scored.columns if c in events["method"].unique()]].astype(int)
    jacc = pd.DataFrame(index=agreement.columns, columns=agreement.columns, dtype=float)
    for a in agreement:
        for b in agreement:
            inter = (agreement[a] & agreement[b]).sum()
            union = (agreement[a] | agreement[b]).sum()
            jacc.loc[a, b] = inter / union if union else np.nan
    save_table(events, "anomaly", "hourly_events")
    save_table(consensus, "anomaly", "consensus")
    save_table(daily, "anomaly", "daily_events")
    save_table(dq.astype({"start": "datetime64[ns]", "end": "datetime64[ns]"}) if len(dq) else dq, "anomaly", "data_quality")
    save_table(rep, "anomaly", "repeated_patterns")
    save_table(jacc.reset_index(names="method"), "anomaly", "method_agreement")
    save_table(scored[["observed", "expected", "robust_z", "n_methods"]], "anomaly", "scored_series")
    summary = {"disclaimer": DISCLAIMER, "baseline": "median of same hour-of-week over previous 4 weeks (causal)",
               "methods": sorted(events["method"].unique().tolist()), "hours_scored": int(len(scored)),
               "flags_by_method": events["method"].value_counts().to_dict(),
               "consensus_rule": ">= 2 detectors agree",
               "practical_significance_floor_kwh": MIN_ABS_DEVIATION_KWH, "n_consensus_anomalies": int(len(consensus)),
               "consensus_by_type": consensus["type"].value_counts().to_dict(),
               "consensus_by_severity": consensus["severity"].value_counts().to_dict(),
               "consensus_by_driver": consensus["driver_submeter"].value_counts().to_dict(),
               "severity_rule": "max(|rolling z|, |robust z|) of the deviation from the causal baseline: >=8 critical, >=6 high, >=4 medium, else low",
               "n_daily_anomalies": int(len(daily)), "n_data_quality_findings": int(len(dq)),
               "n_repeated_patterns": int(len(rep))}
    save_json(summary, "anomaly", "summary")
    return summary
