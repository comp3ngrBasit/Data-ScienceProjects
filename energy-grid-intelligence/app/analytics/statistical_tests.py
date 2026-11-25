"""Statistical hypothesis testing with effect sizes, block-bootstrap CIs and Holm correction.

Daily energy series are serially correlated, so classical tests overstate significance. Each comparison
therefore reports (a) the classical test, (b) an effect size, and (c) a 95 % CI for the difference in
means from a *moving-block bootstrap over weeks*, which preserves within-week dependence. Holm's
step-down procedure controls the family-wise error rate across all tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from app.config import RANDOM_SEED
from app.data import preprocessing as pp
from app.data.feature_engineering import holiday_flags
from app.utils.helpers import save_json, save_table


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def block_bootstrap_diff(s: pd.Series, group_a: pd.Series, n_boot: int = 2000, block: str = "W") -> tuple[float, float]:
    """95 % CI of mean(A) - mean(B) by resampling whole blocks (weeks/months) with replacement."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.DataFrame({"v": s, "a": group_a}).dropna()
    blocks = [g for _, g in df.groupby(df.index.to_period(block))]
    diffs = []
    for _ in range(n_boot):
        pick = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        A, B = pick.loc[pick["a"], "v"], pick.loc[~pick["a"], "v"]
        if len(A) and len(B):
            diffs.append(A.mean() - B.mean())
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def independent_test(name: str, s: pd.Series, mask_a: pd.Series, label_a: str, label_b: str, unit: str,
                     block: str = "W", context: str = "") -> dict:
    a, b = s[mask_a].dropna().to_numpy(), s[~mask_a].dropna().to_numpy()
    w = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1 - 2 * mw.statistic / (len(a) * len(b))  # rank-biserial correlation
    lo, hi = block_bootstrap_diff(s, mask_a, block=block)
    return {"hypothesis": name, "group_a": label_a, "group_b": label_b, "n_a": len(a), "n_b": len(b),
            "mean_a": a.mean(), "mean_b": b.mean(), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "unit": unit, "diff_mean": a.mean() - b.mean(), "diff_pct": (a.mean() / b.mean() - 1) * 100,
            "primary_test": "Mann-Whitney U (two-sided)", "statistic": float(mw.statistic), "p_value": float(mw.pvalue),
            "welch_t": float(w.statistic), "welch_p": float(w.pvalue), "effect_size_cohens_d": cohens_d(a, b),
            "effect_size_rank_biserial": float(rbc), "ci95_diff_low": lo, "ci95_diff_high": hi,
            "ci_method": f"moving-block bootstrap ({'weeks' if block == 'W' else 'months'}), 2000 resamples",
            "context": context}


def paired_test(name: str, a: pd.Series, b: pd.Series, label_a: str, label_b: str, unit: str, context: str = "") -> dict:
    d = (a - b).dropna()
    wil = stats.wilcoxon(d)
    t = stats.ttest_rel(a.loc[d.index], b.loc[d.index])
    ranks = stats.rankdata(np.abs(d))
    rbc = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()  # matched-pairs rank-biserial
    rng = np.random.default_rng(RANDOM_SEED)
    weeks = [g.to_numpy() for _, g in d.groupby(d.index.to_period("W"))]
    boots = [np.concatenate([weeks[i] for i in rng.integers(0, len(weeks), len(weeks))]).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"hypothesis": name, "group_a": label_a, "group_b": label_b, "n_a": len(d), "n_b": len(d),
            "mean_a": float(a.loc[d.index].mean()), "mean_b": float(b.loc[d.index].mean()),
            "median_a": float(a.loc[d.index].median()), "median_b": float(b.loc[d.index].median()), "unit": unit,
            "diff_mean": float(d.mean()), "diff_pct": float((a.loc[d.index].mean() / b.loc[d.index].mean() - 1) * 100),
            "primary_test": "Wilcoxon signed-rank (paired by day)", "statistic": float(wil.statistic),
            "p_value": float(wil.pvalue), "welch_t": float(t.statistic), "welch_p": float(t.pvalue),
            "effect_size_cohens_d": float(d.mean() / d.std(ddof=1)), "effect_size_rank_biserial": float(rbc),
            "ci95_diff_low": lo, "ci95_diff_high": hi, "ci_method": "moving-block bootstrap of daily differences (weeks)",
            "context": context}


def interpret(r: dict) -> str:
    sig = r["p_value_holm"] < 0.05
    ci_excl = (r["ci95_diff_low"] > 0) or (r["ci95_diff_high"] < 0)
    d = abs(r["effect_size_cohens_d"])
    mag = "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
    direction = "higher" if r["diff_mean"] > 0 else "lower"
    if sig and ci_excl:
        return (f"Mean for {r['group_a']} is {direction} than for {r['group_b']} by {abs(r['diff_mean']):.3g} {r['unit']} "
                f"({r['diff_pct']:+.1f} %), {mag} effect; significant after Holm correction and the dependence-robust CI excludes 0.")
    if sig:
        return (f"Classical test significant after Holm correction, but the block-bootstrap CI includes 0: "
                f"evidence is weak once serial dependence is accounted for ({mag} effect).")
    return f"No significant difference after Holm correction ({mag} effect)."


def run() -> dict:
    rows = []
    hd = pp.load_household_daily()
    hd = hd[hd["imputed_share"] < 0.25]["energy_kwh"]
    rows.append(independent_test("H1: household weekday vs weekend daily consumption", hd, hd.index.dayofweek >= 5,
                                 "Weekend days", "Weekdays", "kWh/day"))
    sw = hd[hd.index.month.isin([12, 1, 2, 6, 7, 8])]
    rows.append(independent_test("H2: household summer vs winter daily consumption", sw,
                                 pd.Series(sw.index.month.isin([6, 7, 8]), index=sw.index), "Summer (JJA)", "Winter (DJF)",
                                 "kWh/day", block="M", context="Monthly blocks: seasons are contiguous"))
    hh = pp.load_household_hourly()
    hh = hh[~hh["imputed"]]["energy_kwh"]
    day = hh.index.normalize()
    peak = hh[(hh.index.hour >= 19) & (hh.index.hour < 23)].groupby(day[(hh.index.hour >= 19) & (hh.index.hour < 23)]).mean()
    off = hh[(hh.index.hour < 19) | (hh.index.hour >= 23)].groupby(day[(hh.index.hour < 19) | (hh.index.hour >= 23)]).mean()
    rows.append(paired_test("H3: household peak (19-23 h) vs off-peak hourly consumption", peak, off,
                            "Peak window", "Off-peak hours", "kWh/h", context="Peak window found data-driven in Module 2"))
    gd = pp.grid_daily("DE")["load_mwh"] / 1e3
    rows.append(independent_test("H4: grid weekday vs weekend daily load (DE)", gd, gd.index.dayofweek >= 5,
                                 "Weekend days", "Weekdays", "GWh/day"))
    gsw = gd[gd.index.month.isin([12, 1, 2, 6, 7, 8])]
    rows.append(independent_test("H5: grid summer vs winter daily load (DE)", gsw,
                                 pd.Series(gsw.index.month.isin([6, 7, 8]), index=gsw.index), "Summer (JJA)", "Winter (DJF)",
                                 "GWh/day", block="M"))
    gl = pp.grid_local_hourly("DE")["load_mw"]
    workday = (gl.index.dayofweek < 5) & (holiday_flags(gl.index, "DE") == 0)
    pk = gl[workday & (gl.index.hour >= 8) & (gl.index.hour < 20)]
    op = gl[workday & ((gl.index.hour < 8) | (gl.index.hour >= 20))]
    rows.append(paired_test("H6: grid peak (08-20 h) vs off-peak hourly load on working days (DE)",
                            pk.groupby(pk.index.normalize()).mean(), op.groupby(op.index.normalize()).mean(),
                            "Peak (EEX 08-20)", "Off-peak", "MW"))
    try:
        from app.analytics.weather_analysis import build_daily

        wdf, _ = build_daily("DE")
        wdf = wdf[(wdf["is_offday"] == 0) & (wdf["xmas_period"] == 0) & wdf.index.month.isin([5, 6, 7, 8, 9])]
        t = wdf["temperature"]
        hot, moderate = t >= t.quantile(0.8), t.between(t.quantile(0.3), t.quantile(0.6))
        sub = wdf.loc[hot | moderate, "load_gwh"]
        rows.append(independent_test("H7: grid load on hot vs moderate-temperature working days, May-Sep (DE)", sub,
                                     hot[hot | moderate], f"Hot days (>= {t.quantile(0.8):.1f} degC)",
                                     f"Moderate days ({t.quantile(0.3):.1f}-{t.quantile(0.6):.1f} degC)", "GWh/day",
                                     context="Working days in May-Sep only, to avoid mixing heating demand and calendar effects"))
    except Exception as exc:  # noqa: BLE001
        rows.append({"hypothesis": "H7: hot vs moderate temperature", "status": f"not run: {exc}"})
    res = pd.DataFrame(rows)
    ok = res["p_value"].notna()
    res.loc[ok, "p_value_holm"] = multipletests(res.loc[ok, "p_value"], method="holm")[1]
    res["significant_holm_0.05"] = res["p_value_holm"] < 0.05
    res["interpretation"] = [interpret(r) if pd.notna(r.get("p_value")) else r.get("status") for r in res.to_dict("records")]
    save_table(res, "statistics", "hypothesis_tests")
    out = {"n_tests": int(ok.sum()), "correction": "Holm (family-wise error rate, alpha = 0.05)",
           "n_significant": int(res["significant_holm_0.05"].sum()), "method_note": __doc__.strip()}
    save_json(out, "statistics", "summary")
    return out
