"""Small, dependency-light statistics helpers used by the pattern engine."""
from __future__ import annotations

import math
from collections import Counter

from scipy.stats import binomtest, chi2_contingency
from scipy.spatial.distance import jensenshannon


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for k successes out of n."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def reliability(n: int, fights: int, ci: tuple[float, float] | None, min_n: int) -> str:
    """Plain-language reliability label for an estimated proportion.

    single_fight : every observation comes from one fight — not evidence of a general habit
    insufficient : fewer observations than the configured minimum
    low / medium / high : by sample size, number of fights and CI width
    """
    if n < min_n:
        return "insufficient"
    if fights <= 1:
        return "single_fight"
    width = (ci[1] - ci[0]) if ci else 1.0
    if n >= 30 and fights >= 3 and width <= 0.3:
        return "high"
    if n >= 10 and fights >= 2 and width <= 0.5:
        return "medium"
    return "low"


def binom_greater(k: int, n: int, p0: float) -> float | None:
    """One-sided p-value that the true rate exceeds the baseline p0."""
    if n <= 0 or not (0 < p0 < 1):
        return None
    return float(binomtest(k, n, p0, alternative="greater").pvalue)


def entropy(counts: Counter | dict) -> float:
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c)


def js_distance(a: dict, b: dict) -> float | None:
    keys = sorted(set(a) | set(b))
    sa, sb = sum(a.values()), sum(b.values())
    if not keys or sa == 0 or sb == 0:
        return None
    return round(float(jensenshannon([a.get(k, 0) / sa for k in keys], [b.get(k, 0) / sb for k in keys], base=2)), 4)


def chi_square(table: list[list[int]]) -> dict:
    """Chi-square test of homogeneity with the usual validity check (expected >= 5 in >= 80 % of cells)."""
    rows = [r for r in table if sum(r) > 0]
    cols = [j for j in range(len(rows[0])) if any(r[j] for r in rows)] if rows else []
    rows = [[r[j] for j in cols] for r in rows]
    if len(rows) < 2 or len(cols) < 2:
        return {"valid": False, "reason": "need at least two groups and two action types with observations"}
    chi2, p, dof, expected = chi2_contingency(rows)
    ok = (expected >= 5).mean() >= 0.8
    return {"valid": bool(ok), "chi2": round(float(chi2), 3), "p_value": round(float(p), 5), "dof": int(dof),
            "reason": None if ok else "expected counts too small for a reliable chi-square test"}
