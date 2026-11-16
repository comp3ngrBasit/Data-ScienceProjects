"""Evaluation metrics for regression (forecasting) and classification models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)


def _clean(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {f.shape}")
    m = ~(np.isnan(a) | np.isnan(f))
    return a[m], f[m]


def mae(a, f) -> float:
    a, f = _clean(a, f)
    return float(np.mean(np.abs(a - f))) if a.size else float("nan")


def rmse(a, f) -> float:
    a, f = _clean(a, f)
    return float(np.sqrt(np.mean((a - f) ** 2))) if a.size else float("nan")


def mape(a, f) -> float:
    """Mean absolute percentage error over periods with non-zero actuals (undefined at zero)."""
    a, f = _clean(a, f)
    nz = a != 0
    return float(np.mean(np.abs((a[nz] - f[nz]) / a[nz]))) if nz.any() else float("nan")


def wape(a, f) -> float:
    a, f = _clean(a, f)
    d = np.abs(a).sum()
    return float(np.abs(a - f).sum() / d) if d > 0 else float("nan")


def weekly_wape(a, f) -> float:
    """WAPE on 7-day totals — the accuracy that matters for weekly replenishment."""
    a, f = _clean(a, f)
    n = (a.size // 7) * 7
    if n == 0:
        return wape(a, f)
    return wape(a[:n].reshape(-1, 7).sum(1), f[:n].reshape(-1, 7).sum(1))


def regression_report(a, f) -> dict:
    return {"mae": mae(a, f), "rmse": rmse(a, f), "mape": mape(a, f), "wape": wape(a, f),
            "weekly_wape": weekly_wape(a, f)}


def classification_report(y_true, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, float)
    pred = (p >= threshold).astype(int)
    out = {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)), "accuracy": float((pred == y).mean()),
           "positive_rate": float(y.mean()), "threshold": threshold, "n": int(len(y))}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def best_f1_threshold(y_true, proba) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, float)
    grid = np.linspace(0.05, 0.95, 37)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def youden_threshold(y_true, proba) -> float:
    """Threshold maximising TPR - FPR (Youden's J). Unlike F1-optimisation it does not collapse to
    'predict everything positive' when positives are the majority class."""
    from sklearn.metrics import roc_curve
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, np.asarray(proba, float))
    j = tpr - fpr
    return float(np.clip(thr[int(np.argmax(j))], 0.01, 0.99))
