"""Evaluation metrics for probabilistic next-action predictions (computed on held-out predictions only)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from app.patterns.stats import wilson


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error on the top-1 probability, plus the reliability-diagram bins."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total, out = 0.0, []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        n = int(m.sum())
        if n:
            gap = abs(correct[m].mean() - conf[m].mean())
            total += n / len(y) * gap
            out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": n,
                        "mean_confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(correct[m].mean()), 4)})
    return round(float(total), 4), out


def evaluate(P: np.ndarray, y: np.ndarray, classes: list[str], with_confusion: bool = True) -> dict:
    n = len(y)
    if n == 0:
        return {"n": 0}
    C = len(classes)
    Y = np.zeros_like(P)
    Y[np.arange(n), y] = 1
    pred = P.argmax(axis=1)
    top3 = np.argsort(-P, axis=1)[:, :3]
    t1 = int((pred == y).sum())
    t3 = int(sum(y[i] in top3[i] for i in range(n)))
    present = sorted(set(y.tolist()))
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=present, average="macro", zero_division=0)
    e, rel = ece(P, y)
    out = {
        "n": n, "top1_accuracy": round(t1 / n, 4), "top1_ci95": wilson(t1, n), "top3_accuracy": round(t3 / n, 4),
        "top3_ci95": wilson(t3, n),
        # single true next action: Recall@3 = share of cases where it is among the top 3; Precision@3 = hits / 3
        "recall_at_3": round(t3 / n, 4), "precision_at_3": round(t3 / (3 * n), 4), "macro_precision": round(float(pr), 4), "macro_recall": round(float(rc), 4),
        "macro_f1": round(float(f1), 4), "log_loss": round(float(-np.mean(np.log(np.clip(P[np.arange(n), y], 1e-12, 1)))), 4),
        "brier": round(float(np.mean(np.sum((P - Y) ** 2, axis=1))), 4), "ece": e, "reliability_diagram": rel,
        "classes_in_test": len(present), "mean_confidence": round(float(P.max(axis=1).mean()), 4),
    }
    if with_confusion:
        out["confusion_matrix"] = {"labels": classes, "matrix": confusion_matrix(y, pred, labels=list(range(C))).tolist()}
    return out


def majority_baseline(train_y: np.ndarray, test_y: np.ndarray, C: int) -> dict:
    """Always predict the training class frequencies — the floor any model must beat."""
    freq = np.bincount(train_y, minlength=C).astype(float) + 0.5
    P = np.tile(freq / freq.sum(), (len(test_y), 1))
    return P


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, groups: np.ndarray, B: int = 1000, seed: int = 0) -> dict:
    """Top-1 accuracy difference (a - b) with a 95 % interval from resampling whole fights."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    diffs = []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        diffs.append(correct_a[s].mean() - correct_b[s].mean())
    d = float(correct_a.mean() - correct_b.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": round(d, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "resampled_fights": len(ug),
            "significant": bool(lo > 0 or hi < 0)}
