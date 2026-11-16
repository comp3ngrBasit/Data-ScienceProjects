"""Training and evaluation of action classifiers.

* Data are split **by video / fight group**, never by window or frame (no leakage):
  videos marked ``test`` form a hold-out set; otherwise grouped cross-validation is used
  (leave-one-group-out for <= 10 groups, else 5-fold GroupKFold).
* Augmented copies of a training example never appear in an evaluation fold.
* Classes with too few independent examples are excluded and reported, not silently trained.
* Every number stored here is computed from model outputs on held-out data.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from app.actions.data import Dataset, Sample
from app.actions.models import make_model
from app.actions.taxonomy import BACKGROUND, label

log = logging.getLogger(__name__)


def _arr(samples: list[Sample]):
    return np.stack([s.X for s in samples]), [s.label for s in samples]


def select_classes(ds: Dataset, min_examples: int, min_groups: int) -> tuple[list[str], dict[str, str]]:
    real = [s for s in ds.samples if not s.augmented]
    counts = Counter(s.label for s in real)
    groups = {c: len({s.group for s in real if s.label == c}) for c in counts}
    keep, dropped = [], {}
    for c, n in sorted(counts.items()):
        if c == BACKGROUND:
            continue
        if n < min_examples:
            dropped[c] = f"only {n} example(s) (need {min_examples})"
        elif groups[c] < min_groups:
            dropped[c] = f"examples come from only {groups[c]} video/fight group(s) (need {min_groups})"
        else:
            keep.append(c)
    if counts.get(BACKGROUND):
        keep.append(BACKGROUND)
    return keep, dropped


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for a proportion (used to show how uncertain small-sample metrics are)."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(c - h), 4), round(float(c + h), 4)


def metrics_from(y_true: list[str], y_pred: list[str], conf: np.ndarray, classes: list[str], threshold: float) -> dict:
    labels = classes
    acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    action_labels = [c for c in labels if c != BACKGROUND]
    pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, labels=action_labels, average="macro", zero_division=0) if action_labels else (0, 0, 0, 0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    yt, yp = np.array(y_true), np.array(y_pred)
    for i, c in enumerate(labels):
        # "validated precision": of predictions of class c made with confidence >= threshold, the fraction that were right
        sel = (yp == c) & (conf >= threshold)
        per_class[c] = {
            "label": label(c), "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4), "support": int(s[i]),
            "confident_predictions": int(sel.sum()),
            "precision_at_threshold": round(float((yt[sel] == c).mean()), 4) if sel.any() else None,
        }
    n_correct = int(sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b))
    majority = Counter(y_true).most_common(1)[0][1] / len(y_true) if y_true else None
    return {
        "accuracy": round(float(acc), 4),
        "accuracy_ci95": wilson(n_correct, len(y_true)),
        "majority_class_baseline": round(float(majority), 4) if majority is not None else None,
        "precision_macro": round(float(pa), 4), "recall_macro": round(float(ra), 4), "f1_macro": round(float(fa), 4),
        "precision_weighted": round(float(pw), 4), "recall_weighted": round(float(rw), 4), "f1_weighted": round(float(fw), 4),
        "macro_over": "action classes (background excluded)",
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "display_labels": [label(c) for c in labels], "matrix": cm.tolist()},
        "n_evaluated": len(y_true),
        "threshold": threshold,
    }


def evaluate(ds: Dataset, classes: list[str], algorithm: str, params: dict, device: str, threshold: float,
             progress: Callable[[float], None] | None = None) -> tuple[dict, dict]:
    pool = [s for s in ds.samples if s.label in classes]
    test = [s for s in pool if s.split == "test" and not s.augmented]
    train_pool = [s for s in pool if s.split != "test"]
    y_true, y_pred, confs, fold_info = [], [], [], []

    if test:
        strategy = "hold-out test videos"
        folds = [(train_pool, test)]
    else:
        groups = sorted({s.group for s in train_pool if not s.augmented})
        if len(groups) < 2:
            return {"error": "Need annotated examples from at least 2 different videos/fights to evaluate"}, {"strategy": "none"}
        splitter = LeaveOneGroupOut() if len(groups) <= 10 else GroupKFold(n_splits=5)
        strategy = "leave-one-group-out cross-validation" if len(groups) <= 10 else "5-fold group cross-validation"
        real = [s for s in train_pool if not s.augmented]
        gidx = np.array([s.group for s in real])
        folds = []
        for _tr_i, te_i in splitter.split(np.zeros(len(real)), groups=gidx):
            te_groups = set(gidx[te_i])
            folds.append(([s for s in train_pool if s.group not in te_groups], [real[i] for i in te_i]))

    for k, (tr, te) in enumerate(folds):
        tr_labels = {s.label for s in tr}
        if len(tr_labels) < 2:
            continue
        Xtr, ytr = _arr(tr)
        Xte, yte = _arr(te)
        m = make_model(algorithm, classes, params, device).fit(Xtr, ytr)
        P = m.predict_proba(Xte)
        pred = [classes[i] for i in P.argmax(1)]
        y_true += yte
        y_pred += pred
        confs += list(P.max(1))
        fold_info.append({"train": len(tr), "test": len(te), "test_groups": sorted({s.group for s in te})[:5],
                          "unseen_classes_in_test": sorted({s.label for s in te} - tr_labels)})
        if progress:
            progress((k + 1) / len(folds))
    if not y_true:
        return {"error": "No evaluable folds (each fold needs at least two classes in training)"}, {"strategy": strategy}
    m = metrics_from(y_true, y_pred, np.array(confs), classes, threshold)
    return m, {"strategy": strategy, "folds": len(fold_info), "fold_details": fold_info}


def train_algorithms(ds: Dataset, cfg: dict, device: str, out_root, progress: Callable[[str, float], None]) -> dict:
    """Train + evaluate each algorithm. Returns per-algorithm results and the saved model dirs."""
    classes, dropped = select_classes(ds, cfg["min_examples"], cfg["min_groups"])
    action_classes = [c for c in classes if c != BACKGROUND]
    if not action_classes:
        raise ValueError("No action class has enough annotated examples to train. " +
                         ("; ".join(f"{label(c)}: {why}" for c, why in dropped.items()) or "No annotations found."))
    if len(classes) < 2:
        raise ValueError("Training needs at least two classes (e.g. one action plus background windows from a fully annotated video)")

    results = {}
    algos = cfg["algorithms"]
    for ai, algo in enumerate(algos):
        base = ai / len(algos)
        progress(f"Evaluating {algo}", base)
        t0 = time.monotonic()
        params = cfg.get("params", {}).get(algo, {})
        metrics, ev = evaluate(ds, classes, algo, params, device, cfg["threshold"],
                               lambda f, b=base, a=algo: progress(f"Evaluating {a}", b + f * 0.7 / len(algos)))
        progress(f"Training final {algo}", base + 0.75 / len(algos))
        final_pool = [s for s in ds.samples if s.label in classes and s.split != "test"]
        X, y = _arr(final_pool)
        model = make_model(algo, classes, params, device).fit(X, y)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version = f"{algo}-{ds.version[:6]}-{stamp}"
        d = out_root / version
        model.save(d)
        pos = [s.t1 - s.t0 for s in ds.samples if s.label not in (BACKGROUND,) and s.label in classes and not s.augmented]
        meta = {"version": version, "algorithm": algo, "classes": classes, "feature_config": ds.summary["feature_config"],
                "params": params, "dataset_version": ds.version,
                "window_seconds": round(float(np.median(pos)), 3) if pos else 0.8}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        results[algo] = {
            "version": version, "dir": str(d), "metrics": metrics, "evaluation": ev,
            "train_seconds": round(time.monotonic() - t0, 2), "final_train_samples": len(final_pool),
            "final_train_real": sum(1 for s in final_pool if not s.augmented), "model_info": model.info,
        }
    return {"classes": classes, "dropped": dropped, "results": results}
