"""Phase 5 training & evaluation pipeline.

Protocols (all splits by complete fights / fighters, never by adjacent actions):
1. fight-level cross-validation (GroupKFold over fights) — main comparison; every example is predicted by
   models that never saw its fight;
2. unseen-fighter cross-validation (GroupKFold over fighters) — generalisation to new fighters;
3. temporal hold-out — train on the chronologically earliest fights, test on the latest.
Inside every training portion, a fight-level slice is held back for early stopping and temperature scaling.
The production models are retrained on all data; their reported metrics are the cross-validated ones.
"""
from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from app.actions.taxonomy import CATEGORIES, label
from app.patterns.engine import style_label
from app.predictions.dataset import RARE, Example, FighterStats, action_group_of, choose_classes, map_target
from app.predictions.features import FeatureBuilder
from app.predictions.metrics import evaluate, majority_baseline, paired_bootstrap
from app.predictions.models import make_predictor

log = logging.getLogger(__name__)
DEEP = ("gru", "lstm", "transformer")


def vocabulary() -> list[str]:
    return [c.key for c in CATEGORIES]


def class_label(c: str) -> str:
    return "Other (rare action)" if c == RARE else label(c)


def split_calibration(train: list[Example], frac: float = 0.2) -> tuple[list[Example], list[Example]]:
    """Hold back the chronologically last ~20 % of training fights (>= 1, only if >= 3 fights)."""
    order = []
    for e in train:
        if e.fight_key not in order:
            order.append(e.fight_key)
    if len(order) < 3:
        return train, []
    k = max(1, int(round(len(order) * frac)))
    cal = set(order[-k:])
    return [e for e in train if e.fight_key not in cal], [e for e in train if e.fight_key in cal]


def fit_fold(arch: str, train: list[Example], classes: list[str], cfg: dict, device: str):
    fit, cal = split_calibration(train)
    fb = FeatureBuilder(classes, vocabulary(), FighterStats(train), cfg.get("prior_alpha", 1.0))
    params = dict(cfg.get("params", {}).get(arch, {}))
    m = make_predictor(arch, params, device)
    m.fit(fit, fb, cal or None)
    return m


def run_protocol(name: str, examples: list[Example], splits: list[tuple[list[int], list[int]]], archs: list[str],
                 classes: list[str], cfg: dict, device: str, progress: Callable[[str, float], None]) -> dict:
    """Out-of-fold probabilities for every architecture under one splitting protocol."""
    C = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[map_target(e.target, classes)] for e in examples])
    P = {a: np.zeros((len(examples), C)) for a in archs + ["class_frequency"]}
    covered = np.zeros(len(examples), dtype=bool)
    known = np.zeros(len(examples), dtype=bool)
    fold_info = []
    for f, (tr, te) in enumerate(splits):
        train = [examples[i] for i in tr]
        test = [examples[i] for i in te]
        train_fighters = {e.fighter_key for e in train}
        for i in te:
            known[i] = examples[i].fighter_key in train_fighters
        P["class_frequency"][te] = majority_baseline(y[tr], y[te], C)
        for j, a in enumerate(archs):
            progress(f"{name}: fold {f + 1}/{len(splits)} · {a}", (f + j / len(archs)) / len(splits))
            try:
                m = fit_fold(a, train, classes, cfg, device)
                P[a][te] = m.predict_proba(test)
            except Exception as exc:  # a fold too small for a model: record, don't fake
                log.exception("fold %s %s failed", f, a)
                P[a][te] = np.nan
                fold_info.append({"fold": f, "arch": a, "error": str(exc)[:300]})
        covered[te] = True
        fit_part, cal_part = split_calibration(train)
        fold_info.append({"fold": f, "train_examples": len(fit_part), "validation_examples": len(cal_part), "test_examples": len(test),
                          "train_fights": len({e.fight_key for e in train}), "test_fights": len({e.fight_key for e in test})})
    return {"y": y, "P": P, "covered": covered, "known": known, "folds": fold_info}


def _subset_metrics(P: np.ndarray, y: np.ndarray, mask: np.ndarray, classes: list[str]) -> dict | None:
    if mask.sum() == 0 or np.isnan(P[mask]).any():
        return None
    return evaluate(P[mask], y[mask], classes, with_confusion=False)


def summarise(res: dict, examples: list[Example], classes: list[str], archs: list[str], styles: dict[str, str]) -> dict:
    y, cov = res["y"], res["covered"]
    groups = np.array([action_group_of(map_target(e.target, classes)) for e in examples])
    has_opp = np.array([e.context.opponent_action is not None for e in examples])
    style = np.array([styles.get(e.fighter_key, "insufficient_data") for e in examples])
    fights = np.array([e.fight_key for e in examples])
    out = {}
    for a in archs + ["class_frequency"]:
        P = res["P"][a]
        m = cov & ~np.isnan(P).any(axis=1)
        if m.sum() == 0:
            out[a] = {"status": "failed", "n": 0}
            continue
        r = evaluate(P[m], y[m], classes)
        r["coverage"] = round(float(m.sum() / len(y)), 4)
        r["breakdown"] = {
            "known_fighters": _subset_metrics(P, y, m & res["known"], classes),
            "unseen_fighters": _subset_metrics(P, y, m & ~res["known"], classes),
            "offensive_targets": _subset_metrics(P, y, m & (groups == "offensive"), classes),
            "defensive_targets": _subset_metrics(P, y, m & (groups == "defensive"), classes),
            "with_opponent_context": _subset_metrics(P, y, m & has_opp, classes),
            "without_opponent_context": _subset_metrics(P, y, m & ~has_opp, classes),
            **{f"style_{s}": _subset_metrics(P, y, m & (style == s), classes) for s in sorted(set(style.tolist()))},
        }
        out[a] = r
    # each model vs the Markov baseline: top-1 difference resampled by fight
    if "markov" in archs and out.get("markov", {}).get("n"):
        base = res["P"]["markov"]
        for a in archs + ["class_frequency"]:
            if a == "markov" or not out[a].get("n"):
                continue
            m = cov & ~np.isnan(res["P"][a]).any(axis=1) & ~np.isnan(base).any(axis=1)
            ca = (res["P"][a][m].argmax(1) == y[m]).astype(float)
            cb = (base[m].argmax(1) == y[m]).astype(float)
            out[a]["vs_markov_top1"] = paired_bootstrap(ca, cb, fights[m])
    return out


def train_and_evaluate(examples: list[Example], cfg: dict, device: str, progress: Callable[[str, float], None],
                       fight_dates: dict[str, str | None] | None = None) -> dict:
    t0 = time.monotonic()
    examples = [e for e in examples if e.target]
    classes, counts = choose_classes(examples, int(cfg.get("min_class_count", 5)))
    fights = sorted({e.fight_key for e in examples})
    fighters = sorted({e.fighter_key for e in examples})
    n = len(examples)
    archs = ["markov", "xgboost"] + [a for a in DEEP if a in cfg.get("architectures", DEEP)]
    skipped = {}
    for a in DEEP:
        need = int(cfg.get("min_examples_transformer", 1000) if a == "transformer" else cfg.get("min_examples_deep", 300))
        if a in archs and n < need:
            archs.remove(a)
            skipped[a] = f"needs >= {need} training examples (have {n})"
    data = {"examples": n, "fights": len(fights), "fighters": len(fighters), "classes": classes,
            "class_labels": [class_label(c) for c in classes], "target_counts": counts,
            "sources": dict(Counter(e.source for e in examples)),
            "rare_actions": sorted(c for c, k in counts.items() if c not in classes)}
    if len(fights) < 2 or len(classes) < 2:
        return {"status": "insufficient_data", "data": data, "skipped": skipped, "protocols": {},
                "reason": "Need at least 2 fights and 2 action classes with enough examples to evaluate on unseen fights."}

    # styles for the per-style breakdown (interpretable label from each fighter's observed action mix)
    by_f = defaultdict(list)
    for e in examples:
        by_f[e.fighter_key].append(e.target)
    styles = {}
    for f, ts in by_f.items():
        k = len(ts)
        off = sum(1 for t in ts if action_group_of(t) == "offensive") / k
        de = sum(1 for t in ts if action_group_of(t) == "defensive") / k
        styles[f] = style_label({"offensive_share": off, "defensive_share": de}, k, 20)

    protocols = {}
    gidx = np.array([e.fight_key for e in examples])
    k = min(int(cfg.get("folds", 5)), len(fights))
    splits = list(GroupKFold(n_splits=k).split(np.zeros(n), groups=gidx))
    stage_w = {"fight_cv": 0.5, "fighter_cv": 0.2, "temporal": 0.1}
    base = 0.0

    def sub(name):
        w = stage_w[name]
        b = base
        return lambda s, f: progress(s, min(0.9, b + w * f))

    res = run_protocol("Fight-level CV", examples, splits, archs, classes, cfg, device, sub("fight_cv"))
    protocols["fight_cv"] = {"description": f"{k}-fold cross-validation grouped by fight", "folds": res["folds"],
                             "models": summarise(res, examples, classes, archs, styles)}
    oof = res
    base += stage_w["fight_cv"]
    if len(fighters) >= 2:
        kf = min(int(cfg.get("folds", 5)), len(fighters))
        fsplits = list(GroupKFold(n_splits=kf).split(np.zeros(n), groups=np.array([e.fighter_key for e in examples])))
        r2 = run_protocol("Unseen-fighter CV", examples, fsplits, archs, classes, cfg, device, sub("fighter_cv"))
        protocols["fighter_cv"] = {"description": f"{kf}-fold cross-validation grouped by fighter (all test fighters unseen)",
                                   "folds": r2["folds"], "models": summarise(r2, examples, classes, archs, styles)}
    else:
        protocols["fighter_cv"] = {"description": "not possible — only one fighter in the data", "models": {}}
    base += stage_w["fighter_cv"]
    first_seen = {}
    for i, e in enumerate(examples):
        first_seen.setdefault(e.fight_key, i)
    order = sorted(fights, key=lambda f: ((fight_dates or {}).get(f) or "9999", first_seen[f]))
    if len(order) >= 3:
        cut = max(1, int(round(len(order) * 0.3)))
        test_f = set(order[-cut:])
        tr = [i for i, e in enumerate(examples) if e.fight_key not in test_f]
        te = [i for i, e in enumerate(examples) if e.fight_key in test_f]
        r3 = run_protocol("Temporal hold-out", examples, [(tr, te)], archs, classes, cfg, device, sub("temporal"))
        protocols["temporal"] = {"description": f"train on the earliest {len(order) - cut} fights, test on the latest {cut} "
                                 "(by fight date, else upload order)", "folds": r3["folds"],
                                 "models": summarise(r3, examples, classes, archs, styles)}
    else:
        protocols["temporal"] = {"description": "not possible — fewer than 3 fights", "models": {}}

    main = protocols["fight_cv"]["models"]
    valid = {a: m for a, m in main.items() if a in archs and m.get("n")}
    best = min(valid, key=lambda a: (valid[a]["log_loss"], -valid[a]["top1_accuracy"])) if valid else "markov"
    experimental = (n < int(cfg.get("min_examples_reliable", 1000)) or len(fights) < int(cfg.get("min_fights_reliable", 10)))
    return {"status": "ok", "data": data, "architectures": archs, "skipped": skipped, "protocols": protocols,
            "selected": best, "selection_rule": "lowest log loss under fight-level cross-validation (ties: higher top-1)",
            "experimental": experimental, "styles": styles, "seconds": round(time.monotonic() - t0, 1),
            "_oof": oof}
