"""Deployment gates: a newly trained model replaces the working one only if it passes these checks.

The checks use held-out metrics only (never training-set scores). A model that fails can still be activated
manually with ``force`` (recorded in the deployment history), but training never does that automatically.
"""
from __future__ import annotations

ACTION_MIN_EVALUATED = 20
ACTION_MIN_F1 = 0.2
ACTION_MAX_F1_DROP = 0.05
PREDICTION_MIN_TEST = 50
PREDICTION_MAX_LOGLOSS_RISE = 0.05


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(ok), "detail": detail}


def validate_action_model(metrics: dict | None, active_metrics: dict | None) -> dict:
    m = metrics or {}
    checks = []
    has_eval = bool(m) and "error" not in m and m.get("f1_macro") is not None
    checks.append(_check("held_out_evaluation", has_eval, "evaluated on held-out videos" if has_eval else f"no valid evaluation ({m.get('error', 'missing')})"))
    n = m.get("n_evaluated") or 0
    checks.append(_check("evaluation_size", n >= ACTION_MIN_EVALUATED, f"{n} held-out samples (need >= {ACTION_MIN_EVALUATED})"))
    f1 = m.get("f1_macro")
    checks.append(_check("minimum_quality", f1 is not None and f1 >= ACTION_MIN_F1, f"macro F1 {f1} (need >= {ACTION_MIN_F1})"))
    if active_metrics and active_metrics.get("f1_macro") is not None and f1 is not None:
        a = active_metrics["f1_macro"]
        checks.append(_check("not_worse_than_active", f1 >= a - ACTION_MAX_F1_DROP,
                             f"macro F1 {f1:.3f} vs active {a:.3f} (max drop {ACTION_MAX_F1_DROP})"))
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def validate_prediction_model(arch_metrics: dict | None, floor_metrics: dict | None, active_metrics: dict | None) -> dict:
    m = arch_metrics or {}
    checks = []
    n = m.get("n") or 0
    checks.append(_check("held_out_evaluation", n > 0, f"{n} held-out predictions (fight-level CV)"))
    checks.append(_check("evaluation_size", n >= PREDICTION_MIN_TEST, f"{n} held-out predictions (need >= {PREDICTION_MIN_TEST})"))
    if floor_metrics and floor_metrics.get("log_loss") is not None and m.get("log_loss") is not None:
        checks.append(_check("beats_class_frequency", m["log_loss"] < floor_metrics["log_loss"],
                             f"log loss {m['log_loss']:.3f} vs class-frequency floor {floor_metrics['log_loss']:.3f}"))
    else:
        checks.append(_check("beats_class_frequency", False, "no comparison available"))
    if active_metrics and active_metrics.get("log_loss") is not None and m.get("log_loss") is not None:
        checks.append(_check("not_worse_than_active", m["log_loss"] <= active_metrics["log_loss"] + PREDICTION_MAX_LOGLOSS_RISE,
                             f"log loss {m['log_loss']:.3f} vs active {active_metrics['log_loss']:.3f} (max rise {PREDICTION_MAX_LOGLOSS_RISE})"))
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
