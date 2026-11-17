"""Phase 5 orchestration: data assembly, model registry & bundles, prediction with evidence and reliability,
offline video prediction, evaluation of stored predictions and background jobs."""
from __future__ import annotations

import logging
import math
import queue
import shutil
import statistics
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.actions.taxonomy import BY_KEY, label
from app.core.config import get_settings
from app.db.database import session_scope
from app.db.models import AppSetting, JobStatus
from app.db.prediction_models import (
    ExternalSequence,
    PredictionEvaluation,
    PredictionModel,
    PredictionResult,
    PredictionTrainingJob,
    PredictionVideoJob,
)
from app.db.vision_models import FighterDistance, MovementMetric, TrackingSession
from app.patterns.engine import Obs
from app.patterns.extract import Extraction, apply_filters, extract
from app.patterns.settings import get_pattern_settings
from app.predictions.dataset import (
    MOVEMENT_FIELDS,
    RARE,
    Context,
    Example,
    FighterStats,
    Step,
    VideoContext,
    build_examples,
    dataset_version,
)
from app.predictions.features import FeatureBuilder
from app.predictions.models import load_predictor, make_predictor
from app.predictions.training import class_label, split_calibration, train_and_evaluate, vocabulary
from app.vision.job_manager import current_job as current_vision_job

log = logging.getLogger(__name__)
ACTIVE = (JobStatus.queued, JobStatus.processing)
KEY = "predictions"


def _now():
    return datetime.now(timezone.utc)


# ============================================================================ settings
class PredictionSettings(BaseModel):
    history_length: int = Field(10, ge=2, le=10, description="Previous actions used as input (2-10)")
    min_class_count: int = Field(5, ge=1, le=1000, description="Actions with fewer training examples share an 'Other (rare)' class")
    folds: int = Field(5, ge=2, le=10)
    architectures: list[Literal["gru", "lstm", "transformer"]] = Field(default_factory=lambda: ["gru", "lstm", "transformer"])
    min_examples_deep: int = Field(300, ge=10, description="Sequence networks are trained only with at least this many examples")
    min_examples_transformer: int = Field(1000, ge=10)
    min_examples_reliable: int = Field(1000, ge=10, description="Below this, models are labelled experimental")
    min_fights_reliable: int = Field(10, ge=2)
    include_external: bool = Field(True, description="Include imported real action sequences (e.g. BoxingVI) in training")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    beam_width: int = Field(5, ge=1, le=20)
    min_support: int = Field(5, ge=1, description="Fewer matching historical observations -> insufficient-data warning")


def get_prediction_settings(db: Session) -> PredictionSettings:
    row = db.get(AppSetting, KEY)
    data = PredictionSettings().model_dump()
    if row and row.value:
        data.update({k: v for k, v in row.value.items() if k in PredictionSettings.model_fields})
    try:
        return PredictionSettings.model_validate(data)
    except Exception:
        return PredictionSettings()


def save_prediction_settings(db: Session, s: PredictionSettings) -> PredictionSettings:
    row = db.get(AppSetting, KEY)
    if row is None:
        db.add(AppSetting(key=KEY, value=s.model_dump()))
    else:
        row.value = s.model_dump()
    db.commit()
    return s


def resolve_device(pref: str) -> str:
    try:
        import torch

        if pref in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


# ============================================================================ data assembly
def video_contexts(db: Session, video_ids: set[str]) -> tuple[dict[str, VideoContext], dict]:
    ctx, tracks_of = {}, defaultdict(set)
    for vid in video_ids:
        job = current_vision_job(db, vid)
        if not job:
            continue
        dist = db.execute(select(FighterDistance.track_a, FighterDistance.track_b, FighterDistance.timestamp,
                                 FighterDistance.distance_torso, FighterDistance.closing_speed)
                          .where(FighterDistance.job_id == job.id)).all()
        cols = [getattr(MovementMetric, f) for f in ("track_id", "timestamp", "state", "direction_deg") + MOVEMENT_FIELDS]
        mv = [dict(zip(("track_id", "timestamp", "state", "direction_deg") + MOVEMENT_FIELDS, r, strict=False))
              for r in db.execute(select(*cols).where(MovementMetric.job_id == job.id)).all()]
        ctx[vid] = VideoContext([tuple(r) for r in dist], mv)
        for ts in db.scalars(select(TrackingSession).where(TrackingSession.job_id == job.id)).all():
            if ts.assigned_fighter_id:
                tracks_of[(vid, ts.assigned_fighter_id)].add(ts.track_id)
    return ctx, tracks_of


def external_obs(db: Session) -> list[Obs]:
    rows = db.scalars(select(ExternalSequence).order_by(ExternalSequence.fight_key, ExternalSequence.start)).all()
    return [Obs(fighter_key=r.fighter_key, video_id=f"ext-{r.fight_key}", fight_key=r.fight_key, category=r.category,
                start=r.start, end=r.end, confidence=1.0, source=f"external:{r.dataset}", action_id=f"ext{r.id}", stance=r.stance)
            for r in rows]


def assemble(db: Session, ps: PredictionSettings, include_last: bool = False, video_id: str | None = None) -> tuple[list[Example], Extraction]:
    pset = get_pattern_settings(db)
    cfg = pset.engine_config()
    ex = extract(db, pset)
    obs = apply_filters(ex, videos=[video_id] if video_id else None)
    vctx, tracks_of = video_contexts(db, {o.video_id for o in obs})
    for o in obs:  # unassigned tracks are their own fighter key
        if o.track_id is not None:
            tracks_of[(o.video_id, o.fighter_key)].add(o.track_id)
    examples = build_examples(obs, cfg, vctx, ps.history_length, include_last, tracks_of)
    if ps.include_external and not video_id:
        examples += build_examples(external_obs(db), cfg, None, ps.history_length, include_last)
    return examples, ex


# ============================================================================ bundles
def models_root() -> Path:
    d = get_settings().data_dir / "prediction_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ngram_index(examples: list[Example], K: int = 3) -> dict:
    """(fighter or '*', context tuple) -> {next: count} and fights, for evidence."""
    idx: dict[tuple, Counter] = defaultdict(Counter)
    fights: dict[tuple, set] = defaultdict(set)
    for e in examples:
        cats = [s.category for s in e.history]
        for k in range(1, K + 1):
            if k > len(cats):
                break
            for who in ("*", e.fighter_key):
                key = (who, tuple(cats[-k:]))
                idx[key][e.target] += 1
                fights[(key, e.target)].add(e.fight_key)
    return {"counts": {k: dict(v) for k, v in idx.items()}, "fights": {k: len(v) for k, v in fights.items()}}


def context_profile(examples: list[Example]) -> dict:
    """Per fighter: distance / timing distributions and reactions to opponent actions (for evidence and OOD)."""
    prof: dict[str, dict] = defaultdict(lambda: {"distance": [], "interval": [], "duration": [], "after_opponent": defaultdict(Counter)})
    for e in examples:
        for who in ("*", e.fighter_key):
            p = prof[who]
            if e.context.distance is not None:
                p["distance"].append(e.context.distance)
            h = e.history
            if len(h) >= 2 and h[-1].start is not None and h[-2].end is not None:
                p["interval"].append(h[-1].start - h[-2].end)
            if h[-1].duration is not None:
                p["duration"].append(h[-1].duration)
            if e.context.opponent_action:
                p["after_opponent"][e.context.opponent_action][e.target] += 1

    def q(xs):
        if len(xs) < 3:
            return None
        s = sorted(xs)
        return {"n": len(s), "median": round(statistics.median(s), 3), "p10": round(s[int(0.1 * (len(s) - 1))], 3),
                "p90": round(s[int(0.9 * (len(s) - 1))], 3)}

    return {k: {"distance": q(v["distance"]), "interval": q(v["interval"]), "duration": q(v["duration"]),
                "after_opponent": {a: dict(c) for a, c in v["after_opponent"].items()}} for k, v in prof.items()}


def train_job(job_id: str, progress) -> dict:
    with session_scope() as db:
        job = db.get(PredictionTrainingJob, job_id)
        ps = PredictionSettings.model_validate({k: v for k, v in job.config.items() if k in PredictionSettings.model_fields})
    device = resolve_device(ps.device)
    progress("Assembling dataset from Phases 1-4", 0.01)
    with session_scope() as db:
        examples, ex = assemble(db, ps)
        fight_dates = {o.fight_key: o.fight_date for o in ex.obs}
    examples = [e for e in examples if e.target]
    dv = dataset_version(examples)
    cfg = {**ps.model_dump(), "params": {}}
    report = train_and_evaluate(examples, cfg, device, lambda s, f: progress(s, 0.03 + 0.8 * f), fight_dates)
    report.pop("_oof", None)
    info = {**report.get("data", {}), "dataset_version": dv, "device": device}
    if report["status"] != "ok":
        return {"report": report, "info": info, "models": []}
    # production models: all data, calibration/early stopping on the latest fights
    with session_scope() as db:
        version = (db.scalar(select(func.max(PredictionModel.version))) or 0) + 1
    vdir = models_root() / f"v{version}"
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    classes = report["data"]["classes"]
    stats = FighterStats(examples)
    fit, cal = split_calibration(examples)
    fb = FeatureBuilder(classes, vocabulary(), stats)
    created = []
    for i, arch in enumerate(report["architectures"]):
        progress(f"Training final {arch} model on all data", 0.85 + 0.12 * i / len(report["architectures"]))
        fbi = FeatureBuilder(classes, vocabulary(), stats)
        m = make_predictor(arch, {}, device).fit(fit, fbi, cal or None)
        m.save(vdir / arch)
        created.append((arch, m.temperature, m.params, getattr(m, "importance", lambda: [])() if arch == "xgboost" else []))
    # ranges of context features in training (for out-of-distribution warnings)
    raw = np.stack([fb.static_raw(e, e.fight_key)[:len(fb.context_names())] for e in examples])
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ranges = {n: {"p01": float(np.nanpercentile(raw[:, j], 1)), "p99": float(np.nanpercentile(raw[:, j], 99)),
                      "observed": int(np.sum(~np.isnan(raw[:, j])))}
                  for j, n in enumerate(fb.context_names()) if np.sum(~np.isnan(raw[:, j])) >= 10}
    joblib.dump({"stats": stats, "ngrams": ngram_index(examples), "profile": context_profile(examples), "ranges": ranges,
                 "train_fights": sorted({e.fight_key for e in examples}), "fighter_examples": dict(Counter(e.fighter_key for e in examples)),
                 "classes": classes, "history_length": ps.history_length}, vdir / "bundle.joblib")
    out = []
    with session_scope() as db:
        sel = report["selected"]
        for arch, T, params, imp in created:
            metrics = report["protocols"]["fight_cv"]["models"].get(arch)
            metrics = {**(metrics or {}), "unseen_fighter_cv": _brief(report["protocols"].get("fighter_cv", {}).get("models", {}).get(arch)),
                       "temporal": _brief(report["protocols"].get("temporal", {}).get("models", {}).get(arch)),
                       "feature_importance": imp}
            pm = PredictionModel(version=version, architecture=arch, selected=arch == sel, experimental=report["experimental"],
                                 dataset_version=dv, dataset_info=info, metrics=metrics, params=params, temperature=T,
                                 path=str(vdir / arch), training_job_id=job_id)
            db.add(pm)
            db.flush()
            out.append({"id": pm.id, "architecture": arch, "selected": arch == sel})
        sel_id = next(o["id"] for o in out if o["selected"])
        db.execute(update(PredictionModel).values(is_active=False))
        db.execute(update(PredictionModel).where(PredictionModel.id == sel_id).values(is_active=True))
    MODEL_CACHE.clear()
    return {"report": report, "info": info, "models": out, "version": version}


def _brief(m: dict | None) -> dict | None:
    if not m or not m.get("n"):
        return None
    return {k: m.get(k) for k in ("n", "top1_accuracy", "top3_accuracy", "macro_f1", "log_loss", "brier", "ece")}


# ============================================================================ loading & caching
class _Cache:
    def __init__(self):
        self._d: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def get(self, pm: PredictionModel, device: str):
        with self._lock:
            hit = self._d.get(pm.id)
            if hit:
                return hit
            bundle = joblib.load(Path(pm.path).parent / "bundle.joblib")
            meta_fb = __import__("json").loads((Path(pm.path) / "meta.json").read_text())["features"]
            fb = FeatureBuilder.from_state(meta_fb, bundle["stats"])
            model = load_predictor(Path(pm.path), fb, device)
            markov = model if pm.architecture == "markov" else None
            if markov is None and (Path(pm.path).parent / "markov").exists():
                mfb = FeatureBuilder.from_state(__import__("json").loads((Path(pm.path).parent / "markov" / "meta.json").read_text())["features"], bundle["stats"])
                markov = load_predictor(Path(pm.path).parent / "markov", mfb, device)
            self._d[pm.id] = (model, fb, bundle, markov)
            return self._d[pm.id]

    def clear(self):
        with self._lock:
            self._d.clear()


MODEL_CACHE = _Cache()


def active_model(db: Session, model_id: str | None = None) -> PredictionModel | None:
    if model_id:
        return db.get(PredictionModel, model_id)
    return db.scalar(select(PredictionModel).where(PredictionModel.is_active.is_(True)).limit(1))


# ============================================================================ prediction
def _dist_list(P: np.ndarray, classes: list[str]) -> list[dict]:
    order = np.argsort(-P)
    return [{"action": classes[i], "label": class_label(classes[i]), "probability": round(float(P[i]), 4)} for i in order]


def _accuracy_at_confidence(pm: PredictionModel, conf: float) -> dict | None:
    """Empirical top-1 accuracy of cross-validated predictions with similar confidence (reliability diagram)."""
    for b in (pm.metrics or {}).get("reliability_diagram", []) or []:
        lo, hi = b["bin"]
        if lo < conf <= hi or (lo == 0 and conf <= hi):
            return {"bin": b["bin"], "n": b["n"], "observed_accuracy": b["accuracy"]}
    return None


def evidence_and_reliability(pm: PredictionModel, bundle: dict, e: Example, P: np.ndarray, raw: np.ndarray,
                             classes: list[str], ps: PredictionSettings, fb: FeatureBuilder) -> tuple[dict, dict]:
    cats = [s.category for s in e.history]
    ng, nf = bundle["ngrams"]["counts"], bundle["ngrams"]["fights"]
    top = [classes[i] for i in np.argsort(-P)[:3]]
    combos, support = [], {}
    for k in range(min(3, len(cats)), 0, -1):
        ctx = tuple(cats[-k:])
        for who in (e.fighter_key, "*"):
            cnt = ng.get((who, ctx), {})
            n = sum(cnt.values())
            support[f"{'fighter' if who != '*' else 'all'}_order{k}"] = n
            for t in top:
                if cnt.get(t):
                    combos.append({"sequence": [label(c) for c in ctx] + [class_label(t)], "scope": "this fighter" if who != "*" else "all fighters",
                                   "count": cnt[t], "of": n, "fights": nf.get(((who, ctx), t), 0)})
    combos = sorted(combos, key=lambda c: (-len(c["sequence"]), c["scope"] != "this fighter", -c["count"]))[:6]
    stats: FighterStats = bundle["stats"]
    dist = stats.distribution(e.fighter_key, None)
    after = stats.transitions_from(e.fighter_key, cats[-1], None)
    n_f = sum(dist.values())
    tend = {"fighter_training_examples": n_f,
            "most_frequent_actions": [{"action": label(a), "share": round(c / n_f, 3)} for a, c in dist.most_common(4)] if n_f else [],
            "usual_follow_ups_after_last_action": [{"action": label(a), "count": c, "of": sum(after.values())} for a, c in after.most_common(3)]}
    prof = bundle["profile"].get(e.fighter_key) or bundle["profile"].get("*") or {}
    notes = []
    c = e.context
    if c.distance is not None and prof.get("distance"):
        d = prof["distance"]
        where = "closer than" if c.distance < d["p10"] else "farther than" if c.distance > d["p90"] else "within"
        notes.append(f"Current distance {c.distance:.2f} torso lengths is {where} this fighter's usual range "
                     f"({d['p10']}–{d['p90']}, median {d['median']}, n={d['n']}).")
    if c.opponent_action:
        ao = (prof.get("after_opponent") or {}).get(c.opponent_action, {})
        if ao:
            best = max(ao, key=ao.get)
            notes.append(f"Opponent's last action: {label(c.opponent_action)} ({c.opponent_lag:.2f} s ago). In training, after that "
                         f"opponent action this fighter's next action was most often {class_label(best)} ({ao[best]} of {sum(ao.values())}).")
        else:
            notes.append(f"Opponent's last action: {label(c.opponent_action)} — no training examples of this fighter after it.")
    # out-of-distribution / insufficiency warnings
    warns = []
    unknown = [x for x in cats if x not in BY_KEY]
    if unknown:
        warns.append(f"Unknown action(s) in the history: {', '.join(unknown)}.")
    if not stats.total.get(e.fighter_key):
        warns.append("This fighter has no training examples — prediction relies on all fighters' patterns.")
    if support.get("all_order1", 0) < ps.min_support:
        warns.append(f"The last action was followed by another action only {support.get('all_order1', 0)} time(s) in training.")
    if len(cats) >= 2 and support.get("all_order2", 0) == 0:
        warns.append("The last two actions never occurred together in training.")
    if any(s.start is None for s in e.history):
        warns.append("Action timing was not provided — timing features are missing.")
    vals = dict(zip(fb.context_names(), fb.context_numeric(e), strict=False))
    for n, r in bundle["ranges"].items():
        v = vals.get(n)
        if v is not None and not math.isnan(v) and (v < r["p01"] or v > r["p99"]):
            warns.append(f"{n.replace('_', ' ')} = {v:.2f} is outside the 1st–99th percentile seen in training.")
    m = pm.metrics or {}
    ent = float(-(P * np.log(np.clip(P, 1e-12, 1))).sum() / math.log(len(P))) if len(P) > 1 else 0.0
    acc = _accuracy_at_confidence(pm, float(P.max()))
    info = pm.dataset_info or {}
    if pm.experimental and support.get("all_order1", 0) < ps.min_support or (not stats.total.get(e.fighter_key) and support.get("all_order1", 0) < ps.min_support):
        level = "insufficient"
    elif pm.experimental or warns:
        level = "low"
    elif m.get("ece", 1) <= 0.05 and support.get("all_order1", 0) >= 50:
        level = "high"
    else:
        level = "medium"
    reliability = {
        "level": level, "experimental_model": pm.experimental,
        "predicted_probability": round(float(P.max()), 4), "uncalibrated_model_confidence": round(float(raw.max()), 4),
        "historical_accuracy_at_this_confidence": acc,
        "uncertainty_normalised_entropy": round(ent, 4),
        "supporting_observations": support, "training_examples": info.get("examples"), "training_fights": info.get("fights"),
        "calibration": {"method": "temperature scaling", "temperature": pm.temperature, "cv_ece": m.get("ece"),
                        "cv_brier": m.get("brier"), "cv_log_loss": m.get("log_loss")},
        "cv_top1_accuracy": m.get("top1_accuracy"), "cv_top3_accuracy": m.get("top3_accuracy"),
        "warnings": warns,
        "explanation": "‘Predicted probability’ is the calibrated model output. ‘Historical accuracy at this confidence’ is how often "
                       "the top prediction was actually right for held-out predictions with similar confidence.",
    }
    evidence = {"supporting_combinations": combos, "fighter_tendencies": tend, "context_notes": notes}
    return evidence, reliability


def predict_examples(db: Session, pm: PredictionModel, examples: list[Example], with_evidence: bool = True,
                     contextual: bool = False) -> list[dict]:
    ps = get_prediction_settings(db)
    model, fb, bundle, _ = MODEL_CACHE.get(pm, resolve_device(ps.device))
    classes = fb.classes
    raw = model.raw_proba(examples)
    from app.predictions.models import apply_temperature

    P = apply_temperature(raw, model.temperature)
    out = []
    train_fights = set(bundle["train_fights"])
    for i, e in enumerate(examples):
        dist = _dist_list(P[i], classes)
        r = {"top3": dist[:3], "distribution": dist, "predicted_action": dist[0]["action"],
             "predicted_probability": dist[0]["probability"], "probability_sum": round(float(P[i].sum()), 4),
             "model": {"id": pm.id, "architecture": pm.architecture, "version": f"v{pm.version}-{pm.architecture}"},
             "in_training_data": e.fight_key in train_fights}
        if with_evidence:
            r["evidence"], r["reliability"] = evidence_and_reliability(pm, bundle, e, P[i], raw[i], classes, ps, fb)
        out.append(r)
    if contextual and examples:
        bare = [Example(e.fighter_key, e.fight_key, e.video_id, e.fight_date, e.history, Context(stance=e.context.stance), None, e.t)
                for e in examples]
        P0 = apply_temperature(model.raw_proba(bare), model.temperature)
        for i, r in enumerate(out):
            r["without_context"] = _dist_list(P0[i], classes)[:3]
            r["context_shift"] = [{"action": classes[j], "label": class_label(classes[j]),
                                   "change": round(float(P[i][j] - P0[i][j]), 4)} for j in np.argsort(-np.abs(P[i] - P0[i]))[:3]]
    return out


def next_sequences(db: Session, pm: PredictionModel, e: Example, depth: int, width: int) -> list[dict]:
    """Beam search over future actions; each step feeds the predicted action back as history.

    Timing of hypothetical actions uses the fighter's median duration / interval from training (stated in
    the response); context is held at its current value. These are possible sequences, not outcomes.
    """
    ps = get_prediction_settings(db)
    model, fb, bundle, _ = MODEL_CACHE.get(pm, resolve_device(ps.device))
    from app.predictions.models import apply_temperature

    prof = bundle["profile"].get(e.fighter_key) or bundle["profile"].get("*") or {}
    dur = (prof.get("duration") or {}).get("median", 0.4)
    gap = (prof.get("interval") or {}).get("median", 0.3)
    beams = [(0.0, [], e)]
    finished = []  # sequences ending in the 'rare' class cannot be extended (no concrete action to feed back)
    for _ in range(depth):
        live = [b for b in beams if b[2] is not None]
        if not live:
            break
        P = apply_temperature(model.raw_proba([b[2] for b in live]), model.temperature)
        cand = []
        for (lp, seq, ex), p in zip(live, P, strict=False):
            for j in np.argsort(-p)[:width]:
                c, lpn = fb.classes[j], lp + math.log(max(float(p[j]), 1e-12))
                if c == RARE:
                    finished.append((lpn, seq + [c], None))
                    continue
                last = ex.history[-1]
                start = (last.end if last.end is not None else ex.t) + gap
                h = (ex.history + [Step(c, start, start + dur, 1.0)])[-ps.history_length:]
                cand.append((lpn, seq + [c], Example(ex.fighter_key, ex.fight_key, ex.video_id, ex.fight_date, h, ex.context, None, start + dur)))
        beams = sorted(cand, key=lambda b: -b[0])[:width]
    final = sorted(beams + [f for f in finished if len(f[1]) < depth], key=lambda b: -b[0])[:width]
    return [{"sequence": s_, "labels": [class_label(x) for x in s_], "probability": round(math.exp(lp), 4),
             "assumed_timing": {"duration": dur, "gap": gap}} for lp, s_, _ in final]


# ============================================================================ offline video prediction
def run_video_job(job_id: str, progress) -> dict:
    with session_scope() as db:
        job = db.get(PredictionVideoJob, job_id)
        vid, fk, mid = job.video_id, job.fighter_key, job.model_id
        pm = active_model(db, mid)
        if pm is None:
            raise RuntimeError("No trained prediction model. Train models first.")
        ps = get_prediction_settings(db)
        examples, _ = assemble(db, ps, include_last=True, video_id=vid)
        if fk:
            examples = [e for e in examples if e.fighter_key == fk]
        examples.sort(key=lambda e: e.t)
        progress("Predicting", 0.1)
        preds = []
        for i in range(0, len(examples), 64):
            preds += predict_examples(db, pm, examples[i:i + 64])
            progress("Predicting", 0.1 + 0.8 * min(1.0, (i + 64) / max(1, len(examples))))
        n_eval = n1 = n3 = 0
        for e, r in zip(examples, preds, strict=False):
            pr = PredictionResult(fighter_key=e.fighter_key, video_id=vid, video_job_id=job_id, mode="video", at_time=e.t,
                                  source_action_id=e.action_id, sequence=[{"category": s.category, "start": s.start, "end": s.end} for s in e.history],
                                  context=_ctx_json(e.context), distribution={d["action"]: d["probability"] for d in r["distribution"]},
                                  top3=r["top3"], predicted_action=r["predicted_action"], predicted_probability=r["predicted_probability"],
                                  reliability=r["reliability"], model_id=pm.id, model_version=r["model"]["version"],
                                  in_training_data=r["in_training_data"])
            db.add(pr)
            db.flush()
            # the actual next action comes from later footage; it is compared, never used to change the prediction
            ev = _make_evaluation(pr, e.target, "video", pm)
            db.add(ev)
            if ev.status == "evaluated":
                n_eval += 1
                n1 += ev.correct_top1
                n3 += ev.correct_top3
        return {"predictions": len(examples), "evaluated": n_eval, "top1_correct": n1, "top3_correct": n3,
                "model": f"v{pm.version}-{pm.architecture}", "in_training_data": any(r["in_training_data"] for r in preds)}


def _ctx_json(c: Context) -> dict:
    return {"opponent_action": c.opponent_action, "opponent_lag": c.opponent_lag, "opponent_attacks_3s": c.opponent_attacks_3s,
            "distance": c.distance, "closing_speed": c.closing_speed, "stance": c.stance,
            "movement": {k: (round(v, 4) if v is not None else None) for k, v in (c.movement or {}).items()}}


def _make_evaluation(pr: PredictionResult, actual: str | None, source: str, pm: PredictionModel) -> PredictionEvaluation:
    top = [t["action"] for t in pr.top3]
    if actual is None:
        return PredictionEvaluation(prediction_id=pr.id, fighter_key=pr.fighter_key, status="no_next_action", actual_action=None,
                                    actual_source=source, predicted_action=pr.predicted_action, top3=top,
                                    in_training_data=pr.in_training_data)
    classes = list(pr.distribution.keys())
    mapped = actual if actual in classes else RARE
    return PredictionEvaluation(prediction_id=pr.id, fighter_key=pr.fighter_key, status="evaluated", actual_action=actual,
                                actual_source=source, predicted_action=pr.predicted_action, top3=top,
                                correct_top1=mapped == pr.predicted_action, correct_top3=mapped in top,
                                probability_of_actual=pr.distribution.get(mapped), in_training_data=pr.in_training_data)


# ============================================================================ background jobs
class PredictionJobManager:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        with session_scope() as db:
            for model in (PredictionTrainingJob, PredictionVideoJob):
                db.execute(update(model).where(model.status == JobStatus.processing).values(
                    status=JobStatus.failed, finished_at=_now(), error_message="Interrupted: the application stopped while this job was running."))
            q = [("train", j) for j in db.scalars(select(PredictionTrainingJob.id).where(PredictionTrainingJob.status == JobStatus.queued)).all()]
            q += [("video", j) for j in db.scalars(select(PredictionVideoJob.id).where(PredictionVideoJob.status == JobStatus.queued)).all()]
        for item in q:
            self._q.put(item)
        self._thread = threading.Thread(target=self._loop, name="prediction-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        if self._running:
            self._running = False
            self._q.put(None)
            if self._thread:
                self._thread.join(timeout=timeout)

    def train(self, db: Session) -> PredictionTrainingJob:
        ex = db.scalar(select(PredictionTrainingJob).where(PredictionTrainingJob.status.in_(ACTIVE)))
        if ex:
            return ex
        job = PredictionTrainingJob(config=get_prediction_settings(db).model_dump())
        db.add(job)
        db.commit()
        db.refresh(job)
        self._q.put(("train", job.id))
        return job

    def video(self, db: Session, video_id: str, fighter_key: str | None, model_id: str | None) -> PredictionVideoJob:
        ex = db.scalar(select(PredictionVideoJob).where(PredictionVideoJob.video_id == video_id, PredictionVideoJob.status.in_(ACTIVE)))
        if ex:
            return ex
        job = PredictionVideoJob(video_id=video_id, fighter_key=fighter_key, model_id=model_id)
        db.add(job)
        db.commit()
        db.refresh(job)
        self._q.put(("video", job.id))
        return job

    def _progress(self, model, jid):
        last = [0.0]

        def w(stage, frac):
            if time.monotonic() - last[0] < 0.5 and frac < 0.99:
                return
            last[0] = time.monotonic()
            with session_scope() as db:
                j = db.get(model, jid)
                if j:
                    j.progress = round(min(0.999, frac), 4)
                    if hasattr(j, "current_stage"):
                        j.current_stage = stage[:120]
        return w

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            kind, jid = item
            model = PredictionTrainingJob if kind == "train" else PredictionVideoJob
            with session_scope() as db:
                j = db.get(model, jid)
                if j is None or j.status != JobStatus.queued:
                    continue
                j.status = JobStatus.processing
                if hasattr(j, "started_at"):
                    j.started_at = _now()
            try:
                if kind == "train":
                    res = train_job(jid, self._progress(model, jid))
                    with session_scope() as db:
                        j = db.get(model, jid)
                        j.result, j.dataset_info, j.version = res["report"], res["info"], res.get("version")
                        j.status, j.progress, j.finished_at, j.current_stage = JobStatus.completed, 1.0, _now(), "Done"
                        if res["report"]["status"] != "ok":
                            j.current_stage = "Not enough data to train and evaluate"
                else:
                    res = run_video_job(jid, self._progress(model, jid))
                    with session_scope() as db:
                        j = db.get(model, jid)
                        j.stats, j.status, j.progress, j.finished_at = res, JobStatus.completed, 1.0, _now()
            except Exception as exc:
                log.exception("prediction job %s failed", jid)
                with session_scope() as db:
                    j = db.get(model, jid)
                    j.status, j.error_message, j.finished_at = JobStatus.failed, str(exc)[:2000], _now()
