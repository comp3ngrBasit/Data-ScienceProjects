"""Phase 6 analytics: every number is computed from stored database records of Phases 1-5.

A small time-to-live cache keyed by a cheap database fingerprint avoids recomputing dashboard analytics on
every request; the fingerprint changes whenever actions, analyses, predictions or models change.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.actions.job_manager import active_model as active_action_model
from app.actions.job_manager import current_action_job
from app.actions.taxonomy import group as action_group
from app.actions.taxonomy import label
from app.db.action_models import ActionAnalysisJob, ActionAnnotation, ActionModelVersion, RecognizedAction, VideoAnnotationMeta
from app.db.models import Fighter, JobStatus, Video, VideoStatus
from app.db.pattern_models import PatternAnalysisJob, PatternSnapshot
from app.db.platform_models import ModelDeployment, PipelineRun
from app.db.prediction_models import PredictionEvaluation, PredictionModel, PredictionResult
from app.db.vision_models import MovementMetric, TrackingSession, VisionJob
from app.patterns import engine as E
from app.patterns.extract import OBS_FIELDS, Extraction, apply_filters, extract, is_track_key
from app.patterns.settings import PatternSettings, get_pattern_settings
from app.patterns.stats import wilson
from app.vision.job_manager import current_job as current_vision_job

# ============================================================================ cache
_cache: dict[tuple, tuple[float, str, object]] = {}
_lock = threading.Lock()
TTL = 30.0


def fingerprint(db: Session) -> str:
    parts = [
        db.scalar(select(func.count(RecognizedAction.id))), db.scalar(select(func.max(ActionAnalysisJob.finished_at))),
        db.scalar(select(func.count(ActionAnnotation.id))), db.scalar(select(func.max(ActionAnnotation.updated_at))),
        db.scalar(select(func.max(VisionJob.finished_at))), db.scalar(select(func.count(PredictionResult.id))),
        db.scalar(select(func.count(PredictionModel.id))), db.scalar(select(func.max(PatternAnalysisJob.finished_at))),
        db.scalar(select(func.count(Video.id))), db.scalar(select(func.max(Fighter.updated_at))),
        db.scalar(select(func.count(RecognizedAction.id)).where(RecognizedAction.review_status != "unreviewed")),
        db.scalar(select(func.count(TrackingSession.id)).where(TrackingSession.assigned_fighter_id.is_not(None))),
    ]
    return "|".join(str(p) for p in parts)


def cached(db: Session, key: tuple, fn):
    fp = fingerprint(db)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[1] == fp and now - hit[0] < TTL:
            return hit[2]
    val = fn()
    with _lock:
        _cache[key] = (now, fp, val)
    return val


def _extraction(db: Session) -> tuple[Extraction, PatternSettings]:
    ps = get_pattern_settings(db)
    return extract(db, ps), ps


# ============================================================================ overview
def overview(db: Session) -> dict:
    def build():
        ex, ps = _extraction(db)
        videos_total = db.scalar(select(func.count(Video.id))) or 0
        processed = db.scalar(select(func.count(Video.id)).where(Video.status == VideoStatus.completed)) or 0
        vision_done = {v for (v,) in db.execute(select(VisionJob.video_id).where(VisionJob.status == JobStatus.completed)).all()}
        detected = 0
        for vid in vision_done:
            j = current_action_job(db, vid)
            if j:
                detected += db.scalar(select(func.count(RecognizedAction.id)).where(RecognizedAction.job_id == j.id,
                                                                                 RecognizedAction.review_status != "rejected")) or 0
        verified = db.scalar(select(func.count(ActionAnnotation.id)).where(ActionAnnotation.status == "verified")) or 0
        fights = {v["fight_key"] for v in ex.videos.values() if v["actions"]}
        fighters = {o.fighter_key for o in ex.obs}
        p = E.prepare(apply_filters(ex), ps.engine_config())
        combos = 0
        for fk in p.by_fighter:
            combos += len(E.combinations(E.fighter_sequences(p, fk), ps.engine_config()))
        am = active_action_model(db)
        pm = db.scalar(select(PredictionModel).where(PredictionModel.is_active.is_(True)))
        n_pred = db.scalar(select(func.count(PredictionResult.id))) or 0
        evald = db.execute(select(PredictionEvaluation.correct_top1, PredictionEvaluation.correct_top3)
                           .where(PredictionEvaluation.status == "evaluated", PredictionEvaluation.in_training_data.is_(False))).all()
        runs = db.scalars(select(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(5)).all()
        from app.db.prediction_models import ExternalSequence

        ext = dict(db.execute(select(ExternalSequence.dataset, func.count(ExternalSequence.id)).group_by(ExternalSequence.dataset)).all())
        return {
            "totals": {"videos": videos_total, "videos_processed": processed, "videos_analysed": len(vision_done),
                       "fights_analysed": len(fights), "fighters_analysed": len(fighters),
                       "fighter_profiles": db.scalar(select(func.count(Fighter.id))) or 0,
                       "actions_detected": detected, "verified_annotations": verified, "actions_in_analysis": len(ex.obs),
                       "combinations_identified": combos, "predictions_generated": n_pred,
                       "predictions_evaluated_held_out": len(evald),
                       "held_out_top1": sum(1 for e in evald if e[0]), "held_out_top3": sum(1 for e in evald if e[1]),
                       "imported_sequence_actions": ext},
            "models": {
                "action_recognition": None if am is None else {"version": am.version, "algorithm": am.algorithm,
                                                                "f1_macro": (am.metrics or {}).get("f1_macro"),
                                                                "accuracy": (am.metrics or {}).get("accuracy"),
                                                                "n_evaluated": (am.metrics or {}).get("n_evaluated")},
                "next_move": None if pm is None else {"id": pm.id, "name": f"v{pm.version}-{pm.architecture}", "experimental": pm.experimental,
                                                      "top1": (pm.metrics or {}).get("top1_accuracy"), "top3": (pm.metrics or {}).get("top3_accuracy"),
                                                      "log_loss": (pm.metrics or {}).get("log_loss"), "n": (pm.metrics or {}).get("n")},
            },
            "status": model_status_label(am, pm),
            "pipeline_runs": [{"id": r.id, "video_id": r.video_id, "status": r.status.value, "current_stage": r.current_stage,
                               "progress": r.progress} for r in runs],
            "definitions": {"combinations_identified": f"distinct action combinations per fighter seen >= {ps.min_occurrences} times",
                            "actions_detected": "model-recognised actions in the current analysis of each video (rejected ones excluded)",
                            "fights_analysed": "distinct fights (fight key / video) with at least one usable action",
                            "imported_sequence_actions": "real action sequences imported without video (e.g. BoxingVI) — used for model training, not counted above"},
        }
    return cached(db, ("overview",), build)


def model_status_label(am, pm) -> dict:
    a = "not_trained" if am is None else "evaluated" if (am.metrics or {}).get("f1_macro") is not None else "unevaluated"
    p = "not_trained" if pm is None else ("experimental" if pm.experimental else "evaluated")
    return {"action_recognition": a, "next_move": p}


# ============================================================================ fighters
def _observed_time(db: Session) -> dict[str, float]:
    """Seconds each fighter profile is visible in analysed footage (union of its tracks' spans per video)."""
    out: dict[str, float] = defaultdict(float)
    vids = [v for (v,) in db.execute(select(Video.id)).all()]
    for vid in vids:
        j = current_vision_job(db, vid)
        if not j:
            continue
        spans = defaultdict(list)
        for t in db.scalars(select(TrackingSession).where(TrackingSession.job_id == j.id, TrackingSession.assigned_fighter_id.is_not(None))).all():
            spans[t.assigned_fighter_id].append((t.first_timestamp, t.last_timestamp))
        for fid, iv in spans.items():
            iv.sort()
            cur_s, cur_e, tot = iv[0][0], iv[0][1], 0.0
            for s, e in iv[1:]:
                if s > cur_e:
                    tot += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            out[fid] += tot + cur_e - cur_s
    return out


def fighters_table(db: Session) -> list[dict]:
    def build():
        ex, ps = _extraction(db)
        cfg = ps.engine_config()
        p = E.prepare(apply_filters(ex), cfg)
        obs_time = _observed_time(db)
        n_pred = dict(db.execute(select(PredictionResult.fighter_key, func.count(PredictionResult.id)).group_by(PredictionResult.fighter_key)).all())
        snaps = {s.fighter_key for s in db.scalars(select(PatternSnapshot)).all()}
        rows = {}
        for f in db.scalars(select(Fighter)).all():
            rows[f.id] = {"fighter_key": f.id, "name": f.name, "is_profile": True, "stance": f.stance.value, "weight_class": f.weight_class}
        for fk in p.by_fighter:
            rows.setdefault(fk, {"fighter_key": fk, "name": ex.names.get(fk, fk), "is_profile": False, "stance": None, "weight_class": None})
        out = []
        for fk, r in rows.items():
            obs = p.by_fighter.get(fk, [])
            c = Counter(o.category for o in obs)
            off = sum(1 for o in obs if o.group == "offensive")
            seqs = E.fighter_sequences(p, fk)
            out.append({**r, "actions": len(obs), "videos": len({o.video_id for o in obs}), "fights": len({o.fight_key for o in obs}),
                        "observed_seconds": round(obs_time.get(fk, 0.0), 2),
                        "top_action": label(c.most_common(1)[0][0]) if c else None,
                        "offensive_share": round(off / len(obs), 4) if obs else None,
                        "combinations": len(E.combinations(seqs, cfg)) if seqs else 0,
                        "predictions": n_pred.get(fk, 0), "pattern_analysis": fk in snaps})
        out.sort(key=lambda d: (not d["is_profile"], -d["actions"]))
        return out
    return cached(db, ("fighters",), build)


def global_analytics(db: Session, fighter: str | None = None) -> dict:
    def build():
        ex, ps = _extraction(db)
        cfg = ps.engine_config()
        p = E.prepare(apply_filters(ex), cfg)
        keys = [fighter] if fighter else list(p.by_fighter)
        seqs = [s for k in keys for s in E.fighter_sequences(p, k)]
        combos = []
        for k in keys:
            for c in E.combinations(E.fighter_sequences(p, k), cfg):
                combos.append({**c, "fighter_key": k, "fighter": ex.names.get(k, k)})
        combos.sort(key=lambda c: (-c["occurrences"], -c["length"]))
        states = sorted({o.category for s in seqs for o in s})
        mk = E.markov(seqs, cfg, 1, states) if seqs else None
        lens = Counter(len(s) for s in seqs)
        group_tr = Counter()
        for s in seqs:
            for a, b in zip(s, s[1:], strict=False):
                group_tr[(a.group, b.group)] += 1
        tot = sum(group_tr.values())
        dist = Counter(o.category for s in seqs for o in s)
        opp = {}
        for k in keys[:20]:
            res = E.opponent_analysis(p, k, cfg, ex.footage, ex.names, ps.min_actions_per_opponent)
            if res["opponents"]:
                opp[k] = {"fighter": ex.names.get(k, k), "opponents": [{"opponent": o["opponent"], "actions": o["actions"], "sufficient": o["sufficient"],
                                                                          "counter_rate": o["after_opponent_attacks"]["counter_rate"],
                                                                          "offensive_share": o["offensive_share"]} for o in res["opponents"]],
                          "difference_test": res["difference_test"]}
        return {"fighter": fighter, "sequences": len(seqs), "actions": sum(dist.values()),
                "action_distribution": [{"action": a, "label": label(a), "group": action_group(a), "count": n} for a, n in dist.most_common()],
                "top_combinations": combos[:25], "markov": mk,
                "sequence_length_distribution": [{"length": k, "count": v} for k, v in sorted(lens.items())],
                "group_transitions": [{"from": a, "to": b, "count": n, "share": round(n / tot, 4)} for (a, b), n in group_tr.most_common()],
                "opponent_patterns": opp}
    return cached(db, ("analytics", fighter), build)


# ============================================================================ fighter intelligence
def movement_tendencies(db: Session, fighter_id: str, video_ids: list[str] | None = None) -> dict:
    rows = []
    for (vid,) in db.execute(select(Video.id)).all():
        if video_ids and vid not in video_ids:
            continue
        j = current_vision_job(db, vid)
        if not j:
            continue
        tracks = [t for (t,) in db.execute(select(TrackingSession.track_id).where(TrackingSession.job_id == j.id,
                                                                                  TrackingSession.assigned_fighter_id == fighter_id)).all()]
        if not tracks:
            continue
        rows += db.execute(select(MovementMetric.state, MovementMetric.center_speed, MovementMetric.direction_deg,
                                  MovementMetric.left_wrist_speed, MovementMetric.right_wrist_speed,
                                  MovementMetric.left_ankle_speed, MovementMetric.right_ankle_speed)
                           .where(MovementMetric.job_id == j.id, MovementMetric.track_id.in_(tracks))).all()
    if not rows:
        return {"samples": 0}
    states = Counter(r[0] for r in rows)
    valid = [r for r in rows if r[0] in ("moving", "stationary")]

    def mean(i):
        v = [r[i] for r in valid if r[i] is not None]
        return round(statistics.fmean(v), 3) if v else None

    dirs = Counter()
    for r in valid:
        if r[0] == "moving" and r[2] is not None:
            d = r[2] % 360
            dirs["right" if d < 45 or d >= 315 else "up/away" if d < 135 else "left" if d < 225 else "down/toward camera"] += 1
    n = len(rows)
    return {"samples": n, "state_share": {k: round(v / n, 4) for k, v in states.items()},
            "mean_center_speed": mean(1), "mean_wrist_speed": {"left": mean(3), "right": mean(4)},
            "mean_ankle_speed": {"left": mean(5), "right": mean(6)},
            "screen_direction_when_moving": {k: round(v / max(1, sum(dirs.values())), 4) for k, v in dirs.most_common()},
            "units": "torso lengths per second (scale-normalised); direction is relative to the screen, not the opponent"}


def fighter_intelligence(db: Session, fighter_id: str, videos: list[str] | None = None) -> dict:
    ex, ps = _extraction(db)
    cfg = ps.engine_config()
    f = db.get(Fighter, fighter_id)
    obs_all = apply_filters(ex)
    p_all = E.prepare(obs_all, cfg)
    p = E.prepare(apply_filters(ex, videos=videos), cfg) if videos else p_all
    mine = p.by_fighter.get(fighter_id, [])
    seqs = E.fighter_sequences(p, fighter_id)
    tend = E.tendencies(mine, seqs, cfg, ex.footage, p.obs) if mine else None
    combos = E.combinations(seqs, cfg, len({o.fight_key for o in mine}) or 1)
    # opening actions = first action of each sequence; counters = fighter's offensive responses to opponent attacks
    openings = Counter(s[0].category for s in seqs)
    counters = Counter()
    for r in (tend or {}).get("defensive", {}).get("reactions_to_opponent_attacks", {}).get("by_attack", []):
        for resp in r["responses"]:
            if resp["action"] != E.NONE and action_group(resp["action"]) == "offensive":
                counters[(r["opponent_attack"], resp["action"])] += resp["count"]
    intervals = [o.interval_prev for o in mine if o.interval_prev is not None]
    # videos of this fighter (assigned tracks or attributed actions)
    vids = {o.video_id for o in p_all.by_fighter.get(fighter_id, [])}
    for (vid,) in db.execute(select(TrackingSession.video_id).where(TrackingSession.assigned_fighter_id == fighter_id).distinct()).all():
        vids.add(vid)
    if f:
        vids |= {v.id for v in f.videos}
    vrows = []
    overall = Counter(o.category for o in p_all.by_fighter.get(fighter_id, []))
    for v in db.scalars(select(Video).where(Video.id.in_(vids))).all():
        vo = [o for o in p_all.by_fighter.get(fighter_id, []) if o.video_id == v.id]
        c = Counter(o.category for o in vo)
        meta = ex.videos.get(v.id, {})
        vrows.append({"video_id": v.id, "title": v.title, "duration": v.duration_seconds, "fight_key": meta.get("fight_key"),
                      "fight_date": meta.get("fight_date"), "source": meta.get("source"), "actions": len(vo),
                      "attacks_per_minute": round(sum(1 for o in vo if o.group == "offensive") / (v.duration_seconds / 60), 3) if v.duration_seconds and vo else None,
                      "distribution": [{"action": a, "label": label(a), "count": n} for a, n in c.most_common()],
                      "js_distance_from_overall": E.js_distance(c, overall) if vo else None})
    vrows.sort(key=lambda r: (r["fight_date"] or "", r["title"]))
    timeline = [{"video_id": o.video_id, "fight_key": o.fight_key, "start": round(o.start, 3), "end": round(o.end, 3), "category": o.category,
                 "label": label(o.category), "group": o.group, "confidence": round(o.confidence, 4), "source": o.source,
                 "opponent_action": o.opponent_action} for o in sorted(mine, key=lambda o: (o.fight_key, o.video_id, o.start))]
    off = sum(1 for o in mine if o.group == "offensive")
    dfn = sum(1 for o in mine if o.group == "defensive")
    mv = sum(1 for o in mine if o.group == "movement")
    obs_time = _observed_time(db).get(fighter_id, 0.0)
    snap = db.scalar(select(PatternSnapshot).where(PatternSnapshot.fighter_key == fighter_id).order_by(PatternSnapshot.created_at.desc()).limit(1))
    return {
        "fighter": {"fighter_key": fighter_id, "name": f.name if f else ex.names.get(fighter_id, fighter_id), "is_profile": f is not None,
                    "nickname": f.nickname if f else None, "stance": f.stance.value if f else None, "weight_class": f.weight_class if f else None,
                    "height_cm": f.height_cm if f else None, "reach_cm": f.reach_cm if f else None},
        "filters": {"videos": videos},
        "videos": vrows,
        "observed": {
            "label": "Observed behaviour — counts of recognised / annotated actions",
            "analysed_seconds": round(obs_time, 2), "footage_seconds": round(sum(ex.footage.get(v, 0.0) for v in {o.video_id for o in mine}), 2),
            "actions": len(mine), "fights": len({o.fight_key for o in mine}),
            "strike_frequency": [{"action": a, "label": label(a), "count": n, "share": round(n / len(mine), 4), "ci95": wilson(n, len(mine))}
                                 for a, n in Counter(o.category for o in mine if o.group == "offensive").most_common()],
            "group_distribution": {"offensive": off, "defensive": dfn, "movement": mv, "other": len(mine) - off - dfn - mv},
            "mean_time_between_actions": round(statistics.fmean(intervals), 3) if intervals else None,
            "median_time_between_actions": round(statistics.median(intervals), 3) if intervals else None,
            "interval_samples": len(intervals),
            "opening_actions": [{"action": a, "label": label(a), "count": n, "share": round(n / len(seqs), 4)} for a, n in openings.most_common(6)],
            "counterattacks": [{"after": label(a), "counter": label(b), "count": n} for (a, b), n in counters.most_common(6)],
            "timeline": timeline,
            "movement": movement_tendencies(db, fighter_id, videos) if f else {"samples": 0},
        },
        "patterns": {
            "label": "Statistically identified patterns — with sample sizes and reliability",
            "combinations": combos[:15], "tendencies": tend,
            "opponents": E.opponent_analysis(p, fighter_id, cfg, ex.footage, ex.names, ps.min_actions_per_opponent) if mine else None,
            "stored_analysis_at": snap.created_at.isoformat() if snap else None,
        },
        "warnings": ([] if mine else ["No recognised or annotated actions are attributed to this fighter yet."])
        + (["All data comes from a single fight; patterns are not generalised."] if mine and len({o.fight_key for o in mine}) == 1 else []),
    }


# ============================================================================ pattern-recognition evaluation
def _ngram_occ(seqs: list[list[E.Obs]], n: int) -> list[tuple[str, tuple, float]]:
    out = []
    for s in seqs:
        for i in range(len(s) - n + 1):
            out.append((s[0].fighter_key, tuple(o.category for o in s[i:i + n]), s[i].start))
    return out


def pattern_evaluation(db: Session) -> dict:
    """How well Phase 4 patterns hold up, from real data only.

    (a) combination detection: n-grams found in model detections vs the same video's verified annotations
        (only videos that are fully annotated AND have a detection run); occurrences match when the sequence is
        identical, same fighter, and starts within 1 s;
    (b) frequency estimation: per (fighter, combination) counts from detections vs annotations;
    (c) transition reliability: first-order probabilities estimated on half of the fights vs the other half;
    (d) coverage: share of held-out transitions whose previous action / exact pair was seen in other fights.
    """
    def build():
        ps = get_pattern_settings(db)
        cfg = ps.engine_config()
        out: dict = {}
        metas = db.scalars(select(VideoAnnotationMeta).where(VideoAnnotationMeta.fully_annotated.is_(True))).all()
        both = [m.video_id for m in metas if current_action_job(db, m.video_id)]
        if both:
            det = apply_filters(extract(db, ps.model_copy(update={"source": "detections"}), use_cache=False), videos=both)
            ann = apply_filters(extract(db, ps.model_copy(update={"source": "annotations"}), use_cache=False), videos=both)
            pd_, pa = E.prepare(det, cfg), E.prepare(ann, cfg)
            res = {}
            for n in (2, 3):
                d_occ = _ngram_occ(list(pd_.sequences.values()), n)
                a_occ = _ngram_occ(list(pa.sequences.values()), n)
                used, tp = set(), 0
                for fk, g, t in d_occ:
                    j = next((i for i, (fa, ga, ta) in enumerate(a_occ) if i not in used and fa == fk and ga == g and abs(ta - t) <= 1.0), None)
                    if j is not None:
                        used.add(j)
                        tp += 1
                prec = tp / len(d_occ) if d_occ else None
                rec = tp / len(a_occ) if a_occ else None
                f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
                cd, ca = Counter((fk, g) for fk, g, _ in d_occ), Counter((fk, g) for fk, g, _ in a_occ)
                keys = set(cd) | set(ca)
                errs = [abs(cd[k] - ca[k]) for k in keys]
                res[f"n{n}"] = {"detected": len(d_occ), "ground_truth": len(a_occ), "matched": tp,
                                "precision": round(prec, 4) if prec is not None else None, "precision_ci95": wilson(tp, len(d_occ)) if d_occ else None,
                                "recall": round(rec, 4) if rec is not None else None, "recall_ci95": wilson(tp, len(a_occ)) if a_occ else None,
                                "f1": round(f1, 4) if f1 is not None else None,
                                "frequency_mae": round(statistics.fmean(errs), 3) if errs else None, "combinations_compared": len(keys)}
            out["combination_detection"] = {"videos": len(both), **res, "status": "ok" if any(r["ground_truth"] for r in res.values()) else "insufficient_data"}
        else:
            out["combination_detection"] = {"status": "insufficient_data", "videos": 0,
                                            "reason": "Needs videos that are marked fully annotated (verified annotations) AND have an action-recognition run."}
        ex = extract(db, ps)
        p = E.prepare(apply_filters(ex), cfg)
        fights = sorted({s[0].fight_key for s in p.sequences.values()})
        if len(fights) >= 2:
            half_a = set(fights[::2])
            pairs = {"a": defaultdict(Counter), "b": defaultdict(Counter)}
            for s in p.sequences.values():
                side = "a" if s[0].fight_key in half_a else "b"
                for x, y in zip(s, s[1:], strict=False):
                    pairs[side][x.category][y.category] += 1
            diffs, inside, compared = [], 0, 0
            for ctx in set(pairs["a"]) & set(pairs["b"]):
                na, nb = sum(pairs["a"][ctx].values()), sum(pairs["b"][ctx].values())
                if na < cfg.min_observations or nb < cfg.min_observations:
                    continue
                for nxt in set(pairs["a"][ctx]) | set(pairs["b"][ctx]):
                    pa_, pb_ = pairs["a"][ctx][nxt] / na, pairs["b"][ctx][nxt] / nb
                    diffs.append(abs(pa_ - pb_))
                    lo, hi = wilson(pairs["a"][ctx][nxt], na)
                    inside += lo <= pb_ <= hi
                    compared += 1
            out["transition_reliability"] = {
                "status": "ok" if compared else "insufficient_data", "fights_half_a": len(half_a), "fights_half_b": len(fights) - len(half_a),
                "probabilities_compared": compared, "mean_absolute_difference": round(statistics.fmean(diffs), 4) if diffs else None,
                "share_within_ci95": round(inside / compared, 4) if compared else None,
                "min_observations_per_half": cfg.min_observations,
                "reason": None if compared else f"No previous action has >= {cfg.min_observations} transitions in both halves of the fights."}
            seen_ctx = seen_pair = total = 0
            for fk in fights:
                train = defaultdict(set)
                for s in p.sequences.values():
                    if s[0].fight_key != fk:
                        for x, y in zip(s, s[1:], strict=False):
                            train[x.category].add(y.category)
                for s in p.sequences.values():
                    if s[0].fight_key == fk:
                        for x, y in zip(s, s[1:], strict=False):
                            total += 1
                            seen_ctx += x.category in train
                            seen_pair += y.category in train.get(x.category, set())
            out["coverage"] = {"status": "ok" if total else "insufficient_data", "held_out_transitions": total,
                               "previous_action_seen": round(seen_ctx / total, 4) if total else None,
                               "exact_transition_seen": round(seen_pair / total, 4) if total else None,
                               "method": "leave-one-fight-out"}
        else:
            reason = "Needs at least two fights with action sequences."
            out["transition_reliability"] = {"status": "insufficient_data", "reason": reason}
            out["coverage"] = {"status": "insufficient_data", "reason": reason}
        out["data"] = {"fights": len(fights), "sequences": len(p.sequences), "actions": len(p.obs)}
        return out
    return cached(db, ("pattern_eval",), build)


# ============================================================================ models
def action_model_json(m: ActionModelVersion) -> dict:
    met = m.metrics or {}
    return {"id": m.version, "kind": "action", "name": m.version, "architecture": m.algorithm, "status": m.status, "is_active": m.is_active,
            "dataset_version": m.dataset_version, "created_at": m.created_at.isoformat(), "classes": m.classes,
            "metrics": {k: met.get(k) for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "n_evaluated")},
            "evaluation_strategy": (met.get("evaluation") or {}).get("strategy"), "error": met.get("error"),
            "config": m.config, "path": m.path, "validation": (m.training_info or {}).get("validation"),
            "deployment_status": "active" if m.is_active else ("failed" if m.status != "ready" else "available")}


def prediction_model_json(m: PredictionModel) -> dict:
    met = m.metrics or {}
    return {"id": m.id, "kind": "prediction", "name": f"v{m.version}-{m.architecture}", "version": m.version, "architecture": m.architecture,
            "is_active": m.is_active, "selected": m.selected, "experimental": m.experimental, "dataset_version": m.dataset_version,
            "dataset_info": m.dataset_info, "created_at": m.created_at.isoformat(), "params": m.params, "temperature": m.temperature,
            "path": m.path, "metrics": {k: met.get(k) for k in ("n", "top1_accuracy", "top3_accuracy", "precision_at_3", "recall_at_3",
                                                                 "macro_f1", "log_loss", "brier", "ece")},
            "validation": met.get("validation"), "deployment_status": "active" if m.is_active else "available"}


def deployments(db: Session, kind: str | None = None) -> list[dict]:
    q = select(ModelDeployment).order_by(ModelDeployment.created_at.desc()).limit(50)
    if kind:
        q = q.where(ModelDeployment.kind == kind)
    return [{"id": d.id, "kind": d.kind, "model_id": d.model_id, "previous_model_id": d.previous_model_id, "reason": d.reason,
             "forced": d.forced, "validation": d.validation, "created_at": d.created_at.isoformat()} for d in db.scalars(q).all()]


__all__ = ["overview", "fighters_table", "global_analytics", "fighter_intelligence", "pattern_evaluation", "action_model_json",
           "prediction_model_json", "deployments", "cached", "OBS_FIELDS", "is_track_key"]
