"""Phase 4 orchestration: full fighter analysis, persistence of derived tables, background jobs."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session

from app.actions.taxonomy import label
from app.db.database import session_scope
from app.db.models import JobStatus
from app.db.pattern_models import (
    ActionTransition,
    FighterTendency,
    FightingCombination,
    PatternAnalysisJob,
    PatternSnapshot,
)
from app.patterns import engine as E
from app.patterns.extract import Extraction, apply_filters, extract, is_track_key
from app.patterns.settings import PatternSettings, get_pattern_settings

log = logging.getLogger(__name__)
ACTIVE = (JobStatus.queued, JobStatus.processing)
DEFINITIONS = ("next", "next_offensive", "window")
ENGINE_VERSION = "4.2"  # bump when result format changes: stored snapshots then count as stale


def _now():
    return datetime.now(timezone.utc)


class PatternConflict(Exception):
    pass


# ============================================================================ analysis
def fighter_version(ex: Extraction, settings: PatternSettings, fighter_key: str) -> str:
    """Dataset version relevant to one fighter: their videos (incl. opponents' actions there) + settings."""
    vids = {o.video_id for o in ex.obs if o.fighter_key == fighter_key}
    sub = Extraction(obs=[o for o in ex.obs if o.video_id in vids])
    return sub.version(settings)[:13] + "-" + ENGINE_VERSION.replace(".", "")


def sequence_rows(seqs: list[list[E.Obs]], videos: dict[str, dict]) -> list[dict]:
    out = []
    for s in sorted(seqs, key=lambda q: (q[0].fight_date or "", q[0].video_id, q[0].start)):
        out.append({
            "id": s[0].seq_id, "video_id": s[0].video_id, "video_title": videos.get(s[0].video_id, {}).get("title"),
            "fight_key": s[0].fight_key, "fight_date": s[0].fight_date, "start": round(s[0].start, 3), "end": round(max(o.end for o in s), 3),
            "n_actions": len(s), "categories": [o.category for o in s], "labels": [label(o.category) for o in s],
            "actions": [{"action_id": o.action_id, "category": o.category, "label": label(o.category), "group": o.group,
                         "start": round(o.start, 3), "end": round(o.end, 3), "duration": round(o.duration, 3),
                         "interval_prev": o.interval_prev, "confidence": round(o.confidence, 4), "source": o.source,
                         "opponent_action": o.opponent_action, "opponent_action_label": label(o.opponent_action) if o.opponent_action else None,
                         "opponent_action_lag": o.opponent_action_lag,
                         "distance": o.distance, "stance": o.stance, "track_id": o.track_id} for o in s],
        })
    return out


def analyze_fighter(obs: list[E.Obs], fighter_key: str, settings: PatternSettings, ex: Extraction,
                    progress=lambda stage, frac: None) -> dict:
    cfg = settings.engine_config()
    progress("Preparing sequences", 0.05)
    p = E.prepare(obs, cfg)
    mine = p.by_fighter.get(fighter_key, [])
    seqs = E.fighter_sequences(p, fighter_key)
    fights = sorted({o.fight_key for o in mine})
    videos = sorted({o.video_id for o in mine})
    dates = sorted({o.fight_date for o in mine if o.fight_date})
    progress("Combinations", 0.2)
    combos = E.combinations(seqs, cfg, len(fights))
    patterns = E.prefixspan(seqs, cfg)
    progress("Conditional probabilities", 0.4)
    transitions = {d: {str(k): E.conditional_probabilities(mine, seqs, d, k, cfg) for k in range(1, cfg.max_order + 1)}
                   for d in DEFINITIONS}
    progress("Markov models", 0.6)
    states = sorted({o.category for o in mine})
    markov = {str(k): E.markov(seqs, cfg, k, states) for k in range(1, cfg.max_order + 1)}
    progress("Tendencies", 0.75)
    tend = E.tendencies(mine, seqs, cfg, ex.footage, p.obs)
    progress("Opponent analysis", 0.85)
    opp = E.opponent_analysis(p, fighter_key, cfg, ex.footage, ex.names, settings.min_actions_per_opponent)
    all_probs = [r for d in transitions.values() for rows in d.values() for r in rows]
    rel = Counter(r["reliability"] for r in transitions["next"]["1"])
    src = Counter(o.source for o in mine)
    warnings = []
    if not mine:
        warnings.append("No actions are attributed to this fighter with the current settings and filters.")
    elif len(fights) <= 1:
        warnings.append("All data comes from a single fight: patterns describe that fight only and are not generalised.")
    if mine and len(mine) < 30:
        warnings.append(f"Only {len(mine)} actions — most probabilities are below the reliability thresholds.")
    if src.get("detection"):
        warnings.append(f"{src['detection']} of {len(mine)} actions are unreviewed model detections; "
                        "their accuracy is bounded by the Phase 3 model's validated precision.")
    return {
        "fighter_key": fighter_key, "name": ex.names.get(fighter_key, fighter_key), "is_track": is_track_key(fighter_key),
        "summary": {
            "actions": len(mine), "sequences": len(seqs), "fights": len(fights), "videos": len(videos),
            "fight_keys": fights, "period": {"from": dates[0], "to": dates[-1]} if dates else None,
            "minutes_observed": tend["minutes_observed"], "sources": dict(src),
            "multi_action_sequences": sum(1 for s in seqs if len(s) >= 2),
            "mean_sequence_length": round(sum(len(s) for s in seqs) / len(seqs), 3) if seqs else None,
            "preparation": p.stats,
            "reliability_counts_first_order": dict(rel),
            "probabilities_meeting_thresholds": sum(1 for r in all_probs if r["reliability"] in ("high", "medium", "low")),
            "probabilities_total": len(all_probs),
            "single_fight_patterns": sum(1 for c in combos if c["distinct_fights"] == 1),
            "cross_fight_patterns": sum(1 for c in combos if c["distinct_fights"] >= 2),
        },
        "warnings": warnings,
        "config": settings.model_dump(),
        "combinations": combos,
        "sequential_patterns": patterns,
        "transitions": transitions,
        "markov": markov,
        "tendencies": tend,
        "opponent_analysis": opp,
        "sequences": sequence_rows(seqs, ex.videos),
        "states": states,
        "state_labels": {k: label(k) for k in states} | {E.END: "End of sequence", E.NONE: "No action in window", E.START: "Sequence start"},
    }


def persist_snapshot(db: Session, fighter_key: str, job_id: str, version: str, settings: PatternSettings, result: dict) -> PatternSnapshot:
    snap = PatternSnapshot(fighter_key=fighter_key, job_id=job_id, dataset_version=version, config=settings.model_dump(), result=result)
    db.add(snap)
    db.flush()
    # derived tables hold the latest snapshot only (older snapshots keep the full JSON)
    for model in (ActionTransition, FightingCombination, FighterTendency):
        db.execute(delete(model).where(model.fighter_key == fighter_key))
    trows = []
    for d, by_order in result["transitions"].items():
        for order, rows in by_order.items():
            for r in rows:
                ci = r["ci95"] or (None, None)
                trows.append({"fighter_key": fighter_key, "snapshot_id": snap.id, "definition": d, "order": int(order),
                              "previous_action": ">".join(r["context"]), "next_action": r["next"], "count": r["count"],
                              "context_count": r["context_count"], "probability": r["probability"], "ci_low": ci[0],
                              "ci_high": ci[1], "fights": r["fights"], "reliability": r["reliability"],
                              "observation_window": r["window_seconds"]})
    if trows:
        db.execute(insert(ActionTransition), trows)
    fights = max(1, result["summary"]["fights"])
    crows = [{"fighter_key": fighter_key, "snapshot_id": snap.id, "sequence": c["sequence"], "sequence_key": ">".join(c["sequence"]),
              "length": c["length"], "occurrences": c["occurrences"], "distinct_fights": c["distinct_fights"],
              "support": round(c["occurrences"] / fights, 4), "confidence": c["confidence"], "avg_interval": c["avg_interval"],
              "classification": c["classification"]} for c in result["combinations"]]
    if crows:
        db.execute(insert(FightingCombination), crows)
    period = result["summary"]["period"] or {}
    tend = result["tendencies"]
    drows = []
    for cat, items in (("offensive", tend["offensive"]["most_used"]), ("defensive", tend["defensive"]["most_used"])):
        for t in items:
            drows.append({"fighter_key": fighter_key, "snapshot_id": snap.id, "category": cat, "action": t["action"],
                          "count": t["count"], "frequency": t["share"], "per_minute": t["per_minute"], "fights": t["fights"],
                          "period_start": period.get("from"), "period_end": period.get("to")})
    if drows:
        db.execute(insert(FighterTendency), drows)
    return snap


def latest_snapshot(db: Session, fighter_key: str) -> PatternSnapshot | None:
    return db.scalar(select(PatternSnapshot).where(PatternSnapshot.fighter_key == fighter_key)
                     .order_by(PatternSnapshot.created_at.desc()).limit(1))


def fighter_index(ex: Extraction) -> list[dict]:
    by = {}
    for o in ex.obs:
        d = by.setdefault(o.fighter_key, {"fighter_key": o.fighter_key, "name": ex.names.get(o.fighter_key, o.fighter_key),
                                           "is_track": is_track_key(o.fighter_key), "actions": 0, "fights": set(), "videos": set(),
                                           "dates": set()})
        d["actions"] += 1
        d["fights"].add(o.fight_key)
        d["videos"].add(o.video_id)
        if o.fight_date:
            d["dates"].add(o.fight_date)
    out = []
    for d in by.values():
        out.append({**d, "fights": len(d["fights"]), "videos": len(d["videos"]),
                    "fight_keys": sorted(d["fights"]), "dates": sorted(d["dates"])})
    out.sort(key=lambda d: (d["is_track"], -d["actions"]))
    return out


# ============================================================================ background jobs
class PatternJobManager:
    """One worker thread; analysis jobs are independent of the video pipeline."""

    def __init__(self):
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        with session_scope() as db:
            db.execute(update(PatternAnalysisJob).where(PatternAnalysisJob.status == JobStatus.processing).values(
                status=JobStatus.failed, finished_at=_now(),
                error_message="Interrupted: the application stopped while this job was running. Retry to run again."))
            queued = db.scalars(select(PatternAnalysisJob.id).where(PatternAnalysisJob.status == JobStatus.queued)
                                .order_by(PatternAnalysisJob.created_at)).all()
        for j in queued:
            self._queue.put(j)
        self._thread = threading.Thread(target=self._loop, name="pattern-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        if not self._running:
            return
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=timeout)

    def create(self, db: Session, fighter_key: str, force: bool = False) -> PatternAnalysisJob:
        existing = db.scalar(select(PatternAnalysisJob).where(PatternAnalysisJob.fighter_key == fighter_key,
                                                              PatternAnalysisJob.status.in_(ACTIVE)))
        if existing:
            return existing
        job = PatternAnalysisJob(fighter_key=fighter_key, config={**get_pattern_settings(db).model_dump(), "force": force})
        db.add(job)
        db.commit()
        db.refresh(job)
        self._queue.put(job.id)
        return job

    def _loop(self):
        while True:
            jid = self._queue.get()
            if jid is None:
                break
            try:
                self._run(jid)
            except Exception as exc:  # pragma: no cover
                log.exception("Pattern job %s crashed", jid)
                with session_scope() as db:
                    j = db.get(PatternAnalysisJob, jid)
                    if j:
                        j.status, j.error_message, j.finished_at = JobStatus.failed, str(exc)[:2000], _now()

    def _progress(self, job_id: str, base: float, span: float):
        last = [0.0]

        def write(stage: str, frac: float):
            now = time.monotonic()
            if now - last[0] < 0.5 and frac < 0.99:
                return
            last[0] = now
            with session_scope() as db:
                j = db.get(PatternAnalysisJob, job_id)
                if j:
                    j.progress, j.current_stage = round(min(0.999, base + span * frac), 4), stage

        return write

    def _run(self, job_id: str):
        with session_scope() as db:
            job = db.get(PatternAnalysisJob, job_id)
            if job is None or job.status != JobStatus.queued:
                return
            job.status, job.started_at, job.current_stage, job.progress = JobStatus.processing, _now(), "Extracting actions", 0.02
            fighter_key, force = job.fighter_key, bool((job.config or {}).get("force"))
        t0 = time.monotonic()
        with session_scope() as db:
            settings = PatternSettings.model_validate({k: v for k, v in (db.get(PatternAnalysisJob, job_id).config or {}).items()
                                                       if k in PatternSettings.model_fields})
            ex = extract(db, settings)
        keys = [d["fighter_key"] for d in fighter_index(ex)] if fighter_key == "*" else [fighter_key]
        done, skipped, analysed = [], [], []
        for i, key in enumerate(keys):
            version = fighter_version(ex, settings, key)
            with session_scope() as db:
                snap = latest_snapshot(db, key)
            if snap and snap.dataset_version == version and not force:
                skipped.append(key)
                continue
            obs = apply_filters(ex)
            result = analyze_fighter(obs, key, settings, ex, self._progress(job_id, 0.05 + 0.9 * i / max(1, len(keys)), 0.9 / max(1, len(keys))))
            with session_scope() as db:
                persist_snapshot(db, key, job_id, version, settings, result)
            analysed.append(key)
            done.append({"fighter_key": key, "actions": result["summary"]["actions"], "fights": result["summary"]["fights"]})
        with session_scope() as db:
            job = db.get(PatternAnalysisJob, job_id)
            job.status, job.progress, job.current_stage, job.finished_at = JobStatus.completed, 1.0, "Done", _now()
            job.dataset_version = fighter_version(ex, settings, fighter_key) if fighter_key != "*" else ex.version(settings)[:16]
            job.stats = {"analysed": done, "unchanged_skipped": skipped, "seconds": round(time.monotonic() - t0, 3),
                         "video_cache_hits": ex.cache_hits, "video_cache_misses": ex.cache_misses, "extraction": ex.stats}
            if not keys or (fighter_key != "*" and not any(o.fighter_key == fighter_key for o in ex.obs)):
                job.warnings = ["No actions found for this fighter with the current settings."]
