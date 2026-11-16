"""Sliding-window action detection on tracked pose sequences, and sequence extraction.

For each fighter track the window slides over time (configurable length and hop). Windows
are skipped when the fighter barely moves (motion gate), when too few landmarks are visible,
or when they span a camera cut. The classifier's probabilities decide:

* a labelled action when the best non-background class reaches the confidence threshold;
* ``unknown`` ("Unclassified") for fast movement the model cannot classify confidently;
* nothing otherwise — idle or ordinary movement is never forced into a strike class.

Overlapping windows with the same label are merged into one action, overlapping actions with
different labels are de-duplicated (temporal NMS), and start/end are refined from the speed
profile of the most active limb (onset/offset at 30 % of the peak speed).
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.actions.features import FeatureConfig, TrackPose, limb_speed_profile, window_frames
from app.actions.models import make_model
from app.actions.settings import ActionSettings
from app.actions.taxonomy import BACKGROUND, UNKNOWN, group, label

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    version: str
    algorithm: str
    classes: list[str]
    cfg: FeatureConfig
    window_seconds: float
    model: object
    per_class: dict = field(default_factory=dict)


def load_model(version: str, algorithm: str, path: Path, device: str, metrics: dict | None) -> LoadedModel:
    meta = json.loads((path / "meta.json").read_text())
    fc = meta.get("feature_config", {})
    cfg = FeatureConfig(seq_len=fc.get("seq_len", 16), context=fc.get("context", 0.15), kp_threshold=fc.get("kp_threshold", 0.3))
    m = make_model(algorithm, meta["classes"], meta.get("params"), device).load(path)
    return LoadedModel(version, algorithm, meta["classes"], cfg, float(meta.get("window_seconds") or 0.8), m,
                       (metrics or {}).get("per_class", {}))


@dataclass
class Detection:
    track_id: int
    category: str
    start: float
    end: float
    confidence: float
    runner_up: str | None
    runner_up_confidence: float | None
    limb: str | None
    windows: int = 1
    low_confidence: bool = False


def _iou(a: Detection, b: Detection) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def detect_track(tp: TrackPose, lm: LoadedModel, s: ActionSettings, scene_cut_times: list[float],
                 stats: Counter) -> list[Detection]:
    W = s.window_seconds or lm.window_seconds
    hop = max(W * s.hop_fraction, 1.0 / tp.fps)
    times = tp.times
    if len(times) < 3 or times[-1] - times[0] < W * 0.5:
        stats["tracks_too_short"] += 1
        return []
    starts = np.arange(times[0], max(times[0], times[-1] - W) + 1e-6, hop)
    Xs, meta = [], []
    for st in starts:
        en = st + W
        stats["windows_total"] += 1
        if any(st < c < en for c in scene_cut_times):
            stats["windows_scene_cut"] += 1
            continue
        res = window_frames(tp, st, en, lm.cfg)
        if res is None:
            stats["windows_no_pose"] += 1
            continue
        X, info = res
        if info["visible_frac"] < s.min_visible_fraction:
            stats["windows_low_visibility"] += 1
            continue
        motion = max(info["limb_speeds"].values())
        if motion < s.motion_gate:
            stats["windows_idle"] += 1
            continue
        Xs.append(X)
        meta.append((st, en, motion, max(info["limb_speeds"], key=info["limb_speeds"].get)))
    if not Xs:
        return []
    stats["windows_classified"] += len(Xs)
    P = lm.model.predict_proba(np.stack(Xs))
    classes = lm.classes
    bg = classes.index(BACKGROUND) if BACKGROUND in classes else None
    act_idx = [i for i, c in enumerate(classes) if c != BACKGROUND]

    cands: list[Detection] = []
    for (st, en, motion, limb), p in zip(meta, P, strict=True):
        order = sorted(act_idx, key=lambda i: p[i], reverse=True)
        best = order[0]
        p_best = float(p[best])
        p_bg = float(p[bg]) if bg is not None else 0.0
        second = order[1] if len(order) > 1 else None
        if p_best >= s.confidence_threshold and (bg is None or p_best > p_bg):
            cands.append(Detection(tp.track_id, classes[best], st, en, p_best,
                                   classes[second] if second is not None else None,
                                   float(p[second]) if second is not None else None, limb))
        elif motion >= s.unknown_motion and (bg is None or p_bg < 0.5):
            cands.append(Detection(tp.track_id, UNKNOWN, st, en, p_best, classes[best], p_best, limb, low_confidence=True))
            stats["windows_unknown"] += 1
        else:
            stats["windows_rejected"] += 1

    # merge overlapping windows with the same label
    cands.sort(key=lambda d: d.start)
    merged: list[Detection] = []
    for d in cands:
        last = merged[-1] if merged else None
        if last and last.category == d.category and d.start <= last.end and d.end - last.start <= 1.5 * W:
            last.end = max(last.end, d.end)
            last.windows += 1
            if d.confidence > last.confidence:
                last.confidence, last.runner_up, last.runner_up_confidence, last.limb = d.confidence, d.runner_up, d.runner_up_confidence, d.limb
        else:
            merged.append(Detection(**d.__dict__))

    # temporal NMS across labels: higher confidence wins; 'unknown' always yields to a labelled action
    merged.sort(key=lambda d: (d.category != UNKNOWN, d.confidence), reverse=True)
    kept: list[Detection] = []
    for d in merged:
        if all(_iou(d, k) <= s.nms_iou for k in kept):
            kept.append(d)
        else:
            stats["duplicates_removed"] += 1

    # refine boundaries from the most active limb's speed profile
    for d in kept:
        if not d.limb:
            continue
        t, sp = limb_speed_profile(tp, d.start, d.end, d.limb)
        if len(t) >= 3 and sp.max() > 0:
            pk = int(sp.argmax())
            thr = 0.3 * sp[pk]
            a = pk
            while a > 0 and sp[a - 1] >= thr:
                a -= 1
            b = pk
            while b < len(sp) - 1 and sp[b + 1] >= thr:
                b += 1
            ns, ne = float(t[max(a - 1, 0)]), float(t[min(b + 1, len(t) - 1)])
            if ne - ns < 0.2:  # keep a plausible minimum duration centred on the speed peak
                c = float(t[pk])
                ns, ne = max(d.start, c - 0.1), min(d.end, c + 0.1)
            if ne - ns >= 0.1:
                d.start, d.end = ns, ne
    kept.sort(key=lambda d: d.start)
    return kept


def detect_video(vp, lm: LoadedModel, s: ActionSettings, progress: Callable[[float], None],
                 is_cancelled: Callable[[], bool]) -> tuple[list[Detection], dict, list[str]]:
    warnings: list[str] = []
    stats: Counter = Counter()
    stride = int((vp.vision_job.stats or {}).get("stride") or 1)
    pose_rate = vp.fps / stride
    if pose_rate < 15:
        warnings.append(f"Pose was sampled at {pose_rate:.1f} fps. Strikes last ~0.2-0.5 s, so many will be missed; "
                        "re-run Phase 2 with an analysis rate of 25 fps or 'every frame' for action recognition.")
    tids = list(vp.tracks)
    if s.fighters_only:
        roles = {t: vp.sessions[t].role for t in tids if t in vp.sessions}
        fighters = [t for t in tids if roles.get(t) == "fighter"]
        if not fighters:
            fighters = [t for t in tids if vp.sessions.get(t) and vp.sessions[t].suggested_role == "fighter"]
            if fighters:
                warnings.append("No tracks are confirmed as fighters yet — analysing the suggested fighter tracks. "
                                "Confirm roles on the Fighter Tracking page for reliable fighter association.")
        tids = fighters or tids
    cut_times = [c / vp.fps for c in vp.scene_cuts]
    out: list[Detection] = []
    for i, tid in enumerate(tids):
        if is_cancelled():
            from app.pipeline.base import JobCancelled

            raise JobCancelled()
        out += detect_track(vp.tracks[tid], lm, s, cut_times, stats)
        progress((i + 1) / max(len(tids), 1))
    stats["tracks_analysed"] = len(tids)
    stats["pose_rate_fps"] = round(pose_rate, 2)
    return out, dict(stats), warnings


# ----------------------------------------------------------------------------- sequences
def _sequence_record(key: str, idx: int, current: list[dict]) -> dict:
    return {
        "fighter_key": key, "fighter_id": current[0].get("fighter_id"),
        "track_ids": sorted({a["track_id"] for a in current}), "sequence_index": idx,
        "start_time": round(current[0]["start"], 3), "end_time": round(max(a["end"] for a in current), 3),
        "actions": [{"action_id": a["id"], "category": a["category"], "label": label(a["category"]),
                     "group": group(a["category"]), "start": round(a["start"], 3), "end": round(a["end"], 3),
                     "confidence": round(a["confidence"], 4), "source": a.get("source", "model")} for a in current],
        "intervals": [round(max(0.0, current[i + 1]["start"] - current[i]["end"]), 3) for i in range(len(current) - 1)],
        "n_actions": len(current),
    }


def build_sequences(actions: list[dict], gap: float) -> list[dict]:
    """actions: dicts with fighter_key, fighter_id, track_id, category, start, end, confidence, id.

    Returns one record per (fighter, burst of activity) in chronological order. Callers leave out
    unknown and rejected actions so Phase 4 receives clean sequences; corrected labels replace
    model labels. ``intervals`` are the gaps between consecutive actions (start of next - end of previous).
    """
    by_f: dict[str, list[dict]] = defaultdict(list)
    for a in actions:
        by_f[a["fighter_key"]].append(a)
    seqs = []
    for key, items in by_f.items():
        items.sort(key=lambda a: a["start"])
        current: list[dict] = []
        idx = 0
        for a in items:
            if current and a["start"] - max(x["end"] for x in current) > gap:
                seqs.append(_sequence_record(key, idx, current))
                idx += 1
                current = []
            current.append(a)
        if current:
            seqs.append(_sequence_record(key, idx, current))
    return seqs


def repeated_combinations(seqs: list[dict], n_values=(2, 3)) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for key in {s["fighter_key"] for s in seqs}:
        cnt: Counter = Counter()
        for s in seqs:
            if s["fighter_key"] != key:
                continue
            cats = [a["category"] for a in s["actions"]]
            for n in n_values:
                for i in range(len(cats) - n + 1):
                    cnt[tuple(cats[i:i + n])] += 1
        out[key] = [{"combination": [label(c) for c in k], "keys": list(k), "count": v}
                    for k, v in cnt.most_common() if v >= 2][:20]
    return out
