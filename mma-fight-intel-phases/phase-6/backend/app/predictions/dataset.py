"""Phase 5 dataset: (history, context) -> next action examples.

Leakage rules enforced here:
* the prediction moment is the END of the last observed action; every feature uses only information
  that exists at or before that moment (opponent actions that started before it, distance / pose samples
  at or before it, fighter statistics computed without the example's own fight);
* the target is the fighter's next action in the same sequence (Phase 4 ``next`` definition);
* examples carry their fight and fighter so splits can be made by complete fights / unseen fighters.
"""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.actions.taxonomy import group as action_group
from app.patterns.engine import Obs, PatternConfig, prepare
from app.patterns.extract import OBS_FIELDS

RARE = "__rare__"
PAD = "__pad__"
MOVEMENT_FIELDS = ("center_speed", "left_wrist_speed", "right_wrist_speed", "left_elbow_speed", "right_elbow_speed",
                   "left_shoulder_speed", "right_shoulder_speed", "left_hip_speed", "right_hip_speed",
                   "left_knee_speed", "right_knee_speed", "left_ankle_speed", "right_ankle_speed")


@dataclass
class Step:
    """One observed action in a history (what the models see)."""
    category: str
    start: float | None = None  # None when the caller did not provide timing
    end: float | None = None
    confidence: float = 1.0

    @property
    def duration(self) -> float | None:
        return None if self.start is None or self.end is None else max(0.0, self.end - self.start)


@dataclass
class Context:
    """Situation at the prediction moment. Every field is optional (None = unknown)."""
    opponent_action: str | None = None
    opponent_lag: float | None = None  # seconds between opponent action start and the prediction moment
    opponent_attacks_3s: int | None = None
    distance: float | None = None  # torso lengths
    closing_speed: float | None = None
    stance: str | None = None
    movement: dict[str, float | None] = field(default_factory=dict)  # MOVEMENT_FIELDS + direction_deg
    sequence_position: int | None = None
    time_since_sequence_start: float | None = None


@dataclass
class Example:
    fighter_key: str
    fight_key: str
    video_id: str
    fight_date: str | None
    history: list[Step]  # oldest -> newest, last element = action just completed
    context: Context
    target: str | None  # next action (None for prediction-only examples)
    t: float  # prediction moment (s)
    action_id: str = ""
    target_action_id: str | None = None
    source: str = "db"


# ============================================================================ video context (Phase 2)
class VideoContext:
    """Distances and movement samples of one video, queried strictly backwards in time."""

    def __init__(self, distances: list[tuple] | None = None, movement: list[dict] | None = None):
        self.pairs: dict[tuple[int, int], tuple[list[float], list[tuple]]] = defaultdict(lambda: ([], []))
        for a, b, t, d, cs in sorted(distances or [], key=lambda r: r[2]):
            ts, vals = self.pairs[(min(a, b), max(a, b))]
            ts.append(t)
            vals.append((d, cs))
        self.tracks: dict[int, tuple[list[float], list[dict]]] = defaultdict(lambda: ([], []))
        for m in sorted(movement or [], key=lambda r: r["timestamp"]):
            ts, rows = self.tracks[m["track_id"]]
            ts.append(m["timestamp"])
            rows.append(m)

    def distance(self, track: int | None, others: list[int], t: float, max_age: float = 0.5):
        if track is None:
            return None, None
        best = None
        for o in others:
            ts, vals = self.pairs.get((min(track, o), max(track, o)), ([], []))
            i = bisect_right(ts, t) - 1  # last sample at or before t
            if i >= 0 and t - ts[i] <= max_age and vals[i][0] is not None:
                if best is None or ts[i] > best[0]:
                    best = (ts[i], vals[i])
        return (best[1][0], best[1][1]) if best else (None, None)

    def movement(self, track: int | None, t: float, window: float = 0.5) -> dict[str, float | None]:
        if track is None or track not in self.tracks:
            return {}
        ts, rows = self.tracks[track]
        hi = bisect_right(ts, t)
        lo = bisect_right(ts, t - window)
        sel = [r for r in rows[lo:hi] if r.get("state") not in ("missing", "low_confidence")]
        if not sel:
            return {}
        out: dict[str, float | None] = {}
        for f in MOVEMENT_FIELDS:
            vals = [r[f] for r in sel if r.get(f) is not None]
            out[f] = float(np.mean(vals)) if vals else None
        dirs = [r["direction_deg"] for r in sel if r.get("direction_deg") is not None]
        if dirs:
            out["direction_sin"] = float(np.mean([math.sin(math.radians(d)) for d in dirs]))
            out["direction_cos"] = float(np.mean([math.cos(math.radians(d)) for d in dirs]))
        return out


# ============================================================================ building examples
def build_examples(obs: list[Obs], cfg: PatternConfig, video_ctx: dict[str, VideoContext] | None = None,
                   history_length: int = 10, include_last: bool = False, tracks_of: dict | None = None) -> list[Example]:
    """One example per action that has a successor in its sequence.

    With ``include_last`` the last action of every sequence also yields an example with ``target=None``
    (used for offline video prediction, where the future is unknown at that moment).
    """
    p = prepare([Obs(**{k: getattr(o, k) for k in OBS_FIELDS}) for o in obs], cfg)
    video_ctx = video_ctx or {}
    # streams of each fighter per video (history may span earlier sequences of the same video)
    streams: dict[tuple[str, str], list[Obs]] = defaultdict(list)
    by_video: dict[str, list[Obs]] = defaultdict(list)
    for o in p.obs:
        streams[(o.fighter_key, o.video_id)].append(o)
        by_video[o.video_id].append(o)
    for v in list(streams.values()) + list(by_video.values()):
        v.sort(key=lambda o: o.start)
    tracks_of = tracks_of or {}
    out: list[Example] = []
    for (fk, vid), items in streams.items():
        vc = video_ctx.get(vid)
        seq_start: dict[str, float] = {}
        for i, o in enumerate(items):
            seq_start.setdefault(o.seq_id, o.start)
            nxt = items[i + 1] if i + 1 < len(items) else None
            same_seq = nxt is not None and nxt.seq_id == o.seq_id
            if not same_seq and not include_last:
                continue
            t = o.end
            hist = [Step(h.category, h.start, h.end, h.confidence) for h in items[max(0, i + 1 - history_length):i + 1]]
            ctx = Context(stance=o.stance, sequence_position=o.seq_pos, time_since_sequence_start=round(t - seq_start[o.seq_id], 4))
            if o.opponent_key:
                opp = [x for x in by_video[vid] if x.fighter_key == o.opponent_key and x.start <= t]
                if opp:
                    last = opp[-1]
                    ctx.opponent_action, ctx.opponent_lag = last.category, round(t - last.start, 4)
                    ctx.opponent_attacks_3s = sum(1 for x in opp if x.group == "offensive" and t - x.start <= 3.0)
                else:
                    ctx.opponent_attacks_3s = 0
                if vc is not None:
                    others = sorted(tracks_of.get((vid, o.opponent_key), set()))
                    ctx.distance, ctx.closing_speed = vc.distance(o.track_id, others, t)
            if vc is not None:
                ctx.movement = vc.movement(o.track_id, t)
            out.append(Example(fighter_key=fk, fight_key=o.fight_key, video_id=vid, fight_date=o.fight_date,
                               history=hist, context=ctx, target=nxt.category if same_seq else None, t=t,
                               action_id=o.action_id, target_action_id=nxt.action_id if same_seq else None,
                               source=o.source))
    out.sort(key=lambda e: (e.fight_date or "", e.video_id, e.t))
    return out


def dataset_version(examples: list[Example]) -> str:
    h = hashlib.sha1()
    for e in examples:
        h.update(f"{e.fighter_key}|{e.fight_key}|{e.video_id}|{e.t:.3f}|{'>'.join(s.category for s in e.history)}|{e.target}\n".encode())
    return h.hexdigest()[:16]


def choose_classes(examples: list[Example], min_count: int) -> tuple[list[str], dict[str, int]]:
    """Target classes with at least ``min_count`` examples; rarer actions share one 'rare' class."""
    counts = Counter(e.target for e in examples if e.target)
    keep = sorted(c for c, n in counts.items() if n >= min_count)
    if any(n < min_count for n in counts.values()):
        keep.append(RARE)
    return keep, dict(counts)


def map_target(t: str, classes: list[str]) -> str:
    return t if t in classes else RARE


# ============================================================================ fighter statistics without leakage
class FighterStats:
    """Per-fighter action counts with the possibility to exclude one fight (leave-one-fight-out)."""

    def __init__(self, examples: list[Example]):
        self.total: dict[str, Counter] = defaultdict(Counter)
        self.by_fight: dict[tuple[str, str], Counter] = defaultdict(Counter)
        self.trans_total: dict[str, Counter] = defaultdict(Counter)
        self.trans_by_fight: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for e in examples:
            if e.target is None:
                continue
            self.total[e.fighter_key][e.target] += 1
            self.by_fight[(e.fighter_key, e.fight_key)][e.target] += 1
            k = (e.history[-1].category, e.target)
            self.trans_total[e.fighter_key][k] += 1
            self.trans_by_fight[(e.fighter_key, e.fight_key)][k] += 1

    def distribution(self, fighter: str, exclude_fight: str | None) -> Counter:
        c = Counter(self.total.get(fighter, Counter()))
        if exclude_fight is not None:
            c.subtract(self.by_fight.get((fighter, exclude_fight), Counter()))
        return +c

    def transitions_from(self, fighter: str, last: str, exclude_fight: str | None) -> Counter:
        c = Counter(self.trans_total.get(fighter, Counter()))
        if exclude_fight is not None:
            c.subtract(self.trans_by_fight.get((fighter, exclude_fight), Counter()))
        return Counter({nx: n for (lst, nx), n in (+c).items() if lst == last})


def action_group_of(c: str) -> str:
    return "rare" if c == RARE else action_group(c)
