"""Feature engineering for next-action prediction (tabular + sequence views of the same example)."""
from __future__ import annotations

import math

import numpy as np

from app.actions.taxonomy import group as action_group
from app.predictions.dataset import MOVEMENT_FIELDS, PAD, Example, FighterStats

UNK = "__unk__"
GROUPS = ("offensive", "defensive", "movement", "unknown")
STANCES = ("Orthodox", "Southpaw")
TAB_POSITIONS = 5  # last N actions encoded position-by-position for the tabular model


def _f(v) -> float:
    return float("nan") if v is None else float(v)


class FeatureBuilder:
    """Fitted on training examples only: vocabulary, class list and normalisation statistics."""

    def __init__(self, classes: list[str], vocab: list[str], stats: FighterStats | None, prior_alpha: float = 1.0):
        self.classes = classes
        self.vocab = [PAD, UNK] + [v for v in vocab if v not in (PAD, UNK)]
        self.idx = {v: i for i, v in enumerate(self.vocab)}
        self.stats = stats
        self.alpha = prior_alpha
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    # ------------------------------------------------------------------ shared pieces
    def vid(self, c: str) -> int:
        return self.idx.get(c, 1)

    def context_numeric(self, e: Example) -> list[float]:
        h, c = e.history, e.context
        intervals = [h[i].start - h[i - 1].end for i in range(1, len(h)) if h[i].start is not None and h[i - 1].end is not None]
        last_int = intervals[-1] if intervals else None
        mean_int = float(np.mean(intervals)) if intervals else None
        rhythm = (last_int - float(np.mean(intervals[:-1]))) if len(intervals) >= 2 else None
        mv = c.movement or {}
        vals = [len(h), h[-1].duration, h[-1].confidence, last_int, mean_int, rhythm,
                c.time_since_sequence_start, c.sequence_position, c.opponent_lag, c.opponent_attacks_3s,
                c.distance, c.closing_speed, 1.0 if c.opponent_action else (0.0 if c.opponent_attacks_3s is not None else None)]
        vals += [1.0 if c.stance == s else 0.0 for s in STANCES] if c.stance else [None, None]
        vals += [mv.get(f) for f in MOVEMENT_FIELDS] + [mv.get("direction_sin"), mv.get("direction_cos")]
        return [_f(v) for v in vals]

    def context_names(self) -> list[str]:
        return (["history_length", "last_duration", "last_confidence", "last_interval", "mean_interval", "rhythm_change",
                 "time_since_sequence_start", "sequence_position", "opponent_lag", "opponent_attacks_3s", "distance",
                 "closing_speed", "opponent_active"] + [f"stance_{s.lower()}" for s in STANCES]
                + [f"move_{f}" for f in MOVEMENT_FIELDS] + ["move_direction_sin", "move_direction_cos"])

    def fighter_numeric(self, e: Example, exclude_fight: str | None) -> list[float]:
        C = len(self.classes)
        if self.stats is None:
            return [float("nan")] * (2 * C + 2)
        dist = self.stats.distribution(e.fighter_key, exclude_fight)
        tr = self.stats.transitions_from(e.fighter_key, e.history[-1].category, exclude_fight)
        n, nt = sum(dist.values()), sum(tr.values())

        def share(cnt, tot, c):
            k = cnt.get(c, 0) if c != "__rare__" else sum(v for x, v in cnt.items() if x not in self.classes)
            return (k + self.alpha) / (tot + self.alpha * C)

        return ([share(dist, n, c) for c in self.classes] + [share(tr, nt, c) for c in self.classes]
                + [math.log1p(n), math.log1p(nt)])

    def fighter_names(self) -> list[str]:
        return ([f"fighter_prior_{c}" for c in self.classes] + [f"fighter_after_last_{c}" for c in self.classes]
                + ["fighter_log_examples", "fighter_log_transitions_from_last"])

    # ------------------------------------------------------------------ tabular (XGBoost)
    def tabular(self, e: Example, exclude_fight: str | None) -> np.ndarray:
        V = len(self.vocab)
        pos = np.zeros(TAB_POSITIONS * (V + len(GROUPS)), dtype=np.float32)
        for k in range(TAB_POSITIONS):
            if k < len(e.history):
                s = e.history[-1 - k]
                base = k * (V + len(GROUPS))
                pos[base + self.vid(s.category)] = 1.0
                g = action_group(s.category)
                pos[base + V + (GROUPS.index(g) if g in GROUPS else 3)] = 1.0
        bag = np.zeros(V, dtype=np.float32)
        for s in e.history:
            bag[self.vid(s.category)] += 1.0 / len(e.history)
        opp = np.zeros(V, dtype=np.float32)
        if e.context.opponent_action:
            opp[self.vid(e.context.opponent_action)] = 1.0
        return np.concatenate([pos, bag, opp, np.array(self.context_numeric(e), dtype=np.float32),
                               np.array(self.fighter_numeric(e, exclude_fight), dtype=np.float32)])

    def tabular_names(self) -> list[str]:
        names = []
        for k in range(TAB_POSITIONS):
            names += [f"prev{k + 1}_{v}" for v in self.vocab] + [f"prev{k + 1}_group_{g}" for g in GROUPS]
        names += [f"history_bag_{v}" for v in self.vocab] + [f"opponent_last_{v}" for v in self.vocab]
        return names + self.context_names() + self.fighter_names()

    # ------------------------------------------------------------------ sequence (GRU / LSTM / Transformer)
    def static_raw(self, e: Example, exclude_fight: str | None) -> np.ndarray:
        opp = np.zeros(len(self.vocab), dtype=np.float32)
        if e.context.opponent_action:
            opp[self.vid(e.context.opponent_action)] = 1.0
        return np.concatenate([np.array(self.context_numeric(e) + self.fighter_numeric(e, exclude_fight), dtype=np.float32), opp])

    def fit_normaliser(self, raws: np.ndarray):
        import warnings

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nan_to_num(np.nanmean(raws, axis=0))
            self.std = np.nan_to_num(np.nanstd(raws, axis=0))
        self.std[self.std < 1e-6] = 1.0

    def static(self, raw: np.ndarray) -> np.ndarray:
        missing = np.isnan(raw).astype(np.float32)
        z = np.nan_to_num((raw - self.mean) / self.std)
        return np.concatenate([np.clip(z, -6, 6), missing]).astype(np.float32)

    def steps(self, e: Example, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(category ids, per-step numeric, mask) right-padded to ``length``; mask=1 for real steps."""
        h = e.history[-length:]
        ids = np.zeros(length, dtype=np.int64)
        num = np.zeros((length, 4), dtype=np.float32)
        mask = np.zeros(length, dtype=np.float32)
        off = 0
        for i, s in enumerate(h):
            timed = s.start is not None and s.end is not None and (i == 0 or h[i - 1].end is not None)
            gap = (s.start - h[i - 1].end if i else 0.0) if timed else 0.0
            ids[off + i] = self.vid(s.category)
            num[off + i] = [min(s.duration, 3.0) if timed else 0.0, max(-1.0, min(gap, 5.0)), s.confidence,
                            (e.t - s.end) / 5.0 if timed and e.t is not None else 0.0]
            mask[off + i] = 1.0
        return ids, num, mask

    def state(self) -> dict:
        return {"classes": self.classes, "vocab": self.vocab, "alpha": self.alpha,
                "mean": None if self.mean is None else self.mean.tolist(), "std": None if self.std is None else self.std.tolist()}

    @classmethod
    def from_state(cls, st: dict, stats: FighterStats | None) -> FeatureBuilder:
        fb = cls(st["classes"], st["vocab"], stats, st["alpha"])
        fb.vocab, fb.idx = st["vocab"], {v: i for i, v in enumerate(st["vocab"])}
        fb.mean = None if st["mean"] is None else np.array(st["mean"], dtype=np.float32)
        fb.std = None if st["std"] is None else np.array(st["std"], dtype=np.float32)
        return fb
