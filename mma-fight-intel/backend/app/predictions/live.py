"""Live prediction architecture (prepared, not real-time tested).

Pipeline per incoming frame:  FrameSource -> detector/tracker (Phase 2) -> pose -> action recogniser (Phase 3,
sliding window) -> completed actions -> ``LivePredictionSession.push_action`` -> next-move prediction.

What is implemented and tested today:
* ``LivePredictionSession`` — the action-level part: keeps each fighter's recent action sequence, computes
  the context at the prediction moment and returns a prediction whenever an action completes. It is
  independent of where actions come from.
* ``ReplaySource`` — feeds a session with the actions already recognised in an uploaded video, in time order,
  exactly as a live recogniser would emit them (used by tests; the offline video prediction job applies the
  same rule: only information before the prediction moment is used).

What is NOT implemented: a webcam / stream ``FrameSource`` and incremental (per-frame) versions of the Phase 2
and Phase 3 stages. Those stages currently process whole videos. ``FrameSource`` and ``FrameStage`` define the
interfaces a future live stage has to provide; nothing here claims real-time performance.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.predictions.dataset import Context, Example, Step


class FrameSource(Protocol):
    """A source of (timestamp, BGR frame). A webcam / RTSP implementation would wrap cv2.VideoCapture."""

    def frames(self) -> Iterator[tuple[float, np.ndarray]]: ...


class FrameStage(Protocol):
    """Incremental stage: consumes a frame, returns completed actions (fighter_key, category, start, end, confidence)."""

    def process(self, t: float, frame: np.ndarray) -> list[tuple[str, str, float, float, float]]: ...


@dataclass
class CompletedAction:
    fighter_key: str
    category: str
    start: float
    end: float
    confidence: float = 1.0
    opponent_key: str | None = None
    distance: float | None = None


class LivePredictionSession:
    """Keeps recent sequences and asks ``predict_fn`` for the next move each time an action completes."""

    def __init__(self, predict_fn: Callable[[list[Example]], list[dict]], history_length: int = 10, sequence_gap: float = 2.0):
        self.predict_fn = predict_fn
        self.history_length = history_length
        self.sequence_gap = sequence_gap
        self.history: dict[str, deque[Step]] = defaultdict(lambda: deque(maxlen=history_length))
        self.seq_start: dict[str, float] = {}
        self.seq_pos: dict[str, int] = defaultdict(int)
        self.latest: dict[str, dict] = {}

    def push_action(self, a: CompletedAction) -> dict:
        h = self.history[a.fighter_key]
        if h and a.start - h[-1].end > self.sequence_gap:
            self.seq_pos[a.fighter_key] = 0
            self.seq_start[a.fighter_key] = a.start
        self.seq_start.setdefault(a.fighter_key, a.start)
        h.append(Step(a.category, a.start, a.end, a.confidence))
        opp = self.history.get(a.opponent_key) if a.opponent_key else None
        ctx = Context(distance=a.distance, sequence_position=self.seq_pos[a.fighter_key],
                      time_since_sequence_start=a.end - self.seq_start[a.fighter_key])
        if opp:
            last = [s for s in opp if s.start <= a.end]
            if last:
                ctx.opponent_action, ctx.opponent_lag = last[-1].category, a.end - last[-1].start
        self.seq_pos[a.fighter_key] += 1
        e = Example(a.fighter_key, "__live__", "__live__", None, list(h), ctx, None, a.end)
        r = self.predict_fn([e])[0]
        r["at_time"] = a.end
        self.latest[a.fighter_key] = r
        return r


class ReplaySource:
    """Replays recognised actions of an uploaded video in time order, as a live recogniser would emit them."""

    def __init__(self, actions: Iterable[CompletedAction]):
        self.actions = sorted(actions, key=lambda a: a.end)

    def __iter__(self) -> Iterator[CompletedAction]:
        return iter(self.actions)
