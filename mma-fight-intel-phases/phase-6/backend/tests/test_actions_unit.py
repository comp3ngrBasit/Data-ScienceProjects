"""Phase 3 unit tests: features, sequences, metrics helpers, importer, taxonomy (no model files needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.actions.features import FEATURE_DIM, FeatureConfig, TrackPose, summarise, window_frames
from app.actions.importer import parse_rows
from app.actions.inference import build_sequences, repeated_combinations
from app.actions.taxonomy import BY_KEY, normalise_key
from app.actions.training import metrics_from, wilson
from tests.test_movement import standing_pose


def pose_track(n=30, fps=25.0, conf=0.9, mover=None):
    k = np.stack([standing_pose() if mover is None else mover(i) for i in range(n)]).astype(float)
    return TrackPose(1, np.arange(n), k, np.full((n, 17), conf), fps)


def test_taxonomy_covers_spec_and_unknown():
    for k in ["left_jab", "right_jab", "left_hook", "right_hook", "left_cross", "right_cross", "uppercut", "front_kick",
              "low_kick", "middle_kick", "high_kick", "roundhouse_kick", "knee_strike", "elbow_strike", "spinning_back_kick",
              "takedown_attempt", "high_guard", "defensive_block", "parry", "slip", "duck", "backward_evasion", "sprawl",
              "takedown_defense", "counterattack", "forward_movement", "backward_movement", "lateral_movement",
              "stance_switch", "feint", "reset_stance", "clinch_entry", "unknown"]:
        assert k in BY_KEY
    assert normalise_key("Left Jab") == "left_jab" and normalise_key("left-jab") == "left_jab"
    assert normalise_key("nonsense") is None


def test_window_features_shape_and_invariance():
    X, info = window_frames(pose_track(), 0.1, 0.9, FeatureConfig())
    assert X.shape == (16, FEATURE_DIM) and info["visible_frac"] == 1.0
    assert summarise(X).shape == (FEATURE_DIM * 6,)
    # same pose scaled 2x and shifted: normalised features identical
    tp = pose_track()
    tp2 = TrackPose(1, tp.frames, tp.keypoints * 2 + 100, tp.conf, tp.fps)
    X2, _ = window_frames(tp2, 0.1, 0.9, FeatureConfig())
    assert np.allclose(X, X2, atol=1e-4)


def test_low_confidence_window_is_rejected_not_zero_filled():
    assert window_frames(pose_track(conf=0.1), 0.1, 0.9, FeatureConfig()) is None  # no visible torso -> no sample
    tp = pose_track()
    tp.conf[:, :5] = 0.1  # face hidden: still usable, visibility reflects it
    _, info = window_frames(tp, 0.1, 0.9, FeatureConfig())
    assert 0.9 < info["visible_frac"] < 1.0


def test_moving_wrist_raises_limb_speed():
    from app.vision.constants import KP

    def mover(i):
        k = standing_pose()
        k[KP["right_wrist"], 0] += 20 * i
        return k

    _, info = window_frames(pose_track(mover=mover), 0.1, 0.9, FeatureConfig())
    assert info["limb_speeds"]["right_arm"] > 2.0 and info["limb_speeds"]["left_arm"] < 0.1


def test_sequences_and_intervals():
    acts = [
        {"id": "a", "fighter_key": "f1", "fighter_id": "f1", "track_id": 1, "category": "left_jab", "start": 1.0, "end": 1.3, "confidence": 0.9},
        {"id": "b", "fighter_key": "f1", "fighter_id": "f1", "track_id": 1, "category": "right_cross", "start": 1.5, "end": 1.8, "confidence": 0.8},
        {"id": "c", "fighter_key": "f1", "fighter_id": "f1", "track_id": 3, "category": "left_jab", "start": 6.0, "end": 6.3, "confidence": 0.9},
        {"id": "d", "fighter_key": "f1", "fighter_id": "f1", "track_id": 3, "category": "right_cross", "start": 6.4, "end": 6.7, "confidence": 0.7},
        {"id": "e", "fighter_key": "f2", "fighter_id": "f2", "track_id": 2, "category": "defensive_block", "start": 1.4, "end": 1.6, "confidence": 0.7},
    ]
    seqs = build_sequences(acts, gap=2.0)
    f1 = [s for s in seqs if s["fighter_key"] == "f1"]
    assert len(f1) == 2 and [a["category"] for a in f1[0]["actions"]] == ["left_jab", "right_cross"]
    assert f1[0]["intervals"] == [0.2] and f1[1]["track_ids"] == [3]
    combos = repeated_combinations(seqs)["f1"]
    assert combos[0]["keys"] == ["left_jab", "right_cross"] and combos[0]["count"] == 2


def test_metrics_and_wilson():
    m = metrics_from(["a", "a", "b", "background"], ["a", "b", "b", "background"], np.array([0.9, 0.4, 0.8, 0.7]),
                     ["a", "b", "background"], 0.6)
    assert m["accuracy"] == 0.75 and m["confusion_matrix"]["matrix"][0] == [1, 1, 0]
    assert m["per_class"]["a"]["precision_at_threshold"] == 1.0
    lo, hi = wilson(7, 10)
    assert 0.39 < lo < 0.40 and 0.89 < hi < 0.90


def test_importer_frames_and_boxingvi_mapping():
    df = pd.DataFrame({"start_frame": [30, 60, 90], "end_frame": [44, 74, 104], "punch_class": ["Jab", "Rear Hook", "kick?"]})
    rows, errors = parse_rows(df, "frames", 30.0, "boxingvi_orthodox", None)
    assert [r.category for r in rows] == ["left_jab", "right_hook"]
    assert rows[0].start == 1.0 and abs(rows[0].end - 1.5) < 1e-6
    assert len(errors) == 1 and "kick?" in errors[0]
    rows, _ = parse_rows(df.iloc[:2], "frames", 30.0, "boxingvi_southpaw", None)
    assert [r.category for r in rows] == ["right_jab", "left_hook"]
