"""Phase 5 tests.

The constructed sequences below are TEST FIXTURES with known structure (so the expected behaviour is known);
they verify mechanics — leakage rules, metrics, calibration, model training, API — and are never reported as
model performance. Real-data results are in the README.
"""
from __future__ import annotations

import io
import math
import random
import time

import numpy as np
import pytest

from app.patterns.engine import Obs, PatternConfig
from app.predictions.dataset import Context, Example, FighterStats, Step, VideoContext, build_examples, choose_classes
from app.predictions.features import FeatureBuilder
from app.predictions.metrics import ece, evaluate, paired_bootstrap
from app.predictions.models import MarkovPredictor, apply_temperature, fit_temperature, make_predictor
from app.predictions.training import train_and_evaluate, vocabulary


def O(f, v, c, s, d=0.3, fight=None, opp=None, track=None):
    return Obs(f, v, fight or v, c, s, s + d, 0.9, action_id=f"{v}-{s}-{c}", opponent_key=opp, track_id=track)


def pattern_data(fights=8, seed=0):
    rng = random.Random(seed)
    pats = {"A": [["left_jab", "right_cross", "left_hook"], ["left_jab", "right_cross"]],
            "B": [["low_kick", "left_jab", "right_cross"], ["low_kick", "slip"]]}
    obs = []
    for f, ps in pats.items():
        for k in range(fights):
            t = 0.0
            for _ in range(12):
                for c in rng.choice(ps):
                    obs.append(O(f, f"{f}{k}", c, t))
                    t += 0.5
                t += 3.0
    return obs


# ============================================================================ dataset & leakage
def test_examples_use_only_past_information():
    obs = [O("A", "v", "left_jab", 0.0, opp="B", track=1), O("A", "v", "right_cross", 0.5, opp="B", track=1),
           O("B", "v", "low_kick", 0.6, opp="A", track=2),  # starts AFTER A's jab ends (0.3) -> not visible at t=0.3
           O("A", "v", "left_hook", 1.0, opp="B", track=1)]
    vc = VideoContext([(1, 2, 0.2, 2.0, 0.0), (1, 2, 0.3, 2.5, 0.1), (1, 2, 0.35, 9.9, 0.0)])  # 0.35 is in the future at t=0.3
    ex = build_examples(obs, PatternConfig(), {"v": vc}, 10, tracks_of={("v", "B"): {2}, ("v", "A"): {1}})
    first = next(e for e in ex if e.fighter_key == "A" and e.t == pytest.approx(0.3))
    assert [s.category for s in first.history] == ["left_jab"] and first.target == "right_cross"
    assert first.context.opponent_action is None and first.context.opponent_attacks_3s == 0
    assert first.context.distance == 2.5  # last sample at or before t, never the future 9.9
    second = next(e for e in ex if e.fighter_key == "A" and e.t == pytest.approx(0.8))
    assert second.context.opponent_action == "low_kick" and second.context.opponent_lag == pytest.approx(0.2)
    assert [s.category for s in second.history] == ["left_jab", "right_cross"] and second.target == "left_hook"
    # the last action of a sequence has no target unless include_last
    assert all(e.target for e in ex)
    assert any(e.target is None for e in build_examples(obs, PatternConfig(), None, 10, include_last=True))


def test_fighter_stats_exclude_own_fight():
    ex = build_examples(pattern_data(fights=2), PatternConfig(), None, 10)
    st = FighterStats(ex)
    full = st.distribution("A", None)
    without = st.distribution("A", "A0")
    own = sum(1 for e in ex if e.fighter_key == "A" and e.fight_key == "A0")
    assert sum(full.values()) - sum(without.values()) == own > 0


def test_rare_classes_grouped():
    ex = [Example("A", "f", "v", None, [Step("left_jab", 0, 0.3)], Context(), t, 0.3) for t in ["right_cross"] * 6 + ["spinning_back_fist"]]
    classes, counts = choose_classes(ex, 5)
    assert classes == ["right_cross", "__rare__"] and counts["spinning_back_fist"] == 1


# ============================================================================ metrics & calibration
def test_metrics_hand_computed():
    P = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    y = np.array([0, 1, 1, 0])
    m = evaluate(P, y, ["a", "b", "c"])
    assert m["top1_accuracy"] == 0.5 and m["top3_accuracy"] == 1.0
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.7, 0.3, 0.5, 0.1])), abs=1e-4)
    assert m["brier"] == pytest.approx(np.mean([0.09 + 0.04 + 0.01, 0.36 + 0.49 + 0.01, 0.04 + 0.25 + 0.09, 0.81 + 0.01 + 0.64]), abs=1e-4)
    e, bins = ece(P, y, bins=10)
    # bins: 0.7 (correct), 0.6 (wrong), 0.5 (correct), 0.8 (wrong) -> |1-.7|+|0-.6|+|1-.5|+|0-.8| over 4
    assert e == pytest.approx((0.3 + 0.6 + 0.5 + 0.8) / 4, abs=1e-4) and len(bins) == 4
    assert m["confusion_matrix"]["matrix"] == [[1, 0, 1], [1, 1, 0], [0, 0, 0]]


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    P = np.full((400, 3), 0.02)
    hit = rng.random(400) < 0.6  # right 60 % of the time but always 96 % confident
    pred = np.where(hit, y, (y + 1) % 3)
    P[np.arange(400), pred] = 0.96
    T = fit_temperature(P, y)
    assert T > 1.5
    before = evaluate(P, y, ["a", "b", "c"])
    after = evaluate(apply_temperature(P, T), y, ["a", "b", "c"])
    assert after["log_loss"] < before["log_loss"] and after["ece"] < before["ece"]
    assert after["top1_accuracy"] == before["top1_accuracy"]  # calibration never changes the ranking


def test_paired_bootstrap_by_fight():
    a = np.array([1, 1, 1, 1, 0, 1, 1, 1.0])
    b = np.array([0, 0, 1, 0, 0, 0, 1, 0.0])
    r = paired_bootstrap(a, b, np.array(list("aabbccdd")))
    assert r["difference"] == pytest.approx(0.625) and r["resampled_fights"] == 4


# ============================================================================ models
def test_markov_hand_computed():
    ex = [Example("A", "f1", "v", None, [Step("left_jab", 0, 0.3)], Context(), t, 0.3) for t in ["right_cross"] * 3 + ["left_hook"]]
    fb = FeatureBuilder(["left_hook", "right_cross"], vocabulary(), FighterStats(ex))
    m = MarkovPredictor({"max_order": 1, "min_count": 1, "alpha": 0.5}).fit(ex, fb)
    # global unigram (3+.5)/(4+1)=0.7 ; order-1 WB: lam=4/(4+2) -> 2/3*0.75 + 1/3*0.7 = 0.7333
    # fighter order 0 and order 1 counts are identical -> two more WB steps with the same ML 0.75
    p = 0.7
    for _ in range(3):
        p = 2 / 3 * 0.75 + 1 / 3 * p
    P = m.predict_proba([Example("A", "x", "v", None, [Step("left_jab", 0, 0.3)], Context(), None, 0.3)])
    assert P[0, 1] == pytest.approx(p, abs=1e-4) and P.sum() == pytest.approx(1.0)


def test_xgboost_uses_context_markov_cannot():
    """Next action depends only on distance: close -> uppercut, far -> left jab (Markov sees identical histories)."""
    rng = random.Random(1)
    ex = []
    for fight in range(10):
        for i in range(40):
            d = rng.uniform(0.5, 1.2) if i % 2 else rng.uniform(2.5, 3.5)
            ex.append(Example("A", f"f{fight}", f"v{fight}", None, [Step("left_jab", i, i + 0.3)], Context(distance=d),
                              "uppercut" if d < 2 else "left_jab", i + 0.3))
    r = train_and_evaluate(ex, {"folds": 5, "architectures": []}, "cpu", lambda s, f: None)
    m = r["protocols"]["fight_cv"]["models"]
    assert m["xgboost"]["top1_accuracy"] > 0.95 and m["markov"]["top1_accuracy"] < 0.6
    assert m["xgboost"]["vs_markov_top1"]["significant"] is True


@pytest.mark.parametrize("arch", ["gru", "lstm", "transformer"])
def test_sequence_models_learn_order(arch):
    ex = build_examples(pattern_data(fights=4), PatternConfig(), None, 10)
    classes, _ = choose_classes(ex, 2)
    fb = FeatureBuilder(classes, vocabulary(), FighterStats(ex))
    tr = [e for e in ex if e.fight_key not in ("A3", "B3")]
    te = [e for e in ex if e.fight_key in ("A3", "B3")]
    m = make_predictor(arch, {"epochs": 60, "seed": 0}).fit(tr, fb)
    P = m.predict_proba(te)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    y = m.y(te)
    assert (P.argmax(1) == y).mean() > 0.6  # far above the ~0.25 of chance on this fixture


def test_pipeline_splits_by_fight_and_skips_deep_models_without_data():
    ex = build_examples(pattern_data(fights=5), PatternConfig(), None, 10)
    r = train_and_evaluate(ex, {"folds": 5, "min_examples_deep": 10_000}, "cpu", lambda s, f: None)
    assert r["status"] == "ok" and set(r["skipped"]) == {"gru", "lstm", "transformer"}
    assert r["experimental"] is True
    for f in r["protocols"]["fight_cv"]["folds"]:
        if "test_fights" in f:
            assert f["train_fights"] + f["test_fights"] == 10
    m = r["protocols"]["fight_cv"]["models"]
    assert m["markov"]["top1_accuracy"] > m["class_frequency"]["top1_accuracy"]
    assert r["protocols"]["fighter_cv"]["models"]["markov"]["breakdown"]["known_fighters"] is None  # all unseen
    assert r["protocols"]["temporal"]["models"]


def test_insufficient_data_is_reported():
    ex = build_examples([O("A", "v", "left_jab", 0), O("A", "v", "right_cross", 0.5)], PatternConfig(), None, 10)
    assert train_and_evaluate(ex, {}, "cpu", lambda s, f: None)["status"] == "insufficient_data"


def test_live_session_replay():
    from app.predictions.live import CompletedAction, LivePredictionSession, ReplaySource

    seen = []

    def fake_predict(examples):  # records what the session would send to the model
        seen.append(examples[0])
        return [{"predicted_action": "x"}]

    s = LivePredictionSession(fake_predict, history_length=3)
    src = ReplaySource([CompletedAction("A", "right_cross", 0.5, 0.8, opponent_key="B"), CompletedAction("A", "left_jab", 0, 0.3, opponent_key="B"),
                        CompletedAction("B", "slip", 0.55, 0.7, opponent_key="A"), CompletedAction("A", "left_hook", 5.0, 5.3, opponent_key="B")])
    for a in src:
        s.push_action(a)
    a_examples = [e for e in seen if e.fighter_key == "A"]
    assert [s_.category for s_ in a_examples[1].history] == ["left_jab", "right_cross"]
    assert a_examples[1].context.opponent_action == "slip"
    assert a_examples[2].context.sequence_position == 0  # 4.2 s pause -> new sequence


# ============================================================================ API end-to-end
def _seed(client, fights=6):
    from app.db.action_models import ActionAnnotation, VideoAnnotationMeta
    from app.db.database import session_scope
    from app.db.models import Video, VideoStatus

    fa = client.post("/api/fighters", json={"name": "Alpha", "stance": "Orthodox"}).json()["id"]
    fb = client.post("/api/fighters", json={"name": "Beta"}).json()["id"]
    rng = random.Random(3)
    vids = []
    with session_scope() as db:
        for k in range(fights):
            v = Video(title=f"fight{k}", original_filename=f"f{k}.mp4", stored_path=f"videos/f{k}.mp4", file_ext=".mp4",
                      file_size=1, sha256=f"{k:064d}", status=VideoStatus.completed, duration_seconds=120.0)
            db.add(v)
            db.flush()
            vids.append(v.id)
            db.add(VideoAnnotationMeta(video_id=v.id, fully_annotated=True))
            t = 0.0
            for _ in range(10):
                for c in rng.choice([["left_jab", "right_cross", "left_hook"], ["left_jab", "right_cross"]]):
                    db.add(ActionAnnotation(video_id=v.id, fighter_id=fa, category=c, start_time=t, end_time=t + 0.3, status="verified"))
                    t += 0.5
                db.add(ActionAnnotation(video_id=v.id, fighter_id=fb, category="low_kick", start_time=t, end_time=t + 0.4, status="verified"))
                t += 3.0
    return fa, fb, vids


def _wait(client, url, done):
    for _ in range(600):
        r = client.get(url).json()
        if done(r):
            return r
        time.sleep(0.2)
    raise AssertionError(f"timeout waiting for {url}")


def test_predictions_api_end_to_end(client):
    fa, fb, vids = _seed(client)
    assert client.post(f"/api/predictions/{fa}/predict", json={"history": ["left_jab"]}).status_code == 409  # no model yet
    client.put("/api/predictions/settings", json={"architectures": ["gru"], "min_examples_deep": 20, "folds": 3})
    job = client.post("/api/predictions/models/train").json()
    j = _wait(client, "/api/predictions/models/jobs", lambda r: r[0]["status"] in ("completed", "failed"))[0]
    assert j["status"] == "completed", j["error_message"]
    rep = client.get("/api/predictions/models/evaluation").json()["report"]
    assert rep["status"] == "ok" and set(rep["architectures"]) == {"markov", "xgboost", "gru"} and j["id"] == job["id"]
    for a in ("markov", "xgboost", "gru"):
        m = rep["protocols"]["fight_cv"]["models"][a]
        assert m["n"] > 0 and 0 <= m["top1_accuracy"] <= 1 and "confusion_matrix" in m and "vs_markov_top1" in m or a == "markov"
    versions = client.get("/api/predictions/models/versions").json()
    assert sum(v["is_active"] for v in versions) == 1 and {v["architecture"] for v in versions} == {"markov", "xgboost", "gru"}

    r = client.post(f"/api/predictions/{fa}/predict", json={"history": [{"category": "left_jab", "start": 0, "end": 0.3},
                                                                       {"category": "right_cross", "start": 0.5, "end": 0.8}],
                                                           "context": {"distance": 2.0}, "mode": "contextual"}).json()
    assert len(r["top3"]) <= 3 and sum(d["probability"] for d in r["distribution"]) == pytest.approx(1.0, abs=2e-3)
    assert r["reliability"]["experimental_model"] is True and "supporting_observations" in r["reliability"]
    assert r["evidence"]["supporting_combinations"] and "without_context" in r
    pid = r["prediction_id"]
    ok = client.post(f"/api/predictions/{fa}/evaluate", json={"prediction_id": pid, "actual_action": "left_hook"})
    assert ok.status_code == 200 and ok.json()["evaluation"]["status"] == "evaluated"
    assert client.post(f"/api/predictions/{fa}/evaluate", json={"prediction_id": pid, "actual_action": "left_jab"}).status_code == 409
    assert client.post(f"/api/predictions/{fa}/predict", json={"history": ["not_an_action"]}).status_code == 422

    seq = client.get(f"/api/predictions/{fa}/next-sequence", params={"history": ["left_jab"], "depth": 2}).json()
    assert seq["sequences"] and all(len(s["sequence"]) <= 2 for s in seq["sequences"])

    client.post(f"/api/predictions/video/{vids[0]}/predict")
    tl = _wait(client, f"/api/predictions/video/{vids[0]}/timeline", lambda d: d["job"] and d["job"]["status"] in ("completed", "failed"))
    assert tl["job"]["status"] == "completed", tl["job"]["error_message"]
    preds = tl["predictions"]
    assert preds and all(p["in_training_data"] for p in preds)  # the model trained on this fight -> flagged
    evald = [p for p in preds if p["evaluation"]["status"] == "evaluated"]
    assert evald and all(p["evaluation"]["actual_action"] for p in evald)
    perf = client.get(f"/api/predictions/{fa}/performance").json()
    assert perf["in_training_data"]["n"] == sum(1 for p in evald if p["fighter_key"] == fa)
    assert perf["held_out"]["n"] == 1  # the manual prediction
    hist = client.get(f"/api/predictions/{fa}/history").json()
    assert len(hist) >= len([p for p in preds if p["fighter_key"] == fa])
    st = client.get(f"/api/predictions/{fa}/model-status").json()
    assert st["ready"] and st["known_fighter"]


def test_import_external_sequences(client):
    csv = "start_frame,end_frame,punch_class\n0,9,jab\n15,24,cross\n30,39,lead hook\n300,309,jab\n"
    r = client.post("/api/predictions/datasets/import", files={"file": ("S1.csv", io.BytesIO(csv.encode()), "text/csv")},
                    data={"dataset": "boxingvi", "fighter_key": "boxingvi-S1", "fight_key": "boxingvi-V1", "units": "frames", "fps": "30"})
    assert r.status_code == 200 and r.json()["imported"] == 4
    ov = client.get("/api/predictions").json()
    assert ov["external_datasets"] == [{"dataset": "boxingvi", "actions": 4, "fights": 1}]
    assert math.isclose(1.0, 1.0)


def test_import_boxingvi_irregular_sheet(client):
    """BoxingVI sheets vary in layout: blank columns, a header row, extra helper columns."""
    import pandas as pd

    df = pd.DataFrame([["Start Frame", None, "End Frame", None, "Class", None],
                       [30, None, 39, None, "Jab", 9], [45, None, 54, None, "Cross", 9],
                       [60, None, 50, None, "Lead Hook", -10], [90, None, 99, None, "Rear Uppercut", 9]])
    buf = io.BytesIO()
    df.to_excel(buf, header=False, index=False)
    r = client.post("/api/predictions/datasets/import", files={"file": ("V1.xlsx", buf.getvalue(), "application/octet-stream")},
                    data={"dataset": "boxingvi", "fighter_key": "boxingvi-V1", "fight_key": "boxingvi-V1", "fps": "30",
                          "label_map": "boxingvi_orthodox"})
    body = r.json()
    assert body["imported"] == 3 and body["error_count"] == 1  # the end-before-start row is rejected, not guessed
