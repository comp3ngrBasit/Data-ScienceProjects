"""Phase 6: integration of all phases — dashboard, fighter intelligence, unified pipeline, model evaluation & management.

Dashboard numbers are checked against direct database counts; the pipeline test runs on real footage.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.conftest import footage, upload, wait_for
from tests.test_predictions import _seed


def _poll(fn, done, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        r = fn()
        if done(r):
            return r
        time.sleep(0.3)
    raise AssertionError("timeout")


def _train_prediction(client):
    client.put("/api/predictions/settings", json={"architectures": [], "folds": 3})
    r = client.post("/api/models/train", json={"kind": "prediction"})
    assert r.status_code == 202 and r.json()["kind"] == "prediction"
    return _poll(lambda: client.get("/api/predictions/models/jobs").json()[0], lambda j: j["status"] in ("completed", "failed"))


def test_dashboard_matches_database(client):
    fa, fb, vids = _seed(client)
    from app.db.action_models import ActionAnnotation
    from app.db.database import session_scope

    with session_scope() as db:  # an unrecognised action must be excluded everywhere
        db.add(ActionAnnotation(video_id=vids[0], fighter_id=fa, category="unknown", start_time=200, end_time=200.3, status="verified"))
    with session_scope() as db:
        n_ann = db.query(ActionAnnotation).filter(ActionAnnotation.status == "verified").count()
        n_unknown = db.query(ActionAnnotation).filter(ActionAnnotation.category == "unknown").count()
    ov = client.get("/api/dashboard/overview").json()
    t = ov["totals"]
    assert t["videos"] == 6 and t["verified_annotations"] == n_ann
    assert t["actions_in_analysis"] == n_ann - n_unknown  # unknown actions never enter analysis
    assert t["fights_analysed"] == 6 and t["fighters_analysed"] == 2
    assert t["predictions_generated"] == 0 and ov["models"]["next_move"] is None
    assert ov["status"] == {"action_recognition": "not_trained", "next_move": "not_trained"}
    assert t["combinations_identified"] > 0

    rows = {r["fighter_key"]: r for r in client.get("/api/dashboard/fighters").json()}
    alpha = rows[fa]
    assert alpha["is_profile"] and alpha["videos"] == 6 and alpha["top_action"] in ("Left Jab", "Right Cross")
    assert rows[fb]["actions"] == 60 and rows[fb]["offensive_share"] == 1.0  # 10 low kicks per fight x 6

    an = client.get("/api/dashboard/analytics").json()
    assert an["actions"] == t["actions_in_analysis"]
    assert sum(d["count"] for d in an["action_distribution"]) == an["actions"]
    assert an["markov"]["rows"] and an["top_combinations"][0]["occurrences"] >= an["top_combinations"][-1]["occurrences"]
    assert sum(g["count"] for g in an["group_transitions"]) == sum(s["count"] * (s["length"] - 1) for s in an["sequence_length_distribution"])
    one = client.get("/api/dashboard/analytics", params={"fighter": fb}).json()
    assert one["actions"] == 60


def test_fighter_intelligence_and_predictions(client):
    fa, fb, vids = _seed(client)
    intel = client.get(f"/api/fighters/{fa}/intelligence").json()
    o = intel["observed"]
    assert intel["fighter"]["name"] == "Alpha" and len(intel["videos"]) == 6
    assert o["actions"] == sum(v["actions"] for v in intel["videos"]) and o["fights"] == 6
    assert o["strike_frequency"] and o["opening_actions"][0]["label"] == "Left Jab"  # every burst opens with a jab
    assert o["mean_time_between_actions"] == pytest.approx(0.2, abs=1e-6)
    assert o["timeline"] and intel["patterns"]["combinations"]
    assert o["counterattacks"] == [] or all("counter" in c for c in o["counterattacks"])
    one = client.get(f"/api/fighters/{fa}/intelligence", params={"video": [vids[0]]}).json()
    assert one["observed"]["fights"] == 1 and one["warnings"]
    assert client.get(f"/api/fighters/{fa}/patterns").json()["markov"]["rows"]
    assert client.get("/api/fighters/nope/intelligence").status_code == 404

    p0 = client.get(f"/api/fighters/{fa}/predictions").json()
    assert p0["available"] is False and "No active" in p0["reason"]
    job = _train_prediction(client)
    assert job["status"] == "completed"
    p = client.get(f"/api/fighters/{fa}/predictions").json()
    assert p["available"] and 2 <= len(p["previous_actions"]) <= 5
    assert abs(sum(d["probability"] for d in p["prediction"]["distribution"]) - 1) < 2e-3
    kinds = [x["kind"] for x in p["timeline"]]
    assert kinds[:len(p["previous_actions"])] == ["detected"] * len(p["previous_actions"]) and "predicted" in kinds
    assert "not observed" in p["disclaimer"]
    # prediction at a video timestamp uses only actions up to that time
    pt = client.get(f"/api/fighters/{fa}/predictions", params={"video_id": vids[0], "at": 1.0}).json()
    assert pt["available"] and all(a["end"] <= 1.0 + 1e-6 for a in pt["previous_actions"])


def test_model_evaluation_management_and_rollback(client):
    fa, fb, vids = _seed(client)
    ev0 = client.get("/api/models/evaluation").json()
    assert ev0["action_recognition"] is None and ev0["next_move"] is None
    pe = ev0["pattern_recognition"]
    assert pe["combination_detection"]["status"] == "insufficient_data"  # no detections to compare yet
    assert pe["transition_reliability"]["fights_half_a"] == 3 and pe["coverage"]["held_out_transitions"] > 0
    assert 0 <= pe["coverage"]["exact_transition_seen"] <= 1

    _train_prediction(client)
    ev = client.get("/api/models/evaluation").json()["next_move"]
    m = ev["report"]["protocols"]["fight_cv"]["models"]["markov"]
    assert m["recall_at_3"] == m["top3_accuracy"] and m["precision_at_3"] == pytest.approx(m["top3_accuracy"] / 3, abs=1e-4)
    s = ev["samples"]["fight_cv"]
    assert s["test_examples"] == ev["report"]["data"]["examples"] and s["mean_validation_examples"] > 0
    v1 = client.get("/api/models/versions").json()
    active1 = next(m for m in v1["prediction"] if m["is_active"])
    assert active1["validation"]["passed"] and v1["deployments"][0]["reason"] == "training"

    # second training run -> a new version; rolling back to the first one is recorded
    _train_prediction(client)
    v2 = client.get("/api/models/versions").json()
    assert {m["version"] for m in v2["prediction"]} == {1, 2}
    r = client.post("/api/models/activate", json={"kind": "prediction", "model_id": active1["id"], "reason": "rollback"})
    assert r.status_code == 200 and r.json()["active"] == active1["id"]
    assert client.get("/api/predictions").json()["active_model"]["id"] == active1["id"]
    deps = client.get("/api/models/deployments", params={"kind": "prediction"}).json()
    assert deps[0]["reason"] == "rollback" and deps[0]["model_id"] == active1["id"]
    assert client.post("/api/models/activate", json={"kind": "prediction", "model_id": "nope"}).status_code == 404
    assert client.post("/api/models/train", json={"kind": "other"}).status_code == 422


def test_failed_model_inference_is_reported(client):
    fa, fb, vids = _seed(client)
    _train_prediction(client)
    active = client.get("/api/predictions").json()["active_model"]
    from app.db.database import session_scope
    from app.db.prediction_models import PredictionModel
    from app.predictions.service import MODEL_CACHE

    with session_scope() as db:
        path = Path(db.get(PredictionModel, active["id"]).path)
    for f in path.iterdir():
        f.write_bytes(b"corrupted")
    MODEL_CACHE.clear()
    r = client.get(f"/api/fighters/{fa}/predictions")
    assert r.status_code == 503 and "inference failed" in r.json()["detail"]


def test_interrupted_pipeline_is_marked_failed(client):
    from app.db.database import session_scope
    from app.db.models import JobStatus, Video, VideoStatus
    from app.db.platform_models import PipelineRun
    from app.platform.pipeline import PipelineManager

    with session_scope() as db:
        v = Video(title="x", original_filename="x.mp4", stored_path="videos/x.mp4", file_ext=".mp4", file_size=1, sha256="f" * 64,
                  status=VideoStatus.completed)
        db.add(v)
        db.flush()
        run = PipelineRun(video_id=v.id, status=JobStatus.processing, stages=[], options={})
        db.add(run)
        db.flush()
        rid = run.id
    PipelineManager({}).start()
    with session_scope() as db:
        r = db.get(PipelineRun, rid)
        assert r.status == JobStatus.failed and "Interrupted" in r.error_message


@pytest.mark.vision
def test_real_footage_unified_pipeline(client, tmp_path):
    """Real MMA/boxing footage through the unified pipeline; a corrupted upload fails cleanly."""
    clips = [footage("ufc_single_fighter.mp4"), footage("broadcast_boxing_8s.mp4")]
    bad = tmp_path / "corrupted_fight.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x13" * 4000)
    res = upload(client, *clips, bad).json()["results"]
    ids = [r["video"]["id"] for r in res if r.get("video")]
    assert len(ids) == 3
    states = {i: wait_for(client, i)["status"] for i in ids}
    assert list(states.values()).count("completed") == 2 and list(states.values()).count("failed") == 1
    bad_id = next(i for i, s in states.items() if s == "failed")
    good = [i for i in ids if i != bad_id]
    vs = client.get("/api/vision/settings").json()
    vs.update(analysis_fps=10)
    client.put("/api/vision/settings", json=vs)

    runs = [client.post(f"/api/pipeline/videos/{v}/run").json() for v in good]
    assert all(r["created"] for r in runs)
    assert client.post(f"/api/pipeline/videos/{good[0]}/run").json()["created"] is False  # no duplicate runs
    for v in good:
        run = _poll(lambda v=v: client.get(f"/api/pipeline/videos/{v}").json()[0], lambda r: r["status"] in ("completed", "failed"), 900)
        assert run["status"] == "completed", run
        st = {s["name"]: s for s in run["stages"]}
        assert st["processing"]["status"] == "reused" and st["vision"]["status"] == "completed"
        assert st["actions"]["status"] == "skipped" and "No trained action" in st["actions"]["message"]  # never faked
        assert st["patterns"]["status"] == "completed" and st["prediction"]["status"] == "skipped"
        ca = client.get(f"/api/videos/{v}/complete-analysis").json()
        assert ca["stages"]["vision"] == "completed" and ca["tracks"] and ca["annotated_video_url"]
        assert ca["actions"] == [] and ca["predictions"] == []
        # occlusion / missing landmarks are represented, not hidden
        assert all(t["coverage"] <= 1.0 and t["interruptions"] >= 0 for t in ca["tracks"])
    mv = client.get(f"/api/vision/{good[0]}/movement").json()
    assert mv  # real pose-derived movement metrics exist
    # second run reuses every completed stage
    r2 = client.post(f"/api/pipeline/videos/{good[1]}/run").json()
    run = _poll(lambda: client.get(f"/api/pipeline/videos/{good[1]}").json()[0], lambda r: r["status"] in ("completed", "failed") and r["id"] == r2["id"], 300)
    assert {s["name"]: s["status"] for s in run["stages"]}["vision"] == "reused"
    # the failed video's pipeline fails at processing with the real error
    rb = client.post(f"/api/pipeline/videos/{bad_id}/run", params={"force": True}).json()
    runb = _poll(lambda: client.get(f"/api/pipeline/videos/{bad_id}").json()[0], lambda r: r["status"] in ("completed", "failed") and r["id"] == rb["id"], 120)
    assert runb["status"] == "failed" and runb["error_message"].startswith("Video processing")
    ov = client.get("/api/dashboard/overview").json()["totals"]
    assert ov["videos"] == 3 and ov["videos_processed"] == 2 and ov["videos_analysed"] == 2


def test_reanalysis_keeps_fighter_assignments(client):
    from app.db.database import session_scope
    from app.db.models import JobStatus, Video, VideoStatus
    from app.db.vision_models import TrackingSession, VisionJob
    from app.vision.job_manager import carry_over_assignments

    fid = client.post("/api/fighters", json={"name": "Carry"}).json()["id"]
    with session_scope() as db:
        v = Video(title="x", original_filename="x.mp4", stored_path="videos/x.mp4", file_ext=".mp4", file_size=1, sha256="e" * 64,
                  status=VideoStatus.completed)
        db.add(v)
        db.flush()
        jobs = []
        for _ in range(2):
            j = VisionJob(video_id=v.id, status=JobStatus.completed, attempt=1, settings_snapshot={}, model_name="m")
            db.add(j)
            db.flush()
            jobs.append(j.id)

        def tr(job, tid, a, b, fighter=None, role="unreviewed"):
            db.add(TrackingSession(job_id=job, video_id=v.id, track_id=tid, assigned_fighter_id=fighter, role=role, first_frame=0, last_frame=1,
                                   first_timestamp=a, last_timestamp=b, frames_detected=1, coverage=1, video_coverage=1, avg_confidence=1,
                                   avg_box_height=0.5, avg_pose_confidence=1, status="continuous"))
        tr(jobs[0], 1, 0, 10, fid, "fighter")
        tr(jobs[0], 5, 0, 10, None, "referee")
        tr(jobs[1], 1, 0.2, 9.8)   # same person, same id
        tr(jobs[1], 7, 0.0, 10.0)  # the referee, new id
        tr(jobs[1], 9, 20, 30)     # no overlap: stays unassigned
        db.flush()
        assert carry_over_assignments(db, jobs[0], jobs[1]) == 2
        new = {t.track_id: t for t in db.query(TrackingSession).filter_by(job_id=jobs[1])}
        assert new[1].assigned_fighter_id == fid and new[7].role == "referee" and new[9].assigned_fighter_id is None
