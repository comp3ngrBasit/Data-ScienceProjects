"""Phase 3 API + real-footage integration. Real clips live in MMA_TEST_FOOTAGE_DIR/actions_ds (not shipped)."""
from __future__ import annotations

import time

import pytest

from tests.conftest import FOOTAGE_DIR, upload, wait_for

pytestmark = pytest.mark.vision


def _wait(fn, ok, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        r = fn()
        if ok(r):
            return r
        time.sleep(1)
    raise AssertionError(f"timeout: {r}")


def test_annotation_crud_and_validation(client, videos):
    vid = upload(client, videos["mp4"]).json()["results"][0]["video"]["id"]
    wait_for(client, vid)
    f = client.post("/api/fighters", json={"name": "Annotated Fighter"}).json()
    assert client.post(f"/api/actions/{vid}/annotations", json={"category": "teleport", "start_time": 0, "end_time": 1}).status_code == 422
    assert client.post(f"/api/actions/{vid}/annotations", json={"category": "left_jab", "start_time": 1, "end_time": 0.5}).status_code == 422
    r = client.post(f"/api/actions/{vid}/annotations", json={"category": "Left Jab", "start_time": 0.5, "end_time": 0.9,
                                                            "fighter_id": f["id"], "opponent_reaction": "slip", "confidence": 0.8})
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["category"] == "left_jab" and a["fighter_name"] == "Annotated Fighter" and a["status"] == "verified"
    u = client.put(f"/api/actions/annotations/{a['id']}", json={"category": "right_cross", "status": "needs_review"}).json()
    assert u["category"] == "right_cross" and u["status"] == "needs_review"
    assert client.put(f"/api/actions/annotations/{a['id']}", json={"end_time": 0.1}).status_code == 422
    assert client.get(f"/api/actions/{vid}/annotations").json()["counts"] == {"right_cross": 1}
    assert client.delete(f"/api/actions/annotations/{a['id']}").status_code == 204
    assert client.get(f"/api/actions/{vid}/annotations").json()["items"] == []
    m = client.put(f"/api/actions/{vid}/dataset-meta", json={"fully_annotated": True, "split": "test"}).json()
    assert m["fully_annotated"] and m["split"] == "test"


def test_guards_without_model_or_pose(client, videos):
    vid = upload(client, videos["mp4"]).json()["results"][0]["video"]["id"]
    wait_for(client, vid)
    r = client.post(f"/api/actions/{vid}/analyze")
    assert r.status_code == 409 and "Phase 2" in r.json()["detail"]
    st = client.get("/api/actions/models/status").json()
    assert st["active_model"] is None and st["models"] == []
    assert client.get("/api/actions/models/evaluation").status_code == 404
    # training with no annotations fails honestly instead of producing a fake model
    assert client.post("/api/actions/models/train", json={"algorithms": ["random_forest"]}).status_code == 202
    st = _wait(lambda: client.get("/api/actions/models/status").json(), lambda d: not d["training_active"])
    assert st["training_jobs"][0]["status"] == "failed" and "enough annotated" in st["training_jobs"][0]["error_message"]
    assert st["models"] == []
    tax = client.get("/api/actions/taxonomy").json()
    assert len(tax["categories"]) >= 33 and not any(c["trained"] for c in tax["categories"])


CLIPS = ["strike_g00_high_kick", "strike_g01_high_kick", "strike_g02_low_kick", "strike_g03_low_kick",
         "strike_g05_punch_unspecified", "strike_g07_punch_unspecified"]


def test_real_footage_annotate_train_evaluate_detect(client):
    ds_dir = FOOTAGE_DIR / "actions_ds"
    if not all((ds_dir / f"{c}.mp4").exists() for c in CLIPS):
        pytest.skip("real strike clips not available")
    vs = client.get("/api/vision/settings").json()
    vs.update(analysis_fps=0, min_box_height=0.05, max_people_per_frame=3)
    client.put("/api/vision/settings", json=vs)
    s = client.get("/api/actions/settings").json()
    s.update(min_examples=2, min_groups=2, epochs=15, augment=2)
    client.put("/api/actions/settings", json=s)

    ids = {}
    for c in CLIPS:
        r = upload(client, ds_dir / f"{c}.mp4")
        ids[c] = r.json()["results"][0]["video"]["id"]
    for vid in ids.values():
        wait_for(client, vid)
        assert client.post(f"/api/vision/{vid}/analyze").status_code == 202
    _wait(lambda: client.get("/api/vision/jobs").json(), lambda d: d["counts"]["queued"] + d["counts"]["processing"] == 0)
    for c, vid in ids.items():
        with open(ds_dir / f"{c}.csv", "rb") as fh:
            r = client.post(f"/api/actions/{vid}/import", files={"file": (f"{c}.csv", fh, "text/csv")}, data={"units": "seconds"})
        assert r.status_code == 200 and r.json()["created"] == 2, r.text
        client.put(f"/api/actions/{vid}/dataset-meta", json={"fully_annotated": True, "split_group": c.split("_")[1]})

    ds = client.get("/api/actions/dataset").json()
    assert ds["videos"] == 6 and ds["annotations_used"] >= 10
    assert client.post("/api/actions/models/train", json={"algorithms": ["random_forest", "xgboost", "svm", "lstm", "tcn"]}).status_code == 202
    assert client.post("/api/actions/models/train", json={}).status_code == 409  # one training at a time
    st = _wait(lambda: client.get("/api/actions/models/status").json(), lambda d: not d["training_active"])
    job = st["training_jobs"][0]
    assert job["status"] == "completed", job
    assert set(job["results"]["comparison"]) == {"random_forest", "xgboost", "svm", "lstm", "tcn"}
    assert st["active_model"] is not None and job["warnings"]  # small-data warning is always shown

    ev = client.get("/api/actions/models/evaluation").json()
    m = ev["metrics"]
    assert m["evaluation"]["strategy"].startswith("leave-one-group-out")
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "per_class", "confusion_matrix", "accuracy_ci95"):
        assert k in m
    n = sum(sum(r) for r in m["confusion_matrix"]["matrix"])
    assert n == m["n_evaluated"]  # every evaluated example counted once (no augmented copies)
    for fold in m["evaluation"]["fold_details"]:
        assert len(fold["test_groups"]) == 1  # split by video group

    vid = ids["strike_g00_high_kick"]
    assert client.post(f"/api/actions/{vid}/analyze").status_code == 202
    assert client.post(f"/api/actions/{vid}/analyze").status_code == 409
    st = _wait(lambda: client.get(f"/api/actions/{vid}/status").json(), lambda d: d["active_job"] is None)
    assert st["current"]["status"] == "completed", st["history"][0]
    res = client.get(f"/api/actions/{vid}/results").json()
    assert res["model_version"] == client.get("/api/actions/models/status").json()["active_model"]["version"]
    for a in res["items"]:
        for k in ("video_id", "track_id", "category", "start_time", "end_time", "duration", "confidence",
                  "start_frame", "end_frame", "model_version", "validated_precision"):
            assert k in a
        assert a["end_time"] > a["start_time"] and 0 <= a["confidence"] <= 1
        assert abs(a["duration"] - (a["end_time"] - a["start_time"])) < 1e-3
    tl = client.get(f"/api/actions/{vid}/timeline").json()
    assert any(lane["annotations"] for lane in tl["lanes"])
    seq = client.get(f"/api/actions/{vid}/sequences").json()
    gt = client.get(f"/api/actions/{vid}/sequences", params={"source": "annotations"}).json()
    assert gt["sequences"] and all(s["n_actions"] == len(s["actions"]) for s in gt["sequences"])
    assert client.get(f"/api/actions/{vid}/export", params={"kind": "sequences", "format": "csv"}).status_code == 200

    if res["items"]:
        a = res["items"][0]
        r = client.post(f"/api/actions/actions/{a['id']}/review", json={"decision": "correct", "category": "front_kick"})
        assert r.status_code == 200 and r.json()["category"] == "front_kick" and r.json()["review_status"] == "corrected"
        anns = client.get(f"/api/actions/{vid}/annotations").json()["items"]
        assert any(x["source"] == "correction" and x["category"] == "front_kick" for x in anns)
        client.post(f"/api/actions/actions/{a['id']}/review", json={"decision": "reject"})
        anns = client.get(f"/api/actions/{vid}/annotations").json()["items"]
        assert not any(x["source"] == "correction" for x in anns)
        assert seq is not None

    assert client.delete(f"/api/actions/{vid}/results").status_code == 204
    assert client.get(f"/api/actions/{vid}/results").status_code == 404
    assert client.get(f"/api/actions/{vid}/annotations").json()["items"]  # annotations survive
    assert client.delete(f"/api/videos/{vid}").status_code == 204
    assert client.get("/api/actions/overview").json()["annotations"]["total"] == 10  # deleted video's labels removed
