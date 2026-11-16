"""Phase 2 integration tests: real pretrained YOLO pose model + tracker on real footage.

Footage is not shipped with the repo. Put clips in ``backend/tests/footage`` (or set
``MMA_TEST_FOOTAGE_DIR``); tests needing a missing clip are skipped. The no-people
and error-path tests run everywhere.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tests.conftest import footage, upload, wait_for

pytestmark = pytest.mark.vision


def wait_vision(client, video_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/vision/{video_id}/status").json()
        if st["active_job"] is None and st["history"]:
            return st
        time.sleep(0.5)
    raise AssertionError(f"vision analysis timed out: {st}")


def ingest(client, path):
    r = upload(client, path)
    assert r.status_code == 201, r.text
    vid = r.json()["results"][0]["video"]["id"]
    assert wait_for(client, vid)["status"] == "completed"
    return vid


def fast_settings(client, **kw):
    s = client.get("/api/vision/settings").json()
    s.update(dict(analysis_fps=8, image_size=640), **kw)
    r = client.put("/api/vision/settings", json=s)
    assert r.status_code == 200, r.text


def test_endpoints_registered(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/vision/{video_id}/analyze", "/api/vision/{video_id}/status", "/api/vision/{video_id}/detections",
              "/api/vision/{video_id}/pose", "/api/vision/{video_id}/tracking", "/api/vision/{video_id}/annotated-video",
              "/api/vision/{video_id}/assign-fighter", "/api/vision/{video_id}/analysis"]:
        assert p in paths, p
    sys = client.get("/api/vision/system").json()
    assert "torch" in sys["torch"] and len(sys["keypoints"]) == 17


def test_video_without_people_completes_with_zero_tracks(client, videos):
    fast_settings(client)
    vid = ingest(client, videos["mp4"])
    assert client.post(f"/api/vision/{vid}/analyze").status_code == 202
    st = wait_vision(client, vid)
    cur = st["current"]
    assert cur["status"] == "completed", st
    assert cur["stats"]["detections"] == 0 and cur["stats"]["tracks"] == 0
    assert client.get(f"/api/vision/{vid}/tracking").json()["tracks"] == []
    assert client.get(f"/api/vision/{vid}/annotated-video").status_code == 200


def test_requires_phase1_and_prevents_duplicates(client, videos):
    s = client.get("/api/settings").json()
    client.put("/api/settings", json={**s, "auto_process_on_upload": False})
    vid = upload(client, videos["mp4"]).json()["results"][0]["video"]["id"]
    r = client.post(f"/api/vision/{vid}/analyze")
    assert r.status_code == 409 and "Phase 1" in r.json()["detail"]
    client.post(f"/api/videos/{vid}/process")
    wait_for(client, vid)
    fast_settings(client)
    assert client.post(f"/api/vision/{vid}/analyze").status_code == 202
    assert client.post(f"/api/vision/{vid}/analyze").status_code == 409  # duplicate job blocked
    wait_vision(client, vid)
    assert client.get(f"/api/vision/{vid}/detections").status_code == 200


def test_failure_then_retry(client, videos, data_dir):
    fast_settings(client)
    vid = ingest(client, videos["mp4"])
    stored = next((data_dir / "videos").glob(f"{vid}*"))
    backup = stored.read_bytes()
    stored.write_bytes(b"\x00" * 1000)  # simulate corruption after ingest
    client.post(f"/api/vision/{vid}/analyze")
    st = wait_vision(client, vid)
    failed = st["history"][0]
    assert failed["status"] == "failed" and failed["error_message"]
    assert st["current"] is None
    stored.write_bytes(backup)
    r = client.post(f"/api/vision/jobs/{failed['id']}/retry")
    assert r.status_code == 202 and r.json()["attempt"] == 2
    st = wait_vision(client, vid)
    assert st["current"]["status"] == "completed"


def test_device_resolution(monkeypatch):
    import torch

    from app.vision.runtime import resolve_device

    assert resolve_device("cpu", True).device == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    c = resolve_device("cuda", True)
    assert c.device == "cpu" and c.warning
    assert resolve_device("auto", True).device in ("cpu", "mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    c = resolve_device("auto", True)
    assert c.device == "cuda:0" and c.half is True


# ---------------------------------------------------------------- real footage
def _analyse(client, name, **settings):
    fast_settings(client, **settings)
    vid = ingest(client, footage(name))
    assert client.post(f"/api/vision/{vid}/analyze").status_code == 202
    st = wait_vision(client, vid, timeout=900)
    assert st["current"] and st["current"]["status"] == "completed", st["history"][0]
    return vid, st["current"]


def test_overhead_boxing_hard_case(client):
    """Top-down camera: COCO pose models are trained on side views, so recall is limited here.

    This documents the behaviour rather than claiming success: the run completes, people are
    found and tracked, and suggestions are made — but not every person in every frame.
    """
    vid, job = _analyse(client, "overhead_boxing_10s.mp4", min_box_height=0.03, image_size=960, detection_confidence=0.2)
    stats = job["stats"]
    assert stats["max_people_in_frame"] >= 2
    tr = client.get(f"/api/vision/{vid}/tracking").json()
    assert len([t for t in tr["tracks"] if t["status"] != "fragment"]) >= 2
    assert sum(1 for t in tr["tracks"] if t["suggested_role"] == "fighter") >= 1


def test_broadcast_boxing_pose_detections_storage_and_assignment(client):
    vid, job = _analyse(client, "broadcast_boxing_8s.mp4", min_box_height=0.25)
    stats = job["stats"]
    assert stats["detections"] > 0 and stats["landmarks"] == stats["detections"] * 17

    tr = client.get(f"/api/vision/{vid}/tracking").json()
    fighters = sorted([t for t in tr["tracks"] if t["suggested_role"] == "fighter"], key=lambda t: -t["fighter_score"])
    assert len(fighters) >= 2
    # never more than two suggested fighters on screen at once
    fr_sets = [set(range(t["first_frame"], t["last_frame"] + 1)) for t in fighters]
    assert max(sum(1 for s in fr_sets if f in s) for f in range(0, 240)) <= 2 + 1  # +1 tolerance: spans include gaps

    # detections: required fields
    d = client.get(f"/api/vision/{vid}/detections", params={"limit": 50}).json()
    item = d["items"][0]
    assert {"frame_number", "timestamp", "track_id", "confidence", "bbox", "bbox_norm"} <= item.keys()
    assert 0 <= min(item["bbox_norm"]) and max(item["bbox_norm"]) <= 1.01

    # pose: 17 landmarks, normalised, per tracked person; spec landmarks present
    p = client.get(f"/api/vision/{vid}/pose", params={"track_id": fighters[0]["track_id"], "limit": 20}).json()
    assert p["items"] and all(len(i["keypoints"]) == 17 for i in p["items"])
    names = {k["landmark"] for k in p["items"][0]["keypoints"]}
    for n in ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
              "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]:
        assert n in names
    long = client.get(f"/api/vision/{vid}/pose", params={"format": "long", "landmarks": "left_wrist,right_wrist", "limit": 10}).json()
    assert {r["landmark"] for r in long["items"]} <= {"left_wrist", "right_wrist"}

    # Skeleton alignment: visible keypoints of a confident detection lie inside (or very near) its box
    det = client.get(f"/api/vision/{vid}/detections", params={"track_id": fighters[0]["track_id"], "min_confidence": 0.6, "limit": 5}).json()["items"]
    frame = det[0]["frame_number"]
    fr = client.get(f"/api/vision/{vid}/frame/{frame}").json()
    person = next(x for x in fr["people"] if x["track_id"] == fighters[0]["track_id"])
    x1, y1, x2, y2 = det[0]["bbox_norm"]
    vis = [k for k in person["keypoints"] if k["confidence"] > 0.5]
    assert len(vis) >= 6
    inside = [x1 - 0.03 <= k["x"] <= x2 + 0.03 and y1 - 0.03 <= k["y"] <= y2 + 0.03 for k in vis]
    assert np.mean(inside) > 0.9

    # movement computed and distinguishes states
    mv = client.get(f"/api/vision/{vid}/movement", params={"track_id": fighters[0]["track_id"]}).json()
    assert mv["items"] and {"moving", "stationary"} & {r["state"] for r in mv["items"]}
    assert any(r["right_wrist_speed"] is not None for r in mv["items"])
    dist = client.get(f"/api/vision/{vid}/distances").json()
    assert dist["pairs"]

    # manual assignment of track IDs to fighter profiles
    a = client.post("/api/fighters", json={"name": "Fighter A", "stance": "Orthodox"}).json()
    b = client.post("/api/fighters", json={"name": "Fighter B", "stance": "Orthodox"}).json()
    r = client.post(f"/api/vision/{vid}/assign-fighter", json={"assignments": [
        {"track_id": fighters[0]["track_id"], "fighter_id": a["id"]},
        {"track_id": fighters[1]["track_id"], "fighter_id": b["id"]},
    ]})
    assert r.status_code == 200, r.text
    assigned = {t["track_id"]: t for t in r.json()["tracks"]}
    assert assigned[fighters[0]["track_id"]]["assigned_fighter"]["name"] == "Fighter A"
    assert assigned[fighters[0]["track_id"]]["role"] == "fighter"
    # fighter-level filtering and Phase 1 link
    assert client.get(f"/api/vision/{vid}/detections", params={"fighter_id": a["id"]}).json()["total"] > 0
    assert {f["name"] for f in client.get(f"/api/videos/{vid}").json()["fighters"]} >= {"Fighter A", "Fighter B"}
    others = [t["track_id"] for t in r.json()["tracks"] if t["track_id"] not in assigned or assigned[t["track_id"]]["role"] != "fighter"]
    if others:
        client.post(f"/api/vision/{vid}/assign-fighter", json={"assignments": [{"track_id": others[0], "role": "referee"}]})
    assert client.get(f"/api/vision/{vid}/detections", params={"fighters_only": True}).json()["total"] > 0

    # annotated video, overlay, exports
    av = client.get(f"/api/vision/{vid}/annotated-video", headers={"Range": "bytes=0-1023"})
    assert av.status_code == 206
    ov = client.get(f"/api/vision/{vid}/overlay", params={"start_frame": 0, "end_frame": 120}).json()
    assert ov["frames"] and len(ov["frames"][0][1][0][6]) == 51
    csv = client.get(f"/api/vision/{vid}/export", params={"kind": "movement"})
    assert csv.status_code == 200 and "right_wrist_speed" in csv.text.splitlines()[0]
    npz = client.get(f"/api/vision/{vid}/export", params={"kind": "npz"})
    assert npz.status_code == 200 and npz.content[:2] == b"PK"

    # overview
    ov = client.get("/api/vision/overview").json()
    assert ov["videos_analysed"] == 1 and ov["total_frames_analysed"] == stats["frames_analysed"]
    assert ov["distinct_fighters_identified"] == 2
    assert ov["fighter_tracks_assigned"] >= 2


def test_reanalysis_replaces_results_and_delete(client, data_dir):
    vid, first = _analyse(client, "ufc_single_fighter.mp4")
    assert first["stats"]["detections"] > 0
    assert client.post(f"/api/vision/{vid}/analyze").status_code == 202
    st = wait_vision(client, vid, timeout=900)
    assert st["current"]["id"] != first["id"]
    # old rows gone: every detection belongs to the new job
    assert client.get(f"/api/vision/{vid}/detections").json()["job_id"] == st["current"]["id"]
    assert not (data_dir / "vision" / vid / first["id"]).exists()
    # delete analysis
    assert client.delete(f"/api/vision/{vid}/analysis").status_code == 204
    assert client.get(f"/api/vision/{vid}/tracking").status_code == 404
    assert not (data_dir / "vision" / vid).exists()
    # phase 1 still fine
    assert client.get(f"/api/videos/{vid}").json()["status"] == "completed"


def test_deleting_video_removes_vision_data(client, data_dir):
    vid, _ = _analyse(client, "two_people_outdoor.mp4")
    assert (data_dir / "vision" / vid).exists()
    assert client.delete(f"/api/videos/{vid}").status_code == 204
    assert not (data_dir / "vision" / vid).exists()
    assert client.get(f"/api/vision/{vid}/status").status_code == 404
    assert client.get("/api/vision/overview").json()["videos_analysed"] == 0


def test_persistence_across_restart(make_client):
    with make_client() as c:
        vid, job = _analyse(c, "two_people_outdoor.mp4")
        n = c.get(f"/api/vision/{vid}/detections").json()["total"]
    with make_client() as c2:
        st = c2.get(f"/api/vision/{vid}/status").json()
        assert st["current"]["id"] == job["id"]
        assert c2.get(f"/api/vision/{vid}/detections").json()["total"] == n
        assert c2.get(f"/api/vision/{vid}/annotated-video").status_code == 200


