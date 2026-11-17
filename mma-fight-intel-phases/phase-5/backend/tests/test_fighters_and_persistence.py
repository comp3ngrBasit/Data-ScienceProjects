from __future__ import annotations

from app.pipeline.stages.frame_sampling import compute_sample_indices
from app.schemas.settings import ProcessingSettings
from tests.conftest import upload, wait_for


def test_fighter_crud_and_assignment(client, videos):
    r = client.post("/api/fighters", json={"name": "Alex Pereira", "stance": "Orthodox", "height_cm": 193, "reach_cm": 201, "weight_class": "Light Heavyweight"})
    assert r.status_code == 201, r.text
    alex = r.json()
    assert client.post("/api/fighters", json={"name": "alex pereira"}).status_code == 409
    assert client.post("/api/fighters", json={"name": "X", "stance": "Switch"}).status_code == 422
    assert client.post("/api/fighters", json={"name": "  "}).status_code == 422
    bob = client.post("/api/fighters", json={"name": "Israel Adesanya", "stance": "Southpaw"}).json()

    vid = upload(client, videos["mp4"]).json()["results"][0]["video"]["id"]
    r = client.patch(f"/api/videos/{vid}", json={"fighter_ids": [alex["id"], bob["id"]], "target_fighter_id": alex["id"], "target_fighter_notes": "red corner"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["target_fighter"]["name"] == "Alex Pereira"
    assert {f["name"] for f in d["fighters"]} == {"Alex Pereira", "Israel Adesanya"}
    assert client.patch(f"/api/videos/{vid}", json={"target_fighter_id": "nope"}).status_code == 422

    f = client.get(f"/api/fighters/{alex['id']}").json()
    assert f["video_count"] == 1 and f["target_video_count"] == 1
    assert f["videos"][0]["is_target"] is True
    assert client.get("/api/videos", params={"fighter_id": bob["id"]}).json()["total"] == 1

    upd = client.put(f"/api/fighters/{bob['id']}", json={**{k: bob[k] for k in ("name", "nickname", "stance", "height_cm", "reach_cm", "weight_class", "notes")}, "nickname": "The Last Stylebender"})
    assert upd.status_code == 200 and upd.json()["nickname"] == "The Last Stylebender"

    assert client.delete(f"/api/fighters/{alex['id']}").status_code == 204
    d = client.get(f"/api/videos/{vid}").json()
    assert d["target_fighter"] is None
    assert [f["name"] for f in d["fighters"]] == ["Israel Adesanya"]
    assert client.get(f"/api/fighters/{alex['id']}").status_code == 404
    wait_for(client, vid)


def test_data_persists_across_restart(make_client, videos):
    with make_client() as c:
        fighter = c.post("/api/fighters", json={"name": "Persisted Fighter", "stance": "Southpaw"}).json()
        vid = upload(c, videos["mp4"]).json()["results"][0]["video"]["id"]
        wait_for(c, vid)
        c.patch(f"/api/videos/{vid}", json={"title": "Persisted Title", "target_fighter_id": fighter["id"]})
        before = c.get(f"/api/videos/{vid}").json()

    with make_client() as c2:  # fresh app instance, same data dir
        after = c2.get(f"/api/videos/{vid}").json()
        assert after["title"] == "Persisted Title"
        assert after["status"] == "completed"
        assert after["frame_count"] == before["frame_count"]
        assert after["target_fighter"]["name"] == "Persisted Fighter"
        assert c2.get(after["thumbnail_url"]).status_code == 200
        assert c2.get(f"/api/videos/{vid}/stream").status_code == 200
        assert c2.get("/api/fighters").json()[0]["video_count"] == 1


def test_sampling_indices():
    s = ProcessingSettings(sampling_mode="interval", sample_interval_seconds=1.0, max_sampled_frames=1000)
    assert compute_sample_indices(250, 25.0, s) == list(range(0, 250, 25))
    s = ProcessingSettings(sampling_mode="interval", sample_interval_seconds=1.0, max_sampled_frames=4)
    assert len(compute_sample_indices(250, 25.0, s)) == 4
    s = ProcessingSettings(sampling_mode="uniform", uniform_frame_count=5)
    idx = compute_sample_indices(100, 25.0, s)
    assert idx == [10, 30, 50, 70, 90]
    assert compute_sample_indices(0, 25.0, s) == []


def test_orphan_files_swept_on_startup(make_client, data_dir, videos):
    with make_client() as c:
        vid = upload(c, videos["mp4"]).json()["results"][0]["video"]["id"]
        wait_for(c, vid)
    (data_dir / "videos" / "deadbeef.mp4").write_bytes(b"x")
    (data_dir / "frames" / "deadbeef").mkdir()
    with make_client() as c2:
        assert not (data_dir / "videos" / "deadbeef.mp4").exists()
        assert not (data_dir / "frames" / "deadbeef").exists()
        assert c2.get(f"/api/videos/{vid}/stream").status_code == 200
        assert (data_dir / "frames" / vid).exists()
