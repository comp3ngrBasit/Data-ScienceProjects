from __future__ import annotations

from tests.conftest import make_ffmpeg_video, upload, wait_for


def test_health_and_docs(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in [
        "/api/videos/upload", "/api/videos", "/api/videos/{video_id}", "/api/videos/{video_id}/process",
        "/api/videos/{video_id}/status", "/api/fighters", "/api/fighters/{fighter_id}",
    ]:
        assert p in paths


def test_upload_multiple_and_process(client, videos):
    r = upload(client, videos["mp4"], videos["avi"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["created"] == 2
    ids = [x["video"]["id"] for x in body["results"]]
    assert len(set(ids)) == 2

    for vid in ids:
        st = wait_for(client, vid)
        assert st["status"] == "completed", st
        d = client.get(f"/api/videos/{vid}").json()
        assert d["width"] == 320 and d["height"] == 240
        assert abs(d["fps"] - 25) < 0.5
        assert d["frame_count"] > 0
        assert d["duration_seconds"] > 1.5
        assert d["video_codec"]
        assert d["has_audio"] is False
        assert d["sampled_frame_count"] > 0
        assert d["quality"]["rating"] in ("Good", "Fair", "Poor")
        assert d["thumbnail_url"]
        assert client.get(d["thumbnail_url"]).headers["content-type"] == "image/jpeg"
        frames = client.get(f"/api/videos/{vid}/frames").json()
        assert frames["total"] == d["sampled_frame_count"]
        ts = [f["timestamp_seconds"] for f in frames["items"]]
        assert ts == sorted(ts)
        img = client.get(frames["items"][0]["image_url"])
        assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
        jobs = client.get(f"/api/videos/{vid}/jobs").json()
        assert jobs[0]["status"] == "completed" and jobs[0]["progress"] == 1.0
        assert all(s["status"] == "completed" for s in jobs[0]["stage_log"])

    # Non-H.264 / non-MP4 files get a browser-playable preview proxy
    avi = client.get(f"/api/videos/{ids[1]}").json()
    assert avi["has_preview_proxy"] is True
    r = client.get(f"/api/videos/{ids[1]}/stream", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-type"] == "video/mp4"


def test_h264_with_audio_metadata(client, tmp_path):
    p = make_ffmpeg_video(tmp_path / "ufc_bout.mp4", seconds=3, audio=True)
    r = upload(client, p)
    vid = r.json()["results"][0]["video"]["id"]
    assert wait_for(client, vid)["status"] == "completed"
    d = client.get(f"/api/videos/{vid}").json()
    assert d["video_codec"] == "h264"
    assert d["has_audio"] is True and d["audio_codec"] == "aac"
    assert d["width"] == 640 and d["height"] == 360
    assert abs(d["fps"] - 30) < 0.1
    assert abs(d["duration_seconds"] - 3) < 0.2
    assert d["frame_count"] == 90
    assert d["has_preview_proxy"] is False  # already browser-safe
    assert "ffmpeg" in d["metadata_source"] or "ffprobe" in d["metadata_source"]


def test_mov_upload(client, tmp_path):
    p = make_ffmpeg_video(tmp_path / "clip.mov", seconds=2, audio=False)
    r = upload(client, p, mime="video/quicktime")
    assert r.status_code == 201, r.text
    vid = r.json()["results"][0]["video"]["id"]
    assert wait_for(client, vid)["status"] == "completed"
    assert client.get(f"/api/videos/{vid}").json()["has_preview_proxy"] is True


def test_duplicate_rejected(client, videos):
    first = upload(client, videos["mp4"])
    assert first.status_code == 201
    dup = upload(client, videos["mp4"])
    assert dup.status_code == 409
    assert dup.json()["results"][0]["existing_video_id"] == first.json()["results"][0]["video"]["id"]


def test_invalid_files_rejected(client, videos):
    assert upload(client, videos["txt"]).status_code == 415
    r = upload(client, videos["fake_avi"])
    assert r.status_code == 415
    assert "not a recognised" in r.json()["results"][0]["error"]
    # mixed batch: good file stored, bad one reported
    r = upload(client, videos["mp4"], videos["txt"])
    assert r.status_code == 201
    assert r.json()["created"] == 1 and r.json()["rejected"] == 1
    assert client.get("/api/videos").json()["total"] == 1


def test_corrupted_video_fails_and_retry(client, videos):
    r = upload(client, videos["corrupt"])
    assert r.status_code == 201  # passes header sniffing...
    vid = r.json()["results"][0]["video"]["id"]
    st = wait_for(client, vid)
    assert st["status"] == "failed"  # ...but processing detects it
    assert st["error_message"]
    job_id = st["latest_job"]["id"]

    retry = client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 202
    assert retry.json()["attempt"] == 2
    st = wait_for(client, vid)
    assert st["status"] == "failed"
    assert len(client.get(f"/api/videos/{vid}/jobs").json()) == 2
    # completed jobs can't be "retried"
    assert client.get("/api/jobs").json()["counts"]["failed"] == 2


def test_rename_search_filter_delete(client, videos, data_dir):
    ids = [x["video"]["id"] for x in upload(client, videos["mp4"], videos["avi"]).json()["results"]]
    for v in ids:
        wait_for(client, v)
    r = client.patch(f"/api/videos/{ids[0]}", json={"title": "Main Event Round 1"})
    assert r.status_code == 200 and r.json()["title"] == "Main Event Round 1"
    assert client.patch(f"/api/videos/{ids[0]}", json={"title": "   "}).status_code == 422

    assert client.get("/api/videos", params={"search": "main event"}).json()["total"] == 1
    assert client.get("/api/videos", params={"search": "sparring"}).json()["total"] == 1
    assert client.get("/api/videos", params={"status": "completed"}).json()["total"] == 2
    assert client.get("/api/videos", params={"sort": "title", "order": "asc"}).json()["items"][0]["title"] == "Main Event Round 1"

    assert any((data_dir / "frames" / ids[1]).iterdir())
    assert client.delete(f"/api/videos/{ids[1]}").status_code == 204
    assert client.get(f"/api/videos/{ids[1]}").status_code == 404
    assert not (data_dir / "frames" / ids[1]).exists()
    assert not (data_dir / "thumbnails" / f"{ids[1]}.jpg").exists()
    assert not list((data_dir / "videos").glob(f"{ids[1]}*"))
    assert client.get("/api/videos").json()["total"] == 1


def test_reprocess_conflict_and_settings(client, videos):
    s = client.get("/api/settings").json()
    s.update(sampling_mode="uniform", uniform_frame_count=7, save_frame_images=False)
    assert client.put("/api/settings", json=s).status_code == 200
    vid = upload(client, videos["mp4"]).json()["results"][0]["video"]["id"]
    wait_for(client, vid)
    d = client.get(f"/api/videos/{vid}").json()
    assert d["sampled_frame_count"] == 7
    assert d["sampling_config"]["images_saved"] is False
    assert client.get(f"/api/videos/{vid}/frames").json()["items"][0]["image_url"] is None

    s.update(sampling_mode="interval", sample_interval_seconds=0.5, save_frame_images=True)
    client.put("/api/settings", json=s)
    assert client.post(f"/api/videos/{vid}/process").status_code == 202
    second = client.post(f"/api/videos/{vid}/process")
    assert second.status_code in (202, 409)  # 409 if the first job is still active
    wait_for(client, vid)
    d = client.get(f"/api/videos/{vid}").json()
    assert d["sampling_config"]["mode"] == "interval"
    assert d["sampled_frame_count"] == len(range(0, 75, 12))  # 75 frames, step round(0.5 s * 25 fps)
    assert client.put("/api/settings", json={**s, "sample_interval_seconds": -1}).status_code == 422


def test_manual_upload_without_autoprocess(client, videos):
    s = client.get("/api/settings").json()
    client.put("/api/settings", json={**s, "auto_process_on_upload": False})
    vid = upload(client, videos["mp4"]).json()["results"][0]["video"]["id"]
    assert client.get(f"/api/videos/{vid}/status").json()["status"] == "uploaded"
    assert client.post(f"/api/videos/{vid}/process").status_code == 202
    assert wait_for(client, vid)["status"] == "completed"


def test_upload_size_limit(client, videos):
    s = client.get("/api/settings").json()
    client.put("/api/settings", json={**s, "max_upload_size_mb": 1})
    big = videos["mp4"].parent / "big.mp4"
    big.write_bytes(videos["mp4"].read_bytes() + b"\x00" * (3 * 1024 * 1024))
    assert upload(client, big).status_code == 413


def test_overview_stats(client, videos):
    ids = [x["video"]["id"] for x in upload(client, videos["mp4"], videos["corrupt"]).json()["results"]]
    for v in ids:
        wait_for(client, v)
    o = client.get("/api/stats/overview").json()
    assert o["total_videos"] == 2
    assert o["videos_processed"] == 1
    assert o["videos_failed"] == 1
    assert sum(d["count"] for d in o["uploads_by_day"]) == 2
    assert o["total_sampled_frames"] > 0
    assert len(o["recent_videos"]) == 2


def test_delete_while_processing(client, tmp_path, data_dir):
    from tests.conftest import make_opencv_video

    p = make_opencv_video(tmp_path / "long_fight.avi", seconds=40, fps=30, size=(960, 540), fourcc="MJPG")
    vid = upload(client, p).json()["results"][0]["video"]["id"]
    wait_for(client, vid, statuses=("processing",), timeout=30)
    r = client.delete(f"/api/videos/{vid}")
    assert r.status_code == 204
    assert client.get(f"/api/videos/{vid}").status_code == 404
    import time

    time.sleep(0.5)
    assert not (data_dir / "frames" / vid).exists()
    assert not (data_dir / "previews" / f"{vid}.mp4").exists()
    assert not list((data_dir / "videos").glob(f"{vid}*"))
    assert client.get("/api/health").json()["status"] == "ok"
