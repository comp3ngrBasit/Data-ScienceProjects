# MMA Fight Intelligence — Phase 1

**Application foundation, video upload & processing** for the AI-Powered MMA Fight Intelligence & Next-Move Prediction System.

Phase 1 lets you upload UFC/MMA fight videos, manage a video library, preview footage, create fighter profiles, assign fighters to videos (manually), and process every video in the background into usable data: metadata, thumbnails, sampled frames with timestamps, and basic quality measures. This is the base that Phases 2–6 (detection, pose, strike classification, pattern recognition, prediction) will build on.

> **What's not in Phase 1:** no fighter detection, tracking, pose estimation, strike recognition or prediction. Fighter identification is **manual**. Quality scores are simple signal heuristics, not ML.

---

## Contents
1. [Requirements](#1-requirements)
2. [Install on Windows 11](#2-install-on-windows-11)
3. [Run](#3-run)
4. [Using the app](#4-using-the-app)
5. [API](#5-api)
6. [Configuration](#6-configuration)
7. [Project structure & architecture](#7-project-structure--architecture)
8. [Tests](#8-tests)
9. [Troubleshooting](#9-troubleshooting)
10. [Known limitations](#10-known-limitations)

---

## 1. Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | **3.11+** (3.12 recommended) | `winget install Python.Python.3.12` — tick *Add python.exe to PATH* if using the python.org installer |
| Node.js | **22 LTS** (≥ 20.19) | `winget install OpenJS.NodeJS.LTS` |
| FFmpeg | optional | Bundled automatically via the `imageio-ffmpeg` pip package. Installing a full FFmpeg (`winget install Gyan.FFmpeg`) adds `ffprobe` for richer metadata. |

Disk space: originals are stored as-is, plus small thumbnails, sampled JPEG frames (bounded by settings) and, for AVI/MOV/non-H.264 files, a compressed MP4 preview.

## 2. Install on Windows 11

### Option A — one click
1. Unzip the project to a folder **without non-English characters in the path** (e.g. `C:\Projects\mma-fight-intel`). OpenCV on Windows can fail to open files under non-ASCII paths.
2. Double-click **`setup.bat`**. It will:
   - create a virtual environment in `backend\.venv`
   - install Python packages from `backend\requirements.txt`
   - run `npm install` and `npm run build` in `frontend\`

### Option B — manual (PowerShell)
```powershell
cd C:\Projects\mma-fight-intel

# Backend
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

# Frontend
cd frontend
npm install
npm run build
cd ..
```

## 3. Run

### Normal use — single server
Double-click **`start.bat`**, or:
```powershell
cd backend
.venv\Scripts\python.exe run.py
```
Open **http://127.0.0.1:8000**. The FastAPI server serves both the API and the built React app.
Swagger API docs: **http://127.0.0.1:8000/docs**.

### Development — hot reload
Double-click **`start-dev.bat`** (opens two windows), or run in two terminals:
```powershell
# Terminal 1 – API with auto-reload on :8000
cd backend
.venv\Scripts\python.exe run.py --reload

# Terminal 2 – Vite dev server on :5173 (proxies /api to :8000)
cd frontend
npm run dev
```
Open **http://localhost:5173**.

Stop with `Ctrl+C`. All data persists in `backend\data\` across restarts.

## 4. Using the app

| Page | What it does |
|---|---|
| **Overview** | Totals (videos, fighters, processed, awaiting), uploads over the last 14 days, library status, storage use, recent uploads and recent processing activity. All numbers come from the database. **Upload Fight Footage** button. |
| **Video Library** | Grid or table of videos with thumbnail, title, duration, resolution, frame rate, file size, upload date and status. Search (title/filename), filter by status or fighter, sort, rename, delete, process/retry. |
| **Upload Videos** | Drag-and-drop or browse for multiple MP4/MOV/AVI files. Per-file progress, cancel, retry. Files are checked client-side (type, size) and server-side (extension, container signature, size, duplicate content via SHA-256). Uploads keep going while you use other pages. |
| **Video details** | Player (seekable; uses an MP4 preview for formats browsers can't play), full metadata, assigned fighters + **fighter to analyze** + identification notes, processing status with live stage progress, sampled frames with timestamps (click to seek) and a per-frame sharpness/brightness/contrast chart, basic quality assessment, and processing history (every attempt, stage timings, errors, warnings). |
| **Fighters** | Create/edit/delete profiles: name, nickname, stance (Orthodox / Southpaw / Unknown), height, reach, weight class, notes. Each profile lists its associated videos and which ones it is the analysis target in. |
| **Processing Jobs** | All jobs with status filters (Queued / Processing / Completed / Failed / Cancelled), live progress, stage breakdown, error messages, **Retry** and **Retry all failed**. |
| **Settings** | Frame sampling (fixed interval or N evenly spaced frames, max frames per video, stored width, JPEG quality, store images or stats only), auto-process on upload, preview generation, max upload size, thumbnail width. Shows FFmpeg/OpenCV status and storage use. |

### Processing pipeline
Each job runs these stages in a background worker thread (the UI never blocks):

1. **Metadata** — FFprobe (or `ffmpeg -i` output if only the bundled FFmpeg exists) + OpenCV: duration, frame count, FPS, resolution, rotation, video codec, pixel format, container, bitrate, audio presence/codec/sample rate/channels. OpenCV must be able to decode frames, otherwise the job fails as corrupted/unsupported.
2. **Thumbnail** — the sharpest well-exposed of several candidate frames.
3. **Frame sampling** — configurable; stores JPEGs (optional) plus frame index, timestamp, brightness, contrast and sharpness per frame. Every extracted frame is **not** stored — only the sampled subset, capped by *max frames*.
4. **Quality** — heuristic score from resolution, FPS, bitrate, blur %, dark/overexposed %, decode failures.
5. **Preview** — for AVI, MOV or non-H.264 files, an H.264 MP4 preview (≤ 1280 px wide) for browser playback. The original is never modified.

Statuses: **Uploaded → Queued → Processing → Completed / Failed**. Failed jobs show the reason and can be retried; each retry is a new attempt in the history. Jobs that were running when the app stopped are marked failed ("Interrupted") on next start and can be retried; queued jobs resume automatically.

## 5. API

Interactive docs at `/docs` (Swagger) and `/redoc`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/videos/upload` | Upload one or more files (`files` multipart field). Returns per-file results (`created` / `duplicate` / `rejected`). 201 if anything was stored; otherwise 409/413/415/400 |
| GET | `/api/videos` | List — `search`, `status`, `fighter_id`, `unassigned`, `sort`, `order`, `limit`, `offset` |
| GET | `/api/videos/{video_id}` | Full details incl. metadata, quality, jobs |
| PATCH | `/api/videos/{video_id}` | Rename, notes, `fighter_ids`, `target_fighter_id`, `target_fighter_notes` |
| DELETE | `/api/videos/{video_id}` | Delete video + all derived files (cancels a running job first) |
| POST | `/api/videos/{video_id}/process` | Queue (re)processing / retry |
| GET | `/api/videos/{video_id}/status` | Status + latest job |
| GET | `/api/videos/{video_id}/jobs` | Processing history |
| GET | `/api/videos/{video_id}/stream` | Video stream with HTTP Range (`source=auto\|original\|preview`) |
| GET | `/api/videos/{video_id}/download` | Original file |
| GET | `/api/videos/{video_id}/thumbnail` | Thumbnail JPEG |
| GET | `/api/videos/{video_id}/frames` | Sampled frames (timestamps, stats, image URLs) |
| GET | `/api/videos/{video_id}/frames/{frame_id}/image` | Frame JPEG |
| GET / POST | `/api/fighters` | List / create |
| GET / PUT / DELETE | `/api/fighters/{fighter_id}` | Get (with videos) / update / delete (videos are kept, just unassigned) |
| GET | `/api/jobs` | Jobs list + counts per status |
| GET | `/api/jobs/{job_id}` | One job |
| POST | `/api/jobs/{job_id}/retry` | Retry a failed/cancelled job |
| GET / PUT | `/api/settings` | Processing settings |
| GET | `/api/stats/overview` | Dashboard data |
| GET | `/api/system`, `/api/health` | Tooling info, liveness |

## 6. Configuration

Runtime settings are edited on the **Settings** page and stored in the database (they apply to new jobs; reprocess a video to apply them to it).

Static settings come from environment variables or `backend\.env` (see `backend\.env.example`):

| Variable | Default | |
|---|---|---|
| `MMA_DATA_DIR` | `backend/data` | Videos, thumbnails, frames, previews, logs, SQLite DB |
| `MMA_FFMPEG_PATH` / `MMA_FFPROBE_PATH` | auto | Explicit binaries; otherwise PATH, then bundled imageio-ffmpeg |
| `MMA_WORKER_THREADS` | `1` | Parallel processing jobs |
| `MMA_LOG_LEVEL` | `INFO` | Logs also go to `data/logs/app.log` (rotating) |

## 7. Project structure & architecture

```
mma-fight-intel/
├─ setup.bat · start.bat · start-dev.bat · run-tests.bat
├─ backend/
│  ├─ run.py                     # uvicorn launcher
│  ├─ requirements.txt
│  ├─ app/
│  │  ├─ main.py                 # app factory, lifespan, CORS, upload-size guard, serves frontend/dist
│  │  ├─ core/                   # config (pydantic-settings), logging
│  │  ├─ db/                     # SQLAlchemy engine/session, models
│  │  ├─ schemas/                # Pydantic request/response models
│  │  ├─ api/routes/             # videos, fighters, jobs, system (stats/settings/health)
│  │  ├─ services/
│  │  │  ├─ storage.py           # safe server-generated paths under data/, deletion, orphan sweep
│  │  │  ├─ video_service.py     # upload validation (ext + container signature + size + SHA-256 dedupe)
│  │  │  ├─ media_probe.py       # FFprobe / ffmpeg-banner / OpenCV metadata
│  │  │  ├─ ffmpeg_tools.py      # locate FFmpeg binaries
│  │  │  ├─ job_manager.py       # persistent background job queue (threads)
│  │  │  └─ settings_service.py
│  │  └─ pipeline/
│  │     ├─ base.py              # ProcessingContext, Stage protocol
│  │     ├─ runner.py            # JOB_TYPES registry + PipelineRunner
│  │     └─ stages/              # probe, thumbnail, frame_sampling, quality, preview
│  ├─ tests/                     # pytest suite (generates real test videos)
│  └─ data/                      # created at runtime (gitignored)
└─ frontend/
   ├─ package.json · vite.config.ts · tsconfig.json
   └─ src/
      ├─ api/                    # typed API client
      ├─ context/                # upload queue, toasts
      ├─ components/             # layout, UI primitives
      ├─ pages/                  # Overview, Library, Upload, VideoDetail, Fighters, Jobs, Settings
      └─ lib/, hooks/
```

**Data model:** `videos`, `fighters`, `video_fighters` (fighters appearing in a video), `videos.target_fighter_id` (the one to analyze), `processing_jobs` (one row per attempt with stage log, warnings and a snapshot of the settings used), `sampled_frames` (index, timestamp, image path, stats), `app_settings`.

**Extending in later phases** — without restructuring:
- Add a new job type in `pipeline/runner.py` → `JOB_TYPES["phase2_detection"] = lambda: [DetectionStage(), PoseStage()]`. Stages receive the original video path, metadata and the sampled frames via `ProcessingContext`. The job queue, status tracking, retries and UI work unchanged.
- New result tables (detections, poses, strikes, predictions) reference `videos.id`, `sampled_frames.id` and `fighters.id` by foreign key. `target_fighter_id` + `target_fighter_notes` tell Phase 2 which athlete to track.
- The API layer is separate from processing, so a GPU worker process could later take over `job_manager` without touching the routes.

## 8. Tests

```powershell
run-tests.bat
# or
cd backend
.venv\Scripts\python.exe -m pytest -v
```
The suite generates real MP4/AVI/MOV files with OpenCV/FFmpeg and covers: multi-file upload, metadata accuracy (resolution, FPS, frame count, duration, codec, audio), thumbnails, frame sampling modes and caps, preview proxy and HTTP Range streaming, duplicate detection, invalid/spoofed files, corrupted video → failed job → retry, size limit, rename/search/filter/sort, delete (including mid-processing), fighter CRUD and assignment, the FFmpeg-without-ffprobe and OpenCV-only fallbacks, orphan cleanup, and **persistence across an application restart**.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `API offline` in the sidebar | Start the backend (`start.bat`), check nothing else uses port 8000 (`run.py --port 8010` + set `VITE_API_TARGET=http://127.0.0.1:8010` for dev). |
| Video won't play in the browser | Expected for AVI and some MOV/HEVC files until processing creates the MP4 preview. Ensure *Generate browser preview* is on in Settings and click **Reprocess**. You can always download the original. |
| "FFmpeg not found" in Settings | Re-run `pip install -r requirements.txt` (installs the bundled FFmpeg) or install FFmpeg and add it to PATH / `MMA_FFMPEG_PATH`. |
| A video fails with "could not decode" | The file is truncated/corrupted or uses a codec OpenCV can't decode. Re-encode it (`ffmpeg -i in.mkv -c:v libx264 out.mp4`) and upload again. |
| PowerShell blocks scripts | Use the `.bat` files, or call `.venv\Scripts\python.exe` directly as shown above (no activation needed). |
| Start over | Stop the app and delete `backend\data\`. |

## 10. Known limitations

Reported plainly, as requested:

- **Verified on Linux, not on a physical Windows 11 machine.** The full backend test suite (21 tests) and a browser end-to-end run (upload → process → library → details → fighter assignment → rename → retry → delete → restart persistence) passed with Python 3.13 / Node 22. The code is written for Windows (pathlib everywhere, no POSIX-only calls, hidden FFmpeg console windows, retrying deletes for Windows file locks, bundled FFmpeg, non-ASCII-safe image writes, CRLF `.bat` scripts), but the `.bat` scripts themselves have not been executed on Windows.
- Testing used generated test-pattern videos, not real UFC broadcast footage. Very long (2–3 h) or unusual files may surface edge cases.
- Quality assessment is a heuristic (sharpness/exposure/resolution/bitrate), not a learned model.
- The job queue runs inside the API process (threads; state persisted in SQLite). Fine for a single-user desktop setup; a separate worker (e.g. Celery/RQ) would be the next step for GPU-heavy Phase 2+ workloads.
- Schema is created with `create_all`; add Alembic migrations before changing existing tables in later phases.
- No authentication — intended for local use on `127.0.0.1`.
- Uploads are buffered by the multipart parser to a temp file before being moved into storage, so very large uploads temporarily need about twice their size in free disk space. Requests larger than the configured limit are rejected up front via `Content-Length`.
