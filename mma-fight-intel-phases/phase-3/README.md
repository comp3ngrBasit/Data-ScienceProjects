# MMA Fight Intelligence — Phases 1–3

AI-Powered MMA Fight Intelligence & Next-Move Prediction System.

* **Phase 1 — foundation:** upload UFC/MMA videos, video library, preview, fighter profiles, background processing into metadata, thumbnails, sampled frames and basic quality measures.
* **Phase 2 — computer vision:** person detection with a pretrained **YOLO Pose** model, multi-object tracking with **ByteTrack** or **BoT-SORT**, 17 body landmarks per tracked person, movement analysis (NumPy/Pandas), annotated videos, a frame-by-frame **Fighter Tracking** page, and manual linking of track IDs to fighter profiles.

* **Phase 3 — action recognition:** an MMA action taxonomy (all 33 requested categories + Unknown), a manual **annotation tool**, CSV/XLSX import (BoxingVI format), a training-dataset builder, **Random Forest / XGBoost / SVM** baselines and **Bi-LSTM / TCN** temporal models, evaluation split **by video**, sliding-window action detection per fighter track, review/correction of detections, and chronological **action sequences** for Phase 4.

> **Read this first — Phase 3 model quality.** The system is complete, but a model is only as good as its labelled data. No suitable labelled MMA action dataset could be downloaded in the build environment (see [Phase 3 data](#phase-3-training-data--evaluation-results)), so the included evaluation was done on **20 hand-labelled real strikes** and is a pipeline check, *not* a usable strike recogniser. You train real models with the annotation tool and/or BoxingVI.

> **Not implemented yet (by design):** pattern recognition (Phase 4), prediction (Phase 5). The system never decides on its own who the fighters are — it *suggests* likely fighter tracks and you confirm them.

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
| PyTorch | installed by `setup.bat` | Comes with Ultralytics. On Windows this is the **CPU** build; run `setup-gpu.bat` for NVIDIA GPUs (see below). |
| Internet | first setup only | pip/npm packages and the pretrained pose weights (`yolo26n-pose.pt`, ~8 MB, from the official Ultralytics GitHub releases). |

Disk space: originals are stored as-is, plus small thumbnails, sampled JPEG frames (bounded by settings) and, for AVI/MOV/non-H.264 files, a compressed MP4 preview.

## 2. Install on Windows 11

### Option A — one click
1. Unzip the project to a folder **without non-English characters in the path** (e.g. `C:\Projects\mma-fight-intel`). OpenCV on Windows can fail to open files under non-ASCII paths.
2. Double-click **`setup.bat`**. It will:
   - create a virtual environment in `backend\.venv`
   - install Python packages from `backend\requirements.txt`
   - download the default pretrained pose model into `backend\data\models\`
   - run `npm install` and `npm run build` in `frontend\`

   The first install downloads PyTorch (several hundred MB) and can take a few minutes.

### GPU acceleration (optional, NVIDIA)
Analysis runs on the CPU by default and **automatically uses CUDA** when a CUDA-enabled PyTorch sees a GPU (Settings → Computer vision → Device = `auto`).
To get a CUDA build of PyTorch, run **`setup-gpu.bat`** after `setup.bat` (it installs from `https://download.pytorch.org/whl/cu132`; use `cu126` for older drivers). Check the result on the **Settings** page — it shows the PyTorch version, whether CUDA is available and the GPU name. Apple Silicon (MPS) is detected too.

### Option B — manual (PowerShell)
```powershell
cd C:\Projects\mma-fight-intel

# Backend
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd backend; .venv\Scripts\python.exe download_models.py; cd ..   # optional: pre-download pose weights

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
| **Overview** | Totals (videos, fighters, processed, awaiting), uploads over the last 14 days, library status, storage use, recent uploads and recent processing activity. All numbers come from the database. **Upload Fight Footage** button. A **Computer Vision Analysis** section shows detected fighters, frames analysed, detection/pose confidence, tracking consistency, pose-extraction job status, live progress and an annotated-video preview. |
| **Video Library** | Grid or table of videos with thumbnail, title, duration, resolution, frame rate, file size, upload date and status. Search (title/filename), filter by status or fighter, sort, rename, delete, process/retry. |
| **Upload Videos** | Drag-and-drop or browse for multiple MP4/MOV/AVI files. Per-file progress, cancel, retry. Files are checked client-side (type, size) and server-side (extension, container signature, size, duplicate content via SHA-256). Uploads keep going while you use other pages. |
| **Video details** | Player (seekable; uses an MP4 preview for formats browsers can't play), full metadata, assigned fighters + **fighter to analyze** + identification notes, processing status with live stage progress, sampled frames with timestamps (click to seek) and a per-frame sharpness/brightness/contrast chart, basic quality assessment, and processing history (every attempt, stage timings, errors, warnings). |
| **Fighters** | Create/edit/delete profiles: name, nickname, stance (Orthodox / Southpaw / Unknown), height, reach, weight class, notes. Each profile lists its associated videos and which ones it is the analysis target in. |
| **Processing Jobs** | All jobs with status filters (Queued / Processing / Completed / Failed / Cancelled), live progress, stage breakdown, error messages, **Retry** and **Retry all failed**. |
| **Fighter Tracking** *(Phase 2)* | Pick a processed video and click **Analyse**. Then: play the **original video with a live overlay** (boxes, track IDs, skeletons, landmarks — each toggleable, plus “fighters only”) or the **annotated MP4**; step frame by frame (← →, Shift+← → for analysed frames, Space); a **timeline** shows when each track ID is visible, its interruptions and detected camera cuts; the **Tracks → fighters** panel lets you mark each track as fighter / referee / other and link it to a fighter profile (several tracks can belong to the same fighter); the **frame inspector** lists every person in the current frame with confidence, 17 landmarks, joint speeds, angles, state and the distance between fighters; **movement charts** plot wrist/elbow/shoulder/hip/knee speed, joint angles, guard height, lean and posture change over time. Export detections, pose, movement, distances and tracks as CSV, or a NumPy `.npz` bundle. |
| **Action Recognition** *(Phase 3)* | Original / annotated player with skeleton overlay and a live “current action” badge; **detected-action timeline** per fighter (colour = offensive/defensive/movement/unknown, shape = punch/kick/knee-elbow/grappling/defence; dashed = low confidence; annotations shown underneath) — click any marker to jump there; **Detected actions** list per fighter with model confidence, validated precision, runner-up class and ✓ / ✎ / ✕ review buttons (corrections become training data); **Sequences** (Jab → Cross → … with gaps in seconds, from detections or from your annotations; JSON/CSV export); **Charts** (action frequency, offensive vs defensive per fighter); **Annotate** (play/pause, *Set start* = I, *Set end* = O, category, fighter, track, your certainty, opponent reaction, status; edit/delete; mark the video fully annotated; choose train/test split; import CSV/XLSX). Repeated combinations are listed per fighter. |
| **Action Models** *(Phase 3)* | Dataset preview (examples per class, how many videos they come from, which classes are trainable), train & evaluate any of RF/XGBoost/SVM/Bi-LSTM/TCN in the background, model versions with activate/delete, and full evaluation: accuracy with 95 % CI, majority-class baseline, macro/weighted precision/recall/F1, per-class table, precision at the confidence threshold, confusion matrix. Inference/training settings live here. |
| **Settings** | Frame sampling (fixed interval or N evenly spaced frames, max frames per video, stored width, JPEG quality, store images or stats only), auto-process on upload, preview generation, max upload size, thumbnail width. Shows FFmpeg/OpenCV status and storage use. **Computer vision:** pose model (yolo26 / yolo11, n…x), device (auto/cpu/cuda/mps), FP16, inference size, analysis rate (frames per second of video), detection & landmark confidence, tracker (ByteTrack/BoT-SORT), occlusion memory, minimum person height (filters the crowd), max people per frame, smoothing, gap filling, stationary threshold, annotated-video on/off. Shows PyTorch/CUDA status. |

### Processing pipeline
Each job runs these stages in a background worker thread (the UI never blocks):

1. **Metadata** — FFprobe (or `ffmpeg -i` output if only the bundled FFmpeg exists) + OpenCV: duration, frame count, FPS, resolution, rotation, video codec, pixel format, container, bitrate, audio presence/codec/sample rate/channels. OpenCV must be able to decode frames, otherwise the job fails as corrupted/unsupported.
2. **Thumbnail** — the sharpest well-exposed of several candidate frames.
3. **Frame sampling** — configurable; stores JPEGs (optional) plus frame index, timestamp, brightness, contrast and sharpness per frame. Every extracted frame is **not** stored — only the sampled subset, capped by *max frames*.
4. **Quality** — heuristic score from resolution, FPS, bitrate, blur %, dark/overexposed %, decode failures.
5. **Preview** — for AVI, MOV or non-H.264 files, an H.264 MP4 preview (≤ 1280 px wide) for browser playback. The original is never modified.

Statuses: **Uploaded → Queued → Processing → Completed / Failed**. Failed jobs show the reason and can be retried; each retry is a new attempt in the history. Jobs that were running when the app stopped are marked failed ("Interrupted") on next start and can be retried; queued jobs resume automatically.

### Phase 2 computer-vision pipeline
A separate background worker (so long analyses never block uploads or Phase 1 processing) streams the video frame by frame — it is never loaded into memory — and:

1. **Detects people + pose in one pass** with a pretrained Ultralytics **YOLO Pose** model (COCO-17 keypoints; default `yolo26n-pose`). Only the *person* class is used; tiny/background people are filtered by a minimum height, and only the N largest people are kept per frame.
2. **Tracks** them with **ByteTrack** (default) or **BoT-SORT** via Ultralytics. The “occlusion memory” keeps a lost ID alive for N seconds so short occlusions don’t create new IDs. Keypoints stay attached to their tracked box, so pose is extracted **per tracked person**.
3. Runs on every *k*-th frame (configurable **analysis rate**, default 10 analysed frames per second of video).
4. Writes detections and landmarks to SQLite in **batches** (bulk `executemany` inserts).
5. Detects **camera cuts** (colour-histogram change) — IDs always change across cuts in broadcast footage.
6. Summarises each track (first/last frame, coverage, interruptions and visible segments, average detection/pose confidence) and **suggests** likely fighter tracks: large, long-lived, confident, and at most two on screen at once. A referee can look the same to this heuristic — **you confirm roles manually**.
7. Computes **movement features** with Pandas: landmarks below the confidence threshold become *missing* (never zero), short gaps are interpolated, a centred rolling mean smooths jitter, then per frame: wrist/elbow/shoulder/hip/knee/ankle speeds and body speed (**torso lengths per second**, so they are comparable across resolutions and zoom), movement direction, elbow/knee angles, torso lean, stance width, guard height (hands vs shoulders), posture change, and a state: `moving`, `stationary`, `low_confidence` or `missing`. Distances between the likely fighters (in torso lengths and fraction of frame height) and closing speed are stored per frame.
8. Writes an **annotated H.264 MP4** (boxes, ID labels, skeletons; between analysed frames the last result is held) and a **NumPy `.npz` bundle** of per-track keypoint sequences for Phase 3.

Re-analysing a video keeps the previous results until the new run succeeds. Jobs survive restarts (queued jobs resume; interrupted jobs are marked failed and can be retried). Only one analysis per video can be queued or running.

**Measured speed in development (2-core CPU, no GPU, `yolo26n-pose`, 640 px):** ~75–100 ms per analysed frame → a 1080p video analysed at 10 fps took ~0.5–0.7× real time (an 8-second clip ≈ 15 s, a 20-second clip ≈ 28 s, including the annotated video). A CUDA GPU is typically many times faster, but this was **not measured** — no GPU was available. Real-time processing is not promised.

### Phase 3 action-recognition pipeline
**Labelling → dataset → models → detection → sequences.**

1. **Annotations** (UI, API or CSV/XLSX import): video, fighter, optional track ID, category, start/end time, annotator certainty, optional opponent reaction, status (`verified` are used for training). Per video you can declare it *fully annotated* (so unlabelled moments become “no action” examples) and put it in the train or hold-out test split. Reviewing a detection (✓/✎) writes a `correction` annotation automatically.
2. **Dataset builder** resolves each annotation to a Phase 2 track (explicit track → tracks linked to the fighter → the only person in view → a clearly dominant person), cuts a window with context around it, and turns the pose sequence into features: hip-centred, torso-normalised joint positions and velocities, elbow/knee/shoulder/hip angles, limb extension and height, and landmark visibility, resampled to a fixed number of steps. Temporal-jitter copies augment training only. Background windows come from fully-annotated videos, away from any label.
3. **Models:** RF, XGBoost and SVM on window statistics (mean/std/min/max/change/peak position per feature — an initial benchmark); **Bi-LSTM with attention** and a **dilated TCN** on the full sequence (they learn temporal order, not static poses). PyTorch models use CUDA when available. Classes with too few examples or examples from too few different videos are **not trained** and are listed as such.
4. **Evaluation is split by video/fight group** (hold-out test videos if you mark any, otherwise leave-one-group-out or 5-fold GroupKFold). Augmented copies never reach a test fold. Reported: accuracy (+95 % Wilson CI and majority baseline), macro & weighted precision/recall/F1, per-class metrics, confusion matrix, and per-class **precision at the confidence threshold**.
5. **Detection:** a window slides over every fighter track (length learned from the training data, hop configurable). Windows are skipped when the fighter is nearly still (*motion gate*), when too few landmarks are visible, or when they span a camera cut. A label is reported only if its probability passes the threshold and beats “no action”; fast movement the model can't classify becomes **Unknown / Unclassified**; everything else is left unlabelled — ordinary arm movement is not forced into a punch. Overlapping same-label windows merge, overlapping different labels are de-duplicated (temporal NMS), and start/end are refined from the active limb's speed profile.
6. **Every action record** has: video, fighter track ID (and fighter via track assignment), category, start/end time, duration, frame range, **model confidence** (the model's own probability), **validated precision** (+ number of held-out predictions it is based on), runner-up class, active limb, model version and review status. Confidence and validated precision are deliberately separate.
7. **Sequences:** per fighter (all their tracks merged), chronological, split into bursts after a configurable pause; each stores the ordered actions and the **time intervals** between them. Unknown, rejected and unreviewed low-confidence detections are excluded; corrected labels replace model labels. Repeated 2- and 3-action combinations are counted for Phase 4.

Because strikes last 0.2–0.5 s, run Phase 2 with **Analysis rate 25 fps or 0 (every frame)** on videos you annotate or analyse for actions; the app warns when pose was sampled too sparsely.

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
| **POST** | **`/api/vision/{video_id}/analyze`** | Queue detection + tracking + pose analysis (202; 409 if already queued/running or Phase 1 not finished) |
| **GET** | **`/api/vision/{video_id}/status`** | Progress (stage, %), current results summary, history |
| **GET** | **`/api/vision/{video_id}/detections`** | Boxes (pixel + normalised), confidence, frame, timestamp, track ID — filters: `track_id`, `fighter_id`, `fighters_only`, frame range, `min_confidence`, paging |
| **GET** | **`/api/vision/{video_id}/pose`** | Landmarks (normalised x/y, confidence) per tracked person; `format=grouped\|long`, `landmarks=left_wrist,right_wrist`, same filters |
| **GET** | **`/api/vision/{video_id}/tracking`** | Tracking sessions per ID (segments, coverage, interruptions, role, assigned fighter) + consistency stats + camera cuts |
| **GET** | **`/api/vision/{video_id}/annotated-video`** | Annotated MP4 (Range supported; `download=true`) |
| **POST** | **`/api/vision/{video_id}/assign-fighter`** | `{"assignments":[{"track_id":1,"fighter_id":"…"},{"track_id":3,"role":"referee"}]}` |
| **DELETE** | **`/api/vision/{video_id}/analysis`** | Remove all analyses, results and files for the video (cancels a running one) |
| GET | `/api/vision/{video_id}/movement` · `/distances` · `/frame/{n}` · `/overlay` | Movement features, fighter distances, everything about one frame, compact overlay data |
| GET | `/api/vision/{video_id}/export?kind=detections\|pose\|movement\|distances\|tracks\|npz` | CSV (Pandas) or NumPy bundle |
| GET / POST | `/api/vision/jobs`, `/api/vision/jobs/{job_id}/retry` | All vision jobs; retry failed/cancelled |
| GET / PUT | `/api/vision/settings`, GET `/api/vision/system`, `/api/vision/overview` | Settings, PyTorch/CUDA/model info, dashboard stats |
| **POST** | **`/api/actions/{video_id}/analyze`** | Recognise actions (background; 409 if no model, no Phase 2 results, or already running). `?model_version=` optional |
| **GET** | **`/api/actions/{video_id}/results`** | Action records — filters `track_id`, `fighter_id`, `category`, `group`, `min_confidence`, `include_unknown` |
| **GET** | **`/api/actions/{video_id}/timeline`** | Per-fighter lanes with detections and annotations |
| **GET** | **`/api/actions/{video_id}/sequences`** | Fighter-specific sequences with intervals (`source=model\|annotations`, optional `gap`) |
| **POST** | **`/api/actions/{video_id}/annotations`** | Create annotation (GET lists them) |
| **PUT / DELETE** | **`/api/actions/annotations/{annotation_id}`** | Correct / delete an annotation |
| **GET** | **`/api/actions/models/status`** | Models, active model, training jobs |
| **POST** | **`/api/actions/models/train`** | Train & evaluate (background) — `{"algorithms":["random_forest","xgboost","svm","lstm","tcn"]}` |
| **GET** | **`/api/actions/models/evaluation`** | Metrics of the active model (or `?version=`) |
| POST | `/api/actions/actions/{action_id}/review` | `{"decision":"confirm\|correct\|reject","category":...}` |
| POST | `/api/actions/{video_id}/import` | CSV/XLSX import (`units=frames\|seconds`, `label_map=boxingvi_orthodox\|boxingvi_southpaw`) |
| PUT | `/api/actions/{video_id}/dataset-meta` | `fully_annotated`, `split` (auto/train/test/exclude), `split_group` |
| GET | `/api/actions/{video_id}/status` · `/export` · DELETE `/results` | Job status/history, JSON/CSV export, delete results (annotations kept) |
| GET / POST / DELETE | `/api/actions/taxonomy`, `/dataset`, `/settings`, `/overview`, `/models/{v}/activate`, `/models/{v}` | Categories (+ which are trained), dataset preview, settings, dashboard, model management |


## Phase 3 training data & evaluation results

### Datasets — what was available
| Source | Contents | Could it be used here? |
|---|---|---|
| **BoxingVI** (Kumar et al., 2025) | 6,915 punch clips, 6 classes (jab, cross, lead/rear hook, lead/rear uppercut), 20 YouTube videos, temporal boundaries | Hosted on **Google Drive**, not reachable from the build environment; no licence file in the repository. **Supported by the importer** — see below. |
| **fight-judge** | 5,106 UFC frames with fighter boxes / keypoints (CC BY-NC-SA 4.0) | Hosted on Mendeley (unreachable). It has **no action labels**, so it cannot train an action classifier. |
| **UCF101** | 101 general human actions (incl. boxing bag/speed bag) | Unreachable, and a general action dataset is not a substitute for MMA strike labels. |
| **GA_Capstone “MMA strikes”** (GitHub) | 144 “clips” of 6 strike classes | Inspected and **rejected for training**: every clip is a *single still photo repeated for 24 frames* (no motion), and 8 “right high kick” clips are exact duplicates of “left high kick” clips. |
| GA_Capstone moving demo videos (real UFC/ONE footage, single fighter masked) | 71 one-second segments | After removing duplicates and mirrored copies, **11 distinct real strikes** (2 high kicks, 3 low kicks, 6 punches). Labelled by hand (start/end at ≈ ±3 frames) and used — with each strike's mirror copy kept in the same group — for the pipeline check below. Punches are labelled *Punch (type unspecified)* because jab/cross/hook could not be told apart reliably from those clips. |

### Results on that tiny real dataset (pipeline check)
20 labelled strikes from 11 videos + 7 “no action” windows from one non-fighting video, evaluated with 5-fold group cross-validation (split by video), 27 held-out predictions:

| Model | Accuracy (95 % CI) | Macro F1 (action classes) | Macro precision | Macro recall |
|---|---|---|---|---|
| **TCN** (selected) | 44 % (28–63 %) | 59 % | 56 % | 64 % |
| Bi-LSTM | 44 % (28–63 %) | 55 % | 51 % | 64 % |
| XGBoost | 37 % (22–56 %) | 47 % | 44 % | 58 % |
| SVM (RBF) | 37 % (22–56 %) | 44 % | 45 % | 44 % |
| Random Forest | 30 % (16–48 %) | 31 % | 28 % | 36 % |

Always predicting the most common class scores 37 %. The temporal models did best, but **with 27 test predictions the confidence intervals overlap almost completely — these numbers do not show that any model works, only that training, group-split evaluation and inference run correctly.** The “no action” class came from a single video, so in every fold where it was held out the model had never seen it (0 % recall). Expect a usable model to need on the order of **hundreds of examples per class from dozens of different fights**.

### Training a real model
1. **BoxingVI (recommended start):** download it from the link in its GitHub README. For each RGB video: upload it, run Fighter Tracking with *Analysis rate = 0 (every frame)*, then in Action Recognition → Annotate → *Import*, choose that video's `.xlsx`, answer **frames** and **BoxingVI mapping**. (Lead/rear is mapped to left/right assuming an orthodox stance; for southpaw athletes use `label_map=boxingvi_southpaw` via the API.) Set each video's *split group* to the athlete/session and mark the subjects BoxingVI uses for validation (S16–S20) as **test**. Note BoxingVI is boxing, not MMA: it teaches punch types, not kicks, knees or takedowns.
2. **Your own UFC footage:** annotate in the UI (I/O keys), assign each label to the right fighter, mark videos fully annotated when every action of that fighter is labelled, and keep whole fights together in one split group.
3. Action Models → **Train & evaluate**. Check the dataset table first: classes that lack examples are shown as not trainable.
4. Analyse videos, review detections (✓/✎/✕) and retrain — corrections become new training data.

## 6. Configuration

Runtime settings are edited on the **Settings** page and stored in the database (they apply to new jobs; reprocess a video to apply them to it).

Static settings come from environment variables or `backend\.env` (see `backend\.env.example`):

| Variable | Default | |
|---|---|---|
| `MMA_DATA_DIR` | `backend/data` | Videos, thumbnails, frames, previews, logs, SQLite DB |
| `MMA_FFMPEG_PATH` / `MMA_FFPROBE_PATH` | auto | Explicit binaries; otherwise PATH, then bundled imageio-ffmpeg |
| `MMA_WORKER_THREADS` | `1` | Parallel processing jobs |
| `MMA_LOG_LEVEL` | `INFO` | Logs also go to `data/logs/app.log` (rotating) |
| `MMA_MODELS_DIR` | `data/models` | Where pretrained weights are cached |

## 7. Project structure & architecture

```
mma-fight-intel/
├─ setup.bat · setup-gpu.bat · start.bat · start-dev.bat · run-tests.bat
├─ backend/
│  ├─ run.py                     # uvicorn launcher
│  ├─ download_models.py         # pre-download pose weights
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
│  │  ├─ actions/                # Phase 3: taxonomy, features, data (dataset builder), models (RF/XGB/SVM/LSTM/TCN),
│  │  │                          #   training (group-split evaluation), inference (sliding window, NMS, sequences),
│  │  │                          #   importer (CSV/XLSX, BoxingVI), job_manager, settings, api
│  │  ├─ vision/                 # Phase 2 (no web/DB code in the reusable parts)
│  │  │  ├─ pose_tracker.py      # PoseTracker: YOLO pose + ByteTrack/BoT-SORT, reusable per frame
│  │  │  ├─ movement.py          # pure NumPy/Pandas movement features (reusable for Phase 3)
│  │  │  ├─ analyzer.py          # streams a video: detect → track → pose → batch writes → movement → annotated MP4 → .npz
│  │  │  ├─ annotate.py          # drawing + FFmpeg H.264 writer
│  │  │  ├─ runtime.py           # CUDA/MPS/CPU selection, weight download/cache
│  │  │  ├─ job_manager.py       # separate persistent background queue for vision jobs
│  │  │  ├─ settings.py · constants.py · api.py
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
      ├─ pages/                  # Overview, Library, Upload, VideoDetail, Fighters, Tracking, Actions, ActionModels, Jobs, Settings
      ├─ components/vision/      # overlay player (canvas), timeline, track panel, frame inspector, charts
      ├─ components/actions/     # action timeline, history/review, sequences, charts, annotation panel
      └─ lib/, hooks/
```

**Phase 2 tables:** `vision_jobs` (status, progress, stage, timing, errors, settings snapshot, device, stats), `fighter_detections` (video, job, frame, timestamp, track ID, normalised box, confidence), `pose_landmarks` (video, job, track, frame, timestamp, landmark name/index, normalised x/y, confidence — all 17 COCO keypoints), `tracking_sessions` (track ID, assigned fighter, role, suggested role, first/last frame, coverage, interruptions, segments, status), `movement_metrics`, `fighter_distances`. These are new tables only — an existing Phase 1 database is upgraded in place on first start.

**Phase 3 tables:** `action_annotations`, `video_annotation_meta`, `action_models` (version, algorithm, dataset version, classes, accuracy/precision/recall/F1, per-class metrics, confusion matrix, training info), `action_training_jobs`, `action_analysis_jobs`, `recognized_actions`, `action_sequences` (sequence id, fighter, ordered actions, intervals). Model files live in `data/action_models/<version>/`. Rows are written with bulk inserts.

**Phase 4 hand-off:** `GET /api/actions/{id}/sequences` (or `/export?kind=sequences`) returns clean, chronological, fighter-specific sequences with intervals; `source=annotations` gives the same structure from verified human labels (ground truth).

**Phase 2 → 3 hand-off:** `PoseTracker` and `compute_track_movement` have no database/web dependencies; `GET /api/vision/{id}/pose?fighter_id=…` returns one athlete’s landmark sequence across all their tracks; the `.npz` bundle holds per-track arrays (`t{ID}_frames`, `t{ID}_keypoints` [n,17,2], `t{ID}_keypoint_conf`, `t{ID}_boxes`) plus metadata (fps, size, keypoint names).

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
**50 tests.** Phase 1 tests generate real MP4/AVI/MOV files with OpenCV/FFmpeg and covers: multi-file upload, metadata accuracy (resolution, FPS, frame count, duration, codec, audio), thumbnails, frame sampling modes and caps, preview proxy and HTTP Range streaming, duplicate detection, invalid/spoofed files, corrupted video → failed job → retry, size limit, rename/search/filter/sort, delete (including mid-processing), fighter CRUD and assignment, the FFmpeg-without-ffprobe and OpenCV-only fallbacks, orphan cleanup, and **persistence across an application restart**.

Phase 2 tests run the **real pretrained YOLO pose model** (no mocked detections):
* everywhere: endpoints, a video with no people (completes with zero tracks), Phase-1-first rule, duplicate-job prevention, failure → retry, CPU/CUDA device selection logic, and 9 unit tests of the movement maths (speeds in torso lengths/s, missing vs low-confidence vs stationary, gap filling, smoothing, direction, angles, distance/closing speed);
* with real footage (put clips in `backend\tests\footage\` or set `MMA_TEST_FOOTAGE_DIR`; skipped otherwise): two-fighter broadcast boxing (detection, tracking, 17 landmarks per person, skeleton-inside-box alignment check, movement, distances, manual assignment, annotated video, CSV/NPZ export, dashboard stats), an overhead amateur bout (hard case), a single-fighter UFC clip (re-analysis replaces results, delete analysis), a two-person outdoor clip (deleting a video removes its vision data; persistence across restart). Expected file names are in `tests/test_vision.py`.

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

**Phase 3**
- **No trained, validated MMA action model is shipped.** The included evaluation (20 real strikes, 3 classes) is far too small to establish accuracy; do not use its detections for decisions. Only categories present in the training data can be recognised — currently High Kick, Low Kick and Punch (type unspecified) in the test runs. All other categories exist in the taxonomy, annotation tool and pipeline but have no examples.
- **Feints, defensive actions and movement categories** (forward/backward/lateral movement, stance switch, clinch entry …) need labelled examples like any other class; they are not detected by hand-written rules, to avoid presenting rule outputs as AI recognition.
- Detection on broadcast footage with the test model labelled some boxing exchanges as “Punch (type unspecified)” and several fast movements as *Unknown*; without ground truth for that clip, their correctness is **unverified**.
- Pose-only features miss what pose can't see: gloves blocking at close range, strikes hidden behind the opponent, clinch and ground work. Frame-level visual (CNN) features and video transformers were investigated in the design but **not implemented** — they need far more labelled data than is available here; the feature/model interfaces are ready for them.
- Validated precision per class is only as reliable as the number of held-out predictions behind it (shown as *n*).
- Tests: 10 Phase 3 tests (unit + API + a real-footage train/evaluate/detect/review run) on top of the 40 earlier tests — **50 total, all passing**.

**Phase 2**
- **Tested footage:** a broadcast pro-boxing clip (with someone else’s graphics burned in), an overhead amateur boxing bout, masked single-fighter UFC clips, and an outdoor two-person clip — all real footage, found publicly on GitHub. **No full-length UFC broadcast was available to test**; please try your own fights and report anything odd.
- **Identity is not guaranteed.** IDs are stable while a fighter is clearly visible, but they switch or restart when fighters clinch, cross, overlap heavily, leave the frame, or at camera cuts (cuts are detected and shown on the timeline). In the 8-second broadcast test clip one boxer received two different IDs (across a camera cut) and the other kept one ID but with gaps while occluded. Assign every track of an athlete to the same profile; fighter-level API queries then merge them.
- **“Likely fighter” is a heuristic.** In the broadcast test the referee was among the suggested tracks. Always review roles.
- **Camera angle matters.** Pose models are trained on side/front views. On the **overhead** test footage recall was poor (often only 1–2 of 3 people found at default settings; `yolo26s-pose`, 960 px and a lower confidence help but do not fix it).
- Heavily occluded joints (clinch, cage, the far arm) get low confidence and show as gaps — by design they are not reported as zero movement.
- **GPU path not executed:** the CUDA selection code is unit-tested with a simulated GPU, but no GPU was available to run real inference on CUDA. CPU path is fully tested.
- Movement features are geometric measurements; nothing is classified as a punch/kick yet (Phase 3).
- Weights are downloaded from GitHub on first use (or by `setup.bat`); without internet on first run, analysis fails with a clear message.
- Ultralytics is AGPL-3.0 licensed — relevant if you ever distribute this commercially.

**Phase 1**

- **Verified on Linux, not on a physical Windows 11 machine.** The full backend test suite (50 tests) and browser end-to-end runs (Phase 1: upload → process → library → details → fighter assignment → rename → retry → delete → restart persistence; Phase 2: analyse → progress → overlay player → toggles → frame stepping → track assignment → annotated video → dashboard/jobs/settings) passed with Python 3.13 / Node 22 / PyTorch 2.15 (CPU) / Ultralytics 8.4. The code is written for Windows (pathlib everywhere, no POSIX-only calls, hidden FFmpeg console windows, retrying deletes for Windows file locks, bundled FFmpeg, non-ASCII-safe image writes, CRLF `.bat` scripts), but the `.bat` scripts themselves have not been executed on Windows.
- Testing used generated test-pattern videos, not real UFC broadcast footage. Very long (2–3 h) or unusual files may surface edge cases.
- Quality assessment is a heuristic (sharpness/exposure/resolution/bitrate), not a learned model.
- The job queue runs inside the API process (threads; state persisted in SQLite). Fine for a single-user desktop setup; a separate worker (e.g. Celery/RQ) would be the next step for GPU-heavy Phase 2+ workloads.
- Schema is created with `create_all`; add Alembic migrations before changing existing tables in later phases.
- No authentication — intended for local use on `127.0.0.1`.
- Uploads are buffered by the multipart parser to a temp file before being moved into storage, so very large uploads temporarily need about twice their size in free disk space. Requests larger than the configured limit are rejected up front via `Content-Length`.
