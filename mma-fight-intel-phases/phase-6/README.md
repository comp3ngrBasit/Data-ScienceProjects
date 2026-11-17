# MMA Fight Intelligence — Phases 1–6

AI-Powered MMA Fight Intelligence & Next-Move Prediction System.

* **Phase 1 — foundation:** upload UFC/MMA videos, video library, preview, fighter profiles, background processing into metadata, thumbnails, sampled frames and basic quality measures.
* **Phase 2 — computer vision:** person detection with a pretrained **YOLO Pose** model, multi-object tracking with **ByteTrack** or **BoT-SORT**, 17 body landmarks per tracked person, movement analysis (NumPy/Pandas), annotated videos, a frame-by-frame **Fighter Tracking** page, and manual linking of track IDs to fighter profiles.

* **Phase 3 — action recognition:** an MMA action taxonomy (all 33 requested categories + Unknown), a manual **annotation tool**, CSV/XLSX import (BoxingVI format), a training-dataset builder, **Random Forest / XGBoost / SVM** baselines and **Bi-LSTM / TCN** temporal models, evaluation split **by video**, sliding-window action detection per fighter track, review/correction of detections, and chronological **action sequences** for Phase 4.

* **Phase 4 — fighter pattern intelligence:** per-fighter chronological action sequences (with opponent action, distance and stance where available), combination discovery (2-, 3-, 4+-action), conditional probabilities under three explicitly separate definitions, first- to third-order **Markov** transition models with smoothing, **PrefixSpan** sequential-pattern mining, fighter tendency profiles, opponent-dependent analysis and fighter comparison — every number with its sample size, a 95 % Wilson interval and a reliability label. Background analysis, stored derived tables, incremental re-extraction, and a **Pattern Intelligence** dashboard with an interactive transition heatmap.

* **Phase 5 — AI next-move prediction:** a context-aware next-action engine trained on the actions, pose/movement, distance, opponent actions and fighter statistics from Phases 1–4 — **Markov baseline** (interpolated back-off, smoothing, minimum-sample thresholds), **XGBoost** (engineered context features), **GRU / LSTM** and a **Transformer** (positional encoding, padding masks) when there is enough data — compared on **unseen fights**, **unseen fighters** and a **temporal hold-out** with top-1/top-3, macro P/R/F1, log loss, Brier, ECE and confusion matrices, **temperature-scaled** probabilities, reliability and out-of-distribution warnings, supporting evidence, 2–3-action sequence predictions, offline prediction through uploaded footage with stored, never-edited predictions compared against the actual next action, a model registry, and an **AI Next-Move Prediction** dashboard.

* **Phase 6 — complete platform:** one **unified analysis pipeline** (*Run full analysis* on any video: processing → detection, tracking & pose → action recognition → pattern analysis → next-move prediction, each stage consuming the previous stage's stored output, up-to-date stages reused, no duplicate runs, interrupted runs recoverable), an **Overview Dashboard** with totals and pattern analytics computed from the database, **Fighter Intelligence** profiles (observed behaviour vs statistical patterns vs ML predictions, kept visibly separate; per-video comparison), a **Video Intelligence** page (original/annotated video with tracking & pose overlay, click-to-seek action and combination timelines, corrections, next-move panel at any timestamp with detected vs predicted actions on one timeline), **Model Evaluation** (action recognition, pattern recognition, next-move — held-out data only, sample counts, insufficient-data warnings), **Model Management** (versions, datasets, training, comparison, validation-gated activation, rollback, deployment history), fighter assignments preserved when a video is re-analysed, structured JSON logging, and the requested navigation.

> **Read this first — Phase 5 prediction quality.** Trained on the real **BoxingVI** annotations (5,004 next-punch examples from 10 continuous boxing sessions, every test athlete unseen), the models clearly beat the baselines: Transformer **65.7 % top-1 / 92.2 % top-3**, GRU (selected, lowest log loss) 58.7 % / 93.1 %, Markov baseline 49.5 %, most-frequent-class floor 11.2 % — see the [Phase 5 evaluation](#phase-5-evaluation--results). This shows punch-to-punch prediction generalising to new athletes; it does **not** show MMA next-move prediction (BoxingVI has no kicks, grappling, defence or opponent, and the app's own MMA footage gives only 12 examples).

> **Read this first — Phase 3 model quality.** The system is complete, but a model is only as good as its labelled data. No suitable labelled MMA action dataset could be downloaded in the build environment (see [Phase 3 data](#phase-3-training-data--evaluation-results)), so the included evaluation was done on **20 hand-labelled real strikes** and is a pipeline check, *not* a usable strike recogniser. You train real models with the annotation tool and/or BoxingVI.

> **Read this too — Phase 4 statistics are only as good as the actions behind them.** The pattern engine was verified with hand-counted sequences (see [Phase 4 verification](#phase-4-verification--results)). On the real footage available in the build environment there are only a few dozen recognised actions from short single-fight clips, so every real-data probability is correctly labelled *insufficient* or *single fight*. Meaningful fighter profiles need many recognised (ideally reviewed) actions from several fights per fighter.

> **Not implemented:** live webcam/stream prediction is prepared architecturally but **not** implemented or real-time tested — prediction runs on uploaded footage. There is no user authentication (the app is designed for single-user local use; see Known limitations). The system never decides on its own who the fighters are — it *suggests* likely fighter tracks and you confirm them.

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

### Upgrading an existing Phase 1–3 installation to Phase 4
Stop the app, unzip the new version over the old folder (your `backend\data\` folder — videos, database, models — is kept), then run **`setup.bat`** again: it installs the one new Python dependency (SciPy, used for the statistical tests) and rebuilds the frontend. On the next start the new Phase 4 tables are created automatically; nothing existing is changed. Open **Pattern Intelligence** in the sidebar.

**Phase 6:** same procedure (no new packages). The Phase 6 tables (`pipeline_runs`, `model_deployments`) are created on the next start; existing models stay active.

**Phase 5:** same procedure — unzip over the existing folder and run `setup.bat` again (no new packages are needed beyond those already installed: PyTorch, XGBoost, scikit-learn, SciPy). The Phase 5 tables are created on the next start. Then open **Next-Move Prediction → Models → Train & evaluate models**.

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

### End-to-end workflow (from footage to predictions)
1. **Video Library → Upload** fight videos (several at once).
2. Open a video → **Open full analysis** → **Run full analysis**. Watch the five stage chips: *Video processing → Detection, tracking & pose → Action recognition → Pattern analysis → Next-move prediction*. Stages whose results are already up to date are *reused*; a stage whose prerequisite model does not exist yet is *skipped* with the reason (never faked).
3. **Computer Vision Analysis**: assign each fighter's track(s) to a fighter profile (create profiles under *Fighter profiles*). Assignments survive later re-analysis.
4. **MMA Action Recognition**: annotate examples, train an action model (or **Model Management → Action-recognition models → Start training**). Run the full analysis again so actions, patterns and predictions are produced.
5. **Model Management → Next-move models → Start training** (optionally after importing BoxingVI sequences under *Next-Move Predictions → Models*). A new model becomes active only if it passes the validation checks.
6. Explore: **Overview Dashboard**, **Fighter Intelligence**, **Fighting Pattern Analysis**, **Next-Move Predictions**, and the video's **full analysis** page; check **Model Evaluation**.

### Navigation (Phase 6)
Overview Dashboard · Video Library · Computer Vision Analysis · MMA Action Recognition · Fighting Pattern Analysis · Next-Move Predictions · Fighter Intelligence · Model Evaluation · Model Management · Settings — plus *Upload videos*, *Fighter profiles* and *Processing jobs* in the sidebar's secondary section.

| Phase 6 page | What it does |
|---|---|
| **Overview Dashboard** | Fights analysed, videos processed / analysed, fighters analysed, detected actions (+ verified annotations), combinations identified, predictions generated (held-out top-1), model status of both model types; a fighters table (observed time, actions, top action, offensive share, combinations, predictions — click for the profile); pattern analytics for all fighters or one: most common combinations, sequence-length distribution, offensive/defensive transitions, action distribution, the transition heatmap and opponent-dependent patterns. Running pipelines are shown. The original footage-library panels follow below. |
| **Fighter Intelligence** | Pick a fighter. Profile, analysed time (union of the fighter's tracked spans), actions, mean/median time between actions, offensive vs defensive counts. Three colour-coded kinds of section: **Observed behaviour** (videos with per-video comparison and filtering, strike frequency, opening actions, counterattacks, pose-based movement tendencies, the historical action timeline — click an action to open the video at that moment), **Statistical patterns** (combinations with occurrences, fights, P(last \| prefix) and classification; tendencies; opponent-specific patterns with sufficiency labels) and **ML prediction** (the next-move panel, labelled as a prediction, never a confirmed action). |
| **Video full analysis** (`/videos/{id}/analysis`) | Run/force the full pipeline with live stage status; original or annotated video with detection, track and pose overlay; fighter selection; detected-action timeline (click to jump); combination timeline (each recurring combination of the fighter where it occurs in this video, with its overall count); **review & correct** recognised actions (confirm / correct / reject — saved as verified training data); **next-move panel** — *Predict at the playhead* uses only actions completed by that time: previous actions (solid) and top-3 predictions at an estimated time (dashed) on one timeline, click a prediction for the historical combinations supporting it, context (opponent's last action, distance, stance), reliability; stored predictions at the playhead with the actual outcome. |
| **Model Evaluation** | Action recognition: accuracy (CI), macro precision/recall/F1, per-action table, confusion matrix, sample counts. Pattern recognition: combination detection (detections vs verified annotations of the same videos: precision, recall, F1) and frequency-estimation error, split-half transition-probability reliability, leave-one-fight-out pattern coverage. Next move: training / validation / test sample counts per protocol, unique fights and fighters, Precision@3 / Recall@3, top-1/top-3, log loss, Brier, calibration, full model comparison. Insufficient-data warnings everywhere they apply. |
| **Model Management** | For both model types: active model with its validation checks and file location, roll back to the previous active model, training dataset summary, deployment history, all versions with held-out metrics and validation status, compare any versions side by side, activate a version (a version that fails validation needs an explicit override, which is recorded), start training. |

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
| **Pattern Intelligence** *(Phase 4)* | Fighter list (fighter profiles, plus unassigned tracks listed per video) with analysis state (*Analysed / Out of date / Not analysed*). Per fighter: **filters** (fight, date range, stricter detection confidence), warnings about data limits, summary cards, and tabs — **Tendencies** (most used attacks & defences with share, 95 % CI and rate per minute; attacks per minute; preferred attacking combinations; reactions to opponent attacks; typical pause between actions; sequence diversity; changes across fights with a chi-square homogeneity test), **Combinations** (every n-gram above the thresholds with occurrences, distinct fights, per-fight frequency, average gap, recognition confidence, % followed by another action, P(last \| prefix) with CI, lift, p-value and *Repeated pattern / Single fight / Possibly coincidental*; PrefixSpan patterns with support and confidence), **Transitions** (interactive heatmap — rows previous action(s), columns next action, 1st/2nd/3rd order, raw or smoothed, hover for counts and CI; most common and once-only transitions; conditional-probability table under the chosen definition), **Sequence explorer** (what precedes / follows a chosen action, top 3- and 5-action combinations containing it, every chronological sequence with timing, confidence, opponent action, distance and stance), **Opponents** (per-opponent statistics, observed opponent style, counter / defensive response rates, a difference test — findings only when supported), **Compare** (two fighters: action distributions, defensive tendencies, combinations, most-likely transitions, diversity, Jensen–Shannon distance, chi-square). **Fights & dates** groups videos into fights and sets dates; **Settings** holds all thresholds. **Run analysis / Analyse all** store results in the background. The Overview page has a *Fighter Pattern Intelligence* section. |
| **Next-Move Prediction** *(Phase 5)* | Choose a fighter. **Predict**: build a recent action sequence (plus opponent's last action and distance if known) → top 3 and full calibrated distribution, reliability (calibrated probability vs raw confidence, held-out accuracy at that confidence, matching history counts, warnings), supporting evidence, fighter intelligence, the effect of context, possible 2–3-action sequences, and *What actually happened?* to record the outcome. **Video timeline**: run predictions through an uploaded video, play it — the panel shows the prediction made at the last completed action, and the actual next action is revealed only when playback reaches it; the timeline marks actual actions (solid), predictions (dashed) and correct/wrong outcomes. **Evaluation**: held-out vs in-sample top-1/top-3 with CIs, performance over time, prediction history. **Models**: train & evaluate, comparison table under each protocol, calibration diagram, per-subset metrics, confusion matrix, training data, model registry (activate any version), settings and import of real sequences. The Overview page has an *AI Next-Move Prediction* section. |
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

### Phase 4 pattern-intelligence pipeline
**Actions → observations → sequences → statistics → stored profile.** The engine (`app/patterns/engine.py`) is pure Python/NumPy/SciPy over a list of observations; it does not touch videos, so it runs independently of the vision pipeline.

1. **Source of actions** (Settings → *Data source*): **best** (default) = verified annotations for videos marked *fully annotated*, model detections for every other analysed video (never both for one video, so nothing is double counted); or detections only / annotations only. Detections: reviewer corrections replace model labels, rejected and *Unknown* actions are dropped, unreviewed detections below the minimum confidence (default 0.6) or flagged low-confidence by Phase 3 are excluded (configurable). Phase 3 data is read, never modified.
2. **Attribution:** an action belongs to the fighter profile its track is assigned to (current assignment, so later track assignments apply retroactively); unassigned tracks become their own per-video "fighter" (`track-<video>-<id>`), because a track id can't be matched across videos. Annotations without a track use the video's *fighter to analyze*. Unattributable actions are counted and excluded.
3. **Context per action:** fight (videos with the same *fight key* = one fight; otherwise the Phase 3 split group; otherwise the video), fight date, opponent (the other fighter in the video; with several, the closest by Phase 2 distance at that moment), the opponent's most recent action (and its lag), distance between the fighters in torso lengths (nearest Phase 2 distance sample within 0.5 s) and the fighter's stance (profile).
4. **Cleaning:** actions are ordered by time per fighter and video; same-fighter detections overlapping by more than the IoU threshold (0.5) are merged, keeping the more confident one; identical multi-action sequences that appear in two different videos of the **same fight** (duplicate uploads / overlapping clips) are counted once.
5. **Sequences:** split after a pause longer than the sequence gap (2 s) — a video is never one long sequence. Each action carries its position, duration and the interval since the previous action.
6. **Combinations** (contiguous n-grams, n = 2 … 5): occurrences, distinct fights and videos, frequency per analysed fight, average gap, average duration, average recognition confidence, % followed by another action, **P(last \| prefix) = Count(prefix→last) / Count(prefix)** with Wilson CI, **lift** against how often the last action follows anything, and a one-sided **binomial test**. A combination is a *repeated pattern* only if it occurs in ≥ 2 fights and beats chance at the significance level; one-fight combinations are labelled *single fight*; the rest *possibly coincidental*. Minimum occurrences and minimum fights are configurable.
7. **Sequential patterns (PrefixSpan):** ordered patterns with gaps allowed (e.g. jab … cross with a slip in between), support = share of sequences containing it (each sequence counted once), confidence, distinct fights, average time between elements.
8. **Conditional probabilities** for orders 1–3 under three definitions that are computed and stored separately and never mixed: `next` (next action in the same sequence, `END` if it ends), `next_offensive` (next *offensive* action in the same sequence, `END` if none), `window` (next action starting within the observation window — default 1.5 s — after the previous one ends, `NONE` if nothing starts). Each value has count, context count, probability, Wilson 95 % CI, fights, coverage and a **reliability** label: *insufficient* (fewer than the minimum observations, default 5), *single fight*, *low*, *medium* (≥ 10 obs, ≥ 2 fights, CI width ≤ 0.5) or *high* (≥ 30 obs, ≥ 3 fights, CI width ≤ 0.3).
9. **Markov models:** transition matrices of order 1–3 (higher-order rows only when the context has enough observations) with raw and **additively smoothed** (Lidstone, α = 0.5) probabilities, start distribution, most likely next action per row, most common and once-only transitions, entropy rate.
10. **Tendencies:** offensive (most used attacks, attacks per minute of footage, average attacks per sequence, top attack-only combinations), defensive (most used defences, evasions, counter and defensive responses within 1 s of an opponent's attack, by attack type), behavioural (repeated patterns, pause distribution, action/transition entropy, per-fight action mix with Jensen–Shannon distance from the fighter's overall mix and a chi-square homogeneity test). Nothing is generalised from one fight — every view says so.
11. **Opponent analysis:** per opponent — fights, actions, action mix, offensive share, combinations, counter / defensive response rates with CIs — plus the opponent's *observed* style (aggressive / balanced / defensive from their recognised action mix, only with ≥ 10 actions). A between-opponent difference is reported only when a chi-square test with adequate expected counts supports it and at least two opponents have ≥ 10 of the fighter's actions.
12. **Comparison:** two fighters side by side, Jensen–Shannon distance and chi-square on their action distributions.
13. **Storage & performance:** *Run analysis* queues a background job (separate worker thread) that writes a full snapshot plus `action_transitions`, `fighting_combinations` and `fighter_tendencies`. Each video's extracted observations are cached with a signature of everything they depend on (actions, reviews, annotations, track assignments, fight info, stance) — unchanged videos are not re-extracted. Each fighter's **dataset version** hashes their observations and the settings; analysing an unchanged fighter is skipped, and a changed one shows *Out of date*. Unfiltered views use the stored snapshot when current; filtered views (fight, dates, confidence) and out-of-date fighters are computed on the fly, and the response says which (`meta.mode`).

### Phase 5 next-move prediction pipeline
**Training examples.** Every recognised/annotated action that is followed by another action of the same fighter *in the same sequence* (Phase 4 `next` definition) becomes one example: the input is everything known when that action **ends**, the target is the next action. Inputs:
* **recent action history** — the previous 2–10 actions (configurable) with durations, gaps and recognition confidence;
* **temporal context** — time since the previous action, mean gap, change in rhythm, position in the sequence, time since the sequence started;
* **opponent** — the opponent's most recent action that had *started* by the prediction moment, how long ago, and how many attacks they threw in the last 3 s;
* **physical / spatial (Phase 2)** — distance between the fighters (torso lengths) and closing speed from the last distance sample at or before the moment; mean speeds of centre, wrists, elbows, shoulders, hips, knees, ankles and movement direction over the preceding 0.5 s; stance from the fighter profile;
* **fighter statistics (Phase 4)** — the fighter's action distribution and follow-ups after the current action, computed **without the example's own fight** (so a fight never informs its own features).
Rarely seen actions (fewer than 5 examples, configurable) share an *Other (rare action)* class so probabilities always cover the complete distribution.

**Models.** *Markov baseline*: interpolated (Witten–Bell) back-off over orders 0–3, global then fighter-specific, contexts with too few observations get no weight, additive smoothing at order 0. *XGBoost*: the tabular features above with square-root class weighting. *GRU / LSTM*: action embeddings + per-step timing → recurrent encoder, concatenated with the normalised context vector (missing values zeroed with explicit missing-indicators). *Transformer*: the same inputs with learned positional encoding and padding masks. Sequence networks are trained only with ≥ 300 examples (Transformer ≥ 1,000; configurable). Every model is calibrated with **temperature scaling** fitted on held-back fights; early stopping uses the same held-back fights. CUDA is used automatically when available.

**Evaluation (no invented numbers).** (1) Fight-level cross-validation — GroupKFold over fights, every example predicted by models that never saw its fight; (2) unseen-fighter cross-validation; (3) temporal hold-out — earliest fights train, latest fights test. Reported per model: top-1 and top-3 accuracy (with Wilson CIs), macro precision/recall/F1, log loss, Brier score, expected calibration error + reliability diagram, confusion matrix, and breakdowns for known vs unseen fighters, offensive vs defensive next actions, with vs without opponent context, and fighter style; every model's top-1 difference from the Markov baseline comes with a 95 % interval from resampling whole fights; a class-frequency floor is always shown. The production model is the one with the **lowest held-out log loss** (rule fixed in advance); all architectures are kept in the registry and any can be activated. Models are **experimental** below 1,000 examples or 10 fights.

**Predictions.** A prediction returns the full calibrated distribution (sums to 1), the top 3, the model version and:
* **reliability** — calibrated probability vs the raw model confidence, the observed held-out accuracy of predictions with similar confidence (from the reliability diagram), normalised entropy, the number of matching histories in training (this fighter / all fighters, last 1–3 actions), training-set size, calibration metrics and a level (*high / medium / low / insufficient*);
* **warnings** — unseen fighter, rarely/never seen action histories, missing timing, context values outside the training range (out-of-distribution);
* **evidence** — matching historical combinations with counts and fights, the fighter's most frequent actions and usual follow-ups, how the current distance compares to the fighter's usual range, and what this fighter did after the same opponent action in training;
* **contextual mode** — the same prediction without opponent/position context and the largest probability changes;
* **short sequences** — beam search over 2–3 future actions (joint probabilities; hypothetical actions use the fighter's typical timing) — shown as possibilities, not outcomes.

**Offline video prediction.** *Predict through this video* replays the video's recognised actions in time order and stores a prediction after each one using only what was known at that moment. The actual next action is then compared in a **separate evaluation record** — stored predictions are never modified. Predictions on fights the model was trained on are flagged *in-sample* and reported separately from held-out performance. The timeline plays the video and reveals each actual next action only when playback reaches it.

**Live mode (prepared, not real-time).** `app/predictions/live.py` defines the frame-source and per-frame stage interfaces and implements the action-level part (`LivePredictionSession`, tested with a replay of recognised actions). Webcam/stream capture and incremental per-frame versions of the Phase 2/3 stages are not implemented; no real-time claim is made.

### Phase 6 integration
* **Unified pipeline** (`app/platform/pipeline.py`): one background run per video chains the existing phase jobs — Phase 1 processing, Phase 2 analysis, Phase 3 recognition with the active action model, Phase 4 pattern analysis (only changed fighters are recomputed), Phase 5 prediction through the video with the active prediction model. Each stage starts only after the previous one *completed*; failures stop the run with the stage's own error; runs interrupted by a shutdown are marked failed on restart (re-running reuses completed stages). One active run per video. Progress and stage messages are stored (`pipeline_runs`).
* **Assignments survive re-analysis:** when a video's tracking is re-run, fighter assignments and roles are copied to the new tracks whose time spans overlap the old assigned tracks by ≥ 60 % (same track id preferred); unmatched assignments are dropped, never guessed (count in the analysis stats).
* **Validation-gated deployment:** a newly trained model goes live only if it passes checks on held-out data — action model: a valid held-out evaluation, ≥ 20 evaluated samples, macro F1 ≥ 0.2, not more than 0.05 F1 below the active model; next-move model: ≥ 50 held-out predictions, lower log loss than the class-frequency floor, not more than 0.05 log loss worse than the active model. Otherwise the previous model stays active and the training job says why. Manual activation of a failing model requires `force=true` and is recorded in `model_deployments` (which also records automatic activations and rollbacks).
* **Caching & performance:** dashboard analytics are cached (30 s) and invalidated by a database fingerprint (counts / latest timestamps of actions, analyses, predictions, models, assignments); per-video pattern extraction is cached (Phase 4); videos are always read frame by frame (never fully loaded); heavy work runs in background jobs.
* **Structured logging:** set `MMA_LOG_FORMAT=json` to write one JSON object per log line (with fields such as `run_id`, `video_id`, `stage`); the default text format appends the same fields as `key=value`. Logs: `backend\data\logs\app.log`.
* **Reproducibility:** every stored prediction keeps its input sequence, context, full distribution, model version and in-sample flag; models are versioned with their dataset version and training configuration; training uses fixed seeds. Retraining on the same data in the build environment reproduced the selected GRU's held-out log loss (0.972 both times); top-1 differed by 0.2 points (58.7 % vs 58.5 %) because some CPU kernels are not bit-deterministic.

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
| **GET** | **`/api/patterns/{fighter_id}`** | Pattern summary: counts, fights, period, sources, reliability overview, warnings, top combinations, most likely next actions |
| **GET** | **`/api/patterns/{fighter_id}/combinations`** | n-gram combinations (+ `min_length`, `max_length`) and PrefixSpan patterns |
| **GET** | **`/api/patterns/{fighter_id}/transitions`** | Markov transition matrix, `order=1\|2\|3` (raw + smoothed, CI per cell) |
| **GET** | **`/api/patterns/{fighter_id}/probabilities`** | P(next \| previous): `definition=next\|next_offensive\|window`, `order`, or an explicit context `action=left_jab&action=right_cross` |
| **GET** | **`/api/patterns/{fighter_id}/tendencies`** | Offensive, defensive and behavioural tendencies |
| **GET** | **`/api/patterns/{fighter_id}/sequences`** | Chronological sequences with per-action context; `action=` adds the before/after explorer |
| **GET** | **`/api/patterns/{fighter_id}/opponent-analysis`** | Opponent-dependent statistics and difference test |
| **POST** | **`/api/patterns/{fighter_id}/analyze`** | Queue background analysis (skipped if unchanged; `force=true` to re-run) |
| **GET** | **`/api/patterns/{fighter_id}/analysis-status`** | Latest job, stored snapshot, current dataset version, state `processing\|up_to_date\|stale\|not_analysed` |
| **GET** | **`/api/patterns/compare?fighter_a=&fighter_b=`** | Fighter comparison |
| GET | `/api/patterns` | Fighters with data, analysis state, videos used, extraction statistics, recent jobs |
| GET / PUT | `/api/patterns/settings` | Pattern thresholds and data source |
| GET / PUT | `/api/patterns/fights` · `/api/patterns/fights/{video_id}` | Fight grouping (`fight_key`), `fight_date`, `event` per video |
| GET / POST | `/api/patterns/jobs` · `/api/patterns/analyze-all` | Job list; analyse every fighter (unchanged ones skipped) |

| **POST** | **`/api/predictions/{fighter_id}/predict`** | Body `{history: [actions or {category,start,end,confidence}], context?: {opponent_action, opponent_lag, distance, closing_speed, stance, movement}, mode: immediate\|contextual, save}` → top 3, full distribution, reliability, evidence, model version, `prediction_id` |
| **POST** | **`/api/predictions/video/{video_id}/predict`** | Background job: predict after every recognised action of an uploaded video (`fighter_key`, `model_id` optional) |
| **GET** | **`/api/predictions/{fighter_id}/history`** | Stored predictions with their evaluations |
| **GET** | **`/api/predictions/{fighter_id}/performance`** | Top-1 / top-3 of evaluated predictions (held-out vs in-sample, per model, over time) |
| **GET** | **`/api/predictions/{fighter_id}/model-status`** | Active model, whether the fighter is in the training data, classes |
| **POST** | **`/api/predictions/models/train`** | Background training + evaluation of all architectures |
| **GET** | **`/api/predictions/models/evaluation`** | Full evaluation report (all protocols, all models) |
| **GET** | **`/api/predictions/models/versions`** | Model registry |
| **POST** | **`/api/predictions/{fighter_id}/evaluate`** | Body `{prediction_id, actual_action}` — records the outcome once (409 if already evaluated) |
| **GET** | **`/api/predictions/{fighter_id}/next-sequence`** | `history=…&history=…&depth=2\|3` (+ `opponent_action`, `distance`) → possible future sequences |
| GET | `/api/predictions` · `/fighters` · `/models/jobs` · `/models/{id}` · `/video/{video_id}/timeline` | Overview, fighters, jobs, model detail, timeline of the latest video run |
| PUT / GET | `/api/predictions/settings` | History length, class threshold, folds, architectures, data thresholds, device (auto/cpu/cuda), beam width |
| POST / DELETE | `/api/predictions/models/{id}/activate` · `/api/predictions/datasets/import` · `/api/predictions/datasets/{name}` | Activate a model; import real action sequences without video (CSV/XLSX, e.g. BoxingVI sheets); delete an imported dataset |

| **GET** | **`/api/dashboard/overview`** | Totals across all phases, model status, recent pipeline runs |
| **GET** | **`/api/dashboard/fighters`** | One row per fighter (profiles + unassigned tracks with actions) |
| **GET** | **`/api/dashboard/analytics`** | Pattern analytics for all fighters or `?fighter=` |
| **GET** | **`/api/fighters/{fighter_id}/intelligence`** | Observed behaviour, statistical patterns, videos, movement; `?video=` (repeatable) to filter / compare |
| **GET** | **`/api/fighters/{fighter_id}/patterns`** | Combinations, Markov matrix, tendencies, opponent analysis |
| **GET** | **`/api/fighters/{fighter_id}/predictions`** | Next-move prediction from the latest actions; `?video_id=&at=` for a video timestamp; detected vs predicted timeline; 503 if model inference fails |
| **GET** | **`/api/models/evaluation`** | Action recognition, pattern recognition and next-move evaluation (held-out) |
| **GET** | **`/api/models/versions`** | All versions of both model kinds + deployment history |
| **POST** | **`/api/models/train`** | `{kind: action\|prediction}` — background training, validation-gated activation |
| **POST** | **`/api/models/activate`** | `{kind, model_id, force?, reason: manual\|rollback}` — 409 with the failed checks unless `force` |
| **GET** | **`/api/videos/{video_id}/complete-analysis`** | Video, stage states, tracks, actions, combination timeline, stored predictions, pipeline run |
| POST / GET | `/api/pipeline/videos/{video_id}/run` (`?force=true`) · `/api/pipeline/videos/{video_id}` · `/api/pipeline/runs` | Start the unified pipeline (returns the running one instead of a duplicate); runs and stage status |
| GET | `/api/models/deployments` · `/api/models/datasets` | Deployment history; what the next training would use |

All `/api/patterns/{fighter_id}/…` endpoints and `/compare` accept the filters `fight=` (repeatable), `video=` (repeatable), `date_from=`, `date_to=` (YYYY-MM-DD; undated fights are excluded when a date filter is used) and `min_confidence=`. `{fighter_id}` is a fighter profile id or an unassigned-track key `track-<video_id>-<track_id>`. Every response carries `meta` (`mode` snapshot/live, `stale`, `dataset_version`, applied filters) and `warnings`.


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

## Phase 4 verification & results

**Calculations — verified against hand counts.** `backend/tests/test_patterns.py` builds small action sequences whose statistics were counted by hand and checks the engine's output exactly, e.g. for fighter A — fight 1: *Jab, Cross, Hook* · (4.2 s pause) · *Jab, Cross* · (pause) · *Jab* + a duplicate overlapping Jab detection; fight 2: *Jab, Cross* · (2.3 s pause) · *Slip*:
* the duplicate is removed (more confident kept) and 5 sequences are formed; intervals 0.2 s, 0.2 s;
* Jab→Cross: 3 occurrences in 2 fights, 1.5 per fight, P(Cross \| Jab) = 3/4 = 0.75 with Wilson CI 0.3006–0.9544, labelled *insufficient* (4 < 5 observations); P(END \| Cross) = 2/3; average gap 0.167 s; followed by another action 1/3;
* the `window` definition (1.5 s) gives Cross→Hook 1 and Cross→NONE 2 — different from the `next` definition, as intended; `next_offensive` skips a Slip that `next` reports;
* smoothed Markov cell P(Cross \| Jab) = (3 + 0.5) / (4 + 0.5·5) = 0.5385; rows sum to 1; start distribution Jab 4, Slip 1;
* PrefixSpan finds Jab … Cross in 3/3 sequences when a Slip/Duck sits in between, while contiguous n-grams count it once;
* the same exchange uploaded twice for one fight is counted once; reactions to opponent attacks, attacks per minute, opponent attribution and lag, chi-square refusing to run on tiny expected counts, comparisons and Jensen–Shannon distance are checked the same way.

An API test seeds two fighter profiles with verified annotations in two fully annotated videos and checks the endpoints return the hand-counted numbers, that grouping both videos into one fight changes the cross-fight counts, that date filters apply, that background analysis writes the derived tables and is skipped when nothing changed, that a new annotation makes the stored analysis *stale* while re-extracting only the changed video, and that comparison and error cases behave.

**Real analysed data.** The Phase 3 test database (real broadcast boxing, masked UFC clips) gives 26 usable actions from 13 videos after exclusions (4 *Unknown*, 22 annotations with no fighter or track). After assigning the two boxers' tracks in the 8-second broadcast clip to profiles: Boxer A has 3 recognised punches in one sequence, opponent context (Boxer B's punch, 0.3–1.0 s earlier), distance 2.3–2.8 torso lengths and stance from the profile; **0 of 13 probabilities meet the minimum sample size**, every pattern is *single fight*, opponent analysis is *insufficient*, and the page states this. This is the correct output for that amount of data — it is not evidence about any fighter's style.

## Phase 5 evaluation & results
Full report: [`docs/phase5-evaluation-report.md`](docs/phase5-evaluation-report.md).

**Data:** BoxingVI annotation sheets V1–V10 imported (5,446 real punches, 5 invalid rows rejected) + the app's own footage → **5,016 next-action examples, 12 sessions/fights, 15 fighters/tracks**. Targets: right cross 1,314 · uppercut 1,244 · left jab 1,149 · left hook 922 · right hook 375 (+ 12 app examples).

**Unseen sessions = unseen athletes (5-fold CV, n = 5,016):**

| Model | Top-1 (95 % CI) | Top-3 | Macro F1 | Log loss | Brier | ECE | vs Markov top-1 |
|---|---|---|---|---|---|---|---|
| Markov baseline | 49.5 % (48.1–50.9) | 89.6 % | 0.338 | 1.325 | 0.673 | 0.177 | — |
| XGBoost | 58.7 % (57.4–60.1) | 93.2 % | 0.413 | 1.074 | 0.543 | 0.052 | +9.2 pts (+4.7 to +15.8) |
| **GRU (selected)** | 58.7 % (57.3–60.1) | 93.1 % | 0.423 | **0.972** | 0.500 | 0.084 | +9.2 pts (+4.7 to +16.1) |
| LSTM | 56.4 % (55.1–57.8) | 92.6 % | 0.403 | 1.030 | 0.529 | 0.086 | +6.9 pts (+0.4 to +16.0) |
| Transformer | **65.7 %** (64.4–67.0) | 92.2 % | **0.463** | 0.996 | **0.495** | 0.086 | **+16.2 pts** (+6.2 to +20.8) |
| Class-frequency floor | 11.2 % | 66.3 % | 0.043 | 1.643 | 0.816 | 0.186 | −38.3 pts |

Temporal hold-out (latest 4 sessions, n = 946): Transformer 78.5 % top-1 / log loss 0.680, GRU 76.7 %, XGBoost 75.3 %, LSTM 73.5 %, Markov 68.8 %. The GRU is the production model by the pre-declared rule (lowest held-out log loss); the Transformer (higher top-1, lowest Brier) can be activated in the model registry.

**Limits of this evidence:** boxing punches only — no kicks, clinch, takedowns, defence or opponent context; one session per athlete, so "known fighter" performance cannot be measured; orthodox label mapping assumed for all sessions; the app's own MMA footage (12 examples) is still far too small for an MMA evaluation.

**Machinery verification** (`tests/test_predictions.py`, constructed fixtures with known structure, not performance claims): no leakage of future information, fight-level splits never share fights, metrics match hand calculations, temperature scaling lowers log loss/ECE without changing ranking, hand-computed Markov back-off, XGBoost learns a distance-dependent rule the Markov model cannot, GRU/LSTM/Transformer learn order patterns, irregular BoxingVI sheets are parsed, and the full API.

**Reproduce:** download the BoxingVI `Annotation_files` folder from Google Drive, unzip, then **Next-Move Prediction → Models → Settings & data import** → select V1–V10.xlsx, dataset `boxingvi`, FPS 30 → *Import* → *Train & evaluate models* (≈ 5 min on CPU).

## Phase 6 end-to-end test results
* **Real footage through the unified pipeline** (`tests/test_platform.py::test_real_footage_unified_pipeline`): a real UFC clip and a real broadcast boxing clip plus a corrupted MP4 are uploaded together; the corrupted file fails processing with its error; for both real clips the pipeline completes (processing reused, detection/tracking/pose completed, action recognition *skipped — no trained action model*, patterns completed, prediction *skipped — no active model*); a second request returns the running pipeline instead of a duplicate; a re-run reuses the tracking analysis; the corrupted video's pipeline fails at *Video processing*; the dashboard totals match (3 uploaded, 2 processed, 2 analysed); tracks report coverage and interruptions (occlusions); pose-derived movement metrics exist.
* **All five stages on real footage** (manual run on the build machine's database with an active action model and the BoxingVI-trained next-move model): *broadcast boxing 8s* → processing, tracking & pose, action recognition (6 actions), pattern analysis and stored next-move predictions all *completed* in ≈ 30 s on CPU; the fighter assignments of both boxers were carried over to the new tracks.
* **Integration tests** (`tests/test_platform.py`): dashboard numbers equal direct database counts (an *Unknown* action is excluded everywhere); fighter intelligence values (opening actions, mean gap, per-video filter); prediction availability without a model, timeline kinds, timestamp-restricted predictions; model evaluation (P@3 = top-3 / 3, R@3 = top-3, sample counts), versions, validation-gated activation, **rollback** with deployment history; **failed model inference** (corrupted model files → HTTP 503, no made-up output); **interrupted pipeline** recovery; assignment carry-over on re-analysis.
* **Phase 3 test updated:** with 12 real strikes the newly trained action model now (correctly) fails the deployment gate (< 20 held-out samples) and is *not* activated; the test then shows activation is refused without `force` and recorded when forced.
* **Browser end-to-end** (Playwright, real data): all ten navigation items; dashboard (with fighter filter) → fighter profile → video full analysis (overlay video, seek, *Predict at playhead*, open a prediction's evidence, confirm a recognised action → saved) → model evaluation → model management (compare versions, switch model type) — no browser console errors.

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
│  │  ├─ platform/               # Phase 6: pipeline (unified analysis runs), analytics (dashboard, fighter intelligence,
│  │  │                          #   pattern evaluation, cache), validation (deployment gates), api
│  │  ├─ predictions/            # Phase 5: dataset (examples, leakage rules), features, models (Markov, XGBoost, GRU/LSTM,
│  │  │                          #   Transformer, temperature scaling), metrics, training (CV protocols), service (registry,
│  │  │                          #   prediction, evidence, reliability, video jobs), live (architecture), api
│  │  ├─ patterns/               # Phase 4: stats (Wilson, binomial, chi-square, entropy, JS), engine (pure: sequences,
│  │  │                          #   n-grams, PrefixSpan, conditional probabilities, Markov, tendencies, opponents, compare),
│  │  │                          #   extract (DB → observations, per-video cache), service (snapshots, background jobs), settings, api
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
      ├─ pages/                  # Overview, Library, Upload, VideoDetail, Fighters, Tracking, Actions, ActionModels, Patterns, Jobs, Settings
      ├─ components/vision/      # overlay player (canvas), timeline, track panel, frame inspector, charts
      ├─ components/actions/     # action timeline, history/review, sequences, charts, annotation panel
      ├─ components/patterns/    # transition heatmap, tendencies, combinations, explorer, opponents, compare, fights & settings
      ├─ components/platform/    # dashboard, prediction card (detected vs predicted timeline)
      ├─ pages/                  # + FighterIntelligence, VideoIntelligence, ModelEvaluation, ModelManagement (Phase 6)
      ├─ components/predictions/ # prediction panel, reliability, evaluation report views, calibration chart, confusion matrix
      └─ lib/, hooks/
```

**Phase 2 tables:** `vision_jobs` (status, progress, stage, timing, errors, settings snapshot, device, stats), `fighter_detections` (video, job, frame, timestamp, track ID, normalised box, confidence), `pose_landmarks` (video, job, track, frame, timestamp, landmark name/index, normalised x/y, confidence — all 17 COCO keypoints), `tracking_sessions` (track ID, assigned fighter, role, suggested role, first/last frame, coverage, interruptions, segments, status), `movement_metrics`, `fighter_distances`. These are new tables only — an existing Phase 1 database is upgraded in place on first start.

**Phase 3 tables:** `action_annotations`, `video_annotation_meta`, `action_models` (version, algorithm, dataset version, classes, accuracy/precision/recall/F1, per-class metrics, confusion matrix, training info), `action_training_jobs`, `action_analysis_jobs`, `recognized_actions`, `action_sequences` (sequence id, fighter, ordered actions, intervals). Model files live in `data/action_models/<version>/`. Rows are written with bulk inserts.

**Phase 4 tables:** `fight_info` (video → fight key, date, event), `pattern_video_cache` (per-video extracted observations + input signature), `pattern_analysis_jobs` (fighter or `*`, status, progress, dataset version, stats, timestamps), `pattern_snapshots` (complete stored analysis per run, kept as history), `action_transitions` (fighter, definition, order, previous action(s), next action, count, context count, probability, CI, fights, reliability, observation window), `fighting_combinations` (fighter, ordered sequence, occurrences, distinct fights, support, confidence, average gap, classification), `fighter_tendencies` (fighter, category, action, count, frequency, per minute, fights, analysis period). New tables only — existing databases upgrade in place; everything here can be regenerated.

**Phase 6 tables:** `pipeline_runs` (video, status, progress, per-stage status / job id / message, options, errors, timestamps), `model_deployments` (model kind, model, previous model, reason training/manual/rollback, forced, validation result, time).

**Phase 5 tables:** `prediction_models` (registry: architecture, version, training dataset version, dataset info, cross-validated metrics, temperature, file location, active/selected/experimental, training date), `prediction_training_jobs` (status, progress, stage, settings, dataset info, full evaluation report, errors), `prediction_video_jobs`, `prediction_results` (fighter, video, sequence, context, full distribution, top 3, predicted action & probability, reliability, model version, in-sample flag, timestamp — never edited), `prediction_evaluations` (prediction id, actual next action, predicted action, top 3, top-1/top-3 correctness, probability of the actual action, status, source), `prediction_external_actions` (imported real sequences). Model files: `backend\data\prediction_models\v<N>\<architecture>\` plus `bundle.joblib` (fighter statistics, evidence index, training feature ranges).

**Phase 3 → 4 hand-off:** `GET /api/actions/{id}/sequences` (or `/export?kind=sequences`) returns clean, chronological, fighter-specific sequences with intervals; `source=annotations` gives the same structure from verified human labels (ground truth).

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
**91 tests** (all passing; 84 run without real footage, 7 more need the footage clips). Phase 1 tests generate real MP4/AVI/MOV files with OpenCV/FFmpeg and covers: multi-file upload, metadata accuracy (resolution, FPS, frame count, duration, codec, audio), thumbnails, frame sampling modes and caps, preview proxy and HTTP Range streaming, duplicate detection, invalid/spoofed files, corrupted video → failed job → retry, size limit, rename/search/filter/sort, delete (including mid-processing), fighter CRUD and assignment, the FFmpeg-without-ffprobe and OpenCV-only fallbacks, orphan cleanup, and **persistence across an application restart**.

Phase 2 tests run the **real pretrained YOLO pose model** (no mocked detections):
* everywhere: endpoints, a video with no people (completes with zero tracks), Phase-1-first rule, duplicate-job prevention, failure → retry, CPU/CUDA device selection logic, and 9 unit tests of the movement maths (speeds in torso lengths/s, missing vs low-confidence vs stationary, gap filling, smoothing, direction, angles, distance/closing speed);
* with real footage (put clips in `backend\tests\footage\` or set `MMA_TEST_FOOTAGE_DIR`; skipped otherwise): two-fighter broadcast boxing (detection, tracking, 17 landmarks per person, skeleton-inside-box alignment check, movement, distances, manual assignment, annotated video, CSV/NPZ export, dashboard stats), an overhead amateur bout (hard case), a single-fighter UFC clip (re-analysis replaces results, delete analysis), a two-person outdoor clip (deleting a video removes its vision data; persistence across restart). Expected file names are in `tests/test_vision.py`.

Phase 4 tests (17, `tests/test_patterns.py`, no footage needed — ~2 s): statistics helpers, sequence cleaning and splitting, duplicate-footage handling, combinations, the three probability definitions, second-order contexts, Markov smoothing, PrefixSpan, the explorer, tendencies and reactions, opponent-analysis thresholds, comparison, and a full API/database/background-job test — all against hand-counted expected values (listed in [Phase 4 verification](#phase-4-verification--results)).

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests\test_patterns.py -v
```

Phase 5 tests (17, `tests/test_predictions.py`, no footage needed — ~15 s): leakage rules, leave-own-fight-out statistics, rare-class grouping, hand-computed metrics (top-k, log loss, Brier, ECE, confusion), temperature scaling, fight-resampled bootstrap, hand-computed Markov back-off, XGBoost context learning, GRU/LSTM/Transformer training, fight-grouped splits and skipped deep models, insufficient-data reporting, the live session replay, and the full API.

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests\test_predictions.py -v
```

Phase 6 tests (7, `tests/test_platform.py`; the real-footage pipeline test needs the footage clips): dashboard vs database counts, fighter intelligence, predictions at timestamps, model evaluation/versions/validation/rollback, failed inference, interrupted pipelines, assignment carry-over, and the unified pipeline on real UFC/boxing footage with a corrupted upload.

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests\test_platform.py -v
# full suite incl. real-footage tests (put the clips in a folder first)
$env:MMA_TEST_FOOTAGE_DIR="C:\path\to\footage"; .venv\Scripts\python.exe -m pytest -q
```

### Training the prediction models (Windows 11)
1. Make sure Phases 2–3 have produced actions: analyse videos (**Fighter Tracking**), assign fighter tracks, run **Action Recognition** (review detections if you can), and group clips of the same bout under **Pattern Intelligence → Fights & dates**.
2. Optional: download the BoxingVI `Annotation_files` (one sheet per video) and import them under **Next-Move Prediction → Models → Settings & data import** (FPS 30).
3. **Next-Move Prediction → Models → Train & evaluate models.** Progress is shown; the report lists every model under every protocol. On an NVIDIA GPU the sequence models use CUDA automatically (Settings → Device).
4. Use the **Predict** tab (manual sequences) or **Video timeline** (*Predict through this video*), and check **Evaluation** for held-out accuracy.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `API offline` in the sidebar | Start the backend (`start.bat`), check nothing else uses port 8000 (`run.py --port 8010` + set `VITE_API_TARGET=http://127.0.0.1:8010` for dev). |
| Video won't play in the browser | Expected for AVI and some MOV/HEVC files until processing creates the MP4 preview. Ensure *Generate browser preview* is on in Settings and click **Reprocess**. You can always download the original. |
| "FFmpeg not found" in Settings | Re-run `pip install -r requirements.txt` (installs the bundled FFmpeg) or install FFmpeg and add it to PATH / `MMA_FFMPEG_PATH`. |
| A video fails with "could not decode" | The file is truncated/corrupted or uses a codec OpenCV can't decode. Re-encode it (`ffmpeg -i in.mkv -c:v libx264 out.mp4`) and upload again. |
| PowerShell blocks scripts | Use the `.bat` files, or call `.venv\Scripts\python.exe` directly as shown above (no activation needed). |
| Pattern Intelligence shows no fighters | It needs Phase 3 output: run Action Recognition on analysed videos (or mark videos fully annotated with verified annotations). Annotations without a fighter, track or *fighter to analyze* can't be attributed and are excluded (counted in `/api/patterns` → `extraction`). |
| Every probability says *insufficient* / *single fight* | That is the honest result for little data. Add fights, assign tracks to fighter profiles so a fighter's footage combines across videos, group clips of one bout with **Fights & dates**, or lower the thresholds in Pattern Settings (the labels follow the thresholds). |
| Fighter shows *Out of date* | Actions, reviews, assignments, fight info or settings changed since the stored analysis. Views already show live numbers; click **Run analysis** to refresh the stored tables. |
| Start over | Stop the app and delete `backend\data\`. |

## 10. Known limitations

Reported plainly, as requested:

**Phase 6**
- **No authentication.** The app is a local single-user tool (it binds to 127.0.0.1); the specification asked for authentication "where already supported" — it never was. Do not expose port 8000 to a network.
- **The full pipeline needs trained models for stages 3 and 5.** Without an active action model, action recognition (and therefore new actions, patterns and predictions for that video) is skipped and says so; the app cannot recognise actions before you have annotated and trained.
- **Real MMA evaluation remains limited by data:** the evaluation page shows exactly how little real MMA data there is (27 held-out action-recognition samples; 12 MMA next-action examples). The next-move metrics come mostly from BoxingVI punch sequences.
- Assignment carry-over relies on time-span overlap; if two people swap positions between analyses with very different settings, check assignments after a re-analysis.
- Combination-detection accuracy can only be measured on videos that are both fully annotated and recognised by a model (1 such video in the build database → reported as insufficient).
- Dashboard caching may show numbers up to 30 s old when only data outside the fingerprint changed (e.g. fight dates).

**Phase 5**
- **Predictive performance is established for boxing punch sequences, not for MMA fights.** The evaluation data is BoxingVI (punches only, no opponent); the app's own MMA footage gives 12 examples. MMA-specific actions (kicks, takedowns, defence) and the opponent/position features need many recognised MMA actions before they can be evaluated.
- Predictions can only be as specific as Phase 3's recognised categories (currently mostly "punch (type unspecified)" and kicks on the available footage).
- BoxingVI labels are mapped assuming an orthodox stance for every session (lead = left).
- Live webcam/stream prediction is not implemented; offline replay of uploaded footage is.
- Manually entered sequences have no timing/pose context; predictions then warn that these features are missing.
- Timeline reveal of the actual next action uses the next stored prediction's action start; the very last action of a fighter in a video has no recorded follow-up.

**Phase 4**
- **No real fighter profile could be validated.** The real data available here (a few dozen recognised actions from short single-fight clips, recognised by a pipeline-check model) is far below what the reliability thresholds require; the system reports this instead of presenting patterns. The calculations themselves are verified with hand-counted sequences.
- Patterns inherit Phase 3's limits: unreviewed detections are only as accurate as the model (validated precision is shown in Phase 3), and categories the model can't recognise never appear in patterns. Reviewing detections or using fully annotated videos (*Data source: best*) gives ground-truth patterns.
- Opponent identification needs both fighters' tracks assigned; with more than two attributed people in a video the closest one (Phase 2 distance) is used, which can be wrong during scrambles. Opponent *style* is a simple, interpretable label from the opponent's observed action mix — no clustering model is trained because there are not enough fighters to cluster meaningfully.
- "Repeated pattern" means more frequent than chance relative to the fighter's own baseline (one-sided binomial test); with many combinations tested, some will pass by chance — check the counts, fights and CIs, not only the label.
- Unassigned tracks are analysed per video only; identity across videos requires assigning tracks to a fighter profile.
- Date filtering requires fight dates (Fights & dates); undated fights are excluded when a date filter is active.

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

- **Verified on Linux, not on a physical Windows 11 machine.** The full backend test suite (91 tests, including Phases 4–6) and browser end-to-end runs (Phase 1: upload → process → library → details → fighter assignment → rename → retry → delete → restart persistence; Phase 2: analyse → progress → overlay player → toggles → frame stepping → track assignment → annotated video → dashboard/jobs/settings; Phase 5: dashboard section → prediction with evidence/reliability → record outcome → video prediction run → timeline playback with delayed reveal → evaluation → model report; Phase 4: dashboard section → fighter page → run analysis → stored snapshot → every tab incl. heatmap hover, explorer, opponents, comparison → date filter → fights & settings panels, with no browser console errors) passed with Python 3.13 / Node 22 / PyTorch 2.15 (CPU) / Ultralytics 8.4. The code is written for Windows (pathlib everywhere, no POSIX-only calls, hidden FFmpeg console windows, retrying deletes for Windows file locks, bundled FFmpeg, non-ASCII-safe image writes, CRLF `.bat` scripts), but the `.bat` scripts themselves have not been executed on Windows.
- Testing used generated test-pattern videos, not real UFC broadcast footage. Very long (2–3 h) or unusual files may surface edge cases.
- Quality assessment is a heuristic (sharpness/exposure/resolution/bitrate), not a learned model.
- The job queue runs inside the API process (threads; state persisted in SQLite). Fine for a single-user desktop setup; a separate worker (e.g. Celery/RQ) would be the next step for GPU-heavy Phase 2+ workloads.
- Schema is created with `create_all`; add Alembic migrations before changing existing tables in later phases.
- No authentication — intended for local use on `127.0.0.1`.
- Uploads are buffered by the multipart parser to a temp file before being moved into storage, so very large uploads temporarily need about twice their size in free disk space. Requests larger than the configured limit are rejected up front via `Content-Length`.
