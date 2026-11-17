# MMA Fight Intelligence — phase snapshots

Each folder is the complete project exactly as delivered at the end of that phase. Every phase builds on the previous one, so `phase-6` (identical to [`../mma-fight-intel`](../mma-fight-intel)) contains everything.

| Folder | Phase |
|---|---|
| [phase-1](phase-1) | Application foundation: video upload, library, processing (metadata, thumbnails, frame sampling, quality), fighter profiles, jobs, settings |
| [phase-2](phase-2) | Fighter detection, tracking & pose estimation (YOLO Pose, ByteTrack/BoT-SORT, movement analysis, annotated video, Fighter Tracking page) |
| [phase-3](phase-3) | MMA action recognition & strike classification (taxonomy, annotation tool, RF/XGBoost/SVM + Bi-LSTM/TCN, sliding-window detection, action sequences) |
| [phase-4](phase-4) | Fighting pattern recognition & combination analysis (combinations, conditional probabilities, Markov models, PrefixSpan, tendencies, opponent analysis, comparison) |
| [phase-5](phase-5) | AI next-move prediction (Markov, XGBoost, GRU/LSTM, Transformer; calibrated; evaluated on unseen fights/athletes incl. BoxingVI) |
| [phase-6](phase-6) | Complete platform: unified analysis pipeline, dashboard, fighter intelligence, video intelligence, model evaluation & management |

Setup and run instructions for Windows 11 are in each folder's `README.md`.
