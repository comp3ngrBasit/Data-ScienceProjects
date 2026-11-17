# Phase 5 — Next-move prediction: evaluation report

## Summary

On **unseen athletes**, the models predict the next punch far better than the baselines, and the improvement over the Markov baseline is statistically clear. The training data is real **BoxingVI** annotations: 5,004 next-punch examples from 10 continuous boxing sessions. The Transformer reaches the highest accuracy (**65.7 % top-1, 92.2 % top-3**). The GRU has the best-calibrated probabilities (lowest log loss) and is the selected production model. The Markov baseline gets 49.5 % top-1. Always predicting the most frequent class gets 11.2 %.

**What this does and does not show.** BoxingVI is boxing: shadow boxing, bag work and sparring, with six punch types. It has no kicks, takedowns, defence or opponent. So these results show that the engine can learn and generalise **punch-to-punch patterns** for athletes it has never seen. They do **not** show next-move prediction in MMA fights. The app's own MMA footage still provides only **12** next-action examples, which is not enough to evaluate on its own.

## Data

| Source | Next-action examples | "Fights" | Notes |
|---|---|---|---|
| BoxingVI annotation sheets V1–V10 (imported, real) | 5,004 | 10 sessions = 10 athletes | 5,446 punches; 5 rows rejected (end frame before start frame); 30 fps; orthodox mapping (jab → left jab, cross → right cross, lead/rear hook → left/right hook, uppercuts → uppercut) |
| App footage (Phase 3 detections / reviews) | 12 | 2 | unspecified punches and kicks |
| **Total** | **5,016** | **12** | 15 fighters / tracks |

* **Targets:** right cross 1,314 · uppercut 1,244 · left jab 1,149 · left hook 922 · right hook 375 · unspecified punch 8. Kicks (4) are grouped as "Other (rare)".
* **Context:** for BoxingVI, only the action history and timing are available (no pose, distance or opponent). For the app clips, the Phase 2 and opponent context is also used.

## Protocols

All three protocols test on data the model never trained on:

* **Fight-level 5-fold CV:** each session is held out whole, so every test athlete is unseen. This is the main comparison.
* **Unseen-fighter CV:** gives identical numbers here, because every BoxingVI session is a different athlete.
* **Temporal hold-out:** train on the 8 earliest-imported sessions, test on the last 4.

Within each training portion, the last ~20 % of sessions are held back for early stopping and temperature scaling.

## Results — unseen sessions / athletes (5-fold CV, n = 5,016)

| Model | Top-1 (95 % CI) | Top-3 | Macro F1 | Log loss | Brier | ECE | Top-1 vs Markov (95 % CI, resampled by session) |
|---|---|---|---|---|---|---|---|
| Markov baseline | 49.5 % (48.1–50.9) | 89.6 % | 0.338 | 1.325 | 0.673 | 0.177 | — |
| XGBoost | 58.7 % (57.4–60.1) | **93.2 %** | 0.413 | 1.074 | 0.543 | **0.052** | +9.2 pts (+4.7 to +15.8) |
| **GRU (selected)** | 58.7 % (57.3–60.1) | 93.1 % | 0.423 | **0.972** | 0.500 | 0.084 | +9.2 pts (+4.7 to +16.1) |
| LSTM | 56.4 % (55.1–57.8) | 92.6 % | 0.403 | 1.030 | 0.529 | 0.086 | +6.9 pts (+0.4 to +16.0) |
| Transformer | **65.7 %** (64.4–67.0) | 92.2 % | **0.463** | 0.996 | **0.495** | 0.086 | **+16.2 pts** (+6.2 to +20.8) |
| Class-frequency floor | 11.2 % | 66.3 % | 0.043 | 1.643 | 0.816 | 0.186 | −38.3 pts |

**How the production model was chosen:** the rule, fixed in advance, is the lowest held-out log loss, so the GRU is the production model. The Transformer has clearly higher top-1 accuracy, slightly higher log loss and the lowest Brier score. It can be activated in **Models → Model registry**.

**Per-class recall:**

| Model | Right cross | Uppercut | Right hook | Left hook | Left jab |
|---|---|---|---|---|---|
| Transformer | 0.75 | 0.67 | 0.70 | 0.62 | 0.56 |
| GRU | 0.57 | 0.58 | 0.68 | 0.63 | 0.56 |
| Markov | 0.58 | 0.51 | 0.39 | 0.31 | 0.56 |

The 8 unspecified punches from the app footage are never predicted correctly; they're too few to learn.

## Temporal hold-out (latest 4 sessions, n = 946)

| Model | Top-1 | Top-3 | Log loss | ECE |
|---|---|---|---|---|
| Markov | 68.8 % | 91.1 % | 0.978 | 0.225 |
| XGBoost | 75.3 % | 96.7 % | 0.746 | 0.128 |
| GRU | 76.7 % | 96.2 % | 0.767 | 0.167 |
| LSTM | 73.5 % | 95.7 % | 0.757 | 0.099 |
| Transformer | **78.5 %** | 96.6 % | **0.680** | 0.129 |

Every learned model beats the Markov baseline, and the 95 % interval of each difference, resampled by session, is above zero.

## Limitations

* **Boxing, not MMA:** six punch classes only. There are no kicks, clinch, takedowns or defensive actions in the training data, so the models know nothing about them.
* **No opponent:** BoxingVI has no opponent or spatial context. The "with opponent context" subset is 3 examples, so the context features are untested on real data at scale.
* **Unseen athletes only:** each athlete appears in one session, so "known fighter" performance can't be measured.
* **Label mapping:** the orthodox mapping is assumed for all sessions. If some athletes are southpaw, left/right labels are swapped for them.
* **Calibration:** ECE is 0.05–0.09 on held-out data, so stated probabilities are approximately right but not perfectly calibrated.
* **App footage:** the 12 MMA examples remain far too few for a separate MMA evaluation. Predictions on them are flagged *in-sample* when the model trained on that clip.

## Verification of the machinery (constructed test fixtures, not performance claims)

`backend/tests/test_predictions.py` checks:

* **No leakage:** features use only information available before the prediction moment, and fighter statistics never include the example's own fight.
* **Splits:** fight-grouped splits never share a fight between training and test.
* **Metrics and calibration:** metrics match hand calculations, and temperature scaling improves calibration without changing rankings.
* **Markov baseline:** its back-off probability matches a hand calculation.
* **Context:** XGBoost learns a distance-dependent rule that the Markov model cannot.
* **Sequence models:** GRU, LSTM and Transformer learn order patterns.
* **Import:** irregular BoxingVI sheet layouts are parsed, and invalid rows are rejected.
* **API end to end.**
