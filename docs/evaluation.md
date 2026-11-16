# Evaluation

This document describes how GWD evaluates a trained CNN today: the commands, the test sets, the metrics, the plots, the threshold strategies, and the leaderboard. The scenarios and the SNR definition are in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1).

> **Status:** The simulated test sets come from `EVAL_V1`, which differs from the training data in noise, masses, and SNR range (see "Known limitations" below). The numbers these commands print describe that test set, not the detector performance on real data. No measured results are listed here.

## Commands

| Command | What it does | Writes |
|---|---|---|
| `gwd eval run` | ROC analysis of one model on simulated data; with `--real` also the field test on real data (network) | one plot with six panels, one leaderboard row |
| `gwd eval thresholds` | Six threshold strategies from the ROC curve of one model | one plot, `results/optimal_thresholds.json` |
| `gwd eval compare` | ROC curves of several models on the same simulated test set | two plots |
| `gwd eval leaderboard` | Prints the leaderboard, best ROC AUC first | nothing |

`run`, `thresholds`, and `compare` take `--model` with `latest` (the newest timestamp in the name), a model name in `models_registry/`, or a path to a `.keras` file; `compare` takes it more than once and compares all models without it. `--seed` seeds Python's `random` and the global numpy RNG, so the same seed gives the same test set; without it every run draws a new one. `gwd eval run` runs offline; `--real` adds the field test, which downloads from GWOSC (decision D10: the field test is optional). `gwd eval thresholds --no-save` does not write the JSON file. An unknown model ends the command with exit code 1.

```bash
gwd eval run --model latest --seed 1
gwd eval run --real
gwd eval thresholds --seed 1
gwd eval compare --model gwd_model_20251203-220536 --model latest --seed 1
gwd eval leaderboard
```

The plots go to `plots/evaluation_models/`, with the start time of the command in the file name.

## Test sets

All simulated test sets are drawn from `EVAL_V1` with `make_test_set` in `gwd.data.injection`: each sample gets a signal with probability 1/2, white noise, a 30/30 M☉ chirp with its merger between 2.0 and 3.5 s, and a peak SNR between 0.5 and 2.5. Every sample is z-score normalized. The test sets are generated anew for every command and are never the training files.

| Command | Simulated samples |
|---|---|
| `gwd eval run` | 1000 for the ROC curve, plus 50 signal samples at each of 15 SNR levels from 0.0 to 2.5 for the sensitivity curve |
| `gwd eval thresholds` | 500 |
| `gwd eval compare` | 1000, the same for all models |

## Metrics

All metrics come from the model's score, a number between 0 and 1 for each sample.

- **ROC curve:** the true positive rate (TPR) against the false positive rate (FPR) for every threshold. It does not depend on the choice of one threshold, so two models can be compared without choosing one.
- **ROC AUC:** the area under the ROC curve. It is the probability that a random signal sample scores higher than a random noise sample: 1.0 separates both perfectly, 0.5 is chance.
- **Average precision (AP):** the precision averaged over all recall levels (`average_precision_score` in scikit-learn). It summarizes the precision-recall curve and, unlike ROC AUC, depends on the share of signal samples in the test set.
- **TPR** (recall, sensitivity) = TP / (TP + FN): the share of signal samples that are detected.
- **FPR** = FP / (FP + TN): the share of noise samples that raise a false alarm.
- **Accuracy** = (TP + TN) / all samples, at the operating point.
- **Operating point:** TPR, FPR, accuracy, and the confusion matrix at the fixed threshold `THRESHOLD = 0.15` in `gwd.evaluation.evaluate`. A sample counts as detected if its score is ≥ the threshold (`predict_labels` in `gwd.evaluation.metrics`), as in scikit-learn's `roc_curve`.
- **SNR50 and SNR90:** the first of the 15 SNR levels of the sensitivity curve at which at least 50% or 90% of the signal samples are detected at the operating point. If no level reaches the rate, the value is `N/A` in the output and `-` in the leaderboard. The SNR is the peak amplitude ratio described in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1), not a matched filter SNR.

Look at ROC AUC and the sensitivity together: a model with a higher AUC can still need a higher SNR to reach 90% detection. Accuracy alone can mislead on unbalanced data; the test sets here are balanced 50/50 on average.

### Field test on real data

With `--real`, `gwd eval run` downloads data from GWOSC (network), in `gwd.evaluation.real_data`. Each stretch of H1 data is downloaded with 16 s more on each side, conditioned as a whole (notches at 60, 120, 180, and 240 Hz, then a 20–300 Hz band-pass), and only then cut back to the requested span. The filters ring at the ends of the data they get; the extra 16 s keep that ringing out of the scored windows, as for the training noise, which is conditioned in one piece and then cut. gwpy keeps a local copy of every downloaded file, so a second run reads them from disk.

- **Events:** the 4 s window around each of GW150914, GW170817, GW190521, GW170104, and GW151226, normalized and scored. An event counts as found if its score is ≥ 0.15, the same rule as for the simulated metrics. The events inside the training mass range and those outside it (GW170817, a neutron star merger) are counted separately. An event whose download fails or is too short is skipped and reported, not counted as missed; then its group reads `n/a (k of n loaded)`.
- **False alarms:** 200 s of H1 noise from GPS 1253326755, conditioned the same way and cut into 50 windows of 4 s. The result is the number of windows with a score ≥ 0.15 and the analysed time, for example `3 false alarms in 0.0556 h`. It is not scaled to a rate per year: 50 windows are too few for that. If the noise download fails, it reads `n/a (noise not loaded)`.

Because of the padding, the `>=` rule, and the new texts, field test results since P4.3 cannot be compared with older leaderboard rows.

## Plots

`gwd eval run` writes `plot_evaluate_model_with_roc_<timestamp>.png` with six panels:

1. ROC curve with the AUC, the chance diagonal, and the operating point.
2. Precision-recall curve with the AP.
3. Confusion matrix at the operating point.
4. Sensitivity curve: detection rate against the injected SNR, with SNR50 and SNR90.
5. Score distributions of the signal and noise samples, with the threshold.
6. A table of ROC AUC, AP, accuracy, TPR, FPR, SNR50, and SNR90.

`gwd eval thresholds` writes `plot_optimize_threshold_<timestamp>.png`: the ROC curve with the point of every strategy, TPR, FPR, and Youden's J against the threshold, and a table of the strategies.

`gwd eval compare` writes `plot_compare_roc_<timestamp>.png` with the ROC curves of all models and a bar chart of their AUCs, and `plot_compare_roc_<timestamp>_zoom.png` with the region FPR ≤ 0.1. Each model has the same color in both plots.

## Threshold strategies

`find_optimal_thresholds` in `gwd.evaluation.metrics` proposes one threshold per strategy. None of them uses the infinite first threshold of `roc_curve`.

| Name | Rule |
|---|---|
| `youden` | Maximizes Youden's J = TPR − FPR, the vertical distance to the chance diagonal. |
| `f1` | Maximizes F1 = 2 · precision · recall / (precision + recall) over the thresholds of the precision-recall curve. |
| `fixed_fpr_5` | The highest TPR with FPR ≤ 5%; ties go to the lower FPR. |
| `fixed_fpr_1` | The highest TPR with FPR ≤ 1%; ties go to the lower FPR. |
| `cost_based` | Minimizes 5 · (1 − TPR) + 1 · FPR, so a missed signal costs five times a false alarm. |
| `fixed_tpr_90` | The lowest FPR with TPR ≥ 90%; ties go to the higher TPR. |

If no point of the ROC curve meets an FPR limit, the proposal lies just above the highest score, so it detects nothing and has TPR and FPR 0.

`gwd eval thresholds` also prints which strategy suits which use: `fixed_fpr_1` for a scientific analysis, `youden` for a trigger of follow-up analyses and for teaching, `f1` for statistical studies, and `fixed_tpr_90` for a real-time alert.

### Threshold workflow

1. Train a model (`gwd train`) and evaluate it (`gwd eval run`).
2. Run `gwd eval thresholds` for the same model. It overwrites `results/optimal_thresholds.json`:

   ```json
   {
     "model": "gwd_model_YYYYMMDD-HHMMSS.keras",
     "optimal_thresholds": {
       "youden": {"threshold": 0.0, "tpr": 0.0, "fpr": 0.0, "j_stat": 0.0, "description": "..."},
       "f1": {"threshold": 0.0, "tpr": 0.0, "fpr": 0.0, "f1_score": 0.0, "description": "..."}
     }
   }
   ```

   Every strategy has `threshold`, `tpr`, `fpr`, and `description`; `youden` adds `j_stat`, `f1` adds `f1_score`, and `cost_based` adds `cost`. The file holds one model at a time.
3. Choose the strategy that fits the use. Nothing reads the file yet: `gwd eval run` keeps 0.15, and the Glitch Hunter starts its threshold slider at 0.46.

## Leaderboard

Every `gwd eval run` appends one row to `results/model_leaderboard.csv`; old rows are kept as written, so a model evaluated twice has two rows. `gwd eval leaderboard` prints all rows sorted by `ROC_AUC`, best first, with rows without a numeric AUC at the end.

| Column | Content | Format |
|---|---|---|
| `Model` | File name of the model | `gwd_model_YYYYMMDD-HHMMSS.keras` |
| `Date` | Time of the evaluation | `YYYY-MM-DD HH:MM` |
| `ROC_AUC` | ROC AUC | 4 decimals |
| `Avg_Precision` | Average precision | 4 decimals |
| `Sim_Accuracy` | Accuracy at the operating point | percent, 1 decimal |
| `TPR` | TPR at the operating point | percent, 2 decimals |
| `FPR` | FPR at the operating point | percent, 2 decimals |
| `Sim_SNR50` | SNR50 | 2 decimals, or `-` |
| `Sim_SNR90` | SNR90 | 2 decimals, or `-` |
| `Physik_Gap` | Always `-`, see below | `-` |
| `Real_Events_Found` | Events found in the field test, inside the training range, then outside (OOD) | for example `4/4; OOD 0/1`; a group with a failed download reads `n/a (k of n loaded)`; `n/a` without `--real` |
| `Real_Noise_FAR` | False alarms of the field test in the analysed noise | for example `3 false alarms in 0.0556 h`; `n/a (noise not loaded)` if the noise download failed, `n/a` without `--real` |

If `results/model_leaderboard.csv` does not exist yet and `analysis/model_leaderboard.csv` does, the file is copied there once; the original stays. An old `./model_leaderboard.csv` in the project root is not copied, because it may use another schema.

## Matched Filter baseline (D5)

There is no valid Matched Filter baseline. The script that computed it was deleted, and its results were invalid (`EVAL-6`). The files `models_registry/physics_baseline*.json` are not read. `Physik_Gap` is written as `-`, and `gwd eval leaderboard` also shows `-` for older rows that have a value. Writing a row warns `Physics baseline currently invalid`. Decision D5 in [architecture.md](architecture.md#decisions) (default B) drops the column; until then it stays in the schema.

## Known limitations

- `EVAL_V1` uses white noise, fixed 30/30 M☉ masses, and a peak SNR of 0.5–2.5, while the training data has up to 80% real noise, random masses, and a peak SNR of 1–12 (`DATA-1`, `EVAL-1`, `EVAL-2`, `EVAL-3`). The full list is under "Known skews" in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1).
- The operating point and the field test use the fixed threshold 0.15, not a threshold chosen per model.
- Rows written before P4.3 have field test values from unpadded windows, counted with score > 0.15, with failed downloads counted as missed (`0/5`), and a false alarm rate per year (`< 157788.0/y`). They cannot be compared with newer rows.
- The field test uses five events and 200 s of noise; its numbers are a rough check, not a measured detector performance.
- The test sets change with every run unless `--seed` is given.
