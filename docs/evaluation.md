# Evaluation

This document describes how GWD evaluates a trained CNN today: the commands, the test sets, the metrics, the plots, the threshold strategies, and the leaderboard. The scenarios and the SNR definition are in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1).

> **Status:** Since P5.7 the simulated data is a fixed, hashed test set. The default, `stress_v1`, is `EVAL_V1`, which differs from the training data in noise, masses, and SNR range; `indist_v1` follows the training distribution (see "Test sets" and "Known limitations" below). The numbers these commands print describe that test set, not the detector performance on real data. No measured results are listed here.

## Commands

| Command | What it does | Writes |
|---|---|---|
| `gwd eval build-testset` | Draws a fixed test set once, or loads and checks it if it exists | `data/eval/<set>_<seed>_<n>.npz`, `results/testsets/<set>_<seed>_<n>.json` |
| `gwd eval run` | ROC analysis of one model on a fixed test set, also per bin; with `--real` also the field test on real data (network) | one plot with six panels; one leaderboard row for `stress_v1` |
| `gwd eval thresholds` | Six threshold strategies from the ROC curve of one model | one plot, `results/optimal_thresholds.json` |
| `gwd eval compare` | ROC curves of several models on the same fixed test set | two plots |
| `gwd eval leaderboard` | Prints the leaderboard, best ROC AUC first | nothing |

`run`, `thresholds`, and `compare` take `--model` with `latest` (the newest timestamp in the name), a model name in `models_registry/`, or a path to a `.keras` file; `compare` takes it more than once and compares all models without it. All three and `build-testset` take `--testset` (`stress_v1`, the default, or `indist_v1`), `--seed` (default 0), and `--n` (default 1000), which name the test set; the same three values always give the same samples, and the first command that needs a set builds it. `build-testset` also accepts `--scenario` for `--testset` and takes `--rebuild` to draw a set again. `gwd eval run` runs offline; `--real` adds the field test, which downloads from GWOSC (decision D10: the field test is optional). `gwd eval thresholds --no-save` does not write the JSON file. An unknown model, and a cached test set that does not match its hash or is stale, end the command with exit code 1; a negative `--seed` or an `--n` below 2 with exit code 2.

```bash
gwd eval build-testset --testset indist_v1 --seed 7 --n 64
gwd eval run --model latest --seed 1
gwd eval run --testset indist_v1
gwd eval run --real
gwd eval thresholds --seed 1
gwd eval compare --model gwd_model_20251203-220536 --model latest --seed 1
gwd eval leaderboard
```

The plots go to `plots/evaluation_models/`, with the start time of the command in the file name.

## Test sets

Since P5.7 every simulated number comes from a fixed test set of `gwd.evaluation.testsets` (decision D4, default A). There are two:

| Set | Drawn from | Noise | Masses | Peak SNR | Peak SNR bins |
|---|---|---|---|---|---|
| `indist_v1` | `TRAIN_V1`, the training distribution | 80% real where `data/noise_background/` has chunks, else white | random, 5–100 M☉ | 1–12 | 11, of width 1 |
| `stress_v1` | `EVAL_V1`, the distribution of every simulated number before P5.7 | white | 30/30 M☉ | 0.5–2.5 | 8, of width 0.25 |

`gwd eval build-testset` with `--testset`, `--seed`, and `--n`, or the first `run`, `thresholds`, or `compare` that needs the set, draws it once and writes:

- `data/eval/<set>_<seed>_<n>.npz`: the samples, z-score normalized, as float32, and the parameters of every sample;
- `results/testsets/<set>_<seed>_<n>.json`: the manifest with the format, set, scenario and its hash (`Scenario.hash()`), seed, n, waveform backend, package versions, Git commit, the noise chunks it drew from (a hash of their names and bytes), the bins, the parameters of every sample (label, peak and optimal SNR, masses, merger time, noise source, noise chunk), and the SHA-256 of the set.

Every later command with the same set, seed, and n loads the file instead (a cache hit) and prints `Cache hit:` with the SHA-256. Both folders are local and ignored by Git (decision D9, default A); under D9 = B the manifests could be versioned, the arrays under `data/` could not.

Each set has two parts, each drawn from its own random stream:

- `test`: n samples, balanced, the even ones with a signal and the odd ones noise only. The k-th signal sample draws its peak SNR uniform within peak SNR bin k mod (number of bins), so the peak SNR stays uniform over the range and every bin is filled once there are as many signal samples as bins. `run` takes the ROC curve, the operating point, the efficiency curve, and the metrics per bin from it; `thresholds` and `compare` score it too.
- `calib_bg`: n noise-only samples, only for the threshold calibration that P5.8 adds. Nothing reads it yet.

The SHA-256 covers the format, set, scenario and its hash, seed, and n, and every array with its name, dtype, and shape, not the bytes of the npz file. Loading a set computes it again: if it differs from the manifest, an array was changed after the build, and the command stops with exit code 1 (`ValueError` in Python). A set is stale if its manifest has another format, or if the scenario's fields or, for `indist_v1`, the noise chunks changed since the build; the command stops too. Neither case replaces the set on its own; `gwd eval build-testset --rebuild` (with the same `--testset`, `--seed`, and `--n`) draws it again. A missing manifest or npz means the set is drawn again.

The random streams come from `numpy.random.SeedSequence(seed, spawn_key=(0x67776474, crc32(set), part))` with an MT19937 generator. Training seeds the generator with the plain integer (`gwd data generate`, `gwd train`) or uses `SeedSequence(seed)` without a spawn key (the split), so no training seed gives a test set stream, and no eval command seeds the global numpy RNG any more.

Limits:

- As soon as real noise is drawn (`indist_v1` with chunks in `data/noise_background/`), the hash is reproducible only on a machine with the same chunks, the owner's. Tests and CI build the sets with an empty noise folder, so only with white noise.
- The real chunks of `indist_v1` can be ones the training data used; P6.3 gives each split its own pool.
- 1000 samples per part are 65.5 MB each as float32, so a default set takes about 131 MB in `data/eval/`.

## Metrics

All metrics come from the model's score, a number between 0 and 1 for each sample.

- **ROC curve:** the true positive rate (TPR) against the false positive rate (FPR) for every threshold. It does not depend on the choice of one threshold, so two models can be compared without choosing one.
- **ROC AUC:** the area under the ROC curve. It is the probability that a random signal sample scores higher than a random noise sample: 1.0 separates both perfectly, 0.5 is chance.
- **Average precision (AP):** the precision averaged over all recall levels (`average_precision_score` in scikit-learn). It summarizes the precision-recall curve and, unlike ROC AUC, depends on the share of signal samples in the test set.
- **TPR** (recall, sensitivity) = TP / (TP + FN): the share of signal samples that are detected.
- **FPR** = FP / (FP + TN): the share of noise samples that raise a false alarm.
- **Accuracy** = (TP + TN) / all samples, at the operating point.
- **Operating point:** TPR, FPR, accuracy, and the confusion matrix at the fixed threshold `THRESHOLD = 0.15` in `gwd.evaluation.evaluate`. A sample counts as detected if its score is ≥ the threshold (`predict_labels` in `gwd.evaluation.metrics`), as in scikit-learn's `roc_curve`.
- **Efficiency curve:** the detection rate at the operating point of the signal samples of `test` in each peak SNR bin, plotted at the bin centers. Before P5.7 it came from 50 extra signal samples at each of 15 SNR levels from 0.0 to 2.5.
- **SNR50 and SNR90:** the center of the first peak SNR bin at which at least 50% or 90% of its signal samples are detected at the operating point. If no bin reaches the rate, the value is `N/A` in the output and `-` in the leaderboard. The bins are peak SNRs, the peak amplitude ratio described in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1), not a matched filter SNR. Since P5.5 `gwd eval run` also reports SNR50 and SNR90 as optimal (matched filter) SNR: "Peak SNR50/90" is the bin center, "Optimal SNR50/90" the mean optimal SNR of that bin's samples. The run prints the optimal SNR per peak SNR of the set's signal samples; for the 30/30 M☉ chirps of `stress_v1` in white noise it is about 13.5. The summary at the end shows the peak and the optimal SNR90; the leaderboard stores the peak values only. Because the levels became bins, SNR50 and SNR90 since P5.7 cannot be compared with older leaderboard rows.
- **Metrics per bin (since P5.7):** `gwd eval run` prints tables per peak SNR bin, per optimal SNR bin (5 bins with about equal counts), per total mass bin (4 such bins; a fixed mass is one bin), and per noise source (`real`, `gaussian`). An SNR or mass bin shows its signal samples n, their TPR at the operating point, and the AUC of these samples against all noise samples of `test`; a noise source shows its signal and noise samples, TPR, FPR, and the AUC within that source. A bin whose signal samples all lie outside the training ranges of `TRAIN_V1` (peak SNR 1–12, masses 5–100 M☉ with mass2 ≤ mass1, merger time 2.0–3.6 s) is marked `Extrapolation`, one with some of them `partly extrapolation (x%)`. In `stress_v1` the peak SNR bins below 1.0 are `Extrapolation`.

Look at ROC AUC and the sensitivity together: a model with a higher AUC can still need a higher SNR to reach 90% detection. Accuracy alone can mislead on unbalanced data; the part `test` of every set is balanced, n/2 signal samples (rounded down) and the rest noise only.

### Field test on real data

With `--real`, `gwd eval run` downloads data from GWOSC (network), in `gwd.evaluation.real_data`. Each stretch of H1 data is downloaded with 16 s more on each side, conditioned as a whole (notches at 60, 120, 180, and 240 Hz, then a 20–300 Hz band-pass), and only then cut back to the requested span. The filters ring at the ends of the data they get; the extra 16 s keep that ringing out of the scored windows, as for the training noise, which is conditioned in one piece and then cut. gwpy keeps a local copy of every downloaded file, so a second run reads them from disk.

- **Events:** the 4 s window around each of GW150914, GW170817, GW190521, GW170104, and GW151226, normalized and scored. An event counts as found if its score is ≥ 0.15, the same rule as for the simulated metrics. The events inside the training mass range and those outside it (GW170817, a neutron star merger) are counted separately. An event whose download fails or is too short is skipped and reported, not counted as missed; then its group reads `n/a (k of n loaded)`.
- **False alarms:** 200 s of H1 noise from GPS 1253326755, conditioned the same way and cut into 50 windows of 4 s. The result is the number of windows with a score ≥ 0.15 and the analysed time, for example `3 false alarms in 0.0556 h`. It is not scaled to a rate per year: 50 windows are too few for that. If the noise download fails, it reads `n/a (noise not loaded)`.

Because of the padding, the `>=` rule, and the new texts, field test results since P4.3 cannot be compared with older leaderboard rows.

## Plots

`gwd eval run` writes `plot_evaluate_model_with_roc_<timestamp>.png` with six panels; its title names the test set and the first 12 characters of its SHA-256:

1. ROC curve with the AUC, the chance diagonal, and the operating point.
2. Precision-recall curve with the AP.
3. Confusion matrix at the operating point.
4. Efficiency curve: detection rate against the peak SNR bin centers, with SNR50 and SNR90; the top axis shows the optimal SNR (the factor above).
5. Score distributions of the signal and noise samples, with the threshold.
6. A table of ROC AUC, AP, accuracy, TPR, FPR, and SNR50 and SNR90 as peak and as optimal SNR.

`gwd eval thresholds` writes `plot_optimize_threshold_<timestamp>.png`: the ROC curve with the point of every strategy, TPR, FPR, and Youden's J against the threshold, and a table of the strategies.

`gwd eval compare` writes `plot_compare_roc_<timestamp>.png` with the ROC curves of all models, titled with the test set, and a bar chart of their AUCs, and `plot_compare_roc_<timestamp>_zoom.png` with the region FPR ≤ 0.1. Each model has the same color in both plots.

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
     "testset": {"name": "stress_v1", "seed": 0, "n": 1000, "part": "test", "sha256": "..."},
     "optimal_thresholds": {
       "youden": {"threshold": 0.0, "tpr": 0.0, "fpr": 0.0, "j_stat": 0.0, "description": "..."},
       "f1": {"threshold": 0.0, "tpr": 0.0, "fpr": 0.0, "f1_score": 0.0, "description": "..."}
     }
   }
   ```

   `testset` (since P5.7) names the set whose part `test` gave the scores. Every strategy has `threshold`, `tpr`, `fpr`, and `description`; `youden` adds `j_stat`, `f1` adds `f1_score`, and `cost_based` adds `cost`. The file holds one model at a time.
3. Choose the strategy that fits the use. Nothing reads the file yet: `gwd eval run` keeps 0.15, and the Glitch Hunter starts its threshold slider at 0.46.

## Leaderboard

Every `gwd eval run` on the test set `stress_v1` appends one row to `results/model_leaderboard.csv`; old rows are kept as written, so a model evaluated twice has two rows. All older rows come from `EVAL_V1`, the distribution of `stress_v1`, and the file has no column for the test set, so a run on `indist_v1` adds no row and says so; leaderboard v2 (P5.8) records the set and its hash. The row does not show the seed and n of the set either; the run prints them with the SHA-256. `gwd eval leaderboard` prints all rows sorted by `ROC_AUC`, best first, with rows without a numeric AUC at the end.

| Column | Content | Format |
|---|---|---|
| `Model` | File name of the model | `gwd_model_YYYYMMDD-HHMMSS.keras` |
| `Date` | Time of the evaluation | `YYYY-MM-DD HH:MM` |
| `ROC_AUC` | ROC AUC | 4 decimals |
| `Avg_Precision` | Average precision | 4 decimals |
| `Sim_Accuracy` | Accuracy at the operating point | percent, 1 decimal |
| `TPR` | TPR at the operating point | percent, 2 decimals |
| `FPR` | FPR at the operating point | percent, 2 decimals |
| `Sim_SNR50` | Peak SNR50 (the optimal SNRs are not stored) | 2 decimals, or `-` |
| `Sim_SNR90` | Peak SNR90 | 2 decimals, or `-` |
| `Physik_Gap` | Always `-`, see below | `-` |
| `Real_Events_Found` | Events found in the field test, inside the training range, then outside (OOD) | for example `4/4; OOD 0/1`; a group with a failed download reads `n/a (k of n loaded)`; `n/a` without `--real` |
| `Real_Noise_FAR` | False alarms of the field test in the analysed noise | for example `3 false alarms in 0.0556 h`; `n/a (noise not loaded)` if the noise download failed, `n/a` without `--real` |

If `results/model_leaderboard.csv` does not exist yet and `analysis/model_leaderboard.csv` does, the file is copied there once; the original stays. An old `./model_leaderboard.csv` in the project root is not copied, because it may use another schema.

## Matched Filter baseline (D5)

There is no valid Matched Filter baseline. The script that computed it was deleted, and its results were invalid (`EVAL-6`). The files `models_registry/physics_baseline*.json` are not read. `Physik_Gap` is written as `-`, and `gwd eval leaderboard` also shows `-` for older rows that have a value. Writing a row warns `Physics baseline currently invalid`. Decision D5 in [architecture.md](architecture.md#decisions) (default B) drops the column; until then it stays in the schema.

## Known limitations

- `stress_v1` (`EVAL_V1`, the default) uses white noise, fixed 30/30 M☉ masses, and a peak SNR of 0.5–2.5, while the training data has up to 80% real noise, random masses, and a peak SNR of 1–12 (`DATA-1`, `EVAL-1`, `EVAL-2`, `EVAL-3`). Since P5.7 `indist_v1` evaluates on the training distribution, and the metrics per bin mark the extrapolation. The full list is under "Known skews" in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1).
- The operating point and the field test use the fixed threshold 0.15, not a threshold chosen per model.
- Rows written before P4.3 have field test values from unpadded windows, counted with score > 0.15, with failed downloads counted as missed (`0/5`), and a false alarm rate per year (`< 157788.0/y`). They cannot be compared with newer rows.
- The field test uses five events and 200 s of noise; its numbers are a rough check, not a measured detector performance.
- Real noise chunks reach the CNN at a much smaller scale than white noise (see "Known skews"), so the noise source table of `indist_v1` mixes two very different inputs.
