# Evaluation

This document describes how GWD evaluates a trained CNN today: the commands, the test sets, the metrics, the plots, the threshold calibration, and the leaderboard. The scenarios and the SNR definition are in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1).

> **Status:** Since P5.7 the simulated data is a fixed, hashed test set. The default, `stress_v1`, is `EVAL_V1`, which differs from the training data in noise, masses, and SNR range; `indist_v1` follows the training distribution (see "Test sets" and "Known limitations" below). The numbers these commands print describe that test set, not the detector performance on real data. No measured results are listed here.

## Commands

| Command | What it does | Writes |
|---|---|---|
| `gwd eval build-testset` | Draws a fixed test set once, or loads and checks it if it exists | `data/eval/<set>_<seed>_<n>.npz`, `results/testsets/<set>_<seed>_<n>.json` |
| `gwd eval thresholds` | Calibrates one model's threshold at FPR 1% on the part `calib_bg` of a fixed test set; shows six reference strategies on `test` | the threshold in the model's `.json` in `models_registry/`; one plot; `results/thresholds/<model name>.json` |
| `gwd eval run` | ROC analysis of one model on a fixed test set at the model's threshold, also per bin; with `--real` also the field test on real data (network) | one plot with six panels; one row in `results/leaderboard_v2.csv` |
| `gwd eval compare` | ROC curves of several models on the same fixed test set, each with its operating point at its own threshold | two plots |
| `gwd eval leaderboard` | Prints leaderboard v2, one table per test set, best ROC AUC first | nothing |

`run`, `thresholds`, and `compare` take `--model` with `latest` (the newest timestamp in the name), a model name in `models_registry/`, or a path to a `.keras` file; `compare` takes it more than once and compares all models without it. All three and `build-testset` take `--testset` (`stress_v1`, the default, `indist_v1`, since P6.2 `indist_v2` or `stress_v2`, and since P6.6 `indist_v3` or `stress_v3`), `--seed` (default 0), and `--n` (default 1000), which name the test set; the same three values always give the same samples, and the first command that needs a set builds it. `build-testset` also accepts `--scenario` for `--testset` and takes `--rebuild` to draw a set again. `gwd eval run` runs offline; `--real` adds the field test, which downloads from GWOSC (decision D10: the field test is optional). `gwd eval thresholds --no-save` writes neither the threshold nor the JSON file. Since P6.6 the data follow the model's preprocessing (see "Preprocessing follows the model" below): `run`, `thresholds`, and `compare` score a model trained with whitening on the V3 counterpart of a V2 set, and a model without whitening on the V2 counterpart of a V3 set. An unknown model, and a cached test set that does not match its hash or is stale, end the command with exit code 1; a negative `--seed` or an `--n` below 2 with exit code 2.

```bash
gwd eval build-testset --testset indist_v1 --seed 7 --n 64
gwd eval thresholds --model latest
gwd eval run --model latest --seed 1
gwd eval run --testset indist_v1
gwd eval run --testset indist_v3
gwd eval thresholds --testset stress_v3
gwd eval run --real
gwd eval compare --model gwd_model_20251203-220536 --model latest --seed 1
gwd eval leaderboard
```

The plots go to `plots/evaluation_models/`, with the start time of the command in the file name.

## Test sets

Since P5.7 every simulated number comes from a fixed test set of `gwd.evaluation.testsets` (decision D4, default A). There are two V1 sets, since P6.2 two V2 sets for models trained on `TRAIN_V2` (P6.5), and since P6.6 two V3 sets for models trained on `TRAIN_V3`; a V1 set is drawn and binned in the peak SNR, a V2 or V3 set in the optimal SNR:

| Set | Drawn from | Noise | Masses | SNR | SNR bins |
|---|---|---|---|---|---|
| `indist_v1` | `TRAIN_V1`, the training distribution | 80% real where `data/noise_background/` has chunks, else white | random, 5–100 M☉ | peak 1–12 | 11, of width 1 |
| `stress_v1` | `EVAL_V1`, the distribution of every simulated number before P5.7 | white | 30/30 M☉ | peak 0.5–2.5 | 8, of width 0.25 |
| `indist_v2` | `INDIST_V2`, the same distribution as `TRAIN_V2` | simulated aLIGO design noise, conditioned with the signal | random, 5–100 M☉ | optimal 4–20 | 8, of width 2 |
| `stress_v2` | `STRESS_V2`: `TRAIN_V2` with one extrapolation bin | as `indist_v2` | random, 5–100 M☉ | optimal 2–20 | 9, of width 2; 2–4 lies below the training range |
| `indist_v3` | `INDIST_V3`: the samples of `indist_v2`, whitened | as `indist_v2`, whitened with the aLIGO PSD before the band-pass | random, 5–100 M☉ | optimal 4–20 | 8, of width 2 |
| `stress_v3` | `STRESS_V3`: the samples of `stress_v2`, whitened | as `indist_v3` | random, 5–100 M☉ | optimal 2–20 | 9, of width 2; 2–4 lies below the training range |

The V2 and V3 sets need gwpy and PyCBC; the scenarios are described in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1). The default stays `stress_v1` until P6.5.

A V3 set draws from the random streams of its V2 counterpart (`EvalSet.stream`): with the same seed and n, `indist_v3` holds the same labels, signal parameters, and noise realizations as `indist_v2`, and `stress_v3` as `stress_v2`; only the preprocessing differs (`tests/data/test_scenario_v3.py`). So a V2 and a V3 model can be compared per optimal SNR bin on the same physical events, each with its own preprocessing. The SHA-256 of a V3 set differs from that of its V2 counterpart, because the arrays differ; its manifest names the stream and the preprocessing.

`gwd eval build-testset` with `--testset`, `--seed`, and `--n`, or the first `run`, `thresholds`, or `compare` that needs the set, draws it once and writes:

- `data/eval/<set>_<seed>_<n>.npz`: the samples, z-score normalized, as float32, and the parameters of every sample;
- `results/testsets/<set>_<seed>_<n>.json`: the manifest with the format, set, scenario and its hash (`Scenario.hash()`), seed, n, waveform backend, package versions, Git commit, the noise chunks it drew from (a hash of their names and bytes), the bins, the parameters of every sample (label, peak and optimal SNR, masses, merger time, noise source, noise chunk), and the SHA-256 of the set.

Every later command with the same set, seed, and n loads the file instead (a cache hit) and prints `Cache hit:` with the SHA-256. Both folders are local and ignored by Git (decision D9, default A); under D9 = B the manifests could be versioned, the arrays under `data/` could not.

Each set has two parts, each drawn from its own random stream:

- `test`: n samples, balanced, the even ones with a signal and the odd ones noise only. The k-th signal sample draws its SNR (peak in V1, optimal in V2) uniform within SNR bin k mod (number of bins), so the SNR stays uniform over the range and every bin is filled once there are as many signal samples as bins. `run` takes the ROC curve, the operating point, the efficiency curve, and the metrics per bin from it; `thresholds` and `compare` score it too.
- `calib_bg`: n noise-only samples, only for the threshold calibration of `gwd eval thresholds` (since P5.8, see "Threshold calibration" below). No metric is computed on it.

The SHA-256 covers the format, set, scenario and its hash, seed, and n, and every array with its name, dtype, and shape, not the bytes of the npz file. Loading a set computes it again: if it differs from the manifest, an array was changed after the build, and the command stops with exit code 1 (`ValueError` in Python). A set is stale if its manifest has another format, or if the scenario's fields or, for `indist_v1`, the noise chunks changed since the build; the command stops too. Neither case replaces the set on its own; `gwd eval build-testset --rebuild` (with the same `--testset`, `--seed`, and `--n`) draws it again. A missing manifest or npz means the set is drawn again.

The random streams come from `numpy.random.SeedSequence(seed, spawn_key=(0x67776474, crc32(stream), part))` with an MT19937 generator, where the stream is the set's own name, or for a V3 set the name of its V2 counterpart. Training seeds the generator with the plain integer (`gwd data generate`, `gwd train`) or uses `SeedSequence(seed)` without a spawn key (the split), so no training seed gives a test set stream, and no eval command seeds the global numpy RNG any more.

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
- **Operating point:** TPR, FPR, accuracy, and the confusion matrix at the model's own threshold, which `gwd eval thresholds` calibrated on `calib_bg` (see "Threshold calibration" below); a model without one is evaluated at 0.5, with a warning, and the output and the leaderboard mark it as a fallback. Until P5.8 every model was evaluated at the fixed threshold 0.15. A sample counts as detected if its score is ≥ the threshold (`predict_labels` in `gwd.evaluation.metrics`), as in scikit-learn's `roc_curve`. The TPR is always given with the FPR that the threshold reached on the noise samples of `test` and the number of samples of each class, for example `TPR 63.20% (n 500) at FPR 0.80% (n 500)`.
- **Efficiency curve:** the detection rate at the operating point of the signal samples of `test` in each peak SNR bin, plotted at the bin centers. Before P5.7 it came from 50 extra signal samples at each of 15 SNR levels from 0.0 to 2.5.
- **SNR50 and SNR90:** the peak SNR at which 50% or 90% of the signal samples are detected at the operating point, interpolated linearly between the centers of the last bin below the rate and the first bin that reaches it (since P5.8; before, the center of the first bin that reached it). If the first bin already reaches the rate, it is that bin's center; if no bin reaches it, the value is `N/A` in the output and an empty cell in the leaderboard. They are always printed with the FPR of the threshold, because a threshold with more false alarms reaches every rate at a lower SNR (`evaluation:MISS-1`). The bins of a V1 set are peak SNRs, the peak amplitude ratio described in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1), not a matched filter SNR. Since P5.5 `gwd eval run` also reports SNR50 and SNR90 as optimal (matched filter) SNR: "Peak SNR50/90" is the bin center, "Optimal SNR50/90" the same interpolation over the mean optimal SNR of the bins' samples. On a V2 set (since P6.2) the bins are optimal SNRs: "Optimal SNR50/90" is the bin center, "Peak SNR50/90" the interpolation over the mean peak SNR, and the efficiency curve's x axis is the optimal SNR. The run prints the optimal SNR per peak SNR of the set's signal samples; for the 30/30 M☉ chirps of `stress_v1` in white noise it is about 13.5. The summary at the end shows the peak SNR50 and SNR90 and the optimal SNR90; the leaderboard stores both. Because the levels became bins (P5.7), the threshold became the model's own, and the values are interpolated (P5.8), SNR50 and SNR90 cannot be compared with the old leaderboard.
- **Metrics per bin (since P5.7):** `gwd eval run` prints tables per peak SNR bin, per optimal SNR bin (5 bins with about equal counts), per total mass bin (4 such bins; a fixed mass is one bin), and per noise source (`real`, `gaussian`, and `aligo` in the V2 sets). An SNR or mass bin shows its signal samples n, their TPR at the operating point, and the AUC of these samples against all noise samples of `test`; a noise source shows its signal and noise samples, TPR, FPR, and the AUC within that source. A bin whose signal samples all lie outside the training ranges of `TRAIN_V1` (peak SNR 1–12, masses 5–100 M☉ with mass2 ≤ mass1, merger time 2.0–3.6 s) is marked `Extrapolation`, one with some of them `partly extrapolation (x%)`. In `stress_v1` the peak SNR bins below 1.0 are `Extrapolation`. For the V2 sets the training ranges are those of `TRAIN_V2` (optimal SNR 4–20); in `stress_v2` the optimal SNR bin 2–4 is `Extrapolation`.

Look at ROC AUC and the sensitivity together: a model with a higher AUC can still need a higher SNR to reach 90% detection. Accuracy alone can mislead on unbalanced data; the part `test` of every set is balanced, n/2 signal samples (rounded down) and the rest noise only.

### Field test on real data

With `--real`, `gwd eval run` downloads data from GWOSC (network), in `gwd.evaluation.real_data`. Each stretch of H1 data is downloaded with 16 s more on each side, conditioned as a whole (notches at 60, 120, 180, and 240 Hz, then a 20–300 Hz band-pass), and only then cut back to the requested span. The filters ring at the ends of the data they get; the extra 16 s keep that ringing out of the scored windows, as for the training noise, which is conditioned in one piece and then cut. gwpy keeps a local copy of every downloaded file, so a second run reads them from disk.

- **Events:** the 4 s window around each of GW150914, GW170817, GW190521, GW170104, and GW151226, normalized and scored. An event counts as found if its score is ≥ the model's threshold, the same rule as for the simulated metrics. The events inside the training mass range and those outside it (GW170817, a neutron star merger) are counted separately. An event whose download fails or is too short is skipped and reported, not counted as missed; then its group reads `n/a (k of n loaded)`.
- **False alarms:** 200 s of H1 noise from GPS 1253326755, conditioned the same way and cut into 50 windows of 4 s. The result is the number of windows with a score ≥ the model's threshold and the analysed time, for example `3 false alarms in 0.0556 h`. It is not scaled to a rate per year: 50 windows are too few for that. If the noise download fails, it reads `n/a (noise not loaded)`.

Because of the padding, the `>=` rule, and the new texts, field test results since P4.3 cannot be compared with older leaderboard rows; since P5.8 they also use the model's threshold instead of 0.15.

Since P6.6 the real data get the model's preprocessing. For a model trained with whitening (`TRAIN_V3`) each padded stretch, 36 s for an event and 232 s for the noise, is first whitened with its own PSD: a Welch estimate over the whole padded stretch (Hann segments of 4 s, half overlap, the median of the segments, so a loud event in one or two segments does not raise it), after a Tukey taper of 1 s at each end that lies inside the pads. The notches, the band-pass, and the crop follow as before. On simulated aLIGO noise with a strong part below 10 Hz added, this gives windows that correlate with the windows whitened with the known PSD at about 0.99 (`tests/data/test_scenario_v3.py` asks for more than 0.97).

### Preprocessing follows the model

Since P6.6 a model records in its `.json` what its input went through: `preprocessing` is `bandpass` (notches and band-pass: `TRAIN_V1`, `TRAIN_V2`, and datasets without a manifest) or `whiten_bandpass` (`TRAIN_V3`). `gwd.data.scenarios.preprocessing_of` reads it back through the registry; a model without the key, such as every model trained before P6.6, gets the preprocessing of its `dataset_scenario` if that is a known scenario, else `bandpass`, which is what it always got. Every command that gives a model data picks it from there:

- `gwd eval run` and `gwd eval thresholds` print `Preprocessing of the model: ...`. If the named set has another preprocessing and a counterpart with the same draws exists, they use the counterpart and say so: `--testset indist_v2` scores a whitening model on `indist_v3`, and `--testset indist_v3` scores a model without whitening on `indist_v2`. The leaderboard row and the threshold record name the set that was used. A V1 set has no whitened counterpart; a whitening model scored on it gets a warning.
- `gwd eval compare` builds the named set and, if a model needs it, its counterpart, checks that both have the same labels, and scores each model on the one with its preprocessing; the plot title names both sets.
- The field test (`--real`) whitens the real data for a whitening model, as described above.
- The Glitch Hunter whitens its real data for a whitening model, and its simulation starts in the scenario version the model was trained on.

## Plots

The plots go to `plots/evaluation_models/`, and every command prints the path of each one. `gwd eval run` writes `plot_evaluate_model_with_roc_<model name>_<timestamp>.png` with six panels, so a loop over the models leaves one plot per model; its title names the model, the test set, and the first 12 characters of the set's SHA-256:

1. ROC curve with the AUC, the chance diagonal, and the operating point at the model's threshold.
2. Precision-recall curve with the AP.
3. Confusion matrix at the operating point.
4. Efficiency curve: detection rate against the peak SNR bin centers, with SNR50 and SNR90 and the FPR of the threshold; the top axis shows the optimal SNR (the factor above).
5. Score distributions of the signal and noise samples, with the threshold.
6. A table of ROC AUC, AP, the threshold, accuracy, TPR, FPR, and SNR50 and SNR90 as peak and as optimal SNR.

`gwd eval thresholds` writes `plot_optimize_threshold_<model name>_<timestamp>.png`, titled with the model: the ROC curve on `test` with a star at the calibrated threshold and the point of every reference strategy, TPR, FPR, and Youden's J against the threshold, and a table of the calibrated threshold and the strategies.

`gwd eval compare` writes `plot_compare_roc_<timestamp>.png` with the ROC curves of all models, titled with the test set, each with a star at its operating point at its own threshold, and a bar chart of their AUCs, and `plot_compare_roc_<timestamp>_zoom.png` with the region FPR ≤ 0.1. Each model has the same color in both plots. The printed ranking, by ROC AUC, lists the TPR of each model at its threshold with the FPR it reached.

## Threshold calibration

Since P5.8 every model has its own detection threshold (decision D10, default A). `gwd eval thresholds` scores the part `calib_bg` of the test set, n noise-only samples, and takes the lowest threshold at which at most 1% of them raise an alarm (`THRESHOLD_TARGET_FPR` in `gwd.config`, `threshold_at_fpr` in `gwd.evaluation.metrics`). The threshold is one of their scores, so the TPR is as high as the false alarm limit allows. With 1000 samples, 10 of them may lie at or above it. If even the highest score would exceed the limit (fewer than 100 samples, or an untrained model that gives every sample the same score), the threshold lies just above the highest score and detects nothing. The part `test` is never used to choose it; the command then reports the TPR on the signal samples of `test` and the FPR on its noise samples at that threshold, each with n.

With `--save`, the default, the threshold goes into the model's metadata (`gwd.registry.set_threshold`), next to the keys that training wrote:

| Key | Content |
|---|---|
| `threshold` | The threshold |
| `threshold_fpr` | The target false alarm rate, 0.01 |
| `threshold_fpr_achieved` | The rate it reached on `calib_bg` |
| `threshold_n_background` | The number of `calib_bg` samples |
| `threshold_strategy` | `fixed_fpr on calib_bg` |
| `threshold_testset` | The set: `name`, `seed`, `n`, `part` (`calib_bg`), and `sha256` |
| `threshold_date` | When it was saved |

`gwd eval run`, `gwd eval compare`, and the Glitch Hunter read it with `gwd.registry.get_threshold`. A model without a threshold, such as every model trained before P5.8, gets 0.5 and a warning that names the command; calibrate each old model once, for example `gwd eval thresholds --model gwd_model_20251205-232013 --save`, or all of them with the loop in the [README](../README.md#3-evaluation). A threshold calibrated on one test set can be used on another; `gwd eval run` then says that the FPR on its set can differ from the target. A model trained after its last calibration is a new file and needs its own.

`gwd eval thresholds` also writes a report to `results/thresholds/<model name>.json` (`THRESHOLD_REPORTS_DIR` in `gwd.config`), one per model, and prints its path; a later run of the same model replaces it, and nothing reads it. Until this change every run overwrote the single file `results/optimal_thresholds.json`, so after a loop over the models only the last report was left; that file is no longer written, and an existing one stays as it is. A report looks like this:

```json
{
  "model": "gwd_model_YYYYMMDD-HHMMSS.keras",
  "testset": {"name": "stress_v1", "seed": 0, "n": 1000, "part": "calib_bg and test", "sha256": "..."},
  "calibrated": {"strategy": "fixed_fpr on calib_bg", "threshold": 0.0, "target_fpr": 0.01, "calib_fpr": 0.0, "n_background": 1000, "test_tpr": 0.0, "n_signal": 500, "test_fpr": 0.0, "n_noise": 500},
  "reference_strategies_on_test": {
    "youden": {"threshold": 0.0, "tpr": 0.0, "fpr": 0.0, "j_stat": 0.0, "description": "..."}
  }
}
```

### Reference strategies

For comparison, the command still shows the six strategies of `find_optimal_thresholds` in `gwd.evaluation.metrics`. They are chosen on the part `test`, so their TPR and FPR there are optimistic, and nothing saves or uses them. None of them uses the infinite first threshold of `roc_curve`.

| Name | Rule |
|---|---|
| `youden` | Maximizes Youden's J = TPR − FPR, the vertical distance to the chance diagonal. |
| `f1` | Maximizes F1 = 2 · precision · recall / (precision + recall) over the thresholds of the precision-recall curve. |
| `fixed_fpr_5` | The highest TPR with FPR ≤ 5%; ties go to the lower FPR. |
| `fixed_fpr_1` | The highest TPR with FPR ≤ 1%; ties go to the lower FPR. |
| `cost_based` | Minimizes 5 · (1 − TPR) + 1 · FPR, so a missed signal costs five times a false alarm. |
| `fixed_tpr_90` | The lowest FPR with TPR ≥ 90%; ties go to the higher TPR. |

If no point of the ROC curve meets an FPR limit, the proposal lies just above the highest score, so it detects nothing and has TPR and FPR 0. The command also prints which strategy suits which use: `fixed_fpr_1` for a scientific analysis, `youden` for a trigger of follow-up analyses and for teaching, `f1` for statistical studies, and `fixed_tpr_90` for a real-time alert.

## Leaderboard

Since P5.8 every `gwd eval run` appends one row to `results/leaderboard_v2.csv`, on any test set; old rows are never rewritten, so a model evaluated twice has two rows. The file holds raw numbers: floats, integers, and empty cells for missing values, so `pandas.read_csv` reads the metrics as numbers (`gwd.evaluation.leaderboard.read_leaderboard` also reads the hashes as text and every float exactly as written). Only `gwd eval leaderboard` formats them. It prints one table per test set (name, seed, n, and SHA-256), because rows of different sets are not comparable, and ranks each by ROC AUC, then by TPR at the threshold; rows without a numeric AUC go last. It never ranks by SNR50: a model that fires on everything reaches every SNR level (`evaluation:MISS-1`). The printed columns `peak SNR50` and `peak SNR90` are the CSV columns `snr50` and `snr90`, and `optimal SNR90` is `optimal_snr90`; until this change the first two were labelled only `SNR50` and `SNR90`, and the optimal SNR was not shown.

| Column | Content |
|---|---|
| `schema_version` | 2 |
| `model` | File name of the model, `gwd_model_YYYYMMDD-HHMMSS.keras` |
| `date` | Time of the evaluation, ISO 8601 |
| `testset`, `testset_seed`, `testset_n`, `testset_hash` | The test set and its SHA-256 |
| `eval_scenario` | Its scenario, `EVAL_V1` or `TRAIN_V1` |
| `backend` | The waveform backend that drew it, from its manifest |
| `noise_source` | The noise of its part `test`: `gaussian`, `real`, or `gaussian+real` |
| `roc_auc`, `ap` | ROC AUC and average precision on `test` |
| `threshold` | The model's threshold |
| `threshold_fpr`, `threshold_testset_hash` | Its target FPR and the SHA-256 of the set it was calibrated on; empty for the fallback 0.5 |
| `tpr_at_fpr`, `fpr_achieved` | TPR and FPR at the threshold on `test`, as fractions |
| `n_signal`, `n_noise` | The signal and noise samples of `test` |
| `snr50`, `snr90` | Peak SNR50 and SNR90 at the threshold; empty if not reached |
| `optimal_snr50`, `optimal_snr90` | The same as optimal SNR |
| `real_loaded`, `real_found`, `real_total` | Field test events in the training range: loaded, found, all; empty without `--real` |
| `real_ood_loaded`, `real_ood_found`, `real_ood_total` | The same for the events outside it (OOD) |
| `real_false_alarms`, `real_noise_seconds` | False alarms in the analysed real noise and its length in seconds; empty without `--real` or if the noise did not load |

The 12-column leaderboard from before P5.8, `results/model_leaderboard.csv`, is never written again and stays as an archive; so do the older copies `analysis/model_leaderboard.csv` and `./model_leaderboard.csv`, which are no longer copied anywhere. `gwd eval leaderboard` names the ones that exist and does not show their rows: they used other test sets, the fixed threshold 0.15, formatted strings such as `51.8%`, and a `Physik_Gap` column based on an invalid baseline. `read_legacy` in `gwd.evaluation.leaderboard` still reads them, as strings.

## Matched Filter baseline (D5)

There is no valid Matched Filter baseline. The script that computed it was deleted, and its results were invalid (`EVAL-6`). Decision D5 in [architecture.md](architecture.md#decisions) (default B) dropped it in P5.8: leaderboard v2 has no baseline column, and nothing reads the files `models_registry/physics_baseline*.json`, which stay where they are. The deleted script, `evaluate_classic.py` in the project root, is in the Git history at `45b9202^` (`git show 45b9202^:evaluate_classic.py`). It has a known template bug: it cuts the PyCBC template to its first 4 s, so the template ends before the merger and holds almost none of the signal energy (`EVAL-6`). It also reads and writes paths relative to the working directory, waits for `input()`, and uses another noise model, so checking it out would give a wrong baseline at once. Step P6.4, a rewritten baseline, is dropped (D5 = B); it would be a new project, for which the old script is only a reference.

## Known limitations

- `stress_v1` (`EVAL_V1`, the default) uses white noise, fixed 30/30 M☉ masses, and a peak SNR of 0.5–2.5, while the training data has up to 80% real noise, random masses, and a peak SNR of 1–12 (`DATA-1`, `EVAL-1`, `EVAL-2`, `EVAL-3`). Since P5.7 `indist_v1` evaluates on the training distribution, and the metrics per bin mark the extrapolation. The full list is under "Known skews" in [architecture.md](architecture.md#scenarios-and-snr-definition-current-state-v1).
- The threshold is calibrated on the noise of one test set. On a set with other noise (for example `indist_v1`, mostly real noise, with a threshold from `stress_v1`, white noise) its FPR can be far from 1%; `fpr_achieved` shows the rate it reached.
- With 1000 `calib_bg` samples, the reached FPR has a statistical uncertainty of about ±0.3 percentage points; the false alarm rate at very low FPR (below 1 in 1000) cannot be calibrated with this set.
- Rows of the old leaderboard written before P4.3 have field test values from unpadded windows, counted with score > 0.15, with failed downloads counted as missed (`0/5`), and a false alarm rate per year (`< 157788.0/y`). They cannot be compared with newer rows.
- The field test uses five events and 200 s of noise; its numbers are a rough check, not a measured detector performance.
- gwpy's band-pass (20–300 Hz) is not flat inside its pass band: white noise through the notches and the band-pass has a Welch PSD that varies between about 0.65 and 1.75 times its median across 20–300 Hz (measured in 10 Hz bands on simulated samples). V2 and V3 both keep it, as the real data path does; whitening in V3 makes the input flat up to that ripple.
- Real noise chunks reach the CNN at a much smaller scale than white noise (see "Known skews"), so the noise source table of `indist_v1` mixes two very different inputs.
