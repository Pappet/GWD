# Gravitational Wave Detection (GWD)

GWD trains a convolutional neural network (CNN) to detect gravitational wave chirps from binary black hole mergers in 4 s of detector noise, and evaluates it with ROC metrics on simulated data and on a few real events from GWOSC. It also has three interactive apps for learning: a waveform explorer, an interferometer designer, and the Glitch Hunter, in which the CNN rates simulated or real data.

> **Status:** The project is being cleaned up. The simulated metrics are of limited significance: training and evaluation draw from different distributions, the SNR is a peak amplitude ratio, and there is no Matched Filter baseline. [docs/architecture.md](docs/architecture.md) describes the target structure, the open decisions, and the progress, with the steps that are still open.

## Components

Everything is in one package, `gwd`, in `src/gwd/`. The pipeline runs from left to right:

```text
core  ->  data  ->  training  ->  evaluation
              \        |            /
               '--- registry ------'
apps: use core, data, and registry      cli: the gwd command      main.py: menu
```

| Component | Folder | What it does |
|---|---|---|
| core | `src/gwd/core/` | Physics: waveforms (PyCBC), noise, the interferometer learning model, a time domain simulator. No file or network access. |
| data | `src/gwd/data/` | Scenarios, signal injection, conditioning and normalization, GWOSC access, dataset generation with a manifest, validation, the noise download. |
| training | `src/gwd/training/` | The CNN and its training. |
| evaluation | `src/gwd/evaluation/` | ROC evaluation, threshold calibration, model comparison, the leaderboard. |
| registry | `src/gwd/registry.py` | Finds, selects, and loads models in `models_registry/`, and reads and writes their metadata. |
| apps | `src/gwd/apps/` | The three apps; they only draw and interact. |
| cli | `src/gwd/cli.py`, `src/gwd/env.py` | The `gwd` command and `gwd env`. |
| menu | `main.py` | A text menu that runs `gwd` commands. |

[docs/architecture.md](docs/architecture.md) has the import rules between the components, which `tests/test_architecture.py` enforces.

## Installation

GWD needs Python 3.11 or newer and PyCBC, its only waveform backend, which is a base dependency. Use Linux or WSL: PyCBC is not available on native Windows, so `pip install` leaves it out there. macOS is not tested. Without PyCBC, `gwd data generate`, `gwd eval run`, `gwd eval thresholds`, `gwd eval compare`, and the waveform explorer stop with exit code 1 and print how to install it; the interferometer designer has no aLIGO reference curve; and the Glitch Hunter starts with its simulation off and shows only real GWOSC data.

```bash
# A fresh virtual environment
python -m venv .venv
source .venv/bin/activate

# The package with all extras and the development tools
pip install -e '.[all,dev]'

# Check the environment, the waveform backend, the paths, and the GUI backend
gwd env
```

The base install has numpy, scipy, pandas, matplotlib, scikit-learn, tabulate, and (except on Windows) pycbc. PyCBC draws the waveforms of `gwd data generate`, `gwd eval run`, `gwd eval thresholds`, `gwd eval compare`, the waveform explorer, and the Glitch Hunter's simulation, and the aLIGO design curve (`aLIGOZeroDetHighPower`) of the two demo apps; there is no fallback. Until P6.1 it was the `physics` extra. The extras add what single commands need:

| Extra | Installs | Needed for |
|---|---|---|
| `gwdata` | gwpy | `gwd data fetch-noise`, the field test of `gwd eval run`, and real data in the Glitch Hunter |
| `ml` | tensorflow | `gwd train`, `gwd eval run`, `gwd eval thresholds`, `gwd eval compare`, and 'Ask AI' in the Glitch Hunter |
| `dev` | pytest, ruff | Tests and linting |
| `all` | `gwdata` and `ml` | |

For an NVIDIA GPU, TensorFlow offers `pip install 'tensorflow[and-cuda]'`; this project does not install it for you. Without a GPU, training runs on the CPU, by default in float32.

**GUI backend.** The apps open matplotlib windows and need an interactive backend such as TkAgg. Tk is a system package: `sudo apt install python3-tk` on Debian and Ubuntu, `sudo dnf install python3-tkinter` on Fedora. On WSL, windows need WSLg (`wsl --update`). `gwd env` shows the backend in use. Without an interactive backend, for example with `MPLBACKEND=Agg`, an app exits with code 2 and says what to install.

### Reproducible environment (lockfile)

`requirements-linux-py312.lock` pins every package, including TensorFlow and PyCBC, to the versions the models were trained and the golden test values were recorded with. It is valid for Linux and Python 3.12 only. Install from it in a fresh virtual environment:

```bash
pip install -r requirements-linux-py312.lock
pip install -e . --no-deps
```

CI installs from the same file: the fast job with `-c requirements-linux-py312.lock` (everything except TensorFlow), the slow job with `-r` as above. CI also fails if `pyproject.toml` and the lock disagree.

### Updating the lockfile

The lock is compiled with [uv](https://docs.astral.sh/uv/), on Linux, from `pyproject.toml` with all extras. Update it in the same commit as the change to `pyproject.toml`:

```bash
# After adding, removing, or changing a dependency in pyproject.toml.
# uv keeps the versions already in the output file and only resolves what changed.
uv pip compile pyproject.toml --all-extras --python-version 3.12 -o requirements-linux-py312.lock

# Upgrade one package on purpose (repeat the flag for more packages)
uv pip compile pyproject.toml --all-extras --python-version 3.12 -o requirements-linux-py312.lock --upgrade-package scipy

# Check that the lock still matches pyproject.toml (the same check CI runs; no output means it matches)
uv pip compile pyproject.toml --all-extras --python-version 3.12 \
  -c requirements-linux-py312.lock --no-header --no-annotate -q -o /tmp/lock-check.txt
diff <(grep -E '^[^ #]' requirements-linux-py312.lock) /tmp/lock-check.txt
```

Avoid `--upgrade` for everything: newer numpy, pandas, TensorFlow, or PyCBC versions can change the golden values and the behavior of the trained models. After any update, install the new lock in a fresh virtual environment and run all tests with `GWD_REQUIRE_PYCBC=1 python -m pytest`. If golden values change, update them only in a separate `golden:` commit that states the reason (see [AGENTS.md](AGENTS.md)).

The first lock was compiled with an extra `-c constraints-pc.txt` (the `pip freeze` of the PC training environment without the `nvidia-*` packages, not in the repository), so its versions match that environment. The commands above keep those versions without the file. The first update rewrites the header and drops the `-c constraints-pc.txt` notes from the file; that is expected.

## Usage

Every step is a `gwd` command; `gwd --help` and `gwd <command> --help` list the options. Exit codes: 0 on success, 1 when a command fails (for example an unknown model or a missing dataset), 2 for invalid arguments and for an app without a window, 130 on Ctrl+C.

### 1. Data

```bash
# Optional: download real H1 noise from GWOSC and cut it into chunks (network).
# Overwrites chunks of the same name in data/noise_background/ and keeps all others.
gwd data fetch-noise

# Generate training data (--seed makes it reproducible, --n sets the number of samples).
# The folder must be empty; --overwrite first removes its sample_*.npy and labels.csv.
gwd data generate --seed 1

# The corrected scenario V2 (needs gwpy): simulated aLIGO noise, conditioned together
# with the signal. It is not the default yet; models are trained on it in P6.5.
gwd data generate --scenario train_v2 --seed 1 --out data/training_v2

# Scenario V3 (needs gwpy): V2 whitened with the aLIGO PSD before the band-pass. With the
# same seed it draws the same signals and noise as train_v2.
gwd data generate --scenario train_v3 --seed 1 --out data/training_v3

# Check every sample and the scenario ranges; --plots also writes the report plots
gwd data validate
gwd data validate --plots
```

`gwd data generate` and `gwd eval run` start with a line `Backend: ... | Noise: ...` that names the waveform backend (`pycbc <version>`) and the noise. Without PyCBC they stop before they write, remove, or load anything, print the install hint on stderr, and exit with code 1. Without real noise chunks in `data/noise_background/`, `gwd data generate` prints a warning on stderr. The dataset folder gets a `manifest.json` with the scenario, seed, backend, package versions, Git commit, and the share of real noise; `labels.csv` names the backend, the noise source (`real` or `gaussian`), and the noise chunk of every sample, and since P5.5 its peak and optimal SNR (see "Scenarios and SNR" below). Since P5.6 the samples are float32, 65 536 bytes of data per file; older float64 datasets stay valid. `gwd data generate` writes only into a new or empty folder and otherwise exits with code 1; with `--overwrite` it first removes the folder's `sample_*.npy` files and `labels.csv` and nothing else (other files, subfolders, and `data/noise_background/` stay). `gwd data validate` loads every sample that `labels.csv` lists and checks that it exists, has 16384 float values, and has no NaN or Inf; it also reports `sample_*.npy` files that `labels.csv` does not list, and a `manifest.json` that does not match `labels.csv` or its scenario. If the manifest names a scenario, every signal sample must lie in that scenario's SNR, mass, and merger time ranges, and the classes must be balanced to 50% ± 5%. If `labels.csv` has `peak_snr` and `optimal_snr`, `peak_snr` must equal `snr`, and `optimal_snr` must be positive for every signal sample and 0 for every noise sample. A dataset without a manifest (written before P4.1) is reported as `unknown (v1)` and checked for integrity only. The command lists every finding and exits with code 1 if there is one. The plots go to `plots/validation_dataset/` only with `--plots` (or to the folder given after it). `gwd data fetch-noise` conditions its segment in one piece and writes every 4 s chunk except the first and the last, which hold the ringing of the filters at the ends of the segment; `NNNN` in the file name is the chunk's position in the segment. Files already in the folder are kept, and the files that are not from the new download are named on stderr.

### 2. Training

```bash
gwd train
gwd train --epochs 10 --seed 1
# The same seed gives the same model on the CPU
CUDA_VISIBLE_DEVICES='' gwd train --epochs 10 --seed 1 --deterministic
# Online: fresh train_v3 samples every epoch, drawn by worker processes; no dataset needed
gwd train --online --scenario train_v3 --seed 42
gwd train --online --scenario train_v2 --samples-per-epoch 4096 --workers 8 --epochs 5
```

`gwd train` takes the defaults `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, and `EARLY_STOPPING_PATIENCE` from `src/gwd/config.py`; `--epochs`, `--batch-size`, `--lr`, and `--patience` override them for one run. It trains with mixed precision (`mixed_float16`) only if TensorFlow finds a GPU, otherwise in float32; `--mixed-precision` forces it. It prints the precision it chose in a `Precision:` line. It splits the dataset into 70% train, 15% validation, and 15% test (`VAL_SPLIT` and `TEST_SPLIT` in `src/gwd/config.py`), stratified by class and noise source, and keeps all samples of one real noise chunk in the same part, so no noise realization is seen in training and in the test. The validation part steers early stopping, the checkpoint, and the learning rate; the test part is evaluated once, with the best epoch, after training. A dataset without the `noise_file` column of `labels.csv` (written before P4.1) is split by sample, with a warning that recommends generating it again. Every run is seeded: `--seed` seeds Python, numpy, TensorFlow, and the split; without it, `gwd train` draws a seed and prints it in a `Seed:` line. `--deterministic` also makes TensorFlow's ops deterministic, so the same seed gives the same model on the CPU (it is slower, and on a GPU it is not guaranteed). The samples are loaded as float32. The model goes to `models_registry/gwd_model_YYYYMMDD-HHMMSS.keras`, with a `.json` next to it that records the epochs it really trained, the seed, whether it was drawn, the split (sizes, fractions, `split_grouped`), the validation accuracy and loss of the best epoch and the test accuracy and loss as separate fields, a `dataset_hash` (SHA-256 of `labels.csv` and the data of every sample), and the SHA-256, scenario, backend, and real noise share of the dataset's `manifest.json`; a dataset without a manifest is recorded as `unknown (v1)`. Since P6.6 it also records `preprocessing`, what the model's input went through: `whiten_bandpass` for `TRAIN_V3`, else `bandpass`. The evaluation and the Glitch Hunter read it and give the model data with the same preprocessing. Without a dataset, `gwd train` exits with code 1 and tells you to run `gwd data generate` first. It checks the dataset as `gwd data validate` does for integrity (not for the scenario ranges) and exits with code 1, naming the files, if a sample is missing, unreadable, of the wrong length, or not finite; it no longer skips, cuts, or pads such samples.

`gwd train --online --scenario train_v3` (since P6.7) trains without a dataset: every epoch gets `--samples-per-epoch` fresh samples (default 8192) drawn from the scenario by `--workers` worker processes (default 12, at most the CPU count minus 2), so the CNN cannot learn noise realizations by heart. Validation and test are fixed sets drawn once from the same scenario (`--val-samples`, `--test-samples`, default 2048 each), from random streams disjoint from the training batches; the same `--seed` gives the same batches, whatever the number of workers. Scenarios with real noise (`train_v1`) are refused. On the laptop (12 cores, CPU only) an online epoch of 8192 samples plus validation takes about 53 s with 12 workers, against 31 s for the same training on arrays in memory, because generation and TensorFlow share the CPU; see [architecture.md](docs/architecture.md) for the measurements. The model's `.json` records `training_data: "online"`, a null `dataset_hash` with a `dataset_hash_note`, the scenario as `dataset_scenario` and its `preprocessing`, and the scenario hash, seed streams, and sizes under `online`.

In both modes `gwd train` recomputes the BatchNorm statistics from 1024 fixed training samples before every validation (`BN_RECALIBRATION_SAMPLES` in `src/gwd/config.py`). Without it, the validation accuracy sat at 0.5 for the first epochs, because the moving averages lagged behind the weights, and early stopping could keep a useless early model; the reason and the measurements are in [architecture.md](docs/architecture.md).

### 3. Evaluation

```bash
# Build the default fixed test set once (stress_v1, seed 0, n 1000); the commands below
# also build it on first use and load it afterwards
gwd eval build-testset

# Calibrate the latest model's detection threshold at FPR 1% on the noise-only part
# calib_bg and save it in the model's metadata (--no-save to skip)
gwd eval thresholds

# ROC evaluation of the latest model on the fixed test set (offline), at its threshold
gwd eval run

# The same on the training distribution instead of the stress set
gwd eval run --testset indist_v1

# A V2 and a V3 model on the same events, each with its own preprocessing: the V3 model
# is scored on indist_v3, the same samples as indist_v2, whitened
gwd eval compare --model gwd_model_20261116-204058 --model latest --testset indist_v2

# The same plus the field test on real GWOSC data (network)
gwd eval run --real

# Compare all models in models_registry/ in one ROC plot and a zoom on low FPR
gwd eval compare

# Show leaderboard v2: one table per test set, ranked by ROC AUC
gwd eval leaderboard
```

The latest model is the one with the newest timestamp in its name, `gwd_model_YYYYMMDD-HHMMSS.keras`; other `.keras` files are ordered by their modification time, with a warning. `gwd eval run`, `gwd eval thresholds`, and `gwd eval compare` take `--model` with `latest`, a model name, or a path to a `.keras` file; `compare` takes it more than once. Since P5.7 all simulated numbers come from a fixed, hashed test set: `--testset` is `stress_v1` (the default, the old evaluation distribution `EVAL_V1`) or `indist_v1` (the training distribution `TRAIN_V1`), since P6.2 `indist_v2` or `stress_v2` for models trained on `TRAIN_V2`, and since P6.6 `indist_v3` or `stress_v3` for models trained on `TRAIN_V3`, and `--seed` (default 0) and `--n` (default 1000) pick the set. It is drawn once into `data/eval/`, with its manifest and SHA-256 in `results/testsets/`, and loaded by every later command with the same three values; a set whose arrays no longer match the hash, or that is stale, stops the command with exit code 1 until `gwd eval build-testset --rebuild` (with the same `--testset`, `--seed`, and `--n`) draws it again. `gwd eval run` prints the metrics per peak SNR, optimal SNR, total mass, and noise source and marks bins outside the training ranges as `Extrapolation`. With the same seed and n, `indist_v3` and `stress_v3` hold the same events and noise as `indist_v2` and `stress_v2`, whitened; a model trained with whitening is scored on the V3 set when a V2 set is named, and the other way round, in `run`, `thresholds`, and `compare`, so every model sees its own preprocessing, and V2 and V3 models are compared on the same events. The field test with `--real` whitens the real data for such a model too.

Since P5.8 every model has its own detection threshold. `gwd eval thresholds` picks the lowest threshold at which at most 1% of the noise-only samples of the set's part `calib_bg` raise an alarm, never using the part `test`, and saves it with the false alarm rate it reached, the strategy, and the test set with its SHA-256 in the model's `.json` in `models_registry/`. `gwd eval run`, `gwd eval compare`, and the Glitch Hunter read it from there; a model without one gets 0.5 and a warning. `gwd eval run` reports the TPR, SNR50, and SNR90 at that threshold, always with the FPR it reached on `test` and the number of samples, and every run adds a row with raw numbers and the test set hash to `results/leaderboard_v2.csv`. Models trained before P5.8 have no threshold; calibrate each of them once:

```bash
for m in models_registry/*.keras; do
  gwd eval thresholds --model "$m" --save
done
```

```bash
gwd eval run --model gwd_model_20251205-232013 --seed 1
gwd eval compare --model gwd_model_20251203-220536 --model latest
```

[docs/evaluation.md](docs/evaluation.md) describes the test sets, the metrics, the plots, the threshold calibration, and the leaderboard columns.

### 4. Apps

```bash
# Glitch Hunter: the latest model in models_registry/ rates simulated data or real GWOSC data
gwd app glitch-hunter
gwd app glitch-hunter --model gwd_model_20251205-232013

# Waveform explorer: strain, frequency, and detector signal with sliders for masses, distance, and noise
gwd app waveform

# Interferometer designer: sensitivity curve of a simple learning model with sliders for laser, mirrors, and arm length
gwd app interferometer
```

The interferometer designer is a qualitative learning model, not a calibrated detector model: its curve shows in which direction the sliders and features move the sensitivity, and its dashed reference is PyCBC's aLIGO design curve (`aLIGOZeroDetHighPower`), about 270 times lower at 100 Hz than the default model. The waveform explorer draws its noise from the same aLIGO design curve above 10 Hz; its noise slider multiplies the noise amplitude.

The Glitch Hunter has two simulation modes. 'Training', the start mode, draws from the training distribution `TRAIN_V1`; 'Challenge' draws from `APP_V1`, which is hard on purpose: peak SNR 0.5–2, partly below what the model was trained on. Since P6.2 the switch 'Scenario' draws both modes from V1 or from V2, the corrected scenarios below, and since P6.6 also from V3; it starts in the version the model was trained on (V1 without a model) and is off without gwpy. Real data get the model's preprocessing, so a model trained on `TRAIN_V3` sees whitened real data; a simulated sample whose preprocessing is not the model's says so in its title. 'LIGO random' downloads a known event or a random stretch of O3 noise from GWOSC in the background (about 36 s of data: 4 s plus 16 s on each side for the filters); if the download fails, a simulated sample stands in and the title says why. 'Ask AI' shows the model's score against the alarm threshold, which starts at the model's threshold from `gwd eval thresholds` (0.5, with a warning, if it has none) and moves with the slider, and then the truth, judged right or wrong at the current threshold, also on real data; events outside the training masses are marked OOD. Without a model (or without TensorFlow) the Glitch Hunter starts with 'Ask AI' off; a model named with `--model` that does not exist ends with exit code 1. 'Listen' plays the sample through paplay or aplay (Linux), afplay (macOS), or winsound (Windows), from a temporary file.

### 5. Menu

`python main.py` opens a menu with the data, training, and evaluation steps, the leaderboard, the three apps, and `gwd env`. Each entry runs the `gwd` command with its defaults; one extra entry runs `gwd eval run --real`, the evaluation with the field test. Before "Generate training data" the menu asks whether an existing dataset in `data/training/` may be replaced and adds `--overwrite` on a yes. It works from any folder: each entry runs in the project folder with the same Python, and the menu shows the exit code when a step fails.

## Data and paths

All paths are below the project root, the folder with `pyproject.toml`. `gwd env` prints them.

| Path | Content |
|---|---|
| `data/training/` | The training dataset: `sample_NNNNN.npy`, `labels.csv`, `manifest.json` |
| `data/noise_background/` | Real noise chunks from `gwd data fetch-noise`, `real_noise_NNNN.npy` |
| `data/eval/` | The fixed test sets, `<set>_<seed>_<n>.npz` (since P5.7) |
| `models_registry/` | Trained models (`.keras`) and their metadata (`.json`) |
| `results/` | `leaderboard_v2.csv`, one `gwd eval thresholds` report per model in `thresholds/<model name>.json`, the test set manifests in `testsets/`, and the old `model_leaderboard.csv` and `optimal_thresholds.json` (the single report that every run overwrote until the per-model reports) as archives |
| `plots/` | `validation_dataset/`, `training/`, and `evaluation_models/` |

Git ignores all of these folders; they exist only on the machine that made them. The legacy paths `gw_noise_background/`, `gw_training_data/`, `./model_leaderboard.csv`, and `analysis/model_leaderboard.csv` come from before the cleanup. The code writes none of them; since P5.8 it also no longer writes or copies `results/model_leaderboard.csv`, and `gwd eval leaderboard` only names the old leaderboards that exist as archives. Keep the legacy paths; ignored files cannot be restored from Git.

`GWD_HOME` sets another project root, for example for tests or a second set of data: `GWD_HOME=/path/to/folder gwd env`. `gwd` then warns once on stderr that it uses `GWD_HOME`. Without it, the root is the nearest folder above `src/gwd/config.py` that contains a `pyproject.toml`.

## Scenarios and SNR

Training, evaluation, and the Glitch Hunter draw their samples from three named scenarios, `TRAIN_V1`, `EVAL_V1`, and `APP_V1`, which differ in noise, masses, and SNR range. The samples are scaled to a peak SNR: the peak amplitude of the signal divided by the noise standard deviation, not a matched filter SNR. Since P5.5 `labels.csv` records it as `peak_snr` (and, as before, as `snr`, which stays for older readers) next to `optimal_snr`, the matched filter SNR of the signal: in white noise from its known standard deviation, in a real chunk against that chunk's Welch PSD in 20–300 Hz. Older datasets have only `snr`, which is read as the peak SNR. `gwd data validate` reports the median ratio `optimal_snr / peak_snr` per noise source, and `gwd eval run` reports SNR50 and SNR90 both as peak and as optimal SNR. Since P5.7 the evaluation runs on the fixed test set `stress_v1` (`EVAL_V1`) or `indist_v1` (`TRAIN_V1`) and reports the metrics per bin. Since P6.2 there is also scenario V2 (`TRAIN_V2`, `INDIST_V2`, `STRESS_V2`, `APP_V2`): the chirp is added to raw simulated noise with the aLIGO design PSD, and signal and noise are then notched, band-passed (20–300 Hz), and normalized together, like real data; its samples are scaled to an optimal SNR of 4–20 against that PSD, so the same value means the same detectability for every mass. In a V2 dataset `optimal_snr` is the SNR the signal was scaled to, `peak_snr` is measured, and `noise_source` is `aligo`. V1 stays the default until the models are trained on V2 (P6.5). Since P6.6 scenario V3 (`TRAIN_V3`, `INDIST_V3`, `STRESS_V3`, `APP_V3`) is V2 with one change: the sum of noise and signal is whitened with the aLIGO PSD before the notches and the band-pass, as a matched filter search whitens its data; real data are whitened with a Welch estimate of their own PSD. The motivation is the first V2 model (`gwd_model_20261116-204058`, 10 000 `TRAIN_V2` samples): on `indist_v2` it reached ROC AUC 0.865 and TPR 39.8 % at FPR ≤ 1 %, and needed an optimal SNR of about 14 for 50 % and about 19 for 90 % detection. A matched filter with the right template does far better: in Gaussian noise its statistic has mean ρ_opt and standard deviation 1, so at ρ_opt 8 it lies far above any threshold at FPR 1 %. One suspected cause is that without whitening the strongly coloured aLIGO noise dominates the input at low frequencies. The scenarios and both SNRs are described, with the known skews between them, in [docs/architecture.md](docs/architecture.md#scenarios-and-snr-definition-current-state-v1).

## Tests

```bash
# Fast tests, before every commit
python -m pytest -m 'not slow and not network'

# All tests, before a pull request; needs network access for the network test
python -m pytest
```

| Marker | Tests |
|---|---|
| `slow` | The end-to-end test `tests/test_e2e.py` (the full pipeline as `gwd` commands, then the Glitch Hunter with the trained model and its saved threshold), and a few TensorFlow tests; minutes on a CPU |
| `apps` | The apps under the Agg backend, without a window |
| `golden` | Stored golden values of the PyCBC waveform, the noise, the SNR scaling, a small dataset, and the interferometer |
| `network` | A real download from GWOSC |

The end-to-end, smoke, and golden tests run in a temporary `GWD_HOME`. Tests that need TensorFlow or gwpy are skipped without them. `GWD_REQUIRE_PYCBC=1` makes a missing PyCBC fail the PyCBC tests in `tests/core/` instead of skipping them. Every test that draws a chirp, generates a dataset or test set, or measures the golden values needs PyCBC, as the pipeline does. `tests/test_docs_commands.py` checks that every `gwd` command in this README and in `docs/` parses, and that every relative link works.

CI (`.github/workflows/ci.yml`) runs on every push and pull request: the lockfile check, `ruff format --check . && ruff check .`, and `pytest -m 'not slow and not network and not apps'` with PyCBC but without TensorFlow. The slow, app, and golden tests run in a second job that is started by hand.

## Contributing

Put new code in the `gwd` package in `src/gwd/` and declare dependencies in `pyproject.toml`. Change the documentation in the same pull request as the behavior it describes. Work on a branch: CI must be green before a change is merged into `main`. [AGENTS.md](AGENTS.md) has the rules for commits, tests, data folders, and language.

## License

Copyright (C) 2025–2026 Peter Paprotny

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (SPDX: `GPL-3.0-or-later`).

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [LICENSE](LICENSE) file for the full text.
