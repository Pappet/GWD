# GWD Architecture

This document describes the target structure of the project, where the code is today, and how far the cleanup has come. File and line references were checked at commit `257afbc`.

The detailed steps and all findings are in [aufraeumen/PLAN.md](aufraeumen/PLAN.md) and [aufraeumen/BEFUNDE.md](aufraeumen/BEFUNDE.md), both in German. IDs such as `EVAL-1` refer to the findings list. That folder is removed in P7.4 and stays available in the Git history.

**Work order for AI assistants:** "Run step Px.y. Change only the named files." The rules are in [AGENTS.md](../AGENTS.md).

## Purpose

GWD trains a CNN to detect gravitational wave chirps in noise and evaluates it with ROC metrics. It also includes interactive learning apps. There is currently no valid Matched Filter baseline (see D5).

The workflow is: fetch noise → generate data → validate → train → evaluate → optimize thresholds → compare models.

## Corrections to the old overview

This file replaces `PROJECT_OVERVIEW.md`, which made these false statements:

| Old statement | What is true |
|---|---|
| All executing scripts are in `src/` | `config.py`, `main.py`, and `utility/test_imports.py` are outside `src/`. |
| `requirements.txt` locks dependencies for reproducible results | It pins `pycbc==2.4.0`, which has no usable wheels on Python 3.12, and it does not describe the environment in use. No random seed was set in the code (`core:MISS-3`). P2.1 replaced it with `pyproject.toml` and `requirements-linux-py312.lock`. |
| `gw_training_data` is the ignored data folder | That is a legacy path. Training data is in `data/training/`. |
| `src/evaluate_classic.py` computes the physics limit | It was deleted in `45b9202`, and its results were invalid (`EVAL-6`). |
| CNN and Matched Filter are compared | Not at the moment, see D5. |
| Links to documents next to the overview | They pointed to paths that did not exist. `INSTALLATION_CHECKLIST.md` was removed in P1.4. |

## Components

The target is one installable package, `gwd`, in `src/gwd/` (D8).

| Component | Target | Responsibility |
|---|---|---|
| config | `src/gwd/config.py` | Project root, paths, and constants. Nothing happens on import; writers call `ensure_dirs()` first. |
| core | `src/gwd/core/` | Physics: waveforms, noise, SNR, interferometer learning model, time domain simulator. No file or network access. |
| data | `src/gwd/data/` | Scenarios, the one injection, the one filter chain, GWOSC access, dataset writing, provenance (manifest), and validation. |
| registry | `src/gwd/registry.py` | The one place that finds, selects, and loads models. Metadata and per-model thresholds. |
| training | `src/gwd/training/` | Model definition and training. |
| evaluation | `src/gwd/evaluation/` | Fixed test sets, metrics, thresholds, comparison, leaderboard. |
| apps | `src/gwd/apps/` | GUI, interaction, and audio only. No physics, injection, filtering, or model search of their own. |
| cli | `src/gwd/cli.py`, `src/gwd/env.py`, `main.py` | Dispatch only. `main.py` is a thin menu. |

`core/interferometer.py` and `core/simulation.py` stay in core on purpose. Whether app-specific physics belongs there is a design question (`APP-14`), and the import rules below keep the boundary.

Before the cleanup, the scripts in `src/` found each other through `sys.path` edits with generic top-level names such as `config` and `utils` (`hygiene:HYG-M5`). Since P2.1, tests and scripts both import `gwd.core` (`CORE-5`). From P2.3 to P2.6, the root `config.py` was only a shim for `gwd.config`. Since P2.4, `gwd.data` and `gwd.training` use absolute imports without `sys.path` edits and run with `python -m`, for example `python -m gwd.data.generate`. Since P2.5, all code is in `src/gwd/`: evaluation, apps, and `gwd.data.preprocessing` import absolutely, `src/utils/` is gone, and `src/gwd` has no `sys.path` edits. Since P2.6, the root `config.py` shim and the pytest `pythonpath` entries (`src` and `.`) are gone. Tests import the installed `gwd` package (editable in `.venv`, `pip install -e` in CI), so the old top-level names `config`, `utils`, `dataset`, and `gwd_core` no longer resolve.

**Model registry (since P3.3).** `gwd.registry` is the only model search. `list_models()` returns the `.keras` files in `models_registry/`, oldest first, ordered by the timestamp in the name `gwd_model_YYYYMMDD-HHMMSS`. Before P3.3 the four copies chose by ctime, which changes on chmod, copy, or restore under Linux and WSL (`apps:APPX-4`). On the laptop, ctime picked `gwd_model_20251124-234359` as the newest; the name picks `gwd_model_20251205-232013`. Names without a valid timestamp fall back to mtime with a warning. `resolve()` turns `latest`, a model name, or a path into a file, and `load_model()` imports TensorFlow lazily and loads with `compile=False`, which also avoids Keras' optimizer warning. `read_metadata()` and `write_metadata()` handle the `.json` file next to a model; training names its `.json` with `metadata_path()` since P3.7 and writes it through `write_metadata()` since P4.1. `gwd eval run` and `gwd eval thresholds` take `--model` (default `latest`), and `gwd eval compare` takes it repeatedly (default: all models). The Glitch Hunter loads the latest model through the registry and still exits with rc 1 without one (`APP-8`, P4.4). Since P4.1 it also reads the model's metadata and warns, on the console and in red above the plots, if the real noise share of its training data was 0 (`apps:APPX-1`); for metadata without the share (all models trained before P4.1) it prints a note that it is unknown. No evaluation module and no app imports TensorFlow itself: `tests/test_architecture.py` checks the source, and `tests/test_registry.py` checks at runtime that `run()` of the three evaluation modules handles an unknown model with exit code 1 without loading TensorFlow.

**Command line (since P3.6).** `gwd` (`[project.scripts] gwd = "gwd.cli:main"`) is the one entry point (D8):

| Command | Runs |
|---|---|
| `gwd data fetch-noise [--detector --gps --duration]` | `gwd.data.fetch_noise.fetch_and_slice` (network; overwrites chunks of the same name in `data/noise_background/`) |
| `gwd data generate [--n --seed --out --scenario]` | `gwd.data.generate.generate_dataset`; `--scenario` is `train_v1` (default), `eval_v1`, or `app_v1` |
| `gwd data validate [--data --plots --no-plots]` | `gwd.data.validate.run` |
| `gwd train [--data --epochs --batch-size --lr --patience --mixed-precision --seed]` | `gwd.training.train.train(TrainConfig(...))` with only the options that were given |
| `gwd eval run [--model --real/--no-real --seed]` | `gwd.evaluation.evaluate.run`; `--real` (the field test, network) is the default until P4.3 switches it (D10) |
| `gwd eval thresholds [--model --save/--no-save --seed]` | `gwd.evaluation.thresholds.run`; saves by default and never asks (`EVAL-17`) |
| `gwd eval compare [--model ... --seed]` | `gwd.evaluation.compare.run` |
| `gwd eval leaderboard` | `gwd.evaluation.leaderboard.show` |
| `gwd app waveform\|interferometer\|glitch-hunter` | the app module's `__main__` block, through `runpy` |
| `gwd env` | `gwd.env.main`: versions, extras, waveform backend with the PyCBC version, resolved paths, matplotlib backend |

`gwd.cli` holds only dispatch. Each handler imports its module when it runs, so `gwd --help` and `build_parser()` load no numpy, and `gwd data generate` loads neither TensorFlow nor gwpy (`tests/test_cli.py`). For the `eval` commands, `--seed` calls `seed_everything(seed)`, which seeds Python's `random` and the global numpy RNG. Since P4.1, `gwd data generate` passes the seed to `generate_dataset(seed=...)`, which draws from `np.random.RandomState(seed)` (the same samples as the seeded global RNG), and `gwd train` passes it in `TrainConfig.seed`, which `train()` sets with `tf.keras.utils.set_random_seed` (Python, numpy, and TensorFlow); both record it. Without `--seed` nothing is seeded, as before. P5.3 and P5.6 replace it with an `rng` that is passed down. Exit codes: 0 on success, 1 when a command fails (unknown or missing model, missing dataset, rejected or failed download, no model to compare), 2 for invalid arguments, 130 on Ctrl+C. Since P4.1, `gwd train` checks for `labels.csv` before it imports TensorFlow and reports a missing dataset in one line on stderr instead of a traceback; `gwd data fetch-noise` reports any download error on stderr (`DATA-6`, `tests/data/test_fetch_noise.py` lets gwpy's `fetch_open_data` raise). The modules no longer parse options: `evaluate`, `thresholds`, `compare`, `validate`, `fetch_noise`, and (since P3.7) `train` still run with `python -m` and their defaults, and stop with a pointer to `gwd` if given options. `gwd env` checks packages with `find_spec` and their metadata and imports only `gwd.core.waveforms`, because only that import shows whether PyCBC works. The smoke tests run every pipeline stage and app as a `gwd` command line (`tests/smoke/entrypoints.py`); the `python -c` drivers that set module globals and stubbed the field test are gone.

**Training (since P3.7).** `gwd.training.model.build_model(input_shape, learning_rate=LEARNING_RATE)` builds and compiles the CNN and changes no global state. Its architecture is unchanged and fixed at 1 074 609 parameters by `tests/training/test_model.py`; a new one is a deliberate change (`DATA-26`). `gwd.training.train.train(cfg)` runs one training. `TrainConfig` holds the dataset, model, and plot folders, `epochs`, `batch_size`, `learning_rate`, `patience`, and `mixed_precision`. The defaults are `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, and `EARLY_STOPPING_PATIENCE` in `gwd.config`, which also holds `REDUCE_LR_PATIENCE` and `TEST_SPLIT`; the values are those hard-coded before. Importing the module changes nothing (`DATA-13`): `new_run()` creates the timestamp, the name `gwd_model_YYYYMMDD-HHMMSS`, and the paths when `train()` runs, and the plot uses a matplotlib `Figure` without pyplot. Mixed precision follows D7: `choose_precision()` takes `mixed_float16` if TensorFlow finds a GPU or `--mixed-precision` is set, else float32. Before P3.7, the import set `mixed_float16` everywhere, so CPU and CI runs now train in float32; on a GPU nothing changes. `train()` prints the policy, sets it only for the run, restores the previous one, and records it as `precision_policy` in the `.json`, together with `learning_rate` and `early_stopping_patience`. Since P4.1 the `.json` also has `epochs_trained` (early stopping can end before `epochs`, the maximum), `best_epoch` (1-based, the saved weights), `seed` (null if unseeded), and from the dataset's `manifest.json` its SHA-256 (`dataset_manifest_sha256`), `dataset_scenario`, `waveform_backend`, and `real_noise_fraction`; without a manifest these four are `unknown (v1)` (`DATA-20`). The `gwd train` handler sets `TF_CPP_MIN_LOG_LEVEL=2` and `TF_ENABLE_ONEDNN_OPTS=0` with `setdefault` before it imports TensorFlow, as the module did on import before; values from the caller now win. `python -m gwd.training.train` still runs the defaults, but without those two variables.

**Evaluation metrics (since P4.2).** `gwd.evaluation.metrics` holds the metrics and threshold strategies as pure functions: labels and scores in, numbers out. Importing it loads neither matplotlib nor TensorFlow. `find_optimal_thresholds` moved there from `thresholds.py` and was fixed, not rewritten. Fixed FPR (5% and 1%) takes the highest TPR with FPR ≤ target, ties going to the lower FPR, and fixed TPR (90%) the lowest FPR with TPR ≥ target; before, both took the ROC point nearest to the target, which could lie beyond it (`EVAL-16`). No strategy picks the infinite first threshold of `roc_curve`, which `json` would write as the invalid `Infinity`; if no finite point meets an FPR limit, the threshold lies just above the highest score and detects nothing. F1 comes vectorized from `precision_recall_curve` (`f1_curve`) instead of one `f1_score` call per threshold, and its TPR and FPR are computed at its own threshold instead of the nearest ROC threshold. So the recommended thresholds in `results/optimal_thresholds.json` change; nothing reads them yet (P5.8). `predict_labels` is the one detection rule, score ≥ threshold, as in `roc_curve`. The operating point, the sensitivity curve, and the confusion matrix of `gwd eval run` use it; the latter two counted score > threshold before (`EVAL-15`). The field test keeps `>` until P4.3 rewrites it, and the threshold stays 0.15 until P5.8. `snr_at_rate` finds SNR50 and SNR90 without the bare `except` and keeps 0.0 as a valid value, which the plot and the metrics box showed as missing before (`EVAL-22`). The metrics box shows formatted values instead of raw floats and no longer has the "False Alarms" line, which repeated the FPR; the `far` value is gone from the result, and the leaderboard never read it. `gwd eval compare` loads the models one after another, computes each ROC curve once, and gives each model one color in both panels, by sorted name; before, the bars took their colors in AUC order and the curves in load order. It writes `plot_compare_roc_<timestamp>.png` and `plot_compare_roc_<timestamp>_zoom.png`; before, the zoom overwrote the comparison plot (`EVAL-18`, `HYG-24`). The test sets are still `EVAL_V1`. `tests/evaluation/test_metrics.py` and `tests/evaluation/test_compare.py` check this; the second uses stub models, so it also runs in CI (`EVAL-20`).

**Menu (since P2.4, gwd commands since P3.6).** `main.py` uses only the standard library. `MENU` is a fixed list of `(text, gwd arguments)` with 12 entries: the 11 from P2.5 and `gwd env`. Each entry runs as `subprocess.run([sys.executable, "-m", "gwd.cli", ...])` in the project folder, so the menu works from any working directory and reports the real exit code. The end of input exits with 0 (after an entry, with that entry's exit code) instead of raising `EOFError`; the screen is cleared with `cls` or `clear` without a shell, only on a terminal. `tests/test_menu.py` parses every argument list with `gwd.cli.build_parser()`, so the menu and the CLI cannot drift apart.

**Project root (since P2.3).** `gwd.config` takes `GWD_HOME` if it is set and not blank, and then warns once (`Using GWD_HOME=...`). Otherwise it takes the nearest folder above `src/gwd/config.py` that contains a `pyproject.toml`, which is the repo root in the editable install. If there is none, the import fails and asks for `GWD_HOME`. Without `GWD_HOME`, all paths are the same as before P2.3 (`tests/test_config_paths.py`).

## Import rules

Since P2.6, `tests/test_architecture.py` enforces these rules. It parses every module in `src/gwd/` and `main.py` with `ast` and counts an import as lazy only inside a function body; a `try` block at module level is not lazy. The standard library is always allowed, and anything not listed under "May import" fails, so the test also limits where tensorflow, gwpy, and pycbc may appear at all. Dynamic imports (`__import__`, `importlib`) are not checked. Since P3.4, a separate check fails on any gwpy import statement outside `data/gwosc.py` and `data/preprocessing.py`, lazy or not, in any component including cli. The same test fails on any `sys.path` edit in `src/`, `tests/`, or `main.py`, and on two byte-identical `.py` files (empty `__init__.py` files excepted, `HYG-1`).

Known violations are in the test's `ALLOWLIST`, each with its finding ID and the step that removes it. An entry that no longer matches a real violation fails the test, so the list can only shrink. It must be empty by P7.4. It had 12 entries at P2.6, 8 at P3.3, 5 at P3.4, 3 at P3.6, 2 at P3.7, and since P4.1 it has 1:

| File | Import | Finding | Removed in |
|---|---|---|---|
| `core/waveforms.py` | pycbc at module level (in `try`) | `CORE-17` | P5.2 |


| Component | May import | Must not import |
|---|---|---|
| config | Standard library | Any other `gwd` component, numpy, tensorflow, gwpy, pycbc, matplotlib |
| core | `gwd.core`, standard library, numpy, scipy, pycbc (lazy) | config, data, registry, training, evaluation, apps, cli, tensorflow, matplotlib, gwpy, pandas, sklearn |
| data | core, config, numpy, scipy, pandas, gwpy (lazy, only `gwosc.py` and `preprocessing.py`), matplotlib (lazy, only the validation report) | registry, training, evaluation, apps, cli, tensorflow, sklearn |
| registry | config, standard library, tensorflow (lazy, only `load_model`) | data, training, evaluation, apps, cli, matplotlib, gwpy |
| training | data, core, config, registry, tensorflow, sklearn, matplotlib (lazy), numpy, pandas | evaluation, apps, cli, gwpy |
| evaluation | data, core, config, registry, numpy, sklearn, pandas, matplotlib, tabulate | training, apps, cli, gwpy directly (only through `gwd.data.gwosc`), tensorflow directly (only through registry) |
| apps | core, data, config, registry, numpy, matplotlib, `scipy.io.wavfile`; tensorflow only lazily | training, evaluation, cli, tensorflow at module level, gwpy directly (only through `gwd.data.gwosc`) |
| cli, env | Everything, but only lazily inside handlers | tensorflow, gwpy, pycbc at module level; gwpy directly (only through `gwd.data.gwosc`) |
| `main.py` | Standard library | `gwd` |

## Move table

Every row names a tracked file and where it goes. Files move only with `git mv`.

| Move | Step |
|---|---|
| current: src/gwd_core/__init__.py → target: src/gwd/core/__init__.py | P2.1 |
| current: src/gwd_core/waveforms.py → target: src/gwd/core/waveforms.py | P2.1 |
| current: src/gwd_core/noise.py → target: src/gwd/core/noise.py | P2.1 |
| current: src/gwd_core/interferometer.py → target: src/gwd/core/interferometer.py | P2.1 |
| current: src/gwd_core/simulation.py → target: src/gwd/core/simulation.py | P2.1 |
| current: utility/test_imports.py → target: src/gwd/env.py | P2.1 |
| current: config.py → target: src/gwd/config.py (the temporary root shim was removed in P2.6) | P2.3 |
| current: src/dataset/__init__.py → target: src/gwd/data/__init__.py | P2.4 |
| current: src/dataset/fetch_real_noise.py → target: src/gwd/data/fetch_noise.py | P2.4 |
| current: src/dataset/generate_chirp_dataset.py → target: src/gwd/data/generate.py | P2.4 |
| current: src/dataset/validate_dataset.py → target: src/gwd/data/validate.py | P2.4 |
| current: src/dataset/train_cnn.py → target: src/gwd/training/train.py | P2.4 |
| current: src/evaluate_model_with_roc.py → target: src/gwd/evaluation/evaluate.py | P2.5 |
| current: src/optimize_threshold.py → target: src/gwd/evaluation/thresholds.py | P2.5 |
| current: src/compare_models_roc.py → target: src/gwd/evaluation/compare.py | P2.5 |
| current: src/utils/evaluation.py → target: src/gwd/evaluation/leaderboard.py | P2.5 |
| current: src/utils/data_processing.py → target: src/gwd/data/preprocessing.py | P2.5 |
| current: src/gw_simulator.py → target: src/gwd/apps/waveform_explorer.py | P2.5 |
| current: src/interferometer_simulator.py → target: src/gwd/apps/interferometer_designer.py | P2.5 |
| current: src/glitch_hunter_app.py → target: src/gwd/apps/glitch_hunter/app.py | P2.5 |
| current: tests/test_leaderboard.py → target: tests/evaluation/test_leaderboard.py | P2.5 |
| current: docs/ROC_ANALYSIS_README.md → target: docs/evaluation.md | P7.1 |

These tracked files end without a move of their own:

| File | What happens | Step |
|---|---|---|
| current: requirements.txt → target: replaced by `pyproject.toml` and `requirements-linux-py312.lock` | Removed | P2.1 |
| current: tests/conftest.py → target: none | Removed | P2.1 |
| current: src/show_leaderboard.py → target: src/gwd/evaluation/leaderboard.py | Becomes the function `show()`, then removed | P2.5 |
| current: src/utils/__init__.py → target: none | Removed | P2.5 |
| current: docs/ROC_INTEGRATION_OVERVIEW.md → target: docs/evaluation.md | Useful parts merged, then removed | P7.1 |

Since P5.1 the unit tests for `gwd.core` live in `tests/core/` (`test_waveforms.py`, `test_noise.py`, and `test_interferometer.py` moved there with `git mv`; `test_simulation.py` is new).

**Physics tests (since P5.1).** `tests/core/conftest.py` seeds the global numpy RNG for every test, because `gwd.core` has no `rng` parameter yet (P5.3). The waveform and simulator tests run on both backends, `pycbc` and `newton`, by setting `waveforms.HAS_PYCBC`. On `pycbc`, a silent fallback to Newton fails the test, and with `GWD_REQUIRE_PYCBC=1` a missing pycbc fails instead of skipping. The tests check physical quantities only, never the length of the raw waveform, which depends on the pycbc version: peak at `t_merger` within 1 ms, monotonic frequency up to the merger, `h` proportional to 1/D, the output frequency against the zero-crossing frequency, white noise std, the colored noise spectrum, and the direction in which the interferometer ASD moves with arm length, laser power, and features. Known bugs are `xfail(strict=True, raises=AssertionError)` with the finding ID in the reason: `CORE-1`, `CORE-2`, `CORE-3`, `CORE-8`, `CORE-9`, and `CORE-15`, 10 xfails after parametrization. A fix that makes one pass turns it into a failure until its mark is removed. `pytest tests/core -q -rx` lists them.

## Functions to merge

Each of these has several copies today. P3 leaves exactly one.

| Logic | Copies | Where today | Target |
|---|---|---|---|
| Signal injection | 6 | `generate_chirp_dataset.py:126`, `evaluate_model_with_roc.py:98` and `:142`, `optimize_threshold.py:59`, `compare_models_roc.py:67`, `glitch_hunter_app.py:269` | `gwd.data.injection` (done in P3.5) |
| Conditioning (notch and bandpass) | 3 | `utils/data_processing.py:9`, `fetch_real_noise.py:37`, `glitch_hunter_app.py:169`. `data_processing.py` has no 240 Hz notch. | `condition_strain` in `gwd.data.preprocessing` (done in P3.2) |
| z-score | 10 | `generate_chirp_dataset.py:64`, `train_cnn.py:73`, `evaluate_model_with_roc.py:102`, `:143`, `:290`, `:327`, `optimize_threshold.py:63`, `compare_models_roc.py:71`, `glitch_hunter_app.py:181`, `:283`. Two of them add an epsilon. | `normalize` in `gwd.data.preprocessing` (done in P3.2) |
| Model search | 4 | `utils/evaluation.py:11`, `optimize_threshold.py:32`, `compare_models_roc.py:28`, `glitch_hunter_app.py:49` | `gwd.registry` (done in P3.3) |
| Event list | 2 | `REAL_EVENTS` in `evaluate_model_with_roc.py:45`, `KNOWN_EVENTS` in `glitch_hunter_app.py:24` | `EVENTS` in `gwd.data.gwosc` (done in P3.4) |
| GWOSC download | 4 | `TimeSeries.fetch_open_data` in `fetch_real_noise.py:29`, `evaluate_model_with_roc.py:284` and `:317`, `glitch_hunter_app.py:161` | `fetch_strain` in `gwd.data.gwosc` (done in P3.4) |
| Detection threshold | 3 values | `0.15` in `evaluate_model_with_roc.py:42`, `0.46` in `glitch_hunter_app.py:73`. `0.75` survives only in comments at `evaluate_model_with_roc.py:121` and `glitch_hunter_app.py:73`. | Per-model threshold in `gwd.registry` (P5.8) |

**Preprocessing (since P3.2).** `gwd.data.preprocessing` has two public functions. `condition_strain(ts, fs, notches=NOTCH_FREQS, band=BANDPASS)` resamples a gwpy `TimeSeries` to `fs`, applies one notch per frequency, then the band-pass, and returns a numpy array. It is a band-pass, not whitening. It calls the methods of the series it gets and never imports gwpy itself. The noise download, the field test in `evaluate`, and the Glitch Hunter use it; it is bit-identical to the old download chain. The field test now also notches 240 Hz, the one behaviour change of P3.2. `normalize(x, eps=1e-10)` is the z-score `(x - mean) / (std + eps)`. The eps is absolute: on conditioned strain (std near 1e-21) it dominates the std, so the field test and the Glitch Hunter pass `eps=0` there, as they had no eps before. All other callers use the default; for the synthetic test sets, which had no eps, this changes values by a relative 1e-10.

**GWOSC access (since P3.4).** `gwd.data.gwosc` is the only place that downloads from GWOSC. `fetch_strain(detector, start, end, pad_s=0, cache=True, verbose=False)` imports gwpy inside the function, calls `TimeSeries.fetch_open_data` for `[start - pad_s, end + pad_s)`, and returns the raw `TimeSeries`. It raises `StrainError` (a `ValueError`) if the sample count does not match the requested span at the series' sample rate, or if any sample is NaN; nothing is resized or padded. `pad_s=0` reproduces the 4 s windows of the field test and the Glitch Hunter; the buffer with a crop after conditioning follows in P4.3. `cache=True` lets gwpy keep a local copy of the downloaded file (astropy's download cache), which the old calls did not. `EVENTS` is a tuple of `Event(name, gps, kind, mass1, mass2, in_training_range)` with the nine events of the old Glitch Hunter list, in the same order and with the same GPS times. The masses are rounded source-frame medians from GWTC-1 and GWTC-2, written down from the literature and not verified offline. `in_training_range` means both masses lie in the 5–100 M☉ of `TRAIN_V1` (since P3.5; before, of `generate_random_bbh_parameters`); GW170817 (BNS) and GW190814 (2.6 M☉ secondary) are out of range. The field test in `evaluate` picks its five events from `EVENTS` by name (`FIELD_TEST_EVENTS`, old order); a rejected download scores the event as `None` instead of an `np.resize`d window. The Glitch Hunter draws from `EVENTS` and falls back to the simulation on a rejected download instead of trimming or zero-padding. `fetch_noise` downloads its segment through `fetch_strain` and writes nothing if the download is rejected. Importing `gwd.data.gwosc`, `fetch_noise`, `evaluate`, or the Glitch Hunter no longer loads gwpy (`tests/data/test_gwosc.py`). The download length `NOISE_FETCH_SECONDS` stays in `gwd.config`: moving it next to `START_GPS` in `fetch_noise.py` would bring back the literal 4096 that the P3.1 check forbids in `gwd.data`, and P6.3 replaces the single segment with configurable windows anyway. Tests mock `fetch_open_data` or `fetch_strain`; one real download runs only under the `network` marker, which CI never selects.

## Scenarios and SNR definition (current state, V1)

Training, evaluation, and the Glitch Hunter draw from different distributions. Since P3.5 these are named as `TRAIN_V1`, `EVAL_V1`, and `APP_V1` in `gwd.data.scenarios`, without changing them. P6.2 adds a corrected V2.

| | `TRAIN_V1` | `EVAL_V1` | `APP_V1` |
|---|---|---|---|
| Used by | `gwd.data.generate` | `gwd.evaluation.evaluate` (ROC set and sensitivity curve), `thresholds`, `compare` | `gwd.apps.glitch_hunter.app` (simulation mode) |
| Peak SNR | 1.0–12.0 | 0.5–2.5 | 0.5–2.0 |
| Masses | Random: m1 log-uniform 5–100 M☉, m2 log-uniform 5–m1 | Fixed, 30/30 M☉ | Fixed, 30/30 M☉ |
| Merger time | 2.0–3.6 s | 2.0–3.5 s | 2.0–3.8 s |
| Noise | 80% real if chunks exist, otherwise white | White | White |
| Peak reference (`peak_reference`) | Measured noise std of each sample (`noise_std`) | Nominal std 1: chirp (peak 1) times SNR (`unit`) | Same as `EVAL_V1` |

**Injection (since P3.5).** `gwd.data.injection` is the only place that draws and injects signals. `make_sample(scenario, rng=None, has_signal=None, snr=None, noise_provider=None)` returns a `Sample` with `data = noise + signal`, the noise, the scaled signal, the `InjectionParams` (masses, distance, merger time, SNR), the SNR it scaled to, and `noise_kind` (`real` or `white`). `draw_params` draws masses, then merger time, then SNR; a fixed `snr` (the sensitivity curve) is not drawn. `make_test_set` builds the normalized `(X, y)` that `evaluate`, `thresholds`, and `compare` each used to build in their own loop. `rng` is anything with the numpy `RandomState` methods and defaults to the global `np.random`, as before. `gwd.core` has no `rng` parameter until P5.3, so the module draws white noise and random masses itself, with the same calls as `generate_gaussian_noise` and `generate_random_bbh_parameters`; the pipeline no longer calls the latter. The real noise provider (`real_noise_provider`, `load_random_real_noise`) lists the chunks with `sorted(glob)` and picks one with the same `rng`, so a seed picks the same chunk on every file system; before, `random.choice` from Python's `random` module picked from an unsorted list. `Scenario.hash()` is a SHA-256 prefix of all fields that is the same in every process, unlike Python's `hash()`; `tests/data/test_injection.py` pins the three V1 hashes. A scenario with `condition_signal=True` raises `NotImplementedError` until V2. Since P4.1 a `Sample` also has `noise_file` (the chunk's file name) and `waveform_backend` (`pycbc` or `newton`, from `chirp_and_backend`), and the real noise provider returns `(noise, file name)`. A chunk that is too short or unreadable still falls back to white noise, but now warns and is recorded as white instead of real; the random stream is unchanged.

The draw order for `TRAIN_V1` is the one of the old generator, so its files are bit-identical without real noise (golden (d), and 40 samples on the PyCBC backend), and so is the sensitivity curve. The ROC, threshold, and comparison sets now draw the merger time before the SNR, and the Glitch Hunter draws the label before the noise and with `rand() > 0.5` instead of `np.random.choice`. The distributions are the same, but the same global seed gives other samples there; no seed was set before, so no stored result depended on it. `tests/data/test_injection.py` checks bit-identical batches for the same seed, also with the chunk list in reverse order, the bounds of every drawn parameter, and the V1 scale of real chunks. `test_known_skew` (`xfail(strict=True)`) asks for `TRAIN_V1` and `EVAL_V1` to draw from the same distribution and fails until V2.

**SNR definition.** The `snr` column in `labels.csv` is a peak amplitude ratio: the signal's peak divided by the noise standard deviation. It is not a matched filter SNR, so detectability at the same value depends strongly on the masses and the noise type (`data-training:SNR-DEF`, `evaluation:MISS-2`). P5.5 records `peak_snr` and `optimal_snr`.

**Provenance (since P4.1).** Two switches in the environment decide what the data contains: whether PyCBC can be imported, and whether `data/noise_background/` has chunks. `gwd.core.waveforms` no longer prints: a missing PyCBC is logged at info level on import, so `python -c 'import gwd.core.waveforms'` writes nothing to stdout. `active_backend()` returns `pycbc <version>` or `newton`. A PyCBC error that falls back to Newton warns with `WaveformFallbackWarning`, and `chirp_and_backend()` returns the backend of each chirp; `generate_astrophysical_chirp` returns its first two values. `gwd.data.provenance` prints the line `Backend: ... | Noise: ...` at the start of `gwd data generate` and `gwd eval run`; for Newton, and in `generate` for a scenario with real noise but no chunks, a warning block follows on stderr. `labels.csv` has three columns after the old seven: `waveform_backend` (`pycbc` or `newton`, empty without signal), `noise_source` (`real` or `gaussian`; `noise_kind` `white` is written as `gaussian`), and `noise_file` (the chunk's file name, empty for Gaussian noise). Empty cells read as NaN in pandas unless `keep_default_na=False`. `<out>/manifest.json` holds `format`, `scenario` and `scenario_hash`, `seed`, `n`, `waveform_backend` (`active_backend()`), `signal_backends` (signal samples per backend), `real_noise_fraction` and `real_noise_samples` (what was drawn), `scenario_real_noise_fraction`, `noise_chunks`, `noise_dir`, the versions of gwd, numpy, pycbc, and pandas, `git_commit` and `git_dirty` (null outside a Git checkout), and `labels_sha256`. It has no timestamp, so the same data from the same commit, noise folder, and working tree state gives the same manifest and hash (`tests/data/test_generate.py`). Training copies its SHA-256 into the model's `.json` (see Training above). The golden dataset values gained the three columns in a `golden:` commit; nothing else changed. `tests/data/test_generate.py` checks the columns, the manifest, the start line, and both warnings; the smoke test checks them in the pipeline and in the model's `.json`.

**Known skews:**

- The evaluation SNR range lies mostly below or at the edge of the training range (`DATA-1`, `EVAL-1`).
- Training uses up to 80% real, bandpass-filtered noise, while simulation evaluation uses 100% white noise (`EVAL-2`, `data-training:NOISE-SKEW`).
- Evaluation uses only 30/30 M☉, while training uses random masses (`EVAL-3`).
- The Glitch Hunter's simulation mode produces data outside the training distribution (`APP-2`).
- The peak SNR refers to the measured noise std in training, but to a nominal std of 1 in evaluation and the Glitch Hunter. For white noise of 16384 samples the measured std is within about half a percent of 1; for real chunks the scale skew below applies.
- Real noise is filtered, but the signal is added unfiltered. V2 injects first and then conditions both together.
- Real noise chunks reach the CNN at a much smaller scale than white noise. `generate` normalizes the raw chunks (std near 1e-21) with the default eps, which dominates, so their std is about 1e-11 instead of 1. `train` normalizes again with the same eps. For the 1024 local chunks, the noise-only inputs end up with a std of 0.07–0.21, against 1.0 for white noise. Measured in P3.2 and kept as V1.
- Backend and noise source are switches in the environment (`hygiene:HYG-M1`). Since P4.1 they are printed and recorded (see Provenance above), but not removed: a missing PyCBC still means Newton (P5.2), and missing chunks still mean white noise.

## Starting point

### Facts

| | Finding | Consequence |
|---|---|---|
| **F1** | PC: pycbc 2.10.0 is installed, and TF 2.20 sees the RTX 3060. | The 8 PC models come from an environment with PyCBC. Model metadata does not record the backend, so this rests on the environment, not the model files. |
| **F2** | PC: `data/noise_background/` with 1024 chunks (133 MB), `data/training/` 1.3 GB. Laptop: 512 chunks in the legacy path `gw_noise_background/`. | Since P0.1 the 1024 PC chunks are also in `data/noise_background/` on the laptop, so generated data uses real noise. `gw_noise_background/` stays untouched. |
| **F3** | Machines over time: Linux laptop → Windows PC `Tesseract` with RTX 3060 (WSL Ubuntu 24.04) → now `tuxedo-laptop` (Ryzen AI 9 HX 370, Fedora, no GPU for TF). | Work happens on the laptop, with CPU training. Large training runs (P6.5) happen on the PC under WSL. |
| **F4** | Laptop: 9 models in `models_registry/` (8 from the PC, 24.11.–05.12.2025, and 1 from 21.11.2025), `analysis/model_leaderboard.csv` with a byte-identical copy in `results/` since P1.2, `plots/` from the PC, and two baselines (`physics_baseline.json` from the laptop, `physics_baseline.pc-tesseract.json` from the PC). Legacy paths unchanged: `gw_noise_background/`, `gw_training_data/`, `./model_leaderboard.csv`. Local and ignored: `docs/*.pdf`, `PROJECT_MEMORY.md`. Local and empty: `examples/`. | No step touches the PDFs or `examples/`. Both baselines are invalid (`EVAL-6`), so `Physik_Gap` is always `-`, and old values are hidden in the display. |

### Environment

- Laptop `.venv`: Python 3.12.14, built from `~/gwd-env-pc.txt` (`pip freeze` of the PC environment without the `nvidia-*` packages) plus pytest 8.3.4.
- Key versions: tensorflow 2.20.0, keras 3.12.0, numpy 1.26.4, pandas 2.3.3, pycbc 2.10.0, lalsuite 7.26.1, gwpy 3.0.13. All 9 models load.
- `requirements-linux-py312.lock` (since P2.1) holds these pins. It was compiled with `~/gwd-env-pc.txt` without `nvidia-*` as constraints and matches the laptop `.venv`, which has ruff 0.17.5 since P2.2. The README describes how to install from it and how to update it (P7.3).
- CI (since P2.2, `.github/workflows/ci.yml`): job `test` on every push and pull request first checks that the lockfile matches `pyproject.toml` (since P7.3), then installs `-c requirements-linux-py312.lock -e '.[physics,gwdata,dev]'` (PyCBC, no TensorFlow), runs `ruff format --check . && ruff check .`, then `GWD_REQUIRE_PYCBC=1 pytest -m 'not slow and not network and not apps'`. Job `slow` runs only on `workflow_dispatch` with the full lockfile and `pytest -m 'slow or apps or golden'`.
- Commits use the noreply address, because GitHub rejects pushes with the private one.

### Decisions

"Default applies" means the owner has not answered, so the plan's documented default is in effect.

| ID | Question | Answer | Status | Affects |
|---|---|---|---|---|
| **D1** | PyCBC required, or a corrected Newton backend? | A: PyCBC required, delete Newton. Follows from F1 and F3. | Effectively decided; owner confirms before P6.1 | P5.2, P6.1 |
| **D2** | SNR definition for V2? | B: optimal SNR drives V2. V1 stays. | Default applies | P6.2, P6.5 |
| **D3** | Noise for V2? | B: simulated noise. Real noise (P6.3) is parked. | Default applies | P6.2, P6.3, P6.5 |
| **D4** | Glitch Hunter: game, tool, or both? | A: "Training" and "Challenge" modes. | Default applies | P4.4, P5.7 |
| **D5** | Rewrite or drop the Matched Filter baseline? | B: `Physik_Gap` is dropped. | Default applies | P5.8, P6.4 |
| **D6** | Interferometer: learning model or calibrated? | A: honestly labeled learning model. | Default applies | P4.5, P5.4 |
| **D7** | Python version, lockfile, mixed precision? | B: Python ≥ 3.11, CI on 3.12, plus a lockfile. Like A, mixed precision only with a GPU (since P3.7). | Decided | P2.1, P3.7, P7.3 |
| **D8** | One package or three? Keep the menu? | A: one package `gwd`, one command `gwd`, `main.py` as a menu. | Decided | P2.1 onward |
| **D9** | Version result files? | A: `results/` stays local. | Default applies | P1.1, P5.7, P5.8 |
| **D10** | Threshold, ranking, field test? | A: per-model threshold at 1% FPR, ranking by ROC_AUC, field test optional. | Default applies | P4.3, P4.4, P5.8 |
| **D11** | Language? | B: everything in English. Until P3.6, only lines a step changes anyway get translated. From P3.6, a rewritten module is fully translated in its own `docs:` commit. P7.1 does the rest. `docs/aufraeumen/` stays German. | Decided | All steps |
| **D12** | License? | GPL-3.0-or-later, copyright Peter Paprotny. `LICENSE` and the README section are done (PR #1), and the `pyproject.toml` entry since P2.1 (#17). | Decided | P2.1 |

### Reference commits

| Commit or tag | Meaning | Used by |
|---|---|---|
| `9bde4552ffa551e083dd541b4ea4e622ee9c1172` (`9bde455`) | Line-ending commit from P1.6, no content change | Listed in `.git-blame-ignore-revs` since P2.2 |
| `1db803a1ac70f6277bfd9659c9061b0de730537a` (`1db803a`) | `ruff format` commit from P2.2, formatting only | Listed in `.git-blame-ignore-revs`. The hash changes if the stack is rebased or squash-merged; then update the file |
| `pre-cleanup` (`2eec4b5`) | State before the cleanup | Rollback and restoring tool state |

## Step checklist

Each step PR ticks its own step, so a tick takes effect when that PR is merged.

| Done | Step | Title | PR |
|---|---|---|---|
| [x] | P0.1 | Back up local data, collect facts, rollback tag, rules for AI assistants | #2 |
| [x] | P0.2 | Testability hook: `GWD_HOME` in `config.py` | #3 |
| [x] | P0.3 | Make `validate_dataset` runnable | #5 |
| [x] | P0.4 | Immediate leaderboard fixes and results moratorium | #6, #7 |
| [x] | P0.5 | pytest configuration, pipeline and app smoke tests | #8 |
| [x] | P0.6 | Golden values without a version key | #9 |
| [x] | P1.1 | Rewrite `.gitignore`, remove tool state and `.planning` from the index | #11 |
| [x] | P1.2 | Move leaderboard and thresholds to `results/` | #12 |
| [x] | P1.3 | Remove duplicate scripts in `analysis/` and `apps/` | #13 |
| [x] | P1.4 | Remove dead scripts and misleading AI notes | #14 |
| [x] | P1.5 | Write down the target picture: this document | #16 |
| [x] | P1.6 | Normalize line endings | #15 |
| [x] | P2.1 | `pyproject.toml`, lockfile, `gwd_core` → `gwd.core`, update tests | #17 |
| [x] | P2.2 | ruff baseline, format commit, lean CI | #18 |
| [x] | P2.3 | `config` → `gwd/config.py`, `ensure_dirs()` instead of folders on import | #20 |
| [x] | P2.4 | Move data and training scripts, menu independent of working directory | #21 |
| [x] | P2.5 | Move evaluation, utils, and apps; start of the registry | #22 |
| [x] | P2.6 | Remove the shim and last `sys.path` edits; architecture test | #23 |
| [x] | P3.1 | Central constants instead of hard-coded values | #24 |
| [x] | P3.2 | One conditioning and one normalization function | #27 |
| [x] | P3.3 | Model registry: one model search, selection by timestamp, `--model` | #25 |
| [x] | P3.4 | GWOSC access and shared event list | #28 |
| [x] | P3.5 | Central injection with named V1 scenarios | #29 |
| [x] | P3.6 | Unified `gwd` CLI, `gwd env`, thin menu | #30 |
| [x] | P3.7 | Split training into `model.py` and `train.py` | #31 |
| [x] | P4.1 | Make silent switches visible: logging, manifest, metadata, exit codes | #32 |
| [ ] | P4.2 | Simulation evaluation fixes, `metrics.py` as pure functions | |
| [ ] | P4.3 | Real data field test: padding, honest counting, only with `--real` | |
| [ ] | P4.4 | Untangle Glitch Hunter: GUI, sources, audio; fix threading; modes | |
| [ ] | P4.5 | Testable demo apps | |
| [ ] | P4.6 | Validator and loading: full integrity and range checks | |
| [x] | P5.1 | Physics tests with real asserts | #26 |
| [ ] | P5.2 | Explicit waveform backend, no silent fallback in the pipeline | |
| [ ] | P5.3 | Core details: `time_axis`, `rng`, `t_merger` check, constants, dead code | |
| [ ] | P5.4 | Correct noise normalization; interferometer as an honest learning model | |
| [ ] | P5.5 | Name and measure SNR honestly: `peak_snr` and `optimal_snr` | |
| [ ] | P5.6 | Training: grouped three-way split, seeds, float32, full metadata | |
| [ ] | P5.7 | Fixed, hashed test sets; metrics per SNR bin, mass bin, and noise type | |
| [ ] | P5.8 | Per-model threshold at fixed FPR; leaderboard v2 with raw numbers | |
| [ ] | P6.1 | Delete or correct the Newton backend (D1) | |
| [ ] | P6.2 | Scenario V2: inject first, then condition together | |
| [ ] | P6.3 | Real noise for V2 (D3) | |
| [ ] | P6.4 | Rewrite the Matched Filter baseline (D5) | |
| [ ] | P6.5 | Retrain and re-evaluate, switch to V2, archive old artifacts | |
| [ ] | P7.1 | New README, `docs/evaluation.md`, documentation command test | |
| [x] | P7.2 | License entry in `pyproject.toml` (D12) | #17 (done in P2.1) |
| [x] | P7.3 | CI on the lockfile, README on updating it (D7) | #19 |
| [ ] | P7.4 | Final check: end to end, empty allowlist, checklist | |
