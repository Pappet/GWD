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
| `requirements.txt` locks dependencies for reproducible results | It pins `pycbc==2.4.0`, which has no usable wheels on Python 3.12, and it does not describe the environment in use. No random seed was set in the code (`core:MISS-3`). A lockfile comes in P2.1. |
| `gw_training_data` is the ignored data folder | That is a legacy path. Training data is in `data/training/`. |
| `src/evaluate_classic.py` computes the physics limit | It was deleted in `45b9202`, and its results were invalid (`EVAL-6`). |
| CNN and Matched Filter are compared | Not at the moment, see D5. |
| Links to documents next to the overview | They pointed to paths that did not exist. `INSTALLATION_CHECKLIST.md` was removed in P1.4. |

## Components

The target is one installable package, `gwd`, in `src/gwd/` (D8).

| Component | Target | Responsibility |
|---|---|---|
| config | `src/gwd/config.py` | Project root, paths, and constants. Nothing happens on import. |
| core | `src/gwd/core/` | Physics: waveforms, noise, SNR, interferometer learning model, time domain simulator. No file or network access. |
| data | `src/gwd/data/` | Scenarios, the one injection, the one filter chain, GWOSC access, dataset writing and validation. |
| registry | `src/gwd/registry.py` | The one place that finds, selects, and loads models. Metadata and per-model thresholds. |
| training | `src/gwd/training/` | Model definition and training. |
| evaluation | `src/gwd/evaluation/` | Fixed test sets, metrics, thresholds, comparison, leaderboard. |
| apps | `src/gwd/apps/` | GUI, interaction, and audio only. No physics, injection, filtering, or model search of their own. |
| cli | `src/gwd/cli.py`, `src/gwd/env.py`, `main.py` | Dispatch only. `main.py` is a thin menu. |

`core/interferometer.py` and `core/simulation.py` stay in core on purpose. Whether app-specific physics belongs there is a design question (`APP-14`), and the import rules below keep the boundary.

Today, modules find each other through `sys.path` edits with generic top-level names such as `config`, `utils`, and `dataset` (`hygiene:HYG-M5`), and tests import `src.gwd_core` while scripts import `gwd_core` (`CORE-5`). The move below removes both.

## Import rules

P2.6 enforces these rules with `tests/test_architecture.py`.

| Component | May import | Must not import |
|---|---|---|
| config | Standard library | Any other `gwd` component, numpy, tensorflow, gwpy, pycbc, matplotlib |
| core | `gwd.core`, standard library, numpy, scipy, pycbc (lazy) | config, data, registry, training, evaluation, apps, cli, tensorflow, matplotlib, gwpy, pandas, sklearn |
| data | core, config, numpy, scipy, pandas, gwpy (lazy, only `gwosc.py` and `preprocessing.py`), matplotlib (lazy, only the validation report) | registry, training, evaluation, apps, cli, tensorflow, sklearn |
| registry | config, standard library, tensorflow (lazy, only `load_model`) | data, training, evaluation, apps, cli, matplotlib, gwpy |
| training | data, core, config, registry, tensorflow, sklearn, matplotlib (lazy), numpy, pandas | evaluation, apps, cli, gwpy |
| evaluation | data, core, config, registry, sklearn, pandas, matplotlib, tabulate | training, apps, cli, gwpy directly (only through `gwd.data.gwosc`), tensorflow directly (only through registry) |
| apps | core, data, config, registry, matplotlib, `scipy.io.wavfile` | training, evaluation, cli, tensorflow and gwpy at module level |
| cli, env | Everything, but only lazily inside handlers | tensorflow, gwpy, pycbc at module level |
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
| current: config.py → target: src/gwd/config.py (a temporary root shim stays until P2.6) | P2.3 |
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

The unit tests in `tests/test_waveforms.py`, `tests/test_noise.py`, and `tests/test_interferometer.py` belong in `tests/core/` in the target structure.

## Functions to merge

Each of these has several copies today. P3 leaves exactly one.

| Logic | Copies | Where today | Target |
|---|---|---|---|
| Signal injection | 6 | `generate_chirp_dataset.py:126`, `evaluate_model_with_roc.py:98` and `:142`, `optimize_threshold.py:59`, `compare_models_roc.py:67`, `glitch_hunter_app.py:269` | `gwd.data.injection` (P3.5) |
| Conditioning (notch and bandpass) | 3 | `utils/data_processing.py:9`, `fetch_real_noise.py:37`, `glitch_hunter_app.py:169`. `data_processing.py` has no 240 Hz notch. | `gwd.data.preprocessing` (P3.2) |
| z-score | 10 | `generate_chirp_dataset.py:64`, `train_cnn.py:73`, `evaluate_model_with_roc.py:102`, `:143`, `:290`, `:327`, `optimize_threshold.py:63`, `compare_models_roc.py:71`, `glitch_hunter_app.py:181`, `:283`. Two of them add an epsilon. | `gwd.data.preprocessing` (P3.2) |
| Model search | 4 | `utils/evaluation.py:11`, `optimize_threshold.py:32`, `compare_models_roc.py:28`, `glitch_hunter_app.py:49` | `gwd.registry` (P3.3) |
| Event list | 2 | `REAL_EVENTS` in `evaluate_model_with_roc.py:45`, `KNOWN_EVENTS` in `glitch_hunter_app.py:24` | `gwd.data.gwosc` (P3.4) |
| Detection threshold | 3 values | `0.15` in `evaluate_model_with_roc.py:42`, `0.46` in `glitch_hunter_app.py:73`. `0.75` survives only in comments at `evaluate_model_with_roc.py:121` and `glitch_hunter_app.py:73`. | Per-model threshold in `gwd.registry` (P5.8) |

## Scenarios and SNR definition (current state, V1)

Training, evaluation, and the Glitch Hunter draw from different distributions. P3.5 names these as `TRAIN_V1`, `EVAL_V1`, and `APP_V1` without changing them. P6.2 adds a corrected V2.

| | `TRAIN_V1` | `EVAL_V1` | `APP_V1` |
|---|---|---|---|
| Used by | `generate_chirp_dataset.py` | `evaluate_model_with_roc.py`, `optimize_threshold.py`, `compare_models_roc.py` | `glitch_hunter_app.py` |
| Peak SNR | 1.0–12.0 | 0.5–2.5 | 0.5–2.0 |
| Masses | Random, 5–100 M☉ | Fixed, 30/30 M☉ | Fixed, 30/30 M☉ |
| Merger time | 2.0–3.6 s | 2.0–3.5 s | 2.0–3.8 s |
| Noise | 80% real if chunks exist, otherwise white | White | White |

**SNR definition.** The `snr` column in `labels.csv` is a peak amplitude ratio: the signal's peak divided by the noise standard deviation. It is not a matched filter SNR, so detectability at the same value depends strongly on the masses and the noise type (`data-training:SNR-DEF`, `evaluation:MISS-2`). P5.5 records `peak_snr` and `optimal_snr`.

**Known skews:**

- The evaluation SNR range lies mostly below or at the edge of the training range (`DATA-1`, `EVAL-1`).
- Training uses up to 80% real, bandpass-filtered noise, while simulation evaluation uses 100% white noise (`EVAL-2`, `data-training:NOISE-SKEW`).
- Evaluation uses only 30/30 M☉, while training uses random masses (`EVAL-3`).
- The Glitch Hunter's simulation mode produces data outside the training distribution (`APP-2`).
- Real noise is filtered, but the signal is added unfiltered. V2 injects first and then conditions both together.
- Backend and noise source are silent switches and are not recorded (`hygiene:HYG-M1`). P4.1 logs them.

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
- Until P2.1 creates `requirements-linux-py312.lock`, `~/gwd-env-pc.txt` is the only source of these pins. `requirements.txt` does not describe this environment.
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
| **D7** | Python version, lockfile, mixed precision? | B: Python ≥ 3.11, CI on 3.12, plus a lockfile. | Decided | P2.1, P3.7, P7.3 |
| **D8** | One package or three? Keep the menu? | A: one package `gwd`, one command `gwd`, `main.py` as a menu. | Decided | P2.1 onward |
| **D9** | Version result files? | A: `results/` stays local. | Default applies | P1.1, P5.7, P5.8 |
| **D10** | Threshold, ranking, field test? | A: per-model threshold at 1% FPR, ranking by ROC_AUC, field test optional. | Default applies | P4.3, P4.4, P5.8 |
| **D11** | Language? | B: everything in English. Until P3.6, only lines a step changes anyway get translated. From P3.6, a rewritten module is fully translated in its own `docs:` commit. P7.1 does the rest. `docs/aufraeumen/` stays German. | Decided | All steps |
| **D12** | License? | GPL-3.0-or-later, copyright Peter Paprotny. `LICENSE` and the README section are done (PR #1). Only the `pyproject.toml` entry is open. | Decided | P2.1 |

### Reference commits

| Commit or tag | Meaning | Used by |
|---|---|---|
| `9bde4552ffa551e083dd541b4ea4e622ee9c1172` (`9bde455`) | Line-ending commit from P1.6, no content change | **P2.2 adds it to `.git-blame-ignore-revs`**, together with the `ruff format` commit |
| `pre-cleanup` (`2eec4b5`) | State before the cleanup | Rollback and restoring tool state |

## Step checklist

The checklist is kept up to date with each step. Steps are done once their PR is merged.

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
| [ ] | P1.5 | Write down the target picture: this document | |
| [x] | P1.6 | Normalize line endings | #15 |
| [ ] | P2.1 | `pyproject.toml`, lockfile, `gwd_core` → `gwd.core`, update tests | |
| [ ] | P2.2 | ruff baseline, format commit, lean CI | |
| [ ] | P2.3 | `config` → `gwd/config.py`, `ensure_dirs()` instead of folders on import | |
| [ ] | P2.4 | Move data and training scripts, menu independent of working directory | |
| [ ] | P2.5 | Move evaluation, utils, and apps; start of the registry | |
| [ ] | P2.6 | Remove the shim and last `sys.path` edits; architecture test | |
| [ ] | P3.1 | Central constants instead of hard-coded values | |
| [ ] | P3.2 | One conditioning and one normalization function | |
| [ ] | P3.3 | Model registry: one model search, selection by timestamp, `--model` | |
| [ ] | P3.4 | GWOSC access and shared event list | |
| [ ] | P3.5 | Central injection with named V1 scenarios | |
| [ ] | P3.6 | Unified `gwd` CLI, `gwd env`, thin menu | |
| [ ] | P3.7 | Split training into `model.py` and `train.py` | |
| [ ] | P4.1 | Make silent switches visible: logging, manifest, metadata, exit codes | |
| [ ] | P4.2 | Simulation evaluation fixes, `metrics.py` as pure functions | |
| [ ] | P4.3 | Real data field test: padding, honest counting, only with `--real` | |
| [ ] | P4.4 | Untangle Glitch Hunter: GUI, sources, audio; fix threading; modes | |
| [ ] | P4.5 | Testable demo apps | |
| [ ] | P4.6 | Validator and loading: full integrity and range checks | |
| [ ] | P5.1 | Physics tests with real asserts | |
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
| [ ] | P7.2 | License entry in `pyproject.toml` (D12) | |
| [ ] | P7.3 | CI on the lockfile, README on updating it (D7) | |
| [ ] | P7.4 | Final check: end to end, empty allowlist, checklist | |
