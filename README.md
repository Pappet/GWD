# Gravitational Wave Detection (GWD)

## 📌 High Level Project Overview
The **Gravitational Wave Detection (GWD)** project is an advanced, machine-learning-based signal processing pipeline designed to detect gravitational waves in simulated noisy streams. With recent additions, the project offers professional-grade signal processing metrics (ROC analysis) and baseline physics comparisons (Matched Filter, currently not available) aiming for scientifically comparable evaluations found in modern astrophysics papers.

> **Status:** Simulation metrics are currently of limited significance (train/eval skew, peak SNR, no seeds). The cleanup plan is in [docs/aufraeumen/PLAN.md](docs/aufraeumen/PLAN.md) (in German).

## ✨ Features
- **Machine Learning Detection:** Train Convolutional Neural Networks (CNNs) to classify gravitational wave chirps within noisy data streams.
- **Physics Baseline Comparison (currently not available):** There is currently no valid Matched Filtering baseline, so `Physik_Gap` in the leaderboard is always `-`.
- **Advanced Evaluation (ROC):** Comprehensive Area Under the ROC Curve (ROC AUC) analysis, threshold-independent scoring, Operating Point analyses, and Precision-Recall evaluation.
- **Threshold Optimization:** Automated tools to find the optimal detection threshold based on cost, fixed FPR/TPR, or F1-Score.
- **Multi-Model Comparison:** Visually and quantitatively compare multiple trained models.
- **Scientific Visualizations:** Export publication-ready figures representing SNR distribution, sensitivity curves, and more.

## 🚀 Quickstart Guide

### 1. Installation
GWD needs Python 3.11 or newer. Use Linux or WSL: PyCBC, the waveform backend, is not available on Windows.

```bash
# Provide a fresh virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package with all extras and the development tools
pip install -e '.[all,dev]'

# Check the environment and the active waveform backend
python -m gwd.env
```

The extras are `physics` (pycbc), `gwdata` (gwpy), `ml` (tensorflow), `dev` (pytest, ruff), and `all` (physics, gwdata, ml). For an NVIDIA GPU, TensorFlow also offers `pip install 'tensorflow[and-cuda]'`; this project does not install it for you.

#### Reproducible environment (lockfile)

`requirements-linux-py312.lock` pins every package, including TensorFlow and PyCBC, to the versions the models were trained and the golden test values were recorded with. It is valid for Linux and Python 3.12 only. Install from it in a fresh virtual environment:

```bash
pip install -r requirements-linux-py312.lock
pip install -e . --no-deps
```

CI installs from the same file: the fast job with `-c requirements-linux-py312.lock` (everything except TensorFlow), the slow job with `-r` as above. CI also fails if `pyproject.toml` and the lock disagree.

#### Updating the lockfile

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

Avoid `--upgrade` for everything: newer numpy, pandas, TensorFlow, or PyCBC versions can change the golden values and the behavior of the trained models. After any update, install the new lock in a fresh virtual environment and run all tests with `GWD_REQUIRE_PYCBC=1 python -m pytest`. If golden values change, update them only in a separate `golden:` commit that states the reason (see `AGENTS.md`).

The first lock was compiled with an extra `-c constraints-pc.txt` (the `pip freeze` of the PC training environment without the `nvidia-*` packages, not in the repository), so its versions match that environment. The commands above keep those versions without the file. The first update rewrites the header and drops the `-c constraints-pc.txt` notes from the file; that is expected.

### 2. Generate Data & Train
```bash
# 1. Generate Training Data
python -m gwd.data.generate

# 2. Train the CNN model
python -m gwd.training.train
```

### 3. Evaluate Results
```bash
# Run advanced ROC metric evaluation
python src/evaluate_model_with_roc.py

# Optimize inference threshold
python src/optimize_threshold.py
```

## 🤝 Contribution
Contributions are welcome! Please ensure that any structural or behavioral changes are strictly tied to synchronized documentation updates. All new code should be placed in the `src/` folder, and dependencies are declared in `pyproject.toml`.
Remember: **Never push directly to the `main` branch.** All changes must go through a Pull Request.

## 📄 License
Copyright (C) 2025–2026 Peter Paprotny

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (SPDX: `GPL-3.0-or-later`).

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [LICENSE](LICENSE) file for the full text.
