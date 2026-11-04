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
**Known issue:** `requirements.txt` pins `pycbc==2.4.0`, which has no usable wheels on Python 3.12, so this install can fail. A lockfile is planned.

```bash
# Provide a fresh virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train
```bash
# 1. Generate Training Data
python src/dataset/generate_chirp_dataset.py

# 2. Train the CNN model
python src/dataset/train_cnn.py
```

### 3. Evaluate Results
```bash
# Run advanced ROC metric evaluation
python src/evaluate_model_with_roc.py

# Optimize inference threshold
python src/optimize_threshold.py
```

## 🤝 Contribution
Contributions are welcome! Please ensure that any structural or behavioral changes are strictly tied to synchronized documentation updates. All new code should be placed in the `src/` folder, and modifications to dependencies must be locked down securely in `requirements.txt`.
Remember: **Never push directly to the `main` branch.** All changes must go through a Pull Request.

## 📄 License
Copyright (C) 2025–2026 Peter Paprotny

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (SPDX: `GPL-3.0-or-later`).

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [LICENSE](LICENSE) file for the full text.
