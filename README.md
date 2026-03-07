# Gravitational Wave Detection (GWD)

## 📌 High Level Project Overview
The **Gravitational Wave Detection (GWD)** project is an advanced, machine-learning-based signal processing pipeline designed to detect gravitational waves in simulated noisy streams. With recent additions, the project offers professional-grade signal processing metrics (ROC analysis) and baseline physics comparisons (Matched Filter) aiming for scientifically comparable evaluations found in modern astrophysics papers.

## ✨ Features
- **Machine Learning Detection:** Train Convolutional Neural Networks (CNNs) to classify gravitational wave chirps within noisy data streams.
- **Physics Baseline Comparison:** Compare CNN model performance against the theoretical physics limit using classical Matched Filtering.
- **Advanced Evaluation (ROC):** Comprehensive Area Under the ROC Curve (ROC AUC) analysis, threshold-independent scoring, Operating Point analyses, and Precision-Recall evaluation.
- **Threshold Optimization:** Automated tools to find the optimal detection threshold based on cost, fixed FPR/TPR, or F1-Score.
- **Multi-Model Comparison:** Visually and quantitatively compare multiple trained models.
- **Scientific Visualizations:** Export publication-ready figures representing SNR distribution, sensitivity curves, and more.

## 🚀 Quickstart Guide

### 1. Installation
Ensure you have the exact dependencies installed. We enforce exact version locks to guarantee reproducible builds.

```bash
# Provide a fresh virtual environment
python -m venv venv
source venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train
```bash
# IMPORTANT: Adjust PYTHONPATH so modules resolve from the project root
export PYTHONPATH=.

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
MIT License. Feel free to use and expand upon this analytical framework.
