# GWD Project Overview

## 📖 What it is?
The Gravitational Wave Detection (GWD) application is a combined ML and physics-based analytical framework for recognizing GW chirp signatures in simulated astrophysical noise. By providing both standard CNN models and traditional matched filtering, it facilitates an end-to-end sandbox representing how contemporary observatories (like LIGO/Virgo) filter signals.

## 📊 Project Stats
- **Language**: Python 3
- **Core Strategy**: Machine Learning vs. Classical Analytics (Matched Filter)
- **Deployment Structure**: Component-driven analysis via CLI tools.

## 🏛️ Architectural Decisions
1. **Separation of Source via `src/`**: All executing scripts and logic reside safely within the `src/` directory to prevent pollution of the project root.
2. **Deterministic Processing**: Dependencies are hard-locked (`requirements.txt`) to assure reproducible analytical results.
3. **Threshold-Agnostic Validation**: Standardizing metrics primarily around ROC-AUC to abstract away the sensitivity of simple arbitrary threshold limits.
4. **Environment Isolation**: Local `.env` (ignored from codebases) for secrets (though none required at this exact layer right now) and ignored binary artifacts (`gw_training_data`, `models_registry`).

## 🧱 Detailed Architecture
The pipeline follows a `Train -> Evaluate -> Optimize -> Compare` workflow.
- `gwd_core/`: Core logic containing waveforms generation, colored noise application, and interferometer physics models.
- `dataset/`: Training operations (data generation, CNN training loops).
- **Execution Level**: A suite of `evaluate_*.py` files compute comparisons between CNN outputs and matched filter configurations.

## 📁 Source Files Description (`src/` Folder)
- **`gwd_core/`**: Mathematical simulation models (`interferometer.py`, `noise.py`, `waveforms.py`, `simulation.py`).
- **`dataset/`**: Data building blocks (`train_cnn.py`, `generate_chirp_dataset.py`, `fetch_real_noise.py`).
- **`evaluate_model_with_roc.py`**: Principal evaluation framework generating 6 distinct statistical plots (ROC, PRC, Matrix, Output Dist).
- **`evaluate_classic.py`**: Computes the physics-limit baseline utilizing `gwd_core` templated Matched Filters.
- **`optimize_threshold.py`**: Identifies optimal cutoffs via 6 diverse numerical strategies (Youden's J, F1, Cost, etc.).
- **`compare_models_roc.py`**: Ingests the `models_registry` dir to compare multple previously trained `.keras` iterations against one another.
- **`glitch_hunter_app.py`**: Example application integrating findings.

## 📦 Dependencies & Purpose
- `numpy` / `pandas` / `scipy` : Core linear algebraic data manipulation and array management.
- `tensorflow` : CNN Model infrastructure, building layers, and running loss optimization.
- `scikit-learn` : Primary supplier of precision-recall and ROC scoring metric functions.
- `matplotlib` : Exporting multi-layered scientific visualization plots.
- `gwpy` : Used to fetch and manipulate genuine Open Science Center astrophysical data if needed.
- `pycbc` : Used extensively in `gwd_core/waveforms.py` to synthesize accurate professional chirp waveforms.
- `tabulate` : Clean markdown and terminal rendering of tables matrices.

## 🔗 Additional References
- [ROC_ANALYSIS_README.md](./ROC_ANALYSIS_README.md): Detailed documentation surrounding the ROC-Integration mechanics.
- [ROC_INTEGRATION_OVERVIEW.md](./ROC_INTEGRATION_OVERVIEW.md): Extended Workflow mechanics.
- [INSTALLATION_CHECKLIST.md](./INSTALLATION_CHECKLIST.md): Basic checklist of items prior to run-time.
