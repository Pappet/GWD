# 📋 Installation Checkliste - ROC-Features

## 🎯 Was muss ins Projekt?

### ✅ Neue Dateien (aus `/mnt/user-data/outputs/`):

1. **`evaluate_model_with_roc.py`**
   - Ersetzt: `evaluate_model.py` (alte Version)
   - Location: Root-Verzeichnis des Projekts
   - Status: ⭐ HAUPTFILE - Unbedingt installieren!

2. **`compare_models_roc.py`**
   - Neu, kein Ersatz
   - Location: Root-Verzeichnis
   - Status: Optional, aber sehr nützlich für Multi-Modell Vergleiche

3. **`optimize_threshold.py`**
   - Neu, kein Ersatz
   - Location: Root-Verzeichnis
   - Status: Optional, aber empfohlen für Threshold-Tuning

4. **`evaluate_classic_fixed.py`**
   - Ersetzt: `evaluate_classic.py` (alte Version mit Bugs)
   - Location: Root-Verzeichnis
   - Status: ⚠️ Wichtig! Die alte Version hat PSD-Probleme

5. **`ROC_ANALYSIS_README.md`**
   - Neu, kein Ersatz
   - Location: Root-Verzeichnis (oder `docs/`)
   - Status: Dokumentation - Hilfreich

6. **`ROC_INTEGRATION_OVERVIEW.md`**
   - Neu, kein Ersatz
   - Location: Root-Verzeichnis (oder `docs/`)
   - Status: Übersicht - Hilfreich

## 📂 Vorgeschlagene Projektstruktur (nach Integration):

```
GWD/
├── gwd_core/
│   ├── __init__.py
│   ├── waveforms.py
│   ├── noise.py
│   ├── simulation.py
│   └── interferometer.py
│
├── dataset/
│   ├── __init__.py
│   ├── fetch_real_noise.py
│   ├── generate_chirp_dataset.py
│   └── train_cnn.py
│
├── models_registry/          # Automatisch erstellt
│   ├── gwd_model_*.keras
│   ├── gwd_model_*.json
│   └── physics_baseline.json
│
├── docs/                     # Optional, für Organisation
│   ├── ROC_ANALYSIS_README.md
│   └── ROC_INTEGRATION_OVERVIEW.md
│
├── evaluate_model_with_roc.py    # ⭐ NEU/ERSATZ
├── evaluate_classic_fixed.py     # ⭐ NEU/ERSATZ
├── compare_models_roc.py         # ⭐ NEU
├── optimize_threshold.py         # ⭐ NEU
│
├── glitch_hunter_app.py
├── gw_simulator.py
├── interferometer_simulator.py
├── show_leaderboard.py
├── model_leaderboard.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔄 Schritt-für-Schritt Installation:

### Schritt 1: Alte Dateien sichern (optional)
```bash
# Falls du die alten Versionen behalten möchtest:
mv evaluate_model.py evaluate_model_OLD.py
mv evaluate_classic.py evaluate_classic_OLD.py
```

### Schritt 2: Neue Dateien kopieren
```bash
# Von outputs/ ins Projekt-Root:
cp evaluate_model_with_roc.py ../GWD/
cp evaluate_classic_fixed.py ../GWD/
cp compare_models_roc.py ../GWD/
cp optimize_threshold.py ../GWD/

# Dokumentation (optional):
cp ROC_ANALYSIS_README.md ../GWD/docs/
cp ROC_INTEGRATION_OVERVIEW.md ../GWD/docs/
```

### Schritt 3: Ausführbar machen (Linux/Mac)
```bash
chmod +x evaluate_model_with_roc.py
chmod +x evaluate_classic_fixed.py
chmod +x compare_models_roc.py
chmod +x optimize_threshold.py
```

### Schritt 4: Test
```bash
# Teste ob alles funktioniert:
python evaluate_model_with_roc.py
```

**Erwartete Fehlermeldung (falls kein Modell):**
```
❌ Keine Modelle (.keras) gefunden.
```
→ Das ist OK! Trainiere zuerst ein Modell.

## ✅ Kompatibilitäts-Check:

### Diese Dateien müssen UNVERÄNDERT bleiben:
- ✅ `gwd_core/waveforms.py`
- ✅ `gwd_core/noise.py`
- ✅ `dataset/train_cnn.py`
- ✅ `dataset/generate_chirp_dataset.py`

### Diese Dateien können OPTIONAL aktualisiert werden:
- 📝 `show_leaderboard.py` - Funktioniert mit neuen Spalten
- 📝 `glitch_hunter_app.py` - Threshold kann optimiert werden

## 🔍 Verifizierung:

### Test 1: Imports funktionieren?
```bash
python -c "from sklearn.metrics import roc_curve, auc; print('✓ sklearn OK')"
python -c "import tensorflow as tf; print('✓ TensorFlow OK')"
python -c "from pycbc.filter import matched_filter; print('✓ PyCBC OK')"
```

### Test 2: Sind alle Core-Module da?
```bash
python -c "from gwd_core.waveforms import generate_astrophysical_chirp; print('✓')"
python -c "from gwd_core.noise import generate_colored_noise; print('✓')"
```

### Test 3: Kann ein Modell geladen werden?
```bash
# Erst ein Modell trainieren:
python dataset/train_cnn.py

# Dann evaluieren:
python evaluate_model_with_roc.py
```

## 🐛 Häufige Probleme & Lösungen:

### Problem 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'sklearn'
```
**Lösung:**
```bash
pip install scikit-learn
```

### Problem 2: PyCBC Import Error
```
ImportError: cannot import name 'matched_filter' from 'pycbc.filter'
```
**Lösung:**
```bash
pip install --upgrade pycbc
```

### Problem 3: "No models found"
```
❌ Keine Modelle (.keras) gefunden.
```
**Lösung:** 
```bash
# Erst trainieren:
python dataset/train_cnn.py
```

### Problem 4: LIGO Data Download Fehler
```
⚠️ Konnte LIGO-Daten nicht laden (Internet?)
```
**Lösung:** 
- Internet-Verbindung prüfen
- LIGO Server können manchmal down sein
- Fallback: Skript arbeitet trotzdem weiter mit Simulation

## 📊 Neue Leaderboard-Spalten:

Nach der Integration wird `model_leaderboard.csv` erweitert:

**Alte Spalten:**
- Model
- Date
- Sim_Accuracy
- Sim_SNR50
- Sim_SNR90
- Physik_Gap
- Sim_FalseAlarm
- Real_Events_Found
- Real_Noise_FAR

**Neue Spalten:**
- **ROC_AUC** ⭐
- **Avg_Precision** ⭐
- **TPR** (True Positive Rate)
- **FPR** (False Positive Rate)

**Kompatibilität:** 
- ✅ Alte Einträge bleiben erhalten
- ✅ Neue Spalten werden ergänzt
- ✅ `show_leaderboard.py` funktioniert weiterhin

## 🎓 Quick-Start nach Installation:

```bash
# 1. Ein Modell trainieren
python dataset/train_cnn.py

# 2. ROC-Evaluation
python evaluate_model_with_roc.py

# 3. Threshold optimieren
python optimize_threshold.py

# 4. (Optional) Physik-Baseline einmalig berechnen
python evaluate_classic_fixed.py

# 5. Leaderboard anschauen
python show_leaderboard.py
```

## 🎉 Fertig-Check:

Kreuze ab, wenn du fertig bist:

- [ ] Alte Dateien gesichert (optional)
- [ ] Neue Dateien kopiert
- [ ] Dependencies installiert (sklearn, tensorflow, pycbc)
- [ ] Test-Import erfolgreich
- [ ] Mindestens ein Modell trainiert
- [ ] `evaluate_model_with_roc.py` läuft ohne Fehler
- [ ] ROC-Kurve wird angezeigt
- [ ] Leaderboard enthält neue Spalten

**Wenn alle Punkte ✅ sind → Installation erfolgreich!** 🎊

## 💡 Pro-Tipps:

1. **Git Commit nach Installation:**
   ```bash
   git add .
   git commit -m "Add ROC analysis and formal metrics"
   ```

2. **Backup des Leaderboards:**
   ```bash
   cp model_leaderboard.csv model_leaderboard_backup.csv
   ```

3. **Dokumentation lesen:**
   - `ROC_ANALYSIS_README.md` für Details
   - `ROC_INTEGRATION_OVERVIEW.md` für Workflow

## 📞 Hilfe benötigt?

Falls etwas nicht funktioniert:

1. Prüfe Python-Version: `python --version` (sollte ≥ 3.8 sein)
2. Prüfe Dependencies: `pip list | grep -E "sklearn|tensorflow|pycbc"`
3. Schaue in die Error-Message - meist ist es ein fehlende Dependency
4. Teste einzelne Komponenten mit den Tests oben

**Happy Analyzing!** 🚀
