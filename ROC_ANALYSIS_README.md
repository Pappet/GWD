# ROC-Analyse und formale Metriken

## Übersicht

Dieses Projekt nutzt jetzt professionelle Signalverarbeitungs-Metriken:

- **ROC-Kurven** (Receiver Operating Characteristic)
- **AUC** (Area Under Curve) - Threshold-unabhängiges Qualitätsmaß
- **Precision-Recall Kurven** - Für unbalancierte Datensätze
- **Score Distributions** - Visualisierung der Modell-Konfidenz

## 🎯 Warum ROC-Kurven?

### Problem mit festen Thresholds:
```python
# Alter Ansatz: Ein fester Threshold (z.B. 0.75)
prediction = model.predict(data) > 0.75
```

**Nachteile:**
- Ergebnis hängt stark von Threshold-Wahl ab
- Unfairer Vergleich zwischen Modellen
- Keine Aussage über optimalen Arbeitspunkt

### Lösung: ROC-Kurve
Die ROC-Kurve zeigt **alle möglichen Thresholds gleichzeitig**:

```
TPR (True Positive Rate)  = Wie viele Signale finden wir?
    ^
    |     /----  Perfektes Modell (AUC=1.0)
1.0 |    /
    |   /
    |  /  <- Unser Modell (AUC=0.85)
0.5 |/
    |  /  <- Zufall (AUC=0.5)
    |/_________________>
    0                 1.0  FPR (False Positive Rate)
                           Wie oft liegen wir falsch?
```

**ROC AUC Interpretation:**
- `AUC = 1.0`: Perfekt! Findet alle Signale ohne Fehler
- `AUC = 0.9`: Exzellent (LIGO-Niveau)
- `AUC = 0.8`: Gut
- `AUC = 0.5`: Nicht besser als Zufall

## 📊 Neue Scripts

### 1. `evaluate_model_with_roc.py` - Erweiterte Einzelmodell-Analyse

**Was es macht:**
- Erstellt ROC-Kurve für ein Modell
- Berechnet AUC und Average Precision
- Zeigt Operating Point (gewählter Threshold)
- 6 verschiedene Visualisierungen

**Usage:**
```bash
python evaluate_model_with_roc.py
```

**Output:**
```
🤖 COMPREHENSIVE EVALUATION: gwd_model_20250123-143022.keras
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 FINAL SUMMARY:
  ROC AUC:              0.9234
  Average Precision:    0.9156
  Accuracy:             87.3%
  TPR (Sensitivity):    91.2%
  FPR:                  16.5%
  SNR90:                1.45
  Real Events Found:    4/5
```

**6 Plots:**
1. **ROC-Kurve** - Threshold-unabhängige Performance
2. **Precision-Recall** - Alternative Darstellung
3. **Confusion Matrix** - Bei gewähltem Threshold
4. **Sensitivity Curve** - SNR vs Detection Rate
5. **Score Distribution** - Wie trennt das Modell Signal/Noise?
6. **Metriken-Tabelle** - Alle Zahlen auf einen Blick

### 2. `compare_models_roc.py` - Modell-Vergleich

**Was es macht:**
- Vergleicht ALLE Modelle in einer ROC-Kurve
- Ranking nach AUC
- Zoom auf relevante Region (FPR < 0.1)

**Usage:**
```bash
python compare_models_roc.py
```

**Output:**
```
🏆 ROC CURVE COMPARISON TOOL

✓ Geladen: gwd_model_20250120-120000
✓ Geladen: gwd_model_20250121-140000
✓ Geladen: gwd_model_20250123-160000

📊 FINAL RANKING:
  1. gwd_model_20250123-160000  | AUC: 0.9456
  2. gwd_model_20250121-140000  | AUC: 0.9123
  3. gwd_model_20250120-120000  | AUC: 0.8845
```

## 🔬 Wissenschaftliche Metriken erklärt

### ROC AUC (Area Under Curve)
```python
# Interpretation:
AUC = P(score(signal) > score(noise))
```
**Bedeutung:** Wahrscheinlichkeit, dass ein zufälliges Signal einen höheren Score bekommt als zufälliges Rauschen.

### Average Precision (AP)
```python
# Wichtig bei unbalancierten Daten
AP = Durchschnitt aller Precision-Werte über alle Recall-Level
```
**Vorteil:** Bestraft False Positives härter als ROC AUC

### TPR (True Positive Rate) = Recall = Sensitivity
```python
TPR = TP / (TP + FN)
```
**Bedeutung:** Von allen echten Signalen - wie viele finden wir?

### FPR (False Positive Rate)
```python
FPR = FP / (FP + TN)
```
**Bedeutung:** Von allen Rausch-Segmenten - wie oft schlagen wir Fehlalarm?

## 🎮 Praktische Anwendung

### Beispiel 1: Threshold-Optimierung

Du siehst in der ROC-Kurve:
- Bei `Threshold = 0.5`: TPR=95%, FPR=30% → Viele Fehlalarme
- Bei `Threshold = 0.75`: TPR=85%, FPR=10% → Ausgewogen
- Bei `Threshold = 0.9`: TPR=60%, FPR=2% → Konservativ

**Wahl hängt ab von:**
- Wissenschaftliche Analyse → Niedriger FPR wichtiger (Threshold hoch)
- Trigger für Folge-Analysen → Hoher TPR wichtiger (Threshold niedrig)

### Beispiel 2: Modell-Vergleich

```
Modell A: AUC = 0.92, SNR90 = 1.8
Modell B: AUC = 0.88, SNR90 = 1.5

→ Modell A ist threshold-unabhängig besser (höherer AUC)
→ ABER: Modell B ist sensitiver bei niedrigem SNR!
```

**Lesson:** AUC allein reicht nicht, schaue auch auf Sensitivity!

## 📈 Verbesserungen im Leaderboard

Die CSV enthält jetzt zusätzlich:

| Column | Bedeutung |
|--------|-----------|
| `ROC_AUC` | Threshold-unabhängiges Qualitätsmaß |
| `Avg_Precision` | Präzision über alle Recall-Level |
| `TPR` | True Positive Rate bei gewähltem Threshold |
| `FPR` | False Positive Rate bei gewähltem Threshold |

**Alter Leaderboard:**
```csv
Model,Sim_Accuracy,Sim_SNR90
model_1,85%,1.8
model_2,87%,1.5
```

**Neuer Leaderboard:**
```csv
Model,ROC_AUC,Avg_Precision,TPR,FPR,Sim_SNR90,Physik_Gap
model_1,0.9234,0.9156,91%,16%,1.8,+0.30
model_2,0.8945,0.8823,88%,12%,1.5,+0.00
```

## 🔧 Integration in Workflow

### Kompletter Evaluations-Workflow:

```bash
# 1. Modell trainieren
python dataset/train_cnn.py

# 2. Physik-Baseline berechnen (einmalig)
python evaluate_classic.py

# 3. Modell evaluieren (mit ROC)
python evaluate_model_with_roc.py

# 4. Alle Modelle vergleichen
python compare_models_roc.py

# 5. Leaderboard anschauen
python show_leaderboard.py
```

## 📚 Weiterführende Literatur

**ROC-Kurven in GW-Physik:**
- LIGO Scientific Collaboration Papers verwenden immer ROC/AUC
- Standard bei Machine Learning für GW-Detection
- Vergleichbar mit "Efficiency Curves" in Particle Physics

**Typische Werte in der Literatur:**
- LIGO Matched Filter: AUC ≈ 0.95-0.98 (Physik-Limit)
- Deep Learning Modelle: AUC ≈ 0.90-0.95
- Naive Methoden: AUC ≈ 0.70-0.80

## ⚠️ Häufige Fehler

### ❌ Fehler 1: AUC auf Trainings-Daten
```python
# FALSCH:
y_pred = model.predict(X_train)
auc = roc_auc_score(y_train, y_pred)  # Zu optimistisch!
```

**Fix:** Immer separate Test-Daten verwenden!

### ❌ Fehler 2: Unbalancierte Daten ignorieren
```python
# Bei 90% Rauschen, 10% Signale:
# Accuracy = 90% klingt gut, aber Modell findet kein Signal!
```

**Fix:** Schaue auch auf Precision-Recall Kurve!

### ❌ Fehler 3: Threshold nach ROC festlegen
```python
# ROC zeigt nur was MÖGLICH ist
# Den Threshold musst du basierend auf deiner Anwendung wählen!
```

## 🎯 Zusammenfassung

**Was du jetzt hast:**
✅ Professionelle ROC-Analyse wie in Papers
✅ Threshold-unabhängiger Modell-Vergleich
✅ Mehrere komplementäre Metriken
✅ Visualisierung der Trennschärfe
✅ Wissenschaftlich fundierte Evaluation

**Nächste Schritte:**
1. Trainiere mehrere Modelle mit verschiedenen Architekturen
2. Vergleiche sie mit `compare_models_roc.py`
3. Wähle das beste Modell basierend auf AUC UND deiner Anwendung
4. Optimiere den Threshold basierend auf ROC-Kurve

**Pro-Tipp für Paper/Präsentation:**
Zeige immer die ROC-Kurve! Sie ist der Standard in der Community und zeigt, dass du weißt, was du tust. 🚀
