# Befund-Register

Dies ist das Register der geprüften Befunde zum [Aufräumplan](PLAN.md). Es hat einen Abschnitt je Komponente.

- Die **Schwere** ist der Wert nach der Prüfung.
- **Urteil** sagt, wie der Prüfer den Befund bewertet hat:
  - *bestätigt*: wie beschrieben nachvollzogen;
  - *teilweise*: im Kern richtig, aber Umfang oder Schweregrad korrigiert;
  - *ungeprüft*: nur geringe Schwere, nicht eigens geprüft.
- IDs mit Präfix (z. B. `core:MISS-1`) hat der Prüfer zusätzlich gefunden.
- **Schritt** nennt die Plan-Schritte, die den Befund beheben oder sichtbar machen.
- Zeilennummern beziehen sich auf `c54acd5`.

## Physik-Kern (`src/gwd_core`) und Tests

hoch: 3 · mittel: 12 · niedrig: 8

| ID | Schwere | Urteil | Befund | Ort | Schritt |
|---|---|---|---|---|---|
| CORE-4 | hoch | teilweise | Stiller Wechsel des Physikmodells PyCBC -> Newton (abhängig von Umgebung und Parametern) | `src/gwd_core/waveforms.py`:10-15, 52-53, 119-120 | P0.6, P4.1, P5.2, P6.1 |
| CORE-5 | hoch | teilweise | gwd_core nicht installierbar: Tests und Skripte importieren unterschiedliche Module, apps/analysis/utility sind kaputt | `tests/conftest.py`:4-5 | P1.3, P1.5, P2.1, P2.6, P7.4 |
| core:MISS-1 | hoch | vom Prüfer ergänzt | pycbc==2.4.0 ist auf Python 3.12 nicht installierbar; unter Windows gibt es PyCBC gar nicht, deshalb wird Newton dort zum Standard-Backend | `requirements.txt`:16 | P0.1, P2.1, P5.2, P6.1, P7.3 |
| CORE-1 | mittel | bestätigt | generate_realistic_chirp schneidet den Merger ab und ist toter Code | `src/gwd_core/waveforms.py`:172-248 (v. a. 236-246) | P5.1, P5.3 |
| CORE-2 | mittel | bestätigt | Newton-Fallback: ausgegebene Frequenz um Faktor π^(5/8) ≈ 2,045 zu hoch | `src/gwd_core/waveforms.py`:140-142 | P5.1, P6.1 |
| CORE-3 | mittel | bestätigt | Newton-Fallback: Strain-Amplitude um Faktor 4 zu groß | `src/gwd_core/waveforms.py`:146-148 | P5.1, P6.1 |
| CORE-6 | mittel | bestätigt | Signal-Injektion und SNR-Definition sechsmal außerhalb des Kerns, mit widersprüchlichen Bereichen | `src/dataset/generate_chirp_dataset.py`:23-44, 106-131 | P0.6, P3.5, P5.5 |
| CORE-7 | mittel | bestätigt | Konditionierung der LIGO-Daten dreifach und inkonsistent; simulierte Chirps werden nie gefiltert | `src/utils/data_processing.py`:3-12 | P3.2, P4.3, P6.2 |
| CORE-8 | mittel | bestätigt | Newton-Fallback physikalisch unvollständig: kein f_lower, kein ISCO-Cut, Parameter werden still ignoriert, Taper setzt t0=0 voraus | `src/gwd_core/waveforms.py`:122-169 | P5.1, P6.1 |
| CORE-10 | mittel | bestätigt | InterferometerModel ist ein Spielzeugmodell mit unbegründeten Faktoren und falschen Skalierungen | `src/gwd_core/interferometer.py`:28-76 | P0.6, P5.4 |
| CORE-11 | mittel | bestätigt | Physik in der App: Finesse-Formel und erfundene 'LIGO Design'-Kurve; finesse und R sind im Modell inkonsistent | `src/interferometer_simulator.py`:105-106, 166-171 | P4.5, P5.4 |
| CORE-12 | mittel | bestätigt | TimeDomainSimulator ist eine eigene Welt: ungetestet, eigene Konstanten, unbrauchbare Intensität, unrealistische Rauschskala | `src/gwd_core/simulation.py`:9-52 | P5.1, P5.4 |
| CORE-13 | mittel | bestätigt | Tests prüfen kaum Physik, große Teile werden nie ausgeführt, der getestete Pfad hängt von der Umgebung ab | `tests/test_waveforms.py`:1-19 | P0.6, P5.1 |
| core:MISS-2 | mittel | vom Prüfer ergänzt | Echte Daten werden für Eval und App nur 4 s lang geladen und dann gefiltert, das Trainingsrauschen dagegen am Stück (4096 s) und erst danach geschnitten; dadurch entstehen Filter-Randtransienten nur zur Testzeit | `src/utils/data_processing.py`:3-12 (Aufrufer src/evaluate_model_with_roc.py:281-285, src/glitch_hunter_app.py:161-171) | P4.3 |
| core:MISS-3 | mittel | vom Prüfer ergänzt | Nirgends ein RNG-Seed: Datensätze, ROC-Testsets und Threshold-Optimierung sind nicht reproduzierbar, Leaderboard-AUCs stammen aus verschiedenen Zufallsstichproben | `src/evaluate_model_with_roc.py`:85-104 (analog compare_models_roc.py:59-72, optimize_threshold.py:51-65, generate_chirp_dataset.py:81-131) | P0.6, P3.5, P3.6, P5.3, P5.7 |
| CORE-9 | niedrig | teilweise | generate_colored_noise: noise_level ist nicht die Standardabweichung, das Spektrum ist willkürlich und hat keinen Bezug zum Detektormodell | `src/gwd_core/noise.py`:29-64 | P5.1, P5.4 |
| CORE-14 | niedrig | teilweise | PyCBC-Pfad: freq_out ist außerhalb des Signals und bei Inklination π/2 unbrauchbar; Taper-Logik passt nicht zum Kommentar | `src/gwd_core/waveforms.py`:69-77, 104-109 | P5.3 |
| CORE-15 | niedrig | bestätigt | t_merger außerhalb des Fensters wird still auf den Rand geklemmt; Default setzt t[0]=0 voraus | `src/gwd_core/waveforms.py`:48-49, 84 | P5.1, P5.3 |
| CORE-16 | niedrig | bestätigt | generate_random_bbh_parameters: 'O3-Verteilungen' ist übertrieben, globaler RNG, Spins und Inklination werden vom Aufrufer verworfen | `src/gwd_core/waveforms.py`:251-292 | P3.5, P5.3 |
| CORE-17 | niedrig | bestätigt | Seiteneffekte beim Import und print in Bibliothekscode | `src/gwd_core/waveforms.py`:10-15, 120, 197 | P2.6, P4.1, P5.2 |
| CORE-18 | niedrig | bestätigt | Verstreute und doppelte Konstanten, toter Code, Kosmetik | `src/gwd_core/interferometer.py`:22-23, 29, 38, 43, 54-55 | P2.2, P2.3, P3.1, P5.3, P5.4 |
| CORE-19 | niedrig | bestätigt | Zeitachse über np.linspace(0, dur, n): dt ≠ 1/fs | `src/dataset/generate_chirp_dataset.py`:76 | P5.3 |
| core:MISS-4 | niedrig | vom Prüfer ergänzt | Feldtest enthält GW170817 (Neutronenstern-Binär), das weit außerhalb der Trainingsmassen liegt | `src/evaluate_model_with_roc.py`:46-52 | P3.4, P4.3 |

## Daten und Training (`src/dataset`)

mittel: 14 · niedrig: 16

| ID | Schwere | Urteil | Befund | Ort | Schritt |
|---|---|---|---|---|---|
| DATA-1 | mittel | teilweise | Trainings- und Evaluations-SNR-Bereich passen nicht zusammen, obwohl der Docstring Gleichheit behauptet | `src/dataset/generate_chirp_dataset.py`:26, 110, 123 | P3.5, P6.2 |
| DATA-3 | mittel | bestätigt | validate_dataset stürzt mit installiertem matplotlib 3.11 ab, die Deprecation war per Warnfilter versteckt | `src/dataset/validate_dataset.py`:23, 183 | P0.3 |
| DATA-5 | mittel | bestätigt | Keine Reproduzierbarkeit: weder Generator noch Training setzen Seeds | `src/dataset/generate_chirp_dataset.py`:12, 91, 110, 123; train_cnn.py:249 | P3.6, P5.3, P5.6 |
| DATA-6 | mittel | bestätigt | fetch_real_noise verschluckt Fehler und endet mit Exit 0, main.py meldet Erfolg | `src/dataset/fetch_real_noise.py`:61-62 | P4.1, P6.3 |
| DATA-7 | mittel | bestätigt | config.py legt beim Import Ordner an | `config.py`:29-37 | P0.2, P2.3 |
| DATA-8 | mittel | bestätigt | LEADERBOARD_FILE zeigt in den Duplikat-Ordner analysis/, Aufräumen würde das Leaderboard brechen | `config.py`:19 | P1.2 |
| DATA-9 | mittel | bestätigt | Config-Konstanten SAMPLE_RATE/DURATION/INPUT_SHAPE ungenutzt, 4096/4.0/16384 überall hart codiert | `config.py`:24-26 | P3.1 |
| DATA-10 | mittel | bestätigt | Vorverarbeitung und Normierung mehrfach und uneinheitlich implementiert, 'whitening' ist nur ein Bandpass | `src/dataset/fetch_real_noise.py`:31-40 | P3.2 |
| DATA-11 | mittel | bestätigt | Echtes Rauschen wird ~8× wiederverwendet, ohne Random-Crop und quer über Train/Test | `src/dataset/generate_chirp_dataset.py`:46-67, 94 | P6.3 |
| DATA-13 | mittel | bestätigt | Training liegt in dataset/, hat Seiteneffekte auf Modulebene und hart codierte Hyperparameter | `src/dataset/train_cnn.py`:1-47 | P2.4, P3.7 |
| DATA-17 | mittel | bestätigt | sys.path-Hacks in allen Dataset-Skripten führen zu doppelten Modulobjekten | `src/dataset/generate_chirp_dataset.py`:5-6 | P2.1, P2.3, P2.4, P2.5, P2.6 |
| DATA-23 | mittel | bestätigt | Keine Tests für Datengenerierung, Validierung, Training, config oder main | `tests/` | P0.2, P0.5, P7.4 |
| data-training:NOISE-SKEW | mittel | vom Prüfer ergänzt | Training und Evaluation verwenden unterschiedliche Rauschmodelle, und die Physik-Baseline wurde in einem dritten Setup gemessen | `src/evaluate_model_with_roc.py`:92, 139 (ebenso optimize_threshold.py:53, compare_models_roc.py:61) | P3.5, P6.2 |
| data-training:SNR-DEF | mittel | vom Prüfer ergänzt | Die Spalte 'snr' ist ein Peak-Amplitudenverhältnis und kein (Matched-Filter-)SNR. Ihre Detektierbarkeit hängt stark von den Massen ab, und Validator-Schwellen und Kategorien sind deshalb bedeutungslos. | `src/dataset/generate_chirp_dataset.py`:23-44, 123, 145 | P1.5, P4.6, P5.5, P6.2 |
| DATA-2 | niedrig | teilweise | Validierungsset ist gleich dem Test-Set, die gemeldete Test-Accuracy ist optimistisch verzerrt | `src/dataset/train_cnn.py`:248-249, 265-276, 294, 307, 186 | P5.6, P6.5 |
| DATA-4 | niedrig | teilweise | Validator-Schwellen stammen aus alter SNR-Definition (5–25) und schlagen für den heutigen Generator falsch Alarm | `src/dataset/validate_dataset.py`:64-79, 265-271, 439-447 | P4.6 |
| DATA-12 | niedrig | bestätigt | Signale werden nicht gleich vorverarbeitet wie echtes Rauschen (Bandpass nur auf Rauschen), mögliche Abkürzung für das CNN | `src/dataset/generate_chirp_dataset.py`:115-128 | P6.2 |
| DATA-14 | niedrig | bestätigt | main.py: cwd-abhängige relative Pfade und os.system mit unquotierten Strings | `main.py`:9-19, 43-51 | P2.4, P3.6 |
| DATA-15 | niedrig | bestätigt | main.py ist nur interaktiv nutzbar: keine CLI, input() blockiert, argparse ungenutzt | `main.py`:4, 19, 24, 41 | P3.6 |
| DATA-16 | niedrig | bestätigt | Menü deckt den Funktionsumfang nur teilweise ab | `main.py`:26-38, 43-52 | P2.5, P3.6, P6.4 |
| DATA-18 | niedrig | bestätigt | Generator: ungenutzter Import, bare except, gewürfelte Parameter werden verworfen, Distanz ohne Wirkung | `src/dataset/generate_chirp_dataset.py`:14-18, 66, 106-117 | P2.2, P4.6, P5.3 |
| DATA-19 | niedrig | bestätigt | Zeitvektor mit linspace(endpoint=True) ergibt effektiv 4095.75 Hz statt 4096 Hz | `src/dataset/generate_chirp_dataset.py`:76 | P5.3 |
| DATA-20 | niedrig | bestätigt | Modell-Metadaten-JSON unvollständig und ungenutzt, Modellauswahl per Datei-ctime | `src/dataset/train_cnn.py`:159-193 | P3.3, P4.1, P5.6 |
| DATA-21 | niedrig | ungeprüft | Speicher und Format: float64-Einzeldateien, komplettes Laden via iterrows | `src/dataset/train_cnn.py`:66-89 | P4.6, P5.6 |
| DATA-22 | niedrig | ungeprüft | Validator prüft nur 10 Zufallsdateien und dupliziert Dateinamen/Längenlogik | `src/dataset/validate_dataset.py`:28, 110-151 | P3.1, P4.6 |
| DATA-24 | niedrig | bestätigt | Veraltete Pfade und Angaben zur Datenpipeline in Doku und Meldungen | `docs/` | P1.4, P7.1 |
| DATA-25 | niedrig | ungeprüft | Rauschdownload: fixes Einzelsegment, ungeprüfte Annahme 'keine Events', kein NaN-Check | `src/dataset/fetch_real_noise.py`:15-18, 29 | P3.4, P6.3 |
| DATA-26 | niedrig | bestätigt | Architektur: 97.6 % der Parameter in einer Dense-Schicht, 21.6 MB pro Modell | `src/dataset/train_cnn.py`:115-146 | (bewusst weggelassen) |
| data-training:GITIGNORE-ART | niedrig | vom Prüfer ergänzt | Artefakte werden nur über globale Dateiendungen ignoriert, nicht über Ordner. Auch Konfigurations- und Ergebnisdateien sind dadurch unversionierbar. | `.gitignore`:21-32, 47 | P0.1, P1.1 |
| data-training:LOAD-SILENT | niedrig | vom Prüfer ergänzt | load_data überspringt fehlerhafte oder fehlende Dateien stillschweigend und kürzt bzw. paddet falsche Längen ohne Fehler | `src/dataset/train_cnn.py`:66-85 | P4.6 |

## Evaluation (`src/evaluate_*`, `optimize_threshold`, `compare_models_roc`, Leaderboard)

hoch: 5 · mittel: 16 · niedrig: 10

| ID | Schwere | Urteil | Befund | Ort | Schritt |
|---|---|---|---|---|---|
| EVAL-1 | hoch | teilweise | SNR-Bereich der Evaluation liegt größtenteils unter bzw. am Rand des Trainingsbereichs | `src/evaluate_model_with_roc.py`:95, 130 | P3.5, P5.7, P6.2, P6.5 |
| EVAL-2 | hoch | bestätigt | Rauschmodell im Training (bis 80 % echtes, bandpassgefiltertes H1-Rauschen) unterscheidet sich von der Simulations-Eval (100 % weißes Rauschen) | `src/dataset/generate_chirp_dataset.py`:79, 94-97, 128 | P3.5, P5.7, P6.2, P6.5 |
| EVAL-4 | hoch | teilweise | Echte Events werden als 4-s-Fenster einzeln gefiltert, Rauschen und Trainingsrauschen am Stück: Randartefakte und nicht vergleichbare Zahlen | `src/evaluate_model_with_roc.py`:281-290, 317-327 | P4.3 |
| EVAL-6 | hoch | bestätigt | Matched-Filter-Baseline: Template wird vor dem Merger abgeschnitten und enthält keine Signalenergie | `evaluate_classic.py (gelöscht, 45b9202^)`:46-56 | P0.4, P6.4 |
| evaluation:MISS-2 | hoch | vom Prüfer ergänzt | Peak-SNR ist nicht rauschartinvariant: Gleiche Label-SNR bedeutet in weißem, gefiltertem H1- und farbigem Rauschen sehr unterschiedliche Detektierbarkeit | `src/dataset/generate_chirp_dataset.py`:23-44, 94-97, 128 | P5.5, P6.2 |
| EVAL-3 | mittel | bestätigt | Eval nutzt nur 30/30 M☉. Peak-SNR ist keine physikalische SNR und je nach Masse unterschiedlich skaliert | `src/evaluate_model_with_roc.py`:97, 141 | P3.5, P5.5, P5.7 |
| EVAL-5 | mittel | bestätigt | Vorverarbeitung echter Daten doppelt implementiert und leicht unterschiedlich; kein Whitening trotz Variablenname | `src/utils/data_processing.py`:3-12 | P3.2 |
| EVAL-7 | mittel | bestätigt | Physik_Gap vergleicht Größen, die nicht vergleichbar sind (anderes Rauschen, andere SNR-Skala, andere Verteilung, nicht FAR-abgeglichene Schwellen) | `src/utils/evaluation.py`:105-115, 130 | P0.4, P5.8, P6.4 |
| EVAL-8 | mittel | bestätigt | physics_baseline.json hat keinen Schreiber mehr; Feature ist still deaktiviert, Doku verweist auf nie existierende Datei | `src/evaluate_model_with_roc.py`:36, 393 | P0.4, P5.8, P6.4, P7.1 |
| EVAL-9 | mittel | teilweise | analysis/*.py sind byte-identische, nicht lauffähige Kopien von src/*.py | `analysis/evaluate_model_with_roc.py`:28 | P1.3 |
| EVAL-10 | mittel | bestätigt | Testdaten-Generator 4× kopiert, Konstanten FS/DURATION mehrfach neu definiert | `src/optimize_threshold.py`:46-67 | P3.1, P3.5, P5.7 |
| EVAL-13 | mittel | bestätigt | Bei Netzwerkfehlern trägt das Leaderboard '0/5 Events gefunden' ein, nicht unterscheidbar von Modellversagen | `src/evaluate_model_with_roc.py`:297-301, 347-351 | P4.3 |
| EVAL-15 | mittel | bestätigt | Arbeits-Schwellwert hart codiert und inkonsistent; Ergebnis von optimize_threshold wird nie genutzt | `src/evaluate_model_with_roc.py`:42, 68, 121, 148, 164 | P4.2, P5.8 |
| EVAL-18 | mittel | bestätigt | Zoom-Plot überschreibt den Vergleichsplot; Balkenfarben passen nicht zu den Kurven; doppelte Predictions | `src/compare_models_roc.py`:117-121, 136, 154, 173 | P4.2 |
| EVAL-19 | mittel | bestätigt | Ergebnisdateien verstreut und wegen globaler .gitignore-Regeln nie versioniert | `config.py`:19, 21 | P1.1, P1.2 |
| EVAL-20 | mittel | bestätigt | Keine Tests für die Evaluations-Komponente, keine Seeds, keine Reproduzierbarkeit | `tests/` | P0.5, P4.2, P5.7, P7.4 |
| EVAL-24 | mittel | teilweise | sys.path-Hacks, generisches Paket 'utils' und harte Netz/gwpy-Abhängigkeit | `src/evaluate_model_with_roc.py`:4, 24, 27 | P2.5, P2.6, P3.3, P3.4 |
| EVAL-25 | mittel | bestätigt | (Komponentenübergreifend) 'Test Accuracy' des Trainings ist nicht unabhängig: Testset dient auch als Validierungsset | `src/dataset/train_cnn.py`:248-250, 271-276, 294, 307 | P5.6 |
| evaluation:MISS-1 | mittel | vom Prüfer ergänzt | SNR50/SNR90 werden bei festem Schwellwert ohne FAR-Bezug gemessen, und das Leaderboard rankt danach. Ein Modell, das öfter feuert, wird belohnt. | `src/evaluate_model_with_roc.py`:130-159, 148; show_leaderboard.py:11-15 | P0.4, P5.8, P6.5 |
| evaluation:MISS-3 | mittel | vom Prüfer ergänzt | Löschen von analysis/ bricht die Evaluation (Leaderboard-Ziel liegt dort, Ordner wird nicht angelegt) | `config.py`:19, 30-37 | P0.1, P0.5, P1.2 |
| evaluation:MISS-4 | mittel | vom Prüfer ergänzt | (Komponentenübergreifend) Echte Rausch-Chunks werden ca. 7,8× wiederverwendet; der zufällige Train/Test-Split legt identische Rauschrealisierungen in beide Sets | `src/dataset/generate_chirp_dataset.py`:46-61, 94-95; fetch_real_noise.py:18, 47-57; train_cnn.py:248-250 | P4.3, P5.6, P6.3 |
| EVAL-11 | niedrig | bestätigt | get_latest_model doppelt mit unterschiedlicher Signatur; 'neuestes Modell' über ctime | `src/utils/evaluation.py`:8-20 | P3.3 |
| EVAL-12 | niedrig | teilweise | update_leaderboard_basic ist toter Code und dupliziert update_leaderboard_roc; Leaderboard-Schema driftet | `src/utils/evaluation.py`:22-63 | P0.4 |
| EVAL-14 | niedrig | bestätigt | Escape-Fehler: '\n' wird wörtlich ausgegeben | `src/utils/evaluation.py`:62, 109 | P0.4 |
| EVAL-16 | niedrig | bestätigt | Fixed-FPR/TPR-Strategien wählen den nächsten Punkt statt 'max TPR bei FPR ≤ Ziel'; F1-Schleife quadratisch | `src/optimize_threshold.py`:94-97, 114-131, 151-159 | P4.2 |
| EVAL-17 | niedrig | bestätigt | Interaktives input() am Ende: nicht-interaktiv EOFError nach kompletter Rechnung | `src/optimize_threshold.py`:319-320 | P3.6 |
| EVAL-21 | niedrig | bestätigt | Hochrechnung der False-Alarm-Rate aus 200 s Daten statistisch nicht belastbar | `src/evaluate_model_with_roc.py`:53-55, 335-340 | P4.3 |
| EVAL-22 | niedrig | bestätigt | Randfälle bei SNR50/90 und Längen-Fallback | `src/evaluate_model_with_roc.py`:150-161, 206-207, 243-244, 288-289 | P0.4, P3.4, P4.2, P4.3, P5.8 |
| EVAL-23 | niedrig | bestätigt | show_leaderboard: Modulcode ohne main-Guard und Dateiprüfung, falsches Sortierkriterium, nicht im Menü | `src/show_leaderboard.py`:1-17 | P0.4, P2.5 |
| EVAL-26 | niedrig | bestätigt | Veraltete Verweise und Spuren des Lösch-Commits | `utility/fix_plots.py`:5-6 | P1.4 |
| evaluation:MISS-5 | niedrig | vom Prüfer ergänzt | Leaderboard speichert Metriken als formatierte Strings ('51.8%', '0/5', '< 157788.0/y'). Die CSV ist nicht maschinell sortier- oder aggregierbar. | `src/utils/evaluation.py`:85-97 | P5.8 |

## Interaktive Apps (Glitch Hunter, Simulatoren)

hoch: 2 · mittel: 9 · niedrig: 13

| ID | Schwere | Urteil | Befund | Ort | Schritt |
|---|---|---|---|---|---|
| APP-1 | hoch | bestätigt | apps/*.py sind byte-identische, aber nicht lauffähige Kopien von src/* | `apps/` | P0.5, P1.3 |
| apps:APPX-1 | hoch | vom Prüfer ergänzt | Training injiziert UNGEFILTERTE Chirps in bandpassgefiltertes Echtrauschen. Im Modus 'LIGO Zufall' sind Signal und Rauschen dagegen gemeinsam gefiltert. | `src/dataset/generate_chirp_dataset.py`:94-97, 115-128 (vgl. src/dataset/fetch_real_noise.py:37-40, src/glitch_hunter_app.py:169-181) | P4.1, P6.2 |
| APP-2 | mittel | teilweise | Simulationsmodus des Glitch Hunters erzeugt Daten außerhalb der Trainingsverteilung | `src/glitch_hunter_app.py`:259-277 | P3.5, P4.4, P6.2, P6.5 |
| APP-3 | mittel | bestätigt | Threshold ist hart codiert, nicht an das geladene Modell gekoppelt und im Repo inkonsistent | `src/glitch_hunter_app.py`:55, 73, 92 | P3.3, P5.8 |
| APP-5 | mittel | bestätigt | Race-Condition im Download-Thread führt zu falscher 'Wahrheit' | `src/glitch_hunter_app.py`:112-157, 279-308 | P4.4 |
| APP-6 | mittel | bestätigt | matplotlib-Zeichenaufrufe aus dem Worker-Thread (nicht thread-sicher) | `src/glitch_hunter_app.py`:156, 190, 195, 197-237 | P4.4 |
| APP-10 | mittel | teilweise | Filterkette für echte Daten liegt dreimal vor, eine Kopie weicht ab | `src/glitch_hunter_app.py`:163-181 | P3.2, P4.4 |
| APP-13 | mittel | bestätigt | Spielzeugmodell: Slider wirken im relevanten Band kaum, 'LIGO Design'-Referenz ist ad hoc | `src/gwd_core/interferometer.py`:92-128 | P5.4 |
| apps:APPX-2 | mittel | vom Prüfer ergänzt | Das interaktive GUI-Backend ist nirgends deklariert. Ohne Backend beenden sich die Apps stumm mit Exit-Code 0. | `requirements.txt` | P0.5, P4.5, P7.1 |
| apps:APPX-3 | mittel | vom Prüfer ergänzt | Die 'Wahrheit'-Events liegen teils außerhalb des Trainings-Massenbereichs, und im Echtdaten-Modus wird die KI-Antwort nie bewertet. | `src/glitch_hunter_app.py`:24-34, 292-298 | P3.4, P4.4 |
| apps:APPX-4 | mittel | vom Prüfer ergänzt | Die Auswahl des 'neuesten Modells' über os.path.getctime ist unter Linux und WSL nicht 'Erstellungszeit'. | `src/glitch_hunter_app.py`:55 (ebenso src/utils/evaluation.py:19, src/optimize_threshold.py:43, src/compare_models_roc.py:40) | P3.3, P4.4 |
| APP-4 | niedrig | bestätigt | Slider zeigt 0.50, die Entscheidung nutzt 0.46; Zweig 'UNSICHER' beim Start unerreichbar | `src/glitch_hunter_app.py`:73, 92, 302, 315-323 | P4.4, P5.8 |
| APP-7 | niedrig | bestätigt | Harte Top-Level-Imports von TensorFlow und gwpy | `src/glitch_hunter_app.py`:11, 17 | P2.6, P3.4, P4.4 |
| APP-8 | niedrig | bestätigt | sys.exit(1) im Konstruktor, toter Prüfzweig, veralteter Pfad in der Fehlermeldung | `src/glitch_hunter_app.py`:43-52 | P0.5, P2.3, P3.3, P4.4 |
| APP-9 | niedrig | bestätigt | Audio: temp_sound.wav im CWD, blockierender Aufruf, fragile Fallbacks | `src/glitch_hunter_app.py`:239-253 | P4.4 |
| APP-11 | niedrig | bestätigt | Variable ax_noise überschrieben, deshalb skaliert der Rausch-Plot nicht | `src/gw_simulator.py`:38, 79, 104-106 | P4.5 |
| APP-12 | niedrig | bestätigt | Finesse springt beim ersten Bewegen des Spiegel-Sliders (Physik im GUI, inkonsistent zum Modell) | `src/interferometer_simulator.py`:166-171 | P4.5 |
| APP-14 | niedrig | teilweise | App-spezifische Physik liegt im gemeinsamen Kernpaket | `src/gwd_core/simulation.py` | P1.5, P2.5 |
| APP-15 | niedrig | bestätigt | Emoji-Glyphen fehlen im Standard-Font | `src/interferometer_simulator.py`:87-93 | P4.5 |
| APP-16 | niedrig | bestätigt | Apps sind kaum dokumentiert und unvollständig ins Menü eingebunden | `README.md` | P2.5, P3.6, P7.1 |
| APP-17 | niedrig | bestätigt | sys.path-Hacks und toter Code in den Demo-Apps | `src/gw_simulator.py`:8, 15, 47 | P2.5, P4.5 |
| APP-18 | niedrig | bestätigt | Keine Tests für die Apps | `tests/` | P0.5, P7.4 |
| APP-19 | niedrig | bestätigt | Kleinere Schwächen beim Modus 'LIGO Zufall' | `src/glitch_hunter_app.py`:36-37, 141-157, 199-210, 234 | P4.4 |
| apps:APPX-5 | niedrig | vom Prüfer ergänzt | Die Apps starten nur, wenn das CWD der Repo-Root ist. Dorthin wandert dann auch temp_sound.wav. | `main.py`:14, 50-51 | P2.4, P2.5 |

## Repo-Hygiene, Doku, Abhängigkeiten

hoch: 5 · mittel: 10 · niedrig: 15

| ID | Schwere | Urteil | Befund | Ort | Schritt |
|---|---|---|---|---|---|
| HYG-1 | hoch | teilweise | 7 byte-identische, nicht lauffähige Kopien der src-Skripte | `analysis/, apps/` | P0.1, P1.3, P2.6, P7.4 |
| HYG-2 | hoch | bestätigt | Leaderboard-Ausgabe liegt im Code-Ordner analysis/ – blockiert das Löschen der Duplikate | `config.py`:19 | P1.2 |
| HYG-4 | hoch | bestätigt | Die beworbene Matched-Filter-Baseline existiert nicht (mehr) | `README.md`:8 | P0.4, P5.8, P6.4 |
| HYG-6 | hoch | teilweise | Pins lassen sich unter Python 3.13 nicht per Wheel installieren; Reproduzierbarkeit wird nur behauptet | `requirements.txt`:1-22 | P0.4, P2.1, P7.1, P7.3 |
| hygiene:HYG-M1 | hoch | vom Prüfer ergänzt | Zwei stille Umgebungsschalter bestimmen die Trainings- und Evaluationsdaten, ohne protokolliert zu werden (pycbc vorhanden? echtes Rauschen geladen?) | `src/gwd_core/waveforms.py`:10-15, 52-70; src/dataset/generate_chirp_dataset.py:79, 94-97, 141-149 | P0.1, P2.2, P3.5, P4.1, P5.2 |
| HYG-3 | mittel | bestätigt | 49 von 54 Doku-Befehlen zeigen auf nicht existierende Pfade | `docs/*.md`:FINAL_SUMMARY 91-305, INSTALLATION_CHECKLIST 83-237, ROC_ANALYSIS_README 60-208, ROC_INTEGRATION_OVERVIEW 19-330 | P1.4, P7.1 |
| HYG-5 | mittel | bestätigt | Zustand des Agent-Tools (ruflo/claude-flow) ist committet | `.swarm/, .claude-flow/` | P0.1, P1.1 |
| HYG-7 | mittel | teilweise | Keine Trennung in Laufzeit, Dev und Optional; schwere Pakete für alle | `requirements.txt`:6-21 | P2.1 |
| HYG-8 | mittel | bestätigt | .gitignore: CRLF, zu breite Endungsmuster, fehlende Verzeichnisse, Altlasten | `.gitignore`:1-50 | P1.1 |
| HYG-9 | mittel | teilweise | PROJECT_OVERVIEW enthält mehrere Falschaussagen und kaputte Links | `PROJECT_OVERVIEW.md`:12-44 | P1.5, P7.1 |
| HYG-10 | mittel | bestätigt | README unvollständig und teils veraltet; Lizenz und Regeln ohne Grundlage | `README.md`:8-54 | P0.4, P7.1, P7.2 |
| HYG-13 | mittel | bestätigt | Vier redundante Liefernotizen aus einem KI-Chat statt Projektdoku | `docs/` | P1.4, P7.1 |
| HYG-14 | mittel | bestätigt | Unbelegte bzw. erfundene Zahlen; Schwellwerte in Doku und Code passen nicht zusammen | `docs/ROC_ANALYSIS_README.md`:44, 214, 218-221 | P1.4, P6.5, P7.1 |
| HYG-20 | mittel | bestätigt | Fehlende Projekt-Infrastruktur: Packaging, CI, Lizenz, Beitragsregeln | `(Projekt-Root)` | P2.1, P2.2, P2.6, P7.2 |
| hygiene:HYG-M2 | mittel | vom Prüfer ergänzt | Reproduzierbarkeit wird versprochen, im Code wird aber nirgends ein Zufalls-Seed gesetzt | `README.md, PROJECT_OVERVIEW.md`:README 17; PROJECT_OVERVIEW 13 | P0.4, P1.4, P3.6, P5.6, P5.7 |
| HYG-11 | niedrig | bestätigt | Selbstbericht des KI-Refactors mit nachweislich falschen Behauptungen | `.planning/walkthrough.md`:7-16 | P1.1 |
| HYG-12 | niedrig | bestätigt | Commit c4e3dd8 enthält requirements.txt mit Merge-Konflikt-Markern | `requirements.txt` | P2.2 |
| HYG-15 | niedrig | bestätigt | Leaderboard-Schema falsch beschrieben; zwei Schreib-Funktionen mit unterschiedlichen Spalten | `docs/INSTALLATION_CHECKLIST.md`:195-219 | P0.4, P7.1 |
| HYG-16 | niedrig | teilweise | pytest sammelt ein Nicht-Test-Skript ein und führt es aus; der Core-Check ist veraltet | `utility/test_imports.py`:37 | P0.5, P2.1, P3.6 |
| HYG-17 | niedrig | bestätigt | Erledigtes Einmal-Migrationsskript, das in place schreibt | `utility/fix_plots.py`:5-6, 26-27 | P1.4 |
| HYG-18 | niedrig | bestätigt | Sprachmix Deutsch/Englisch quer durch alle Schichten | `src/ (gesamt), Doku` | P7.1 |
| HYG-19 | niedrig | bestätigt | Gemischte Zeilenenden, fehlende End-Newlines, keine Formatierungs- und Lint-Regeln | `src/gwd_core/*.py, src/dataset/*.py, .gitignore` | P1.6, P2.2 |
| HYG-21 | niedrig | bestätigt | Menü (main.py) hängt vom aktuellen Verzeichnis ab, ist unvollständig und undokumentiert | `main.py`:4, 14, 43-51 | P2.4, P2.5, P3.6 |
| HYG-22 | niedrig | bestätigt | Wörtliches '\n' in der Ausgabe; nur FileNotFoundError wird abgefangen | `src/utils/evaluation.py`:34, 62, 78, 109 | P0.4 |
| HYG-23 | niedrig | bestätigt | Commit-Messages ohne Aussage; Tool-Init und Code-Fixes vermischt | `(Git-Historie)` | P0.1, P1.5 |
| HYG-24 | niedrig | bestätigt | Zoom-Plot überschreibt den Vergleichsplot (Doku verspricht beide) | `src/compare_models_roc.py`:136, 173 | P4.2 |
| HYG-25 | niedrig | teilweise | Veraltete und irreführende Kommentare in requirements.txt | `requirements.txt`:13-20 | P2.1 |
| hygiene:HYG-M3 | niedrig | vom Prüfer ergänzt | Ergebnis der Schwellwert-Optimierung landet im Plot-Ordner und wird von nichts gelesen | `config.py`:21; src/optimize_threshold.py:330; src/glitch_hunter_app.py:73 | P1.2, P5.8 |
| hygiene:HYG-M4 | niedrig | vom Prüfer ergänzt | Physik_Gap erhält immer ein '+', auch wenn der Gap negativ ist | `src/utils/evaluation.py`:33, 77 | P0.4 |
| hygiene:HYG-M5 | niedrig | vom Prüfer ergänzt | sys.path.append mit generischen Top-Level-Namen (config, utils, dataset): installierte Pakete würden die lokalen Module verdecken | `src/evaluate_model_with_roc.py, src/dataset/train_cnn.py, src/show_leaderboard.py`:4/27; 3-4; 3 | P1.5, P2.1, P2.3, P2.4, P2.5, P2.6 |
