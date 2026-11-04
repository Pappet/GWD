# Aufräumplan GWD

> **Status:** Entwurf vom 03.11.2026, noch nichts davon umgesetzt.
> **Stand der Analyse:** `main` @ `c54acd5` (13 Commits).
> **Befund-Register mit allen IDs:** [BEFUNDE.md](BEFUNDE.md)
>
> Dieser Ordner (`docs/aufraeumen/`) ist ein Arbeitsdokument. In Schritt P1.5 wird daraus `docs/architecture.md`, in P7.4 wird der Ordner gelöscht. Die Git-Historie behält ihn.

## Inhalt

1. [Wie der Plan entstanden ist](#1-wie-der-plan-entstanden-ist)
2. [Diagnose: Was ist vermischt?](#2-diagnose-was-ist-vermischt)
3. [Zielbild](#3-zielbild)
4. [Fahrplan im Überblick](#4-fahrplan-im-überblick)
5. [Entscheidungen für den Besitzer](#5-entscheidungen-für-den-besitzer)
6. [Schritte im Detail](#6-schritte-im-detail)
7. [Risiken und Gegenmittel](#7-risiken-und-gegenmittel)
8. [Bewusst weggelassen](#8-bewusst-weggelassen)
9. [Umsetzung mit mehreren Agenten](#9-umsetzung-mit-mehreren-agenten)

---

## 1. Wie der Plan entstanden ist

1. **Bestandsaufnahme.** Fünf Analysten haben je eine Komponente untersucht: Physik-Kern, Daten und Training, Evaluation, Apps sowie Repo-Hygiene. Sie haben den Code gelesen und in einer Kopie des Repos ausgeführt. Danach hat je ein Prüfer versucht, jeden Befund zu widerlegen. Dabei wurden Schweregrade korrigiert und übersehene Probleme ergänzt.
   - Ergebnis: **138 geprüfte Befunde**, jeder mit Datei und Zeile (siehe [BEFUNDE.md](BEFUNDE.md)).
2. **Planung.** Drei Planentwürfe sind unabhängig voneinander entstanden, mit je eigenem Schwerpunkt: *inkrementell und sicher*, *Architektur zuerst* und *fachliche Korrektheit zuerst*.
   - Eine Jury hat die Entwürfe bewertet und zusammengeführt. Basis wurde „inkrementell“; die anderen beiden Entwürfe haben das Zielbild und die Korrektheitstests beigesteuert.
   - Danach hat ein Vollständigkeitsprüfer jede Befund-ID gegen den Plan abgeglichen und die Reihenfolge geprüft. Seine Einwände sind eingearbeitet.
3. **Stichproben.** Die zentralen Befunde wurden zusätzlich direkt am Code nachgeprüft. Auszug:
   - Die SNR-Bereiche 1–12 / 0,5–2,5 / 0,5–2,0 stehen so im Code.
   - `validation_data=(X_test, y_test)` steht in `train_cnn.py:294`.
   - Im ganzen Code wird kein einziger Seed gesetzt.
   - Mit matplotlib 3.11 stürzt `validate_dataset.py:183` ab.

Abgedeckt sind 137 der 138 Befunde. DATA-26 (CNN-Architektur) ist bewusst ausgeklammert, siehe [Abschnitt 8](#8-bewusst-weggelassen).

---

## 2. Diagnose: Was ist vermischt?

Im Repo stecken **fünf Komponenten**, die heute ineinanderlaufen:

| Komponente | Liegt heute in | Was vermischt ist |
|---|---|---|
| **Physik-Kern** (Wellenformen, Rauschen, Interferometer) | `src/gwd_core/` | Der Kern selbst ist sauber abgegrenzt. Seine Logik ist aber zusätzlich in Datensatz, Evaluation und Apps nachgebaut: Injektion/SNR-Skalierung 6×, Filterkette 3×, Finesse und Referenzkurve in der App. |
| **Daten und Training** | `src/dataset/` (inklusive `train_cnn.py`), `src/utils/data_processing.py` | Das Training liegt im Ordner `dataset/`, die Konstanten sind hart codiert statt aus `config.py` gelesen, und das Laden scheitert still. |
| **Evaluation** | `src/evaluate_model_with_roc.py`, `optimize_threshold.py`, `compare_models_roc.py`, `show_leaderboard.py`, `src/utils/evaluation.py` **und** `analysis/` | Jedes Skript hat eine eigene Injektion, ein eigenes SNR-Raster und eine eigene Modellsuche. `analysis/` enthält byte-identische, nicht lauffähige Kopien, und das Leaderboard zeigt ausgerechnet dorthin. |
| **Interaktive Apps** | `src/glitch_hunter_app.py`, `gw_simulator.py`, `interferometer_simulator.py` **und** `apps/` | Die Apps bringen eigene Physik, Filterung, Modellsuche und Schwelle mit. `apps/` enthält byte-identische, nicht lauffähige Kopien. |
| **Infrastruktur und Doku** | `config.py`, `main.py`, `requirements.txt`, `README.md`, `PROJECT_OVERVIEW.md`, `docs/`, `.planning/`, `.swarm/`, `.claude-flow/`, `utility/` | Committeter Tool-Zustand, KI-Liefernotizen mit erfundenen Werten, nicht installierbare Pins und 49 von 54 Doku-Befehlen, die ins Leere zeigen. |

### Die wichtigsten Befunde

**Struktur: das, was „Durcheinander“ heißt**

- **Duplikate.** Commit `c4e3dd8` hat Dateien nach `src/` kopiert statt verschoben. `analysis/*.py` und `apps/*.py` sind byte-identische Kopien von 7 Skripten und scheitern alle mit `ModuleNotFoundError: gwd_core` (HYG-1, APP-1, EVAL-9).
- **Zwei Modulidentitäten.** Die Tests importieren `src.gwd_core`, die Skripte `gwd_core` über `sys.path`-Hacks in jeder Datei. Das Paket ist nicht installierbar (CORE-5).
- **Leaderboard im Duplikat-Ordner.** `LEADERBOARD_FILE` zeigt nach `analysis/`. Wer den Ordner naiv löscht, bricht das Leaderboard und verliert die lokale CSV (HYG-2).
- **Nicht installierbar.** `requirements.txt` lässt sich auf keiner einzigen Python-Version vollständig aus Wheels installieren; `pycbc==2.4.0` hat keine Wheels für 3.12 und gar keine für Windows (core:MISS-1).
- **Doku und Tool-Zustand.**
  - 49 von 54 Befehlen in der Doku funktionieren nicht.
  - Das README verspricht „reproducible builds“ und eine MIT-Lizenz, eine LICENSE-Datei gibt es aber nicht.
  - `.swarm/` und `.claude-flow/` (ruflo) sowie KI-Selbstberichte sind committet (HYG-3 bis HYG-15).

**Fachlich: wichtiger als die Ordnerstruktur**

- **Train/Eval/App-Skew.** Die drei Programmteile ziehen ihre Daten aus verschiedenen Verteilungen. Die Evaluation misst deshalb fast nur Extrapolation (DATA-1, EVAL-1, APP-2, data-training:NOISE-SKEW).

  | | Peak-SNR | Rauschen |
  |---|---|---|
  | Training | 1–12 | 80 % echtes, gefiltertes H1-Rauschen |
  | Evaluation | 0,5–2,5 | 100 % weißes Rauschen |
  | Glitch Hunter | 0,5–2,0 | weißes Rauschen |

- **„SNR“ ist kein SNR.** Gemeint ist die Peak-Amplitude geteilt durch σ, nicht das Matched-Filter-SNR. Peak-SNR 1 entspricht je nach Massen ρ_opt ≈ 10–17. Das Training deckt damit ρ_opt ≈ 10–200 ab (data-training:SNR-DEF, evaluation:MISS-2).
- **Mögliche Abkürzung für das CNN.** Das Training addiert *ungefilterte* Chirps zu bandpassgefiltertem Rauschen. Oberhalb von 300 Hz liegen 2–5 % Signalenergie, die es in echten Daten so nie gibt (apps:APPX-1, DATA-12).
- **Kein ehrlicher Split.**
  - Das Validierungsset ist gleich dem Testset (`train_cnn.py:294`). Die gemeldete Test-Accuracy ist dadurch geschönt (DATA-2).
  - Dieselben Rausch-Chunks aus *einem* 68-Minuten-Segment werden etwa 8× quer über Train und Test verwendet (DATA-11).
  - Seeds gibt es nirgends.
- **Stiller Newton-Fallback.** Fehlt pycbc oder wirft es einen Fehler, rechnet der Code still mit einer Newton-Näherung. Die ist falsch: Frequenz ×2,045, Amplitude ×4, kein ISCO-Abbruch. Welches Backend aktiv war, wird nirgends protokolliert (CORE-2/3/4/8, hygiene:HYG-M1).
- **Vier verschiedene Schwellen.** Evaluation 0,15, Glitch Hunter 0,46, ein Kommentar sagt 0,75, die Doku 0,7234. Das Ergebnis von `optimize_threshold` liest niemand (EVAL-15, APP-3, hygiene:HYG-M3).
- **Das Ranking belohnt Dauerfeuer.** Das Leaderboard sortiert nach SNR50, gemessen bei fester Schwelle ohne FPR-Bezug. Ein Modell, das immer „Signal“ sagt, landet vorn (evaluation:MISS-1).
- **`Physik_Gap` ist ungültig.** Der Wert stammt aus einer Matched-Filter-Baseline, deren Skript seit `45b9202` gelöscht ist und die einen Template-Bug hatte (Match 0,07–0,14) (EVAL-6).

**Robustheit**

- `validate_dataset.py` stürzt mit aktuellem matplotlib ab (DATA-3).
- `fetch_real_noise` schluckt Fehler und endet mit Exit 0 (DATA-6). Netzfehler im Feldtest erscheinen als „0/5 gefunden“ (EVAL-13).
- Glitch Hunter:
  - Race-Condition im Download-Thread und matplotlib-Aufrufe aus dem Worker-Thread (APP-5, APP-6);
  - `temp_sound.wav` landet im Arbeitsverzeichnis;
  - ohne Modell `sys.exit` im Konstruktor.
- Ohne GUI-Backend beenden sich die Apps stumm mit Exit 0 (apps:APPX-2).
- `config.py` legt schon beim Import Ordner an (DATA-7). Modelle werden nach `ctime` ausgewählt, und das springt unter Linux und WSL (apps:APPX-4).

**Was gut ist und bleibt**

- Der PyCBC-Pfad legt den Merger exakt; gemessen liegt die Abweichung im Median bei 0,07 ms.
- Der Kern importiert weder TensorFlow noch matplotlib.
- Die alten `.keras`-Modelle bestehen nur aus Standardschichten und lassen sich weiter laden.

---

## 3. Zielbild

### 3.1 Leitidee

Leitmotiv ist die Komponententrennung. Am Ende steht EIN installierbares Paket gwd mit klar getrennten Komponenten: config (Pfade/Konstanten), core (Physik), data (Szenarien, Injektion, Konditionierung, GWOSC, Datensätze), registry (Modelle und Schwellen), training, evaluation, apps (nur GUI) und cli (nur Dispatch). Die Abhängigkeitsrichtung steht in docs/architecture.md. Ein ast-basierter Architektur-Test erzwingt sie ohne neue Abhängigkeit. Er verbietet außerdem sys.path-Hacks und byte-identische Module, damit sich c4e3dd8 („kopiert statt verschoben“) nicht wiederholt. Bekannte Altlasten stehen in einer ALLOWLIST, die nur schrumpfen darf.

Die Daten des Besitzers sind tabu. Migrationen kopieren. Neue Formate landen in neuen Dateien oder Ordnern (results/leaderboard_v2.csv, data/training_v2, data/noise_raw). Alte Modelle werden archiviert statt gelöscht. Die .gitignore deckt auch die Altpfade gw_training_data/, gw_noise_background/ und die CSV/JSON-Dateien im Root ab.

Regeln für KI-Assistenten stehen knapp in AGENTS.md: git mv statt Kopie, nichts in Datenordnern löschen, ein Thema pro Commit, Schritt-ID nennen, schnelle Tests vor jedem Commit und alle vor dem PR, Golden-Werte nur in 'golden:'-Commits.

Augenmaß: keine Frameworks (kein MLflow, DVC, Hydra, Docker, mypy oder import-linter). Es gibt eine CI-Datei (schneller Job ohne TensorFlow, langsamer Job manuell), einen Architektur-Test, einen Doku-Befehlstest, einen Golden-Master ohne Versionsschlüssel und keine Statistik auf Forschungsniveau.

### 3.2 Zielstruktur

```text
GWD/
├── pyproject.toml              # Distribution gwd (src-Layout, packages.find include=['gwd*']), requires-python >=3.11
│                               # Basis: numpy, scipy, pandas, matplotlib>=3.9, scikit-learn, tabulate
│                               # Extras: [physics]=pycbc>=2.5 (nicht Windows) · [gwdata]=gwpy · [ml]=tensorflow · [dev]=pytest,ruff · [all]
│                               # [project.scripts] gwd = gwd.cli:main; [tool.pytest] testpaths, Marker slow/apps/golden/network; [tool.ruff]
├── README.md                   # einzige Einstiegsdoku: Zweck, Komponenten, Plattform Linux/WSL, Installation je Extra, gwd-Befehle, Apps, Tests, Datenablage/GWD_HOME
├── AGENTS.md (+ CLAUDE.md)     # Kernregeln für KI-Assistenten und Sprachregel; CLAUDE.md verweist nur (ruflo-Teile → CLAUDE.local.md, ignoriert)
├── LICENSE                     # nur nach D12
├── .gitignore .gitattributes .git-blame-ignore-revs   # LF; Verzeichnisregeln inkl. Altpfade vor f724856; nur Binärendungen global
├── .github/workflows/ci.yml    # Job test: ruff + pytest ohne TF, mit pycbc, GWD_REQUIRE_PYCBC=1 · Job slow: workflow_dispatch, [all]
├── main.py                     # dünnes Menü, nur Stdlib: feste MENU-Liste von gwd-Argumenten → subprocess [python -m gwd.cli …]
├── docs/
│   ├── architecture.md         # Komponenten, Importregeln (= tests/test_architecture.py), Verschiebetabelle, Fakten F1–F4, Antworten D1–D12, Szenarien V1/V2, SNR-Definition, Schritt-Checkliste
│   └── evaluation.md           # Metriken, Plots, Schwellen-Workflow, Leaderboard-v2-Schema, Baseline-Status, gemessene Ergebnisse
├── src/gwd/
│   ├── config.py               # Wurzel: GWD_HOME > Elternordner mit pyproject.toml > Fehler; Pfade, Konstanten, ensure_dirs(); keine Import-Seiteneffekte
│   ├── registry.py             # Modelle listen/auswählen (Zeitstempel im Namen)/laden (TF lazy, compile=False); Metadaten; Schwelle je Modell
│   ├── env.py                  # gwd env: Versionen, Extras, aktives Wellenform-Backend, Pfade, GUI-Backend (← utility/test_imports.py)
│   ├── cli.py                  # gwd data|train|eval|app|env – nur Dispatch, Importe lazy; build_parser()
│   ├── core/                   # PHYSIK (← src/gwd_core): reine Funktionen, rng als Parameter, keine I/O, kein config, kein print
│   │   ├── constants.py timebase.py population.py
│   │   ├── waveforms.py        # generate_chirp(…, backend=) ohne stillen Fallback
│   │   ├── newtonian.py        # nur bei D1=B
│   │   ├── noise.py snr.py     # white_noise, colored_noise_from_psd; peak_snr, optimal_snr_*, scale_to_snr
│   │   └── interferometer.py simulation.py   # qualitatives Lernmodell (finesse-Property, reference_asd), TimeDomainSimulator
│   ├── data/
│   │   ├── scenarios.py        # Scenario-Dataclass + Hash: TRAIN_V1/EVAL_V1/APP_V1 (= heute), TRAIN_V2/INDIST_V2/STRESS_V2/APP_V2
│   │   ├── injection.py        # draw_params, make_sample – einzige Stelle für Signal + Rauschen
│   │   ├── preprocessing.py    # condition_strain, conditioned_crop (Puffer), normalize – einzige Filterkette
│   │   ├── gwosc.py            # fetch_strain (gwpy lazy, Cache, NaN-Prüfung), EVENTS mit Typ/Massen/OOD-Flag
│   │   ├── fetch_noise.py generate.py   # labels.csv + manifest.json
│   │   └── validate.py         # Integrität + Bereichsprüfung gegen das Szenario; Report-Plots optional
│   ├── training/ {model.py, train.py}   # build_model(); train(cfg): gruppierter 3-Wege-Split, Seeds, Metadaten, Mixed Precision nur mit GPU/Flag
│   ├── evaluation/
│   │   ├── metrics.py testsets.py       # reine Metriken/Strategien; feste, gehashte Testsets (calib_bg, test)
│   │   ├── evaluate.py thresholds.py compare.py real_data.py leaderboard.py   # Leaderboard v2 mit Rohzahlen
│   │   └── baseline_mf.py      # nur bei D5=A
│   └── apps/
│       ├── _backend.py waveform_explorer.py interferometer_designer.py
│       └── glitch_hunter/ {app.py, sources.py, audio.py}
├── tests/
│   ├── core/ data/ training/ evaluation/ apps/   # Unit-Tests je Komponente
│   ├── smoke/ golden/          # Pipeline-/App-Smoke in tmp-GWD_HOME; Golden-Kennwerte (values.py, regen.py)
│   └── test_architecture.py test_menu.py test_cli.py test_docs_commands.py test_e2e.py
└── lokal, ignoriert, NIE löschen: data/{training,training_v2,noise_background,noise_raw,eval}/ · models_registry/{,archive_v1/}
    · plots/ · results/{model_leaderboard.csv (v1-Archiv), leaderboard_v2.csv, optimal_thresholds.json, testsets/}
    · analysis/model_leaderboard.csv · Altpfade gw_training_data/ gw_noise_background/ ./model_leaderboard.csv ./optimal_thresholds.json · .swarm/ .claude-flow/
```

### 3.3 Importregeln (werden in P2.6 per Test erzwungen)

Importregeln:

- config → Stdlib
- core → numpy/scipy/pycbc (lazy)
- data → core, config (gwpy lazy nur in gwosc.py und preprocessing.py)
- registry → config (TF lazy)
- training → data, core, config, registry
- evaluation → data, core, config, registry
- apps → core, data, config, registry
- cli/env → alles, nur lazy
- main.py → nur Stdlib

### 3.4 Was entfällt

Entfällt: analysis/*.py, apps/*.py, utility/, .planning/, src/dataset/, src/utils/, src/gwd_core/ (→ src/gwd/core), Root-config.py, requirements.txt (ggf. → Lockfile nach D7), tests/conftest.py, docs/FINAL_SUMMARY.md, docs/INSTALLATION_CHECKLIST.md, docs/ROC_* (→ evaluation.md), PROJECT_OVERVIEW.md (→ docs/architecture.md)

### 3.5 Komponenten und Grenzen

#### Konfiguration (gwd.config)

**Ort:** `src/gwd/config.py`

Bestimmt die Projektwurzel unabhängig vom Arbeitsverzeichnis: GWD_HOME, sonst der nächste Elternordner mit pyproject.toml, sonst ein Fehler mit Hinweis. Enthält alle Pfade (data/, models_registry/, plots/, results/ sowie Altpfad-Konstanten für die Leaderboard-Migration) und alle Konstanten (SAMPLE_RATE, DURATION, N_SAMPLES, NOTCH_FREQS, BANDPASS, DEFAULT_SEED, WAVEFORM_BACKEND, DEFAULT_SCENARIO, Hyperparameter-Defaults). ensure_dirs() legt Ordner nur auf Abruf an. Beim Import passiert nichts: keine Ordner, keine Migration.

- **Darf importieren:** Python-Standardbibliothek
- **Darf nicht importieren:** gwd.core, gwd.data, gwd.registry, gwd.training, gwd.evaluation, gwd.apps, gwd.cli, numpy, tensorflow, gwpy, pycbc, matplotlib
- **Externe Abhängigkeiten:** –
- **Kommt heute aus:** `config.py`

#### Physik-Kern (gwd.core)

**Ort:** `src/gwd/core/`

Wellenformen mit explizitem Backend: PyCBC, nach D1 ggf. ein korrigiertes Newton-Backend. Zufallsparameter mit rng, Zeitachse, Konstanten, weißes und PSD-basiertes Rauschen mit korrekter Normierung, SNR-Definitionen und Skalierung, das Interferometer-Lernmodell (finesse-Property, reference_asd) und der TimeDomainSimulator. Abtastrate, Dauer und rng kommen als Argument. Keine Datei- oder Netz-I/O, kein config, kein print, kein globaler RNG.

- **Darf importieren:** gwd.core.*, Standardbibliothek, numpy, scipy, pycbc/lal (nur lazy im Backend bzw. in reference_asd)
- **Darf nicht importieren:** gwd.config, gwd.data, gwd.registry, gwd.training, gwd.evaluation, gwd.apps, gwd.cli, tensorflow, matplotlib, gwpy, pandas, sklearn
- **Externe Abhängigkeiten:** numpy, scipy, pycbc>=2.5 (Extra [physics]; Linux/macOS/WSL)
- **Kommt heute aus:** `src/gwd_core/__init__.py`, `src/gwd_core/waveforms.py`, `src/gwd_core/noise.py`, `src/gwd_core/interferometer.py`, `src/gwd_core/simulation.py`, `scale_signal_to_snr() aus src/dataset/generate_chirp_dataset.py`, `Finesse-Formel und ligo_ref aus src/interferometer_simulator.py`

#### Daten (gwd.data)

**Ort:** `src/gwd/data/`

Alles, was aus Physik ML-Daten macht. Benannte, gehashte Szenarien: V1 entspricht dem heutigen Verhalten, V2 heißt „erst injizieren, dann gemeinsam konditionieren“. Die Injektion ist die EINZIGE Quelle für Signal + Rauschen (Datensatz, Testsets, Glitch Hunter). Konditionierung und Normierung sind die EINZIGE Filterkette, auch mit Puffer. Dazu der GWOSC-Zugriff mit Cache und NaN-Prüfung, eine gemeinsame Event-Liste mit OOD-Flag, der Rauschdownload, das Schreiben von Datensätzen (labels.csv + manifest.json) sowie Integritäts- und Bereichsprüfung.

- **Darf importieren:** gwd.core, gwd.config, numpy, scipy, pandas, gwpy (lazy, nur gwosc.py und preprocessing.py), matplotlib (lazy, nur Validierungsreport)
- **Darf nicht importieren:** gwd.registry, gwd.training, gwd.evaluation, gwd.apps, gwd.cli, tensorflow, sklearn
- **Externe Abhängigkeiten:** numpy, scipy, pandas, matplotlib (Report), gwpy (Extra [gwdata])
- **Kommt heute aus:** `src/dataset/__init__.py`, `src/dataset/fetch_real_noise.py`, `src/dataset/generate_chirp_dataset.py`, `src/dataset/validate_dataset.py`, `src/utils/data_processing.py`, `Inline-Injektions- und Filterkopien in src/evaluate_model_with_roc.py, src/optimize_threshold.py, src/compare_models_roc.py, src/glitch_hunter_app.py`, `REAL_EVENTS (evaluate_model_with_roc.py) und KNOWN_EVENTS (glitch_hunter_app.py)`

#### Modell-Registry (gwd.registry)

**Ort:** `src/gwd/registry.py`

Die einzige Stelle, die Modelle findet, auswählt und lädt: list_models, latest_model (nach dem Zeitstempel im Dateinamen statt nach ctime), resolve('latest'|Name|Pfad), load_model(compile=False, TF lazy). Dazu Metadaten lesen und schreiben, get_threshold/set_threshold je Modell (Fallback mit Warnung) und die Kennzeichnung von Legacy-Modellen ohne Manifest.

- **Darf importieren:** gwd.config, Standardbibliothek, tensorflow (nur lazy in load_model)
- **Darf nicht importieren:** gwd.data, gwd.training, gwd.evaluation, gwd.apps, gwd.cli, matplotlib, gwpy
- **Externe Abhängigkeiten:** tensorflow (Extra [ml], lazy)
- **Kommt heute aus:** `get_latest_model() aus src/utils/evaluation.py`, `get_latest_model() aus src/optimize_threshold.py`, `load_all_models() aus src/compare_models_roc.py`, `Modellsuche in src/glitch_hunter_app.py:43-58`

#### Training (gwd.training)

**Ort:** `src/gwd/training/`

build_model() ohne Seiteneffekte; die Architektur ist per count_params-Test fixiert. train(cfg) prüft zuerst die Integrität des Datensatzes, lädt in float32 und teilt stratifiziert und nach Rausch-Chunk gruppiert in drei Mengen (mit Warnung bei Altdatensätzen). Seeds werden gesetzt; .keras und vollständige Metadaten werden über die Registry gespeichert. Mixed Precision nur bei GPU oder per Flag.

- **Darf importieren:** gwd.data, gwd.core, gwd.config, gwd.registry, tensorflow, sklearn, matplotlib (lazy, Trainingskurve), numpy, pandas
- **Darf nicht importieren:** gwd.evaluation, gwd.apps, gwd.cli, gwpy
- **Externe Abhängigkeiten:** tensorflow (Extra [ml]), scikit-learn
- **Kommt heute aus:** `src/dataset/train_cnn.py`

#### Evaluation (gwd.evaluation)

**Ort:** `src/gwd/evaluation/`

Feste, gehashte Testsets (calib_bg, test) aus gwd.data.injection; ihre Manifeste mit den Parametern je Sample liegen unter results/testsets/. Metriken und Schwellen-Strategien sind reine Funktionen. Kommandos: evaluate, thresholds (Schwelle pro Modell bei fester FPR auf calib_bg) und compare. Der Echtdaten-Feldtest läuft nur mit --real. Das Leaderboard v2 speichert Rohzahlen; die alte v1-Datei bleibt unangetastet als Archiv liegen. Optional die Matched-Filter-Baseline (D5).

- **Darf importieren:** gwd.data, gwd.core, gwd.config, gwd.registry, sklearn, pandas, matplotlib, tabulate, pycbc (nur baseline_mf)
- **Darf nicht importieren:** gwd.training, gwd.apps, gwd.cli, gwpy direkt (nur über gwd.data.gwosc), tensorflow direkt (nur über gwd.registry)
- **Externe Abhängigkeiten:** scikit-learn, pandas, matplotlib, tabulate (Basis), tensorflow (über registry), gwpy (optional, --real), pycbc (nur Baseline)
- **Kommt heute aus:** `src/evaluate_model_with_roc.py`, `src/optimize_threshold.py`, `src/compare_models_roc.py`, `src/show_leaderboard.py`, `src/utils/evaluation.py (Leaderboard-Teil)`, `analysis/*.py (Duplikate, entfallen)`, `evaluate_classic.py (nur Historie 45b9202^, Referenz bei D5=A)`

#### Interaktive Apps (gwd.apps)

**Ort:** `src/gwd/apps/`

Waveform-Explorer, Interferometer-Designer und Glitch Hunter (app.py GUI, sources.py Sim/GWOSC, audio.py) sind nur für Darstellung, Interaktion und Audio zuständig. build_figure() ist von main() und plt.show() getrennt. Plot-Updates laufen nur im GUI-Thread. Ohne interaktives Backend gibt es eine verständliche Meldung und Exit 2. Ohne pycbc zeigen die Demo-Apps sichtbar 'Newton-Näherung' an (bis D1). Keine eigene Physik, Injektion, Filterkette oder Modellsuche und kein sys.exit in GUI-Code.

- **Darf importieren:** gwd.core, gwd.data (scenarios, injection, preprocessing, gwosc), gwd.registry, gwd.config, matplotlib, scipy.io.wavfile
- **Darf nicht importieren:** gwd.training, gwd.evaluation, gwd.cli, tensorflow und gwpy auf Modulebene, sys.path-Manipulation
- **Externe Abhängigkeiten:** matplotlib plus interaktives Backend (python3-tk bzw. WSLg, als Systempaket dokumentiert), Glitch Hunter optional: [ml] für 'KI fragen', [gwdata] für 'LIGO Zufall', Audio über aplay/paplay/afplay/winsound
- **Kommt heute aus:** `src/gw_simulator.py`, `src/interferometer_simulator.py`, `src/glitch_hunter_app.py`, `apps/glitch_hunter.py, apps/gw_waveform_sim.py, apps/interferometer_sim.py (Duplikate, entfallen)`

#### CLI, Diagnose und Menü

**Ort:** `src/gwd/cli.py, src/gwd/env.py, main.py`

Ein Einstiegspunkt gwd mit den Subcommands data (fetch-noise, generate, validate), train, eval (run, thresholds, compare, leaderboard, build-testset, ggf. baseline), app (waveform, interferometer, glitch-hunter) und env. Die Handler importieren ihr Modul erst beim Aufruf, geben ehrliche Exit-Codes zurück und nutzen kein input(). main.py ist ein dünnes Menü nur mit Standardbibliothek: eine feste Liste von gwd-Argumentlisten, aufgerufen per subprocess [python -m gwd.cli …]. tests/test_menu.py prüft jede Liste gegen gwd.cli.build_parser(). Keine Fachlogik.

- **Darf importieren:** gwd.cli/gwd.env: alle gwd-Unterpakete, nur lazy im Handler, argparse, subprocess, main.py: nur Standardbibliothek
- **Darf nicht importieren:** tensorflow/gwpy/pycbc auf Modulebene, Fachlogik-Kopien, os.system mit relativen Skriptpfaden, main.py: kein Import von gwd
- **Externe Abhängigkeiten:** –
- **Kommt heute aus:** `main.py`, `utility/test_imports.py`

#### Tests und Werkzeuge

**Ort:** `tests/ (core/, data/, training/, evaluation/, apps/, smoke/, golden/, test_architecture.py, test_menu.py, test_cli.py, test_docs_commands.py, test_e2e.py)`

Smoke-Tests für Pipeline und Apps. Golden-Kennwerte: exakt für die Pfade, die nur von numpy abhängen, mit Toleranz für PyCBC, ohne Versionsschlüssel; mit GWD_REQUIRE_PYCBC=1 wird ein Skip zum Fehler. Dazu Unit-Tests je Komponente, der Architektur-Wächter (Importregeln, kein sys.path, keine Byte-Duplikate, schrumpfende ALLOWLIST), der Menü-/CLI-Abgleich, der Doku-Befehlstest und ein Ende-zu-Ende-Test. Marker slow/apps/golden/network, feste Seeds, immer GWD_HOME=tmp.

- **Darf importieren:** gwd.*, pytest
- **Darf nicht importieren:** src.*, sys.path-Manipulation (ab P2.6), Netzzugriffe ohne Marker network, echte Besitzerordner
- **Externe Abhängigkeiten:** pytest, ruff
- **Kommt heute aus:** `tests/conftest.py (entfällt)`, `tests/test_waveforms.py`, `tests/test_noise.py`, `tests/test_interferometer.py`

#### Repo-Infrastruktur und Doku

**Ort:** `pyproject.toml, README.md, AGENTS.md/CLAUDE.md, docs/, .gitignore, .gitattributes, .git-blame-ignore-revs, .github/workflows/`

Installierbarkeit mit Extras, CI, verzeichnisbasierte Ignore-Regeln inklusive der Altpfade vor f724856, LF-Zeilenenden, Regeln für KI-Assistenten, eine Einstiegsdoku plus zwei Fachdokus. Die Doku-Befehle sichert ein Test ab. Nur belegte Aussagen.

- **Darf importieren:** –
- **Darf nicht importieren:** Tool-Zustand im Index (.swarm/, .claude-flow/, CLAUDE.local.md), Artefakte im Index (data/, models_registry/, plots/, results/, Altpfade – außer nach D9), KI-Liefernotizen, unbelegte Zahlen, nicht existierende Skripte (evaluate_classic_fixed.py)
- **Externe Abhängigkeiten:** GitHub Actions, ruff
- **Kommt heute aus:** `requirements.txt`, `README.md`, `PROJECT_OVERVIEW.md`, `docs/FINAL_SUMMARY.md`, `docs/INSTALLATION_CHECKLIST.md`, `docs/ROC_ANALYSIS_README.md`, `docs/ROC_INTEGRATION_OVERVIEW.md`, `.planning/walkthrough.md`, `.swarm/*`, `.claude-flow/*`, `.gitignore`, `utility/fix_plots.py`

---

## 4. Fahrplan im Überblick

Ablauf in acht Phasen; nach jedem PR läuft das Projekt:

- P0 absichern: Backup inklusive der Altpfade vor f724856, Fakten F1–F4 zur Umgebung, Smoke-Tests in einem tmp-GWD_HOME, schlanke Golden-Kennwerte, Moratorium für den ungültigen Physik_Gap.
- P1 entrümpeln.
- P2 per git mv in das Paket umziehen (verhaltensgleich).
- P3 jede Logik genau einmal; der Train/Eval/App-Skew wird als benannte V1-Szenarien sichtbar.
- P4 stille Schalter sichtbar machen und Bugs beheben.
- P5 ehrlich messen: explizites Backend, rng, korrekte Rauschnormierung, peak_snr und optimal_snr, gruppierter 3-Wege-Split, feste Testsets, Schwelle pro Modell, Leaderboard v2.
- P6 fachliche Korrektur als additive Szenario-Version V2 (erst injizieren, dann gemeinsam konditionieren); V1 bleibt Default, bis der Besitzer neu trainiert.
- P7 Doku und Abschluss.

| Phase | Titel | Schritte | ohne Entscheidung | mit Default | wartet auf Antwort |
|---|---|---|---|---|---|
| **P0** | Sicherheitsnetz, Fakten und Sofortkorrekturen | 6 | 6 | – | – |
| **P1** | Ballast entfernen und Zielbild festschreiben | 6 | 5 | P1.1 | – |
| **P2** | Ein installierbares Paket gwd: Umzug per git mv, Grenzen per Test | 6 | 5 | P2.1 | – |
| **P3** | Eine Quelle pro Logik (verhaltensgleich, Golden-gesichert) | 7 | 5 | P3.6, P3.7 | – |
| **P4** | Robustheit und sichtbare Fehler (Szenarien unverändert) | 6 | 3 | P4.3, P4.4, P4.5 | – |
| **P5** | Physik-Kern und Messprotokoll ehrlich machen (entscheidungsarm, angekündigte Verhaltensänderungen) | 8 | 4 | P5.2, P5.4, P5.7, P5.8 | – |
| **P6** | Fachliche Korrekturen als additive Szenario-Version V2 | 5 | 0 | P6.2 | P6.1, P6.3, P6.4, P6.5 |
| **P7** | Doku und Abschluss | 4 | 1 | P7.1 | – (P7.2 und P7.3 seit D12/D7 frei; LICENSE schon in PR #1) |

Entscheidungen: Es gibt 12 Fragen. Sie werden direkt nach P0.1 gestellt, zusammen mit den dort per Befehl erhobenen Fakten: F1 PyCBC aktiv?, F2 Rauschen der Altmodelle, F3 Windows nativ?, F4 lokale Ergebnisdateien.

- 29 von 48 Schritten sind entscheidungsfrei.
- 13 laufen mit einem dokumentierten Default. Im Text steht dann „Nutzt Default von Dx“. Fällt die Antwort anders aus, braucht es einen kleinen Folge-PR, den die Entscheidung benennt.
- 6 Schritte warten laut blocked_by_decision auf eine Antwort. Ohne Antwort gilt der Default, der dort meist „Schritt bleibt geparkt“ heißt.

Defaults sind additiv, erhalten den Status quo oder entfernen nachweislich Ungültiges (D5: Physik_Gap).

**Minimalpfad für reine Ordnung: P0–P3 plus P7.1, in sich geschlossen. P4 bringt Robustheit, P5 ehrliches Messen, P6 die fachliche Korrektur.**

**Grundregeln für jeden Schritt**

- Ein Schritt ist ein PR, der sofort nach `main` gemergt wird. Danach läuft das Projekt, und die Tests sind grün.
- Nur verschieben, was der Schritt nennt, und nur per `git mv`. Nichts in Datenordnern löschen.
- Verhaltensänderungen gibt es nur in angekündigten Schritten. Ändern sich Golden-Werte, geschieht das in einem eigenen `golden:`-Commit mit Begründung.
- Arbeitsauftrag an einen KI-Assistenten: „Führe Schritt Px.y aus dem Aufräumplan aus. Ändere nur die genannten Dateien. Führe die Prüfung aus und zeige das Ergebnis.“

---

## 5. Entscheidungen für den Besitzer

### Stand der Antworten (04.11.2026)

**Entscheidungen**

| | Antwort | Wirkung |
|---|---|---|
| **D1** | faktisch **A** (siehe F1/F3): überall Linux bzw. WSL, pycbc vorhanden | P6.1 kann als Variante A laufen, sobald der Besitzer es bestätigt |
| **D7** | **B**: Python ≥ 3.11, CI auf 3.12, plus Lockfile | P7.3 ist nicht mehr blockiert und kann direkt nach P2.2 laufen; Grundlage ist die PC-Umgebung (siehe Umgebung unten) |
| **D8** | **A**: ein Paket `gwd` | wie geplant |
| **D11** | **B: alles Englisch**: Bezeichner, Spalten, Kommentare, Konsolenausgaben, Doku (README, `docs/architecture.md`, `docs/evaluation.md`, AGENTS.md), Commit-Messages und PR-Texte | Neue Dateien sind englisch. **Übersetzungsregel:** Bis P3.6 werden nur die Zeilen übersetzt, die ein Schritt ohnehin ändert, damit verhaltensgleiche Diffs lesbar bleiben. Ab P3.6 wird ein Modul, das ein Schritt umschreibt, in einem eigenen `docs:`-Commit im selben PR vollständig übersetzt. Den Rest erledigt P7.1. Dieser Arbeitsordner bleibt deutsch und entfällt in P7.4. |
| **D12** | **GPL-3.0-or-later**, Copyright Peter Paprotny | LICENSE und README-Abschnitt sind umgesetzt (PR #1). Offen bleibt nur `license = "GPL-3.0-or-later"` in pyproject.toml (P2.1). |

**Fakten**

| | Befund | Folge |
|---|---|---|
| **F1** | PC: pycbc 2.10.0, TF 2.20 sieht die RTX 3060 | Die vorhandenen Modelle sind mit PyCBC-Wellenformen entstanden, nicht mit dem fehlerhaften Newton-Pfad |
| **F2** | PC: `data/noise_background/` mit 1024 Chunks (133 MB), `data/training/` 1,3 GB. Laptop: 512 Chunks im **Altpfad** `gw_noise_background/` | Seit P0.1 liegen die 1024 PC-Chunks auf dem Laptop in `data/noise_background/`. Die Pipeline nutzt damit wieder echtes Rauschen. `gw_noise_background/` bleibt unangetastet. |
| **F3** | Rechner-Verlauf: Linux-Laptop → Windows-PC `Tesseract` mit RTX 3060 (WSL Ubuntu 24.04) → jetzt `tuxedo-laptop` (Ryzen AI 9 HX 370, Fedora, keine GPU für TF) | Gearbeitet wird auf dem Laptop (CPU-Training). Große Trainingsläufe (P6.5) laufen auf dem PC unter WSL. |
| **F4** | Laptop nach P0.1: 9 Modelle in `models_registry/` (8 vom PC, 24.11.–05.12.2025, und 1 vom 21.11.2025), `analysis/model_leaderboard.csv` (PC), `plots/` (PC), zwei Baselines (`physics_baseline.json` vom Laptop, `physics_baseline.pc-tesseract.json` vom PC). Altpfade unverändert: `gw_noise_background/`, `gw_training_data/`, `./model_leaderboard.csv`. Ungetrackt: `examples/`, `docs/Projekt Glitch Hunter.pdf`, `docs/Projektbericht.pdf` | Bis P0.4 keine Evaluation laufen lassen, sonst entsteht aus der Baseline ein ungültiger `Physik_Gap`. Die PDFs und `examples/` fasst kein Schritt an. |
| **Git** | Der lokale Laptop-Commit `7788a90` enthielt nur claude-flow-Logs; er wurde aufgelöst und liegt im Branch `sicherung/stuff-7788a90`. Die 11 „geänderten“ Dateien auf dem PC hatten nur andere Zeilenenden und wurden verworfen. | nichts verloren |

**Umgebung (ab P0.1)**

- Laptop-`.venv` mit Python 3.12.14, aufgebaut aus `~/gwd-env-pc.txt` (`pip freeze` der PC-Umgebung, ohne die `nvidia-*`-Pakete) plus pytest 8.3.4. Wichtige Versionen: tensorflow 2.20.0, keras 3.12.0, **numpy 1.26.4**, pandas 2.3.3, pycbc 2.10.0, gwpy 3.0.13. Alle 9 Modelle laden.
- `requirements.txt` beschreibt diese Umgebung **nicht** (pycbc 2.4.0, numpy 2.2.6). Das Lockfile in P7.3 entsteht aus `~/gwd-env-pc.txt`, bis dahin ist diese Datei die einzige Quelle der Pins.
- `.venv-broken-py313` ist das alte, defekte venv, aufbewahrt und ignoriert.
- Git-Commits laufen über die noreply-Adresse `23141484+Pappet@users.noreply.github.com`, weil GitHub Pushes mit privater Adresse ablehnt.

**Fortschritt**

| Schritt | PR | Stand |
|---|---|---|
| Plan, Register, Lizenz | #1 | gemergt |
| P0.1 Backup, Import, Tag `pre-cleanup` (→ `2eec4b5`), AGENTS.md, CLAUDE.md | #2 | gemergt |
| P0.2 GWD_HOME-Haken | #3 | Review von einem zweiten Agenten: freigegeben, drei unkritische Hinweise (siehe P2.3) |

Für P0 und P1 braucht es keine Antwort. Ideal ist eine Antwort auf **D7 und D8 vor P2.1**. Bleibt eine Frage offen, gilt der genannte Default: Er ist additiv, erhält den Status quo oder entfernt nachweislich Ungültiges.

**Fakten, die vorher erhoben werden (P0.1, in der WSL-Umgebung):**

- **F1:** Ist pycbc installiert? Wenn nicht, sind alle bisherigen Datensätze und Modelle mit dem fehlerhaften Newton-Pfad entstanden.
- **F2:** Mit welchem Rauschen wurden die vorhandenen Modelle trainiert (`data/noise_background` bzw. der Altordner `gw_noise_background/`)?
- **F3:** Laufen Apps oder Pipeline auch nativ unter Windows?
- **F4:** Welche lokalen Ergebnisdateien gibt es (Leaderboards, `physics_baseline.json`, `optimal_thresholds.json`)?

| ID | Frage (kurz) | Empfehlung | Default ohne Antwort | blockiert |
|---|---|---|---|---|
| **D1** | PyCBC Pflicht oder korrigiertes Newton-Backend? | A: PyCBC Pflicht, Newton löschen (falls F3 = nein) | Pipeline ohne stillen Fallback; Demo-Apps zeigen Newton sichtbar an; P6.1 geparkt | P6.1 |
| **D2** | SNR-Definition für V2 (Peak vs. ρ_opt) | B: ρ_opt steuert V2 | B (nur V2, additiv; V1 bleibt) | P6.5 |
| **D3** | Rauschen für V2 (echt / simuliert / beides) | C: erst simuliertes aLIGO-Rauschen, dann zusätzlich Echtrauschen | B: simuliertes Rauschen; Echtrauschen (P6.3) geparkt | P6.3, P6.5 |
| **D4** | Glitch Hunter: Spiel, Werkzeug oder beides? | A: Modi „Training“ und „Challenge“; indist- und stress-Set | A | – |
| **D5** | Matched-Filter-Baseline neu schreiben oder streichen? | Jetzt B (streichen), A später als Folgeprojekt | B: Physik_Gap entfällt | P6.4 |
| **D6** | Interferometer: Lernmodell oder kalibriert? | A: ehrlich gekennzeichnetes Lernmodell | A | – |
| **D7** | Python-Version, Lockfile, Mixed Precision | B: Python ≥ 3.11, CI auf 3.12, plus Lockfile | A: ohne Lockfile; P7.3 geparkt | P7.3 |
| **D8** | Ein Paket `gwd` oder drei Pakete; Menü behalten? | A: ein Paket `gwd`, ein Befehl `gwd`, `main.py` als Menü | A | – |
| **D9** | Ergebnisdateien versionieren? | B: `leaderboard_v2.csv` und Testset-Manifeste versionieren | A: `results/` bleibt lokal | – |
| **D10** | Schwelle, Ranking, Feldtest | A: Schwelle je Modell bei FPR 1 %, Ranking nach ROC_AUC, Feldtest optional | A | – |
| **D11** | Sprache (DE/EN) und Push-Regel | A: Bezeichner Englisch, Doku/Ausgaben Deutsch; „CI grün“ statt Push-Verbot | A | – |
| **D12** | Lizenz | A (MIT), falls das Repo öffentlich ist; sonst C | keine LICENSE; P7.2 geparkt | P7.2 |

### D1: PyCBC Pflicht oder korrigiertes Newton-Backend?

Wellenform-Backend und Plattform. Soll PyCBC Pflicht werden? Dann läuft die Pipeline nur unter Linux, macOS oder WSL, weil PyCBC und lalsuite keine Windows-Wheels haben. Oder bleibt ein korrigiertes Newton-Backend als explizit wählbare Lern-Option, z. B. für die Demo-Apps unter Windows nativ? Der heutige Newton-Pfad ist physikalisch falsch (Frequenz ×2,045, Amplitude ×4, kein ISCO). Grundlage: F1 (lief bisher überhaupt PyCBC?) und F3 (wird nativ unter Windows gearbeitet?).

**Optionen:**

- A: PyCBC Pflicht, Newton löschen (in der Historie erhalten); Windows nur über WSL
- B: PyCBC für Datensatz und Evaluation; Newton korrigiert (CORE-2/3/8), nur explizit wählbar, in Szenarien verboten
- C: Optionaler Fallback wie heute, nur laut gewarnt (nicht empfohlen: gleiche Labels, andere Physik, Korrelation −0,004)

**Empfehlung:** A, sofern F3 'nein' ergibt. Die Arbeitsumgebung ist WSL (b113e2b), TF mit GPU braucht unter Windows ohnehin WSL2, und die ML-Pipeline braucht PyCBC. B nur, wenn die Demo-Apps ausdrücklich nativ unter Windows laufen sollen. In jedem Fall gibt es in der Pipeline keinen stillen Fallback mehr (P5.2).

**Default ohne Antwort:** Die Pipeline hat keinen stillen Fallback mehr und bricht ohne pycbc mit Hinweis ab (P5.2). Die Demo-Apps nutzen ohne pycbc weiter Newton, aber sichtbar mit Hinweis im Fenster; das ist der Status quo und wirkt in P5.2. Der Newton-Code bleibt unverändert, P6.1 bleibt geparkt.

**Blockiert:** P6.1

### D2: SNR-Definition für V2 (Peak vs. ρ_opt)

Welche SNR-Definition steuert die neue Szenario-Version V2 (Datensatz, Evaluation, Glitch Hunter)? V1 bleibt in jedem Fall unverändert: Peak-Amplitude/σ ist massen- und rauschartabhängig; Training 1–12 entspricht ρ_opt ≈ 10–200, Eval 0,5–2,5 etwa 5–43. P5.5 misst und speichert ohnehin beide Größen.

**Optionen:**

- A: Peak-SNR (peak_snr) auch in V2; ρ_opt wird nur berichtet
- B: ρ_opt (PSD-gewichtet, gegen die PSD der Konditionierung) steuert V2; Bereiche z. B. Training ρ 5–30, Eval-Bins über denselben Bereich plus ein Extrapolations-Bin

**Empfehlung:** B. Erst damit sind Training, Evaluation, Glitch Hunter und eine Matched-Filter-Baseline vergleichbar, und die Labels sind rauschartinvariant (evaluation:MISS-2). Nach P5.5 ist das ein Feld in data/scenarios.py.

**Default ohne Antwort:** B für V2. Das ist additiv, V1 bleibt unverändert; ein Feld in data/scenarios.py, wirkt in P6.2. Das Neutraining (P6.5) beginnt erst nach Rückfrage.

**Blockiert:** P6.5

### D3: Rauschen für V2 (echt / simuliert / beides)

Welches Rauschen sieht V2? Heute addiert das Training einen UNGEFILTERTEN Chirp zu bandpassgefiltertem H1-Rauschen (mögliche Abkürzung über 300 Hz), die Evaluation nutzt 100 % weißes Rauschen, und dieselben 1024 Chunks aus einem 68-min-Segment werden ~8× quer über Train und Test verwendet. In allen Optionen gilt für V2: erst injizieren, dann gemeinsam konditionieren. Grundlage: F2 (wurden die Altmodelle überhaupt mit Echtrauschen trainiert?).

**Optionen:**

- A: Echtes H1-Rauschen aus mehreren Rohsegmenten (Neu-Download), disjunkte Pools je Split
- B: Simuliertes farbiges Rauschen nach aLIGO-PSD mit derselben Konditionierung (offline, reproduzierbar, CI-tauglich); echte Daten nur im Feldtest und im Glitch Hunter
- C: B und A als zwei Domänen, getrennt berichtet (B zuerst, A sobald der GWOSC-Download klappt)
- D: Keine V2; Status quo V1, nur protokolliert und dokumentiert

**Empfehlung:** C. B ist der reproduzierbare, offline- und CI-taugliche Start. A ergänzt die Domäne, die der Glitch Hunter mit echten Daten ohnehin zeigt.

**Default ohne Antwort:** B: P6.2 baut V2 mit simuliertem farbigem Rauschen. P6.3 (Echtrauschen-Pools) bleibt geparkt, P6.5 beginnt erst nach Rückfrage.

**Blockiert:** P6.3, P6.5

### D4: Glitch Hunter: Spiel, Werkzeug oder beides?

Wozu dient der Glitch Hunter, und welche Testsets gibt es? Ist er ein Spiel (bewusst schwer, Peak-SNR 0,5–2,0) oder ein Werkzeug, um das Modell unter Trainingsbedingungen zu prüfen? Bleiben OOD-Events (GW170817, GW190814) im Feldtest? Die Frage ist bewusst auf Modi und Testsets beschränkt; die Trainingsverteilung entscheiden D2 und D3.

**Optionen:**

- A: Additiv: Glitch Hunter mit den Modi 'Training' (Start) und 'Challenge' (heutige Verteilung, als Extrapolation gekennzeichnet); Evaluation auf indist- und stress-Set getrennt; OOD-Events getrennt ausgewiesen
- B: Nur Modus 'Training' und nur das indist-Set; OOD-Events entfernen
- C: Status quo (nur die heutige schwere Verteilung), Skew nur dokumentiert

**Empfehlung:** A. Beide Bedürfnisse bleiben bedient und sind ehrlich gekennzeichnet; nichts wird entfernt.

**Default ohne Antwort:** A, wirkt in P4.4 und P5.7. Der Start-Modus lässt sich mit einer Zeile ändern.

**Blockiert:** nichts (nur Default-Nutzung)

### D5: Matched-Filter-Baseline neu schreiben oder streichen?

Matched-Filter-Physik-Baseline: neu schreiben oder das Feature samt Physik_Gap und README-Versprechen streichen? Die alte evaluate_classic.py (gelöscht in 45b9202) hatte einen Template-Bug (match 0,07–0,14), ein anderes Rauschmodell und eine andere SNR-Skala und darf deshalb nicht einfach wiederhergestellt werden.

**Optionen:**

- A: Neu schreiben als evaluation/baseline_mf.py auf denselben festen Testsets wie das CNN, Oracle-Template als Obergrenze, Vergleich per ROC bzw. TPR bei gleicher FPR
- B: Feature entfernen (BASELINE_FILE, Physik_Gap, Doku-Versprechen); die Historie bleibt als Referenz
- C: Alte Datei wiederherstellen (nicht empfohlen: erzeugt sofort eine falsche Baseline)

**Empfehlung:** Jetzt B, das setzen P0.4 und P5.8 ohnehin um. A als Folgeprojekt nach P6.2, weil der Vergleich mit der Physik-Grenze erst auf identischen V2-Samples aussagekräftig ist.

**Default ohne Antwort:** B. Damit wird ein nachweislich ungültiges Feature entfernt, statt den Status quo zu halten. P6.4 entfällt bis auf einen Satz in docs/evaluation.md.

**Blockiert:** P6.4

### D6: Interferometer: Lernmodell oder kalibriert?

Bleibt das Interferometer-Modell ein didaktisches Spielzeug? Heute ist es bei 100 Hz etwa 270-mal schlechter als aLIGO, und die Armlänge hat kaum Einfluss. Oder soll es ein an aLIGO kalibriertes Rauschmodell werden, das auch Simulations- und Trainingsrauschen speisen kann?

**Optionen:**

- A: Lernmodell, ehrlich gekennzeichnet; qualitative Tests, Finesse als Property, echte aLIGO-Referenzkurve über pycbc.psd, korrekt normiertes farbiges Rauschen
- B: Physikalisch kalibriertes PSD-Modell (Verschiebungsrauschen/L, korrekte f-Potenzen), optional als Trainingsrauschen

**Empfehlung:** A. Das passt zum Lern- und Demozweck der Apps; B wäre eine Neuentwicklung der Physik und kein Aufräumen.

**Default ohne Antwort:** A, wirkt in P4.5 und P5.4. B steht unter 'bewusst weggelassen'.

**Blockiert:** nichts (nur Default-Nutzung)

### D7: Python-Version, Lockfile, Mixed Precision

Zielumgebung: Welche Python-Version gilt? WSL nutzt 3.12, die Analyseumgebung 3.13; requirements.txt ist auf keiner Version vollständig per Wheel installierbar. Soll es ein Lockfile geben? Mixed Precision immer oder nur mit GPU?

**Optionen:**

- A: requires-python >=3.11, CI auf 3.12, nur Versionsbereiche in pyproject; Mixed Precision automatisch bei GPU
- B: wie A, zusätzlich ein Lockfile (uv pip compile) für Linux/3.12, aus dem die CI installiert
- C: Python 3.13 als Ziel (pycbc ≥ 2.10)

**Empfehlung:** B. Mit uv kostet das Lockfile einen Befehl. Es ersetzt das falsche Reproduzierbarkeitsversprechen durch ein echtes und nimmt den Golden-Toleranzen die Versionsdrift ab.

**Default ohne Antwort:** A, wirkt in P2.1 und P3.7; P7.3 bleibt geparkt.

**Blockiert:** P7.3

### D8: Ein Paket `gwd` oder drei Pakete; Menü behalten?

Paketaufteilung und Einstieg: ein Paket gwd mit den Unterpaketen core, data, training, evaluation, apps plus config, registry und cli? Oder drei Top-Level-Pakete gwd_core, gwd_pipeline und gwd_apps (weniger Umbenennungen)? Bleibt main.py als Menü? Die Antwort sollte möglichst vor P2.1 vorliegen.

**Optionen:**

- A: Ein Paket gwd mit klar benannten Komponenten; ein Befehl gwd; main.py bleibt als dünnes Menü über die CLI
- B: Drei Top-Level-Pakete gwd_core, gwd_pipeline, gwd_apps; eigene Console-Scripts je App
- C: Apps in ein eigenes Repo bzw. Paket

**Empfehlung:** A. Ein eindeutiger Namensraum (HYG-M5), feinere Komponentengrenzen als ein Sammelpaket gwd_pipeline und eine einzige Installation. Die Umbenennung gwd_core → gwd.core fällt in denselben PR wie die ohnehin nötige Test-Umstellung.

**Default ohne Antwort:** A, wirkt in P2.1 und P3.6. Ein späterer Wechsel auf 'nur CLI' heißt nur, main.py zu löschen.

**Blockiert:** nichts (nur Default-Nutzung)

### D9: Ergebnisdateien versionieren?

Sollen kleine Ergebnisdateien versioniert werden (Leaderboard v2, Testset-Manifeste)? Heute versteckt die globale *.csv/*.json-Regel sie.

**Optionen:**

- A: results/ bleibt komplett lokal (ignoriert)
- B: results/leaderboard_v2.csv und results/testsets/ werden versioniert (.gitignore: results/* plus !results/leaderboard_v2.csv und !results/testsets/); Arrays, Plots und das alte Leaderboard bleiben ignoriert
- C: Zusätzlich Plots versionieren

**Empfehlung:** B. Der Fortschritt wird in Git sichtbar, und es geht nur um kleine Textdateien.

**Default ohne Antwort:** A, wirkt in P1.1 (.gitignore) und P5.7. Die Manifeste liegen deshalb von Anfang an unter results/testsets/ statt unter data/, weil sich Dateien unter der Regel data/ nicht wieder einschließen lassen. Ein späterer Wechsel sind drei .gitignore-Zeilen.

**Blockiert:** nichts (nur Default-Nutzung)

### D10: Schwelle, Ranking, Feldtest

Arbeitsschwelle, Ranking und Feldtest: Woher kommt die Schwelle? Heute ist sie in der Evaluation 0.15, im Glitch Hunter 0.46, früher 0.75, laut Doku 0.7234, und das Ergebnis von optimize_threshold liest niemand. Wonach wird gerankt? Ist der Echtdaten-Feldtest (Netz, GWOSC) Pflicht?

**Optionen:**

- A: Schwelle pro Modell bei fester FPR (1 %) auf einem reinen Rausch-Kalibrierset, gespeichert in der Modell-JSON; Ranking nach ROC_AUC bzw. TPR@FPR; Feldtest nur mit --real, OOD-Events getrennt
- B: Schwelle pro Modell nach Youden oder F1
- C: Eine globale Schwelle in config

**Empfehlung:** A. Eine FPR-basierte Schwelle macht SNR50/90 und TPR zwischen Modellen mit unterschiedlicher Score-Kalibrierung vergleichbar (evaluation:MISS-1), und die Evaluation läuft offline.

**Default ohne Antwort:** A mit FPR = 1 %, ohne gespeicherte Schwelle Fallback 0,5 mit Warnung, Feldtest per Default aus (--no-real). Wirkt in P4.3, P4.4 und P5.8.

**Blockiert:** nichts (nur Default-Nutzung)

### D11: Sprache (DE/EN) und Push-Regel

Sprache und README-Regeln: Deutsch oder Englisch für Doku, Kommentare, Konsolenausgaben und Spaltennamen? Heute sind Kommentare überwiegend Deutsch, Bezeichner und Tests Englisch, das README Englisch und docs/ Deutsch. Bleibt im Solo-Projekt die Regel 'nie direkt auf main pushen'?

**Optionen:**

- A: Bezeichner und CSV-/Leaderboard-Spalten Englisch; Doku, Kommentare und Ausgaben Deutsch; Push-Regel → 'CI muss grün sein'
- B: Alles Englisch (öffentliches GitHub-Repo)
- C: Alles Deutsch

**Empfehlung:** A, schrittweise beim Anfassen von Dateien, kein Big-Bang.

**Default ohne Antwort:** A für neue Doku und neue Spalten, wirkt in P5.8 und P7.1.

**Blockiert:** nichts (nur Default-Nutzung)

### D12: Lizenz

Lizenz: Ist die MIT-Lizenz aus README.md:54 gewollt, und mit welchem Copyright-Namen und Jahr? Das ist eine rechtliche Entscheidung und deshalb bewusst von D11 getrennt.

**Optionen:**

- A: MIT mit Name und Jahr (LICENSE, README, pyproject)
- B: Eine andere Lizenz
- C: Keine Lizenz; die README-Zeile wird zu 'Alle Rechte vorbehalten'

**Empfehlung:** A, falls das Repo öffentlich ist oder geteilt werden soll; sonst C, damit README und Realität übereinstimmen.

**Default ohne Antwort:** Keine LICENSE-Datei, die README-Zeile bleibt unverändert, P7.2 bleibt geparkt.

**Blockiert:** P7.2

---

## 6. Schritte im Detail

Aufwand: **S** ≈ eine Stunde bis ein Abend, **M** ≈ ein bis zwei Abende, **L** ≈ ein Wochenende. Befund-IDs verweisen auf [BEFUNDE.md](BEFUNDE.md).

### P0: Sicherheitsnetz, Fakten und Sofortkorrekturen

**Ziel:** Bevor etwas bewegt wird:

- lokale Daten sichern, auch die Altpfade aus der Zeit vor f724856;
- Fakten zur Umgebung erheben, die D1 und D3 entscheiden;
- den heutigen Ablauf (generate, validate, train, evaluate, thresholds, compare, leaderboard) und alle drei Apps automatisiert in einem tmp-Verzeichnis ausführen;
- die numpy-stabilen Physikwerte als Golden-Kennwerte festhalten.

Sofort gestoppt wird die Erzeugung nachweislich ungültiger Zahlen (Physik_Gap). Verhalten ändern nur der GWD_HOME-Haken (Default identisch), der matplotlib-Fix und die Leaderboard-Korrekturen.

**Fertig, wenn:** `python -m pytest -m 'slow or apps or golden'` ist beim Besitzer grün und erzeugt nichts im Repo. Backup und der gepushte Tag pre-cleanup existieren. F1–F4 sind dokumentiert, die 12 Fragen gestellt. Physik_Gap wird nicht mehr aus einer ungültigen Baseline berechnet.

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P0.1](#p01) | Lokale Daten sichern, Fakten erheben, Rollback-Tag, Regeln für KI-Assistenten | S | – | – |
| [P0.2](#p02) | Testbarkeits-Haken: GWD_HOME in config.py | S | P0.1 | – |
| [P0.3](#p03) | validate_dataset lauffähig machen (tick_labels, kein globaler Warnfilter) | S | P0.2 | – |
| [P0.4](#p04) | Leaderboard-Sofortkorrekturen und Ergebnis-Moratorium | S | P0.2 | – |
| [P0.5](#p05) | pytest-Konfiguration, Pipeline- und App-Smoke (Charakterisierung) | M | P0.3, P0.4 | – |
| [P0.6](#p06) | Schlanke Golden-Kennwerte ohne Versionsschlüssel | M | P0.5 | – |

<a id="p01"></a>
#### P0.1: Lokale Daten sichern, Fakten erheben, Rollback-Tag, Regeln für KI-Assistenten

*Aufwand S · hängt ab von – · wartet auf –*

Beim Besitzer (WSL), vor jeder Änderung:

(1) Inventar: `git status --porcelain --ignored > ~/gwd-inventar-vorher.txt`.

(2) Backup außerhalb des Repos. Es umfasst auch die Altpfade aus der Zeit vor f724856, als die Skripte relativ zum Arbeitsverzeichnis schrieben: `tar czf ~/gwd-backup-$(date +%F).tgz --ignore-failed-read analysis/*.csv models_registry plots data/noise_background gw_noise_background model_leaderboard.csv optimal_thresholds.json .swarm .claude-flow CLAUDE.md .mcp.json .claude`. data/training und gw_training_data/ sind regenerierbar (~1,3 GB); davon nur labels.csv sichern.

(3) Fakten erheben, Ergebnis in ~/gwd-fakten.txt. Sie sind die Grundlage für D1 und D3 und kommen in P1.5 nach docs/architecture.md:

- F1 Umgebung: `python -V; pip freeze > ~/gwd-env-vorher.txt; python -c 'import pycbc; print(pycbc.__version__)'`. Fehlt pycbc, sind alle vorhandenen Datensätze und Modelle mit dem fehlerhaften Newton-Fallback entstanden.
- F2 Rauschen der Altmodelle: `ls data/noise_background | wc -l; ls gw_noise_background | wc -l; grep -h dataset_folder models_registry/*.json`.
- F3 Frage an den Besitzer: Laufen Apps oder Pipeline auch nativ unter Windows (winsound-Zweig, 'cls')?
- F4 Lokale Ergebnisdateien laut Inventar: Leaderboards in analysis/ und im Root, physics_baseline.json, optimal_thresholds.json.

(4) `git tag pre-cleanup && git push origin pre-cleanup`. Remote-Sessions und der Guard in P1.3 brauchen den Tag.

(5) AGENTS.md mit den Kernregeln:

- Dateien nur per `git mv` verschieben, nie kopieren (c4e3dd8).
- Nie `git clean -x` oder `rm -rf` auf analysis/, data/, models_registry/, plots/, results/ und die Altpfade.
- Ein Thema pro Commit mit Präfix chore/fix/refactor/test/docs/golden (c54acd5).
- Schritt-ID im PR nennen.
- Vor jedem Commit `pytest -m 'not slow and not network'`, vor dem PR alle Tests.
- Golden-Werte nur in 'golden:'-Commits mit Begründung.
- Kein Tool-Zustand im Commit, keine unbelegten Aussagen in der Doku.

CLAUDE.md: Existiert lokal bereits eine (ruflo init legt oft CLAUDE.md, .mcp.json und .claude/ an), wird sie nicht überschrieben. Oben kommt 'siehe AGENTS.md' hinzu, ruflo-spezifische Teile wandern nach CLAUDE.local.md (ab P1.1 ignoriert). Andernfalls eine neue CLAUDE.md mit genau diesem Verweis.

Danach die 12 Entscheidungsfragen zusammen mit F1–F4 stellen.

- **Dateien:** `AGENTS.md`, `CLAUDE.md`
- **Behebt:** HYG-23, HYG-1, HYG-5, evaluation:MISS-3, data-training:GITIGNORE-ART, hygiene:HYG-M1, core:MISS-1
- **Risiko:** Gering. Hauptrisiko wäre ein vergessenes Backup; deshalb hängen alle löschenden Schritte an P0.1.
- **Prüfung:** `tar tzf ~/gwd-backup-*.tgz | grep -c 'keras$'` entspricht `ls models_registry/*.keras | wc -l`; `git ls-remote --tags origin pre-cleanup` liefert eine Zeile; `grep -cE '^F[1-4]' ~/gwd-fakten.txt` → 4; `git show --stat HEAD` enthält nur AGENTS.md und CLAUDE.md.

<a id="p02"></a>
#### P0.2: Testbarkeits-Haken: GWD_HOME in config.py

*Aufwand S · hängt ab von P0.1 · wartet auf –*

Die einzige Produktionsänderung für die Testbarkeit: `ROOT_DIR = os.environ.get('GWD_HOME') or os.path.dirname(os.path.abspath(__file__))`. Ohne die Variable bleibt alles wie heute. Mit ihr laufen Tests in einem tmp-Verzeichnis: config legt beim Import nur dort Ordner an (DATA-7 für Tests entschärft), echte Modelle und Leaderboards bleiben unberührt. Hier setzt die Wurzelauflösung in P2.3 an.

- **Dateien:** `config.py`
- **Behebt:** DATA-7, DATA-23
- **Risiko:** Sehr gering, der Default ist unverändert.
- **Prüfung:** `T=$(mktemp -d); GWD_HOME=$T python -c 'import os,config; assert config.MODELS_DIR.startswith(os.environ["GWD_HOME"])'`; `python -c 'import config; print(config.ROOT_DIR)'` zeigt den Repo-Pfad; `python -m pytest -q tests` → 6 passed; `git diff --stat HEAD~1` zeigt nur config.py.

<a id="p03"></a>
#### P0.3: validate_dataset lauffähig machen (tick_labels, kein globaler Warnfilter)

*Aufwand S · hängt ab von P0.2 · wartet auf –*

validate_dataset.py:183: `boxplot(..., labels=[...])` wird zu `tick_labels=`. Das braucht matplotlib ≥ 3.9; Pin 3.10.7 und die installierte 3.11 erfüllen das. Z. 23 `warnings.filterwarnings('ignore')` wird entfernt, weil er die Deprecation versteckt hat. Dieser Schritt muss VOR dem Smoke-Test liegen, sonst ist die Testbasis von Anfang an rot. Die veralteten SNR-Schwellen (DATA-4) bleiben bewusst bis P4.6 unverändert.

- **Dateien:** `src/dataset/validate_dataset.py`
- **Behebt:** DATA-3
- **Risiko:** Gering. Mehr Konsolenwarnungen sind beabsichtigt.
- **Prüfung:** `export GWD_HOME=$(mktemp -d); python -c 'import sys,random,numpy as np; sys.path.insert(0,"src/dataset"); random.seed(1); np.random.seed(1); import generate_chirp_dataset as g; g.generate_dataset(64, g.DATA_DIR_TRAIN)' && MPLBACKEND=Agg python src/dataset/validate_dataset.py 2>&1 | grep -c Traceback` → 0 (der Exit-Code kann wegen DATA-4 bei kleinen Stichproben 1 sein); `ls $GWD_HOME/plots/validation_dataset/*.png | wc -l` → 5; `grep -n filterwarnings src/dataset/validate_dataset.py` leer.

<a id="p04"></a>
#### P0.4: Leaderboard-Sofortkorrekturen und Ergebnis-Moratorium

*Aufwand S · hängt ab von P0.2 · wartet auf –*

(a) update_leaderboard_roc liest physics_baseline.json nicht mehr. Physik_Gap ist immer '-', dazu kommt eine einmalige Warnung 'Physik-Baseline derzeit ungültig (EVAL-6, siehe D5)'. Eine lokal noch vorhandene Datei aus der Zeit vor 45b9202 würde sonst ungeprüft einen falschen Gap eintragen. Der Parameter baseline_file bleibt vorerst, damit sich die Aufrufer nicht ändern; er entfällt in P5.8.

(b) Bugfixes im selben Modul:

- update_leaderboard_basic löschen (toter Code);
- das wörtlich ausgegebene Backslash-n durch einen echten Zeilenumbruch ersetzen;
- `if sim_metrics.get('snr50')` durch eine is-not-None-Prüfung ersetzen, weil 0.0 gültig ist.

(c) show_leaderboard.py:

- main-Guard;
- ohne Datei eine verständliche Meldung und Exit 0;
- Sortierung nach ROC_AUC absteigend statt nach Sim_SNR50, das immer feuernde Modelle belohnt (evaluation:MISS-1).

(d) README:

- Statushinweis 'Simulationsmetriken derzeit nur eingeschränkt aussagekräftig (Train/Eval-Skew, Peak-SNR, keine Seeds) – Aufräumplan in docs/aufraeumen/PLAN.md' (docs/architecture.md entsteht erst in P1.5; P7.1 ersetzt den Verweis);
- Matched-Filter als 'derzeit nicht vorhanden' markieren;
- 'exact version locks/reproducible builds' und `export PYTHONPATH=.` streichen;
- Bekanntes Problem bei der Installation vermerken: `pycbc==2.4.0` aus requirements.txt hat unter Python 3.12 keine nutzbaren Wheels, ein Lockfile ist geplant (P7.3); requirements.txt selbst bleibt unverändert;
- die Lizenzzeile bleibt (D12).

Das CSV-Schema bleibt (12 Spalten, formatierte Strings); Rohzahlen kommen erst in P5.8.

- **Dateien:** `src/utils/evaluation.py`, `src/show_leaderboard.py`, `README.md`, `tests/test_leaderboard.py`
- **Behebt:** EVAL-6, EVAL-7, EVAL-8, EVAL-12, EVAL-14, EVAL-22, EVAL-23, HYG-22, hygiene:HYG-M4, evaluation:MISS-1, HYG-4, HYG-6, HYG-10, hygiene:HYG-M2, HYG-15
- **Risiko:** Gering. Der Besitzer sieht vorübergehend keinen Physik_Gap mehr; dieser Wert war nie gültig. Weil sich dadurch src/show_leaderboard.py von analysis/show_leaderboard.py unterscheidet, prüft der Guard in P1.3 gegen den Tag pre-cleanup.
- **Prüfung:** `python -m pytest tests/test_leaderboard.py -q`: Eine Dummy-physics_baseline.json mit SNR90=1.0 in tmp_path ergibt Physik_Gap == '-'; snr50=0.0 ergibt '0.00'; die Konsolenausgabe enthält kein wörtliches Backslash-n. `GWD_HOME=$(mktemp -d) python src/show_leaderboard.py; echo $?` → Meldung und 0; `git grep -n update_leaderboard_basic -- ':!docs/aufraeumen'` leer (Plan und Befundliste nennen die Funktion weiterhin); `grep -niE 'exact version locks|reproducible builds|PYTHONPATH' README.md` leer.

<a id="p05"></a>
#### P0.5: pytest-Konfiguration, Pipeline- und App-Smoke (Charakterisierung)

*Aufwand M · hängt ab von P0.3, P0.4 · wartet auf –*

(a) pyproject.toml enthält vorerst NUR [tool.pytest.ini_options]: testpaths=['tests'], Marker slow/apps/golden/network, addopts='-ra'. Damit sammelt pytest utility/test_imports.py nicht mehr ein. tests/conftest.py bleibt bis P2.1.

(b) tests/smoke/entrypoints.py ist die einzige Tabelle 'Stufe → Aufruf'. Bis P3.6 stehen dort kleine Wegwerf-Treiber:

- Skriptmodul importieren, Seeds setzen (random, np.random, tf.keras.utils.set_random_seed);
- train_cnn.EPOCHS=1 patchen;
- evaluate_real_world durch einen Stub ohne Netz ersetzen ({'found':0,'total':0,'far_text':'n/a','event_scores':{}});
- main() aufrufen; optimize_threshold bekommt 'y' über stdin.

Spätere Umzüge ändern nur diese Tabelle.

(c) tests/smoke/test_pipeline.py (Marker slow). Session-Fixture: GWD_HOME=tmp_path, MPLBACKEND=Agg, CUDA_VISIBLE_DEVICES=''. Jede Stufe läuft als Subprozess: generate (n=64, fester Seed, so gewählt, dass der alte Validator nicht zufällig 'KRITISCH' meldet) → validate → train (1 Epoche) → evaluate → thresholds → compare → show_leaderboard.

Geprüft wird nur die Struktur:

- labels.csv mit 64 Zeilen, 32 Signalen und 7 Spalten;
- Sample-Shape (16384,) float64;
- genau ein .keras und ein .json;
- Leaderboard-CSV mit 1 Zeile und 12 Spalten;
- optimal_thresholds.json und ein Vergleichsplot;
- kein 'Traceback' in stderr.

REIHENFOLGE-FALLE: Die Fixture legt $GWD_HOME/analysis/ vorab an, weil LEADERBOARD_FILE dorthin zeigt und niemand den Ordner anlegt (evaluation:MISS-3). Diese Zeile entfällt bewusst in P1.2.

Ein zweiter Fall mit 4 künstlichen real_noise_*.npy in data/noise_background lässt den Echtrausch-Pfad einmal laufen.

(d) tests/smoke/test_apps.py (Marker apps):

- gw_simulator.py und interferometer_simulator.py als Subprozess mit MPLBACKEND=Agg → rc 0 ohne Traceback;
- glitch_hunter_app.py mit einem Dummy-Modell aus der Fixture (Input (16384,1) → GlobalAveragePooling1D → Dense(1, sigmoid)) in $GWD_HOME/models_registry → rc 0.

Ohne Modell liefert der Glitch Hunter heute rc 1 (APP-8, bewusst geändert in P4.4). Dass die Apps ohne interaktives Backend stumm mit 0 enden (APPX-2), behebt P4.5.

- **Dateien:** `pyproject.toml`, `tests/smoke/__init__.py`, `tests/smoke/conftest.py`, `tests/smoke/entrypoints.py`, `tests/smoke/test_pipeline.py`, `tests/smoke/test_apps.py`
- **Behebt:** DATA-23, EVAL-20, HYG-16, evaluation:MISS-3, APP-18, APP-1, APP-8, apps:APPX-2
- **Risiko:** Laufzeit 2–4 min auf CPU. TF ist nicht deterministisch, deshalb gibt es nur Strukturprüfungen. Die Treiber sind Wegwerf-Code bis P3.6. Interaktive Backends werden nicht geprüft.
- **Prüfung:** `python -m pytest -m 'slow or apps' tests/smoke -q --durations=5` grün; danach ist `git status --porcelain` leer; `python -m pytest --collect-only -q | grep -c utility` → 0. Gegenprobe: den App-Eintrag in entrypoints.py testweise auf apps/gw_waveform_sim.py umbiegen → rot (ModuleNotFoundError gwd_core).

<a id="p06"></a>
#### P0.6: Schlanke Golden-Kennwerte ohne Versionsschlüssel

*Aufwand M · hängt ab von P0.5 · wartet auf –*

Neue Dateien: tests/golden/values.py als Python-Literal (nicht .json/.npz, weil diese Endungen bis P1.1 global ignoriert werden), tests/golden/test_golden.py (Marker golden) und tests/golden/regen.py.

EXAKT (rtol 1e-9) werden nur Pfade geprüft, die allein von numpy abhängen und deren Zufallsstrom stabil ist (np.random.seed/RandomState):

(a) Newton-Pfad von generate_astrophysical_chirp (Monkeypatch HAS_PYCBC=False) mit festen Parametern: Summe der Quadrate, argmax, 5 Stützstellen, f_inst am Peak;

(b) generate_gaussian_noise und generate_colored_noise unter Seed;

(c) scale_signal_to_snr und die Eval-Formel noise+sig*snr mit festen Eingaben;

(d) generate_dataset(n=8) mit erzwungenem Newton, ohne Echtrauschen, unter Seed: labels.csv-Werte und Kennwerte der ersten 2 Samples;

(e) InterferometerModel-Default-ASD bei 10/30/100/300/1000 Hz.

Der PyCBC-Pfad wird nur physikalisch mit Toleranz geprüft: Peak-Index ±1, max|h| rtol 1e-3, f am Peak rtol 1e-2.

Es gibt keinen Versionsschlüssel: Eine Abweichung macht den Test rot, nicht übersprungen. Übersprungen wird nur, wenn pycbc fehlt; mit GWD_REQUIRE_PYCBC=1 (CI, P2.1) wird daraus ein Fehler. Neu erzeugt wird nur in Commits mit Präfix 'golden:' und Begründung. Fehlt pycbc beim Besitzer (F1), werden die PyCBC-Kennwerte im ersten Commit von P2.1 erfasst.

Zweck: P1–P3 müssen alle Werte unverändert lassen.

- **Dateien:** `tests/golden/__init__.py`, `tests/golden/values.py`, `tests/golden/regen.py`, `tests/golden/test_golden.py`
- **Behebt:** CORE-13, CORE-4, CORE-6, CORE-10, core:MISS-3
- **Risiko:** Ein KI-Assistent könnte bei Rot reflexhaft neu erzeugen. Dagegen stehen die AGENTS.md-Regel und der lesbare Diff von values.py. Umgebungsdrift bei PyCBC fangen die Toleranzen ab; ein Versionsschlüssel mit stillem Skip entfällt bewusst.
- **Prüfung:** `python -m pytest -m golden -rs -q` grün, Skips höchstens mit Grund 'pycbc fehlt'. Mutationsprobe: die Newton-Amplitude in waveforms.py lokal ändern → rot; zurücksetzen → grün. Ohne pycbc schlägt `GWD_REQUIRE_PYCBC=1 python -m pytest -m golden -q` fehl, statt zu überspringen.

### P1: Ballast entfernen und Zielbild festschreiben

**Ziel:** Duplikate, Tool-Zustand, tote Skripte und irreführende KI-Notizen entfernen. Das Zielbild mit Importregeln, Verschiebetabelle, Fakten und Entscheidungen festschreiben; es ist der Anker für alle späteren KI-Arbeitsaufträge. Zeilenenden normalisieren. Das Code-Verhalten bleibt unverändert, die lokalen Daten des Besitzers (auch Altpfade) bleiben nachweislich unversehrt.

**Fertig, wenn:** Keine Byte-Duplikate, kein Tool-Zustand und keine KI-Selbstberichte mehr im Index. Das Leaderboard liegt unter results/, die lokale CSV ist kopiert und unversehrt. docs/architecture.md beschreibt Zielbild, Importregeln, Verschiebetabelle, Fakten und Entscheidungen. Nur noch LF. Alle P0-Tests sind grün, das Inventar aus P0.1 zeigt keinen Verlust.

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P1.1](#p11) | .gitignore neu fassen (inkl. Altpfade); Tool-Zustand und .planning aus dem Index | S | P0.1 | – |
| [P1.2](#p12) | Leaderboard und Schwellen-JSON nach results/ verlegen – vor dem Löschen von analysis/ | S | P0.5, P1.1 | – |
| [P1.3](#p13) | Byte-Duplikate analysis/*.py und apps/*.py löschen (Guard gegen den Ausgangsstand) | S | P1.2 | – |
| [P1.4](#p14) | Tote Skripte und irreführende KI-Liefernotizen entfernen | S | P0.4 | – |
| [P1.5](#p15) | Zielbild festschreiben: docs/architecture.md (aus PROJECT_OVERVIEW.md) | M | P1.3 | – |
| [P1.6](#p16) | Zeilenenden normalisieren (reiner Format-Commit) | S | P1.1 | – |

<a id="p11"></a>
#### P1.1: .gitignore neu fassen (inkl. Altpfade); Tool-Zustand und .planning aus dem Index

*Aufwand S · hängt ab von P0.1 · wartet auf –*

Vorher: `git status --porcelain --ignored > /tmp/vorher.txt`. Die neue .gitignore wird in LF geschrieben. Nutzt Default von D9 (results/ komplett lokal).

- Global bleiben nur Binärendungen: *.npy, *.npz, *.h5, *.hdf5, *.keras, *.pkl, *.parquet, *.pt, *.pth, *.ckpt, *.onnx, *.wav.
- Verzeichnisregeln: data/, models_registry/, plots/, results/, analysis/*.csv (Übergang, schützt die lokale CSV).
- Altpfade aus der Zeit vor f724856: gw_training_data/, gw_noise_background/, datasets/, models/, /model_leaderboard.csv, /optimal_thresholds.json, /plot_*.png. cc6a3bc hat diese Ordnerregeln gestrichen; seither verstecken nur die globalen *.csv/*.json/*.png-Muster die Dateien.
- Tool-Zustand: .swarm/, .claude-flow/, CLAUDE.local.md. .mcp.json und .claude/ nur, wenn sie laut Vorher-Inventar lokal von ruflo stammen.
- Caches: .pytest_cache/, .ruff_cache/, *.egg-info/, build/, dist/, temp_sound.wav.
- Entfernt werden: globale *.json/*.csv/*.png, *.deb/*.deb.1, KI-Chat-Kommentare.

Dann `git rm -r --cached .swarm .claude-flow` (nur aus dem Index, der lokale ruflo-Zustand bleibt) und `git rm -r .planning` (falscher KI-Selbstbericht). Kein History-Rewrite.

- **Dateien:** `.gitignore`, `.swarm/ (nur Index)`, `.claude-flow/ (nur Index)`, `.planning/walkthrough.md`
- **Behebt:** HYG-8, data-training:GITIGNORE-ART, HYG-5, HYG-11, EVAL-19
- **Risiko:** Ohne die globalen *.json/*.csv-Muster könnten lokale Dateien als '??' auftauchen und mit `git add -A` committet werden. Gegenmittel: Verzeichnis- und Altpfadregeln im selben Commit und ein Vorher/Nachher-Vergleich.
- **Prüfung:** `git status --porcelain --ignored > /tmp/nachher.txt; diff <(grep '^??' /tmp/vorher.txt) <(grep '^??' /tmp/nachher.txt)` zeigt keine neuen Daten-, Ergebnis- oder Tool-Dateien; `git check-ignore analysis/model_leaderboard.csv models_registry/m.json models_registry/m.keras data/training/labels.csv plots/a.png results/x.csv .swarm/memory.db temp_sound.wav gw_noise_background/r.npy gw_training_data/labels.csv model_leaderboard.csv optimal_thresholds.json plot_x.png | wc -l` → 13; `git check-ignore docs/img/a.png tests/fixtures/a.json; echo $?` → 1; `git ls-files .swarm .claude-flow .planning | wc -l` → 0; `test -d .swarm` (lokal noch da); `file .gitignore | grep -c CRLF` → 0.

<a id="p12"></a>
#### P1.2: Leaderboard und Schwellen-JSON nach results/ verlegen – vor dem Löschen von analysis/

*Aufwand S · hängt ab von P0.5, P1.1 · wartet auf –*

In config:

- RESULTS_DIR = ROOT_DIR/results;
- LEADERBOARD_FILE → results/model_leaderboard.csv;
- OPTIMAL_THRESHOLDS_FILE → results/optimal_thresholds.json;
- LEGACY_LEADERBOARD_FILES = (analysis/model_leaderboard.csv, ./model_leaderboard.csv).

Die Schreiber (update_leaderboard_roc, optimize_threshold) legen den Zielordner unmittelbar vor dem Schreiben an, nicht beim config-Import.

Einmal-Migration ohne Datenverlust. Sie läuft NICHT beim Import (das wäre wieder ein Seiteneffekt, DATA-7), sondern in einer Hilfsfunktion in src/utils/evaluation.py, die Leaderboard-Schreiber und show_leaderboard vor dem ersten Zugriff aufrufen:

- Fehlt results/model_leaderboard.csv und existiert analysis/model_leaderboard.csv, wird mit shutil.copy2 KOPIERT und ein Hinweis ausgegeben; das Original bleibt.
- Liegt zusätzlich ein Alt-Leaderboard im Root (vor f724856, ggf. anderes Schema), gibt es nur einen Hinweis und keine Übernahme.
- optimal_thresholds.json liest niemand (HYG-M3); sie wird daher nicht migriert.

Im Smoke-Test entfällt die Vorab-Anlage von $GWD_HOME/analysis/, und test_pipeline.py prüft die neuen Pfade. Dieser Schritt muss vor P1.3 gemergt sein.

- **Dateien:** `config.py`, `src/utils/evaluation.py`, `src/optimize_threshold.py`, `src/show_leaderboard.py`, `tests/smoke/conftest.py`, `tests/smoke/test_pipeline.py`, `tests/test_results_paths.py`
- **Behebt:** HYG-2, DATA-8, evaluation:MISS-3, EVAL-19, hygiene:HYG-M3
- **Risiko:** Alte und neue Datei koexistieren eine Weile. Das Risiko ist gering, weil die Kopie idempotent ist und nichts gelöscht wird.
- **Prüfung:** `python -m pytest -m slow tests/smoke -q` grün ohne Vorab-Anlage von analysis/. `python -m pytest tests/test_results_paths.py -q` prüft: Kopie entsteht, Original byte-identisch; ein Root-Leaderboard löst nur einen Hinweis aus; `import config` legt results/ nicht an. Beim Besitzer zeigt `python src/show_leaderboard.py` die bisherigen Einträge, danach meldet `cmp analysis/model_leaderboard.csv results/model_leaderboard.csv` identisch.

<a id="p13"></a>
#### P1.3: Byte-Duplikate analysis/*.py und apps/*.py löschen (Guard gegen den Ausgangsstand)

*Aufwand S · hängt ab von P1.2 · wartet auf –*

Die Schutzabfrage prüft gegen den Ausgangsstand, weil P0.4 und P1.2 die src-Seiten von show_leaderboard.py und optimize_threshold.py bereits geändert haben:

(1) `git diff --quiet pre-cleanup -- analysis apps`: keine lokalen Änderungen an den Kopien seit dem Tag, auch keine uncommitteten.

(2) Für alle 7 Paare `cmp <(git show pre-cleanup:analysis/X.py) <(git show pre-cleanup:src/X.py)`. Für apps/ gelten die Paare glitch_hunter.py↔glitch_hunter_app.py, gw_waveform_sim.py↔gw_simulator.py und interferometer_sim.py↔interferometer_simulator.py.

Schlägt eine Prüfung fehl, wird abgebrochen und dem Besitzer der Diff gezeigt. Erst danach `git rm analysis/*.py apps/*.py`. NIE `rm -rf analysis`, dort liegt ggf. die lokale Leaderboard-CSV. Keine weiteren Änderungen in diesem PR.

- **Dateien:** `analysis/compare_models_roc.py`, `analysis/evaluate_model_with_roc.py`, `analysis/optimize_threshold.py`, `analysis/show_leaderboard.py`, `apps/glitch_hunter.py`, `apps/gw_waveform_sim.py`, `apps/interferometer_sim.py`
- **Behebt:** HYG-1, APP-1, EVAL-9, CORE-5
- **Risiko:** Gering nach P1.2. Der Guard schützt lokale Änderungen an den Kopien.
- **Prüfung:** `git ls-files analysis apps | wc -l` → 0; `git grep -lE 'analysis/|apps/' -- '*.py' | sort` liefert höchstens config.py und tests/test_results_paths.py (Altpfad-Konstante und ihr Test); `python -m pytest -m 'slow or apps or golden' -q` grün; beim Besitzer zeigt `ls analysis/` weiterhin model_leaderboard.csv, falls sie vorher existierte.

<a id="p14"></a>
#### P1.4: Tote Skripte und irreführende KI-Liefernotizen entfernen

*Aufwand S · hängt ab von P0.4 · wartet auf –*

Gelöscht werden:

- utility/fix_plots.py: ein erledigter Einmal-Codemod, der Dateien in place überschreibt;
- docs/FINAL_SUMMARY.md und docs/INSTALLATION_CHECKLIST.md: KI-Chat-Liefernotizen mit erfundenen Werten und Kopieranleitungen aus /mnt/user-data/outputs. Die Historie behält sie; ein Archivordner würde KI-Assistenten weiter in die Irre führen.

docs/ROC_ANALYSIS_README.md und docs/ROC_INTEGRATION_OVERVIEW.md bleiben bis P7.1, weil ihre Erklärungen zu Metriken, Plots und Strategien stimmen. Sie bekommen oben den Hinweis 'Befehle veraltet; evaluate_classic_fixed.py hat nie existiert; Matched-Filter-Baseline derzeit nicht vorhanden'.

utility/test_imports.py bleibt bis P2.1 (danach gwd.env).

- **Dateien:** `utility/fix_plots.py`, `docs/FINAL_SUMMARY.md`, `docs/INSTALLATION_CHECKLIST.md`, `docs/ROC_ANALYSIS_README.md`, `docs/ROC_INTEGRATION_OVERVIEW.md`
- **Behebt:** HYG-17, EVAL-26, HYG-13, HYG-3, HYG-14, DATA-24, hygiene:HYG-M2
- **Risiko:** Gering, die Historie behält alles.
- **Prüfung:** `git ls-files utility/fix_plots.py docs/FINAL_SUMMARY.md docs/INSTALLATION_CHECKLIST.md | wc -l` → 0; `head -5 docs/ROC_ANALYSIS_README.md docs/ROC_INTEGRATION_OVERVIEW.md | grep -c veraltet` → 2; `python -m pytest -q` grün.

<a id="p15"></a>
#### P1.5: Zielbild festschreiben: docs/architecture.md (aus PROJECT_OVERVIEW.md)

*Aufwand M · hängt ab von P1.3 · wartet auf –*

`git mv PROJECT_OVERVIEW.md docs/architecture.md`, danach neu schreiben. Quelle ist dieser Plan (docs/aufraeumen/PLAN.md). Inhalt:

- die Komponenten config, core, data, registry, training, evaluation, apps, cli;
- die Importregel-Tabelle, die P2.6 als Test umsetzt;
- die Verschiebetabelle 'heute: <Pfad> → Ziel: <Pfad>';
- die Liste der zusammenzuführenden Funktionen: Injektion 6×, Konditionierung 3×, z-Score ~10×, Modellsuche 4×, Event-Liste 2×, Schwellen 0.15/0.46/0.75;
- den Abschnitt 'Szenarien und SNR-Definition (Ist-Zustand V1)' mit den Skews und Issue-IDs;
- die Ausgangslage F1–F4 aus P0.1 und die Antworten auf D1–D12 (bzw. 'Default gilt');
- eine schlichte Schritt-Checkliste (Schritt-ID, erledigt).

Die Falschaussagen von PROJECT_OVERVIEW werden korrigiert (alles in src/, gw_training_data, evaluate_classic, kaputte Links). KI-Arbeitsaufträge lauten danach 'Führe Schritt Px.y aus, ändere nur die genannten Dateien'.

- **Dateien:** `PROJECT_OVERVIEW.md → docs/architecture.md`
- **Behebt:** HYG-9, APP-14, CORE-5, hygiene:HYG-M5, HYG-23, data-training:SNR-DEF
- **Risiko:** Details können sich durch spätere Antworten ändern. Die Checkliste wird mitgepflegt, nicht jede Issue-ID einzeln.
- **Prüfung:** `test -f docs/architecture.md && test ! -e PROJECT_OVERVIEW.md`; `python -c 'import re,subprocess; t=open("docs/architecture.md").read(); f=set(subprocess.check_output(["git","ls-files"]).decode().split()); m=[p for p in re.findall("heute: ([^ |]+)", t) if p not in f]; print(m); assert not m'` (jeder heute-Pfad existiert); `for i in $(seq 1 12); do grep -qw D$i docs/architecture.md || echo fehlt D$i; done` gibt nichts aus; `grep -cE 'F[1-4]' docs/architecture.md` ≥ 4.

<a id="p16"></a>
#### P1.6: Zeilenenden normalisieren (reiner Format-Commit)

*Aufwand S · hängt ab von P1.1 · wartet auf –*

.gitattributes mit `* text=auto eol=lf` sowie `*.npy binary` und `*.keras binary`. Danach `git add --renormalize .` als eigener Commit ohne inhaltliche Änderung. Er muss vor der ersten inhaltlichen Bearbeitung der 6 CRLF-Dateien liegen (src/gwd_core/{interferometer,noise,simulation,waveforms}.py, src/dataset/{fetch_real_noise,generate_chirp_dataset}.py), damit spätere Diffs und die Rename-Erkennung beim git mv sauber bleiben. Den Besitzer bitten, Windows- und WSL-Editoren auf LF einzustellen. Fehlende End-Newlines kommen mit dem Format-Commit in P2.2.

- **Dateien:** `.gitattributes`, `src/gwd_core/*.py`, `src/dataset/fetch_real_noise.py`, `src/dataset/generate_chirp_dataset.py`
- **Behebt:** HYG-19
- **Risiko:** Sehr gering. Der Commit kommt später in .git-blame-ignore-revs.
- **Prüfung:** `git ls-files --eol | grep -c 'i/crlf'` → 0; `git diff HEAD~1 --ignore-cr-at-eol --numstat` zeigt außer .gitattributes nur 0/0-Zeilen; `python -m pytest -m 'not network' -q` grün inkl. golden.

### P2: Ein installierbares Paket gwd: Umzug per git mv, Grenzen per Test

**Ziel:** Das Paket ist per `pip install -e` installierbar, es gibt eine Modulidentität und keine sys.path-Hacks mehr. Jede Datei liegt per git mv in ihrer Komponente. Die Importregeln sichert ein Test, die CI prüft jeden Push. Die Golden-Werte bleiben wirksam und unverändert, Daten und Modelle des Besitzers werden weiter gefunden.

**Fertig, wenn:** In einem frischen venv sind `pip install -e '.[all,dev]'` und `GWD_REQUIRE_PYCBC=1 pytest -m 'not network' -rs` grün, ohne Golden-Skips; die CI ist grün. `ls src` zeigt nur gwd/. Es gibt keinen sys.path-Hack und keine Root-config, und `import utils`/`dataset`/`config` schlägt außerhalb des Repos fehl. Der Architektur-Test ist grün mit dokumentierter ALLOWLIST. Alte Modelle laden.

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P2.1](#p21) | pyproject + gwd_core → gwd.core + Tests umstellen + conftest löschen (ein PR) | M | P1.3, P1.6, P0.6 | – |
| [P2.2](#p22) | ruff-Basis, Format-Commit und schlanke CI | S | P2.1 | – |
| [P2.3](#p23) | config → gwd/config.py mit fester Projektwurzel; ensure_dirs() statt makedirs beim Import | M | P2.2 | – |
| [P2.4](#p24) | Daten- und Trainingsskripte per git mv nach gwd.data / gwd.training; Menü cwd-unabhängig | M | P2.3 | – |
| [P2.5](#p25) | Evaluation, utils und Apps per git mv nach gwd.evaluation / gwd.data / gwd.apps; Registry-Keim | M | P2.4 | – |
| [P2.6](#p26) | Übergangs-Weiche und letzte sys.path-Hacks entfernen; Architektur-Test mit Ratsche | S | P2.5 | – |

<a id="p21"></a>
#### P2.1: pyproject + gwd_core → gwd.core + Tests umstellen + conftest löschen (ein PR)

*Aufwand M · hängt ab von P1.3, P1.6, P0.6 · wartet auf –*

Nutzt Default von D7 (requires-python >=3.11, Versionsbereiche) und D8 (ein Paket gwd). Die Antworten möglichst vorher einholen.

Commit 1 ('golden: neue Umgebung', nur falls nötig): ein frisches venv (`python3.12 -m venv`) mit der neuen pyproject-Installation, der Code bleibt unverändert. Dann `GWD_REQUIRE_PYCBC=1 pytest -m golden -rs`. Fehlen die PyCBC-Kennwerte (F1: kein pycbc) oder weichen sie über die Toleranz ab, `python tests/golden/regen.py` ausführen und mit Versionsangaben committen. So bleibt Golden in P2–P3 wirksam, statt übersprungen zu werden.

pyproject.toml:

- [build-system] setuptools;
- [project] name='gwd', requires-python='>=3.11';
- Basis numpy, scipy, pandas, matplotlib>=3.9, scikit-learn, tabulate (Untergrenzen statt Exakt-Pins);
- Extras physics=['pycbc>=2.5; platform_system != "Windows"'] (2.4.0 hat für keine aktuelle Python-Version Wheels), gwdata=['gwpy'], ml=['tensorflow'], dev=['pytest','ruff'], all;
- [tool.setuptools.packages.find] where=['src'], include=['gwd*'], sonst würden src/dataset und src/utils als Top-Level-Pakete mitgebaut (HYG-M5);
- pytest pythonpath=['src','.'] nur als Übergang bis P2.6.

Commit 2, rein mechanisch: `git mv src/gwd_core src/gwd/core` und ein leeres src/gwd/__init__.py.

Commit 3:

- alle `from gwd_core.` → `from gwd.core.`;
- in den Tests `from src.gwd_core.` → `from gwd.core.` und `from src.utils.` → `from utils.`;
- tests/conftest.py löschen, nur in diesem PR (Prüfer-Einwand core);
- requirements.txt löschen; im README die Installation auf `pip install -e '.[all,dev]'` umstellen, mit Hinweis auf Linux/WSL wegen pycbc, tensorflow[and-cuda] nur als Hinweis, den Beitragsabschnitt ohne requirements-Locks;
- `git mv utility/test_imports.py src/gwd/env.py` und den Import reparieren. Die Backend-Zeile ('Profi-Wellenformen vs. Newton') bleibt, weil sie die einzige Anzeige des aktiven Backends ist.

Hinweis an den Besitzer: Ergab F1 'kein pycbc', erzeugt die Pipeline ab jetzt PyCBC- statt Newton-Wellenformen. Das ist gewollt, macht alte und neue Datensätze aber unvergleichbar; das wird in docs/architecture.md festgehalten.

- **Dateien:** `pyproject.toml`, `src/gwd/__init__.py`, `src/gwd_core/ → src/gwd/core/`, `tests/conftest.py`, `tests/test_*.py`, `src/**/*.py (nur Importzeilen)`, `requirements.txt`, `README.md`, `utility/test_imports.py → src/gwd/env.py`, `tests/golden/values.py`, `docs/architecture.md`
- **Behebt:** CORE-5, HYG-20, HYG-6, HYG-7, HYG-25, HYG-16, core:MISS-1, DATA-17, hygiene:HYG-M5
- **Risiko:** Eine Neuinstallation kann TensorFlow oder CUDA der WSL-GPU-Umgebung verändern. Gegenmittel: erst ein frisches venv, das alte bleibt, bis alles grün ist.
- **Prüfung:** `python3.12 -m venv /tmp/v && /tmp/v/bin/pip install -e '.[all,dev]' && GWD_REQUIRE_PYCBC=1 /tmp/v/bin/python -m pytest -m 'not network' -rs -q` grün, ohne Skips bei golden; `git grep -nE 'gwd_core|src[.]gwd_core' -- src tests main.py` leer; `test ! -e tests/conftest.py && test ! -e requirements.txt`; `/tmp/v/bin/pip check` ohne Konflikte; `/tmp/v/bin/python -m gwd.env` nennt das aktive Backend.

<a id="p22"></a>
#### P2.2: ruff-Basis, Format-Commit und schlanke CI

*Aufwand S · hängt ab von P2.1 · wartet auf –*

(1) `ruff format` als reiner Format-Commit inklusive fehlender End-Newlines; der Hash kommt in .git-blame-ignore-revs. Er liegt bewusst vor den übrigen git mv, damit diese reine Renames bleiben.

(2) [tool.ruff] mit schmalem Regelsatz (E4, E7, E9, F, W292, I). Die Reihenfolge ist wichtig: erst format, das die E701-Befunde beseitigt, dann `ruff check --fix`. Von Hand bleiben:

- 3 nackte `except:` → `except Exception:` (generate_chirp_dataset.py, evaluate_model_with_roc.py, glitch_hunter_app.py);
- die ungenutzten Variablen text (validate_dataset.py), detected (evaluate_model_with_roc.py) und M (waveforms.py).

(3) .github/workflows/ci.yml mit zwei Jobs, ubuntu-latest, Python 3.12, pip-Cache:

- test (jeder Push/PR): `pip install -e '.[physics,gwdata,dev]'` (ohne TensorFlow, aber mit pycbc, damit der produktive PyCBC-Pfad getestet wird), `ruff format --check . && ruff check .`, `GWD_REQUIRE_PYCBC=1 pytest -m 'not slow and not network and not apps'`. Tests, die TF brauchen, nutzen importorskip.
- slow (nur workflow_dispatch): `pip install -e '.[all,dev]'` und `pytest -m 'slow or apps or golden'`.

- **Dateien:** `.github/workflows/ci.yml`, `pyproject.toml`, `.git-blame-ignore-revs`, `src/**/*.py`, `tests/**/*.py`, `main.py`
- **Behebt:** HYG-20, HYG-19, HYG-12, DATA-18, CORE-18, hygiene:HYG-M1
- **Risiko:** Golden ohne Versionsschlüssel kann in der CI durch neuere pycbc-Versionen rot werden. Das ist gewollt, es gibt keinen stillen Skip: Toleranz prüfen oder angekündigt neu erzeugen; bei Dauerproblemen D7=B (Lockfile).
- **Prüfung:** `ruff format --check . && ruff check .` mit Exit 0; `python -m pytest -m 'not network' -q` grün inkl. golden; `gh run list --limit 1` → success; im Wegwerf-Branch macht ein absichtlich kaputter Test die CI rot.

<a id="p23"></a>
#### P2.3: config → gwd/config.py mit fester Projektwurzel; ensure_dirs() statt makedirs beim Import

*Aufwand M · hängt ab von P2.2 · wartet auf –*

Commit 1: `git mv config.py src/gwd/config.py`. Die Wurzel wird NICHT mehr aus dem Modulpfad abgeleitet, sonst landen Daten still unter src/gwd/ oder in site-packages (Prüfer-Einwand data-training). Reihenfolge: GWD_HOME, sonst der nächste Elternordner von __file__ mit pyproject.toml (editable Install), sonst ein Fehler 'GWD_HOME setzen'. Im Root bleibt eine 2-Zeilen-Weiche config.py (`from gwd.config import *`, entfällt in P2.6), damit alle Skripte unverändert laufen. tests/test_config_paths.py: Ohne GWD_HOME zeigen DATA_DIR_TRAIN, DATA_DIR_NOISE, MODELS_DIR, PLOTS_DIR, LEADERBOARD_FILE und LEGACY_LEADERBOARD_FILES exakt auf dieselben absoluten Pfade wie vorher.

Aus dem Review von P0.2 (PR #3) zusätzlich:

- Ist `GWD_HOME` gesetzt, wird beim ersten Import einmal gewarnt („Using GWD_HOME=…“), damit eine vergessene Variable nicht unbemerkt alle Daten umleitet.
- Ein leerer oder nur aus Leerzeichen bestehender Wert gilt als nicht gesetzt (`os.environ.get('GWD_HOME', '').strip()`).
- Beides bekommt einen Fall in tests/test_config_paths.py.

Commit 2: Die makedirs-Aufrufe auf Modulebene fallen weg. `ensure_dirs(*paths)` wird nur dort aufgerufen, wo geschrieben wird: Rauschdownload, Training, Validierung, Evaluation, Thresholds, Compare. Der tote Zweig `if not os.path.exists(MODELS_DIR)` im Glitch Hunter wird dadurch erreichbar. Die Smoke-Tests im leeren GWD_HOME finden vergessene Stellen.

- **Dateien:** `config.py`, `src/gwd/config.py`, `src/dataset/*.py`, `src/evaluate_model_with_roc.py`, `src/optimize_threshold.py`, `src/compare_models_roc.py`, `tests/test_config_paths.py`, `tests/test_config_no_side_effects.py`
- **Behebt:** DATA-17, DATA-7, CORE-18, APP-8, hygiene:HYG-M5
- **Risiko:** Eine falsch aufgelöste Wurzel ließe Modelle scheinbar verschwinden; der Pfadtest fängt das ab. Ein vergessenes ensure_dirs fällt im Smoke-Test auf.
- **Prüfung:** Ohne GWD_HOME: `python -c 'import os, config, gwd.config as c; assert config.MODELS_DIR == c.MODELS_DIR == os.path.abspath("models_registry")'`. Dann `export GWD_HOME=$(mktemp -d); python -c 'import gwd.config'; ls -A $GWD_HOME | wc -l` → 0; `python -m pytest tests/test_config_paths.py tests/test_config_no_side_effects.py -q`; `python -m pytest -m 'slow or apps' -q` grün; `git log --follow --oneline -- src/gwd/config.py | tail -1` zeigt f724856.

<a id="p24"></a>
#### P2.4: Daten- und Trainingsskripte per git mv nach gwd.data / gwd.training; Menü cwd-unabhängig

*Aufwand M · hängt ab von P2.3 · wartet auf –*

Commit 1, nur git mv:

- src/dataset/fetch_real_noise.py → src/gwd/data/fetch_noise.py
- generate_chirp_dataset.py → data/generate.py
- validate_dataset.py → data/validate.py
- src/dataset/__init__.py → src/gwd/data/__init__.py
- src/dataset/train_cnn.py → src/gwd/training/train.py (Training gehört nicht nach dataset/)

src/utils/data_processing.py bleibt bewusst bis P2.5, weil evaluate_model_with_roc.py:31 es per `from utils.data_processing import` nutzt.

Commit 2:

- Die sys.path-Hacks dieser Module werden durch absolute Importe ersetzt (`from gwd.config import …`, `from gwd.core.waveforms import …`). Jedes Modul bleibt per `python -m gwd.data.generate` usw. startbar.
- main.py bekommt eine Tabelle MENU. Bereits umgezogene Module stehen als ('module', 'gwd.data.generate'), die übrigen als ('path', 'src/evaluate_model_with_roc.py') relativ zu __file__. Gestartet wird per `subprocess.run([sys.executable, …])` statt os.system, mit ehrlichem Exit-Code und unabhängig vom Arbeitsverzeichnis.
- entrypoints.py und die README-Befehle werden nachgezogen.

Keine inhaltlichen Änderungen.

- **Dateien:** `src/dataset/* → src/gwd/data/*`, `src/dataset/train_cnn.py → src/gwd/training/train.py`, `src/gwd/training/__init__.py`, `main.py`, `README.md`, `tests/smoke/entrypoints.py`, `tests/test_menu.py`
- **Behebt:** DATA-13, DATA-17, DATA-14, HYG-21, apps:APPX-5, hygiene:HYG-M5
- **Risiko:** Wegen der editable-.pth (src/ auf sys.path) bestünde ein find_spec-Test auch für nicht umgezogene Top-Level-Skripte. Deshalb führt MENU nur Module mit Präfix gwd. als 'module', alles andere als Pfad. Alte .keras-Modelle enthalten nur Standard-Keras-Schichten; eine Ladeprobe bestätigt das.
- **Prüfung:** `python -m pytest -m 'slow or apps or golden' -q` grün; `git grep -n 'sys.path' -- src/gwd` leer; `git show -M --stat HEAD~1` listet nur Renames; `python -m pytest tests/test_menu.py -q` (jeder 'module'-Eintrag beginnt mit gwd. und ist per find_spec auffindbar, jeder 'path'-Eintrag existiert relativ zu main.py); `R=$(git rev-parse --show-toplevel); (cd /tmp && echo 0 | python $R/main.py); echo $?` → 0; beim Besitzer `python -c 'import tensorflow as tf; tf.keras.models.load_model("models_registry/<altes>.keras")'`.

<a id="p25"></a>
#### P2.5: Evaluation, utils und Apps per git mv nach gwd.evaluation / gwd.data / gwd.apps; Registry-Keim

*Aufwand M · hängt ab von P2.4 · wartet auf –*

Commit 1, nur git mv:

- src/evaluate_model_with_roc.py → src/gwd/evaluation/evaluate.py
- optimize_threshold.py → evaluation/thresholds.py
- compare_models_roc.py → evaluation/compare.py
- src/utils/evaluation.py → evaluation/leaderboard.py
- src/utils/data_processing.py → src/gwd/data/preprocessing.py
- src/gw_simulator.py → src/gwd/apps/waveform_explorer.py
- src/interferometer_simulator.py → src/gwd/apps/interferometer_designer.py
- src/glitch_hunter_app.py → src/gwd/apps/glitch_hunter/app.py
- tests/test_leaderboard.py → tests/evaluation/test_leaderboard.py
- dazu die __init__.py-Dateien.

Commit 2:

- show_leaderboard.py wird als Funktion show() mit main-Guard in leaderboard.py übernommen, danach `git rm src/show_leaderboard.py`.
- get_latest_model wandert unverändert nach src/gwd/registry.py; die Semantik ändert erst P3.3.
- `git rm src/utils/__init__.py`, der generische Top-Level-Name utils verschwindet.
- Importe werden absolut, die sys.path-Hacks fallen weg. gwpy und TF bleiben vorerst oben (ALLOWLIST in P2.6).
- main.py: Alle Einträge sind jetzt ('module', 'gwd.…'); die fehlenden Menüpunkte Interferometer-Designer und Leaderboard kommen hinzu.
- entrypoints.py, test_results_paths.py und die README-Befehle werden nachgezogen.

core/interferometer.py und core/simulation.py bleiben bewusst im Kern. APP-14 ist eine Designfrage; die Grenze sichert die Importregel.

- **Dateien:** `src/evaluate_model_with_roc.py → src/gwd/evaluation/evaluate.py`, `src/optimize_threshold.py → src/gwd/evaluation/thresholds.py`, `src/compare_models_roc.py → src/gwd/evaluation/compare.py`, `src/utils/evaluation.py → src/gwd/evaluation/leaderboard.py`, `src/utils/data_processing.py → src/gwd/data/preprocessing.py`, `src/gw_simulator.py → src/gwd/apps/waveform_explorer.py`, `src/interferometer_simulator.py → src/gwd/apps/interferometer_designer.py`, `src/glitch_hunter_app.py → src/gwd/apps/glitch_hunter/app.py`, `src/show_leaderboard.py`, `src/utils/__init__.py`, `src/gwd/registry.py`, `src/gwd/evaluation/__init__.py`, `src/gwd/apps/__init__.py`, `src/gwd/apps/glitch_hunter/__init__.py`, `main.py`, `README.md`, `tests/smoke/entrypoints.py`, `tests/test_results_paths.py`, `tests/test_leaderboard.py → tests/evaluation/test_leaderboard.py`, `tests/test_menu.py`
- **Behebt:** EVAL-24, EVAL-23, APP-17, APP-16, DATA-16, HYG-21, apps:APPX-5, APP-14, hygiene:HYG-M5, DATA-17
- **Risiko:** Größter Umzugs-PR (8 Renames). Gegenmittel: Commit 1 enthält nur Renames, Smoke und Golden sichern ab. Der Glitch Hunter braucht beim Import weiter TF und gwpy (bis P3.4/P4.4).
- **Prüfung:** `python -m pytest -m 'slow or apps or golden' -q` grün; `ls src` → gwd; `find src -maxdepth 1 -name '*.py' | wc -l` → 0; `git grep -n 'sys.path' -- src/gwd` leer; `git show -M --stat HEAD~1` zeigt nur Renames; `GWD_HOME=$(mktemp -d) python -m gwd.evaluation.leaderboard; echo $?` → Meldung 'noch kein Leaderboard' und 0; `python -m pytest tests/test_menu.py -q` (11 Einträge, alle 'module').

<a id="p26"></a>
#### P2.6: Übergangs-Weiche und letzte sys.path-Hacks entfernen; Architektur-Test mit Ratsche

*Aufwand S · hängt ab von P2.5 · wartet auf –*

Die Root-config.py wird gelöscht, die letzten `from config import` werden zu `from gwd.config import`, und 'src' und '.' entfallen aus dem pytest-pythonpath.

Neu ist tests/test_architecture.py: ast-basiert, ~50 Zeilen, ohne neue Abhängigkeit. Er erzwingt die Importregeln aus docs/architecture.md:

- config → nur Stdlib;
- core → Stdlib, numpy, scipy, pycbc (lazy), gwd.core;
- data → core, config;
- registry → config;
- training → data, core, config, registry;
- evaluation → data, core, config, registry, nie training;
- apps → core, data, config, registry, nie training/evaluation;
- cli/env → alles, nur lazy;
- main.py → nur Stdlib.

Zusätzlich prüft er:

- kein sys.path in src/, tests/ und main.py;
- keine zwei byte-identischen .py-Dateien (leere __init__.py ausgenommen), damit sich c4e3dd8 nicht wiederholt;
- Importanweisungen von tensorflow, gwpy und pycbc (auch lazy) nur an erlaubten Stellen.

Bekannte Altlasten stehen in einer kommentierten ALLOWLIST mit Issue-ID. Sie darf nur schrumpfen und muss in P7.4 leer sein: TF/gwpy oben in apps/glitch_hunter/app.py und evaluation/evaluate.py (APP-7, EVAL-24), print in core (CORE-17).

- **Dateien:** `config.py`, `tests/test_architecture.py`, `pyproject.toml`, `src/**/*.py (Importzeilen)`, `docs/architecture.md`
- **Behebt:** CORE-5, HYG-1, hygiene:HYG-M5, DATA-17, EVAL-24, HYG-20, APP-7, CORE-17
- **Risiko:** Gering. Zu strenge Regeln werden bewusst an einer Stelle gelockert, nicht umgangen.
- **Prüfung:** `python -m pytest tests/test_architecture.py -q` grün; `test ! -e config.py`; `git grep -n 'sys.path' -- src tests main.py` leer; `cd /tmp && python -c 'import importlib.util as u; assert all(u.find_spec(m) is None for m in ("utils","dataset","config","gwd_core"))'`. Gegenproben: `import matplotlib` testweise in src/gwd/core/noise.py → rot; eine .py-Datei testweise kopieren → rot.

### P3: Eine Quelle pro Logik (verhaltensgleich, Golden-gesichert)

**Ziel:** Konstanten, Konditionierung, Normierung, Modellsuche, GWOSC-Zugriff mit Event-Liste, Injektion und Kommandozeile gibt es danach je genau einmal, in ihrer Komponente. Der Train/Eval/App-Skew wird als benannte V1-Szenarien an EINER Stelle sichtbar, aber nicht geändert. Die CLI gwd ersetzt die Smoke-Treiber.

**Fertig, wenn:** `git grep` findet keine Kopien von Injektion, Filterkette, z-Score, Modellsuche, Event-Liste oder Abtastkonstanten außerhalb ihres Moduls. Golden ist unverändert; der Notch 240 im Feldtest liegt nicht im Golden. Die Smoke-Tests laufen ausschließlich über die CLI. Der Skew ist in data/scenarios.py und im xfail-Test dokumentiert. Zusammen mit P7.1 ist der Minimalpfad 'Ordnung' hier abgeschlossen.

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P3.1](#p31) | Zentrale Konstanten statt hart codierter 4096/4.0/16384 | S | P2.6 | – |
| [P3.2](#p32) | Eine Konditionierungs- und eine Normierungsfunktion (gwpy bleibt, lazy) | M | P3.1 | – |
| [P3.3](#p33) | Modell-Registry: eine Modellsuche, Auswahl nach Zeitstempel im Namen, --model | S | P2.6 | – |
| [P3.4](#p34) | GWOSC-Zugriff und gemeinsame Event-Liste (gwd.data.gwosc) | M | P3.2 | – |
| [P3.5](#p35) | Injektion zentralisieren mit benannten V1-Szenarien (Skew sichtbar, nicht behoben) | L | P3.4, P0.6 | – |
| [P3.6](#p36) | Einheitliche CLI gwd, gwd env, dünnes Menü; Smoke-Tests auf die CLI umstellen | M | P3.3, P3.4, P3.5 | – |
| [P3.7](#p37) | Training aufteilen: model.py / train.py ohne Modul-Seiteneffekte | S | P3.6 | – |

<a id="p31"></a>
#### P3.1: Zentrale Konstanten statt hart codierter 4096/4.0/16384

*Aufwand S · hängt ab von P2.6 · wartet auf –*

config bekommt N_SAMPLES = int(SAMPLE_RATE*DURATION), NOTCH_FREQS=(60,120,180,240) und BANDPASS=(20,300). Alle hart codierten 4096/4.0/16384 in data, training, evaluation und apps werden durch config-Konstanten ersetzt (früher fetch_real_noise.py:33-34, generate:74-75, train:76, validate:127/285/333/336, evaluate:40-41, thresholds:28-29, compare:24-25, glitch_hunter:62). 'labels.csv' kommt nur noch aus LABELS_FILE. Die Zeitachse bleibt vorerst linspace, weil arange eine Verhaltensänderung ist (P5.3). gwd.core bekommt Abtastrate und Dauer weiter als Argument und importiert config nicht.

- **Dateien:** `src/gwd/config.py`, `src/gwd/data/*.py`, `src/gwd/training/train.py`, `src/gwd/evaluation/*.py`, `src/gwd/apps/*.py`, `src/gwd/apps/glitch_hunter/app.py`
- **Behebt:** DATA-9, EVAL-10, CORE-18, DATA-22
- **Risiko:** Gering, die Werte sind identisch.
- **Prüfung:** `git grep -nwE '4096|16384' -- src/gwd/data src/gwd/training src/gwd/evaluation src/gwd/apps` leer; `python -m pytest -m 'golden or slow or apps' -q` grün.

<a id="p32"></a>
#### P3.2: Eine Konditionierungs- und eine Normierungsfunktion (gwpy bleibt, lazy)

*Aufwand M · hängt ab von P3.1 · wartet auf –*

gwd/data/preprocessing.py hat genau zwei öffentliche Funktionen:

- `condition_strain(ts, fs, notches=NOTCH_FREQS, band=BANDPASS)`: gwpy lazy, resample, Notch-Kette, Bandpass, Rückgabe ndarray. Die Variable 'white_data' wird umbenannt, weil nicht geweißt wird.
- `normalize(x, eps=1e-10)`: z-Normierung.

Umgestellt werden fetch_noise (Inline-Kette), der Feldtest in evaluate, der Glitch Hunter (Inline-Kopie) sowie die z-Score-Kopien in generate, train, evaluate, thresholds, compare und Glitch Hunter. Bewusst kein scipy-Nachbau der gwpy-Kette.

Kleine, im PR benannte Verhaltensänderung: Der Feldtest bekommt jetzt auch notch(240) wie Training und App. Die Randartefakte durch das Filtern von nur 4 s behebt P4.3.

Unit-Test: Eine synthetische gwpy-TimeSeries durch condition_strain ist bitgleich mit der alten fetch-Kette.

- **Dateien:** `src/gwd/data/preprocessing.py`, `src/gwd/data/fetch_noise.py`, `src/gwd/data/generate.py`, `src/gwd/training/train.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/evaluation/thresholds.py`, `src/gwd/evaluation/compare.py`, `src/gwd/apps/glitch_hunter/app.py`, `tests/data/test_preprocessing.py`
- **Behebt:** DATA-10, EVAL-5, APP-10, CORE-7
- **Risiko:** Ein vereinheitlichtes eps kann winzige numerische Abweichungen erzeugen. Golden prüft mit rtol 1e-9; eine Abweichung wird im PR begründet.
- **Prüfung:** `python -m pytest tests/data/test_preprocessing.py -q`; `git grep -nE '[.]notch[(]' -- src` trifft nur preprocessing.py; `git grep -nE '/ *[(]?np[.]std' -- src/gwd | grep -v preprocessing.py` leer; `python -m pytest -m 'golden or slow or apps' -q` grün.

<a id="p33"></a>
#### P3.3: Modell-Registry: eine Modellsuche, Auswahl nach Zeitstempel im Namen, --model

*Aufwand S · hängt ab von P2.6 · wartet auf –*

src/gwd/registry.py stellt list_models(), latest_model(), resolve('latest'|Name|Pfad), load_model(path, compile=False) mit lazy TF-Import sowie read_metadata()/write_metadata() bereit. latest_model sortiert nach dem Zeitstempel im Dateinamen gwd_model_YYYYMMDD-HHMMSS, nicht nach ctime, das unter Linux/WSL bei chmod, Kopie oder Wiederherstellung springt (APPX-4). Für Altnamen ohne Zeitstempel gilt ein Fallback auf mtime mit Warnung.

Die Registry ersetzt get_latest_model, die abweichende Variante in thresholds, load_all_models in compare und die Suche im Glitch Hunter. evaluate, thresholds und compare bekommen ein optionales `--model` (Default 'latest') und importieren TensorFlow danach nicht mehr selbst; der ALLOWLIST-Eintrag 'TF oben' für evaluate.py entfällt. compile=False vermeidet auch die beobachtete Optimizer-Warnung.

Beabsichtigt und im PR genannt: 'Neuestes Modell' kann selten ein anderes sein als bisher.

- **Dateien:** `src/gwd/registry.py`, `src/gwd/evaluation/*.py`, `src/gwd/apps/glitch_hunter/app.py`, `tests/test_registry.py`, `tests/test_architecture.py`
- **Behebt:** EVAL-11, apps:APPX-4, APP-8, APP-3, DATA-20, EVAL-24
- **Risiko:** 'Neuestes Modell' kann sich ändern, wenn ctime und Name auseinanderliefen. Das ist beabsichtigt.
- **Prüfung:** `python -m pytest tests/test_registry.py -q`: zwei tmp-Dateien gwd_model_20251124-… und gwd_model_20260301-…; nach os.utime/chmod auf die ältere wählt latest weiter 20260301. `git grep -n getctime -- src` leer; `python -X importtime -c 'import gwd.registry, gwd.evaluation.evaluate' 2>&1 | grep -c tensorflow` → 0; `python -m pytest -m slow -q` grün.

<a id="p34"></a>
#### P3.4: GWOSC-Zugriff und gemeinsame Event-Liste (gwd.data.gwosc)

*Aufwand M · hängt ab von P3.2 · wartet auf –*

gwd/data/gwosc.py bekommt `fetch_strain(detector, start, end, pad_s=0, cache=True)` mit lazy gwpy und Längen- und NaN-Prüfung (Exception statt stillem np.resize). Neben preprocessing.py ist das die einzige gwpy-Stelle des Pakets.

EVENTS ist eine Liste mit Name, GPS, Typ (BBH/BNS/NSBH), groben Massen und dem Flag in_training_range. GW170817 und GW190814 sind als out-of-distribution markiert; die Massen sind als GWTC-Literaturwerte gekennzeichnet und offline nicht verifiziert. Die Liste ersetzt REAL_EVENTS (evaluate) und KNOWN_EVENTS (Glitch Hunter) sowie den Download in fetch_noise.

Verhaltensgleich: pad_s=0 reproduziert die heutigen 4-s-Fenster; der Puffer folgt in P4.3. Die ALLOWLIST-Einträge 'gwpy oben' für evaluate.py und glitch_hunter/app.py entfallen.

- **Dateien:** `src/gwd/data/gwosc.py`, `src/gwd/data/fetch_noise.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/apps/glitch_hunter/app.py`, `tests/data/test_gwosc.py`, `tests/test_architecture.py`
- **Behebt:** apps:APPX-3, core:MISS-4, EVAL-24, APP-7, DATA-25, EVAL-22
- **Risiko:** GWOSC ist in Tests nicht erreichbar. Deshalb wird alles gemockt; echte Downloads laufen nur mit dem Marker network.
- **Prüfung:** `python -m pytest tests/data/test_gwosc.py -q`: gemocktes fetch_open_data durchläuft Längen- und NaN-Prüfung; GW170817 hat in_training_range=False. test_architecture erlaubt gwpy-Importanweisungen nur noch in data/gwosc.py und data/preprocessing.py (env.py prüft per find_spec, nicht per import); `python -X importtime -c 'import gwd.evaluation.evaluate' 2>&1 | grep -c gwpy` → 0.

<a id="p35"></a>
#### P3.5: Injektion zentralisieren mit benannten V1-Szenarien (Skew sichtbar, nicht behoben)

*Aufwand L · hängt ab von P3.4, P0.6 · wartet auf –*

gwd/data/scenarios.py enthält eine eingefrorene Dataclass Scenario (name, version, snr_kind='peak', snr_range, mass_mode random|fixed(30,30), t_merger_range, noise_mix real/gauss, condition_signal=False) mit hash(). Drei Szenarien bilden den IST-Zustand exakt nach:

- TRAIN_V1: Peak-SNR 1–12, Massen log-uniform 5–100, t_m 2.0–3.6, 80 % Echtrauschen falls vorhanden, Signal ungefiltert addiert.
- EVAL_V1: 0.5–2.5, 30/30, t_m 2.0–3.5, weiß.
- APP_V1: 0.5–2.0, 30/30, t_m 2.0–3.8, weiß.

gwd/data/injection.py enthält draw_params(scenario, rng) und make_sample(...) mit der heutigen Peak-Skalierung sowie load_random_real_noise als noise_provider. Die Chunk-Liste wird sortiert (sorted(glob)), damit gleiche Seeds unabhängig vom Dateisystem gleiche Chunks ziehen.

Umgestellt wird je in einem eigenen Commit: Generator, evaluate (ROC-Set und Sensitivitätskurve), thresholds, compare, Glitch Hunter. Die RNG-Aufrufreihenfolge bleibt erhalten, damit Golden (d) bitgleich bleibt.

test_known_skew mit xfail(strict=True, reason='V1-Skew, Korrektur als V2 in P6.2') fordert TRAIN_V1==EVAL_V1 und dokumentiert den Altbestand. Der falsche Docstring 'entspricht der Logik in evaluate_model_with_roc.py' wird korrigiert.

Nebeneffekt: Danach rufen nur noch injection, simulation und Tests generate_astrophysical_chirp/generate_gaussian_noise auf. Die Umbenennungen in P5 kommen damit ohne Alias-Zoo aus (Prüfer-Einwand core).

- **Dateien:** `src/gwd/data/scenarios.py`, `src/gwd/data/injection.py`, `src/gwd/data/generate.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/evaluation/thresholds.py`, `src/gwd/evaluation/compare.py`, `src/gwd/apps/glitch_hunter/app.py`, `tests/data/test_injection.py`, `tests/golden/test_golden.py`
- **Behebt:** CORE-6, EVAL-10, DATA-1, EVAL-1, EVAL-2, EVAL-3, APP-2, data-training:NOISE-SKEW, CORE-16, hygiene:HYG-M1, core:MISS-3
- **Risiko:** Höchstes Risiko der Phase, weil 5 Aufrufer betroffen sind. Gegenmittel: ein Commit pro Aufrufer, Golden mit festen Parametern, Smoke nach jedem Commit.
- **Prüfung:** `git grep -nE 'uniform[(]0[.]5|uniform[(]1[.]0, *12|scale_signal_to_snr' -- src | grep -vE 'data/(scenarios|injection)[.]py'` leer; `python -m pytest -m 'golden or slow or apps' -q` grün; `python -m pytest -q -rx` listet test_known_skew als xfail; `python -m pytest tests/data/test_injection.py -q`: gleicher Seed → bitgleiche Batches, auch bei umgekehrter Dateireihenfolge im Rauschordner; gezogene SNR, Massen und t_merger liegen je Szenario in den Grenzen.

<a id="p36"></a>
#### P3.6: Einheitliche CLI gwd, gwd env, dünnes Menü; Smoke-Tests auf die CLI umstellen

*Aufwand M · hängt ab von P3.3, P3.4, P3.5 · wartet auf –*

Nutzt Default von D8 (ein Befehl gwd, main.py bleibt als dünnes Menü).

src/gwd/cli.py mit argparse-Subcommands, die ihr Modul lazy importieren (`gwd data generate` lädt kein TensorFlow):

- `gwd data fetch-noise [--detector --gps --duration]`, `gwd data generate --n --seed --out [--scenario]`, `gwd data validate [--data --plots]`
- `gwd train [--data --epochs --batch-size --seed]`
- `gwd eval run [--model --testset --real/--no-real --seed]`, `gwd eval thresholds [--model --save/--no-save]`, `gwd eval compare`, `gwd eval leaderboard`
- `gwd app waveform|interferometer|glitch-hunter`
- `gwd env`: Versionen, Extras, aktives Wellenform-Backend mit pycbc-Version, aufgelöste Pfade, GUI-Backend

[project.scripts] gwd = 'gwd.cli:main'; build_parser() ist eine eigene Funktion. Seeds setzt vorerst minimal seed_everything(seed) (random, np.random, tf.keras.utils.set_random_seed); die saubere rng-Durchreichung folgt in P5.3/P5.6. Das input() in thresholds entfällt, --save ist Default.

main.py bleibt reine Standardbibliothek:

- eine feste Liste MENU von (Text, Argumentliste), aufgerufen per subprocess [sys.executable, '-m', 'gwd.cli', …];
- ehrliche Exit-Codes, EOF wird abgefangen, 'cls'/'clear' bleiben.

tests/test_menu.py prüft jede Argumentliste mit gwd.cli.build_parser().parse_args() (kein SystemExit), damit Menü und CLI nicht auseinanderlaufen.

Die Smoke-Tests wechseln auf echte CLI-Aufrufe; die Monkeypatch-Treiber werden gelöscht.

- **Dateien:** `src/gwd/cli.py`, `src/gwd/env.py`, `pyproject.toml`, `main.py`, `src/gwd/evaluation/thresholds.py`, `tests/smoke/entrypoints.py`, `tests/smoke/test_pipeline.py`, `tests/smoke/test_apps.py`, `tests/test_cli.py`, `tests/test_menu.py`
- **Behebt:** DATA-15, DATA-16, DATA-14, HYG-21, EVAL-17, APP-16, DATA-5, core:MISS-3, hygiene:HYG-M2, HYG-16
- **Risiko:** Der Besitzer muss sich an neue Befehle gewöhnen; das Menü bleibt deshalb als vertrauter Einstieg.
- **Prüfung:** `gwd --help` listet alle Kommandos. `python -m pytest tests/test_cli.py tests/test_menu.py -q`: jedes Subcommand mit --help → 0, ein absichtlich fehlschlagender Aufruf → ≠ 0, jede MENU-Liste ist parsebar. `gwd eval thresholds --no-save </dev/null; echo $?` → 0 (früher EOFError). `export GWD_HOME=$(mktemp -d); gwd data generate --n 16 --seed 3 --out $GWD_HOME/a && gwd data generate --n 16 --seed 3 --out $GWD_HOME/b && diff $GWD_HOME/a/labels.csv $GWD_HOME/b/labels.csv` leer. `python main.py </dev/null; echo $?` → 0 ohne Traceback; `python -m pytest -m 'slow or apps' -q` grün; `git grep -n 'input(' -- src/gwd` leer.

<a id="p37"></a>
#### P3.7: Training aufteilen: model.py / train.py ohne Modul-Seiteneffekte

*Aufwand S · hängt ab von P3.6 · wartet auf –*

Nutzt Default von D7 (Mixed Precision automatisch nur mit GPU).

- training/model.py enthält build_model(input_shape) ohne Seiteneffekte; die Architektur bleibt unverändert (DATA-26 ist bewusst nicht Teil des Aufräumens).
- training/train.py enthält train(cfg); Zeitstempel, Modellname und Pfade entstehen erst zur Laufzeit.
- Mixed Precision gilt nur, wenn eine GPU gefunden wird oder --mixed-precision gesetzt ist. Auf der RTX unter WSL bleibt alles wie bisher, auf CPU und in der CI wird float32 verwendet.
- Die Hyperparameter (BATCH_SIZE, EPOCHS, LR, Patience) liegen als Defaults in config und sind per CLI überschreibbar.

Charakterisierungstest: build_model((16384,1)).count_params() == 1_074_609.

- **Dateien:** `src/gwd/training/model.py`, `src/gwd/training/train.py`, `src/gwd/config.py`, `tests/training/test_model.py`
- **Behebt:** DATA-13
- **Risiko:** Gering. Auf CPU ändert sich die Präzision (float32 statt mixed_float16); das ist beabsichtigt.
- **Prüfung:** `python -c 'import tensorflow as tf, gwd.training.train; assert tf.keras.mixed_precision.global_policy().name == "float32"'`; `python -m pytest tests/training/test_model.py -q` (importorskip tensorflow); `python -m pytest -m slow -q` grün; beim Besitzer meldet `gwd train --epochs 1` auf der GPU mixed_float16.

### P4: Robustheit und sichtbare Fehler (Szenarien unverändert)

**Ziel:** Die stillen Schalter (Wellenform-Backend, Rauschquelle) werden protokolliert. Alle bekannten Bugs ohne Fachentscheidung in Evaluation, Feldtest, Apps und Validator werden behoben und mit Tests abgesichert. Die Apps werden headless testbar und scheitern laut statt stumm.

**Fertig, wenn:** Jeder Datensatz und jedes Modell nennt Backend und Rauschquelle. Für alle Nicht-Fach-Bugs aus EVAL, APP und DATA gibt es einen Test. Die Apps sind headless testbar und brechen ohne GUI-Backend laut ab. Der Validator ist deterministisch, der Feldtest frei von Randtransienten. Die Szenarien sind unverändert; Golden hat sich nur im angekündigten Interferometer-Fall geändert.

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P4.1](#p41) | Stille Schalter sichtbar: Backend und Rauschquelle protokollieren, Manifest, Metadaten, Exit-Codes | M | P3.6 | – |
| [P4.2](#p42) | Simulations-Evaluation: kleine Fehler beheben, metrics.py als reine Funktionen | M | P3.6 | – |
| [P4.3](#p43) | Echtdaten-Feldtest: Puffer statt Randtransienten, ehrliche Zählung, nur mit --real | M | P3.4, P4.2 | – |
| [P4.4](#p44) | Glitch Hunter entflechten: GUI, Quellen, Audio; Race-Condition und Thread-Plotting beheben; Modi | L | P3.3, P3.5, P4.1, P4.3 | – |
| [P4.5](#p45) | Demo-Apps testbar: build_figure(), ax_noise-Bug, Finesse-Property, Emojis, GUI-Backend-Prüfung | M | P2.5, P0.6, P3.6 | – |
| [P4.6](#p46) | Validator und Laden: vollständige Integrität, Bereichsprüfung aus dem Szenario, keine Waisen | M | P4.1 | – |

<a id="p41"></a>
#### P4.1: Stille Schalter sichtbar: Backend und Rauschquelle protokollieren, Manifest, Metadaten, Exit-Codes

*Aufwand M · hängt ab von P3.6 · wartet auf –*

- core/waveforms.py: print → logging/warnings; active_backend() liefert 'pycbc x.y' oder 'newton'.
- Generator: zusätzliche Spalten in labels.csv: waveform_backend, noise_source (real|gaussian), noise_file (Chunk-Name). Die Spalten sind additiv, alte Leser ignorieren sie. Dazu <out>/manifest.json mit Seed, n, Szenario-Name und -Hash, Backend, pycbc- und numpy-Version, Git-Commit und Anteil Echtrauschen.
- Am Start von generate und evaluate steht eine gut sichtbare Zeile 'Backend: … | Rauschen: …', bei Newton zusätzlich eine laute Warnung.
- train schreibt in die Modell-JSON die tatsächlich trainierten Epochen, Seed, Manifest-Hash, Backend und Rauschanteil; ohne Manifest steht dort 'unbekannt (v1)'.
- fetch-noise endet bei Fehlern mit Exit ≠ 0, das Menü meldet das ehrlich.
- Der Glitch Hunter warnt, wenn das geladene Modell laut Metadaten ohne Echtrauschen trainiert wurde (APPX-1).

- **Dateien:** `src/gwd/core/waveforms.py`, `src/gwd/data/generate.py`, `src/gwd/data/fetch_noise.py`, `src/gwd/training/train.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/apps/glitch_hunter/app.py`, `tests/smoke/test_pipeline.py`, `tests/data/test_fetch_noise.py`
- **Behebt:** hygiene:HYG-M1, CORE-4, CORE-17, DATA-6, DATA-20, apps:APPX-1
- **Risiko:** Gering. Es kommen nur Spalten und Felder hinzu, nichts fällt weg.
- **Prüfung:** Der Smoke-Test prüft die neuen Spalten und manifest.json (backend nicht leer). Ein Test lässt fetch_open_data eine Exception werfen, dann liefert `gwd data fetch-noise` rc ≠ 0. `python -c 'import gwd.core.waveforms' | wc -c` → 0 (keine stdout-Ausgabe beim Import).

<a id="p42"></a>
#### P4.2: Simulations-Evaluation: kleine Fehler beheben, metrics.py als reine Funktionen

*Aufwand M · hängt ab von P3.6 · wartet auf –*

Das Testszenario (EVAL_V1) bleibt unverändert.

(1) compare: Der Zoom-Plot bekommt den eigenen Namen …_zoom.png. Scores werden je Modell einmal berechnet, jedes Modell hat eine feste Farbe nach Namen, die Modelle werden nacheinander geladen.

(2) evaluation/metrics.py: find_optimal_thresholds wird übernommen, nicht neu geschrieben (Prüfer-Hinweis: split statt rewrite).

- Fixed-FPR wählt die maximale TPR mit FPR ≤ Ziel, Fixed-TPR analog.
- inf-Schwellen werden gefiltert.
- F1 wird vektorisiert aus Precision und Recall berechnet.
- Operating Point, Sensitivität und Confusion-Matrix nutzen einheitlich '>='; die redundante FAR-Spalte entfällt.

(3) Die Metrikbox zeigt formatierte Werte statt roher Floats; das nackte except wird entfernt.

- **Dateien:** `src/gwd/evaluation/metrics.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/evaluation/thresholds.py`, `src/gwd/evaluation/compare.py`, `tests/evaluation/test_metrics.py`
- **Behebt:** EVAL-18, HYG-24, EVAL-16, EVAL-15, EVAL-22, EVAL-20
- **Risiko:** Die empfohlenen Schwellen von thresholds ändern sich; sie werden heute aber nirgends gelesen.
- **Prüfung:** `python -m pytest tests/evaluation/test_metrics.py -q`: Ziel-FPR 1 % → gewählte FPR ≤ 0,01, keine inf-Schwelle; das vektorisierte F1 stimmt an Stichproben mit sklearn.f1_score überein. Im Smoke-Test erzeugt `gwd eval compare` zwei PNGs mit verschiedenen Namen.

<a id="p43"></a>
#### P4.3: Echtdaten-Feldtest: Puffer statt Randtransienten, ehrliche Zählung, nur mit --real

*Aufwand M · hängt ab von P3.4, P4.2 · wartet auf –*

Nutzt Default von D10: Der Feldtest ist optional, Default ist --no-real, die Evaluation läuft offline.

evaluation/real_data.py entsteht aus evaluate_real_world:

- Events aus data.gwosc.EVENTS mit Puffer (pad_s=16) laden, mit derselben condition_strain konditionieren und erst dann die mittleren 4 s ausschneiden (neu: data.preprocessing.conditioned_crop). Damit sind die Randtransienten weg, die heute nur zur Testzeit entstehen (core:MISS-2, EVAL-4).
- Zu kurze Daten führen zum Überspringen des Events statt zu np.resize. Der Rauschtest läuft genauso.
- FAR wird als 'k Fehlalarme in T h analysierter Daten' angegeben, ohne Hochrechnung pro Jahr.
- Download-Fehler erscheinen als 'n/a (k von n geladen)' statt '0/5'.
- OOD-Events (GW170817, GW190814) werden getrennt ausgewiesen; Downloads werden gecacht.

fetch_noise verwirft beim nächsten Download Chunk 0 und den letzten Chunk, die Filter-Randtransienten enthalten (evaluation:MISS-4); die vorhandenen Chunks des Besitzers bleiben.

Das ist ein reiner Bugfix am Testpfad, das Training ändert sich nicht. Bewusste, im PR benannte Verhaltensänderung: Die Echtdaten-Zahlen alter Modelle ändern sich.

- **Dateien:** `src/gwd/evaluation/real_data.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/data/preprocessing.py`, `src/gwd/data/gwosc.py`, `src/gwd/data/fetch_noise.py`, `tests/evaluation/test_real_data.py`
- **Behebt:** EVAL-4, core:MISS-2, EVAL-13, EVAL-21, EVAL-22, core:MISS-4, CORE-7, evaluation:MISS-4
- **Risiko:** Die Echtdaten-Zahlen sind mit alten Leaderboard-Zeilen nicht vergleichbar. Das steht im PR; ab P5.8 trennt das Leaderboard v2 die Zahlen per eigener Datei.
- **Prüfung:** `python -m pytest tests/evaluation/test_real_data.py -q`:

- ein gemocktes 36-s-Segment ergibt genau 16384 Samples;
- bei synthetischem rotem Rauschen mit LF-Anteil (3e-19 bei 0,2 Hz) liegt das RMS-Verhältnis Rand/Mitte (je 0,1 s) in [0,9; 1,1]; ohne Puffer ist es reproduzierbar > 2 (Gegenprobe);
- ein Download-Fehler für 2 von 5 Events ergibt real_loaded=3 und kein '0/5';
- fetch_noise schreibt aus einem gemockten 40-s-Segment 8 statt 10 Chunks.

`gwd eval run` ohne --real läuft offline durch.

<a id="p44"></a>
#### P4.4: Glitch Hunter entflechten: GUI, Quellen, Audio; Race-Condition und Thread-Plotting beheben; Modi

*Aufwand L · hängt ab von P3.3, P3.5, P4.1, P4.3 · wartet auf –*

Nutzt Default von D4 (Modi 'Training' und 'Challenge') und D10 (Schwelle ab P5.8 aus der Registry, bis dahin 0.46 als benannte Konstante).

gwd/apps/glitch_hunter/ wird in app.py (nur GUI), sources.py und audio.py aufgeteilt.

- sources.py: SimSource nutzt data.injection mit dem Szenario je Modus: 'Training' = TRAIN_V1 ist der Start-Modus, 'Challenge' = APP_V1 ist sichtbar als 'bewusst schwer/Extrapolation' gekennzeichnet. GwoscSource nutzt data.gwosc und conditioned_crop aus P4.3.
- audio.py: tempfile statt temp_sound.wav im CWD, nicht blockierend per Popen, explizite Plattformzweige winsound/aplay/paplay/afplay ohne nacktes except.
- Race-Condition: Der Download-Thread füllt nur lokale Variablen und übergibt (data, label, name, generation) per queue.Queue. Ein fig.canvas.new_timer im GUI-Thread übernimmt und zeichnet. Veraltete Generationen werden verworfen, die Buttons sind während des Ladens gesperrt, und aus dem Thread kommen keine matplotlib-Aufrufe mehr (update_plot_threadsafe → _render_current_sample).
- Das Modell ist optional und kommt über die Registry (--model). Ohne Modell ist 'KI fragen' deaktiviert statt sys.exit(1). TF wird lazy geladen.
- Bewertung: Der Slider-Bereich enthält die Schwelle (valmin ≤ threshold), das 'UNSICHER'-Band liegt relativ zur Schwelle, und die Wahrheitsbox wird bei Slider-Änderungen neu bewertet. Im Echtdaten-Modus wird die KI-Antwort ebenfalls bewertet, OOD-Events sind markiert.
- Der Spektrogramm-Titel richtet sich nach der Quelle; der Fallback nutzt SimSource statt reinem Rauschen.

Die ALLOWLIST-Einträge der App entfallen.

- **Dateien:** `src/gwd/apps/glitch_hunter/app.py`, `src/gwd/apps/glitch_hunter/sources.py`, `src/gwd/apps/glitch_hunter/audio.py`, `tests/apps/test_glitch_hunter.py`, `tests/test_architecture.py`
- **Behebt:** APP-5, APP-6, APP-7, APP-8, APP-9, APP-4, APP-19, APP-2, APP-10, apps:APPX-3, apps:APPX-4
- **Risiko:** Interaktive Backends (TkAgg unter WSLg) lassen sich in der CI nicht testen. Queue und Generationszähler sind deshalb so geschnitten, dass sie ohne GUI testbar sind; dazu kommt eine manuelle Abnahme beim Besitzer.
- **Prüfung:** `MPLBACKEND=Agg python -m pytest -m apps tests/apps/test_glitch_hunter.py -q` mit Dummy-Modell und gemocktem fetch_open_data (0,5 s Verzögerung):

- 'LIGO Zufall' und sofort 'KI fragen' zeigen keine 'WAHRHEIT: SIGNAL (SNR: 0.00)' für das alte Sample;
- ein Doppelklick startet keinen zweiten Download;
- ohne Modell gibt es kein SystemExit;
- der Modus 'Training' liefert Peak-SNR ≥ 1, 'Challenge' 0,5–2,0;
- play_sound legt nichts im CWD an.

`python -X importtime -c 'import gwd.apps.glitch_hunter.app' 2>&1 | grep -cE 'tensorflow|gwpy'` → 0. Manuell: `gwd app glitch-hunter` unter WSLg, dreimal schnell 'LIGO Zufall'.

<a id="p45"></a>
#### P4.5: Demo-Apps testbar: build_figure(), ax_noise-Bug, Finesse-Property, Emojis, GUI-Backend-Prüfung

*Aufwand M · hängt ab von P2.5, P0.6, P3.6 · wartet auf –*

Nutzt Default von D6 (Lernmodell).

Waveform-Explorer und Interferometer-Designer bekommen build_figure() → (fig, state), getrennt von main() mit plt.show().

Fixes:

- Die Slider-Achse heißt jetzt ax_s_noise, damit der Rausch-Plot wieder skaliert (APP-11).
- Die ungenutzte intensity entfällt.
- Emojis werden durch Text ersetzt.
- Die Finesse wird eine Property aus mirror_reflectivity in core/interferometer.py, ohne Physik im GUI-Callback. Das ist eine bewusste Änderung: Startwert 3140 statt 450, die GUI springt nicht mehr beim ersten Slider-Kontakt. Weil die Finesse über cavity_gain ins Shot-Rauschen eingeht, ändert sich Golden (e); das geschieht im Commit 'golden: Finesse als Property'.

Neu ist apps/_backend.py mit require_interactive_backend(). Bei nicht-interaktivem Backend gibt es eine verständliche Meldung (python3-tk bzw. WSLg) und Exit 2 statt stummem Exit 0. Die Smoke-Tests setzen GWD_HEADLESS_OK=1. `gwd env` zeigt das Backend.

- **Dateien:** `src/gwd/apps/waveform_explorer.py`, `src/gwd/apps/interferometer_designer.py`, `src/gwd/apps/_backend.py`, `src/gwd/core/interferometer.py`, `tests/apps/test_demo_apps.py`, `tests/golden/values.py`, `tests/smoke/conftest.py`
- **Behebt:** APP-11, APP-12, CORE-11, APP-15, APP-17, apps:APPX-2
- **Risiko:** Die Default-Kurve des Interferometers ändert sich wegen der Finesse. Das wird im PR erklärt, der Golden-Wert bewusst neu erzeugt.
- **Prüfung:** `MPLBACKEND=Agg python -m pytest -m apps tests/apps/test_demo_apps.py -W error::UserWarning -q`:

- nach Entfernung=1000 und Rauschen=0,1 passt das ylim des Rausch-Plots auf ±20 % zu den Daten;
- finesse ≈ π√R/(1−R) für R=0,999;
- set_val auf denselben Wert ändert finesse nicht;
- keine 'Glyph missing'-Warnung.

`MPLBACKEND=Agg gwd app waveform; echo $?` → 2 mit Hinweistext. `git log -1 --format=%s -- tests/golden/values.py` beginnt mit 'golden:'.

<a id="p46"></a>
#### P4.6: Validator und Laden: vollständige Integrität, Bereichsprüfung aus dem Szenario, keine Waisen

*Aufwand M · hängt ab von P4.1 · wartet auf –*

data/validate.py bekommt check_integrity(folder) als testbare Funktion mit Befundliste. Geprüft werden Existenz, Shape, dtype und NaN/Inf für ALLE in labels.csv gelisteten Dateien sowie die Konsistenz zum Manifest.

Die Verteilung wird gegen das Szenario aus manifest.json geprüft: min/max von SNR, Massen und t_merger liegen in den Szenario-Grenzen, die Klassen sind auf ±5 % balanciert. Das ersetzt die festen Median-Schwellen 5/8 und die Kategorien bis '>20' aus der alten U(5,25)-Logik; die Regeln gibt es nur einmal. Datensätze ohne Manifest gelten als 'v1-unbekannt' und werden nur auf Integrität geprüft.

Der Plot 'SNR vs. Distanz' entfällt, weil die Distanz bei normalize=True keine Wirkung hat. Die übrigen Report-Plots sind optional (--plots).

train ruft check_integrity als Vorbedingung auf und bricht bei fehlenden Dateien oder falscher Länge ab, statt still zu kürzen oder aufzufüllen.

Der Generator schreibt nur in leere Ordner oder mit --overwrite; dabei werden nur sample_*.npy und labels.csv im Zielordner entfernt, nie data/noise_background. Das Menü fragt bei 'Trainingsdaten generieren' nach und übergibt dann --overwrite (main.py darf input() nutzen, die CLI nicht).

- **Dateien:** `src/gwd/data/validate.py`, `src/gwd/data/generate.py`, `src/gwd/training/train.py`, `main.py`, `tests/data/test_validate.py`
- **Behebt:** DATA-22, DATA-4, data-training:LOAD-SILENT, DATA-21, data-training:SNR-DEF, DATA-18
- **Risiko:** Der strengere Check kann alte lokale Datensätze als fehlerhaft melden. Das ist gewollt; die Meldung nennt die betroffenen Dateien.
- **Prüfung:** `python -m pytest tests/data/test_validate.py -q`: Eine gelöschte Datei, ein NaN-Sample oder eine falsche Länge ergeben je einen Befund und rc ≠ 0; ein korrekter Datensatz mit n=64 liefert unabhängig vom Zufallsmedian rc 0. `gwd data generate --n 8 --out <nicht-leer>` ohne --overwrite → Abbruch mit rc ≠ 0; `python -m pytest -m slow -q` grün.

### P5: Physik-Kern und Messprotokoll ehrlich machen (entscheidungsarm, angekündigte Verhaltensänderungen)

**Ziel:** Reine Physik-Bugfixes, ein explizites Backend, rng statt globalem Zufall, gemessene SNR-Definitionen, ein ehrlicher Trainingssplit und ein reproduzierbares Messprotokoll mit festen Testsets, Schwelle pro Modell und Leaderboard v2. Die Szenarien bleiben V1. Golden-Werte ändern sich nur in angekündigten 'golden:'-Commits. Neue Modelle entstehen in neuen Dateien, Altes bleibt nutzbar.

**Fertig, wenn:** Kein stiller Fallback in der Pipeline; die Demo-Apps zeigen Newton sichtbar an. Der Kern hat keinen globalen RNG und kein print. Die Physik-Tests prüfen echte Größen. labels.csv enthält peak_snr und optimal_snr. Der 3-Wege-Split ist nach Rausch-Chunk gruppiert (mit Warnung für Altdatensätze). Es gibt feste, gehashte Testsets, eine Schwelle pro Modell und ein Leaderboard v2 mit Rohzahlen. Golden hat sich nur in angekündigten Commits geändert (time_axis, Rauschnormierung, float32). Offene xfails gibt es nur noch für CORE-2/3/8 (D1) und test_known_skew (V1-Altbestand).

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P5.1](#p51) | Physik-Tests mit echten Asserts (bekannte Fehler als strict xfail) | M | P2.6 | – |
| [P5.2](#p52) | Wellenform-Backend explizit, kein stiller Fallback in der Pipeline | M | P5.1, P4.1, P4.5 | – |
| [P5.3](#p53) | Kern-Details: time_axis, rng, t_merger-Prüfung, Taper/freq_out, Konstanten, toter Code | M | P5.2, P3.5 | – |
| [P5.4](#p54) | Rauschnormierung korrigieren; Interferometer als ehrliches Lernmodell | M | P5.1, P5.3, P4.5 | – |
| [P5.5](#p55) | SNR ehrlich benennen und messen: peak_snr plus optimal_snr | M | P5.4, P3.5 | – |
| [P5.6](#p56) | Training: gruppierter 3-Wege-Split, Seeds, float32, vollständige Metadaten | M | P3.7, P4.6, P4.1 | – |
| [P5.7](#p57) | Feste, gehashte Testsets; Metriken pro SNR-, Massen-Bin und Rauschart | M | P5.5, P4.2 | – |
| [P5.8](#p58) | Schwelle pro Modell bei fester FPR; Leaderboard v2 mit Rohzahlen | M | P5.7, P3.3, P4.4 | – |

<a id="p51"></a>
#### P5.1: Physik-Tests mit echten Asserts (bekannte Fehler als strict xfail)

*Aufwand M · hängt ab von P2.6 · wartet auf –*

tests/core/ wird angelegt; die drei bestehenden Testdateien ziehen per git mv dorthin und werden verschärft. Fester rng, beide Backends parametrisiert.

Geprüft wird:

- Peak bei t_merger ±1 ms;
- f_inst steigt bis zum Merger monoton;
- h ∝ 1/D bei normalize=False;
- std(white) = σ;
- Interferometer: ASD sinkt mit P und L, die Features senken die ASD;
- ein Smoke-Test für TimeDomainSimulator.

Die trivialen Asserts (np.any(adv != base), |mean|<0.5, max>0) werden ersetzt.

Bekannte Fehler laufen als xfail(strict=True) mit Issue-ID: Newton-Frequenz ×2,045 (CORE-2), Amplitude ×4 (CORE-3), fehlender ISCO-Cut (CORE-8), generate_realistic_chirp ohne Merger (CORE-1), stilles Klemmen von t_merger (CORE-15), std des farbigen Rauschens abhängig von n und fs (CORE-9). Sie dokumentieren die Fehler, bis die Korrektur sie grün macht; strict erzwingt dann das Entfernen der Markierung.

- **Dateien:** `tests/core/test_waveforms.py`, `tests/core/test_noise.py`, `tests/core/test_interferometer.py`, `tests/core/test_simulation.py`
- **Behebt:** CORE-13, CORE-12, CORE-9, CORE-2, CORE-3, CORE-1, CORE-8, CORE-15
- **Risiko:** Die Wellenformlänge hängt von der pycbc-Version ab. Deshalb werden nur physikalische Größen geprüft, keine Array-Längen.
- **Prüfung:** `python -m pytest tests/core -q -rx` grün, alle xfails mit Issue-ID aufgelistet; `python -m pytest tests/core/test_simulation.py -q` führt simulation.py erstmals aus.

<a id="p52"></a>
#### P5.2: Wellenform-Backend explizit, kein stiller Fallback in der Pipeline

*Aufwand M · hängt ab von P5.1, P4.1, P4.5 · wartet auf –*

generate_chirp(t, t_merger, params, backend=…) bekommt den Default aus config.WAVEFORM_BACKEND ('pycbc'); der Aufrufer übergibt ihn, core liest config nicht. pycbc wird lazy im Backend importiert. Fehlt pycbc oder wirft get_td_waveform einen Fehler, gibt es eine WaveformBackendError mit Installationshinweis (Linux/WSL) statt print plus Fallback. Das Modul-Global HAS_PYCBC entfällt; Tests und Golden setzen das Backend explizit. Die Pipeline (Datensatz, Testsets, Evaluation) nutzt nur 'pycbc'.

Nutzt Default von D1 für die Demo-Apps: Waveform-Explorer und TimeDomainSimulator wählen ohne pycbc ausdrücklich backend='newton' und zeigen im Fenster 'Newton-Näherung, physikalisch ungenau'. Das erhält den heutigen Stand unter Windows nativ (F3), ohne D1 vorwegzunehmen; bei D1=A entfällt der Zweig in P6.1.

- **Dateien:** `src/gwd/core/waveforms.py`, `src/gwd/core/simulation.py`, `src/gwd/data/injection.py`, `src/gwd/config.py`, `src/gwd/apps/waveform_explorer.py`, `tests/core/test_waveforms.py`, `tests/apps/test_demo_apps.py`, `tests/golden/test_golden.py`
- **Behebt:** CORE-4, hygiene:HYG-M1, CORE-17, core:MISS-1
- **Risiko:** Ohne pycbc (Windows nativ) lassen sich keine Datensätze mehr erzeugen. Das ist gewollt und steht im README; die Demo-Apps laufen sichtbar gekennzeichnet weiter.
- **Prüfung:** `python -m pytest tests/core -q`: approximant='GibtEsNicht' erzeugt eine Exception statt eines Newton-Ergebnisses; sys.modules['pycbc']=None erzeugt WaveformBackendError mit Hinweistext. `python -m pytest tests/apps/test_demo_apps.py -q` mit blockiertem pycbc: Die Figur baut, der Hinweistext enthält 'Newton'. `python -X importtime -c 'import gwd.core.waveforms' 2>&1 | grep -c pycbc` → 0.

<a id="p53"></a>
#### P5.3: Kern-Details: time_axis, rng, t_merger-Prüfung, Taper/freq_out, Konstanten, toter Code

*Aufwand M · hängt ab von P5.2, P3.5 · wartet auf –*

Keine Besitzerentscheidung nötig, aber das Verhalten ändert sich; die Golden-Werte werden im Commit 'golden: time_axis/arange und Konstanten' bewusst aktualisiert.

- generate_realistic_chirp und sein Import werden gelöscht (defekt, nirgends aufgerufen, in der Historie erhalten).
- Liegt t_merger außerhalb von [t0, t_end], gibt es ValueError statt stillem Klemmen; der Default wird relativ zu t0 berechnet.
- core/timebase.time_axis(fs, duration) = np.arange(n)/fs ersetzt überall linspace (effektiv 4096 statt 4095,75 Hz).
- core/population.random_bbh_params(rng) bekommt einen ehrlichen Docstring (log-uniform statt 'O3'). Spins und Inklination werden nicht gewürfelt (V1 ist spinlos und face-on, dokumentiert); die Distanz wird als reine Metadatenspalte gekennzeichnet.
- rng-Parameter statt globalem np.random im ganzen Kern.
- freq_out ist außerhalb des Signals NaN; der Taper ist ein Tukey-Fenster fester Dauer.
- core/constants.py für c, G, M_SUN, MPC und λ; ungenutzte Variablen (M, self.c, '* 1.0 *') und der falsche Dateikopf fallen weg.

Weil die Injektion seit P3.5 zentral ist, werden die wenigen Aufrufer direkt umgestellt.

- **Dateien:** `src/gwd/core/waveforms.py`, `src/gwd/core/population.py`, `src/gwd/core/timebase.py`, `src/gwd/core/constants.py`, `src/gwd/core/interferometer.py`, `src/gwd/core/simulation.py`, `src/gwd/data/injection.py`, `src/gwd/data/generate.py`, `tests/core/*`, `tests/golden/values.py`
- **Behebt:** CORE-1, CORE-14, CORE-15, CORE-16, CORE-18, CORE-19, DATA-19, DATA-18, core:MISS-3, DATA-5
- **Risiko:** Neu erzeugte Datensätze unterscheiden sich minimal von alten (etwa 0,006 % in der Abtastung); das Manifest hält es fest. Für bestehende Modelle ist das irrelevant.
- **Prüfung:** `git grep -nE 'generate_realistic_chirp|linspace[(]0' -- src` leer; `git grep -nE 'np[.]random[.](rand|randn|uniform|normal|randint|choice|seed)[(]' -- src/gwd/core` leer; `python -m pytest tests/core -q`: t_merger=10 bei 4-s-Fenster → ValueError; gleicher rng-Seed → bitgleich; die xfails für CORE-1 und CORE-15 sind entfernt. `python -m pytest -m 'slow or golden' -q` grün.

<a id="p54"></a>
#### P5.4: Rauschnormierung korrigieren; Interferometer als ehrliches Lernmodell

*Aufwand M · hängt ab von P5.1, P5.3, P4.5 · wartet auf –*

Nutzt Default von D6 (Lernmodell).

core/noise.py:

- white_noise(n, sigma, rng);
- colored_noise_from_psd(n, fs, psd, rng) mit korrekter einseitiger Normierung: Die std entspricht der Vorgabe, unabhängig von n und fs; der xfail für CORE-9 wird grün;
- das bisherige 1/f-Spektrum wird eine benannte PSD-Funktion statt 'seismisch'.

InterferometerModel:

- Docstring 'qualitatives Lernmodell' mit der Liste der Ad-hoc-Faktoren, Konstanten aus constants.py;
- reference_asd(f) über pycbc.psd.aLIGOZeroDetHighPower (lazy) ersetzt die erfundene 'LIGO Design'-Kurve der App;
- qualitative Tests: Monotonie in P und L, Richtung der Features, Größenordnung gegenüber der Referenz.

TimeDomainSimulator: time_axis, rng, Rauschen aus colored_noise_from_psd, die ungenutzte Intensität entfällt. Die Demo-Apps zeigen die neue Referenzkurve.

Golden (b) und (e) werden im Commit 'golden: Rauschnormierung' aktualisiert.

- **Dateien:** `src/gwd/core/noise.py`, `src/gwd/core/interferometer.py`, `src/gwd/core/simulation.py`, `src/gwd/apps/interferometer_designer.py`, `src/gwd/apps/waveform_explorer.py`, `tests/core/test_noise.py`, `tests/core/test_interferometer.py`, `tests/golden/values.py`
- **Behebt:** CORE-9, CORE-10, CORE-11, CORE-12, APP-13, CORE-18
- **Risiko:** Die Demo-Apps sehen danach anders aus, etwa mit anderen Achsenbereichen. Die App-Tests sichern die Funktion ab, der Golden-Commit erklärt die Änderung.
- **Prüfung:** `python -m pytest tests/core -q -rx` ohne xfail für CORE-9: Die Welch-PSD stimmt für (n, fs) ∈ {(16384, 4096), (16384, 1024), (2048, 4096)} auf ±15 %; reference_asd(100) ≈ 3,9e-24 ±20 % (importorskip pycbc). `python -m pytest -m apps -q` grün.

<a id="p55"></a>
#### P5.5: SNR ehrlich benennen und messen: peak_snr plus optimal_snr

*Aufwand M · hängt ab von P5.4, P3.5 · wartet auf –*

core/snr.py enthält:

- peak_snr: die heutige Definition, ehrlich als Peak-Amplitude/σ dokumentiert;
- optimal_snr_white(h, σ) = √Σh²/σ;
- optimal_snr_psd(h, psd, fs, f_low, f_high);
- scale_to_snr(h, target, kind, …).

scale_signal_to_snr wandert hierher.

Messvorschrift für V1-Samples:

- bei weißem Rauschen analytisch über σ;
- bei echtem, gefiltertem Rauschen gegen die Welch-PSD des tatsächlich verwendeten Chunks, beschränkt auf das Band BANDPASS (20–300 Hz), weil das Signal in V1 ungefiltert ist und ρ oberhalb des Bandes explodiert (evaluation:MISS-2).

Der Generator schreibt peak_snr (die Spalte 'snr' bleibt als Alias) und optimal_snr; die Evaluation berichtet beide. Steuergröße bleibt Peak-SNR (V1). Der Schritt ist rein additiv, macht den Unterschied messbar und liefert die Datengrundlage für D2.

- **Dateien:** `src/gwd/core/snr.py`, `src/gwd/data/injection.py`, `src/gwd/data/generate.py`, `src/gwd/data/validate.py`, `src/gwd/evaluation/evaluate.py`, `tests/core/test_snr.py`
- **Behebt:** data-training:SNR-DEF, evaluation:MISS-2, EVAL-3, CORE-6
- **Risiko:** Normierungskonventionen (einseitig/zweiseitig, df). Abgesichert durch den Abgleich mit pycbc.filter.sigma und einen Test gegen eine bekannte farbige PSD.
- **Prüfung:** `python -m pytest tests/core/test_snr.py -q`:

- measure(scale_to_snr(h, x)) == x je Definition;
- ρ_opt/peak für 30+30 in weißem Rauschen ≈ 13,5 ±10 % (reproduziert den Befund);
- optimal_snr_psd stimmt mit pycbc.filter.sigma auf ±1 % überein;
- für farbiges Rauschen aus bekannter PSD liegt die Welch-Variante auf ±10 % am analytischen Wert.

Im Smoke-Test hat labels.csv die Spalten peak_snr und optimal_snr mit Werten > 0 für Signale.

<a id="p56"></a>
#### P5.6: Training: gruppierter 3-Wege-Split, Seeds, float32, vollständige Metadaten

*Aufwand M · hängt ab von P3.7, P4.6, P4.1 · wartet auf –*

- Split: stratifiziert train/val/test 70/15/15, gruppiert nach noise_file (GroupShuffleSplit), damit keine Echtrausch-Realisierung in zwei Mengen landet. Fehlt die Spalte (Altdatensatz des Besitzers ohne die P4.1-Spalten), gibt es die Warnung 'ungruppiert, neu generieren empfohlen' und einen stratifizierten Split; die Modell-JSON vermerkt split_grouped=false.
- val steuert EarlyStopping, Checkpoint und ReduceLR; test wird genau einmal am Ende ausgewertet; val_accuracy und test_accuracy sind getrennte Felder.
- Seeds kommen aus --seed (rng, tf.keras.utils.set_random_seed). Mit --deterministic gilt zusätzlich tf.config.experimental.enable_op_determinism(). Die Seeds werden gespeichert.
- Laden in ein vorallokiertes float32-Array. Der Generator schreibt ab jetzt float32 (Commit 'golden: float32', ändert Golden (d)); alte float64-Datensätze bleiben lesbar. Archivmodelle optional mit include_optimizer=False.

Keine Besitzerentscheidung nötig: Das Leaderboard nutzt test_set_accuracy nicht (Prüfer-Korrektur DATA-2), alte Modelle bleiben nutzbar, neue Modelle sind neue Dateien.

- **Dateien:** `src/gwd/training/train.py`, `src/gwd/data/generate.py`, `src/gwd/config.py`, `src/gwd/cli.py`, `tests/training/test_split.py`, `tests/golden/values.py`
- **Behebt:** DATA-2, EVAL-25, DATA-5, DATA-20, DATA-21, evaluation:MISS-4, hygiene:HYG-M2
- **Risiko:** Der zusätzliche Val-Split kostet Trainingsdaten; bei 10 000 Samples ist das unkritisch. Volle Determinismus ist auf der GPU nicht garantiert, deshalb wird nur auf der CPU geprüft.
- **Prüfung:** `python -m pytest tests/training/test_split.py -q`: Klassenbalance je Split ±2 %, keine noise_file-Gruppe in zwei Splits, ohne Spalte Warnung und split_grouped=false. Determinismus auf kleinem Datensatz: `export GWD_HOME=$(mktemp -d); gwd data generate --n 64 --seed 1 --out $GWD_HOME/d`, danach zweimal `CUDA_VISIBLE_DEVICES='' gwd train --data $GWD_HOME/d --epochs 1 --seed 1 --deterministic` → val_loss-Differenz < 1e-6. Die Modell-JSON enthält seed, dataset_hash, val_accuracy und test_accuracy; eine neue Sample-Datei hat 65 536 Byte Nutzdaten.

<a id="p57"></a>
#### P5.7: Feste, gehashte Testsets; Metriken pro SNR-, Massen-Bin und Rauschart

*Aufwand M · hängt ab von P5.5, P4.2 · wartet auf –*

Nutzt Default von D4 (indist- und stress-Set) und D9 (Manifeste lokal).

evaluation/testsets.py: build(scenario, n, seed) erzeugt einmal data/eval/<szenario>_<seed>_<n>.npz mit SHA256. Das Manifest (Hash, Szenario, Seed, Parameter je Sample inkl. Massen, t_merger, peak_snr, optimal_snr, noise_source) liegt unter results/testsets/; unter data/ ließe es sich bei D9=B nicht versionieren.

Zwei Teile:

- calib_bg: nur Rauschen, ausschließlich für die Schwellen-Kalibrierung;
- test: balanciert, für ROC/AUC, FPR-Kontrolle und die SNR-Effizienzkurve aus seinen Signal-Samples.

Der Seed-Strom ist disjunkt zu allen Trainingsseeds.

Standard-Sets:

- indist_v1: TRAIN_V1-Verteilung inklusive Rauschmischung;
- stress_v1: = EVAL_V1, vergleichbar mit alten Zahlen.

Metriken werden pro Set, SNR-Bin (peak und optimal), Massen-Bin und noise_source berichtet (EVAL-2); Bins unter dem Trainingsminimum heißen 'Extrapolation'. Neue Befehle `gwd eval build-testset` und `gwd eval run --testset`.

Dokumentierte Grenzen:

- Sobald Echtrauschen beteiligt ist, ist der Hash nur auf dem Rechner des Besitzers reproduzierbar (lokale Chunks). Tests und CI bauen die Sets mit leerem Rauschordner, also nur mit Gauß-Rauschen.
- Echte Chunks können sich in V1 mit Trainings-Chunks überschneiden; das behebt erst P6.3.

- **Dateien:** `src/gwd/evaluation/testsets.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/evaluation/thresholds.py`, `src/gwd/evaluation/compare.py`, `src/gwd/config.py`, `src/gwd/cli.py`, `tests/evaluation/test_testsets.py`
- **Behebt:** core:MISS-3, EVAL-10, EVAL-1, EVAL-2, EVAL-3, EVAL-20, hygiene:HYG-M2
- **Risiko:** Die Cache-Dateien veralten, wenn sich ein Szenario ändert. Dateiname und Hash enthalten deshalb Szenario-Name und -Version; der Hash steht im Leaderboard.
- **Prüfung:** Zweimal `gwd eval build-testset --scenario indist_v1 --seed 7 --n 64` mit leerem Rauschordner → gleicher Hash und Cache-Treffer; ein manipuliertes Array führt beim Laden zu ValueError; `python -m pytest tests/evaluation/test_testsets.py -q` (alle SNR-Bins besetzt, calib_bg enthält nur Label 0, Manifest unter results/testsets/); zweimal `gwd eval run --testset stress_v1 --no-real` → identische ROC_AUC.

<a id="p58"></a>
#### P5.8: Schwelle pro Modell bei fester FPR; Leaderboard v2 mit Rohzahlen

*Aufwand M · hängt ab von P5.7, P3.3, P4.4 · wartet auf –*

Nutzt Default von D10 (FPR 1 %) und D11 (englische Spaltennamen).

Schwellen:

- `gwd eval thresholds --model X --save` bestimmt die Schwelle auf calib_bg (nie auf test) bei fester FPR. threshold, threshold_fpr, Strategie und Testset-Hash landen über registry.set_threshold in den Modell-Metadaten.
- evaluate, compare und der Glitch Hunter lesen sie über registry.get_threshold (Fallback 0,5 mit Warnung); die harten Werte 0.15/0.46 entfallen.
- Berichtet werden TPR@FPR mit tatsächlich erreichter FPR und n sowie SNR50/90 bei dieser Schwelle (Interpolation), immer mit FPR daneben. Gerankt wird nach ROC_AUC bzw. TPR@FPR, nie nach SNR50.

Leaderboard v2:

- neue Datei results/leaderboard_v2.csv mit Rohzahlen (float/int/leer) und schema_version=2;
- Spalten u. a. model, date, testset_hash, eval_scenario, backend, noise_source, roc_auc, ap, threshold, tpr_at_fpr, fpr_achieved, snr50, snr90, real_loaded, real_found;
- formatiert wird nur in show().

Die alte results/model_leaderboard.csv wird nie mehr beschrieben und bleibt als Archiv liegen (kein --legacy-Modus). Physik_Gap, BASELINE_FILE und der Parameter baseline_file entfallen; der Moratoriums-Test aus P0.4 geht im v2-Test auf. Das v2-Schema kommt in docs/evaluation.md (bzw. in P7.1, falls es die Datei noch nicht gibt). Für Altmodelle steht im README die Schleife `gwd eval thresholds --model <m> --save`.

- **Dateien:** `src/gwd/evaluation/thresholds.py`, `src/gwd/evaluation/leaderboard.py`, `src/gwd/evaluation/metrics.py`, `src/gwd/registry.py`, `src/gwd/evaluation/evaluate.py`, `src/gwd/evaluation/compare.py`, `src/gwd/apps/glitch_hunter/app.py`, `src/gwd/config.py`, `tests/evaluation/test_leaderboard.py`, `tests/evaluation/test_thresholds.py`, `README.md`, `docs/evaluation.md`
- **Behebt:** EVAL-15, APP-3, APP-4, hygiene:HYG-M3, evaluation:MISS-1, evaluation:MISS-5, EVAL-7, EVAL-8, EVAL-22, HYG-4
- **Risiko:** Ohne gespeicherte Schwelle sind alte Modelle nicht vergleichbar. Deshalb läuft `gwd eval thresholds --save` einmal für alle Altmodelle. Die alte CSV bleibt unverändert.
- **Prüfung:** `git grep -nE 'THRESHOLD *= *0[.]|detection_threshold *= *0[.]' -- src` leer. `python -m pytest tests/evaluation/test_thresholds.py tests/evaluation/test_leaderboard.py -q`:

- die Strategie trifft auf synthetischen Scores FPR ≤ 1 %;
- ein untrainiertes Modell mit TPR=FPR=1 wird nicht gut gerankt (Regressionstest MISS-1);
- ein Test setzt registry.get_threshold auf 0.123 und prüft, dass evaluate, compare und der Glitch Hunter genau diesen Wert verwenden;
- in einem tmp-GWD_HOME gilt pd.read_csv(results/leaderboard_v2.csv)['roc_auc'].dtype.kind == 'f';
- eine vorhandene results/model_leaderboard.csv bleibt byte-identisch.

`git grep -nE 'physics_baseline|Physik_Gap|BASELINE_FILE' -- src` leer.

### P6: Fachliche Korrekturen als additive Szenario-Version V2

**Ziel:** Die Kernkorrektur 'erst injizieren, dann gemeinsam konditionieren' wird ohne Besitzerentscheidung als neue, additive Szenario-Version V2 mit simuliertem aLIGO-Rauschen gebaut; V1-Modelle und V1-Zahlen bleiben vergleichbar. Nur Echtrauschen-Pools, Newton-Pfad, Matched-Filter-Baseline und Neutraining warten auf den Besitzer. Neue Daten landen in neuen Ordnern, alte Modelle werden archiviert.

**Fertig, wenn:** V2 mit simuliertem Rauschen existiert und hat grüne Abnahmetests (Label-Invarianz, keine HF-Abkürzung, Konsistenz Train/Eval). D1, D3 (Echtrauschen), D5 und das Neutraining sind umgesetzt oder bewusst per Default geparkt. V1-Modelle, V1-Datensätze und das v1-Leaderboard sind archiviert, nicht gelöscht.

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P6.1](#p61) | Newton-Backend löschen oder korrigieren (D1) | M | P5.2, P5.3 | D1 |
| [P6.2](#p62) | Szenario V2: erst injizieren, dann gemeinsam konditionieren (simuliertes aLIGO-Rauschen) | L | P5.4, P5.5, P5.7, P4.3 | – |
| [P6.3](#p63) | Echtrauschen für V2: mehrere Rohsegmente, Cache, disjunkte Pools je Split (D3 = A oder C) | M | P6.2, P3.4, P4.1 | D3 |
| [P6.4](#p64) | Matched-Filter-Baseline neu schreiben (D5) | L | P6.2, P5.8 | D5 |
| [P6.5](#p65) | Neutraining und Neubewertung, Umstellung auf V2, alte Artefakte archivieren (Besitzer, GPU) | L | P6.2, P5.6, P5.8 | D2, D3 |

<a id="p61"></a>
#### P6.1: Newton-Backend löschen oder korrigieren (D1)

*Aufwand M · hängt ab von P5.2, P5.3 · wartet auf D1*

Variante A: Der Newton-Pfad wird samt Tests und dem Demo-App-Zweig aus P5.2 gelöscht (in der Historie erhalten). backend akzeptiert nur 'pycbc', pycbc wird Basisabhängigkeit, das README nennt Linux/macOS/WSL als Voraussetzung.

Variante B: Newton wird nach core/newtonian.py herausgelöst und korrigiert:

- Frequenzformel mit Faktor 1/π (heute um 2,045 zu hoch);
- Amplitude 1/r statt 4/r;
- Start bei f_lower, Abbruch bei f_ISCO, Taper relativ zum Fensterbeginn;
- Spins, Inklination oder Approximant führen zu ValueError;
- nur explizit wählbar, in Szenarien verboten.

Die xfails aus P5.1 werden grün, strict erzwingt das Entfernen der Marker. Die Golden-Werte des Newton-Pfads werden in einem 'golden:'-Commit neu erzeugt.

Ohne Antwort bleibt der Schritt geparkt; Newton ist dann nur in den Demo-Apps sichtbar gekennzeichnet erreichbar.

- **Dateien:** `src/gwd/core/waveforms.py`, `src/gwd/core/newtonian.py`, `src/gwd/apps/waveform_explorer.py`, `pyproject.toml`, `tests/core/test_waveforms.py`, `tests/golden/values.py`, `README.md`
- **Behebt:** CORE-2, CORE-3, CORE-8, core:MISS-1, CORE-4
- **Risiko:** Bei B kann ein subtiler physikalischer Fehler bleiben; die Lehrbuch-Tests gegen PyCBC begrenzen das. A schließt Windows nativ aus.
- **Prüfung:** A: `git grep -ni newton -- src` leer und `python -m pytest -q` grün. B: `python -m pytest tests/core -q -rx` zeigt keine xfails für CORE-2/3/8 mehr; freq_out und die Nulldurchgangsfrequenz stimmen bei τ=0,2–1 s auf 5 % überein; Hüllkurve/PyCBC bei τ=0,3 s liegt in [0,8; 1,2]; spin1z=0,5 erzeugt ValueError.

<a id="p62"></a>
#### P6.2: Szenario V2: erst injizieren, dann gemeinsam konditionieren (simuliertes aLIGO-Rauschen)

*Aufwand L · hängt ab von P5.4, P5.5, P5.7, P4.3 · wartet auf –*

Nutzt Default von D3 (B: simuliertes farbiges Rauschen) und D2 (V2 steuert über ρ_opt).

Der Schritt ist additiv: Neue Szenarien TRAIN_V2, INDIST_V2, STRESS_V2 und APP_V2 kommen nach data/scenarios.py. V1 bleibt unverändert und Default von `gwd data generate`, bis der Besitzer in P6.5 umstellt.

make_sample im V2-Ablauf:

(1) farbiges Rauschen der Länge duration+2·pad aus der aLIGO-PSD (colored_noise_from_psd, P5.4);

(2) physikalisches h(t) dazu, skaliert per scale_to_snr mit snr_kind aus dem Szenario (ρ_opt gegen dieselbe PSD bzw. peak);

(3) Signal + Rauschen GEMEINSAM durch conditioned_crop (P4.3), dann normalize.

Damit gibt es keine ungefilterten Chirp-Anteile über 300 Hz mehr als Abkürzung fürs CNN, und das Label bedeutet unabhängig von der Rauschart dasselbe.

Weitere Punkte:

- Das Eval-Raster deckt den Trainingsbereich in Bins ab, dazu ein explizites Extrapolations-Bin.
- Die Testsets indist_v2 und stress_v2 entstehen über P5.7 und speichern die Parameter je Sample (für P6.4).
- Neuer Test test_v2_consistent: TRAIN_V2 und INDIST_V2 sind bis auf Seed und n identisch verteilt. test_known_skew bleibt als Dokumentation des V1-Altbestands xfail.
- Der Glitch Hunter bekommt einen Umschalter V1/V2; Default bleibt V1 bis P6.5.

Echtrauschen als zweite V2-Domäne folgt in P6.3, nur nach D3.

- **Dateien:** `src/gwd/data/scenarios.py`, `src/gwd/data/injection.py`, `src/gwd/data/preprocessing.py`, `src/gwd/evaluation/testsets.py`, `src/gwd/apps/glitch_hunter/sources.py`, `tests/data/test_scenario_v2.py`, `docs/architecture.md`
- **Behebt:** DATA-12, apps:APPX-1, evaluation:MISS-2, data-training:SNR-DEF, data-training:NOISE-SKEW, DATA-1, EVAL-1, EVAL-2, APP-2, CORE-7
- **Risiko:** Normierungsfehler bei PSD oder ρ_opt würden eine neue, subtilere Verzerrung erzeugen; dagegen stehen die Abnahmetests. Weil V2 neu ist und V1 Default bleibt, ändert der Schritt nichts an bestehenden Modellen oder Zahlen.
- **Prüfung:** `python -m pytest tests/data/test_scenario_v2.py -q`:

(a) gleicher Seed → bitgleiches Sample;

(b) Label-Invarianz: gemessenes ρ_opt/Ziel in [0,9; 1,1] für weißes und farbiges Rauschen bei ρ=8;

(c) keine HF-Abkürzung: Signalenergie oberhalb 300 Hz im konditionierten Sample < 1 %;

(d) test_v2_consistent grün.

`export GWD_HOME=$(mktemp -d); gwd data generate --scenario train_v2 --n 64 --seed 1 --out $GWD_HOME/v2 && gwd data validate --data $GWD_HOME/v2` → Exit 0; `python -m pytest -m golden -q` grün (V1 unverändert).

<a id="p63"></a>
#### P6.3: Echtrauschen für V2: mehrere Rohsegmente, Cache, disjunkte Pools je Split (D3 = A oder C)

*Aufwand M · hängt ab von P6.2, P3.4, P4.1 · wartet auf D3*

Nur bei D3 = A oder C.

- `gwd data fetch-noise --raw` speichert ROHE, längere Segmente aus mehreren konfigurierbaren GPS-Fenstern (optional mehrere Detektoren) in den NEUEN Ordner data/noise_raw/. data/noise_background/ des Besitzers bleibt unberührt.
- Segmente mit NaN oder Lücken werden verworfen und protokolliert, die Randbereiche jedes Downloads gesperrt.
- Die Segmentpools für train, val und test sind disjunkt; die Zuordnung steht im Manifest.
- Statt fester Chunks gibt es zufällige Ausschnitte der Länge duration+2·pad; die segment_id kommt in labels.csv.

Darauf baut die zweite V2-Domäne auf (TRAIN_V2_REAL, INDIST_V2_REAL) mit demselben make_sample-Ablauf wie in P6.2: Rohsegment statt simuliertem Rauschen, ρ_opt gegen die Welch-PSD des Segments. Die Domäne wird getrennt berichtet.

- **Dateien:** `src/gwd/data/fetch_noise.py`, `src/gwd/data/gwosc.py`, `src/gwd/data/injection.py`, `src/gwd/data/scenarios.py`, `tests/data/test_noise_pools.py`
- **Behebt:** DATA-11, DATA-25, evaluation:MISS-4, DATA-6
- **Risiko:** Netz und Proxy; GWOSC kann zeitweise nicht erreichbar sein. Gegenmittel: Cache, Offline-Fixtures in Tests und ein neuer Ordner, damit nichts überschrieben wird.
- **Prüfung:** `python -m pytest tests/data/test_noise_pools.py -q -m 'not network'`: Mit Fixture-TimeSeries überlappen die Pools train/test in GPS nicht, ein NaN-Segment wird verworfen, keine segment_id liegt in zwei Splits. `GWD_OFFLINE=1 gwd data fetch-noise --raw; test $? -ne 0`; beim Besitzer entspricht `ls data/noise_background | wc -l` dem Inventar aus P0.1.

<a id="p64"></a>
#### P6.4: Matched-Filter-Baseline neu schreiben (D5)

*Aufwand L · hängt ab von P6.2, P5.8 · wartet auf D5*

Nur bei D5=A. evaluation/baseline_mf.py wird NEU geschrieben. Die alte evaluate_classic.py (45b9202^) dient nur als Referenz: Sie hat einen Template-Bug (match 0,07–0,14), CWD-Pfade, input() und ein anderes Rauschmodell; ein git checkout würde sofort eine falsche Baseline erzeugen.

- Das Template wird um den Merger zugeschnitten oder per get_fd_waveform erzeugt, mit derselben PSD und Konditionierung wie V2.
- Die Baseline läuft auf denselben Testset-Dateien (calib_bg/test) wie das CNN; die Templates entstehen aus den im Testset-Manifest gespeicherten Parametern.
- Score: max|ρ(t)| im erlaubten Merger-Fenster.
- Modi: 'oracle' (wahres Template, physikalische Obergrenze) und optional eine kleine Template-Bank.
- Verglichen wird per ROC bzw. TPR bei gleicher FPR; Leaderboard-v2-Zeilen tragen model_type='matched_filter'. Ein Abstand wird nur bei identischem testset_hash und gleicher FPR ausgewiesen.
- Kein TensorFlow. Befehl: `gwd eval baseline`.

Bei D5=B ist nichts zu tun außer einem Satz in docs/evaluation.md (Historie 45b9202^, bekannter Bug).

- **Dateien:** `src/gwd/evaluation/baseline_mf.py`, `src/gwd/evaluation/leaderboard.py`, `src/gwd/cli.py`, `docs/evaluation.md`, `tests/evaluation/test_baseline_mf.py`
- **Behebt:** EVAL-6, EVAL-7, EVAL-8, HYG-4, DATA-16
- **Risiko:** Physikalisch anspruchsvoll (PyCBC-Konventionen, Längen). Abgesichert durch Match- und Erwartungswert-Tests.
- **Prüfung:** `python -m pytest tests/evaluation/test_baseline_mf.py -q`:

- match(Signal, Template) > 0,95 für 15/15, 30/30 und 50/50;
- der Oracle-Score liegt bei ρ=10 in Gauß-Rauschen im Mittel in [9,5; 11];
- 'tensorflow' ist nicht in sys.modules.

`gwd eval baseline --testset indist_v2 --mode oracle` schreibt eine v2-Zeile.

<a id="p65"></a>
#### P6.5: Neutraining und Neubewertung, Umstellung auf V2, alte Artefakte archivieren (Besitzer, GPU)

*Aufwand L · hängt ab von P6.2, P5.6, P5.8 · wartet auf D2, D3*

Beim Besitzer auf der GPU, nach Rückfrage zu D2 und D3. Ohne Antwort: V2 mit ρ_opt und simuliertem Rauschen; bei D3 ∈ {A, C} zusätzlich erst nach P6.3. Reihenfolge:

1. ggf. `gwd data fetch-noise --raw`
2. `gwd data generate --scenario train_v2 --seed … --out data/training_v2`
3. `gwd data validate --data data/training_v2`
4. `gwd train --data data/training_v2` mit 2–3 Seeds; die Streuung zeigt, welche Modellunterschiede überhaupt bedeutsam sind
5. `gwd eval thresholds --save`
6. `gwd eval run` auf indist_v2, stress_v2 und stress_v1 (Vergleich mit V1-Modellen)
7. `gwd eval compare`

Danach werden config.DEFAULT_SCENARIO und der Start-Modus des Glitch Hunters auf V2 gestellt. Alte Modelle kommen per mv nach models_registry/archive_v1/ (nicht löschen) und bleiben per --model nutzbar; der alte Datensatz bleibt. docs/evaluation.md bekommt eine Tabelle mit gemessenen Zahlen (Testset-Hash, Seeds, n) statt erfundener 'erwarteter' Werte. Der README-Statushinweis aus P0.4 wird durch einen Verweis darauf ersetzt.

- **Dateien:** `models_registry/ (lokal)`, `results/ (lokal)`, `src/gwd/config.py`, `src/gwd/apps/glitch_hunter/sources.py`, `docs/evaluation.md`, `README.md`
- **Behebt:** DATA-2, EVAL-1, EVAL-2, APP-2, HYG-14, evaluation:MISS-1
- **Risiko:** Rechenzeit und GPU-Umgebung. Das Ergebnis kann ernüchternd sein; gerade das ist ehrlich und wird dokumentiert. Alte Modelle bleiben im Archiv nutzbar.
- **Prüfung:** `gwd eval leaderboard` zeigt ≥ 2 V2-Zeilen mit gleichem testset_hash und Backend 'pycbc'; `ls models_registry/archive_v1/*.keras | wc -l` entspricht der Zahl alter Modelle laut Inventar P0.1; `grep -c 'nur eingeschränkt aussagekräftig' README.md` → 0; `python -c 'import gwd.config as c; print(c.DEFAULT_SCENARIO)'` → train_v2.

### P7: Doku und Abschluss

**Ziel:** Die Doku beschreibt nur Existierendes und ist per Test an die CLI gebunden. Sprachregel, Lizenz und optionales Lockfile sind geklärt. Eine Abschlussprüfung belegt, dass Zielbild und Ist-Zustand übereinstimmen.

**Fertig, wenn:** Das README ist die einzige Einstiegsdoku, docs/ enthält genau architecture.md und evaluation.md, und der Doku-Befehlstest ist grün. Die ALLOWLIST ist leer, der E2E-Test grün. Lizenz und Lockfile entsprechen D12 bzw. D7.

| Schritt | Titel | Aufwand | hängt ab von | Entscheidung |
|---|---|---|---|---|
| [P7.1](#p71) | README neu, docs/evaluation.md, Sprachregel, Doku-Befehlstest | M | P3.6 | – |
| [P7.2](#p72) | Lizenz festlegen (D12) | S | P7.1 | D12 ✔ (LICENSE in PR #1; nur pyproject offen) |
| [P7.3](#p73) | Lockfile für reproduzierbare Umgebung (D7) | S | P2.2 | D7 ✔ |
| [P7.4](#p74) | Abschlussprüfung: Ende-zu-Ende, leere ALLOWLIST, Schritt-Checkliste | M | P7.1, P5.8, P4.4, P4.5, P5.4, P5.6 | – |

<a id="p71"></a>
#### P7.1: README neu, docs/evaluation.md, Sprachregel, Doku-Befehlstest

*Aufwand M · hängt ab von P3.6 · wartet auf –*

Nutzt Default von D11. Die erste Fassung entsteht direkt nach P3.6; danach zieht jeder Schritt die Doku nach, der Doku-Test erzwingt das.

README:

- Zweck und Komponentenbild (core → data → training → evaluation; registry; apps; cli);
- Plattform Linux/WSL (pycbc);
- Installation je Extra inklusive GUI-Backend (python3-tk/WSLg), optional tensorflow[and-cuda];
- typischer Ablauf als gwd-Befehle; die Apps;
- Tests und Marker;
- Datenablage, Altpfade und GWD_HOME;
- Verweis auf Szenarien und SNR-Definition in docs/architecture.md.

Die Regel 'nur per Pull Request' wird zu 'CI muss grün sein'.

`git mv docs/ROC_ANALYSIS_README.md docs/evaluation.md`, die erhaltenswerten Teile von ROC_INTEGRATION_OVERVIEW.md einarbeiten, dann git rm. docs/evaluation.md enthält Metriken, Plots, Strategien, den Schwellen-Workflow, das Leaderboard-Schema und den Baseline-Status nach D5, ohne unbelegte LIGO-Zahlen.

AGENTS.md bekommt die Sprachregel (D11 = B: alles Englisch, also Bezeichner, Spalten, Kommentare, Ausgaben und Doku). Sie wird nur beim Anfassen von Dateien umgesetzt; CLI-Hilfetexte und Menü folgen ihr.

Neu ist tests/test_docs_commands.py: Es extrahiert alle `gwd …`- und `python -m …`-Zeilen aus README und docs, prüft `--help` bzw. find_spec und alle relativen Links.

- **Dateien:** `README.md`, `docs/ROC_ANALYSIS_README.md → docs/evaluation.md`, `docs/ROC_INTEGRATION_OVERVIEW.md`, `AGENTS.md`, `src/gwd/cli.py`, `main.py`, `tests/test_docs_commands.py`
- **Behebt:** HYG-3, HYG-9, HYG-10, HYG-13, HYG-14, HYG-15, HYG-18, DATA-24, APP-16, apps:APPX-2, EVAL-8, HYG-6
- **Risiko:** Gering. Der Doku-Test in der CI verhindert, dass wieder 49 von 54 Befehlen ins Leere zeigen.
- **Prüfung:** `python -m pytest tests/test_docs_commands.py -q` grün; `git ls-files docs` → docs/architecture.md, docs/evaluation.md (bis P7.4 zusätzlich docs/aufraeumen/); `git grep -nE 'evaluate_classic|PYTHONPATH|exact version|requirements.txt|python src/' -- README.md docs` leer; `grep -n Sprache AGENTS.md` trifft den neuen Abschnitt.

<a id="p72"></a>
#### P7.2: Lizenz festlegen (D12)

*Aufwand S · hängt ab von P7.1 · wartet auf D12*

Nur mit ausdrücklicher Antwort auf D12:

- A: LICENSE (MIT mit Copyright-Halter und Jahr), Lizenzabschnitt im README, `license` in pyproject.toml.
- B: dasselbe mit der gewählten Lizenz.
- C: Die README-Zeile wird zu 'Alle Rechte vorbehalten'.

Ohne Antwort wird keine Lizenz erfunden, und die bestehende Zeile bleibt.

- **Dateien:** `LICENSE`, `README.md`, `pyproject.toml`
- **Behebt:** HYG-10, HYG-20
- **Risiko:** Keines.
- **Prüfung:** Bei A: `test -f LICENSE && grep -q MIT LICENSE && grep -qi license pyproject.toml`; bei C: `test ! -e LICENSE && grep -c MIT README.md` → 0.

<a id="p73"></a>
#### P7.3: Lockfile für reproduzierbare Umgebung (D7)

*Aufwand S · hängt ab von P2.2 · wartet auf – (D7 = B entschieden)*

D7 = B ist entschieden. Ausgangspunkt sind die Pins aus `~/gwd-env-pc.txt` (PC-Umgebung, mit der die Modelle trainiert wurden; siehe Abschnitt 5, Umgebung): Sie kommen als Untergrenzen bzw. Constraints in den Compile, damit das Lock nicht stillschweigend auf neuere numpy- oder pandas-Hauptversionen springt. `uv pip compile pyproject.toml --all-extras --python-version 3.12 -c constraints-pc.txt -o requirements-linux-py312.lock`. Beide CI-Jobs installieren daraus, das README beschreibt die Aktualisierung. Damit stimmt die Aussage 'reproduzierbar' tatsächlich, und die Golden-Toleranzen müssen nicht mehr Versionsdrift abfangen. Ein Lock mit TF und pycbc gilt nur für Linux; das steht im Dateinamen.

- **Dateien:** `requirements-linux-py312.lock`, `.github/workflows/ci.yml`, `README.md`
- **Behebt:** HYG-6, core:MISS-1
- **Risiko:** Gering.
- **Prüfung:** Frisches venv: `pip install -r requirements-linux-py312.lock && pip install -e . --no-deps && GWD_REQUIRE_PYCBC=1 python -m pytest -m 'not slow and not network' -q` grün; CI grün.

<a id="p74"></a>
#### P7.4: Abschlussprüfung: Ende-zu-Ende, leere ALLOWLIST, Schritt-Checkliste

*Aufwand M · hängt ab von P7.1, P5.8, P4.4, P4.5, P5.4, P5.6 · wartet auf –*

tests/test_e2e.py (Marker slow) läuft in einem tmp-GWD_HOME durch: gwd data generate 64 → validate → train 1 Epoche → eval build-testset → eval thresholds --save → eval run --no-real → leaderboard → Glitch-Hunter build_figure mit diesem Modell und der gespeicherten Schwelle.

Außerdem:

- Die ALLOWLIST in test_architecture.py ist leer, Übergangs-Aliase sind entfernt.
- Smoke-Fälle, die die E2E-Strecke vollständig abdeckt, entfallen.
- Die Schritt-Checkliste in docs/architecture.md ist vollständig; jeder offene Schritt ist an eine offene Entscheidung gebunden.
- Erlaubte xfails sind nur noch Newton (CORE-2/3/8, bis D1) und test_known_skew (V1-Altbestand).

Läuft P6 später, wird diese Prüfung wiederholt.

Zum Schluss wird docs/aufraeumen/ (dieser Plan und das Befund-Register) per `git rm -r` entfernt; die Historie behält ihn, docs/architecture.md enthält die abgehakte Checkliste.

- **Dateien:** `tests/test_e2e.py`, `tests/test_architecture.py`, `tests/smoke/*`, `docs/architecture.md`, `docs/aufraeumen/`
- **Behebt:** CORE-5, DATA-23, APP-18, EVAL-20, HYG-1
- **Risiko:** Gering.
- **Prüfung:** `python -m pytest -m slow tests/test_e2e.py -q` grün; `python -m pytest tests/test_architecture.py -q` grün mit leerer ALLOWLIST; `git grep -nE 'generate_astrophysical_chirp|generate_gaussian_noise' -- src tests` leer; `python -m pytest -q -rx` listet nur die genannten xfails; CI grün.; `git ls-files docs/aufraeumen | wc -l` → 0.

---

## 7. Risiken und Gegenmittel

- **Lokale, ungetrackte Daten des Besitzers gehen verloren: analysis/model_leaderboard.csv, data/noise_background, models_registry/, plots/ und die Altpfade vor f724856 (gw_noise_background/, gw_training_data/, ./model_leaderboard.csv, ./optimal_thresholds.json). Mögliche Ursachen: `rm -rf analysis`, `git clean -x`, ein Generator, der bestehende Ordner überschreibt, oder versehentliches Committen nach dem .gitignore-Umbau.**
  *Gegenmittel:* Backup inklusive Altpfade, Inventar und gepushter Tag pre-cleanup (P0.1), Regeln in AGENTS.md. Gelöscht wird nur per `git rm` mit einem Guard gegen den Tag (P1.3). Migrationen kopieren (P1.2). Die Binärendungen bleiben global ignoriert, Altpfade bekommen eigene Regeln, und P1.1 vergleicht `git status --ignored` vorher und nachher. Der Generator schreibt nur in leere Ordner oder mit --overwrite (P4.6). Neue Formate landen in neuen Ordnern, alte Modelle im Archiv.

- **Nach dem Umzug von config zeigen die Pfade woanders hin (src/gwd/ oder site-packages); Modelle und Daten scheinen zu fehlen oder werden doppelt angelegt.**
  *Gegenmittel:* Die Wurzel wird über GWD_HOME bzw. den Ordner mit pyproject.toml aufgelöst, und ein Charakterisierungstest vergleicht die absoluten Pfade mit dem Zustand vor dem Umzug (P2.3). Die Übergangs-Weiche hält alle Skripte bis P2.6 lauffähig; `gwd env` zeigt die aufgelösten Pfade.

- **Ein 'Refactor'-PR ändert heimlich Verhalten, z. B. durch eine andere RNG-Aufrufreihenfolge bei der Injektion (P3.5), ein vereinheitlichtes eps (P3.2) oder unsortierte Rausch-Chunks.**
  *Gegenmittel:* Golden mit exakten Kennwerten für die numpy-stabilen Pfade, ein Commit pro Aufrufer, Smoke nach jedem Commit, sortierte Chunk-Liste. Verhaltensänderungen (time_axis, Finesse, Rauschnormierung, float32, Puffer beim Feldtest) kommen nur in eigenen, angekündigten Schritten mit 'golden:'-Commit.

- **Golden-Tests werden wirkungslos: Sie werden nach dem Umgebungswechsel in P2.1 übersprungen, in der CI gar nicht ausgeführt oder bei Rot reflexhaft neu erzeugt.**
  *Gegenmittel:* Es gibt keinen Versionsschlüssel und keinen stillen Skip. Exakte Werte gibt es nur für numpy-stabile Pfade, PyCBC wird mit Toleranz geprüft, und GWD_REQUIRE_PYCBC=1 macht in CI und P2.1 aus einem Skip einen Fehler. Der erste Commit von P2.1 erzeugt die Werte im neuen venv auf unverändertem Code. Neu erzeugt wird nur in 'golden:'-Commits mit Begründung; das regelt AGENTS.md, und der Diff wird im Review gelesen. Optional ein Lockfile (D7=B).

- **Smoke- und E2E-Tests sind langsam oder flaky: TF ist nicht deterministisch, Netzwerkzugriffe scheitern, der alte Validator meldet bei kleinen Stichproben zufällig KRITISCH (DATA-4).**
  *Gegenmittel:* Es werden nur Strukturen geprüft, keine Metrikwerte. Feste Seeds, CPU erzwungen, Feldtest per Stub bzw. --no-real, Marker slow/apps/golden/network. Die CI führt langsame Tests nur per workflow_dispatch aus. Der Validator wird in P4.6 deterministisch.

- **Ein KI-Assistent kopiert wieder statt zu verschieben (c4e3dd8), vermischt Themen (c54acd5), committet Tool-Zustand oder schreibt geschönte Selbstberichte (.planning/walkthrough.md).**
  *Gegenmittel:* AGENTS.md (CLAUDE.md verweist darauf). Jeder Auftrag nennt die Schritt-ID und deren Prüfbefehl. Der Architektur-Test findet byte-identische Module, sys.path-Hacks und verbotene Importe; die ALLOWLIST darf nur schrumpfen. `git show -M --stat` muss Renames zeigen, Tool-Ordner sind ignoriert, die CI läuft auf jedem Push.

- **Die Neuinstallation per pyproject-Extras beschädigt die funktionierende WSL-GPU-Umgebung (TensorFlow/CUDA).**
  *Gegenmittel:* Zuerst ein frisches venv; das alte bleibt, bis alles grün ist (P2.1). tensorflow[and-cuda] steht nur als README-Hinweis; pip freeze aus P0.1 erlaubt die Rekonstruktion.

- **PyCBC fehlte bisher beim Besitzer (F1). Dann sind alle vorhandenen Modelle mit dem fehlerhaften Newton-Pfad entstanden, und ab P2.1 erzeugt die Pipeline andere Daten; auf Windows nativ bricht die Datenerzeugung nach P5.2 ab.**
  *Gegenmittel:* F1 wird in P0.1 erhoben und in docs/architecture.md festgehalten, P2.1 kündigt den Wechsel an. Die Plattformfrage entscheidet D1. P5.2 bricht mit Installationshinweis ab, statt still umzuschalten; die Demo-Apps laufen sichtbar gekennzeichnet weiter. `gwd env` dient als Diagnose.

- **Alte .keras-Modelle lassen sich nicht mehr laden, oder alte und neue Zahlen werden vermischt, etwa weil sich Echtdaten-Zahlen durch den Puffer-Fix (P4.3) ändern.**
  *Gegenmittel:* Die Modelle bestehen nur aus Standard-Keras-Schichten; das bestätigt die Ladeprobe in P2.4. Das Leaderboard v2 ist eine eigene Datei mit testset_hash und schema_version, die alte CSV wird nie mehr beschrieben. Legacy-Modelle sind in der Registry gekennzeichnet und bleiben per --model nutzbar.

- **V2 wird ohne ausdrückliche Besitzerentscheidung gebaut (Defaults D2=B, D3=B) und könnte eine neue, subtilere Verzerrung einführen oder nicht den Vorstellungen des Besitzers entsprechen.**
  *Gegenmittel:* V2 ist strikt additiv: neue Szenarien und Ordner, V1 bleibt Default bis zum Neutraining (P6.5), das erst nach Rückfrage beginnt. Abnahmetests für Label-Invarianz, HF-Abkürzung und Konsistenz Train/Eval; ρ_opt wird mit pycbc.filter.sigma abgeglichen. Eine andere Antwort auf D2/D3 kostet ein Szenario-Feld bzw. aktiviert P6.3.

- **Entscheidungsstau, oder der Umbau bleibt im Hobby-Tempo halb fertig liegen.**
  *Gegenmittel:* 29 von 48 Schritten sind entscheidungsfrei, 13 laufen mit dokumentiertem Default, nur 6 warten. Jeder Schritt hinterlässt ein lauffähiges Repo mit grüner CI. Der Minimalpfad P0–P3 plus P7.1 ist in sich geschlossen, und die Checkliste in docs/architecture.md zeigt jederzeit den Stand.

- **Ein langer Aufräum-Branch kollidiert mit eigener Weiterentwicklung des Besitzers, oder ein Schritt muss zurückgenommen werden.**
  *Gegenmittel:* Kleine PRs, sofort nach main gemergt. Während P1–P2 möglichst keine Feature-Arbeit (ein bis zwei Wochenenden). Jeder Schritt ist per `git revert` umkehrbar; für den Notfall gibt es den gepushten Tag pre-cleanup.

---

## 8. Bewusst weggelassen

- History-Rewrite (git filter-repo) für die .swarm-Blobs: Das Pack ist nur 138 KiB groß; ein Force-Push auf main bringt mehr Risiko als Nutzen. Aus dem Index nehmen plus .gitignore genügt (P1.1).
- Neue CNN-Architektur (DATA-26: 97,6 % der Parameter in einer Dense-Schicht) und Hyperparameter-Suche: Das ist ML-Forschung, kein Aufräumen, und erst nach einem ehrlichen Messprotokoll (P5) aussagekräftig. Die Architektur ist per count_params-Test fixiert (P3.7), damit sie später bewusst geändert wird.
- Native Windows-Unterstützung für PyCBC/lalsuite: Es gibt keine Wheels, der Weg ist WSL (D1). Die Demo-Apps laufen nativ nur mit sichtbar gekennzeichnetem Newton-Pfad weiter.
- Experiment-Tracking (MLflow, DVC, W&B), Hydra- oder TOML-Szenariodateien, Docker, pre-commit, mypy, import-linter, Coverage-Schwellen: zu viel Aufwand für ein Solo-Hobbyprojekt. Manifest, Modell-Metadaten-JSON, eingefrorene Szenario-Dataclasses mit Hash, ein ast-Architektur-Test und eine CI-Datei reichen.
- Statistik auf Forschungsniveau: Wilson- und Poisson-Intervalle, KS-Tests in Validator und Injektionstests, Bootstrap- oder logistische Fits für SNR50/90, SeedSequence zur Rekonstruktion einzelner Samples. Bereichsprüfungen, n neben jeder Rate, gespeicherte Testset-Parameter und 2–3 Seeds beim Neutraining genügen.
- Zwei parallel gepflegte Leaderboard-Schemata bzw. ein --legacy-Modus: Die alte CSV bleibt nur als Archivdatei liegen, das Leaderboard v2 beginnt frisch.
- Golden-Master mit Versionsschlüsseln und Skip-Logik: Er hätte gerade in P2–P3 nichts geschützt; stattdessen exakte Werte nur für numpy-stabile Pfade plus Toleranzen für PyCBC.
- Statustabelle über alle 138 Issue-IDs und eine in jedem PR gepflegte Status-Spalte: Die Zuordnung Issue → Schritt steht in diesem Plan; in docs/architecture.md genügt eine Schritt-Checkliste.
- CI-Extras wie Größenwächter, wöchentlicher Job oder eine Backend-Matrix: ein schneller Job ohne TensorFlow mit pycbc und ein manueller slow-Job reichen.
- scipy-Nachbau der gwpy-Filterkette: Abweichungsrisiko für Echtdaten-Zahlen ohne Nutzen; gwpy bleibt lazy in gwd.data.
- interferometer.py und simulation.py aus gwd.core nach gwd.apps verschieben (APP-14): eine Designfrage ohne Sicherheitsgewinn; gwd.core bleibt die Physik-Bibliothek, die Importregeln genügen.
- Apps in ein eigenes Repo bzw. Paket auslagern (D8, Option C): verdoppelt den Pflegeaufwand.
- Big-Bang-Übersetzung Deutsch/Englisch (HYG-18): Nur die Regel wird festgelegt (D11, P7.1) und beim Anfassen von Dateien umgesetzt.
- Belastbare FAR-Schätzung mit Time-Slides und mehreren Detektoren, Segmentwahl über Data-Quality-Flags, vollständige O3-Timeline für 'LIGO Zufall' (EVAL-21, DATA-25, APP-19): Forschungsniveau. P4.3 und P6.3 machen die Angaben nur ehrlich (k Fehlalarme in T Stunden, NaN-Prüfung, OOD-Kennzeichnung).
- Netzwerk-SNR, Antennenmuster F+/Fx (CORE-14), Spins, Inklination, Präzession und Higher Modes im Datensatz (Alternative zu DATA-18): eine spätere Szenario-Version. V1 und V2 sind spinlos und face-on und dokumentieren das.
- Kalibriertes Interferometer-Rauschmodell (D6, Option B): eine Neuentwicklung der Physik.
- HDF5-, TFRecord- oder tf.data-Pipeline: float32 plus optional memmap reicht für 10–20k Samples.
- Offline-Modus des Glitch Hunters mit gecachten Event-Snippets und Audio über sounddevice: neue Features. P4.4 macht nur den vorhandenen Weg robust; der Fallback nutzt die Simulation.
- Wiederherstellung von evaluate_model.py, generate_realistic_chirp und evaluate_classic.py im Originalzustand: tot oder fehlerhaft, über die Historie erreichbar.
- Historische Konfliktmarker in c4e3dd8 (HYG-12): kein Handlungsbedarf am Code; die CI (P2.2) verhindert eine Wiederholung.
- CONTRIBUTING.md, Branch-Protection, PyPI-Veröffentlichung, Sphinx- oder API-Doku: Für ein Solo-Projekt genügen README, zwei docs-Dateien, AGENTS.md und eine grüne CI.
- Weiterbetrieb oder Integration von ruflo/claude-flow im Repo: Der Zustand wird nur aus dem Index genommen und bleibt lokal; die Projektregeln für KI-Assistenten stehen in AGENTS.md.

---

## 9. Umsetzung mit mehreren Agenten

Ein Lead klärt die Fragen, mehrere Agenten arbeiten in getrennten Worktrees. Die Abhängigkeiten im Plan lassen das nur eingeschränkt zu, deshalb hier die ehrliche Rechnung.

### Wellen (aus `depends_on`)

| Welle | Schritte | parallel möglich? |
|---|---|---|
| 1–12 | P0.1 → P0.2/P1.1 → P0.3/P0.4/P1.6 → P0.5/P1.4 → P0.6/P1.2 → P1.3 → P1.5/P2.1 → P2.2 → P2.3/P7.3 → P2.4 → P2.5 → P2.6 | kaum; höchstens 2–3 kleine Schritte gleichzeitig |
| 13 | P3.1, P3.3, P5.1 | P5.1 ja; P3.1 und P3.3 berühren dieselben Dateien |
| 14–17 | P3.2 → P3.4 → P3.5 → P3.6 | nein |
| 18 | P3.7, P4.1, P4.2, P4.5, P7.1 | ja, bis zu 4 Agenten; P4.1 überschneidet sich mit P3.7 (`train.py`) und P4.2 (`evaluate.py`) |
| 19 | P4.3, P4.6, P5.2, P7.2 | ja |
| 20 | P4.4, P5.3, P5.6 | P4.4 parallel; P5.3 und P5.6 nacheinander (`generate.py`, Golden-Werte) |
| 21–25 | P5.4/P6.1 → P5.5 → P5.7 → P5.8/P6.2 → P6.3/P6.4/P6.5/P7.4 | teilweise |

**Folgerung:**

- P0 bis P3.6 (der Umzug) ist im Kern ein einziger Strang. Läuft er parallel, gibt es bei jedem `git mv` Merge-Konflikte. Dafür ist **ein** Agent besser, der Schritt für Schritt nach `main` merged.
- Ab P3.6 lohnen sich **3–4 Agenten**.
- Schritte, die dieselbe Datei ändern, gehören in dieselbe Welle **nacheinander**, auch wenn `depends_on` sie nicht verknüpft. Das gilt besonders für alles, was `tests/golden/values.py` ändert (P4.5, P5.3, P5.4, P5.6, P6.1).

### Fallstricke bei Worktrees

- **Lokale Daten fehlen im Worktree.** Ignorierte Ordner (`data/`, `models_registry/`, `plots/`, Leaderboard-CSV) gibt es nur im Haupt-Checkout. Die Tests laufen ohnehin in einem tmp-`GWD_HOME`. Die Prüfungen „beim Besitzer“ laufen aber im Haupt-Checkout und gehören dem Lead bzw. dem Besitzer: P0.1, die Migration in P1.2, der Guard in P1.3, die Ladeprobe alter Modelle in P2.4, die manuelle Abnahme in P4.4 und P6.5.
- **Kein gemeinsames venv nach P2.1.** `pip install -e .` bindet `gwd` an den Pfad *eines* Worktrees. Ein gemeinsames venv lädt dann still den Code eines anderen Agenten. Deshalb braucht jeder Worktree ein eigenes venv (mit `uv` und gemeinsamem Cache ist das billig) oder `PYTHONPATH=src` ohne editable Install.
- **Golden-Werte und `.gitignore`** sind gemeinsame Dateien. Änderungen daran nur in eigenen, nacheinander gemergten PRs.
- **Ein Schritt ist ein Task.** Der Task-Text besteht aus Schritt-ID, Dateiliste, Beschreibung und Prüfbefehl aus Abschnitt 6. Der Agent darf nur die genannten Dateien ändern und muss die Prüfung ausführen. Ein Task mit „wartet auf Dx“ geht erst nach der Antwort an einen Agenten.
- **Aufgaben des Leads:** Entscheidungsfragen, Merge-Reihenfolge, Prüfung von `golden:`-Commits und Abnahme am Haupt-Checkout.

