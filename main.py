# Datei: GWD/main.py
import sys
import os
import argparse

# Pfadesetup damit Importe funktionieren
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_script(module_path):
    """Hilfsfunktion um Skripte sauber auszuführen"""
    print(f"\n🚀 Starte: {module_path}...\n" + "-"*50)
    # Wir nutzen os.system für vollständige Isolation der Skripte
    # python <script>
    ret = os.system(f"{sys.executable} {module_path}")
    if ret == 0:
        print("\n✅ Erfolgreich abgeschlossen.")
    else:
        print("\n❌ Fehler aufgetreten.")
    input("\n[Enter] drücken um zum Menü zurückzukehren...")

def main_menu():
    while True:
        # Screen clear (optional)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("==========================================")
        print("🌊 GRAVITATIONAL WAVE DETECTOR - HUB 🌊")
        print("==========================================")
        print("1. [DATA]   Echtes Rauschen laden (Fetch Real Noise)")
        print("2. [DATA]   Trainingsdaten generieren")
        print("3. [DATA]   Datensatz validieren")
        print("4. [TRAIN]  CNN Modell trainieren")
        print("5. [EVAL]   Modell evaluieren (ROC & Plots)")
        print("6. [EVAL]   Threshold Optimierung")
        print("7. [EVAL]   Modell-Vergleich (Compare)")
        print("8. [APP]    Glitch Hunter (Interaktiv)")
        print("9. [APP]    Waveform Simulator")
        print("0. Beenden")
        print("==========================================")
        
        choice = input("Wähle eine Option: ")
        
        if choice == '1': run_script("dataset/fetch_real_noise.py")
        elif choice == '2': run_script("dataset/generate_chirp_dataset.py")
        elif choice == '3': run_script("dataset/validate_dataset.py")
        elif choice == '4': run_script("dataset/train_cnn.py")
        elif choice == '5': run_script("analysis/evaluate_model_with_roc.py")
        elif choice == '6': run_script("analysis/optimize_threshold.py")
        elif choice == '7': run_script("analysis/compare_models_roc.py")
        elif choice == '8': run_script("apps/glitch_hunter.py")
        elif choice == '9': run_script("apps/gw_waveform_sim.py")
        elif choice == '0': sys.exit()
        else: print("Ungültige Eingabe!")

if __name__ == "__main__":
    main_menu()