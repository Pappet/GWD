import os
import glob
import json
import datetime
import pandas as pd  # type: ignore
from typing import Optional

def get_latest_model(models_dir: str) -> Optional[str]:
    """Sucht das neueste Modell im Registry-Ordner."""
    if not os.path.exists(models_dir):
        print(f"❌ Ordner '{models_dir}' nicht gefunden.")
        return None
    
    files = glob.glob(os.path.join(models_dir, "*.keras"))
    if not files:
        print("❌ Keine Modelle (.keras) gefunden.")
        return None
        
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def update_leaderboard_basic(model_name: str, sim_metrics: dict, real_metrics: dict, 
                             leaderboard_file: str, baseline_file: str):
    """Schreibt alle grundlegenden Ergebnisse in die CSV-Historie."""
    phys_gap = "-"
    try:
        with open(baseline_file, "r") as f:
            base_data = json.load(f)
            base_90 = base_data.get("SNR90", -1)
            
            if base_90 > 0 and sim_metrics.get('snr90'):
                gap = sim_metrics['snr90'] - base_90
                phys_gap = f"+{gap:.2f}"
    except FileNotFoundError:
        pass 

    entry = {
        "Model": model_name,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        
        "Sim_Accuracy": f"{sim_metrics['acc']:.1%}",
        "Sim_SNR50": f"{sim_metrics['snr50']:.2f}" if sim_metrics.get('snr50') else "-",
        "Sim_SNR90": f"{sim_metrics['snr90']:.2f}" if sim_metrics.get('snr90') else "-",
        
        "Physik_Gap": phys_gap,
        
        "Sim_FalseAlarm": f"{sim_metrics['far']:.2%}",
        
        "Real_Events_Found": f"{real_metrics['found']}/{real_metrics['total']}",
        "Real_Noise_FAR": real_metrics['far_text']
    }
    
    df_new = pd.DataFrame([entry])
    
    if os.path.exists(leaderboard_file):
        df_old = pd.read_csv(leaderboard_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(leaderboard_file, index=False)
    else:
        df_new.to_csv(leaderboard_file, index=False)
        
    print(f"\\n🏆 Leaderboard aktualisiert: {leaderboard_file}")
    if phys_gap != "-":
        print(f"   -> Abstand zur Physik-Grenze: {phys_gap} SNR-Einheiten")

def update_leaderboard_roc(model_name: str, sim_metrics: dict, real_metrics: dict,
                           leaderboard_file: str, baseline_file: str):
    """Schreibt erweiterte Metriken in die CSV."""
    phys_gap = "-"
    try:
        with open(baseline_file, "r") as f:
            base_data = json.load(f)
            base_90 = base_data.get("SNR90", -1)
            
            if base_90 > 0 and sim_metrics.get('snr90'):
                gap = sim_metrics['snr90'] - base_90
                phys_gap = f"+{gap:.2f}"
    except FileNotFoundError:
        pass

    entry = {
        "Model": model_name,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        
        "ROC_AUC": f"{sim_metrics['roc_auc']:.4f}",
        "Avg_Precision": f"{sim_metrics['avg_precision']:.4f}",
        
        "Sim_Accuracy": f"{sim_metrics['acc']:.1%}",
        "TPR": f"{sim_metrics['tpr']:.2%}",
        "FPR": f"{sim_metrics['fpr']:.2%}",
        
        "Sim_SNR50": f"{sim_metrics['snr50']:.2f}" if sim_metrics.get('snr50') else "-",
        "Sim_SNR90": f"{sim_metrics['snr90']:.2f}" if sim_metrics.get('snr90') else "-",
        "Physik_Gap": phys_gap,
        
        "Real_Events_Found": f"{real_metrics['found']}/{real_metrics['total']}",
        "Real_Noise_FAR": real_metrics['far_text']
    }
    
    df_new = pd.DataFrame([entry])
    
    if os.path.exists(leaderboard_file):
        df_old = pd.read_csv(leaderboard_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(leaderboard_file, index=False)
    else:
        df_new.to_csv(leaderboard_file, index=False)
        
    print(f"\\n🏆 Leaderboard aktualisiert: {leaderboard_file}")
    print(f"   -> ROC AUC: {sim_metrics['roc_auc']:.4f}")
    if phys_gap != "-":
        print(f"   -> Abstand zur Physik: {phys_gap} SNR")
