import os
import glob
import datetime
import warnings
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

def update_leaderboard_roc(model_name: str, sim_metrics: dict, real_metrics: dict,
                           leaderboard_file: str, baseline_file: str):
    """Schreibt erweiterte Metriken in die CSV."""
    # baseline_file is unused until P5.8 removes it: the baseline is not valid.
    warnings.warn("Physics baseline currently invalid (EVAL-6, see D5)")

    entry = {
        "Model": model_name,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        
        "ROC_AUC": f"{sim_metrics['roc_auc']:.4f}",
        "Avg_Precision": f"{sim_metrics['avg_precision']:.4f}",
        
        "Sim_Accuracy": f"{sim_metrics['acc']:.1%}",
        "TPR": f"{sim_metrics['tpr']:.2%}",
        "FPR": f"{sim_metrics['fpr']:.2%}",
        
        "Sim_SNR50": f"{sim_metrics['snr50']:.2f}" if sim_metrics.get('snr50') is not None else "-",
        "Sim_SNR90": f"{sim_metrics['snr90']:.2f}" if sim_metrics.get('snr90') is not None else "-",
        "Physik_Gap": "-",
        
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
        
    print(f"\n🏆 Leaderboard updated: {leaderboard_file}")
    print(f"   -> ROC AUC: {sim_metrics['roc_auc']:.4f}")
