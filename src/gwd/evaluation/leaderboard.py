import datetime
import os
import shutil
import sys
import warnings

import pandas as pd  # type: ignore

from gwd.config import LEADERBOARD_FILE, LEGACY_LEADERBOARD_FILES


def migrate_legacy_leaderboard(leaderboard_file: str = LEADERBOARD_FILE) -> None:
    """Copy the old leaderboard to the configured path once. Nothing is deleted.

    Only the configured path migrates, so writing a leaderboard elsewhere never copies anything.
    """
    if os.path.abspath(leaderboard_file) != os.path.abspath(LEADERBOARD_FILE):
        return
    if os.path.exists(leaderboard_file):
        return

    analysis_file, root_file = LEGACY_LEADERBOARD_FILES
    if os.path.exists(analysis_file):
        os.makedirs(os.path.dirname(leaderboard_file), exist_ok=True)
        shutil.copy2(analysis_file, leaderboard_file)
        print(
            f"📋 Copied leaderboard from {analysis_file} to {leaderboard_file}. The original stays in place."
        )
    if os.path.exists(root_file):
        print(
            f"ℹ️ Old leaderboard at {root_file} was not copied, because it may use a different schema."
        )


def update_leaderboard_roc(
    model_name: str,
    sim_metrics: dict,
    real_metrics: dict,
    leaderboard_file: str,
    baseline_file: str,
):
    """Schreibt erweiterte Metriken in die CSV."""
    # baseline_file is unused until P5.8 removes it: the baseline is not valid.
    warnings.warn("Physics baseline currently invalid (EVAL-6, see D5)")
    migrate_legacy_leaderboard(leaderboard_file)

    entry = {
        "Model": model_name,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ROC_AUC": f"{sim_metrics['roc_auc']:.4f}",
        "Avg_Precision": f"{sim_metrics['avg_precision']:.4f}",
        "Sim_Accuracy": f"{sim_metrics['acc']:.1%}",
        "TPR": f"{sim_metrics['tpr']:.2%}",
        "FPR": f"{sim_metrics['fpr']:.2%}",
        "Sim_SNR50": f"{sim_metrics['snr50']:.2f}"
        if sim_metrics.get("snr50") is not None
        else "-",
        "Sim_SNR90": f"{sim_metrics['snr90']:.2f}"
        if sim_metrics.get("snr90") is not None
        else "-",
        "Physik_Gap": "-",
        "Real_Events_Found": f"{real_metrics['found']}/{real_metrics['total']}",
        "Real_Noise_FAR": real_metrics["far_text"],
    }

    df_new = pd.DataFrame([entry])

    os.makedirs(os.path.dirname(os.path.abspath(leaderboard_file)), exist_ok=True)

    if os.path.exists(leaderboard_file):
        df_old = pd.read_csv(leaderboard_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(leaderboard_file, index=False)
    else:
        df_new.to_csv(leaderboard_file, index=False)

    print(f"\n🏆 Leaderboard updated: {leaderboard_file}")
    print(f"   -> ROC AUC: {sim_metrics['roc_auc']:.4f}")


def show():
    """Print the leaderboard sorted by ROC_AUC, best first. Returns the exit code."""
    migrate_legacy_leaderboard()
    if not os.path.exists(LEADERBOARD_FILE):
        print(
            f"No leaderboard yet: {LEADERBOARD_FILE} does not exist. Run 'python -m gwd.evaluation.evaluate' first."
        )
        return 0

    df = pd.read_csv(LEADERBOARD_FILE)
    # Display only: the values in the CSV are based on an invalid baseline
    df["Physik_Gap"] = "-"

    # 1. Create a temporary numeric column for sorting
    # 'errors="coerce"' turns "-" into NaN (Not a Number)
    df["sort_val"] = pd.to_numeric(df["ROC_AUC"], errors="coerce")

    # 2. Sort by the numeric values, best first
    # na_position='last' puts models that failed (returned "-") at the bottom
    df_sorted = df.sort_values(
        by="sort_val", ascending=False, na_position="last", kind="stable"
    )

    # 3. Drop the helper column and print
    print(df_sorted.drop(columns=["sort_val"]).to_markdown(index=False))
    print("Physik_Gap hidden: values were based on an invalid baseline (EVAL-6).")
    return 0


if __name__ == "__main__":
    sys.exit(show())
