import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import LEADERBOARD_FILE

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils.evaluation import migrate_legacy_leaderboard


def main():
    migrate_legacy_leaderboard()
    if not os.path.exists(LEADERBOARD_FILE):
        print(f"No leaderboard yet: {LEADERBOARD_FILE} does not exist. Run src/evaluate_model_with_roc.py first.")
        return 0

    df = pd.read_csv(LEADERBOARD_FILE)
    # Display only: the values in the CSV are based on an invalid baseline
    df["Physik_Gap"] = "-"

    # 1. Create a temporary numeric column for sorting
    # 'errors="coerce"' turns "-" into NaN (Not a Number)
    df["sort_val"] = pd.to_numeric(df["ROC_AUC"], errors="coerce")

    # 2. Sort by the numeric values, best first
    # na_position='last' puts models that failed (returned "-") at the bottom
    df_sorted = df.sort_values(by="sort_val", ascending=False, na_position='last', kind="stable")

    # 3. Drop the helper column and print
    print(df_sorted.drop(columns=["sort_val"]).to_markdown(index=False))
    print("Physik_Gap hidden: values were based on an invalid baseline (EVAL-6).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
