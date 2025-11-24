import pandas as pd

df = pd.read_csv("model_leaderboard.csv")

# 1. Create a temporary numeric column for sorting
# 'errors="coerce"' turns "-" into NaN (Not a Number)
df["sort_val"] = pd.to_numeric(df["Sim_SNR50"], errors="coerce")

# 2. Sort by the numeric values
# na_position='last' puts models that failed (returned "-") at the bottom
df_sorted = df.sort_values(by="sort_val", ascending=True, na_position='last')

# 3. Drop the helper column and print
print(df_sorted.drop(columns=["sort_val"]).to_markdown(index=False))