# ==========================================
# Parallel Coordinates Plot (color by Risk)
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# 1) Load data

df = pd.read_csv(r"C:\Users\HP\Downloads\HDD_Lab_Customer-data - HDD_Lab_Customer-data.csv")

# 2) Pick at least 5 numerical variables (you can change these)
vars_5plus = [
    "Income",
    "Spending_Score",
    "Credit_Score",
    "Transactions_Value",
    "Debt",
    "Savings",
]  # 6 vars (>=5)

# Keep only columns that exist and are numeric
vars_5plus = [c for c in vars_5plus if c in df.columns]
vars_5plus = vars_5plus[:6]  # keep 5–6 for readability

# 3) Prepare plotting dataframe: selected variables + Risk_Class
plot_df = df[vars_5plus + ["Risk_Class"]].dropna().copy()

# Optional: sample if you have too many lines (helps readability)
# plot_df = plot_df.sample(300, random_state=42)

# 4) Normalize variables to 0–1 so axes are comparable
for c in vars_5plus:
    mn, mx = plot_df[c].min(), plot_df[c].max()
    if mx > mn:
        plot_df[c] = (plot_df[c] - mn) / (mx - mn)
    else:
        plot_df[c] = 0.0  # constant column safety

# Risk_Class should be treated like a category for coloring
plot_df["Risk_Class"] = plot_df["Risk_Class"].astype(str)

# 5) Plot
plt.figure(figsize=(10, 5))
parallel_coordinates(plot_df, class_column="Risk_Class", cols=vars_5plus, alpha=0.25)
plt.title("Parallel Coordinates Plot (normalized) colored by Risk_Class")
plt.ylabel("Normalized value (0 to 1)")
plt.tight_layout()
plt.show()
