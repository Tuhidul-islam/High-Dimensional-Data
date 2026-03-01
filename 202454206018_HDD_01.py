
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load data

df = pd.read_csv(r"C:\Users\HP\Downloads\HDD_Lab_Customer-data - HDD_Lab_Customer-data.csv")

# (a) Scatter plots
# (a)(i) Income vs Spending_Score
plt.figure(figsize=(7, 5))
plt.scatter(df["Income"], df["Spending_Score"], alpha=0.6)
plt.xlabel("Income")
plt.ylabel("Spending_Score")
plt.title("Income vs Spending_Score")
plt.tight_layout()
#plt.show()

# (a)(ii) Income vs Spending_Score colored by Risk_Class
plt.figure(figsize=(7, 5))
for rc, g in df.groupby("Risk_Class"):
    plt.scatter(g["Income"], g["Spending_Score"], alpha=0.6, label=f"Risk_Class {rc}")
plt.xlabel("Income")
plt.ylabel("Spending_Score")
plt.title("Income vs Spending_Score (colored by Risk_Class)")
plt.legend(title="Group", fontsize=8)
plt.tight_layout()
#plt.show()

# Optional: correlation check for (a)(i)
print("Corr(Income, Spending_Score) =", df["Income"].corr(df["Spending_Score"]))

# -------------------------
# (b) Distributions
# -------------------------

# (b)(i) Credit_Score distribution (hist + boxplot)
plt.figure(figsize=(7, 5))
plt.hist(df["Credit_Score"].dropna(), bins=30)
plt.xlabel("Credit_Score")
plt.ylabel("Count")
plt.title("Distribution of Credit_Score")
plt.tight_layout()
# plt.show()

plt.figure(figsize=(5, 5))
plt.boxplot(df["Credit_Score"].dropna(), vert=True)
plt.ylabel("Credit_Score")
plt.title("Credit_Score (boxplot)")
plt.tight_layout()
#plt.show()

# (b)(ii) Transactions_Value across Market_Segment (boxplots)
segments = sorted(df["Market_Segment"].dropna().unique())
data = [df.loc[df["Market_Segment"] == s, "Transactions_Value"].dropna().values for s in segments]

plt.figure(figsize=(9, 5))
plt.boxplot(data, labels=[str(s) for s in segments], showfliers=True)
plt.xlabel("Market_Segment")
plt.ylabel("Transactions_Value")
plt.title("Transactions_Value across Market_Segment (boxplots)")
plt.tight_layout()
#plt.show()

# -------------------------
# (c) Correlation heatmap (10 numerical variables)
# -------------------------

# Choose 10 "important" numeric variables (adjust if your teacher wants different ones)
vars_10 = [
    "Income",
    "Spending_Score",
    "Credit_Score",
    "Transactions_Value",
    "Savings",
    "Debt",
    "Fraud_Risk_Score",
    "Loyalty_Points",
    "Satisfaction_Score",
    "Dropout_Probability",
]

# Safety: keep only those that exist
vars_10 = [c for c in vars_10 if c in df.columns]
if len(vars_10) < 10:
    # fallback: fill up with other numeric columns
    remaining = [c for c in df.select_dtypes(include=[np.number]).columns if c not in vars_10]
    vars_10 += remaining[: (10 - len(vars_10))]
vars_10 = vars_10[:10]

corr = df[vars_10].corr(numeric_only=True)

plt.figure(figsize=(8, 6))
im = plt.imshow(corr.values, vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(vars_10)), vars_10, rotation=45, ha="right")
plt.yticks(range(len(vars_10)), vars_10)
plt.title("Correlation Heatmap (10 numerical variables)")

# annotate values
for i in range(len(vars_10)):
    for j in range(len(vars_10)):
        plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.show()

# Show top correlated pairs (by absolute value) among ALL numeric columns
num = df.select_dtypes(include=[np.number])
corr_all = num.corr(numeric_only=True)
cols = corr_all.columns.tolist()
pairs = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append((cols[i], cols[j], corr_all.iloc[i, j]))
pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

print("\nTop 10 correlations (absolute) among numeric variables:")
for a, b, r in pairs_sorted[:10]:
    print(f"{a:20s} vs {b:20s}  r = {r: .3f}")
