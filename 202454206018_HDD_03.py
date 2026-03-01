# PCA + Scree Plot + PC1/PC2 Plot by Risk_Class + PC1 Loadings + PCR
# Dataset path (as provided):
# /mnt/data/HDD_Lab_Customer-data - HDD_Lab_Customer-data.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# -----------------------------
# 1) Load data
# -----------------------------
path = "/mnt/data/HDD_Lab_Customer-data - HDD_Lab_Customer-data.csv"
df = pd.read_csv(r"C:\Users\HP\Downloads\HDD_Lab_Customer-data - HDD_Lab_Customer-data.csv")

target_col = "Dropout_Probability"
risk_col = "Risk_Class"

# -----------------------------
# 2) Select numeric predictors for PCA
#    Exclude target + risk label + coded categoricals
# -----------------------------
coded_categoricals = ["Market_Segment", "Region_Code", "Channel _ Preference", "Risk_Class"]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Numeric predictors excluding target and coded categoricals/risk
pca_features = [c for c in numeric_cols if c not in coded_categoricals + [target_col]]

X = df[pca_features].copy()
y = df[target_col].copy()
risk = df[risk_col].copy()

# Basic NA handling (drop rows with NA in relevant columns)
data = pd.concat([X, y, risk], axis=1).dropna()
X = data[pca_features]
y = data[target_col]
risk = data[risk_col]


# -----------------------------
# 3) (a) Standardize + PCA (full)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
X_pcs = pca_full.fit_transform(X_scaled)

expl_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

print("Explained variance (first 10 PCs):", expl_var[:10])
print("Cumulative variance (first 10 PCs):", cum_var[:10])


# -----------------------------
# 4) (b) Scree plot + # PCs for >=80% variance
# -----------------------------
k80 = int(np.argmax(cum_var >= 0.80) + 1)
print(f"\nNumber of PCs to reach >=80% variance: {k80}")
print(f"Cumulative variance at PC{k80}: {cum_var[k80-1]:.4f}")

components = np.arange(1, len(expl_var) + 1)

plt.figure(figsize=(8, 4.5))
plt.plot(components, expl_var, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot (Explained Variance Ratio)")
plt.xticks(components)
plt.tight_layout()
#plt.show()

plt.figure(figsize=(8, 4.5))
plt.plot(components, cum_var, marker="s")
plt.axhline(0.80, linestyle="--")
plt.axvline(k80, linestyle="--")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained")
plt.xticks(components)
plt.ylim(0, 1.05)
plt.tight_layout()
#plt.show()


# -----------------------------
# 5) (c) Plot first two PCs, color by Risk_Class
# -----------------------------
pc_df = pd.DataFrame(X_pcs[:, :2], columns=["PC1", "PC2"])
pc_df["Risk_Class"] = risk.values

plt.figure(figsize=(7, 5))
for cls in sorted(pc_df["Risk_Class"].unique()):
    subset = pc_df[pc_df["Risk_Class"] == cls]
    plt.scatter(subset["PC1"], subset["PC2"], s=18, alpha=0.7, label=f"Risk_Class {cls}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PC1 vs PC2 (Colored by Risk_Class)")
plt.legend()
plt.tight_layout()
#plt.show()


# -----------------------------
# 6) (d) Variables contributing most to PC1 (loadings)
# -----------------------------
pc1_loadings = pd.Series(pca_full.components_[0], index=X.columns, name="PC1_loading")
top_pc1 = pc1_loadings.abs().sort_values(ascending=False).head(10)

print("\nTop 10 variables by absolute PC1 loading:")
print(pd.DataFrame({
    "variable": top_pc1.index,
    "PC1_loading": pc1_loadings[top_pc1.index].values,
    "abs_loading": top_pc1.values
}).to_string(index=False))


# -----------------------------
# 7) (e) Principal Component Regression (PCR) using k80 PCs
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pcr = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=k80)),
    ("lr", LinearRegression())
])

pcr.fit(X_train, y_train)
y_pred = pcr.predict(X_test)

pcr_r2 = r2_score(y_test, y_pred)
pcr_rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"\nPCR (n_components={k80}) Test R^2:  {pcr_r2:.4f}")
print(f"PCR (n_components={k80}) Test RMSE: {pcr_rmse:.4f}")


# Optional: Compare to ordinary linear regression on original predictors
ols = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

ols_r2 = r2_score(y_test, y_pred_ols)
ols_rmse = mean_squared_error(y_test, y_pred_ols, squared=False)

print(f"\nOLS (all predictors) Test R^2:  {ols_r2:.4f}")
print(f"OLS (all predictors) Test RMSE: {ols_rmse:.4f}")


# Optional: 5-fold cross-validated R^2
cv = KFold(n_splits=5, shuffle=True, random_state=42)
pcr_cv_r2 = cross_val_score(pcr, X, y, cv=cv, scoring="r2").mean()
ols_cv_r2 = cross_val_score(ols, X, y, cv=cv, scoring="r2").mean()

print(f"\n5-fold CV mean R^2 (PCR): {pcr_cv_r2:.4f}")
print(f"5-fold CV mean R^2 (OLS): {ols_cv_r2:.4f}")
