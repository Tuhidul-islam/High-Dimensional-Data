import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\HDD_Lab_Customer-data - HDD_Lab_Customer-data.csv")

# Features (use all numeric columns; if you have a label column, drop it)
X = df.drop(columns=["Risk_Class"], errors="ignore")

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA reduction (2D for clustering + visualization in PCA space)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Elbow method
inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(7,5))
plt.plot(list(K), inertias, marker="o")
plt.title("Elbow Method (K-means on PCA-reduced data)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (Within-cluster SSE)")
plt.xticks(list(K))
plt.show()

k = 4  # <-- change this based on your elbow plot

kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_pca)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=12)
plt.title(f"K-means Clusters in PCA Space (k={k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Optional: print cluster counts
import numpy as np
unique, counts = np.unique(labels, return_counts=True)
print("Cluster sizes:", dict(zip(unique, counts)))

