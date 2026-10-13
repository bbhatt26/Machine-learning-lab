"""
Project 1: Customer Segmentation using K-Means Clustering
MCA201P - Lab Sheet 07
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

os.makedirs("outputs", exist_ok=True)
if not os.path.exists("customers.csv"):
    import generate_dataset
sns.set_theme(style="whitegrid")

# 1. Load & clean -------------------------------------------------------------
df = pd.read_csv("customers.csv")
print("Raw shape:", df.shape, "| missing:", df.isna().sum().sum(), "| duplicates:", df.duplicated().sum())
df = df.drop_duplicates().copy()
df["Annual_Income_k"] = df["Annual_Income_k"].fillna(df["Annual_Income_k"].median())
print("Clean shape:", df.shape)
print(df.describe().round(2))

features = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency"]

# quick EDA
sns.pairplot(df[features]); plt.savefig("outputs/01_pairplot.png", dpi=120); plt.close()
plt.figure(figsize=(6, 5)); sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Feature correlation"); plt.tight_layout(); plt.savefig("outputs/02_correlation.png", dpi=120); plt.close()

# 2. Feature scaling ----------------------------------------------------------
X = StandardScaler().fit_transform(df[features])

# 3. Optimal k: elbow + silhouette -------------------------------------------
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
best_k = list(ks)[int(np.argmax(sil))]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia (WCSS)")
ax[1].plot(ks, sil, "s-g"); ax[1].axvline(best_k, ls="--", c="r")
ax[1].set(title=f"Silhouette score (best k={best_k})", xlabel="k", ylabel="Silhouette")
plt.tight_layout(); plt.savefig("outputs/03_optimal_k.png", dpi=120); plt.close()
print("Optimal k (silhouette):", best_k)

# 4. Final K-Means ------------------------------------------------------------
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df["Cluster"] = km.labels_

# 5. Visualise clusters -------------------------------------------------------
pca = PCA(n_components=2, random_state=42); P = pca.fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=df["Cluster"], palette="tab10", ax=ax[0])
cp = pca.transform(km.cluster_centers_)
ax[0].scatter(cp[:, 0], cp[:, 1], c="k", marker="X", s=200, label="Centroids")
ax[0].set(title="K-Means clusters (PCA 2-D)", xlabel="PC1", ylabel="PC2"); ax[0].legend()
sns.scatterplot(data=df, x="Annual_Income_k", y="Spending_Score", hue="Cluster", palette="tab10", ax=ax[1])
ax[1].set_title("Income vs Spending Score")
plt.tight_layout(); plt.savefig("outputs/04_kmeans_clusters.png", dpi=120); plt.close()

# 6. Cluster profiling / business insights ------------------------------------
profile = df.groupby("Cluster")[features].mean().round(1)
profile["Size"] = df["Cluster"].value_counts().sort_index()
def label(r):
    inc = "high" if r.Annual_Income_k > 70 else "low" if r.Annual_Income_k < 40 else "mid"
    sp = "high" if r.Spending_Score > 65 else "low" if r.Spending_Score < 35 else "mid"
    return {("high", "high"): "Premium / VIP customers (retain with exclusive perks)",
            ("high", "low"): "Wealthy but frugal (target with personalised offers)",
            ("low", "high"): "Young enthusiastic spenders (loyalty programs, discounts)",
            ("low", "low"): "Low-value / budget customers (low-cost engagement)"}.get((inc, sp), "Average mid-tier customers (upsell / cross-sell)")
profile["Segment"] = profile.apply(label, axis=1)
print("\nCluster profile:\n", profile.to_string())
profile.to_csv("outputs/cluster_profile.csv")
df.to_csv("outputs/customers_with_clusters.csv", index=False)
plt.figure(figsize=(8, 4.5)); sns.heatmap(profile[features], annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Average feature values per cluster"); plt.tight_layout(); plt.savefig("outputs/05_cluster_heatmap.png", dpi=120); plt.close()

# 7. Compare clustering techniques -------------------------------------------
models = {
    "K-Means": km.labels_,
    "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=best_k).fit_predict(X),
    "Gaussian Mixture": GaussianMixture(n_components=best_k, random_state=42).fit_predict(X),
    "DBSCAN": DBSCAN(eps=0.9, min_samples=5).fit_predict(X),
}
res = []
for name, lab in models.items():
    mask = lab != -1
    n_cl = len(set(lab[mask]))
    if n_cl > 1:
        res.append([name, n_cl, (~mask).sum(), silhouette_score(X[mask], lab[mask]),
                    davies_bouldin_score(X[mask], lab[mask]), calinski_harabasz_score(X[mask], lab[mask])])
    else:
        res.append([name, n_cl, (~mask).sum(), np.nan, np.nan, np.nan])
cmp = pd.DataFrame(res, columns=["Method", "Clusters", "Noise pts", "Silhouette (up)", "Davies-Bouldin (down)", "Calinski-Harabasz (up)"]).round(3)
print("\nEvaluation metrics:\n", cmp.to_string(index=False))
cmp.to_csv("outputs/evaluation_metrics.csv", index=False)

fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, lab) in zip(axs.ravel(), models.items()):
    sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=lab, palette="tab10", ax=ax, legend=False); ax.set_title(name)
plt.tight_layout(); plt.savefig("outputs/06_method_comparison.png", dpi=120); plt.close()

plt.figure(figsize=(10, 4)); dendrogram(linkage(X, "ward"), truncate_mode="lastp", p=30)
plt.title("Hierarchical clustering dendrogram (Ward)"); plt.tight_layout()
plt.savefig("outputs/07_dendrogram.png", dpi=120); plt.close()
print("\nAll outputs saved in ./outputs")
