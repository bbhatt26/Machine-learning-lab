"""
Project 2: News Article Clustering System (TF-IDF + K-Means / Hierarchical)
MCA201P - Lab Sheet 07
"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (silhouette_score, davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score)
from scipy.cluster.hierarchy import dendrogram, linkage

os.makedirs("outputs", exist_ok=True)
if not os.path.exists("news_articles.csv"):
    import generate_dataset
sns.set_theme(style="whitegrid")

# 1. Load & preprocess ---------------------------------------------------------
df = pd.read_csv("news_articles.csv").dropna(subset=["text"]).drop_duplicates("text")
def clean(t):
    t = re.sub(r"[^a-z\s]", " ", t.lower())
    return " ".join(w for w in t.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2)
df["clean"] = df["text"].apply(clean)
print("Articles:", len(df))

# 2. TF-IDF (rows are L2-normalised = scaling) --------------------------------
tfidf = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
X = tfidf.fit_transform(df["clean"]); terms = np.array(tfidf.get_feature_names_out())
print("TF-IDF matrix:", X.shape)

# 3. Optimal k ------------------------------------------------------------------
ks = range(2, 11); inertia, sil = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(m.inertia_); sil.append(silhouette_score(X, m.labels_))
best_k = list(ks)[int(np.argmax(sil))]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, sil, "s-g"); ax[1].axvline(best_k, ls="--", c="r"); ax[1].set(title=f"Silhouette (best k={best_k})", xlabel="k")
plt.tight_layout(); plt.savefig("outputs/01_optimal_k.png", dpi=120); plt.close()
print("Optimal k:", best_k)

# 4. Models ---------------------------------------------------------------------
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
Xd = X.toarray()
hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(Xd)
df["kmeans_cluster"], df["hier_cluster"] = km.labels_, hc.labels_

# 5. Top terms per cluster ------------------------------------------------------
order = km.cluster_centers_.argsort()[:, ::-1]
lines = []
for c in range(best_k):
    lines.append(f"Cluster {c} ({(km.labels_ == c).sum()} articles): " + ", ".join(terms[order[c, :10]]))
print("\nTop terms per cluster (K-Means):\n" + "\n".join(lines))
open("outputs/top_terms_per_cluster.txt", "w").write("\n".join(lines))

# 6. Visualisation (SVD 2-D) ----------------------------------------------------
P = TruncatedSVD(2, random_state=42).fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=km.labels_, palette="tab10", ax=ax[0]); ax[0].set_title("K-Means clusters")
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=hc.labels_, palette="tab10", ax=ax[1]); ax[1].set_title("Hierarchical clusters")
plt.tight_layout(); plt.savefig("outputs/02_cluster_visualisation.png", dpi=120); plt.close()

plt.figure(figsize=(10, 4)); dendrogram(linkage(Xd, "ward"), truncate_mode="lastp", p=30)
plt.title("Dendrogram (Ward linkage)"); plt.tight_layout(); plt.savefig("outputs/03_dendrogram.png", dpi=120); plt.close()

fig, axs = plt.subplots(1, best_k, figsize=(3.2 * best_k, 3.5))
for c, ax in zip(range(best_k), np.atleast_1d(axs)):
    idx = order[c, :8]; ax.barh(terms[idx][::-1], km.cluster_centers_[c, idx][::-1]); ax.set_title(f"Cluster {c}")
plt.tight_layout(); plt.savefig("outputs/04_top_terms.png", dpi=120); plt.close()

# 7. Evaluation & comparison -----------------------------------------------------
rows = []
for name, lab in [("K-Means", km.labels_), ("Hierarchical (Ward)", hc.labels_)]:
    rows.append([name, silhouette_score(X, lab), davies_bouldin_score(Xd, lab),
                 adjusted_rand_score(df["true_category"], lab),
                 normalized_mutual_info_score(df["true_category"], lab),
                 homogeneity_score(df["true_category"], lab)])
cmp = pd.DataFrame(rows, columns=["Method", "Silhouette (up)", "Davies-Bouldin (down)", "ARI (up)", "NMI (up)", "Homogeneity (up)"]).round(3)
print("\nEvaluation metrics:\n", cmp.to_string(index=False)); cmp.to_csv("outputs/evaluation_metrics.csv", index=False)

ct = pd.crosstab(df["true_category"], df["kmeans_cluster"])
plt.figure(figsize=(6, 4)); sns.heatmap(ct, annot=True, fmt="d", cmap="Blues"); plt.title("True category vs K-Means cluster")
plt.tight_layout(); plt.savefig("outputs/05_confusion_heatmap.png", dpi=120); plt.close()
df.drop(columns="clean").to_csv("outputs/news_with_clusters.csv", index=False)

# 8. Predict a new article --------------------------------------------------------
new = "The government minister presented a new election bill in parliament"
print("\nNew article ->", new, "-> cluster", km.predict(tfidf.transform([clean(new)]))[0])
