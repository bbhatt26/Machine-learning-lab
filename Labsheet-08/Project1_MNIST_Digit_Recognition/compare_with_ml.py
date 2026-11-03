"""Compare the Deep Neural Network with traditional ML models (Logistic Regression,
Random Forest, SVM, KNN) on the same data."""
import os, time, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from common import load_data, flatten, SEED

data = load_data()
X_tr, y_tr, X_te, y_te = flatten(data["x_train"]), data["y_train"], flatten(data["x_test"]), data["y_test"]
SUB = 10000  # SVM / KNN are slow on 60k images -> train them on a 10k subset (stated in the report)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=200), len(X_tr)),
    "Random Forest": (RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED), len(X_tr)),
    f"SVM (RBF, {SUB} samples)": (SVC(kernel="rbf", C=5, gamma="scale"), SUB),
    f"KNN (k=3, {SUB} samples)": (KNeighborsClassifier(3, n_jobs=-1), SUB),
}
rows = []
for name, (clf, n) in models.items():
    t0 = time.time(); clf.fit(X_tr[:n], y_tr[:n]); tt = time.time() - t0
    t0 = time.time(); acc = accuracy_score(y_te, clf.predict(X_te)); pt = time.time() - t0
    rows.append(dict(model=name, test_accuracy=acc, train_time_s=tt, predict_time_s=pt)); print(rows[-1])

# DNN: use the saved model from train_dnn.py (train it first if it is missing)
path = "models/mnist_dnn.keras"
if not os.path.exists(path):
    import subprocess, sys; subprocess.run([sys.executable, "train_dnn.py"], check=True)
import json; from tensorflow.keras.models import load_model
dnn = load_model(path); t0 = time.time(); _, acc = dnn.evaluate(X_te, y_te, verbose=0); pt = time.time() - t0
tt = json.load(open("outputs/dnn_results.json"))["training_time_sec"]
rows.append(dict(model="Deep Neural Network (Keras)", test_accuracy=acc, train_time_s=tt, predict_time_s=pt))

df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
df.to_csv("outputs/ml_vs_dl_comparison.csv", index=False); print("\n", df.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
cols = ["#22c55e" if "Neural" in m else "#94a3b8" for m in df.model]
ax[0].barh(df.model, df.test_accuracy * 100, color=cols); ax[0].invert_yaxis()
ax[0].set(xlabel="Test accuracy (%)", title="Accuracy: Traditional ML vs Deep Learning", xlim=(max(0, df.test_accuracy.min() * 100 - 5), 100.5))
for i, v in enumerate(df.test_accuracy * 100): ax[0].text(v + .05, i, f"{v:.2f}", va="center", fontsize=9)
ax[1].barh(df.model, df.train_time_s, color=cols); ax[1].invert_yaxis(); ax[1].set(xlabel="Training time (s)", title="Training time")
plt.tight_layout(); plt.savefig("outputs/08_ml_vs_dl.png", dpi=110); plt.close()
