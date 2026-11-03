"""Hyper-parameter tuning: vary one factor at a time (activation, optimizer, dropout,
learning rate, architecture) and compare validation / test accuracy."""
import time, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_data, flatten, build_model

EPOCHS = 6
data = load_data()
X_tr, y_tr, X_te, y_te = flatten(data["x_train"]), data["y_train"], flatten(data["x_test"]), data["y_test"]
base = dict(hidden=(256, 128), activation="relu", dropout=0.3, lr=1e-3, optimizer="adam")

experiments = {
    "activation": [dict(activation=a) for a in ["relu", "tanh", "sigmoid"]],
    "optimizer":  [dict(optimizer=o, lr=l) for o, l in [("sgd", 0.01), ("rmsprop", 1e-3), ("adam", 1e-3)]],
    "dropout":    [dict(dropout=d) for d in [0.0, 0.2, 0.5]],
    "learning_rate": [dict(lr=l) for l in [0.1, 0.01, 0.001, 0.0001]],
    "architecture": [dict(hidden=h) for h in [(64,), (128, 64), (256, 128), (512, 256, 128)]],
}
rows, curves = [], {}
for factor, options in experiments.items():
    for opt in options:
        cfg = {**base, **opt}
        value = list(opt.values())[0]            # the factor being varied
        label = f"{factor}={value}"
        t0 = time.time()
        m = build_model(X_tr.shape[1], **cfg)
        h = m.fit(X_tr, y_tr, validation_split=0.1, epochs=EPOCHS, batch_size=128, verbose=0)
        _, te_acc = m.evaluate(X_te, y_te, verbose=0)
        rows.append(dict(factor=factor, setting=str(value), train_acc=h.history["accuracy"][-1],
                         val_acc=h.history["val_accuracy"][-1], test_acc=te_acc, time_s=time.time() - t0))
        curves[label] = h.history["val_accuracy"]
        print(f"{label:35s} val={rows[-1]['val_acc']:.4f} test={te_acc:.4f}")

df = pd.DataFrame(rows); df.to_csv("outputs/hyperparameter_results.csv", index=False)
best = df.loc[df.test_acc.idxmax()]
print("\nBEST:", best.factor, best.setting, f"-> test accuracy {best.test_acc:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (factor, g) in zip(axes.ravel(), df.groupby("factor", sort=False)):
    x = range(len(g)); ax.bar([i - .2 for i in x], g.val_acc, .4, label="Validation"); ax.bar([i + .2 for i in x], g.test_acc, .4, label="Test")
    ax.set(xticks=list(x), title=factor, ylim=(max(0, df.val_acc.min() - .03), 1.0)); ax.set_xticklabels(g.setting, rotation=20, fontsize=8)
    ax.legend(fontsize=8)
axes.ravel()[-1].axis("off"); plt.suptitle("Hyper-parameter tuning (accuracy)"); plt.tight_layout()
plt.savefig("outputs/06_hyperparameter_tuning.png", dpi=110); plt.close()

plt.figure(figsize=(9, 5))
for k, v in curves.items():
    if k.startswith(("learning_rate", "activation", "optimizer")): plt.plot(range(1, EPOCHS + 1), v, marker="o", label=k)
plt.xlabel("Epoch"); plt.ylabel("Validation accuracy"); plt.title("Validation accuracy curves"); plt.legend(fontsize=7, ncol=2); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/07_tuning_curves.png", dpi=110); plt.close()
