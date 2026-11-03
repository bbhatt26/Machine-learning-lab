"""Project 1: Handwritten digit recognition with a Deep Neural Network.
Trains, evaluates, plots learning curves + confusion matrix and saves the model."""
import time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from common import load_data, flatten, build_model, save_json

OUT, MODELS = "outputs", "models"
data = load_data()
x_tr, y_tr, x_te, y_te = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
print(f"Dataset: {data['name']} | train {x_tr.shape} | test {x_te.shape}")

# ---- sample images
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_tr[i], cmap="gray"); ax.set_title(int(y_tr[i])); ax.axis("off")
plt.suptitle(f"Sample training images ({data['name']})"); plt.tight_layout(); plt.savefig(f"{OUT}/01_sample_images.png", dpi=110); plt.close()

# ---- model
X_tr, X_te = flatten(x_tr), flatten(x_te)
model = build_model(X_tr.shape[1], hidden=(512, 256, 128), activation="relu", dropout=0.3, lr=1e-3, optimizer="adam")
model.summary()

cb = [EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
      ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)]
t0 = time.time()
hist = model.fit(X_tr, y_tr, validation_split=0.1, epochs=30, batch_size=128, callbacks=cb, verbose=2)
train_time = time.time() - t0

# ---- evaluation
test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
probs = model.predict(X_te, verbose=0); pred = probs.argmax(1)
print(f"\nTEST ACCURACY: {test_acc:.4f} | TEST LOSS: {test_loss:.4f} | training time {train_time:.1f}s")
report = classification_report(y_te, pred, digits=4); print(report)
open(f"{OUT}/classification_report.txt", "w").write(report)

# ---- learning curves
h = hist.history; ep = range(1, len(h["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ep, h["accuracy"], marker="o", label="Train"); ax[0].plot(ep, h["val_accuracy"], marker="s", label="Validation")
ax[0].set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep, h["loss"], marker="o", label="Train"); ax[1].plot(ep, h["val_loss"], marker="s", label="Validation")
ax[1].set(title="Model Loss", xlabel="Epoch", ylabel="Cross-entropy loss"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/02_learning_curves.png", dpi=110); plt.close()

# ---- confusion matrix
cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/03_confusion_matrix.png", dpi=110); plt.close()

# ---- per-class accuracy
per_class = cm.diagonal() / cm.sum(1)
fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(range(10), per_class * 100, color="#3b82f6")
ax.set(xticks=range(10), xlabel="Digit", ylabel="Accuracy (%)", title="Per-class Accuracy", ylim=(max(0, per_class.min() * 100 - 5), 100.5))
plt.tight_layout(); plt.savefig(f"{OUT}/04_per_class_accuracy.png", dpi=110); plt.close()

# ---- misclassified examples
wrong = np.where(pred != y_te)[0][:16]
if len(wrong):
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.8))
    for ax in axes.ravel(): ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(x_te[i], cmap="gray"); ax.set_title(f"T:{y_te[i]} P:{pred[i]}", fontsize=9, color="red")
    plt.suptitle("Misclassified test images (T = true, P = predicted)"); plt.tight_layout()
    plt.savefig(f"{OUT}/05_misclassified.png", dpi=110); plt.close()

# ---- save model + metrics
model.save(f"{MODELS}/mnist_dnn.keras")
save_json({"dataset": data["name"], "test_accuracy": float(test_acc), "test_loss": float(test_loss),
           "epochs_run": len(ep), "training_time_sec": train_time, "parameters": int(model.count_params()),
           "most_confused_digit": int(np.argmin(per_class))}, f"{OUT}/dnn_results.json")
print("Saved model -> models/mnist_dnn.keras")
