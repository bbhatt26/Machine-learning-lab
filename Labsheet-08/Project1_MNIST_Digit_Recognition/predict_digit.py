"""Use the saved model. Examples:
    python predict_digit.py                 # predicts 10 random test images
    python predict_digit.py --image my_digit.png   # predicts your own handwritten image"""
import argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from common import load_data

ap = argparse.ArgumentParser(); ap.add_argument("--image"); args = ap.parse_args()
model = load_model("models/mnist_dnn.keras")
data = load_data(); h, w = data["x_test"].shape[1:]

if args.image:
    img = ImageOps.grayscale(Image.open(args.image)).resize((w, h))
    a = np.asarray(img, dtype="float32") / 255.0
    if a.mean() > 0.5: a = 1 - a                      # MNIST = white digit on black background
    p = model.predict(a.reshape(1, -1), verbose=0)[0]
    print(f"Predicted digit: {p.argmax()}  (confidence {p.max():.1%})")
    plt.imshow(a, cmap="gray"); plt.title(f"Prediction: {p.argmax()} ({p.max():.0%})"); plt.axis("off")
    plt.savefig("outputs/09_custom_prediction.png", dpi=110)
else:
    idx = np.random.default_rng(0).choice(len(data["x_test"]), 10, replace=False)
    X = data["x_test"][idx]; p = model.predict(X.reshape(10, -1), verbose=0)
    fig, axes = plt.subplots(1, 10, figsize=(14, 2.2))
    for ax, x, t, pr in zip(axes, X, data["y_test"][idx], p):
        ax.imshow(x, cmap="gray"); ax.axis("off"); ax.set_title(f"{pr.argmax()} ({pr.max():.0%})", color="green" if pr.argmax() == t else "red", fontsize=9)
    plt.suptitle("Predictions (confidence)"); plt.tight_layout(); plt.savefig("outputs/09_sample_predictions.png", dpi=110)
    print("True     :", data["y_test"][idx]); print("Predicted:", p.argmax(1))
