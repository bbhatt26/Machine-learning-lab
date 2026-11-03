"""Shared helpers: data loading + model builder (Project 1)."""
import os, json, numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
SEED = 42
np.random.seed(SEED)

def load_data():
    """Return dict(x_train, y_train, x_test, y_test, name). Pixels scaled to 0-1.
    Uses MNIST (60k train / 10k test). If it cannot be downloaded (offline),
    falls back to scikit-learn's small 8x8 'digits' dataset so the code still runs."""
    try:
        from tensorflow.keras.datasets import mnist
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
        x_tr, x_te = x_tr.astype("float32") / 255.0, x_te.astype("float32") / 255.0
        name = "MNIST"
    except Exception as e:                                   # offline fallback
        print(f"[WARN] Could not download MNIST ({type(e).__name__}). Using sklearn digits (8x8) instead.")
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        d = load_digits()
        x = (d.images / 16.0).astype("float32")
        x_tr, x_te, y_tr, y_te = train_test_split(x, d.target, test_size=0.2, stratify=d.target, random_state=SEED)
        name = "sklearn-digits (fallback)"
    return dict(x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te, name=name)

def flatten(x):
    return x.reshape(len(x), -1)

def build_model(input_dim, hidden=(512, 256, 128), activation="relu", dropout=0.3,
                lr=1e-3, optimizer="adam", batch_norm=True):
    """Fully-connected DNN: [Dense -> (BatchNorm) -> Activation -> Dropout] x N -> Softmax(10)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    tf.keras.utils.set_random_seed(SEED)
    m = models.Sequential([layers.Input(shape=(input_dim,))])
    for units in hidden:
        m.add(layers.Dense(units, activation=None if batch_norm else activation))
        if batch_norm:
            m.add(layers.BatchNormalization()); m.add(layers.Activation(activation))
        if dropout > 0:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(10, activation="softmax"))          # softmax -> class probabilities
    opt = {"adam": optimizers.Adam, "sgd": lambda learning_rate: optimizers.SGD(learning_rate, momentum=0.9),
           "rmsprop": optimizers.RMSprop}[optimizer](learning_rate=lr)
    m.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
