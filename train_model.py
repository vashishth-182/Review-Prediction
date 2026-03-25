import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

SEED = 42 #aa seed value set karva thi reproducibility aave che je random nakhyu chhe
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.h5")
WORD_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "word_index.json")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")


def _ensure_artifacts_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _classification_report(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    y_true = y_true.astype(np.int32)
    y_pred = (y_score >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = float((tp + tn) / max(1, (tp + tn + fp + fn)))
    precision = float(tp / max(1, (tp + fp)))
    recall = float(tp / max(1, (tp + fn)))
    f1 = float((2 * precision * recall) / max(1e-12, (precision + recall)))

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _pick_threshold(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = np.linspace(0.1, 0.9, 17)
    reports = [_classification_report(y_true, y_score, float(t)) for t in thresholds]
    best = max(reports, key=lambda r: (r["f1"], r["accuracy"]))
    return best["threshold"], best, reports


def main():
    _ensure_artifacts_dir()

    # IMDB data loading
    max_features = 10000
    maxlen = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    # Save word index (used by app.py for raw text -> ids)
    word_index = imdb.get_word_index()
    with open(WORD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(word_index, f)

    # LSTM model
    model = Sequential(
        [
            Embedding(max_features, 32),
            LSTM(32),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    ckpt_path = os.path.join(ARTIFACTS_DIR, "_best_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate and pick threshold
    y_score = model.predict(x_test, verbose=0).reshape(-1)
    best_threshold, best_report, _ = _pick_threshold(np.asarray(y_test), y_score)

    model.save(MODEL_PATH)

    meta = {
        "max_features": int(max_features),
        "maxlen": int(maxlen),
        "threshold": float(best_threshold),
        "model_format": "h5",
        "train": {
            "epochs_ran": int(len(history.history.get("loss", []))),
            "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
        },
        "test": best_report,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved word index: {WORD_INDEX_PATH}")
    print(f"Saved meta: {META_PATH}")
    print(
        "Test (picked threshold): "
        f"thr={best_report['threshold']:.2f} "
        f"acc={best_report['accuracy']:.3f} "
        f"prec={best_report['precision']:.3f} "
        f"rec={best_report['recall']:.3f} "
        f"f1={best_report['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
