import json
import os

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.h5")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")


def classification_report(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
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


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        raise SystemExit(
            "Missing artifacts. Run: python train_model.py "
            "(expected artifacts/model.h5 and artifacts/meta.json)."
        )

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    max_features = int(meta.get("max_features", 10000))
    maxlen = int(meta.get("maxlen", 500))
    threshold = float(meta.get("threshold", 0.5))

    model = load_model(MODEL_PATH)

    (_, _), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    y_score = model.predict(x_test, verbose=0).reshape(-1)
    report = classification_report(np.asarray(y_test), y_score, threshold)

    cm = report["confusion_matrix"]
    print(
        "test "
        f"thr={report['threshold']:.2f} "
        f"acc={report['accuracy']:.3f} "
        f"prec={report['precision']:.3f} "
        f"rec={report['recall']:.3f} "
        f"f1={report['f1']:.3f} "
        f"cm(tp={cm['tp']},tn={cm['tn']},fp={cm['fp']},fn={cm['fn']})"
    )


if __name__ == "__main__":
    main()

