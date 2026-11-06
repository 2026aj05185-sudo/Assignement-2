import io

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
TARGET = "quality"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def read_csv(content):
    if isinstance(content, bytes):
        content = io.BytesIO(content)
    return pd.read_csv(content, sep=None, engine="python")


def validate_data(frame, classes=None):
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    try:
        cleaned = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("All features and quality values must be numeric.") from error
    if not np.isfinite(cleaned.to_numpy(dtype=float)).all():
        raise ValueError("Missing, infinite, and NaN values are not supported.")
    quality = cleaned[TARGET]
    if not ((quality == np.floor(quality)) & quality.between(0, 10)).all():
        raise ValueError("Quality must be a whole-number score between 0 and 10.")
    if classes is not None and not quality.isin(classes).all():
        raise ValueError("Quality contains labels the models were not trained on: "
                         + ", ".join(map(str, sorted(set(quality) - set(classes)))))
    return cleaned[FEATURES], quality.astype(int)


def evaluate_model(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = model.classes_
    auc = None
    if set(target) == set(classes):
        auc = float(roc_auc_score(
            target, probabilities, labels=classes, multi_class="ovr", average="macro"
        ))
    metrics = {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": auc,
        "Precision": float(precision_score(
            target, predictions, labels=classes, average="macro", zero_division=0
        )),
        "Recall": float(recall_score(
            target, predictions, labels=classes, average="macro", zero_division=0
        )),
        "F1": float(f1_score(
            target, predictions, labels=classes, average="macro", zero_division=0
        )),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
    return metrics, predictions, probabilities