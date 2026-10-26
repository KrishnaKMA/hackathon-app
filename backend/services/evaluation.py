"""Deterministic classification metrics; rows=true labels, columns=predictions."""

import numpy as np


def classification_metrics(actual, predicted, classes):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Non-empty, equally sized truth and prediction lists required")
    indices = {name: i for i, name in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, prediction in zip(actual, predicted, strict=True):
        if truth not in indices or prediction not in indices:
            raise ValueError("Unknown class label")
        matrix[indices[truth], indices[prediction]] += 1
    support = matrix.sum(axis=1)
    guessed = matrix.sum(axis=0)
    correct = np.diag(matrix)
    precision = np.divide(
        correct, guessed, out=np.zeros(len(classes)), where=guessed != 0
    )
    recall = np.divide(correct, support, out=np.zeros(len(classes)), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes)),
        where=precision + recall != 0,
    )
    return {
        "class_order": classes,
        "sample_count": len(actual),
        "confusion_matrix": matrix.tolist(),
        "matrix_axes": "rows=true, columns=predicted",
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
        "accuracy": float(correct.sum() / len(actual)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / len(actual)),
        "zero_division": 0,
        "missing_classes": [name for i, name in enumerate(classes) if support[i] == 0],
    }


def compare_results(current, previous):
    if current["dataset"]["sha256"] != previous["dataset"]["sha256"]:
        raise ValueError("Cannot compare evaluations using different datasets")
    if current["metrics"]["class_order"] != previous["metrics"]["class_order"]:
        raise ValueError("Class order differs")
    return {
        "previous_run": previous["run_id"],
        "previous_model": previous["model"],
        "delta": {
            key: current["metrics"][key] - previous["metrics"][key]
            for key in ("accuracy", "macro_f1", "weighted_f1")
        },
        "per_class_f1_delta": {
            name: current["metrics"]["per_class"][name]["f1"]
            - previous["metrics"]["per_class"][name]["f1"]
            for name in current["metrics"]["class_order"]
        },
    }
