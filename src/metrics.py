"""Evaluation metrics and spec-threshold checking."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, classification_report)

from . import config as C


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = recall_score(y_true, y_pred, average=None,
                           labels=[0, 1], zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_consistent": float(recalls[0]),
        "recall_mismatch": float(recalls[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
    }
    metrics["min_per_class_recall"] = min(metrics["recall_consistent"],
                                          metrics["recall_mismatch"])
    metrics["passes"] = check_thresholds(metrics)
    return metrics


def check_thresholds(m: dict) -> dict:
    t = C.THRESHOLDS
    checks = {
        "accuracy": m["accuracy"] >= t["accuracy"],
        "macro_f1": m["macro_f1"] >= t["macro_f1"],
        "per_class_recall": m["min_per_class_recall"] >= t["per_class_recall"],
    }
    checks["ALL_PASS"] = all(checks.values())
    return checks


def report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Consistent", "Mismatch"], zero_division=0)
