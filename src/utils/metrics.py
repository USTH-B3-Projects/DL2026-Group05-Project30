import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from src.training.config import RESULTS_METRICS_DIR, CLASS_NAMES

def compute_metrics(labels: list, preds: list) -> dict:
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    recall = recall_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1":        round(float(f1_per_class[i]), 4),
        }
        for i in range(len(CLASS_NAMES))
    }

    return {
        "accuracy":         round(float(acc), 4),
        "f1_macro":         round(float(f1_macro), 4),
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
    }

def save_metrics(metrics: dict, model_name: str) -> str:
    
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_METRICS_DIR, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")
    return path

def print_report(labels: list, preds: list) -> None:
    # Print a sklearn classification report to stdout.
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
