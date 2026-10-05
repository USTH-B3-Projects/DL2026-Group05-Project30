import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.config import RESULTS_FIGURES_DIR, CLASS_NAMES

def _save(fig: plt.Figure, filename: str) -> str:
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    path = os.path.join(RESULTS_FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Figure saved → {path}")
    return path

def plot_training_curves(history: dict, model_name: str) -> plt.Figure:
    
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="Validation loss", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Training curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    _save(fig, f"{model_name}_training_curves.png")
    return fig

def plot_confusion_matrix(cm: list, model_name: str) -> plt.Figure:
    
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion matrix")
    fig.tight_layout()

    _save(fig, f"{model_name}_confusion_matrix.png")
    return fig
