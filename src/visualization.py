"""
Visualization utilities: plots, confusion matrices, and Grad-CAM.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys train_loss, val_loss, train_acc, val_acc.
        save_path: Optional path to save the figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train accuracy", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val accuracy", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_confusion_matrix(labels, preds, class_names: list = None,
                          save_path: str = None):
    """
    Plot a confusion matrix heatmap.

    Args:
        labels: True labels.
        preds: Predicted labels.
        class_names: Optional list of class name strings.
        save_path: Optional path to save the figure.
    """
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def print_classification_report(labels, preds, class_names: list = None):
    """Print sklearn classification report."""
    print(classification_report(labels, preds, target_names=class_names, digits=4))


def plot_roc_curves(labels, probs, class_names: list = None,
                    save_path: str = None):
    """
    Plot one-vs-rest ROC curves for multi-class classification.

    Args:
        labels: True labels (list of ints).
        probs: Predicted probabilities (N x num_classes tensor or array).
        class_names: Optional list of class names.
        save_path: Optional path to save the figure.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()

    labels_array = np.array(labels)
    num_classes = probs.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        binary_labels = (labels_array == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_sample_images(dataset, n: int = 16, class_names: list = None):
    """
    Display a grid of sample images from a dataset.

    Args:
        dataset: PyTorch Dataset returning (image, label).
        n: Number of images to show.
        class_names: Optional label name mapping.
    """
    from src.transforms import get_inverse_normalize

    inv_normalize = get_inverse_normalize()

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i in range(n):
        img, label = dataset[i]
        # Undo normalization for display
        img = inv_normalize(img)
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()

        axes[i].imshow(img)
        title = class_names[label] if class_names else str(label)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
