"""
Training and evaluation functions.
"""

import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs: int = 10, device: str = "cpu",
                scheduler=None, early_stop_patience: int = 0):
    """
    Train a model with validation tracking and optional early stopping.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        optimizer: Optimizer.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        num_epochs: Number of epochs.
        device: 'cuda' or 'cpu'.
        scheduler: Optional learning rate scheduler.
        early_stop_patience: Stop if val loss doesn't improve for this many
                             epochs. Set to 0 to disable.

    Returns:
        model: Model with best validation weights loaded.
        history: Dict with train_loss, val_loss, train_acc, val_acc lists.
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels).item()
                total_samples += batch_size

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            print(f"  {phase:>5s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
                # Record current learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                history["lr"].append(current_lr)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

                # Best model tracking
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Step the scheduler (if provided) after each epoch
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break

    elapsed = time.time() - since
    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


@torch.no_grad()
def evaluate_model(model, dataloader, device: str = "cpu"):
    """
    Evaluate model on a dataset and return predictions + labels.

    Returns:
        all_labels: List of true labels.
        all_preds: List of predicted labels.
        all_probs: Tensor of prediction probabilities (N x num_classes).
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)

    acc = sum(1 for l, p in zip(all_labels, all_preds) if l == p) / len(all_labels)
    print(f"Accuracy: {acc:.4f}  ({sum(1 for l, p in zip(all_labels, all_preds) if l == p)}/{len(all_labels)})")

    return all_labels, all_preds, all_probs


def save_model(model, path: str):
    """Save model state dict to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path: str, device: str = "cpu"):
    """Load model state dict from disk."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


@torch.no_grad()
def mc_dropout_predict(model, inputs, n_forward: int = 30):
    """
    Monte Carlo Dropout: run multiple forward passes with dropout enabled
    to estimate uncertainty.

    Args:
        model: Model (must have dropout layers).
        inputs: Input tensor (batch).
        n_forward: Number of stochastic forward passes.

    Returns:
        mean_probs: Mean predicted probabilities (batch x num_classes).
        std_probs: Std of predicted probabilities (batch x num_classes).
    """
    model.train()  # Enable dropout

    all_probs = []
    for _ in range(n_forward):
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.unsqueeze(0))

    all_probs = torch.cat(all_probs, dim=0)  # (n_forward, batch, num_classes)
    mean_probs = all_probs.mean(dim=0)
    std_probs = all_probs.std(dim=0)

    model.eval()
    return mean_probs, std_probs
