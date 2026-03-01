"""
Script-based 15-class baseline smoke run.

Runs a small ResNet18 training/evaluation pass and saves:
1) model checkpoint
2) metrics snapshot JSON
3) experiment log JSON (with git commit + split manifest hash + seed)

Run from Main/:
    py -3.13 -m src.run_baseline_smoke
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision.models import resnet18

try:
    from src.datasets import create_dataloaders
    from src.experiment_log import ExperimentLog, sha256_file, to_project_relative
    from src.training import evaluate_model, save_model, train_model
    from src.transforms import get_train_transform, get_val_transform
    from src.utils import (
        CLASS_NAMES,
        CSV_DIR,
        MODELS_DIR,
        NUM_CLASSES,
        PROJECT_ROOT,
        RESULTS_DIR,
        get_device,
        set_seed,
    )
except ImportError:
    from .datasets import create_dataloaders
    from .experiment_log import ExperimentLog, sha256_file, to_project_relative
    from .training import evaluate_model, save_model, train_model
    from .transforms import get_train_transform, get_val_transform
    from .utils import (
        CLASS_NAMES,
        CSV_DIR,
        MODELS_DIR,
        NUM_CLASSES,
        PROJECT_ROOT,
        RESULTS_DIR,
        get_device,
        set_seed,
    )


def _sample_split(df: pd.DataFrame, n_max: int, seed: int, label_column: str) -> pd.DataFrame:
    if n_max <= 0 or len(df) <= n_max:
        return df.copy().reset_index(drop=True)

    sampled_parts = []
    for _, group in df.groupby(label_column):
        frac = len(group) / len(df)
        take = max(1, int(round(frac * n_max)))
        sampled_parts.append(group.sample(n=min(take, len(group)), random_state=seed))

    sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=seed)
    if len(sampled) > n_max:
        sampled = sampled.sample(n=n_max, random_state=seed)
    return sampled.reset_index(drop=True)


def _inverse_frequency_class_weights(
    df: pd.DataFrame,
    label_column: str,
    num_classes: int,
    device,
) -> torch.Tensor:
    counts = df[label_column].value_counts().to_dict()
    total = float(len(df))
    weights = []
    for class_idx in range(num_classes):
        count = float(counts.get(class_idx, 0))
        if count <= 0:
            weights.append(0.0)
        else:
            # Balanced heuristic: N / (C * n_c)
            weights.append(total / (num_classes * count))
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    if tensor.sum().item() > 0:
        tensor = tensor / tensor.mean()
    return tensor


def run_baseline_smoke(
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
    max_train: int = 256,
    max_val: int = 96,
    max_test: int = 96,
    class_weighting: str = "none",
):
    set_seed(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"resnet18_15class_smoke_seed{seed}_{timestamp}"
    label_column = "class_label"

    out_dir = os.path.join(RESULTS_DIR, "baseline_smoke")
    sample_dir = os.path.join(out_dir, "sample_splits")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(CSV_DIR, "plantvillage_train.csv")
    val_csv = os.path.join(CSV_DIR, "plantvillage_val.csv")
    test_csv = os.path.join(CSV_DIR, "plantvillage_test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_df_s = _sample_split(train_df, max_train, seed, label_column)
    val_df_s = _sample_split(val_df, max_val, seed, label_column)
    test_df_s = _sample_split(test_df, max_test, seed, label_column)

    train_csv_s = os.path.join(sample_dir, f"train_smoke_{timestamp}.csv")
    val_csv_s = os.path.join(sample_dir, f"val_smoke_{timestamp}.csv")
    test_csv_s = os.path.join(sample_dir, f"test_smoke_{timestamp}.csv")
    train_df_s.to_csv(train_csv_s, index=False)
    val_df_s.to_csv(val_csv_s, index=False)
    test_df_s.to_csv(test_csv_s, index=False)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv_s,
        val_csv=val_csv_s,
        test_csv=test_csv_s,
        train_transform=get_train_transform(),
        val_transform=get_val_transform(),
        label_column=label_column,
        batch_size=batch_size,
        num_workers=0,
    )

    device = get_device()
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    class_weight_values = None
    if class_weighting == "inverse_frequency":
        class_weights = _inverse_frequency_class_weights(
            train_df_s, label_column=label_column, num_classes=NUM_CLASSES, device=device
        )
        class_weight_values = [float(v) for v in class_weights.detach().cpu().tolist()]
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        device=device,
        early_stop_patience=0,
    )

    labels, preds, _ = evaluate_model(model, test_loader, device=device)
    test_acc = float(accuracy_score(labels, preds))
    test_f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    checkpoint_path = os.path.join(MODELS_DIR, f"{run_name}.pth")
    save_model(model, checkpoint_path)

    split_manifest_path = os.path.join(
        RESULTS_DIR, "split_manifests", "latest_split_manifest.json"
    )
    split_manifest_hash = (
        sha256_file(split_manifest_path) if os.path.exists(split_manifest_path) else None
    )

    smoke_note = (
        "Smoke run for pipeline validation only. "
        "Low metrics are expected because training uses random initialization, "
        f"{epochs} epoch(s), and sampled subsets (train={len(train_df_s)})."
    )

    metrics_snapshot = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "label_column": label_column,
        "class_weighting": class_weighting,
        "class_weights": class_weight_values,
        "device": str(device),
        "train_samples": int(len(train_df_s)),
        "val_samples": int(len(val_df_s)),
        "test_samples": int(len(test_df_s)),
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
        "notes": smoke_note,
        "split_manifest_path": to_project_relative(split_manifest_path, repo_dir=PROJECT_ROOT),
        "split_manifest_sha256": split_manifest_hash,
        "history": history,
        "classification_report": report,
        "checkpoint_path": to_project_relative(checkpoint_path, repo_dir=PROJECT_ROOT),
    }

    metrics_path = os.path.join(out_dir, f"metrics_snapshot_{run_name}.json")
    latest_metrics_path = os.path.join(out_dir, "latest_metrics_snapshot.json")

    cm_json_path = os.path.join(out_dir, f"confusion_matrix_{run_name}.json")
    latest_cm_json_path = os.path.join(out_dir, "latest_confusion_matrix.json")
    cm_png_path = os.path.join(out_dir, f"confusion_matrix_{run_name}.png")
    latest_cm_png_path = os.path.join(out_dir, "latest_confusion_matrix.png")

    cm_payload = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "class_names": CLASS_NAMES,
        "matrix": cm.tolist(),
    }
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2)
    with open(latest_cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"Confusion Matrix - {run_name}")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=150)
    plt.close()
    shutil.copyfile(cm_png_path, latest_cm_png_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_snapshot, f, indent=2)
    with open(latest_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_snapshot, f, indent=2)

    log = ExperimentLog(run_name)
    log.set_hyperparams(
        model="resnet18",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        label_column=label_column,
        class_weighting=class_weighting,
        class_weights=class_weight_values,
        sampled_train_rows=int(len(train_df_s)),
        sampled_val_rows=int(len(val_df_s)),
        sampled_test_rows=int(len(test_df_s)),
    )
    log.set_notes(smoke_note)
    log.set_environment()
    log.set_git_commit(repo_dir=PROJECT_ROOT)
    log.set_split_artifacts(
        split_paths={"train": train_csv_s, "val": val_csv_s, "test": test_csv_s},
        seed=seed,
        label_column=label_column,
        repo_dir=PROJECT_ROOT,
    )
    log.set_file_artifact("split_manifest_file", split_manifest_path, repo_dir=PROJECT_ROOT)
    log.set_file_artifact("metrics_snapshot_file", metrics_path, repo_dir=PROJECT_ROOT)
    log.set_file_artifact("confusion_matrix_json_file", cm_json_path, repo_dir=PROJECT_ROOT)
    log.set_file_artifact("confusion_matrix_png_file", cm_png_path, repo_dir=PROJECT_ROOT)
    log.set_file_artifact("model_checkpoint_file", checkpoint_path, repo_dir=PROJECT_ROOT)
    log.set_metrics(
        test_accuracy=test_acc,
        test_f1_macro=test_f1_macro,
        test_samples=int(len(test_df_s)),
    )
    log_path = log.save()
    latest_log_path = os.path.join(out_dir, "latest_experiment_log.json")
    shutil.copyfile(log_path, latest_log_path)

    print(f"Smoke run complete: {run_name}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Metrics snapshot: {metrics_path}")
    print(f"Confusion matrix JSON: {cm_json_path}")
    print(f"Confusion matrix PNG: {cm_png_path}")
    print(f"Experiment log: {log_path}")
    print(f"Latest experiment log: {latest_log_path}")

    return {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "confusion_matrix_json_path": cm_json_path,
        "confusion_matrix_png_path": cm_png_path,
        "log_path": log_path,
        "latest_log_path": latest_log_path,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run script-based baseline smoke training.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-val", type=int, default=96)
    parser.add_argument("--max-test", type=int, default=96)
    parser.add_argument(
        "--class-weighting",
        type=str,
        choices=["none", "inverse_frequency"],
        default="none",
        help="Class-imbalance mitigation mode for CrossEntropyLoss.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline_smoke(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        class_weighting=args.class_weighting,
    )
