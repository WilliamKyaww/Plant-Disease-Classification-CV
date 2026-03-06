"""
Phase 2 multi-architecture benchmark runner (15-class, frozen splits).

Run from repo root:
    py -3.13 -m src.run_phase2_benchmark --dry-run
"""

import argparse
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from src.datasets import create_dataloaders
    from src.experiment_log import ExperimentLog, sha256_file, to_project_relative
    from src.model_registry import (
        CANONICAL_MODELS,
        build_model,
        canonicalize_model_name,
        count_trainable_parameters,
        is_vit_model,
    )
    from src.phase2_reporting import write_phase2_summary
    from src.split_guard import resolve_project_path, validate_frozen_splits
    from src.training import evaluate_model, save_model, train_model
    from src.transforms import get_train_transform, get_val_transform
    from src.utils import CLASS_NAMES, NUM_CLASSES, PROJECT_ROOT, set_seed, get_device
except ImportError:
    from .datasets import create_dataloaders
    from .experiment_log import ExperimentLog, sha256_file, to_project_relative
    from .model_registry import (
        CANONICAL_MODELS,
        build_model,
        canonicalize_model_name,
        count_trainable_parameters,
        is_vit_model,
    )
    from .phase2_reporting import write_phase2_summary
    from .split_guard import resolve_project_path, validate_frozen_splits
    from .training import evaluate_model, save_model, train_model
    from .transforms import get_train_transform, get_val_transform
    from .utils import CLASS_NAMES, NUM_CLASSES, PROJECT_ROOT, set_seed, get_device


def _parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _inverse_frequency_class_weights(
    train_csv_path: str,
    label_column: str,
    num_classes: int,
    device,
) -> tuple[torch.Tensor | None, list[float] | None]:
    df = pd.read_csv(train_csv_path)
    counts = df[label_column].value_counts().to_dict()
    total = float(len(df))

    weights = []
    for class_idx in range(num_classes):
        count = float(counts.get(class_idx, 0))
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    if tensor.sum().item() <= 0:
        return None, None
    tensor = tensor / tensor.mean()
    return tensor, [float(v) for v in tensor.detach().cpu().tolist()]


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


def _prepare_split_paths_for_seed(
    split_paths: dict,
    label_column: str,
    seed: int,
    out_dir: str,
    max_train: int = 0,
    max_val: int = 0,
    max_test: int = 0,
) -> tuple[dict, dict]:
    """
    Return split paths for this seed.
    If max_* values are > 0, sampled CSVs are generated and returned.
    """
    if max_train <= 0 and max_val <= 0 and max_test <= 0:
        row_counts = {
            "train": int(len(pd.read_csv(split_paths["train"]))),
            "val": int(len(pd.read_csv(split_paths["val"]))),
            "test": int(len(pd.read_csv(split_paths["test"]))),
        }
        return split_paths, row_counts

    sample_dir = os.path.join(out_dir, "sample_splits", f"seed_{seed}")
    os.makedirs(sample_dir, exist_ok=True)
    sampled_paths = {}
    row_counts = {}

    for split_name, n_max in (
        ("train", max_train),
        ("val", max_val),
        ("test", max_test),
    ):
        df = pd.read_csv(split_paths[split_name])
        sampled = _sample_split(df, n_max=n_max, seed=seed, label_column=label_column)
        sampled_path = os.path.join(sample_dir, f"{split_name}.csv")
        sampled.to_csv(sampled_path, index=False)
        sampled_paths[split_name] = sampled_path
        row_counts[split_name] = int(len(sampled))

    return sampled_paths, row_counts


def _build_scheduler(
    scheduler_name: str,
    optimizer,
    epochs: int,
    patience: int,
):
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if scheduler_name == "plateau":
        plateau_patience = max(1, patience // 2)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=plateau_patience,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _select_learning_rate(model_name: str, lr_cnn: float, lr_vit: float) -> float:
    return lr_vit if is_vit_model(model_name) else lr_cnn


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_confusion_matrix_png(cm, path: str, title: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _run_dry_check(args, models: list[str], split_paths: dict, out_dir: str, manifest: dict):
    set_seed(42)
    device = get_device()

    train_loader, _, _ = create_dataloaders(
        train_csv=split_paths["train"],
        val_csv=split_paths["val"],
        test_csv=split_paths["test"],
        train_transform=get_train_transform(),
        val_transform=get_val_transform(),
        label_column="class_label",
        batch_size=min(8, args.batch_size),
        num_workers=args.num_workers,
    )

    sample_inputs, _ = next(iter(train_loader))
    sample_inputs = sample_inputs.to(device)

    model_checks = []
    for model_name in models:
        model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=args.pretrained)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(sample_inputs)
        model_checks.append(
            {
                "model": model_name,
                "params_trainable": int(count_trainable_parameters(model)),
                "selected_lr": _select_learning_rate(model_name, args.lr_cnn, args.lr_vit),
                "forward_shape": list(out.shape),
            }
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": True,
        "models": models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "class_weighting": args.class_weighting,
        "scheduler": args.scheduler,
        "pretrained": args.pretrained,
        "amp_requested": args.amp,
        "device": str(device),
        "manifest_path": to_project_relative(manifest["manifest_path_abs"], repo_dir=PROJECT_ROOT),
        "manifest_sha256": manifest["manifest_sha256"],
        "split_paths": {
            k: to_project_relative(v, repo_dir=PROJECT_ROOT) for k, v in split_paths.items()
        },
        "model_checks": model_checks,
    }

    report_path = os.path.join(out_dir, "phase2_dry_run_report.json")
    _write_json(report_path, report)
    print(f"Dry-run report saved: {report_path}")


def _run_benchmark(args, models: list[str], split_paths: dict, out_dir: str, manifest: dict):
    device = get_device()
    label_column = "class_label"
    run_rows = []
    split_paths_by_seed = {}

    # Build split paths once per seed so every architecture uses identical files.
    for seed in args.seeds:
        seed_split_paths, sampled_row_counts = _prepare_split_paths_for_seed(
            split_paths=split_paths,
            label_column=label_column,
            seed=seed,
            out_dir=out_dir,
            max_train=args.max_train,
            max_val=args.max_val,
            max_test=args.max_test,
        )
        split_paths_by_seed[seed] = {
            "paths": seed_split_paths,
            "row_counts": sampled_row_counts,
            "sha256": {k: sha256_file(v) for k, v in seed_split_paths.items()},
        }

    for model_name in models:
        model_folder = os.path.join(out_dir, "runs", model_name)
        os.makedirs(model_folder, exist_ok=True)

        for seed in args.seeds:
            run_dir = os.path.join(model_folder, f"seed_{seed}")
            metrics_path = os.path.join(run_dir, "metrics.json")
            if args.resume and os.path.exists(metrics_path):
                print(f"Skipping existing run: {model_name} seed {seed}")
                continue

            os.makedirs(run_dir, exist_ok=True)
            set_seed(seed)
            run_split_info = split_paths_by_seed[seed]
            run_split_paths = run_split_info["paths"]
            sampled_row_counts = run_split_info["row_counts"]
            split_hashes = run_split_info["sha256"]

            train_loader, val_loader, test_loader = create_dataloaders(
                train_csv=run_split_paths["train"],
                val_csv=run_split_paths["val"],
                test_csv=run_split_paths["test"],
                train_transform=get_train_transform(),
                val_transform=get_val_transform(),
                label_column=label_column,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=args.pretrained)
            model = model.to(device)
            selected_lr = _select_learning_rate(model_name, args.lr_cnn, args.lr_vit)

            class_weights_tensor = None
            class_weights_values = None
            if args.class_weighting == "inverse_frequency":
                class_weights_tensor, class_weights_values = _inverse_frequency_class_weights(
                    train_csv_path=run_split_paths["train"],
                    label_column=label_column,
                    num_classes=NUM_CLASSES,
                    device=device,
                )

            criterion = (
                nn.CrossEntropyLoss(weight=class_weights_tensor)
                if class_weights_tensor is not None
                else nn.CrossEntropyLoss()
            )
            optimizer = optim.AdamW(
                model.parameters(),
                lr=selected_lr,
                weight_decay=args.weight_decay,
            )
            scheduler = _build_scheduler(
                scheduler_name=args.scheduler,
                optimizer=optimizer,
                epochs=args.epochs,
                patience=args.patience,
            )

            t0 = time.perf_counter()
            model, history = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.epochs,
                device=device,
                scheduler=scheduler,
                early_stop_patience=args.patience,
                use_amp=args.amp,
            )
            labels, preds, _ = evaluate_model(model, test_loader, device=device)
            elapsed_sec = time.perf_counter() - t0

            test_acc = float(accuracy_score(labels, preds))
            test_f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
            report = classification_report(
                labels,
                preds,
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

            checkpoint_path = os.path.join(
                PROJECT_ROOT,
                "models",
                "phase2",
                model_name,
                f"seed_{seed}",
                "best.pth",
            )
            save_model(model, checkpoint_path)
            checkpoint_size_bytes = int(os.path.getsize(checkpoint_path))

            cm_json_path = os.path.join(run_dir, "confusion_matrix.json")
            cm_png_path = os.path.join(run_dir, "confusion_matrix.png")
            _write_json(
                cm_json_path,
                {
                    "model": model_name,
                    "seed": seed,
                    "class_names": CLASS_NAMES,
                    "matrix": cm.tolist(),
                },
            )
            _save_confusion_matrix_png(
                cm,
                cm_png_path,
                title=f"Phase 2 - {model_name} - seed {seed}",
            )

            metrics_payload = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": selected_lr,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "patience": args.patience,
                "class_weighting": args.class_weighting,
                "class_weights": class_weights_values,
                "pretrained": args.pretrained,
                "amp_requested": args.amp,
                "amp_active": bool(args.amp and str(device).startswith("cuda")),
                "device": str(device),
                "train_samples": int(len(train_loader.dataset)),
                "val_samples": int(len(val_loader.dataset)),
                "test_samples": int(len(test_loader.dataset)),
                "sampled_row_counts": sampled_row_counts,
                "split_hashes": split_hashes,
                "source_split_paths": {
                    k: to_project_relative(v, repo_dir=PROJECT_ROOT)
                    for k, v in run_split_paths.items()
                },
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1_macro,
                "elapsed_seconds": float(elapsed_sec),
                "trainable_params": int(count_trainable_parameters(model)),
                "model_size_bytes": checkpoint_size_bytes,
                "manifest_path": to_project_relative(
                    manifest["manifest_path_abs"], repo_dir=PROJECT_ROOT
                ),
                "manifest_sha256": manifest["manifest_sha256"],
                "history": history,
                "classification_report": report,
                "checkpoint_path": to_project_relative(checkpoint_path, repo_dir=PROJECT_ROOT),
                "confusion_matrix_json_path": to_project_relative(cm_json_path, repo_dir=PROJECT_ROOT),
                "confusion_matrix_png_path": to_project_relative(cm_png_path, repo_dir=PROJECT_ROOT),
            }
            _write_json(metrics_path, metrics_payload)

            log = ExperimentLog(f"phase2_{model_name}_seed{seed}")
            log.set_hyperparams(
                model=model_name,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=selected_lr,
                weight_decay=args.weight_decay,
                scheduler=args.scheduler,
                patience=args.patience,
                class_weighting=args.class_weighting,
                class_weights=class_weights_values,
                pretrained=args.pretrained,
                amp_requested=args.amp,
                amp_active=bool(args.amp and str(device).startswith("cuda")),
            )
            log.set_notes("Phase 2 benchmark run")
            log.set_environment()
            log.set_git_commit(repo_dir=PROJECT_ROOT)
            log.set_split_artifacts(
                split_paths=run_split_paths,
                seed=int(manifest.get("seed", 42)),
                label_column=label_column,
                repo_dir=PROJECT_ROOT,
            )
            log.set_file_artifact(
                "split_manifest_file",
                manifest["manifest_path_abs"],
                repo_dir=PROJECT_ROOT,
            )
            log.set_file_artifact("metrics_file", metrics_path, repo_dir=PROJECT_ROOT)
            log.set_file_artifact("confusion_matrix_json_file", cm_json_path, repo_dir=PROJECT_ROOT)
            log.set_file_artifact("confusion_matrix_png_file", cm_png_path, repo_dir=PROJECT_ROOT)
            log.set_file_artifact("model_checkpoint_file", checkpoint_path, repo_dir=PROJECT_ROOT)
            log.set_metrics(
                test_accuracy=test_acc,
                test_f1_macro=test_f1_macro,
                elapsed_seconds=float(elapsed_sec),
            )
            log.save(custom_dir=run_dir)

            run_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "test_accuracy": test_acc,
                    "test_f1_macro": test_f1_macro,
                    "elapsed_seconds": float(elapsed_sec),
                    "trainable_params": int(count_trainable_parameters(model)),
                    "model_size_bytes": checkpoint_size_bytes,
                    "metrics_path": to_project_relative(metrics_path, repo_dir=PROJECT_ROOT),
                }
            )

    if not run_rows:
        print("No new runs executed.")
        return

    paths = write_phase2_summary(run_rows=run_rows, out_dir=out_dir)
    print(f"Saved: {paths['model_seed_csv']}")
    print(f"Saved: {paths['summary_csv']}")
    print(f"Saved: {paths['leaderboard_csv']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 2 multi-architecture benchmark.")
    parser.add_argument(
        "--models",
        type=str,
        default="resnet18,resnet50,efficientnet_b0,vit_small_patch16_224",
    )
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-cnn", type=float, default=3e-4)
    parser.add_argument("--lr-vit", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--class-weighting",
        type=str,
        choices=["none", "inverse_frequency"],
        default="inverse_frequency",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "plateau"],
        default="cosine",
    )
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained_group.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("results", "phase2"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.path.join("results", "split_manifests", "latest_split_manifest.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.amp:
        device_preview = get_device()
        if str(device_preview).startswith("cuda"):
            print("AMP requested: enabled for CUDA training.")
        else:
            print("AMP requested, but CUDA is unavailable. Training will run in FP32.")

    model_names_raw = _parse_csv_list(args.models)
    seed_values = [int(x) for x in _parse_csv_list(args.seeds)]
    if not model_names_raw:
        raise ValueError("No models provided via --models.")
    if not seed_values:
        raise ValueError("No seeds provided via --seeds.")

    models = [canonicalize_model_name(name) for name in model_names_raw]
    unsupported = [name for name in models if name not in CANONICAL_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported models after canonicalization: {unsupported}")

    args.seeds = seed_values
    out_dir = resolve_project_path(args.out_dir, project_root=PROJECT_ROOT)
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = resolve_project_path(args.manifest_path, project_root=PROJECT_ROOT)
    manifest = validate_frozen_splits(manifest_path=manifest_path, project_root=PROJECT_ROOT)
    split_paths = manifest["split_paths_abs"]

    print("Phase 2 configuration")
    print("-" * 60)
    print(f"Models: {models}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Class weighting: {args.class_weighting}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Out dir: {out_dir}")

    if args.dry_run:
        _run_dry_check(args=args, models=models, split_paths=split_paths, out_dir=out_dir, manifest=manifest)
    else:
        _run_benchmark(args=args, models=models, split_paths=split_paths, out_dir=out_dir, manifest=manifest)


if __name__ == "__main__":
    main()
