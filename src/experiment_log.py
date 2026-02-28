"""
Lightweight experiment tracking: logs hyperparameters, metrics, and environment 
info to a JSON file per run.

Usage:
    from src.experiment_log import ExperimentLog

    log = ExperimentLog("resnet18_seed42")
    log.set_hyperparams(model="resnet18", lr=1e-4, epochs=30, seed=42)
    log.set_environment()             # auto-captures Python, CUDA, GPU info
    log.set_git_commit()              # captures current git commit hash
    log.set_split_artifacts(          # captures split hashes + class counts
        split_paths={
            "train": "CSV/plantvillage_train.csv",
            "val": "CSV/plantvillage_val.csv",
            "test": "CSV/plantvillage_test.csv",
        },
        seed=42,
        label_column="class_label",
    )
    log.set_metrics(accuracy=0.95, f1_macro=0.93)
    log.save()
"""

import json
import os
import platform
import sys
import csv
import hashlib
import subprocess
from collections import Counter
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

try:
    from src.utils import RESULTS_DIR
except ImportError:
    from .utils import RESULTS_DIR

LOGS_DIR = os.path.join(RESULTS_DIR, "experiment_logs")


def sha256_file(filepath: str, block_size: int = 65536) -> str:
    """Compute SHA256 hash for an artifact file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def get_git_commit_hash(repo_dir: str = None) -> str:
    """Return current git commit hash, or 'unknown' if unavailable."""
    cwd = repo_dir or os.getcwd()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "unknown"


def summarize_split_csv(csv_path: str, label_column: str = "class_label") -> dict:
    """
    Return row count and label distribution for a split CSV.
    Uses csv module to avoid hard dependency on pandas.
    """
    row_count = 0
    label_counts = Counter()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if label_column in row:
                label_counts[row[label_column]] += 1

    return {
        "rows": row_count,
        "label_counts": dict(sorted(label_counts.items(), key=lambda kv: kv[0])),
    }


class ExperimentLog:
    def __init__(self, name: str):
        self.name = name
        self.data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "hyperparams": {},
            "metrics": {},
            "environment": {},
            "artifacts": {},
            "git_commit": "unknown",
            "notes": "",
        }

    def set_hyperparams(self, **kwargs):
        """Record hyperparameters (model, lr, batch_size, seed, etc.)."""
        self.data["hyperparams"].update(kwargs)

    def set_metrics(self, **kwargs):
        """Record evaluation metrics (accuracy, f1, auc, etc.)."""
        self.data["metrics"].update(kwargs)

    def set_notes(self, notes: str):
        """Add freeform notes about this run."""
        self.data["notes"] = notes

    def set_environment(self):
        """Auto-capture Python version, PyTorch version, CUDA info, GPU model."""
        env = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pytorch_version": torch.__version__ if torch is not None else "not_installed",
            "cuda_available": bool(torch and torch.cuda.is_available()),
        }
        if torch is not None and torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 2
            )
        else:
            env["cuda_version"] = None
            env["gpu_name"] = None
            env["gpu_memory_gb"] = None
        self.data["environment"] = env

    def set_git_commit(self, commit_hash: str = None, repo_dir: str = None):
        """Record git commit hash for reproducibility."""
        self.data["git_commit"] = commit_hash or get_git_commit_hash(repo_dir=repo_dir)

    def set_split_artifacts(
        self,
        split_paths: dict,
        seed: int,
        label_column: str = "class_label",
        repo_dir: str = None,
    ):
        """
        Record split CSV artifacts: path, SHA256, row count, and class counts.
        split_paths format:
            {"train": "...csv", "val": "...csv", "test": "...csv"}
        """
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "label_column": label_column,
            "git_commit": self.data.get("git_commit")
            if self.data.get("git_commit") != "unknown"
            else get_git_commit_hash(repo_dir=repo_dir),
            "splits": {},
        }

        for split_name, path in split_paths.items():
            if not os.path.exists(path):
                manifest["splits"][split_name] = {
                    "path": path,
                    "exists": False,
                }
                continue

            summary = summarize_split_csv(path, label_column=label_column)
            manifest["splits"][split_name] = {
                "path": path,
                "exists": True,
                "sha256": sha256_file(path),
                "rows": summary["rows"],
                "label_counts": summary["label_counts"],
            }

        self.data["artifacts"]["split_manifest"] = manifest

    def save(self, custom_dir: str = None):
        """Save experiment log as JSON."""
        out_dir = custom_dir or LOGS_DIR
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.name}.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Experiment log saved: {path}")
        return path

    def __repr__(self):
        return f"ExperimentLog({self.name})"
