"""
Lightweight experiment tracking: logs hyperparameters, metrics, and environment 
info to a JSON file per run.

Usage:
    from src.experiment_log import ExperimentLog

    log = ExperimentLog("resnet18_seed42")
    log.set_hyperparams(model="resnet18", lr=1e-4, epochs=30, seed=42)
    log.set_environment()             # auto-captures Python, CUDA, GPU info
    log.set_metrics(accuracy=0.95, f1_macro=0.93)
    log.save()
"""

import json
import os
import platform
import sys
from datetime import datetime

import torch

try:
    from src.utils import RESULTS_DIR
except ImportError:
    from .utils import RESULTS_DIR

LOGS_DIR = os.path.join(RESULTS_DIR, "experiment_logs")


class ExperimentLog:
    def __init__(self, name: str):
        self.name = name
        self.data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "hyperparams": {},
            "metrics": {},
            "environment": {},
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
            "platform": platform.platform(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 2
            )
        self.data["environment"] = env

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
