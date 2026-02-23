"""
Configuration constants, seed management, and path helpers.
"""

import os
import random
import torch
import numpy as np


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Datasets")
CSV_DIR = os.path.join(PROJECT_ROOT, "CSV")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ─── Dataset folders ─────────────────────────────────────────────────────────

# All crop-disease folder names in the Datasets/ directory
ALL_FOLDERS = [
    "BellPepper_Healthy",
    "BellPepper_BacterialSpot",
    "Tomato_Healthy",
    "Tomato_EarlyBlight",
    "Tomato_LateBlight",
    "Potato_Healthy",
    "Potato_EarlyBlight",
    "Potato_LateBlight",
]

# Multi-class label mapping: disease name -> integer
DISEASE_CLASSES = {
    "healthy": 0,
    "bacterialspot": 1,
    "earlyblight": 2,
    "lateblight": 3,
}

NUM_DISEASE_CLASSES = len(DISEASE_CLASSES)

# Crop types
CROP_CLASSES = {
    "bellpepper": 0,
    "tomato": 1,
    "potato": 2,
}


# ─── ImageNet normalization ──────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Directory helpers ────────────────────────────────────────────────────────

def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [CSV_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
