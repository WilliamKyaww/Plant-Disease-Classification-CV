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

FOLDER_METADATA = {
    "Pepper__bell___healthy": {"crop": "bellpepper", "disease": "healthy", "label": 0, "clean_name": "Bell Pepper - Healthy"},
    "Pepper__bell___Bacterial_spot": {"crop": "bellpepper", "disease": "bacterialspot", "label": 1, "clean_name": "Bell Pepper - Bacterial Spot"},
    "Tomato_healthy": {"crop": "tomato", "disease": "healthy", "label": 0, "clean_name": "Tomato - Healthy"},
    "Tomato_Early_blight": {"crop": "tomato", "disease": "earlyblight", "label": 1, "clean_name": "Tomato - Early Blight"},
    "Tomato_Late_blight": {"crop": "tomato", "disease": "lateblight", "label": 1, "clean_name": "Tomato - Late Blight"},
    "Potato___healthy": {"crop": "potato", "disease": "healthy", "label": 0, "clean_name": "Potato - Healthy"},
    "Potato___Early_blight": {"crop": "potato", "disease": "earlyblight", "label": 1, "clean_name": "Potato - Early Blight"},
    "Potato___Late_blight": {"crop": "potato", "disease": "lateblight", "label": 1, "clean_name": "Potato - Late Blight"},
}

# All crop-disease folder names in the Datasets/ directory
ALL_FOLDERS = list(FOLDER_METADATA.keys())

# Mapping for pretty presentation
CLASS_NAME_MAPPING = {k: v["clean_name"] for k, v in FOLDER_METADATA.items()}

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
