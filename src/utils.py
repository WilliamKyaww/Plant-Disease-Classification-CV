"""
Configuration constants, seed management, and path helpers.
"""

import os
import random
try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


# ─── Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(PROJECT_ROOT, "CSV")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset path: auto-detect Colab, env var override, or local default
_COLAB_DATASETS = "/content/Datasets"
_LOCAL_DATASETS = os.path.join(PROJECT_ROOT, "Datasets")

if os.environ.get("DATASETS_DIR"):
    DATASETS_DIR = os.environ["DATASETS_DIR"]
elif os.path.isdir(_COLAB_DATASETS):
    DATASETS_DIR = _COLAB_DATASETS
else:
    DATASETS_DIR = _LOCAL_DATASETS

# Google Colab?
IS_COLAB = os.path.isdir("/content")


# ─── Dataset folders ────────────────────────────────────────────────
# Each entry maps a PlantVillage folder name to its metadata.
#   crop:          plant species
#   disease:       specific disease (or "healthy")
#   binary_label:  0 = healthy, 1 = diseased
#   class_label:   integer for multi-class classification (15 classes)
#   clean_name:    human-readable display name

FOLDER_METADATA = {
    # ── Bell Pepper (2 classes) ──
    "Pepper__bell___healthy":        {"crop": "pepper",  "disease": "healthy",            "binary_label": 0, "class_label": 0,  "clean_name": "Pepper - Healthy"},
    "Pepper__bell___Bacterial_spot": {"crop": "pepper",  "disease": "bacterial_spot",     "binary_label": 1, "class_label": 1,  "clean_name": "Pepper - Bacterial Spot"},

    # ── Potato (3 classes) ──
    "Potato___healthy":              {"crop": "potato",  "disease": "healthy",            "binary_label": 0, "class_label": 2,  "clean_name": "Potato - Healthy"},
    "Potato___Early_blight":         {"crop": "potato",  "disease": "early_blight",       "binary_label": 1, "class_label": 3,  "clean_name": "Potato - Early Blight"},
    "Potato___Late_blight":          {"crop": "potato",  "disease": "late_blight",        "binary_label": 1, "class_label": 4,  "clean_name": "Potato - Late Blight"},

    # ── Tomato (10 classes) ──
    "Tomato_healthy":                                    {"crop": "tomato",  "disease": "healthy",            "binary_label": 0, "class_label": 5,  "clean_name": "Tomato - Healthy"},
    "Tomato_Bacterial_spot":                             {"crop": "tomato",  "disease": "bacterial_spot",     "binary_label": 1, "class_label": 6,  "clean_name": "Tomato - Bacterial Spot"},
    "Tomato_Early_blight":                               {"crop": "tomato",  "disease": "early_blight",       "binary_label": 1, "class_label": 7,  "clean_name": "Tomato - Early Blight"},
    "Tomato_Late_blight":                                {"crop": "tomato",  "disease": "late_blight",        "binary_label": 1, "class_label": 8,  "clean_name": "Tomato - Late Blight"},
    "Tomato_Leaf_Mold":                                  {"crop": "tomato",  "disease": "leaf_mold",          "binary_label": 1, "class_label": 9,  "clean_name": "Tomato - Leaf Mold"},
    "Tomato_Septoria_leaf_spot":                          {"crop": "tomato",  "disease": "septoria_leaf_spot", "binary_label": 1, "class_label": 10, "clean_name": "Tomato - Septoria Leaf Spot"},
    "Tomato_Spider_mites_Two_spotted_spider_mite":        {"crop": "tomato",  "disease": "spider_mites",       "binary_label": 1, "class_label": 11, "clean_name": "Tomato - Spider Mites"},
    "Tomato__Target_Spot":                               {"crop": "tomato",  "disease": "target_spot",        "binary_label": 1, "class_label": 12, "clean_name": "Tomato - Target Spot"},
    "Tomato__Tomato_YellowLeaf__Curl_Virus":              {"crop": "tomato",  "disease": "yellow_leaf_curl",   "binary_label": 1, "class_label": 13, "clean_name": "Tomato - Yellow Leaf Curl Virus"},
    "Tomato__Tomato_mosaic_virus":                        {"crop": "tomato",  "disease": "mosaic_virus",       "binary_label": 1, "class_label": 14, "clean_name": "Tomato - Mosaic Virus"},
}

# Derived constants
ALL_FOLDERS = list(FOLDER_METADATA.keys())
NUM_CLASSES = len(FOLDER_METADATA)  # 15

# Pretty class names in label order
CLASS_NAMES = [None] * NUM_CLASSES
for folder, meta in FOLDER_METADATA.items():
    CLASS_NAMES[meta["class_label"]] = meta["clean_name"]

# Reverse lookup: class_label -> folder name
LABEL_TO_FOLDER = {meta["class_label"]: folder for folder, meta in FOLDER_METADATA.items()}

# Crop mapping
CROP_CLASSES = {
    "pepper": 0,
    "potato": 1,
    "tomato": 2,
}
NUM_CROPS = len(CROP_CLASSES)


# ─── ImageNet normalization ─────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


# ─── Reproducibility ────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Return the best available device (CUDA > CPU), or 'cpu' if torch unavailable."""
    if torch is None:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Directory helpers ──────────────────────────────────────────────

def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [CSV_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
