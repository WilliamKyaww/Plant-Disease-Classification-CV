"""
Model registry for Phase 2 architecture benchmarking.
"""

import torch.nn as nn
import timm
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


CANONICAL_MODELS = (
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "vit_small_patch16_224",
)

MODEL_ALIASES = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "vit-small": "vit_small_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_small_patch16_224": "vit_small_patch16_224",
}


def canonicalize_model_name(model_name: str) -> str:
    """Map aliases to canonical model names used by this project."""
    key = model_name.strip().lower()
    if key not in MODEL_ALIASES:
        supported = ", ".join(CANONICAL_MODELS)
        raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}")
    return MODEL_ALIASES[key]


def is_vit_model(model_name: str) -> bool:
    """Return True for ViT-family models."""
    return canonicalize_model_name(model_name).startswith("vit_")


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Build a supported model with the classifier head set to num_classes.
    """
    name = canonicalize_model_name(model_name)

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        return timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    if name == "vit_small_patch16_224":
        return timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    # Defensive fallback if canonical list is changed but logic is not.
    raise ValueError(f"Unsupported model name: {name}")


def count_trainable_parameters(model) -> int:
    """Count trainable parameters only."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
