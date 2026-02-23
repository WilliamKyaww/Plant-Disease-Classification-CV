"""
Data augmentation and transform pipelines.
"""

import torchvision.transforms as T
from src.utils import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


def get_train_transform(strong: bool = False):
    """
    Training transform with data augmentation.

    Args:
        strong: If True, use aggressive augmentation (ColorJitter, Affine, etc.)
    """
    transforms = [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(15),
    ]

    if strong:
        transforms.extend([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
        ])

    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return T.Compose(transforms)


def get_val_transform():
    """Validation/test transform (no augmentation, just resize + normalize)."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inverse_normalize():
    """
    Returns a transform that undoes ImageNet normalization.
    Useful for visualizing images (e.g., Grad-CAM overlays).
    """
    return T.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1.0 / s for s in IMAGENET_STD],
    )
