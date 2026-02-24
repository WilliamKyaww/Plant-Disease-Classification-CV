"""
PyTorch Dataset classes for plant disease detection.
"""

import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

# General-purpose dataset for plant disease classification
class PlantDiseaseDataset(Dataset):
    """
    Reads a CSV with columns: image_path, crop, disease, label
    and returns (image_tensor, label) pairs.

    The 'label_column' parameter controls which column is used as
    the target label, enabling reuse for:
      - Binary classification (label_column="label")
      - Multi-class disease classification (label_column="disease_label")
      - Severity estimation (label_column="severity")
    """

    def __init__(self, csv_file: str, label_column: str = "label",
                 transform=None, root_dir: str = None):

        self.df = pd.read_csv(csv_file)
        self.label_column = label_column
        self.transform = transform
        self.root_dir = root_dir

        # Drop rows where label is NaN (useful for severity column)
        self.df = self.df.dropna(subset=[label_column]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # Prepend root directory if provided
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        else:
            
            # Paths work on both local and Colab automatically
            if img_path.startswith("Datasets/") or img_path.startswith("Datasets\\"):
                try:
                    from src.utils import DATASETS_DIR
                except ImportError:
                    from .utils import DATASETS_DIR
                relative = img_path.split("/", 1)[1] if "/" in img_path else img_path.split("\\", 1)[1]
                img_path = os.path.join(DATASETS_DIR, relative)

        label = int(row[self.label_column])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_samples(self) -> int:
        return len(self.df)

    @property
    def class_counts(self) -> dict:
        """Return a dictionary of {label: count}."""
        return self.df[self.label_column].value_counts().to_dict()


def create_dataloaders(train_csv: str, val_csv: str, test_csv: str,
                       train_transform, val_transform,
                       label_column: str = "label",
                       batch_size: int = 32, num_workers: int = 0,
                       root_dir: str = None):
   
    from torch.utils.data import DataLoader

    train_ds = PlantDiseaseDataset(train_csv, label_column, train_transform, root_dir)
    val_ds = PlantDiseaseDataset(val_csv, label_column, val_transform, root_dir)
    test_ds = PlantDiseaseDataset(test_csv, label_column, val_transform, root_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
