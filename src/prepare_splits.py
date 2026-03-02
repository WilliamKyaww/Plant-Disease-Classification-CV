"""
Generate multi-class label CSV and stratified train/val/test splits.

Produces:
  CSV/plantvillage_multiclass_labels.csv   (all images)
  CSV/plantvillage_train.csv               (70%)
  CSV/plantvillage_val.csv                 (15%)
  CSV/plantvillage_test.csv                (15%)

Run from repo root:
    python -m src.prepare_splits
"""

import os
import csv
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.utils import PROJECT_ROOT, DATASETS_DIR, CSV_DIR, RESULTS_DIR, FOLDER_METADATA, set_seed, ensure_dirs
    from src.experiment_log import sha256_file, get_git_commit_hash, to_project_relative
except ImportError:
    from .utils import PROJECT_ROOT, DATASETS_DIR, CSV_DIR, RESULTS_DIR, FOLDER_METADATA, set_seed, ensure_dirs
    from .experiment_log import sha256_file, get_git_commit_hash, to_project_relative


SEED = 42


def build_label_csv() -> pd.DataFrame:
    """Scan all class folders and build a labelled DataFrame."""
    rows = []

    for folder, meta in FOLDER_METADATA.items():
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"  Warning: {folder_path} does not exist. Skipping.")
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Always use forward slashes in CSV for cross-platform portability.
            image_path = f"Datasets/{folder}/{fname}"
            rows.append({
                "image_path": image_path,
                "crop": meta["crop"],
                "disease": meta["disease"],
                "binary_label": meta["binary_label"],
                "class_label": meta["class_label"],
                "clean_name": meta["clean_name"],
            })

    df = pd.DataFrame(rows)
    print(f"Built label CSV: {len(df)} images across {df['class_label'].nunique()} classes.")
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Create stratified 70/15/15 train/val/test splits."""
    set_seed(seed)

    # Stratify by class_label to preserve class distribution
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["class_label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["class_label"], random_state=seed
    )

    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def verify_no_leakage(train_df, val_df, test_df):
    """Verify no image paths appear in multiple splits."""
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    tv_overlap = train_paths & val_paths
    tt_overlap = train_paths & test_paths
    vt_overlap = val_paths & test_paths

    if tv_overlap or tt_overlap or vt_overlap:
        print(f"  LEAKAGE DETECTED!")
        print(f"    Train-Val overlap: {len(tv_overlap)}")
        print(f"    Train-Test overlap: {len(tt_overlap)}")
        print(f"    Val-Test overlap: {len(vt_overlap)}")
        return False
    else:
        print("  No leakage detected - splits are clean.")
        return True


def main():
    ensure_dirs()

    print("=" * 50)
    print("BUILDING MULTI-CLASS LABELS + SPLITS")
    print("=" * 50)

    # Build full label CSV
    df = build_label_csv()
    all_csv = os.path.join(CSV_DIR, "plantvillage_multiclass_labels.csv")
    df.to_csv(all_csv, index=False)
    print(f"Saved: {all_csv}")

    # Stratified splits
    print()
    train_df, val_df, test_df = make_splits(df)

    # Save
    train_csv = os.path.join(CSV_DIR, "plantvillage_train.csv")
    val_csv = os.path.join(CSV_DIR, "plantvillage_val.csv")
    test_csv = os.path.join(CSV_DIR, "plantvillage_test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved: {train_csv}, {val_csv}, {test_csv}")

    # Leakage check
    print()
    verify_no_leakage(train_df, val_df, test_df)

    # Class distribution per split
    print("\nClass distribution per split:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {name}:")
        for label in sorted(split_df["class_label"].unique()):
            count = len(split_df[split_df["class_label"] == label])
            clean = split_df[split_df["class_label"] == label]["clean_name"].iloc[0]
            print(f"    {clean:40s} {count:>5d}")

    # Save split manifest artifact (hash + date + seed + counts + commit).
    manifest_dir = os.path.join(RESULTS_DIR, "split_manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(manifest_dir, f"split_manifest_seed{SEED}_{timestamp}.json")
    latest_path = os.path.join(manifest_dir, "latest_split_manifest.json")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "git_commit": get_git_commit_hash(repo_dir=os.path.dirname(CSV_DIR)),
        "label_column": "class_label",
        "splits": {
            "train": {
                "path": to_project_relative(train_csv, repo_dir=PROJECT_ROOT),
                "sha256": sha256_file(train_csv),
                "rows": int(len(train_df)),
                "class_counts": train_df["class_label"].value_counts().sort_index().to_dict(),
            },
            "val": {
                "path": to_project_relative(val_csv, repo_dir=PROJECT_ROOT),
                "sha256": sha256_file(val_csv),
                "rows": int(len(val_df)),
                "class_counts": val_df["class_label"].value_counts().sort_index().to_dict(),
            },
            "test": {
                "path": to_project_relative(test_csv, repo_dir=PROJECT_ROOT),
                "sha256": sha256_file(test_csv),
                "rows": int(len(test_df)),
                "class_counts": test_df["class_label"].value_counts().sort_index().to_dict(),
            },
        },
        "all_labels_csv": {
            "path": to_project_relative(all_csv, repo_dir=PROJECT_ROOT),
            "sha256": sha256_file(all_csv),
            "rows": int(len(df)),
            "class_counts": df["class_label"].value_counts().sort_index().to_dict(),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved split manifest: {manifest_path}")
    print(f"Updated latest manifest: {latest_path}")
    print("\nDone. Splits are FROZEN - do not regenerate.")


if __name__ == "__main__":
    main()

