"""
Dataset integrity checks: duplicate detection, corrupt file audit,
and class count verification.

Run from the Main/ directory:
    python -m src.integrity
"""

import os
import hashlib
from collections import defaultdict
from PIL import Image

try:
    from src.utils import DATASETS_DIR, FOLDER_METADATA
except ImportError:
    from .utils import DATASETS_DIR, FOLDER_METADATA


def file_hash(filepath: str, block_size: int = 65536) -> str:
    """Compute MD5 hash of a file (fast, sufficient for duplicate detection)."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_missing_folders() -> list:
    """Check all expected folders exist and are non-empty."""
    issues = []
    for folder in FOLDER_METADATA:
        path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(path):
            issues.append(f"MISSING folder: {folder}")
        else:
            count = len([f for f in os.listdir(path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if count == 0:
                issues.append(f"EMPTY folder: {folder}")
    return issues


def check_corrupt_images(verbose: bool = False) -> list:
    """Try to open every image; report any that fail."""
    corrupt = []
    total = 0
    for folder in FOLDER_METADATA:
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            total += 1
            fpath = os.path.join(folder_path, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception as e:
                corrupt.append((fpath, str(e)))
                if verbose:
                    print(f"  CORRUPT: {fpath} ({e})")
    print(f"  Checked {total} images, found {len(corrupt)} corrupt.")
    return corrupt


def check_duplicates_across_classes() -> dict:
    """
    Find exact-duplicate images across different class folders.
    Returns dict of {hash: [list of file paths]}.
    """
    hash_to_files = defaultdict(list)
    total = 0

    for folder in FOLDER_METADATA:
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(folder_path, fname)
            h = file_hash(fpath)
            hash_to_files[h].append(fpath)
            total += 1

    # Only keep entries with duplicates
    duplicates = {h: paths for h, paths in hash_to_files.items()
                  if len(paths) > 1}

    # Count cross-class duplicates specifically
    cross_class = {}
    for h, paths in duplicates.items():
        folders = set(os.path.basename(os.path.dirname(p)) for p in paths)
        if len(folders) > 1:
            cross_class[h] = paths

    print(f"  Hashed {total} images.")
    print(f"  Total duplicate groups (same file, any location): {len(duplicates)}")
    print(f"  Cross-class duplicate groups (DANGER): {len(cross_class)}")

    return cross_class


def get_class_counts() -> dict:
    """Return {folder_name: image_count} for all classes."""
    counts = {}
    for folder in FOLDER_METADATA:
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(folder_path):
            counts[folder] = 0
        else:
            counts[folder] = len([f for f in os.listdir(folder_path)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return counts


def run_all_checks(verbose: bool = True):
    """Run complete dataset integrity audit."""
    print("=" * 60)
    print("DATASET INTEGRITY CHECK")
    print("=" * 60)

    # 1. Missing/empty folders
    print("\n[1/4] Checking folders...")
    issues = check_missing_folders()
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ All 15 class folders present and non-empty.")

    # 2. Class counts
    print("\n[2/4] Class counts:")
    counts = get_class_counts()
    total = 0
    for folder, count in sorted(counts.items()):
        meta = FOLDER_METADATA[folder]
        print(f"  {meta['clean_name']:40s} {count:>5d}  (class {meta['class_label']})")
        total += count
    print(f"  {'TOTAL':40s} {total:>5d}")

    # 3. Corrupt images
    print("\n[3/4] Checking for corrupt images...")
    corrupt = check_corrupt_images(verbose=verbose)

    # 4. Cross-class duplicates
    print("\n[4/4] Checking for cross-class duplicates (this may take a minute)...")
    cross_class = check_duplicates_across_classes()
    if cross_class:
        print(f"\n  ⚠️  Found {len(cross_class)} cross-class duplicate groups!")
        for h, paths in list(cross_class.items())[:5]:
            print(f"\n  Hash: {h}")
            for p in paths:
                print(f"    - {p}")
        if len(cross_class) > 5:
            print(f"  ... and {len(cross_class) - 5} more groups.")
    else:
        print("  ✅ No cross-class duplicates found.")

    # Summary
    print("\n" + "=" * 60)
    passed = (len(issues) == 0 and len(corrupt) == 0 and len(cross_class) == 0)
    if passed:
        print("✅ ALL CHECKS PASSED — dataset is clean.")
    else:
        print("⚠️  ISSUES FOUND — review above before proceeding.")
    print("=" * 60)

    return {
        "missing_folders": issues,
        "corrupt_images": corrupt,
        "cross_class_duplicates": cross_class,
        "class_counts": counts,
        "total_images": total,
        "passed": passed,
    }


if __name__ == "__main__":
    run_all_checks()
