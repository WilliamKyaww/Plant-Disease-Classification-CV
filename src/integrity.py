"""
Dataset integrity checks:
- missing/empty folders
- corrupt image detection
- exact duplicate detection across classes (SHA256)
- near-duplicate detection across classes (dHash Hamming distance)

Run from repo root:
    python -m src.integrity
"""

import hashlib
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image

try:
    from src.utils import DATASETS_DIR, FOLDER_METADATA
except ImportError:
    from .utils import DATASETS_DIR, FOLDER_METADATA


VALID_EXTS = (".jpg", ".jpeg", ".png")


def _iter_images() -> List[Tuple[str, str]]:
    """Return list of (folder_name, image_path) for all configured class folders."""
    files = []
    for folder in FOLDER_METADATA:
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(VALID_EXTS):
                files.append((folder, os.path.join(folder_path, fname)))
    return files


def file_hash(filepath: str, block_size: int = 65536) -> str:
    """Compute SHA256 hash of a file for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def dhash_int(filepath: str, hash_size: int = 8) -> int:
    """
    Compute dHash (difference hash) as an integer.
    Uses grayscale resize to (hash_size+1, hash_size) and compares adjacent pixels.
    """
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(filepath) as img:
        gray = img.convert("L").resize((hash_size + 1, hash_size), resampling)
        pixels = list(gray.getdata())

    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            bit = 1 if left > right else 0
            value = (value << 1) | bit
    return value


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def check_missing_folders() -> List[str]:
    """Check all expected folders exist and are non-empty."""
    issues = []
    for folder in FOLDER_METADATA:
        path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(path):
            issues.append(f"MISSING folder: {folder}")
        else:
            count = len([f for f in os.listdir(path) if f.lower().endswith(VALID_EXTS)])
            if count == 0:
                issues.append(f"EMPTY folder: {folder}")
    return issues


def check_corrupt_images(verbose: bool = False) -> List[Tuple[str, str]]:
    """Try to open every image; report any that fail."""
    corrupt = []
    total = 0
    for _, fpath in _iter_images():
        total += 1
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception as exc:
            corrupt.append((fpath, str(exc)))
            if verbose:
                print(f"  CORRUPT: {fpath} ({exc})")

    print(f"  Checked {total} images, found {len(corrupt)} corrupt.")
    return corrupt


def check_duplicates_across_classes() -> Dict[str, List[str]]:
    """
    Find exact-duplicate files across different class folders.
    Returns dict: {sha256_hash: [paths...]} for cross-class duplicates only.
    """
    hash_to_files = defaultdict(list)

    for folder, fpath in _iter_images():
        hash_to_files[file_hash(fpath)].append((folder, fpath))

    duplicates = {h: items for h, items in hash_to_files.items() if len(items) > 1}

    cross_class = {}
    for digest, items in duplicates.items():
        folders = {folder for folder, _ in items}
        if len(folders) > 1:
            cross_class[digest] = [p for _, p in items]

    print(f"  Total duplicate groups (exact file hash): {len(duplicates)}")
    print(f"  Cross-class duplicate groups (exact hash, DANGER): {len(cross_class)}")
    return cross_class


def check_near_duplicates_across_classes(
    max_distance: int = 5,
    hash_size: int = 8,
    max_preview_pairs: int = 20,
) -> Dict[str, object]:
    """
    Find visually near-duplicate images across different classes using dHash distance.
    Returns summary dict with total pair count and preview examples.
    """
    records = []
    unreadable = 0

    for folder, fpath in _iter_images():
        try:
            records.append((folder, fpath, dhash_int(fpath, hash_size=hash_size)))
        except Exception:
            unreadable += 1

    total_pairs = 0
    preview = []
    n = len(records)

    # Brute-force pairwise scan. For ~5k images this is tractable for one-off audits.
    for i in range(n):
        folder_a, path_a, hash_a = records[i]
        for j in range(i + 1, n):
            folder_b, path_b, hash_b = records[j]
            if folder_a == folder_b:
                continue
            dist = hamming_distance(hash_a, hash_b)
            if dist <= max_distance:
                total_pairs += 1
                if len(preview) < max_preview_pairs:
                    preview.append(
                        {
                            "distance": dist,
                            "class_a": folder_a,
                            "class_b": folder_b,
                            "path_a": path_a,
                            "path_b": path_b,
                        }
                    )

    print(f"  dHash records computed: {n} (unreadable during hashing: {unreadable})")
    print(
        "  Near-duplicate cross-class pairs "
        f"(dHash distance <= {max_distance}): {total_pairs}"
    )

    return {
        "hash_type": "dHash",
        "hash_size": hash_size,
        "max_distance": max_distance,
        "records": n,
        "unreadable_for_hashing": unreadable,
        "total_pairs": total_pairs,
        "preview_pairs": preview,
    }


def get_class_counts() -> Dict[str, int]:
    """Return image count per configured class folder."""
    counts = {}
    for folder in FOLDER_METADATA:
        folder_path = os.path.join(DATASETS_DIR, folder)
        if not os.path.isdir(folder_path):
            counts[folder] = 0
        else:
            counts[folder] = len(
                [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTS)]
            )
    return counts


def run_all_checks(
    verbose: bool = True,
    run_near_duplicates: bool = True,
    near_duplicate_distance: int = 5,
):
    """Run full dataset integrity audit and return structured results."""
    print("=" * 60)
    print("DATASET INTEGRITY CHECK")
    print("=" * 60)

    print("\n[1/5] Checking folders...")
    issues = check_missing_folders()
    if issues:
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print("  OK: All 15 class folders present and non-empty.")

    print("\n[2/5] Class counts:")
    counts = get_class_counts()
    total = 0
    for folder, count in sorted(counts.items()):
        meta = FOLDER_METADATA[folder]
        print(f"  {meta['clean_name']:40s} {count:>5d}  (class {meta['class_label']})")
        total += count
    print(f"  {'TOTAL':40s} {total:>5d}")

    print("\n[3/5] Checking for corrupt images...")
    corrupt = check_corrupt_images(verbose=verbose)

    print("\n[4/5] Checking exact duplicates across classes...")
    cross_class_exact = check_duplicates_across_classes()
    if cross_class_exact:
        print(f"  WARNING: Found {len(cross_class_exact)} exact cross-class duplicate groups.")
    else:
        print("  OK: No exact cross-class duplicate groups found.")

    near_duplicates = None
    if run_near_duplicates:
        print("\n[5/5] Checking near-duplicates across classes...")
        near_duplicates = check_near_duplicates_across_classes(
            max_distance=near_duplicate_distance
        )
        if near_duplicates["total_pairs"] > 0:
            print(
                "  WARNING: Found near-duplicate cross-class pairs. "
                "Review preview pairs in returned results."
            )
        else:
            print("  OK: No near-duplicate cross-class pairs found.")
    else:
        print("\n[5/5] Near-duplicate check skipped.")

    exact_dupe_fail = len(cross_class_exact) > 0
    near_dupe_fail = bool(near_duplicates and near_duplicates["total_pairs"] > 0)
    passed = (len(issues) == 0) and (len(corrupt) == 0) and (not exact_dupe_fail) and (not near_dupe_fail)

    print("\n" + "=" * 60)
    if passed:
        print("ALL CHECKS PASSED - dataset is clean.")
    else:
        print("ISSUES FOUND - review warnings above before proceeding.")
    print("=" * 60)

    return {
        "missing_folders": issues,
        "corrupt_images": corrupt,
        "cross_class_duplicates_exact": cross_class_exact,
        "cross_class_duplicates_near": near_duplicates,
        "class_counts": counts,
        "total_images": total,
        "passed": passed,
    }


if __name__ == "__main__":
    run_all_checks()
