"""
Frozen split manifest validation utilities for Phase 2 benchmarking.
"""

import json
import os
import hashlib

try:
    from src.experiment_log import sha256_file
    from src.utils import PROJECT_ROOT
except ImportError:
    from .experiment_log import sha256_file
    from .utils import PROJECT_ROOT


def resolve_project_path(path: str, project_root: str = PROJECT_ROOT) -> str:
    """Resolve path relative to project root unless already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_hash_variants(filepath: str) -> set[str]:
    """
    Return hash variants that are equivalent under CSV line-ending normalization.
    This makes split validation stable across Windows (CRLF) and Linux (LF).
    """
    with open(filepath, "rb") as f:
        raw = f.read()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        _sha256_bytes(raw),
        _sha256_bytes(lf),
        _sha256_bytes(crlf),
    }


def validate_frozen_splits(
    manifest_path: str,
    project_root: str = PROJECT_ROOT,
    expected_splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict:
    """
    Validate split CSV existence and SHA256 hashes against manifest.

    Returns manifest payload with:
    - manifest_path_abs
    - manifest_sha256
    - split_paths_abs
    """
    manifest_abs = resolve_project_path(manifest_path, project_root=project_root)
    if not os.path.exists(manifest_abs):
        raise FileNotFoundError(f"Split manifest not found: {manifest_abs}")

    with open(manifest_abs, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    split_paths_abs = {}
    for split_name in expected_splits:
        split_info = manifest.get("splits", {}).get(split_name, {})
        rel_path = split_info.get("path")
        expected_sha = split_info.get("sha256")
        if not rel_path or not expected_sha:
            raise ValueError(f"Manifest missing path/sha256 for split '{split_name}'.")

        split_abs = resolve_project_path(rel_path, project_root=project_root)
        if not os.path.exists(split_abs):
            raise FileNotFoundError(f"Split CSV missing: {split_abs}")

        observed_sha = sha256_file(split_abs)
        if observed_sha != expected_sha:
            # Accept line-ending-only differences for CSV files.
            if split_abs.lower().endswith(".csv") and expected_sha in _csv_hash_variants(split_abs):
                print(
                    f"Warning: split '{split_name}' hash differs by line endings only; "
                    "accepting normalized CSV match."
                )
            else:
                raise ValueError(
                    f"Split hash mismatch for '{split_name}'. "
                    f"expected={expected_sha}, observed={observed_sha}"
                )
        split_paths_abs[split_name] = split_abs

    manifest["manifest_path_abs"] = manifest_abs
    manifest["manifest_sha256"] = sha256_file(manifest_abs)
    manifest["split_paths_abs"] = split_paths_abs
    return manifest
