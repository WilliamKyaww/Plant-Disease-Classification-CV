"""
Run dataset integrity checks and persist report artifacts (JSON + text).

Run from repo root:
    py -3.13 -m src.integrity_report
"""

import argparse
import json
import os
from datetime import datetime

try:
    from src.integrity import run_all_checks
    from src.utils import DATASETS_DIR, PROJECT_ROOT, RESULTS_DIR
    from src.experiment_log import get_git_commit_hash, to_project_relative
except ImportError:
    from .integrity import run_all_checks
    from .utils import DATASETS_DIR, PROJECT_ROOT, RESULTS_DIR
    from .experiment_log import get_git_commit_hash, to_project_relative


def _relativize_paths(obj):
    """
    Recursively rewrite project-local absolute paths to repo-relative strings
    for portability in report artifacts.
    """
    if isinstance(obj, dict):
        return {k: _relativize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize_paths(v) for v in obj]
    if isinstance(obj, str):
        looks_like_path = (
            "\\" in obj
            or "/" in obj
            or (len(obj) > 2 and obj[1] == ":" and obj[2] in ("\\", "/"))
            or obj.startswith(".")
        )
        if looks_like_path:
            return to_project_relative(obj, repo_dir=PROJECT_ROOT)
        return obj
    return obj


def _build_text_summary(payload: dict) -> str:
    """Build a concise human-readable integrity summary."""
    report = payload["report"]
    near = report.get("cross_class_duplicates_near")

    lines = []
    lines.append("Dataset Integrity Report")
    lines.append("=" * 60)
    lines.append(f"Timestamp: {payload['timestamp']}")
    lines.append(f"Git commit: {payload['git_commit']}")
    lines.append(f"Dataset dir: {payload['dataset_dir']}")
    lines.append(f"Passed: {report['passed']}")
    lines.append("")

    lines.append("Counts")
    lines.append("-" * 60)
    lines.append(f"Total images: {report['total_images']}")
    lines.append(f"Missing/empty folders: {len(report['missing_folders'])}")
    lines.append(f"Corrupt images: {len(report['corrupt_images'])}")
    lines.append(
        "Exact cross-class duplicate groups: "
        f"{len(report['cross_class_duplicates_exact'])}"
    )
    if near is None:
        lines.append("Near-duplicate scan: skipped")
    else:
        lines.append(
            "Near-duplicate cross-class pairs "
            f"(dHash <= {near['max_distance']}): {near['total_pairs']}"
        )
    lines.append("")

    if report["missing_folders"]:
        lines.append("Missing/Empty Folders")
        lines.append("-" * 60)
        for item in report["missing_folders"]:
            lines.append(f"- {item}")
        lines.append("")

    if report["corrupt_images"]:
        lines.append("Corrupt Images (first 20)")
        lines.append("-" * 60)
        for path, err in report["corrupt_images"][:20]:
            lines.append(f"- {path} :: {err}")
        lines.append("")

    if report["cross_class_duplicates_exact"]:
        lines.append("Exact Cross-Class Duplicate Groups (first 5)")
        lines.append("-" * 60)
        for digest, paths in list(report["cross_class_duplicates_exact"].items())[:5]:
            lines.append(f"hash={digest}")
            for p in paths:
                lines.append(f"  - {p}")
        lines.append("")

    if near and near["preview_pairs"]:
        lines.append("Near-Duplicate Preview Pairs (first 20)")
        lines.append("-" * 60)
        for pair in near["preview_pairs"]:
            lines.append(
                f"- d={pair['distance']} | {pair['class_a']} <-> {pair['class_b']}"
            )
            lines.append(f"  A: {pair['path_a']}")
            lines.append(f"  B: {pair['path_b']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_and_save_report(
    run_near_duplicates: bool = True,
    near_duplicate_distance: int = 5,
    out_dir: str = None,
    verbose: bool = False,
) -> dict:
    """
    Run integrity checks and write timestamped + latest report artifacts.

    Returns metadata with written file paths.
    """
    report = run_all_checks(
        verbose=verbose,
        run_near_duplicates=run_near_duplicates,
        near_duplicate_distance=near_duplicate_distance,
    )

    output_dir = out_dir or os.path.join(RESULTS_DIR, "integrity_reports")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(repo_dir=os.path.dirname(RESULTS_DIR)),
        "dataset_dir": to_project_relative(DATASETS_DIR, repo_dir=PROJECT_ROOT),
        "run_config": {
            "run_near_duplicates": run_near_duplicates,
            "near_duplicate_distance": near_duplicate_distance,
            "verbose": verbose,
        },
        "report": _relativize_paths(report),
    }

    json_path = os.path.join(output_dir, f"integrity_report_{timestamp}.json")
    txt_path = os.path.join(output_dir, f"integrity_report_{timestamp}.txt")
    latest_json = os.path.join(output_dir, "latest_integrity_report.json")
    latest_txt = os.path.join(output_dir, "latest_integrity_report.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = _build_text_summary(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    with open(latest_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Saved integrity JSON: {json_path}")
    print(f"Updated latest JSON: {latest_json}")
    print(f"Saved integrity text: {txt_path}")
    print(f"Updated latest text: {latest_txt}")

    return {
        "json_path": json_path,
        "txt_path": txt_path,
        "latest_json": latest_json,
        "latest_txt": latest_txt,
        "passed": report["passed"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run and save dataset integrity artifacts.")
    parser.add_argument(
        "--skip-near-duplicates",
        action="store_true",
        help="Skip near-duplicate (dHash) scanning.",
    )
    parser.add_argument(
        "--near-distance",
        type=int,
        default=5,
        help="Hamming distance threshold for dHash near-duplicate pairs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Custom output directory for integrity report artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose corrupt-image logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_save_report(
        run_near_duplicates=not args.skip_near_duplicates,
        near_duplicate_distance=args.near_distance,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )
