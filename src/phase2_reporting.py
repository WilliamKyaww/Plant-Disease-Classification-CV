"""
Phase 2 reporting and aggregation helpers.

Uses only the Python standard library so summary rebuilds still work even if a
Colab notebook kernel has a temporary pandas/numpy binary mismatch after pip
installs.
"""

import csv
import os
from collections import defaultdict
from statistics import mean, stdev


RUN_FIELDNAMES = [
    "model",
    "seed",
    "test_accuracy",
    "test_f1_macro",
    "elapsed_seconds",
    "trainable_params",
    "model_size_bytes",
    "metrics_path",
]

SUMMARY_FIELDNAMES = [
    "model",
    "accuracy_mean",
    "accuracy_std",
    "f1_macro_mean",
    "f1_macro_std",
    "elapsed_mean_sec",
    "elapsed_std_sec",
    "params",
    "model_size_bytes",
    "runs",
]


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values))


def _normalize_run_rows(run_rows: list[dict]) -> list[dict]:
    normalized_rows = []
    for row in run_rows:
        normalized_rows.append(
            {
                "model": str(row["model"]),
                "seed": int(row["seed"]),
                "test_accuracy": float(row["test_accuracy"]),
                "test_f1_macro": float(row["test_f1_macro"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "trainable_params": int(row["trainable_params"]),
                "model_size_bytes": int(row["model_size_bytes"]),
                "metrics_path": str(row.get("metrics_path", "")),
            }
        )
    return normalized_rows


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_phase2_summary(run_rows: list[dict], out_dir: str) -> dict:
    """
    Persist per-run metrics and aggregated summaries.

    Returns dictionary of output CSV paths.
    """
    if not run_rows:
        raise ValueError("run_rows is empty; nothing to summarize.")

    os.makedirs(out_dir, exist_ok=True)

    normalized_rows = _normalize_run_rows(run_rows)
    normalized_rows.sort(key=lambda row: (row["model"], row["seed"]))

    model_seed_csv = os.path.join(out_dir, "model_seed_metrics.csv")
    _write_csv(model_seed_csv, RUN_FIELDNAMES, normalized_rows)

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in normalized_rows:
        grouped_rows[row["model"]].append(row)

    summary_rows = []
    for model in sorted(grouped_rows):
        model_rows = grouped_rows[model]
        accuracies = [row["test_accuracy"] for row in model_rows]
        f1_scores = [row["test_f1_macro"] for row in model_rows]
        elapsed_values = [row["elapsed_seconds"] for row in model_rows]

        summary_rows.append(
            {
                "model": model,
                "accuracy_mean": float(mean(accuracies)),
                "accuracy_std": _safe_std(accuracies),
                "f1_macro_mean": float(mean(f1_scores)),
                "f1_macro_std": _safe_std(f1_scores),
                "elapsed_mean_sec": float(mean(elapsed_values)),
                "elapsed_std_sec": _safe_std(elapsed_values),
                "params": max(row["trainable_params"] for row in model_rows),
                "model_size_bytes": max(row["model_size_bytes"] for row in model_rows),
                "runs": len(model_rows),
            }
        )

    summary_csv = os.path.join(out_dir, "model_summary_mean_std.csv")
    _write_csv(summary_csv, SUMMARY_FIELDNAMES, summary_rows)

    leaderboard_rows = sorted(
        summary_rows,
        key=lambda row: (-row["f1_macro_mean"], -row["accuracy_mean"], row["model"]),
    )
    leaderboard_csv = os.path.join(out_dir, "leaderboard.csv")
    _write_csv(leaderboard_csv, SUMMARY_FIELDNAMES, leaderboard_rows)

    return {
        "model_seed_csv": model_seed_csv,
        "summary_csv": summary_csv,
        "leaderboard_csv": leaderboard_csv,
    }
