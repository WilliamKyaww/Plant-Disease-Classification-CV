"""
Phase 2 reporting and aggregation helpers.
"""

import os

import pandas as pd


def write_phase2_summary(run_rows: list[dict], out_dir: str) -> dict:
    """
    Persist per-run metrics and aggregated summaries.

    Returns dictionary of output CSV paths.
    """
    if not run_rows:
        raise ValueError("run_rows is empty; nothing to summarize.")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(run_rows)
    model_seed_csv = os.path.join(out_dir, "model_seed_metrics.csv")
    df.to_csv(model_seed_csv, index=False)

    summary = (
        df.groupby("model")
        .agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            f1_macro_mean=("test_f1_macro", "mean"),
            f1_macro_std=("test_f1_macro", "std"),
            elapsed_mean_sec=("elapsed_seconds", "mean"),
            elapsed_std_sec=("elapsed_seconds", "std"),
            params=("trainable_params", "max"),
            model_size_bytes=("model_size_bytes", "max"),
            runs=("seed", "count"),
        )
        .reset_index()
    )

    # For partial/early runs, std can be NaN when runs=1.
    for col in ("accuracy_std", "f1_macro_std", "elapsed_std_sec"):
        if col in summary.columns:
            summary[col] = summary[col].fillna(0.0)

    summary_csv = os.path.join(out_dir, "model_summary_mean_std.csv")
    summary.to_csv(summary_csv, index=False)

    leaderboard = summary.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    leaderboard_csv = os.path.join(out_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_csv, index=False)

    return {
        "model_seed_csv": model_seed_csv,
        "summary_csv": summary_csv,
        "leaderboard_csv": leaderboard_csv,
    }
