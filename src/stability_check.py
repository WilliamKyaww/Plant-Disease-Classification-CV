"""
Repeat-run stability check for baseline smoke training.

Runs baseline smoke training for multiple seeds and reports mean/std metrics.

Run from repo root:
    py -3.13 -m src.stability_check --seeds 41,42,43
"""

import argparse
import json
import os
from datetime import datetime
from statistics import mean, stdev

try:
    from src.experiment_log import get_git_commit_hash
    from src.run_baseline_smoke import run_baseline_smoke
    from src.utils import PROJECT_ROOT, RESULTS_DIR
except ImportError:
    from .experiment_log import get_git_commit_hash
    from .run_baseline_smoke import run_baseline_smoke
    from .utils import PROJECT_ROOT, RESULTS_DIR


def _safe_std(values):
    return float(stdev(values)) if len(values) > 1 else 0.0


def run_stability_check(
    seeds,
    epochs=1,
    batch_size=32,
    lr=1e-3,
    max_train=128,
    max_val=64,
    max_test=64,
):
    runs = []
    for seed in seeds:
        result = run_baseline_smoke(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            max_train=max_train,
            max_val=max_val,
            max_test=max_test,
        )
        runs.append(result)

    accs = [r["test_accuracy"] for r in runs]
    f1s = [r["test_f1_macro"] for r in runs]
    payload = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(repo_dir=PROJECT_ROOT),
        "config": {
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_train": max_train,
            "max_val": max_val,
            "max_test": max_test,
        },
        "per_run": runs,
        "summary": {
            "n_runs": len(runs),
            "test_accuracy_mean": float(mean(accs)),
            "test_accuracy_std": _safe_std(accs),
            "test_f1_macro_mean": float(mean(f1s)),
            "test_f1_macro_std": _safe_std(f1s),
        },
    }
    return payload


def save_stability_report(payload):
    out_dir = os.path.join(RESULTS_DIR, "stability_checks")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"stability_check_{ts}.json")
    txt_path = os.path.join(out_dir, f"stability_check_{ts}.txt")
    latest_json = os.path.join(out_dir, "latest_stability_check.json")
    latest_txt = os.path.join(out_dir, "latest_stability_check.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    s = payload["summary"]
    lines = [
        "Baseline Stability Check",
        "=" * 60,
        f"Timestamp: {payload['timestamp']}",
        f"Git commit: {payload['git_commit']}",
        f"Seeds: {payload['config']['seeds']}",
        "",
        "Summary",
        "-" * 60,
        f"Runs: {s['n_runs']}",
        f"Test accuracy mean +/- std: {s['test_accuracy_mean']:.4f} +/- {s['test_accuracy_std']:.4f}",
        f"Test F1 macro mean +/- std: {s['test_f1_macro_mean']:.4f} +/- {s['test_f1_macro_std']:.4f}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(latest_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved stability JSON: {json_path}")
    print(f"Updated latest JSON: {latest_json}")
    print(f"Saved stability text: {txt_path}")
    print(f"Updated latest text: {latest_txt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline repeat-run stability check.")
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=64)
    parser.add_argument("--max-test", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    payload = run_stability_check(
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
    )
    save_stability_report(payload)
