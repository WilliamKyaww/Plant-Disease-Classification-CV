"""
Colab path sanity + smoke checks with artifact output.

This script can run in Colab or locally (with --repo-main).
It validates repository root/imports/split CSVs and attempts one dataset sample load.

Run from Main/:
    py -3.13 -m src.colab_smoke --repo-main .
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    from src.experiment_log import get_git_commit_hash
except ImportError:
    from .experiment_log import get_git_commit_hash


def _default_candidates():
    return [
        "/content/Plant-Disease-Detection-CV/Main",
        "/content/Plant Disease Detection CV/Main",
        "/content/drive/MyDrive/Plant-Disease-Detection-CV/Main",
    ]


def detect_repo_main(repo_main_arg: str = None) -> str:
    if repo_main_arg:
        candidate = os.path.abspath(repo_main_arg)
        return candidate if os.path.isdir(os.path.join(candidate, "src")) else ""

    for candidate in _default_candidates():
        if os.path.isdir(os.path.join(candidate, "src")):
            return candidate
    return ""


def run_colab_smoke(repo_main_arg: str = None) -> dict:
    repo_main = detect_repo_main(repo_main_arg=repo_main_arg)
    executed_in_colab = os.path.isdir("/content")
    checks = []

    def add_check(name: str, passed: bool, details: str = ""):
        checks.append({"name": name, "passed": bool(passed), "details": details})

    add_check("repo_main_detected", bool(repo_main), f"repo_main={repo_main}")
    if not repo_main:
        return {
            "timestamp": datetime.now().isoformat(),
            "executed_in_colab": executed_in_colab,
            "passed": False,
            "checks": checks,
        }

    if repo_main not in sys.path:
        sys.path.insert(0, repo_main)

    os.chdir(repo_main)
    add_check("cwd_set", os.getcwd() == repo_main, f"cwd={os.getcwd()}")

    src_ok = os.path.isdir(os.path.join(repo_main, "src"))
    add_check("src_folder_exists", src_ok, "")

    csv_paths = [
        os.path.join(repo_main, "CSV", "plantvillage_train.csv"),
        os.path.join(repo_main, "CSV", "plantvillage_val.csv"),
        os.path.join(repo_main, "CSV", "plantvillage_test.csv"),
    ]
    csv_ok = all(os.path.exists(p) for p in csv_paths)
    add_check("split_csvs_exist", csv_ok, ", ".join(csv_paths))

    import_ok = False
    sample_ok = False
    sample_err = ""
    try:
        from src.datasets import PlantDiseaseDataset
        from src.transforms import get_val_transform

        import_ok = True
        ds = PlantDiseaseDataset(csv_paths[2], label_column="class_label", transform=get_val_transform())
        _img, _label = ds[0]
        sample_ok = True
    except Exception as exc:
        sample_err = str(exc)

    add_check("src_imports_ok", import_ok, "")
    add_check("dataset_sample_load_ok", sample_ok, sample_err)

    passed = all(item["passed"] for item in checks)
    report = {
        "timestamp": datetime.now().isoformat(),
        "executed_in_colab": executed_in_colab,
        "repo_main": repo_main,
        "git_commit": get_git_commit_hash(repo_dir=repo_main),
        "checks": checks,
        "passed": passed,
    }
    return report


def save_report(report: dict) -> dict:
    repo_main = report.get("repo_main") or os.getcwd()
    out_dir = os.path.join(repo_main, "results", "colab_smoke")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"colab_smoke_{timestamp}.json")
    txt_path = os.path.join(out_dir, f"colab_smoke_{timestamp}.txt")
    latest_json = os.path.join(out_dir, "latest_colab_smoke.json")
    latest_txt = os.path.join(out_dir, "latest_colab_smoke.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "Colab Smoke Report",
        "=" * 60,
        f"Timestamp: {report['timestamp']}",
        f"Executed in Colab: {report['executed_in_colab']}",
        f"Repo main: {report.get('repo_main', '')}",
        f"Git commit: {report.get('git_commit', '')}",
        f"Passed: {report['passed']}",
        "",
        "Checks",
        "-" * 60,
    ]
    for item in report["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- [{status}] {item['name']}: {item.get('details', '')}")
    text = "\n".join(lines) + "\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(latest_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved colab smoke JSON: {json_path}")
    print(f"Updated latest JSON: {latest_json}")
    print(f"Saved colab smoke text: {txt_path}")
    print(f"Updated latest text: {latest_txt}")

    return {
        "json_path": json_path,
        "txt_path": txt_path,
        "latest_json": latest_json,
        "latest_txt": latest_txt,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Colab path sanity smoke checks.")
    parser.add_argument(
        "--repo-main",
        type=str,
        default=None,
        help="Path to Main/ (optional for local/simulated execution).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_colab_smoke(repo_main_arg=args.repo_main)
    save_report(report)
    if not report["passed"]:
        raise SystemExit(1)
