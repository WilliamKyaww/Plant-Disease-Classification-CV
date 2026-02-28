# Plant Disease Detection

Computer vision project for plant disease classification with reproducible training/evaluation workflows.

## Scope (Current)
1. Primary task: 15-class crop+disease classification (PlantVillage folder-level classes).
2. Side analysis: 4-disease subset (healthy, bacterial_spot, early_blight, late_blight).
3. Severity (0-3): parked and optional, only after Phase 1-3 gates are met.

## Repository Layout
```text
Main/
  src/                         Core source modules (single source of truth)
  notebooks/                   Notebook orchestration layer
    01_dataset_preparation.ipynb
    02_data_pipeline.ipynb
    03_finetuning_resnet.ipynb
    04_severity_labelling.ipynb  (deprecated)
    05_severity_review.ipynb     (deprecated)
    06_severity_small_experiment.ipynb (deprecated)
  CSV/                         Split CSV artifacts
  Datasets/                    Image folders (not tracked in git)
  models/                      Saved model checkpoints
  results/                     Logs, figures, split manifests
  requirements.txt
```

## Environment
Recommended local interpreter: Python 3.13 on this machine.

Install:
```bash
cd Main
py -3.13 -m pip install -r requirements.txt
```

## Data Setup
Place PlantVillage folders under:
`Main/Datasets/<folder_name>/image.jpg`

Expected class folder names are defined in `src/utils.py` under `FOLDER_METADATA`.

## Phase 1 Workflow (Executable)
Run from `Main/`:

Quick one-command runner (forces Python 3.13):
```bash
run_phase1.bat
```

Equivalent step-by-step commands:
1. Integrity audit (missing/corrupt/exact duplicates/near duplicates) with report artifacts:
```bash
py -3.13 -m src.integrity_report
```

2. Build frozen multi-class labels + stratified splits + split manifest:
```bash
py -3.13 -m src.prepare_splits
```

3. Run script-based baseline smoke training (1 epoch, 15-class):
```bash
py -3.13 -m src.run_baseline_smoke --epochs 1 --batch-size 32 --max-train 128 --max-val 64 --max-test 64
```

4. Run Colab path sanity smoke report:
```bash
py -3.13 -m src.colab_smoke --repo-main .
```

5. Run repeat-run stability check (3 seeds):
```bash
py -3.13 -m src.stability_check --seeds 41,42,43 --epochs 1 --batch-size 32 --max-train 128 --max-val 64 --max-test 64
```

Generated artifacts:
1. `CSV/plantvillage_multiclass_labels.csv`
2. `CSV/plantvillage_train.csv`
3. `CSV/plantvillage_val.csv`
4. `CSV/plantvillage_test.csv`
5. `results/split_manifests/latest_split_manifest.json`
6. `results/integrity_reports/latest_integrity_report.json`
7. `results/integrity_reports/latest_integrity_report.txt`
8. `results/baseline_smoke/latest_metrics_snapshot.json`
9. `results/baseline_smoke/latest_experiment_log.json`
10. `results/baseline_smoke/latest_confusion_matrix.json`
11. `results/baseline_smoke/latest_confusion_matrix.png`
12. `results/colab_smoke/latest_colab_smoke.json`
13. `results/colab_smoke/latest_colab_smoke.txt`
14. `results/stability_checks/latest_stability_check.json`
15. `results/stability_checks/latest_stability_check.txt`

## Notebook Status
1. `notebooks/01_dataset_preparation.ipynb` calls `src.integrity` and `src.prepare_splits`.
2. `notebooks/02_data_pipeline.ipynb` validates dataloaders using `class_label`.
3. `notebooks/03_finetuning_resnet.ipynb` runs a baseline training flow using `src` modules and `ExperimentLog`.
4. `04-06` are intentionally deprecated until the optional severity phase is unlocked.

## Colab Note
`Main/Google Colab/Plant_Disease.ipynb` is context/support only.
It now auto-detects a `Main/` repository root and imports from `src` from that location.

## Reproducibility Logging
Use `src/experiment_log.py` to record:
1. Hyperparameters and seed
2. Metrics
3. Environment metadata
4. Git commit hash
5. Split artifact metadata (CSV hashes + class counts + seed)
