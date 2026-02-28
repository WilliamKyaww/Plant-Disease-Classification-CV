# Plant Disease Detection (Final Year Project)

Computer vision project for plant disease classification with reproducible training/evaluation workflows.

## Scope (Current)
1. Primary task: 15-class crop+disease classification (PlantVillage folder-level classes).
2. Side analysis: 4-disease subset (healthy, bacterial_spot, early_blight, late_blight).
3. Severity (0-3): parked and optional, only after Phase 1-3 gates are met.

## Repository Layout
```text
Main/
  src/                         Core source modules (single source of truth)
  CSV/                         Split CSV artifacts
  Datasets/                    Image folders (not tracked in git)
  models/                      Saved model checkpoints
  results/                     Logs, figures, split manifests
  01_dataset_preparation.ipynb
  02_data_pipeline.ipynb
  03_finetuning_resnet.ipynb
  04_severity_labelling.ipynb  (deprecated)
  05_severity_review.ipynb     (deprecated)
  06_severity_small_experiment.ipynb (deprecated)
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

1. Integrity audit (missing/corrupt/exact duplicates/near duplicates):
```bash
py -3.13 -m src.integrity
```

2. Build frozen multi-class labels + stratified splits + split manifest:
```bash
py -3.13 -m src.prepare_splits
```

Generated artifacts:
1. `CSV/plantvillage_multiclass_labels.csv`
2. `CSV/plantvillage_train.csv`
3. `CSV/plantvillage_val.csv`
4. `CSV/plantvillage_test.csv`
5. `results/split_manifests/latest_split_manifest.json`

## Notebook Status
1. `01_dataset_preparation.ipynb` calls `src.integrity` and `src.prepare_splits`.
2. `02_data_pipeline.ipynb` validates dataloaders using `class_label`.
3. `03_finetuning_resnet.ipynb` runs a baseline training flow using `src` modules and `ExperimentLog`.
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
