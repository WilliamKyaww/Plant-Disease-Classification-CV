# 🌿 Plant Disease Detection using Computer Vision

A deep learning system for classifying plant diseases from leaf images and estimating disease severity, using transfer learning with modern vision architectures (CNNs and Vision Transformers).

## Project Overview

This project investigates:
1. **Multi-class disease classification** — identifying specific diseases across multiple crop types
2. **Architecture comparison** — benchmarking ResNet, EfficientNet, and Vision Transformers
3. **Explainability** — using Grad-CAM to visualize what the model focuses on
4. **Severity estimation** — predicting disease severity using multi-task learning
5. **Uncertainty quantification** — measuring model confidence via Monte Carlo Dropout

### Dataset
- **PlantVillage** dataset containing leaf images across multiple crop/disease combinations
- Crops: Bell Pepper, Tomato, Potato
- Conditions: Healthy, Bacterial Spot, Early Blight, Late Blight

## Project Structure

```
Main/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                          # Shared source code
│   ├── __init__.py
│   ├── datasets.py               # PyTorch Dataset classes
│   ├── transforms.py             # Data augmentation pipelines
│   ├── training.py               # Training & evaluation loops
│   ├── visualization.py          # Plotting & Grad-CAM utilities
│   └── utils.py                  # Config, seeds, path helpers
├── notebooks/                    # Jupyter notebooks (main pipeline)
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_pipeline.ipynb
│   ├── 04_training_multiclass.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_gradcam.ipynb
│   ├── 07_architecture_comparison.ipynb
│   ├── 08_severity_labelling.ipynb
│   └── 09_severity_model.ipynb
├── models/                       # Saved model weights
├── CSV/                          # Generated data splits
├── Datasets/                     # Image data (gitignored)
├── experiments/                  # Exploratory / pilot experiments
└── results/                      # Final figures for dissertation
```

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended (but CPU works for small experiments)

### Installation
```bash
pip install -r requirements.txt
```

### Data
1. Download the PlantVillage dataset
2. Extract into `Datasets/` with one subfolder per crop-disease combination
3. Run `notebooks/01_dataset_preparation.ipynb` to generate CSV splits

## Running the Pipeline
Execute notebooks in numerical order (01 → 09). Each notebook imports shared code from `src/`.

## Key Results

| Model | Accuracy | F1 (macro) | Parameters |
|-------|----------|------------|------------|
| ResNet18 | TBD | TBD | 11.7M |
| ResNet50 | TBD | TBD | 25.6M |
| EfficientNet-B0 | TBD | TBD | 5.3M |
| ViT-Small | TBD | TBD | 22M |

## Author
Final Year Project — BSc Computer Science & AI, Loughborough University
