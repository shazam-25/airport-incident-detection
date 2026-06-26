# Multi-Stream Deep Learning Framework for Real-Time Airport Incident Detection

An M.Tech Final Year Implementation Project demonstrating real-time incident detection and classification across three concurrent airside video streams using a unified, lightweight deep learning architecture optimized for edge computing constraints.

## 🚀 System Architecture Performance Map
- **Shared Backbone:** Modified CSPDarknet Feature Extractor (~11.5M Parameters)
- **Task Prediction Branches:**
    1. **Stream 1: Airport Turnaround Monitoring** (13 Spatial Classes)
    2. **Stream 2: Industrial PPE Compliance Verification** (4 Safety Classes)
    3. **Stream 3: Runway Forign Object Debris (FOD) Detection** (31 Micro-Object Classes)

## 📁 Repository Directory Structure
Refer to the layout map below for module organization:
```text
airport-incident-detection/
├── configs/
│   ├── turnaround.yaml          # Generated dynamically via data pipeline
│   ├── ppe.yaml
│   └── fod.yaml
├── data/                        # Excluded from git via .gitignore
│   ├── raw/                     # Immutable target download source
│   └── processed/               # Standardized, stratified 640x640 splits
├── models/
│   └── yolov8s.pt               # Pretrained weights baseline checkpoint
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Clean, sequential step-by-step pipeline runner
│   └── 02_model_sandbox.ipynb       # Model architecture test & prototyping playground
├── src/
│   ├── __init__.py
│   ├── data/                    # Data processing and loading core modules
│   │   ├── __init__.py
│   │   ├── extractor.py         # Extract dataset from Kaggle
│   │   ├── analyzer.py          # Custom dataset analyzer
│   │   ├── preprocessor.py      # Data preprocessing pipeline
│   │   └── dataset.py           # Multi-task dataset ingestion pipelines with custom batch collation routines
│   ├── models/                  # Custom multi-task neural network topology
│   │   ├── __init__.py
│   │   └── model.py             # Multi-task structural neural network blueprint
│   │   └── lossed.py            # Decoupled joint loss equations configured with task gradient balancing weights ($\alpha, \beta, \gamma$)
│   ├── utils/                   # Shared pipeline infrastructure and edge components
│   │   ├── __init__.py
│   │   ├── common.py            # Renamed from utils.py
│   │   └── stream_handler.py    # Asynchronous, multi-thread double-buffered frame-buffer handling engine.
│   └── training/
│       ├── __init__.py
│       └── debug_overfit.py     # Renamed from overfitting_test.py
├── requirements.txt             # Explicit dependency list
└── README.md                    # Technical project blueprint documentation
```

## ⚙️ Core Operational Dependencies
Install the required system dependency matrices prior to executing training or inference loops:
```bash
pip install -r requirements.txt
```
Launch TensorBoard to monitor the validation curves
```bash
tensorboard --logdir=./models/logs