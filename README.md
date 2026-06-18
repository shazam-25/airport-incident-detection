## REPOSITORY STRUCTURE
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
│   │   ├── extractor.py         # Renamed from data_extraction.py
│   │   ├── analyzer.py          # Renamed from data_analysis.py
│   │   ├── preprocessor.py      # Renamed from preprocess_pipeline.py
│   │   └── dataset.py           # Renamed from data_loader.py
│   ├── models/                  # Custom multi-task neural network topology
│   │   ├── __init__.py
│   │   └── multi_task_yolo.py   # Renamed from multi_head_yolo.py
│   ├── utils/                   # Shared pipeline infrastructure and edge components
│   │   ├── __init__.py
│   │   ├── common.py            # Renamed from utils.py
│   │   └── stream_buffer.py     # Pure terminal-based Threaded Producer-Consumer Queue
│   └── training/
│       ├── __init__.py
│       └── debug_overfit.py     # Renamed from overfitting_test.py
├── requirements.txt
└── README.md