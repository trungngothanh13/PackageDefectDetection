# PackageDefectDetection

## Download data.rar for training
+ Download data.rar [here](https://drive.google.com/file/d/172O0GeNdwMa_cpLKgQeMGHVzkqt7FEqm/view?usp=sharing)
+ Extract to root ```(PackageDefectDetection/...)```

```bash
python -m venv PDD_VENV
PDD_VENV/scripts/activate
pip install ultralytics opencv-python pillow

cd demo
python classification_demo.py
python detection_demo.py
```

## Troubleshoot + Retrain model
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

python train_classification.py
python train_detection.py
```

## Inference
See training models stats in:
```bash
runs/
├── package_defect_detection/
│   └── train_20251101_130621/
└── package_quality_classification/
    └── train_20251031_133632/
```

Quick summary:
+ Classification model accuracy: 83%
+ Detection model accuracy: 30%