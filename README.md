# PackageDefectDetection

## How to run
+ Download data.rar [here](https://drive.google.com/file/d/172O0GeNdwMa_cpLKgQeMGHVzkqt7FEqm/view?usp=sharing)
+ Extract to root ```(PackageDefectDetection/...)```

```bash
python -m venv PDD_VENV
PDD_VENV/scripts/activate
pip install ultralytics opencv-python pillow

cd demo
python simple_demo.py
```

## Troubleshoot + Retrain model
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
python check_data.py
python train_classification.py
```

See stats like Confusion Matrix... in ```runs``` directory