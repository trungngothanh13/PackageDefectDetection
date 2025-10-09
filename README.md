# PackageDefectDetection

## How to run
Download data.rar [here](https://drive.google.com/file/d/1nxY_r0Z-m2DEnGy0841Una24Mg-6XkeN/view?usp=sharing)
Extract to root ```(PackageDefectDetection/data/...)```

```bash
python -m venv PDD_VENV
PDD_VENV/scripts/activate
pip install ultralytics opencv-python pillow

cd demo
python app.py
```

## Troubleshoot + Retrain model
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
python train.py
```

See stats like Recall-Precision Curve, Confusion Matrix... in ```runs``` directory