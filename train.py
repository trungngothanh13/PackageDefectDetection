"""
YOLOv8 Training Script for Package Defect Detection
"""

from ultralytics import YOLO
import torch
import os
from datetime import datetime


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Dataset configuration
DATA_YAML = "data/dataset.yaml"

# Model selection (choose one)
# 'yolov8n.pt' - Nano (fastest, least accurate)
# 'yolov8s.pt' - Small (good balance) ← RECOMMENDED
# 'yolov8m.pt' - Medium (more accurate, slower)
# 'yolov8l.pt' - Large (very accurate, very slow)
MODEL_SIZE = "yolov8s.pt"

# Training hyperparameters
EPOCHS = 100              # Number of training epochs
BATCH_SIZE = 16           # Adjust based on your GPU memory (8/16/32)
IMG_SIZE = 640            # Image size (640 is standard)
DEVICE = 0                # 0 for GPU, 'cpu' for CPU

# Training settings
PATIENCE = 20             # Early stopping patience
SAVE_PERIOD = 10          # Save checkpoint every N epochs

# Augmentation (YOLOv8 uses these by default, can customize)
AUGMENT = True

# Project organization
PROJECT_NAME = "package_defect_detection"
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def check_environment():
    """Check training environment"""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Check CUDA/GPU
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU detected. Training will use CPU (much slower)")
        print("   Consider using Google Colab or a machine with GPU")
    
    # Check dataset.yaml
    if os.path.exists(DATA_YAML):
        print(f"✓ Dataset config found: {DATA_YAML}")
    else:
        print(f"❌ Dataset config not found: {DATA_YAML}")
        print("   Please create dataset.yaml first!")
        return False
    
    # Check dataset directories
    required_dirs = [
        "data/dataset/train/images",
        "data/dataset/train/labels",
        "data/dataset/val/images",
        "data/dataset/val/labels"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            num_files = len(os.listdir(dir_path))
            print(f"✓ {dir_path}: {num_files} files")
        else:
            print(f"❌ {dir_path}: NOT FOUND")
            return False
    
    print("=" * 70)
    return True


def train_model():
    """Train YOLOv8 model"""
    
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: {MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Device: {'GPU' if DEVICE == 0 and torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Output: runs/{PROJECT_NAME}/{RUN_NAME}")
    print("=" * 70)
    
    # Load pretrained model
    print("\n📦 Loading pretrained model...")
    model = YOLO(MODEL_SIZE)
    print(f"✓ Loaded {MODEL_SIZE}")
    
    # Start training
    print("\n🚀 Starting training...\n")
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        
        # Project organization
        project=f"runs/{PROJECT_NAME}",
        name=RUN_NAME,
        
        # Training settings
        patience=PATIENCE,
        save=True,
        save_period=SAVE_PERIOD,
        
        # Optimization
        optimizer='auto',
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation
        augment=AUGMENT,
        hsv_h=0.015,       # HSV hue augmentation
        hsv_s=0.7,         # HSV saturation augmentation
        hsv_v=0.4,         # HSV value augmentation
        degrees=0.0,       # Rotation (+/- deg)
        translate=0.1,     # Translation (+/- fraction)
        scale=0.5,         # Scale (+/- gain)
        flipud=0.0,        # Flip up-down (probability)
        fliplr=0.5,        # Flip left-right (probability)
        mosaic=1.0,        # Mosaic augmentation (probability)
        
        # Validation
        val=True,
        plots=True,        # Save training plots
        
        # Other
        verbose=True,
        seed=42            # For reproducibility
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    return results


def evaluate_model():
    """Evaluate trained model on test set"""
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Find best model weights
    best_model_path = f"runs/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
    
    if not os.path.exists(best_model_path):
        print(f"❌ Model not found: {best_model_path}")
        return
    
    print(f"📊 Evaluating model: {best_model_path}")
    
    # Load best model
    model = YOLO(best_model_path)
    
    # Validate on test set
    metrics = model.val(
        data=DATA_YAML,
        split='test',
        save_json=True,
        save_hybrid=True
    )
    
    # Print metrics
    print("\n📈 Test Set Performance:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    print("\n📊 Per-Class mAP50:")
    for i, map_value in enumerate(metrics.box.maps):
        class_name = ['optimal', 'good', 'deformed', 'dented', 'poked', 'cut'][i]
        print(f"   {class_name}: {map_value:.4f}")
    
    print("=" * 70)


def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("YOLO PACKAGE DEFECT DETECTION - TRAINING")
    print("=" * 70)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Step 2: Train model
    try:
        results = train_model()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return
    
    # Step 3: Evaluate model
    try:
        evaluate_model()
    except Exception as e:
        print(f"\n⚠️  Evaluation failed with error: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📁 OUTPUT LOCATION")
    print("=" * 70)
    print(f"Training results: runs/{PROJECT_NAME}/{RUN_NAME}/")
    print(f"Best model: runs/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"Last model: runs/{PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
    print("\n💡 Next steps:")
    print("   1. Review training plots in the results folder")
    print("   2. Check confusion matrix and prediction examples")
    print("   3. Integrate best.pt model into your demo app")
    print("=" * 70)


if __name__ == "__main__":
    main()