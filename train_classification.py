"""
YOLOv8 Classification Training Script for Package Quality Classification
Trains a model to classify packages into: good, ok, bad
"""

from ultralytics import YOLO
import torch
import os
from datetime import datetime


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Dataset configuration
# For classification, just point to the root directory (not a YAML file!)
DATA_PATH = "data/classification"

# Model selection (choose one)
# 'yolov8n-cls.pt' - Nano (fastest, least accurate) ← START HERE
# 'yolov8s-cls.pt' - Small (good balance)
# 'yolov8m-cls.pt' - Medium (more accurate, slower)
# 'yolov8l-cls.pt' - Large (very accurate, very slow)
MODEL_SIZE = "yolov8l-cls.pt"

# Training hyperparameters
EPOCHS = 100              # Number of training epochs
BATCH_SIZE = 16           # Batch size (adjust based on GPU memory: 8/16/32/64)
IMG_SIZE = 320            # Image size for classification (224 is standard)
DEVICE = 0                # 0 for GPU, 'cpu' for CPU

# Training settings
PATIENCE = 100             # Early stopping patience (stops if no improvement for N epochs)
SAVE_PERIOD = 10          # Save checkpoint every N epochs

# Project organization
PROJECT_NAME = "package_quality_classification"
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# ENVIRONMENT CHECK
# ============================================================================

def check_environment():
    """Check training environment and dependencies"""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Check CUDA/GPU
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("No GPU detected. Training will use CPU (much slower)")
        print("   Recommend using GPU for faster training")
    
    # Check dataset.yaml
    if os.path.exists(DATA_PATH):
        print(f"✓ Dataset directory found: {DATA_PATH}")
    else:
        print(f"Dataset directory not found: {DATA_PATH}")
        print("   Please ensure data/classification/ exists!")
        return False
    
    # Check dataset directories
    required_dirs = [
        "data/classification/train",
        "data/classification/val",
        "data/classification/test"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            # Count images in subdirectories (good, ok, bad)
            print(f"{dir_path}")
            for class_dir in os.listdir(dir_path):
                class_path = os.path.join(dir_path, class_dir)
                if os.path.isdir(class_path):
                    num_images = len([f for f in os.listdir(class_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  └─ {class_dir}: {num_images} images")
        else:
            print(f"❌ {dir_path}: NOT FOUND")
            return False
    
    print("=" * 70)
    return True


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """Train YOLOv8 classification model"""
    
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: {MODEL_SIZE}")
    print(f"Task: Image Classification (good/ok/bad)")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Device: {'GPU' if DEVICE == 0 and torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Output: runs/{PROJECT_NAME}/{RUN_NAME}/")
    print("=" * 70)
    
    # Load pretrained model
    print("\nLoading pretrained classification model...")
    model = YOLO(MODEL_SIZE)
    print(f"✓ Loaded {MODEL_SIZE}")
    
    # Start training
    print("\nStarting training...\n")
    
    results = model.train(
        data=DATA_PATH,
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
        optimizer='AdamW',      # Changed from 'auto' - AdamW often better
        lr0=0.0005,            # Lower initial learning rate (was 0.001)
        lrf=0.001,             # Lower final learning rate (was 0.01)
        momentum=0.937,
        weight_decay=0.001,    # Increased regularization (was 0.0005)
        
        # Data augmentation - STRONGER augmentation
        augment=True,
        hsv_h=0.03,         # Increased hue (was 0.015)
        hsv_s=0.8,          # Increased saturation (was 0.7)
        hsv_v=0.5,          # Increased value (was 0.4)
        degrees=15,         # More rotation (was 10)
        translate=0.2,      # More translation (was 0.1)
        scale=0.7,          # More scaling (was 0.5)
        flipud=0.0,         # Flip up-down (probability)
        fliplr=0.5,         # Flip left-right (probability)
        
        # Validation
        val=True,
        plots=True,         # Save training plots
        
        # Other
        verbose=True,
        seed=42,            # For reproducibility
        workers=4           # Number of data loading workers
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("YOLOV8 PACKAGE QUALITY CLASSIFICATION - TRAINING")
    print("=" * 70)
    
    # Step 1: Check environment
    print("\n[1/3] Checking environment...")
    if not check_environment():
        print("\nEnvironment check failed. Please fix the issues above.")
        return
    
    # Step 2: Train model
    print("\n[2/3] Training model...")
    try:
        results = train_model()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n" + "=" * 70)
    print("📁 OUTPUT LOCATION")
    print("=" * 70)
    print(f"Training results: runs/{PROJECT_NAME}/{RUN_NAME}/")
    print(f"Best model: runs/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"Last model: runs/{PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
    print(f"Training plots: runs/{PROJECT_NAME}/{RUN_NAME}/")


if __name__ == "__main__":
    main()