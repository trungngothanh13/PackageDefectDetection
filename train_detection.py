"""
YOLOv8 Detection Training Script for Package Defect Detection
Trains a model to detect and localize defects: torn, crushed, dented
"""

from ultralytics import YOLO
import torch
import os
from datetime import datetime


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Dataset configuration
DATA_YAML = "data/detection/dataset.yaml"

# Model selection (choose one)
# 'yolov8n.pt' - Nano (fastest, least accurate) ← START HERE
# 'yolov8s.pt' - Small (good balance)
# 'yolov8m.pt' - Medium (more accurate, slower)
# 'yolov8l.pt' - Large (very accurate, very slow)
MODEL_SIZE = "yolov8n.pt"

# Training hyperparameters
EPOCHS = 100              # Number of training epochs
BATCH_SIZE = 16           # Adjust based on GPU memory (8/16/32)
IMG_SIZE = 640            # Image size (640 is standard for detection)
DEVICE = 0                # 0 for GPU, 'cpu' for CPU

# Training settings
PATIENCE = 100             # Early stopping patience
SAVE_PERIOD = 10          # Save checkpoint every N epochs

# Project organization
PROJECT_NAME = "package_defect_detection"
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
        print("   Recommend using GPU for detection training")
    
    # Check dataset.yaml
    if os.path.exists(DATA_YAML):
        print(f"✓ Dataset config found: {DATA_YAML}")
    else:
        print(f"Dataset config not found: {DATA_YAML}")
        print("   Please ensure dataset.yaml exists!")
        return False
    
    # Check dataset directories
    required_dirs = [
        "data/detection/train/images",
        "data/detection/train/labels",
        "data/detection/val/images",
        "data/detection/val/labels"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            num_files = len([f for f in os.listdir(dir_path) 
                           if not f.startswith('.')])
            print(f"{dir_path}: {num_files} files")
        else:
            print(f"{dir_path}: NOT FOUND")
            return False
    
    print("=" * 70)
    return True


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """Train YOLOv8 detection model"""
    
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: {MODEL_SIZE}")
    print(f"Task: Object Detection (torn, crushed, dented)")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Device: {'GPU' if DEVICE == 0 and torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Output: runs/{PROJECT_NAME}/{RUN_NAME}/")
    print("=" * 70)
    
    # Load pretrained model
    print("\nLoading pretrained detection model...")
    model = YOLO(MODEL_SIZE)
    print(f"✓ Loaded {MODEL_SIZE}")
    
    # Start training
    print("\nStarting training...\n")
    
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
        optimizer='AdamW',
        lr0=0.01,           # Initial learning rate
        lrf=0.01,           # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        
        # Data augmentation
        augment=True,
        hsv_h=0.015,        # HSV hue augmentation
        hsv_s=0.7,          # HSV saturation augmentation
        hsv_v=0.4,          # HSV value augmentation
        degrees=0.0,        # Rotation (+/- deg)
        translate=0.1,      # Translation (+/- fraction)
        scale=0.5,          # Scale (+/- gain)
        shear=0.0,          # Shear (+/- deg)
        perspective=0.0,    # Perspective (+/- fraction)
        flipud=0.0,         # Flip up-down (probability)
        fliplr=0.5,         # Flip left-right (probability)
        mosaic=1.0,         # Mosaic augmentation (probability)
        mixup=0.0,          # MixUp augmentation (probability)
        copy_paste=0.0,     # Copy-paste augmentation (probability)
        
        # Loss weights (can adjust for class imbalance)
        box=7.5,            # Box loss gain
        cls=0.5,            # Class loss gain
        dfl=1.5,            # DFL loss gain
        
        # Validation
        val=True,
        plots=True,         # Save training plots
        
        # Other
        verbose=True,
        seed=42,            # For reproducibility
        workers=4           # Number of data loading workers
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    return results


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model():
    """Evaluate trained model on validation and test sets"""
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Find best model weights
    best_model_path = f"runs/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
    
    if not os.path.exists(best_model_path):
        print(f"Model not found: {best_model_path}")
        return
    
    print(f"Evaluating model: {best_model_path}")
    
    # Load best model
    model = YOLO(best_model_path)
    
    # Validate on validation set
    print("\nValidation Set Performance:")
    val_metrics = model.val(data=DATA_YAML, split='val')
    
    # Print detection metrics
    print("\n" + "=" * 70)
    print("DETECTION METRICS (VALIDATION)")
    print("=" * 70)
    
    if hasattr(val_metrics, 'box'):
        print(f"\nOverall Performance:")
        print(f"  mAP50:     {val_metrics.box.map50:.4f} ({val_metrics.box.map50*100:.2f}%)")
        print(f"  mAP50-95:  {val_metrics.box.map:.4f} ({val_metrics.box.map*100:.2f}%)")
        print(f"  Precision: {val_metrics.box.mp:.4f} ({val_metrics.box.mp*100:.2f}%)")
        print(f"  Recall:    {val_metrics.box.mr:.4f} ({val_metrics.box.mr*100:.2f}%)")
        
        print(f"\nPer-Class mAP50:")
        class_names = ['torn', 'crushed', 'dented']
        if hasattr(val_metrics.box, 'maps'):
            for i, map_value in enumerate(val_metrics.box.maps):
                if i < len(class_names):
                    print(f"  {class_names[i]:8s}: {map_value:.4f} ({map_value*100:.2f}%)")
    
    print("=" * 70)
    
    # Test set evaluation (if exists)
    if os.path.exists("data/detection/test"):
        print("\nTest Set Performance:")
        test_metrics = model.val(data=DATA_YAML, split='test')
        
        print("\n" + "=" * 70)
        print("DETECTION METRICS (TEST)")
        print("=" * 70)
        
        if hasattr(test_metrics, 'box'):
            print(f"\n📊 Overall Performance:")
            print(f"  mAP50:     {test_metrics.box.map50:.4f} ({test_metrics.box.map50*100:.2f}%)")
            print(f"  mAP50-95:  {test_metrics.box.map:.4f} ({test_metrics.box.map*100:.2f}%)")
            print(f"  Precision: {test_metrics.box.mp:.4f} ({test_metrics.box.mp*100:.2f}%)")
            print(f"  Recall:    {test_metrics.box.mr:.4f} ({test_metrics.box.mr*100:.2f}%)")
            
            print(f"\n📊 Per-Class mAP50:")
            if hasattr(test_metrics.box, 'maps'):
                for i, map_value in enumerate(test_metrics.box.maps):
                    if i < len(class_names):
                        print(f"  {class_names[i]:8s}: {map_value:.4f} ({map_value*100:.2f}%)")
        
        print("=" * 70)
    
    # Summary with visual file locations
    print("\n" + "=" * 70)
    print("📊 VISUAL RESULTS")
    print("=" * 70)
    print(f"Confusion Matrix: runs/{PROJECT_NAME}/{RUN_NAME}/confusion_matrix.png")
    print(f"Training Curves:  runs/{PROJECT_NAME}/{RUN_NAME}/results.png")
    print(f"PR Curve:         runs/{PROJECT_NAME}/{RUN_NAME}/PR_curve.png")
    print(f"F1 Curve:         runs/{PROJECT_NAME}/{RUN_NAME}/F1_curve.png")
    print(f"Sample Predictions: runs/{PROJECT_NAME}/{RUN_NAME}/val_batch*_pred.jpg")
    print("=" * 70)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("YOLOV8 PACKAGE DEFECT DETECTION - TRAINING")
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
    
    # Step 3: Evaluate model
    print("\n[3/3] Evaluating model...")
    try:
        evaluate_model()
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 70)
    print("OUTPUT LOCATION")
    print("=" * 70)
    print(f"Training results: runs/{PROJECT_NAME}/{RUN_NAME}/")
    print(f"Best model: runs/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"Last model: runs/{PROJECT_NAME}/{RUN_NAME}/weights/last.pt")

if __name__ == "__main__":
    main()