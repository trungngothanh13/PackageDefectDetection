"""
Inference module for YOLOv8 model predictions
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional

from config import (
    MODEL_PATH, MODEL_AVAILABLE, CLASS_NAMES,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, IMG_SIZE
)

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Using pre-labeled data only.")


class ModelInference:
    """Handle model inference for defect detection"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        if YOLO_AVAILABLE and MODEL_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            print(f"Loading model from: {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
            self.model_loaded = True
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model_loaded = False
    
    def predict(self, image_path: str) -> List[Dict]:
        """
        Run inference on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detection dictionaries with keys:
            - class_id: int (0-5)
            - class_name: str
            - confidence: float (0-1)
            - x_center: float (0-1)
            - y_center: float (0-1)
            - width: float (0-1)
            - height: float (0-1)
        """
        if not self.model_loaded:
            return []
        
        try:
            # Run inference
            results = self.model.predict(
                source=image_path,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )
            
            # Parse results
            detections = []
            
            if len(results) > 0:
                result = results[0]  # First image
                
                # Get image dimensions
                img_height, img_width = result.orig_shape
                
                # Extract boxes
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get box coordinates (xyxy format)
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Convert to YOLO format (normalized center, width, height)
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # Get class and confidence
                        class_id = int(boxes.cls[i].cpu().numpy())
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        detections.append({
                            'class_id': class_id,
                            'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                            'confidence': confidence,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
            
            return detections
            
        except Exception as e:
            print(f"Inference error: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if model inference is available"""
        return self.model_loaded


# Global inference instance
inference_engine = ModelInference()