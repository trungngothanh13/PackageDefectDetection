"""
Utility functions for Package Defect Detection Demo
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from config import CLASS_NAMES, CLASS_COLORS, BOX_THICKNESS, LABEL_FONT_SCALE, LABEL_THICKNESS


def parse_yolo_label(label_path: str) -> List[Dict]:
    """
    Parse YOLO format label file.
    
    Args:
        label_path: Path to .txt label file
        
    Returns:
        List of detection dictionaries with keys:
        - class_id: int (0-5)
        - class_name: str
        - x_center: float (0-1)
        - y_center: float (0-1)
        - width: float (0-1)
        - height: float (0-1)
    """
    detections = []
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return detections
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Invalid label format: {line}")
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            detections.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    return detections


def yolo_to_pixel_coords(detection: Dict, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized coordinates to pixel coordinates.
    
    Args:
        detection: Detection dict from parse_yolo_label
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    x_center = detection['x_center'] * img_width
    y_center = detection['y_center'] * img_height
    width = detection['width'] * img_width
    height = detection['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2


def draw_bounding_boxes(image: np.ndarray, detections: List[Dict], visible_classes: Dict[int, bool]) -> np.ndarray:
    """
    Draw bounding boxes on image for visible classes.
    
    Args:
        image: OpenCV image (numpy array)
        detections: List of detection dicts from parse_yolo_label
        visible_classes: Dict mapping class_id to visibility bool
        
    Returns:
        Image with bounding boxes drawn
    """
    img_with_boxes = image.copy()
    img_height, img_width = image.shape[:2]
    
    for detection in detections:
        class_id = detection['class_id']
        
        # Skip if this class is not visible
        if not visible_classes.get(class_id, True):
            continue
        
        # Get pixel coordinates
        x1, y1, x2, y2 = yolo_to_pixel_coords(detection, img_width, img_height)
        
        # Get color for this class
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        
        # Prepare label text
        label = detection['class_name']
        
        # Get label size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
        )
        
        # Draw label background
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            (255, 255, 255),  # White text
            LABEL_THICKNESS
        )
    
    return img_with_boxes


def count_detections(detections: List[Dict]) -> Dict[str, int]:
    """
    Count detections per class.
    
    Args:
        detections: List of detection dicts
        
    Returns:
        Dict mapping class_name to count
    """
    counts = {}
    for detection in detections:
        class_name = detection['class_name']
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        OpenCV image or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None
    
    return image


def get_label_path_from_image(image_path: str, labels_dir: str) -> str:
    """
    Get corresponding label file path from image path.
    
    Args:
        image_path: Path to image file
        labels_dir: Directory containing label files
        
    Returns:
        Path to label file
    """
    image_filename = os.path.basename(image_path)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_filename)
    return label_path