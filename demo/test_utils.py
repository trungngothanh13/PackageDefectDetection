"""
Test script to verify config and utils work correctly
"""

import os
import cv2
from config import RAW_IMAGES_DIR, RAW_LABELS_DIR, CLASS_NAMES, DEFAULT_VISIBLE_CLASSES
from utils import (
    parse_yolo_label, 
    load_image, 
    draw_bounding_boxes, 
    count_detections,
    get_label_path_from_image
)


def test_basic_functionality():
    """Test basic loading and parsing"""
    
    print("=" * 60)
    print("Testing Package Defect Detection Demo Utils")
    print("=" * 60)
    
    # Check if directories exist
    print(f"\n1. Checking paths...")
    print(f"   Images dir: {RAW_IMAGES_DIR}")
    print(f"   Labels dir: {RAW_LABELS_DIR}")
    print(f"   Images dir exists: {os.path.exists(RAW_IMAGES_DIR)}")
    print(f"   Labels dir exists: {os.path.exists(RAW_LABELS_DIR)}")
    
    # List some images
    if os.path.exists(RAW_IMAGES_DIR):
        images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\n2. Found {len(images)} images")
        if images:
            print(f"   First few: {images[:5]}")
    else:
        print("\n⚠️  Images directory not found! Please create it and add your images.")
        return
    
    # Test with first image
    if images:
        test_image_name = images[0]
        image_path = os.path.join(RAW_IMAGES_DIR, test_image_name)
        label_path = get_label_path_from_image(image_path, RAW_LABELS_DIR)
        
        print(f"\n3. Testing with image: {test_image_name}")
        print(f"   Label path: {label_path}")
        print(f"   Label exists: {os.path.exists(label_path)}")
        
        # Load image
        image = load_image(image_path)
        if image is not None:
            print(f"   ✓ Image loaded: {image.shape}")
        else:
            print(f"   ✗ Failed to load image")
            return
        
        # Parse label
        detections = parse_yolo_label(label_path)
        print(f"\n4. Parsed {len(detections)} detections:")
        for i, det in enumerate(detections):
            print(f"   [{i}] Class {det['class_id']} ({det['class_name']}): "
                  f"center=({det['x_center']:.3f}, {det['y_center']:.3f}), "
                  f"size=({det['width']:.3f} x {det['height']:.3f})")
        
        # Count detections
        counts = count_detections(detections)
        print(f"\n5. Detection counts:")
        for class_name, count in counts.items():
            print(f"   • {class_name}: {count}")
        
        # Draw boxes
        print(f"\n6. Drawing bounding boxes...")
        img_with_boxes = draw_bounding_boxes(image, detections, DEFAULT_VISIBLE_CLASSES)
        
        # Save test output
        output_path = "test_output.jpg"
        cv2.imwrite(output_path, img_with_boxes)
        print(f"   ✓ Saved test image to: {output_path}")
        print(f"   Open it to verify bounding boxes are drawn correctly!")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Ready to build the GUI.")
        print("=" * 60)


if __name__ == "__main__":
    test_basic_functionality()