"""
Configuration file for Package Defect Detection Demo
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw", "images")
RAW_LABELS_DIR = os.path.join(DATA_DIR, "raw", "labels")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.txt")

# ============================================================================
# MODEL PATHS
# ============================================================================
# Point this to your trained model
# After training, it will be at: runs/package_defect_detection/train_XXXXXX/weights/best.pt
MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "package_defect_detection", "train_20251008_203202", "weights", "best.pt")

# If model doesn't exist, we'll use pre-labeled data
MODEL_AVAILABLE = os.path.exists(MODEL_PATH) if MODEL_PATH else False

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================
CLASS_NAMES = [
    "optimal",   # 0
    "good",      # 1
    "deformed",  # 2
    "dented",    # 3
    "poked",     # 4
    "cut"        # 5
]

# Color for each class (BGR format for OpenCV)
# Using distinct colors for easy visual distinction
CLASS_COLORS = {
    0: (0, 255, 0),      # optimal - Green
    1: (0, 255, 255),    # good - Yellow
    2: (0, 0, 255),      # deformed - Red
    3: (0, 165, 255),    # dented - Orange
    4: (255, 0, 255),    # poked - Magenta
    5: (255, 0, 0)       # cut - Blue
}

# ============================================================================
# UI SETTINGS
# ============================================================================
WINDOW_TITLE = "Package Defect Detection - Demo v2.0 (AI Inference)"
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

# Bounding box drawing settings
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 2

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to show detection
IOU_THRESHOLD = 0.45         # Non-maximum suppression threshold
IMG_SIZE = 640               # Model input size

# ============================================================================
# DEFAULT STATES
# ============================================================================
# All classes visible by default
DEFAULT_VISIBLE_CLASSES = {i: True for i in range(len(CLASS_NAMES))}

# ============================================================================
# VERDICT LOGIC SETTINGS
# ============================================================================
# Define verdict based on detected defects
VERDICT_RULES = {
    "pass": ["optimal", "good"],           # Package passes if only these
    "fail": ["deformed", "dented", "poked", "cut"]  # Fails if any of these
}