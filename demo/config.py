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
WINDOW_TITLE = "Package Defect Detection - Demo v1.0"
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

# Bounding box drawing settings
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 2

# ============================================================================
# DEFAULT STATES
# ============================================================================
# All classes visible by default
DEFAULT_VISIBLE_CLASSES = {i: True for i in range(len(CLASS_NAMES))}

# ============================================================================
# FUTURE EXPANSION PLACEHOLDERS
# ============================================================================
# Model inference settings (for future use)
MODEL_PATH = None  # Will point to trained model later
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Verdict logic settings (for future use)
VERDICT_RULES = {
    "optimal": "No defects detected",
    "good": "Minor defects acceptable",
    "deformed": "Significant defects found"
}