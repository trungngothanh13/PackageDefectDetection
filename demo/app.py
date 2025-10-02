"""
Package Defect Detection - Demo Application
Main GUI using Tkinter
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from config import (
    WINDOW_TITLE, CANVAS_WIDTH, CANVAS_HEIGHT,
    CLASS_NAMES, RAW_IMAGES_DIR, RAW_LABELS_DIR,
    DEFAULT_VISIBLE_CLASSES
)
from utils import (
    load_image, parse_yolo_label, draw_bounding_boxes,
    count_detections, get_label_path_from_image
)


class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        
        # State variables
        self.current_image_path = None
        self.current_image = None
        self.current_detections = []
        self.visible_classes = DEFAULT_VISIBLE_CLASSES.copy()
        
        # Build UI
        self._build_ui()
        
    def _build_ui(self):
        """Build the user interface"""
        
        # Top Frame - File selection
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Button(
            top_frame, 
            text="📁 Select Image", 
            command=self.select_image,
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=5
        ).pack(side=tk.LEFT)
        
        self.current_file_label = tk.Label(
            top_frame, 
            text="Current: None", 
            font=("Arial", 10),
            fg="#666"
        )
        self.current_file_label.pack(side=tk.LEFT, padx=20)
        
        # Separator
        tk.Frame(self.root, height=2, bg="#ccc").pack(fill=tk.X, pady=5)
        
        # Toggle Frame - Class visibility checkboxes
        toggle_frame = tk.Frame(self.root, padx=10, pady=10)
        toggle_frame.pack(fill=tk.X)
        
        tk.Label(
            toggle_frame, 
            text="Toggle Classes:", 
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT)
        
        self.class_vars = {}
        for i, class_name in enumerate(CLASS_NAMES):
            var = tk.BooleanVar(value=True)
            self.class_vars[i] = var
            
            cb = tk.Checkbutton(
                toggle_frame,
                text=class_name,
                variable=var,
                command=self.update_display,
                font=("Arial", 9)
            )
            cb.pack(side=tk.LEFT, padx=5)
        
        # Separator
        tk.Frame(self.root, height=2, bg="#ccc").pack(fill=tk.X, pady=5)
        
        # Image Display Frame
        image_frame = tk.Frame(self.root, padx=10, pady=10)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            image_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#f0f0f0"
        )
        self.canvas.pack()
        
        # Info Frame - Detection information
        info_frame = tk.Frame(self.root, padx=10, pady=10, bg="#f9f9f9")
        info_frame.pack(fill=tk.X)
        
        tk.Label(
            info_frame,
            text="Detection Info:",
            font=("Arial", 10, "bold"),
            bg="#f9f9f9"
        ).pack(anchor=tk.W)
        
        self.info_text = tk.Text(
            info_frame,
            height=5,
            font=("Arial", 9),
            bg="#f9f9f9",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.X, pady=5)
        
        # Separator
        tk.Frame(self.root, height=2, bg="#ccc").pack(fill=tk.X, pady=5)
        
        # Future Expansion Frame - Verdict (placeholder)
        verdict_frame = tk.Frame(self.root, padx=10, pady=10, bg="#e8f4f8")
        verdict_frame.pack(fill=tk.X)
        
        tk.Label(
            verdict_frame,
            text="[Verdict Section - Reserved for Future]",
            font=("Arial", 9, "italic"),
            fg="#888",
            bg="#e8f4f8"
        ).pack()
        
    def select_image(self):
        """Open file dialog to select an image"""
        initial_dir = RAW_IMAGES_DIR if os.path.exists(RAW_IMAGES_DIR) else "."
        
        file_path = filedialog.askopenfilename(
            title="Select Package Image",
            initialdir=initial_dir,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_and_display_image(file_path)
    
    def load_and_display_image(self, image_path):
        """Load image and its labels, then display"""
        
        # Load image
        image = load_image(image_path)
        if image is None:
            messagebox.showerror("Error", f"Failed to load image:\n{image_path}")
            return
        
        # Get label path
        label_path = get_label_path_from_image(image_path, RAW_LABELS_DIR)
        
        # Parse labels
        detections = parse_yolo_label(label_path)
        
        # Update state
        self.current_image_path = image_path
        self.current_image = image
        self.current_detections = detections
        
        # Update UI
        filename = os.path.basename(image_path)
        self.current_file_label.config(text=f"Current: {filename}")
        
        # Display
        self.update_display()
        self.update_info()
    
    def update_display(self):
        """Redraw image with bounding boxes based on visible classes"""
        
        if self.current_image is None:
            return
        
        # Update visible classes from checkboxes
        for class_id, var in self.class_vars.items():
            self.visible_classes[class_id] = var.get()
        
        # Draw bounding boxes
        img_with_boxes = draw_bounding_boxes(
            self.current_image,
            self.current_detections,
            self.visible_classes
        )
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        img_height, img_width = img_rgb.shape[:2]
        scale = min(CANVAS_WIDTH / img_width, CANVAS_HEIGHT / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        img_resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            CANVAS_WIDTH // 2,
            CANVAS_HEIGHT // 2,
            image=self.photo,
            anchor=tk.CENTER
        )
    
    def update_info(self):
        """Update detection information panel"""
        
        if not self.current_detections:
            info = "No detections found (or label file missing)"
        else:
            counts = count_detections(self.current_detections)
            total = len(self.current_detections)
            
            info = f"Total detections: {total}\n\n"
            for class_name, count in sorted(counts.items()):
                info += f"• {class_name}: {count}\n"
        
        # Update text widget
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = DefectDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()