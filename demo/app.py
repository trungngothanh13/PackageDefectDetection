"""
Package Defect Detection - Demo Application
Main GUI using Tkinter with AI Inference
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from config import (
    WINDOW_TITLE, CANVAS_WIDTH, CANVAS_HEIGHT,
    CLASS_NAMES, RAW_IMAGES_DIR, RAW_LABELS_DIR,
    DEFAULT_VISIBLE_CLASSES, MODEL_AVAILABLE, VERDICT_RULES
)
from utils import (
    load_image, parse_yolo_label, draw_bounding_boxes,
    count_detections, get_label_path_from_image
)
from inference import inference_engine


class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        
        # State variables
        self.current_image_path = None
        self.current_image = None
        self.current_detections = []
        self.visible_classes = DEFAULT_VISIBLE_CLASSES.copy()
        self.use_ai_inference = True  # Toggle between AI and pre-labeled
        
        # Build UI
        self._build_ui()
        
        # Show model status
        self._show_model_status()
        
    def _build_ui(self):
        """Build the user interface"""
        
        # Top Frame - File selection
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Button(
            top_frame, 
            text="Select Image", 
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
        
        # Mode toggle button
        self.mode_button = tk.Button(
            top_frame,
            text="AI Mode: ON",
            command=self.toggle_mode,
            font=("Arial", 9, "bold"),
            bg="#2196F3",
            fg="white",
            padx=15,
            pady=5
        )
        self.mode_button.pack(side=tk.RIGHT)
        
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
        
        # Verdict Frame
        self.verdict_frame = tk.Frame(self.root, padx=10, pady=10, bg="#e8f4f8")
        self.verdict_frame.pack(fill=tk.X)
        
        self.verdict_label = tk.Label(
            self.verdict_frame,
            text="Verdict: No image loaded",
            font=("Arial", 11, "bold"),
            fg="#666",
            bg="#e8f4f8"
        )
        self.verdict_label.pack()
        
    def _show_model_status(self):
        """Show model availability status"""
        if inference_engine.is_available():
            messagebox.showinfo(
                "Model Ready",
                "AI Model loaded successfully!\n\n"
            )
    
    def toggle_mode(self):
        """Toggle between AI inference and pre-labeled data"""
        if not inference_engine.is_available():
            messagebox.showwarning(
                "AI Not Available",
                "Trained model not found.\n"
                "Please train a model first!"
            )
            return
        
        self.use_ai_inference = not self.use_ai_inference
        
        if self.use_ai_inference:
            self.mode_button.config(text="AI Mode: ON", bg="#2196F3")
        else:
            self.mode_button.config(text="Pre-labeled Mode", bg="#FF9800")
        
        # Reload current image with new mode
        if self.current_image_path:
            self.load_and_display_image(self.current_image_path)
    
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
        """Load image and get detections (AI or pre-labeled), then display"""
        
        # Load image
        image = load_image(image_path)
        if image is None:
            messagebox.showerror("Error", f"Failed to load image:\n{image_path}")
            return
        
        # Get detections based on mode
        if self.use_ai_inference and inference_engine.is_available():
            # Use AI inference
            print("Running AI inference...")
            detections = inference_engine.predict(image_path)
            mode_text = "AI Inference"
        else:
            # Use pre-labeled data
            print("Loading pre-labeled data...")
            label_path = get_label_path_from_image(image_path, RAW_LABELS_DIR)
            detections = parse_yolo_label(label_path)
            mode_text = "Pre-labeled"
        
        # Update state
        self.current_image_path = image_path
        self.current_image = image
        self.current_detections = detections
        
        # Update UI
        filename = os.path.basename(image_path)
        self.current_file_label.config(text=f"Current: {filename} ({mode_text})")
        
        # Display
        self.update_display()
        self.update_info()
        self.update_verdict()
    
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
            info = "No detections found"
        else:
            counts = count_detections(self.current_detections)
            total = len(self.current_detections)
            
            info = f"Total detections: {total}\n\n"
            for class_name, count in sorted(counts.items()):
                info += f"• {class_name}: {count}\n"
            
            # Show confidence if AI mode
            if self.use_ai_inference and self.current_detections:
                avg_conf = sum(d.get('confidence', 0) for d in self.current_detections) / len(self.current_detections)
                info += f"\nAverage confidence: {avg_conf:.2%}"
        
        # Update text widget
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)
    
    def update_verdict(self):
        """Update quality verdict based on detections"""
        
        if not self.current_detections:
            self.verdict_label.config(
                text="Verdict: No detections",
                fg="#666",
                bg="#e8f4f8"
            )
            self.verdict_frame.config(bg="#e8f4f8")
            return
        
        # Count defect types
        counts = count_detections(self.current_detections)
        
        # Check for critical defects
        critical_defects = [d for d in VERDICT_RULES["fail"] if d in counts]
        
        if critical_defects:
            # Package fails
            defect_list = ", ".join(critical_defects)
            self.verdict_label.config(
                text=f"Verdict: Defects found ({defect_list})",
                fg="white",
                bg="#f44336"
            )
            self.verdict_frame.config(bg="#f44336")
        else:
            # Package passes
            self.verdict_label.config(
                text="Verdict: No critical defects",
                fg="white",
                bg="#4CAF50"
            )
            self.verdict_frame.config(bg="#4CAF50")


def main():
    root = tk.Tk()
    app = DefectDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()