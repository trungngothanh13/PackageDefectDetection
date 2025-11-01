"""
Simple Package Quality Classification Demo
Choose an image and see which class it belongs to
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "../runs/package_quality_classification/train_20251031_133632/weights/best.pt"

# ============================================================================
# APPLICATION
# ============================================================================

class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Package Classifier")
        self.root.geometry("600x500")
        
        # Create main frame with scrollbar
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=1)
        
        # Add scrollbar
        self.scrollbar = tk.Scrollbar(self.main_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas
        self.canvas = tk.Canvas(self.main_frame, yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Configure scrollbar
        self.scrollbar.config(command=self.canvas.yview)
        
        # Create frame inside canvas for content
        self.content_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.content_frame, anchor='nw')
        
        # Load model
        try:
            self.model = YOLO(MODEL_PATH)
            print("✓ Model loaded")
        except:
            messagebox.showerror("Error", f"Model not found:\n{MODEL_PATH}")
            self.model = None
        
        # UI
        tk.Button(
            self.content_frame, 
            text="Choose Image", 
            command=self.choose_image,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=10
        ).pack(pady=20, padx=10, anchor='w')
        
        self.image_label = tk.Label(self.content_frame, bg="white")
        self.image_label.pack(pady=10, padx=10, anchor='w')
        
        self.result_label = tk.Label(
            self.content_frame,
            text="No prediction yet",
            font=("Arial", 20, "bold"),
            fg="#666"
        )
        self.result_label.pack(pady=10, padx=10)
        
        # Bind frame configuration
        self.content_frame.bind('<Configure>', self.on_frame_configure)
        
    def on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All", "*.*")]
        )
        if path:
            self.classify(path)
    
    def classify(self, path):
        if not self.model:
            return
        
        try:
            # Show image
            img = Image.open(path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            # Predict
            results = self.model.predict(source=path, verbose=False)
            probs = results[0].probs
            class_name = results[0].names[probs.top1]
            confidence = probs.top1conf.item()
            
            # Show result
            colors = {'good': '#4CAF50', 'ok': '#FF9800', 'bad': '#f44336'}
            color = colors.get(class_name, '#666')
            
            self.result_label.config(
                text=f"{class_name.upper()} ({confidence*100:.1f}%)",
                fg=color
            )
            
            print(f"✓ {class_name} ({confidence*100:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ============================================================================
# MAIN
# ============================================================================

root = tk.Tk()
app = ClassifierApp(root)
root.mainloop()