"""
Simple Defect Detection Demo
Shows detected defects with bounding boxes and counts
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import os


# ============================================================================
# CONFIGURATION
# ============================================================================
# Update this to your detection model path
MODEL_PATH = "../runs/package_defect_detection/train_20251101_130621/weights/best.pt"

# Detection confidence threshold (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.25  # Adjust this if too many/few detections


# ============================================================================
# APPLICATION
# ============================================================================

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Defect Detection Demo")
        self.root.geometry("800x700")
        
        # Load model
        try:
            self.model = YOLO(MODEL_PATH)
            print("✓ Model loaded")
        except:
            messagebox.showerror("Error", f"Model not found:\n{MODEL_PATH}")
            self.model = None
        
        # UI
        tk.Button(
            root,
            text="Choose Image",
            command=self.choose_image,
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10
        ).pack(pady=20)
        
        # Image display
        self.image_label = tk.Label(root, bg="white")
        self.image_label.pack(pady=10)
        
        # Results frame
        self.result_frame = tk.Frame(root, bg="#f9f9f9", relief=tk.RAISED, borderwidth=2)
        self.result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Summary label
        self.summary_label = tk.Label(
            self.result_frame,
            text="No detections yet",
            font=("Arial", 14, "bold"),
            bg="#f9f9f9",
            fg="#666"
        )
        self.summary_label.pack(pady=10)
        
        # Details text
        self.details_text = tk.Text(
            self.result_frame,
            height=6,
            font=("Arial", 10),
            bg="#f9f9f9",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.details_text.pack(fill=tk.X, padx=10, pady=10)
    
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select Package Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All", "*.*")]
        )
        if path:
            self.detect(path)
    
    def detect(self, path):
        if not self.model:
            return
        
        try:
            # Run detection
            results = self.model.predict(
                source=path,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )
            
            result = results[0]
            
            # Load original image
            img = Image.open(path)
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                small_font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
                small_font = font
            
            # Count detections by class
            detections = {'torn': [], 'crushed': [], 'dented': []}
            
            # Draw bounding boxes
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Get class and confidence
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    # Store detection
                    detections[class_name].append(confidence)
                    
                    # Color by class
                    colors = {
                        'torn': '#FF6B6B',      # Red
                        'crushed': '#FFA500',   # Orange
                        'dented': '#4ECDC4'     # Teal
                    }
                    color = colors.get(class_name, '#999999')
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label background
                    label = f"{class_name} {confidence:.2f}"
                    bbox = draw.textbbox((x1, y1 - 25), label, font=small_font)
                    draw.rectangle(bbox, fill=color)
                    
                    # Draw label text
                    draw.text((x1, y1 - 25), label, fill="white", font=small_font)
            
            # Display image
            display_size = (700, 500)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            # Update results
            self.update_results(detections)
            
            # Print to console
            total = sum(len(v) for v in detections.values())
            print(f"\n✓ Detected {total} defects:")
            for class_name, confs in detections.items():
                if confs:
                    print(f"  {class_name}: {len(confs)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{e}")
            import traceback
            traceback.print_exc()
    
    def update_results(self, detections):
        """Update result display"""
        
        # Count total
        total = sum(len(v) for v in detections.values())
        
        # Update summary
        if total == 0:
            self.summary_label.config(
                text="✓ No defects detected",
                fg="#4CAF50"
            )
        else:
            self.summary_label.config(
                text=f"⚠️ Found {total} defect{'s' if total != 1 else ''}",
                fg="#f44336"
            )
        
        # Build details text
        details = []
        
        for class_name in ['torn', 'crushed', 'dented']:
            count = len(detections[class_name])
            if count > 0:
                avg_conf = sum(detections[class_name]) / count
                details.append(f"{class_name.upper()}: {count} detected (avg confidence: {avg_conf:.1%})")
        
        if not details:
            details.append("No defects found in this image.")
            details.append("\nThis package appears to be in good condition.")
        else:
            details.append("\n" + "─" * 50)
            details.append("Recommendation: Inspect package before shipping")
        
        # Update text widget
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, "\n".join(details))
        self.details_text.config(state=tk.DISABLED)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("=" * 70)
        print("MODEL NOT FOUND")
        print("=" * 70)
        print(f"Expected: {MODEL_PATH}")
        print("\nPlease update MODEL_PATH in the script.")
        print("=" * 70)
    
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()