"""
Check YOLO predictions on actual fake images
"""
import sys
sys.path.insert(0, r"C:\Users\korea\Documents\GitHub\gen-clear-drive\pytorch-CycleGAN-and-pix2pix")

import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# Load YOLO model
yolo = YOLO(r"C:\Users\korea\Documents\GitHub\gen-clear-drive\yolo11s.pt")

# Check on a random night image
night_img_dir = Path(r"C:\Users\korea\Documents\GitHub\gen-clear-drive\datasets\yolo_bdd100k\clear_night\images\test")
night_imgs = list(night_img_dir.glob("*.jpg"))[:5]  # First 5 images

print("=" * 60)
print("Testing YOLO on REAL night images")
print("=" * 60)

for img_path in night_imgs:
    print(f"\n📷 {img_path.name}")
    
    # Run YOLO
    results = yolo(str(img_path), verbose=False)
    
    # Check predictions
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            print(f"  Predictions: {len(boxes)} objects")
            
            # Count by class
            class_counts = {}
            for cls in classes:
                cls_id = int(cls)
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            
            print(f"  Class distribution: {class_counts}")
            print(f"  Confidence range: [{confs.min():.2f}, {confs.max():.2f}]")
        else:
            print(f"  No predictions")
    else:
        print(f"  No predictions")

# Now check what GT labels say
print("\n" + "=" * 60)
print("Checking GT labels for same images")
print("=" * 60)

label_dir = Path(r"C:\Users\korea\Documents\GitHub\gen-clear-drive\datasets\yolo_bdd100k\clear_night\labels\test")

for img_path in night_imgs:
    label_path = label_dir / (img_path.stem + ".txt")
    print(f"\n📋 {label_path.name}")
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 0:
            classes = [int(line.split()[0]) for line in lines if line.strip()]
            
            # Count by class
            class_counts = {}
            for cls in classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print(f"  GT objects: {len(lines)}")
            print(f"  GT class distribution: {class_counts}")
        else:
            print(f"  Empty label file")
    else:
        print(f"  No label file")

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)
print("If YOLO pred classes are very different from GT classes,")
print("then class matching will fail and we'll get 0 TP!")
