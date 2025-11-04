"""
Quick validation script for YOLO Loss V2 class matching
"""
import sys
sys.path.insert(0, r"C:\Users\korea\Documents\GitHub\gen-clear-drive\pytorch-CycleGAN-and-pix2pix")

import torch
from models.yolo_loss_v2 import YOLODetectionLossV2

# Create loss function
loss_fn = YOLODetectionLossV2()

# Test case: Same class match
print("=" * 60)
print("TEST 1: Same class (should match)")
print("=" * 60)

# Mock predictions: [[cx, cy, w, h, obj, ...class_scores...]]
# Let's say we have 10 classes (0-9)
# Prediction: class=2 (car), bbox=[0.5, 0.5, 0.2, 0.3], conf=0.9
pred = torch.tensor([
    # [cx, cy, w, h, obj, cls0, cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9]
    [0.5, 0.5, 0.2, 0.3, 0.9, 0.1, 0.1, 0.8, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # class=2 (max at index 7=2+5)
]).unsqueeze(0)  # Add batch dim

# Ground truth: class=2 (car), bbox=[0.5, 0.5, 0.2, 0.3]
targets = torch.tensor([
    # [img_idx, class, cx, cy, w, h]
    [0, 2, 0.5, 0.5, 0.2, 0.3]
])

try:
    boxes, confs, classes = loss_fn._parse_predictions(pred)
    print(f"✓ Parsed predictions:")
    print(f"  Boxes: {boxes}")
    print(f"  Confs: {confs}")
    print(f"  Classes: {classes}")
    print(f"  Expected class: 2, Got: {classes[0].item()}")
    
    if classes[0].item() == 2:
        print("✅ Class extraction CORRECT!")
    else:
        print("❌ Class extraction WRONG!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 2: Different class (should NOT match)")
print("=" * 60)

# Prediction: class=0 (person), GT: class=2 (car)
pred2 = torch.tensor([
    [0.5, 0.5, 0.2, 0.3, 0.9, 0.8, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # class=0
]).unsqueeze(0)

targets2 = torch.tensor([
    [0, 2, 0.5, 0.5, 0.2, 0.3]  # class=2
])

try:
    boxes2, confs2, classes2 = loss_fn._parse_predictions(pred2)
    print(f"✓ Parsed predictions:")
    print(f"  Pred class: {classes2[0].item()}, GT class: 2")
    
    # Manually check class matching
    pred_class = classes2[0]
    gt_class = targets2[0, 1].long()
    
    if pred_class == gt_class:
        print("❌ Classes matched (should NOT match!)")
    else:
        print("✅ Classes did NOT match (correct!)")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 3: Full loss computation")
print("=" * 60)

try:
    # Test with matching classes
    loss1 = loss_fn(pred, targets)
    print(f"Loss (matching classes): {loss1.item():.4f}")
    
    # Test with non-matching classes  
    loss2 = loss_fn(pred2, targets2)
    print(f"Loss (different classes): {loss2.item():.4f}")
    
    print("\n✅ Full loss computation works!")
    
except Exception as e:
    print(f"❌ Error in loss computation: {e}")
    import traceback
    traceback.print_exc()
