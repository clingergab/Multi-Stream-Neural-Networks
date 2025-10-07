"""
Pre-computed class weights for NYU Depth V2 dataset.
Use these to handle severe class imbalance (192x ratio).
"""

import torch

# Inverse frequency weights (normalized)
# Generated from full dataset analysis
NYU_CLASS_WEIGHTS = torch.tensor([
    1.0626,  # 0: basement (7 samples)
    0.0615,  # 1: bathroom (121 samples)
    0.0194,  # 2: bedroom (383 samples) - most common
    0.2066,  # 3: bookstore (36 samples)
    1.4876,  # 4: cafe (5 samples)
    0.1518,  # 5: classroom (49 samples)
    1.2397,  # 6: computer_lab (6 samples)
    1.4876,  # 7: conference_room (5 samples)
    1.8596,  # 8: dinette (4 samples)
    0.0636,  # 9: dining_room (117 samples)
    2.4794,  # 10: excercise_room (3 samples)
    1.8596,  # 11: foyer (4 samples)
    0.2755,  # 12: furniture_store (27 samples)
    0.1488,  # 13: home_office (50 samples)
    1.4876,  # 14: home_storage (5 samples)
    3.7191,  # 15: indoor_balcony (2 samples) - most rare
    0.0331,  # 16: kitchen (225 samples)
    2.4794,  # 17: laundry_room (3 samples)
    0.0337,  # 18: living_room (221 samples)
    0.0954,  # 19: office (78 samples)
    0.7438,  # 20: office_kitchen (10 samples)
    0.2399,  # 21: playroom (31 samples)
    2.4794,  # 22: printer_room (3 samples)
    0.4375,  # 23: reception_room (17 samples)
    1.4876,  # 24: student_lounge (5 samples)
    0.2975,  # 25: study (25 samples)
    1.0626,  # 26: study_room (7 samples)
])

# Alternative: Remove ultra-rare classes (optional)
# Classes with <5 samples: [4, 7, 8, 10, 11, 14, 15, 17, 22, 24]
ULTRA_RARE_CLASSES = [4, 7, 8, 10, 11, 14, 15, 17, 22, 24]


def get_class_weights(device='cpu'):
    """Get class weights for NYU dataset."""
    return NYU_CLASS_WEIGHTS.to(device)


def create_weighted_loss(device='cpu'):
    """Create weighted CrossEntropyLoss for NYU dataset."""
    import torch.nn as nn
    weights = get_class_weights(device)
    return nn.CrossEntropyLoss(weight=weights)


# Statistics
print("NYU Depth V2 Class Weight Statistics:")
print(f"  Total classes: 27")
print(f"  Imbalance ratio: 192x (bedroom: 383 samples, indoor_balcony: 2 samples)")
print(f"  Top 3 classes: 57.2% of dataset")
print(f"  Classes with <5 samples: {len(ULTRA_RARE_CLASSES)}")
print(f"\nUsage:")
print(f"  from class_weights_nyu import create_weighted_loss")
print(f"  criterion = create_weighted_loss(device='cuda')")
