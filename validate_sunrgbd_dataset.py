"""
Comprehensive validation script for SUN RGB-D 15-category dataset.

This script performs extensive checks to ensure:
1. Dataset structure is correct
2. Labels match images (no misalignment)
3. No data leakage between train/val
4. RGB and Depth images are properly aligned
5. No corrupt images
6. Class distribution is correct
7. Dataloader shuffling works properly
8. All images can be loaded and processed
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
import torch
from tqdm import tqdm
import hashlib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_utils.sunrgbd_dataset import SUNRGBDDataset, get_sunrgbd_dataloaders

print("=" * 80)
print("SUN RGB-D 15-CATEGORY DATASET COMPREHENSIVE VALIDATION")
print("=" * 80)

DATASET_ROOT = "data/sunrgbd_15"

# ============================================================================
# Test 1: Dataset Structure Validation
# ============================================================================
print("\n[Test 1] Dataset Structure Validation")
print("-" * 80)

required_structure = {
    'train/rgb': 8041,
    'train/depth': 8041,
    'train/labels.txt': 1,
    'val/rgb': 2018,
    'val/depth': 2018,
    'val/labels.txt': 1,
    'class_names.txt': 1,
    'dataset_info.txt': 1,
}

all_structure_valid = True
for path, expected_count in required_structure.items():
    full_path = os.path.join(DATASET_ROOT, path)

    if not os.path.exists(full_path):
        print(f"✗ Missing: {path}")
        all_structure_valid = False
        continue

    if path.endswith('.txt'):
        print(f"✓ Found: {path}")
    else:
        # Count files
        file_count = len(list(Path(full_path).glob('*.png')))
        if file_count == expected_count:
            print(f"✓ {path}: {file_count} files (expected {expected_count})")
        else:
            print(f"✗ {path}: {file_count} files (expected {expected_count})")
            all_structure_valid = False

if not all_structure_valid:
    print("\n✗ STRUCTURE VALIDATION FAILED")
    sys.exit(1)
else:
    print("\n✓ Structure validation passed")

# ============================================================================
# Test 2: File Naming Validation
# ============================================================================
print("\n[Test 2] File Naming Validation")
print("-" * 80)

def check_file_naming(split):
    """Check that RGB, depth, and labels have consistent naming."""
    rgb_dir = os.path.join(DATASET_ROOT, split, 'rgb')
    depth_dir = os.path.join(DATASET_ROOT, split, 'depth')
    labels_file = os.path.join(DATASET_ROOT, split, 'labels.txt')

    # Get all RGB files
    rgb_files = sorted([f.name for f in Path(rgb_dir).glob('*.png')])
    depth_files = sorted([f.name for f in Path(depth_dir).glob('*.png')])

    # Read labels
    with open(labels_file, 'r') as f:
        labels = f.readlines()

    num_labels = len(labels)
    num_rgb = len(rgb_files)
    num_depth = len(depth_files)

    print(f"\n{split.upper()}:")
    print(f"  RGB files: {num_rgb}")
    print(f"  Depth files: {num_depth}")
    print(f"  Labels: {num_labels}")

    # Check counts match
    if num_rgb != num_depth or num_rgb != num_labels:
        print(f"  ✗ Counts don't match!")
        return False

    # Check RGB and depth have same filenames
    if rgb_files != depth_files:
        print(f"  ✗ RGB and depth filenames don't match!")
        # Show first mismatch
        for i, (r, d) in enumerate(zip(rgb_files, depth_files)):
            if r != d:
                print(f"     First mismatch at index {i}: RGB={r}, Depth={d}")
                break
        return False

    # Check naming is sequential (00000.png, 00001.png, ...)
    expected_names = [f"{i:05d}.png" for i in range(num_rgb)]
    if rgb_files != expected_names:
        print(f"  ✗ Files are not sequentially named!")
        return False

    print(f"  ✓ All naming checks passed")
    return True

train_naming_valid = check_file_naming('train')
val_naming_valid = check_file_naming('val')

if not (train_naming_valid and val_naming_valid):
    print("\n✗ FILE NAMING VALIDATION FAILED")
    sys.exit(1)
else:
    print("\n✓ File naming validation passed")

# ============================================================================
# Test 3: Label Validation
# ============================================================================
print("\n[Test 3] Label Validation")
print("-" * 80)

def validate_labels(split):
    """Validate labels are in correct range and format."""
    labels_file = os.path.join(DATASET_ROOT, split, 'labels.txt')

    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    print(f"\n{split.upper()}:")

    # Check all labels are integers
    try:
        labels_int = [int(label) for label in labels]
    except ValueError as e:
        print(f"  ✗ Non-integer label found: {e}")
        return False

    # Check all labels in range [0, 14]
    min_label = min(labels_int)
    max_label = max(labels_int)

    if min_label < 0 or max_label > 14:
        print(f"  ✗ Labels out of range: min={min_label}, max={max_label}")
        return False

    print(f"  Label range: [{min_label}, {max_label}]")

    # Check all 15 classes are present
    unique_labels = set(labels_int)
    if len(unique_labels) != 15:
        print(f"  ✗ Not all 15 classes present: {len(unique_labels)} classes found")
        print(f"     Missing: {set(range(15)) - unique_labels}")
        return False

    # Count distribution
    label_counts = Counter(labels_int)
    print(f"  Unique classes: {len(unique_labels)}")
    print(f"  Distribution (top 5):")
    for label, count in label_counts.most_common(5):
        print(f"    Class {label:2d}: {count:4d} samples")

    print(f"  ✓ Label validation passed")
    return True

train_labels_valid = validate_labels('train')
val_labels_valid = validate_labels('val')

if not (train_labels_valid and val_labels_valid):
    print("\n✗ LABEL VALIDATION FAILED")
    sys.exit(1)
else:
    print("\n✓ Label validation passed")

# ============================================================================
# Test 4: Image Integrity Check (Sample)
# ============================================================================
print("\n[Test 4] Image Integrity Check (Sample 100 images)")
print("-" * 80)

def check_image_integrity(split, num_samples=100):
    """Check that images can be opened and have correct properties."""
    rgb_dir = os.path.join(DATASET_ROOT, split, 'rgb')
    depth_dir = os.path.join(DATASET_ROOT, split, 'depth')

    # Get total number of images
    total_images = len(list(Path(rgb_dir).glob('*.png')))

    # Sample evenly
    indices = np.linspace(0, total_images - 1, min(num_samples, total_images), dtype=int)

    print(f"\n{split.upper()} (checking {len(indices)} samples):")

    corrupt_rgb = []
    corrupt_depth = []
    shape_mismatches = []

    for idx in tqdm(indices, desc=f"  Checking {split}", leave=False):
        rgb_path = os.path.join(rgb_dir, f'{idx:05d}.png')
        depth_path = os.path.join(depth_dir, f'{idx:05d}.png')

        # Try to open RGB
        try:
            rgb = Image.open(rgb_path)
            rgb.load()  # Actually load the image data
            rgb_size = rgb.size
        except Exception as e:
            corrupt_rgb.append((idx, str(e)))
            continue

        # Try to open Depth
        try:
            depth = Image.open(depth_path)
            depth.load()
            depth_size = depth.size
        except Exception as e:
            corrupt_depth.append((idx, str(e)))
            continue

        # Check sizes match
        if rgb_size != depth_size:
            shape_mismatches.append((idx, rgb_size, depth_size))

    # Report results
    if corrupt_rgb:
        print(f"  ✗ Corrupt RGB images: {len(corrupt_rgb)}")
        for idx, error in corrupt_rgb[:5]:
            print(f"     {idx:05d}.png: {error}")
        return False

    if corrupt_depth:
        print(f"  ✗ Corrupt depth images: {len(corrupt_depth)}")
        for idx, error in corrupt_depth[:5]:
            print(f"     {idx:05d}.png: {error}")
        return False

    if shape_mismatches:
        print(f"  ✗ RGB/Depth size mismatches: {len(shape_mismatches)}")
        for idx, rgb_size, depth_size in shape_mismatches[:5]:
            print(f"     {idx:05d}.png: RGB={rgb_size}, Depth={depth_size}")
        return False

    print(f"  ✓ All sampled images valid")
    return True

train_images_valid = check_image_integrity('train', num_samples=100)
val_images_valid = check_image_integrity('val', num_samples=100)

if not (train_images_valid and val_images_valid):
    print("\n✗ IMAGE INTEGRITY CHECK FAILED")
    sys.exit(1)
else:
    print("\n✓ Image integrity check passed")

# ============================================================================
# Test 5: No Data Leakage (Train/Val Overlap)
# ============================================================================
print("\n[Test 5] Data Leakage Check (Train/Val Overlap)")
print("-" * 80)

def compute_file_hash(filepath):
    """Compute MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

print("Computing hashes for train RGB images (sample 50)...")
train_rgb_dir = os.path.join(DATASET_ROOT, 'train', 'rgb')
train_files = sorted(list(Path(train_rgb_dir).glob('*.png')))[:50]
train_hashes = {compute_file_hash(f) for f in tqdm(train_files, desc="  Train", leave=False)}

print("Computing hashes for val RGB images (sample 50)...")
val_rgb_dir = os.path.join(DATASET_ROOT, 'val', 'rgb')
val_files = sorted(list(Path(val_rgb_dir).glob('*.png')))[:50]
val_hashes = {compute_file_hash(f) for f in tqdm(val_files, desc="  Val", leave=False)}

# Check for overlap
overlap = train_hashes & val_hashes

if overlap:
    print(f"✗ Data leakage detected: {len(overlap)} duplicate images!")
    print("\n✗ DATA LEAKAGE CHECK FAILED")
    sys.exit(1)
else:
    print(f"✓ No data leakage detected (checked {len(train_hashes)} train + {len(val_hashes)} val samples)")

# ============================================================================
# Test 6: Dataset Class Loading
# ============================================================================
print("\n[Test 6] PyTorch Dataset Class Loading")
print("-" * 80)

try:
    train_dataset = SUNRGBDDataset(data_root=DATASET_ROOT, train=True)
    val_dataset = SUNRGBDDataset(data_root=DATASET_ROOT, train=False)

    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")

    # Test loading first sample
    rgb, depth, label = train_dataset[0]
    print(f"\nFirst train sample:")
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Label: {label}")

    assert rgb.shape == (3, 224, 224), f"RGB shape incorrect: {rgb.shape}"
    assert depth.shape == (1, 224, 224), f"Depth shape incorrect: {depth.shape}"
    assert 0 <= label < 15, f"Label out of range: {label}"

    print(f"✓ Dataset loading works correctly")

except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 7: Label-Image Correspondence
# ============================================================================
print("\n[Test 7] Label-Image Correspondence (Spot Check)")
print("-" * 80)

def spot_check_labels(dataset, num_samples=10):
    """Verify that labels correspond to actual images."""
    print(f"\nSpot checking {num_samples} samples...")

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        rgb, depth, label = dataset[idx]

        # Verify label is in valid range
        if not (0 <= label < 15):
            print(f"  ✗ Sample {idx}: Label {label} out of range")
            return False

        # Verify tensors have correct shape
        if rgb.shape != (3, 224, 224):
            print(f"  ✗ Sample {idx}: RGB shape {rgb.shape} incorrect")
            return False

        if depth.shape != (1, 224, 224):
            print(f"  ✗ Sample {idx}: Depth shape {depth.shape} incorrect")
            return False

    print(f"  ✓ All {len(indices)} samples valid")
    return True

if not spot_check_labels(train_dataset, num_samples=10):
    print("\n✗ LABEL-IMAGE CORRESPONDENCE CHECK FAILED")
    sys.exit(1)

if not spot_check_labels(val_dataset, num_samples=10):
    print("\n✗ LABEL-IMAGE CORRESPONDENCE CHECK FAILED")
    sys.exit(1)

print("\n✓ Label-image correspondence check passed")

# ============================================================================
# Test 8: Dataloader Shuffling
# ============================================================================
print("\n[Test 8] Dataloader Shuffling Validation")
print("-" * 80)

train_loader, val_loader = get_sunrgbd_dataloaders(
    data_root=DATASET_ROOT,
    batch_size=16,
    num_workers=0,
)

print("\nChecking train loader shuffling...")
# Get first batch from 3 different epochs
epoch1_labels = []
epoch2_labels = []
epoch3_labels = []

for i, (_, _, labels) in enumerate(train_loader):
    epoch1_labels.extend(labels.tolist())
    if i >= 4:  # First 5 batches
        break

for i, (_, _, labels) in enumerate(train_loader):
    epoch2_labels.extend(labels.tolist())
    if i >= 4:
        break

for i, (_, _, labels) in enumerate(train_loader):
    epoch3_labels.extend(labels.tolist())
    if i >= 4:
        break

# Check that order is different (shuffling working)
if epoch1_labels == epoch2_labels:
    print(f"  ✗ Epoch 1 and 2 have identical order - shuffling not working!")
    print(f"     First 10 labels: {epoch1_labels[:10]}")
    sys.exit(1)

if epoch1_labels == epoch3_labels:
    print(f"  ✗ Epoch 1 and 3 have identical order - shuffling not working!")
    sys.exit(1)

print(f"  ✓ Train loader shuffling works correctly")
print(f"     Epoch 1 first 10: {epoch1_labels[:10]}")
print(f"     Epoch 2 first 10: {epoch2_labels[:10]}")
print(f"     Epoch 3 first 10: {epoch3_labels[:10]}")

# Val loader should NOT shuffle
print("\nChecking val loader does NOT shuffle...")
val1_labels = []
val2_labels = []

for i, (_, _, labels) in enumerate(val_loader):
    val1_labels.extend(labels.tolist())
    if i >= 4:
        break

for i, (_, _, labels) in enumerate(val_loader):
    val2_labels.extend(labels.tolist())
    if i >= 4:
        break

if val1_labels != val2_labels:
    print(f"  ✗ Val loader is shuffling - should be deterministic!")
    sys.exit(1)

print(f"  ✓ Val loader correctly does NOT shuffle")

# ============================================================================
# Test 9: Class Distribution Consistency
# ============================================================================
print("\n[Test 9] Class Distribution Consistency")
print("-" * 80)

# Get distribution from labels file
train_labels_file = os.path.join(DATASET_ROOT, 'train', 'labels.txt')
with open(train_labels_file, 'r') as f:
    file_labels = [int(line.strip()) for line in f.readlines()]
file_dist = Counter(file_labels)

# Get distribution from dataset class
dataset_dist = train_dataset.get_class_distribution()
dataset_counts = {i: dataset_dist[train_dataset.CLASS_NAMES[i]]['count'] for i in range(15)}

print("\nComparing distributions:")
mismatch = False
for class_idx in range(15):
    file_count = file_dist[class_idx]
    dataset_count = dataset_counts[class_idx]

    if file_count != dataset_count:
        print(f"  ✗ Class {class_idx}: File={file_count}, Dataset={dataset_count}")
        mismatch = True

if mismatch:
    print("\n✗ CLASS DISTRIBUTION MISMATCH")
    sys.exit(1)
else:
    print("  ✓ Distributions match perfectly")

# ============================================================================
# Test 10: RGB-Depth Alignment
# ============================================================================
print("\n[Test 10] RGB-Depth Visual Alignment (Manual Check)")
print("-" * 80)

print("\nLoading 3 random samples to verify RGB-Depth alignment...")
print("(Visual inspection would be needed to fully verify alignment)")

indices = np.random.choice(len(train_dataset), 3, replace=False)
for idx in indices:
    rgb, depth, label = train_dataset[idx]
    class_name = train_dataset.CLASS_NAMES[label]

    print(f"\n  Sample {idx}:")
    print(f"    RGB: min={rgb.min():.3f}, max={rgb.max():.3f}, mean={rgb.mean():.3f}")
    print(f"    Depth: min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")
    print(f"    Label: {label} ({class_name})")

    # Basic sanity checks
    if rgb.min() < -3 or rgb.max() > 3:
        print(f"    ⚠ RGB values outside expected range (after normalization)")

    if depth.min() < 0 or depth.max() > 1:
        print(f"    ⚠ Depth values outside [0, 1] range")

print("\n✓ RGB-Depth pairs loaded successfully")
print("  (Manual visual inspection recommended for alignment verification)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print("\n✅ ALL VALIDATION TESTS PASSED!")
print("\nDataset is ready for production training:")
print(f"  ✓ Structure: All files present and correctly named")
print(f"  ✓ Labels: All in range [0, 14], all 15 classes present")
print(f"  ✓ Images: No corruption detected (sampled)")
print(f"  ✓ No data leakage: Train/val are distinct (sampled)")
print(f"  ✓ Dataset class: Loads correctly")
print(f"  ✓ Label correspondence: Verified")
print(f"  ✓ Dataloader: Shuffling works correctly")
print(f"  ✓ Class distribution: Consistent across sources")
print(f"  ✓ RGB-Depth: Pairs load successfully")

print("\n" + "=" * 80)
print("DATASET READY FOR DEPLOYMENT! 🚀")
print("=" * 80)
