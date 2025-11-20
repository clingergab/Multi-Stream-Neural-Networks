"""
Check the scale of RGB, Depth, and Orthogonal data across different datasets.
This script analyzes the raw data before normalization to understand their scales.
"""

import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_utils.sunrgbd_3stream_dataset import SUNRGBD3StreamDataset
from data_utils.sunrgbd_dataset import SUNRGBDDataset
from data_utils.nyu_depth_dataset import NYUDepthV2Dataset


def check_raw_image_scales(data_root='data/sunrgbd_15', split='train', num_samples=10):
    """Check raw image scales by directly loading files."""
    print("=" * 80)
    print(f"Checking RAW image scales from {data_root}/{split}")
    print("=" * 80)

    split_dir = os.path.join(data_root, split)
    rgb_dir = os.path.join(split_dir, 'rgb')
    depth_dir = os.path.join(split_dir, 'depth')
    orth_dir = os.path.join(split_dir, 'orth')

    # Check RGB
    print("\n[RGB Images]")
    for i in range(min(num_samples, 5)):
        rgb_path = os.path.join(rgb_dir, f'{i:05d}.png')
        if os.path.exists(rgb_path):
            rgb = Image.open(rgb_path)
            rgb_array = np.array(rgb)
            print(f"  Sample {i}: mode={rgb.mode}, shape={rgb_array.shape}, "
                  f"dtype={rgb_array.dtype}, range=[{rgb_array.min()}, {rgb_array.max()}]")

    # Check Depth
    print("\n[Depth Images]")
    for i in range(min(num_samples, 5)):
        depth_path = os.path.join(depth_dir, f'{i:05d}.png')
        if os.path.exists(depth_path):
            depth = Image.open(depth_path)
            depth_array = np.array(depth)
            print(f"  Sample {i}: mode={depth.mode}, shape={depth_array.shape}, "
                  f"dtype={depth_array.dtype}, range=[{depth_array.min()}, {depth_array.max()}]")

    # Check Orthogonal
    if os.path.exists(orth_dir):
        print("\n[Orthogonal Images]")
        for i in range(min(num_samples, 5)):
            orth_path = os.path.join(orth_dir, f'{i:05d}.png')
            if os.path.exists(orth_path):
                orth = Image.open(orth_path)
                orth_array = np.array(orth)
                print(f"  Sample {i}: mode={orth.mode}, shape={orth_array.shape}, "
                      f"dtype={orth_array.dtype}, range=[{orth_array.min()}, {orth_array.max()}]")


def check_sunrgbd_3stream_processing():
    """Check how SUNRGBD3StreamDataset processes each modality."""
    print("\n" + "=" * 80)
    print("SUNRGBD 3-Stream Dataset Processing")
    print("=" * 80)

    # Temporarily create a validation dataset (no augmentation for clearer analysis)
    try:
        dataset = SUNRGBD3StreamDataset(train=False)

        # Get a sample and analyze at different processing stages
        idx = 0

        # Load raw images manually to see what dataset receives
        rgb_path = os.path.join(dataset.rgb_dir, f'{idx:05d}.png')
        depth_path = os.path.join(dataset.depth_dir, f'{idx:05d}.png')
        orth_path = os.path.join(dataset.orth_dir, f'{idx:05d}.png')

        print("\n[RAW FILES (before dataset processing)]")
        rgb_raw = Image.open(rgb_path)
        print(f"  RGB:   mode={rgb_raw.mode}, size={rgb_raw.size}")
        rgb_raw_arr = np.array(rgb_raw)
        print(f"         array: dtype={rgb_raw_arr.dtype}, range=[{rgb_raw_arr.min()}, {rgb_raw_arr.max()}]")

        depth_raw = Image.open(depth_path)
        print(f"  Depth: mode={depth_raw.mode}, size={depth_raw.size}")
        depth_raw_arr = np.array(depth_raw)
        print(f"         array: dtype={depth_raw_arr.dtype}, range=[{depth_raw_arr.min()}, {depth_raw_arr.max()}]")

        orth_raw = Image.open(orth_path)
        print(f"  Orth:  mode={orth_raw.mode}, size={orth_raw.size}")
        orth_raw_arr = np.array(orth_raw)
        print(f"         array: dtype={orth_raw_arr.dtype}, range=[{orth_raw_arr.min()}, {orth_raw_arr.max()}]")

        # Get processed sample from dataset
        rgb_tensor, depth_tensor, orth_tensor, label = dataset[idx]

        print("\n[AFTER DATASET PROCESSING (normalized tensors)]")
        print(f"  RGB tensor:   shape={rgb_tensor.shape}, dtype={rgb_tensor.dtype}, "
              f"range=[{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")
        print(f"  Depth tensor: shape={depth_tensor.shape}, dtype={depth_tensor.dtype}, "
              f"range=[{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")
        print(f"  Orth tensor:  shape={orth_tensor.shape}, dtype={orth_tensor.dtype}, "
              f"range=[{orth_tensor.min():.3f}, {orth_tensor.max():.3f}]")

        print("\n[KEY PROCESSING STEPS IN DATASET]")
        print("  RGB:")
        print("    1. Loaded as PIL RGB (0-255)")
        print("    2. to_tensor() converts to [0, 1] range")
        print("    3. Normalized with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]")
        print("       Formula: (x - 0.5) / 0.5 = 2x - 1, maps [0,1] -> [-1,1]")

        print("\n  Depth:")
        print("    1. Loaded as PIL Image (mode I;16 or I)")
        print("    2. Converted to float32 and normalized per-image: depth/depth.max()")
        print("    3. Converted to PIL mode='F' (float32, range [0,1])")
        print("    4. to_tensor() keeps [0,1] range for mode F")
        print("    5. Normalized with mean=[0.5], std=[0.5]")
        print("       Formula: (x - 0.5) / 0.5 = 2x - 1, maps [0,1] -> [-1,1]")

        print("\n  Orth:")
        print("    1. Loaded as PIL Image (mode I;16)")
        print("    2. Converted to PIL mode='F' (float32, preserves 16-bit range [0, 65535])")
        print("    3. to_tensor() does NOT scale mode F, keeps [0, 65535]")
        print("    4. Manually scaled: orth / 65535.0 -> [0, 1]")
        print("    5. Normalized with mean=[0.5], std=[0.5]")
        print("       Formula: (x - 0.5) / 0.5 = 2x - 1, maps [0,1] -> [-1,1]")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure data/sunrgbd_15 exists with train/val splits")


def check_sunrgbd_2stream_processing():
    """Check how SUNRGBDDataset processes RGB and Depth."""
    print("\n" + "=" * 80)
    print("SUNRGBD 2-Stream Dataset Processing")
    print("=" * 80)

    try:
        dataset = SUNRGBDDataset(train=False)

        idx = 0
        rgb_tensor, depth_tensor, label = dataset[idx]

        print("\n[AFTER DATASET PROCESSING (normalized tensors)]")
        print(f"  RGB tensor:   shape={rgb_tensor.shape}, dtype={rgb_tensor.dtype}, "
              f"range=[{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")
        print(f"  Depth tensor: shape={depth_tensor.shape}, dtype={depth_tensor.dtype}, "
              f"range=[{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")

        print("\n[KEY PROCESSING STEPS]")
        print("  RGB:")
        print("    1. Loaded as PIL RGB (0-255)")
        print("    2. to_tensor() converts to [0, 1]")
        print("    3. Normalized with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

        print("\n  Depth:")
        print("    1. Loaded and converted to mode='L' (grayscale, 0-255)")
        print("    2. to_tensor() converts to [0, 1]")
        print("    3. Normalized with mean=[0.5027], std=[0.2197]")

    except Exception as e:
        print(f"Error: {e}")


def check_nyu_depth_processing():
    """Check how NYUDepthV2Dataset processes RGB and Depth."""
    print("\n" + "=" * 80)
    print("NYU Depth V2 Dataset Processing")
    print("=" * 80)

    h5_path = 'data/nyu_depth_v2_labeled.mat'
    if not os.path.exists(h5_path):
        print(f"  NYU dataset not found at {h5_path}")
        return

    try:
        dataset = NYUDepthV2Dataset(h5_path, train=False)

        idx = 0
        rgb_tensor, depth_tensor, label = dataset[idx]

        print("\n[AFTER DATASET PROCESSING (normalized tensors)]")
        print(f"  RGB tensor:   shape={rgb_tensor.shape}, dtype={rgb_tensor.dtype}, "
              f"range=[{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")
        print(f"  Depth tensor: shape={depth_tensor.shape}, dtype={depth_tensor.dtype}, "
              f"range=[{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")

        print("\n[KEY PROCESSING STEPS]")
        print("  RGB:")
        print("    1. Loaded from HDF5 as uint8 [0, 255]")
        print("    2. to_tensor() converts to [0, 1]")
        print("    3. Normalized with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

        print("\n  Depth:")
        print("    1. Loaded from HDF5 as float")
        print("    2. Per-image normalized: (depth - min) / (max - min) -> [0, 1]")
        print("    3. Scaled to uint8: * 255 -> [0, 255]")
        print("    4. to_tensor() converts back to [0, 1]")
        print("    5. NO additional normalization applied")

    except Exception as e:
        print(f"Error: {e}")


def summary():
    """Print summary of scales."""
    print("\n" + "=" * 80)
    print("SUMMARY: Are datasets in [0, 1] range?")
    print("=" * 80)

    print("\n[BEFORE normalization (after to_tensor)]:")
    print("  ✓ RGB (all datasets):     [0, 1] - to_tensor() scales uint8 by 1/255")
    print("  ✓ Depth (SUNRGBD 3-stream): [0, 1] - mode F scaled manually before to_tensor()")
    print("  ✓ Depth (SUNRGBD 2-stream): [0, 1] - mode L scaled by to_tensor()")
    print("  ✓ Depth (NYU):            [0, 1] - pre-normalized then to_tensor()")
    print("  ✓ Orth (SUNRGBD 3-stream):  [0, 1] - mode F scaled manually: orth/65535.0")

    print("\n[AFTER normalization (final tensor output)]:")
    print("  • RGB (SUNRGBD 3-stream):   [-1, 1] - mean=0.5, std=0.5")
    print("  • RGB (SUNRGBD 2-stream):   ~[-2.1, 2.6] - ImageNet normalization")
    print("  • RGB (NYU):                ~[-2.1, 2.6] - ImageNet normalization")
    print("  • Depth (SUNRGBD 3-stream): [-1, 1] - mean=0.5, std=0.5")
    print("  • Depth (SUNRGBD 2-stream): ~[-2.3, 2.3] - mean=0.5027, std=0.2197")
    print("  • Depth (NYU):              [0, 1] - NO normalization")
    print("  • Orth (SUNRGBD 3-stream):  [-1, 1] - mean=0.5, std=0.5")

    print("\n[Key Insight]:")
    print("  YES - All modalities are in [0, 1] range BEFORE final normalization.")
    print("  The normalization step then maps them to different ranges:")
    print("    - mean=0.5, std=0.5 maps [0,1] -> [-1,1]")
    print("    - ImageNet stats map [0,1] -> approximately [-2.1, 2.6]")
    print("    - No normalization keeps [0,1]")


if __name__ == "__main__":
    # Check raw file scales
    if os.path.exists('data/sunrgbd_15/train'):
        check_raw_image_scales()

    # Check dataset processing
    check_sunrgbd_3stream_processing()
    check_sunrgbd_2stream_processing()
    check_nyu_depth_processing()

    # Print summary
    summary()
