"""
Detailed inspection of raw data scaling for Depth and Orth streams.
This will help us understand if the scaling is correct and why we're seeing large ranges.
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def inspect_raw_files(data_root='data/sunrgbd_15', split='train', num_samples=50):
    """Inspect raw depth and orth files to understand their actual ranges."""
    print("=" * 80)
    print(f"INSPECTING RAW FILES: {data_root}/{split}")
    print("=" * 80)

    split_dir = os.path.join(data_root, split)
    depth_dir = os.path.join(split_dir, 'depth')
    orth_dir = os.path.join(split_dir, 'orth')

    depth_raw_mins, depth_raw_maxs = [], []
    orth_raw_mins, orth_raw_maxs = [], []
    depth_modes = set()
    orth_modes = set()

    print(f"\nChecking {num_samples} samples...")

    for idx in tqdm(range(num_samples), desc="Inspecting files"):
        # Depth
        depth_path = os.path.join(depth_dir, f'{idx:05d}.png')
        if os.path.exists(depth_path):
            depth_img = Image.open(depth_path)
            depth_modes.add(depth_img.mode)
            depth_arr = np.array(depth_img)
            depth_raw_mins.append(depth_arr.min())
            depth_raw_maxs.append(depth_arr.max())

        # Orth
        orth_path = os.path.join(orth_dir, f'{idx:05d}.png')
        if os.path.exists(orth_path):
            orth_img = Image.open(orth_path)
            orth_modes.add(orth_img.mode)
            orth_arr = np.array(orth_img)
            orth_raw_mins.append(orth_arr.min())
            orth_raw_maxs.append(orth_arr.max())

    print("\n" + "-" * 80)
    print("RAW DEPTH FILES")
    print("-" * 80)
    print(f"Image modes found: {depth_modes}")
    print(f"Raw value range: [{min(depth_raw_mins)}, {max(depth_raw_maxs)}]")
    print(f"Across {len(depth_raw_mins)} samples:")
    print(f"  Min values range: [{min(depth_raw_mins)}, {max(depth_raw_mins)}]")
    print(f"  Max values range: [{min(depth_raw_maxs)}, {max(depth_raw_maxs)}]")

    print("\n" + "-" * 80)
    print("RAW ORTH FILES")
    print("-" * 80)
    print(f"Image modes found: {orth_modes}")
    print(f"Raw value range: [{min(orth_raw_mins)}, {max(orth_raw_maxs)}]")
    print(f"Across {len(orth_raw_mins)} samples:")
    print(f"  Min values range: [{min(orth_raw_mins)}, {max(orth_raw_mins)}]")
    print(f"  Max values range: [{min(orth_raw_maxs)}, {max(orth_raw_maxs)}]")

    return {
        'depth_modes': depth_modes,
        'depth_raw_min': min(depth_raw_mins),
        'depth_raw_max': max(depth_raw_maxs),
        'orth_modes': orth_modes,
        'orth_raw_min': min(orth_raw_mins),
        'orth_raw_max': max(orth_raw_maxs)
    }


def test_scaling_logic(raw_info):
    """Test the scaling logic we use in the dataset."""
    print("\n" + "=" * 80)
    print("TESTING SCALING LOGIC")
    print("=" * 80)

    # Depth scaling
    print("\n[DEPTH SCALING]")
    print("Current logic in dataset:")
    print("  1. Load as PIL Image (mode I;16 or I)")
    print("  2. Convert to float32 numpy array")
    print("  3. Divide by 65535.0 (16-bit max)")
    print("  4. Clip to [0.0, 1.0]")

    print(f"\nWith raw range [{raw_info['depth_raw_min']}, {raw_info['depth_raw_max']}]:")
    depth_scaled_min = np.clip(raw_info['depth_raw_min'] / 65535.0, 0.0, 1.0)
    depth_scaled_max = np.clip(raw_info['depth_raw_max'] / 65535.0, 0.0, 1.0)
    print(f"  After scaling: [{depth_scaled_min:.6f}, {depth_scaled_max:.6f}]")

    if raw_info['depth_raw_max'] == 65535:
        print("  ✓ Using full 16-bit range, scaling is appropriate")
    else:
        print(f"  ⚠️  Max value {raw_info['depth_raw_max']} < 65535")
        print(f"     Consider scaling by {raw_info['depth_raw_max']} instead of 65535")

    # Orth scaling
    print("\n[ORTH SCALING]")
    print("Current logic in dataset:")
    print("  1. Load as PIL Image (mode I;16)")
    print("  2. Convert to float32 numpy array")
    print("  3. Divide by 65535.0 (16-bit max)")
    print("  4. Clip to [0.0, 1.0]")

    print(f"\nWith raw range [{raw_info['orth_raw_min']}, {raw_info['orth_raw_max']}]:")
    orth_scaled_min = np.clip(raw_info['orth_raw_min'] / 65535.0, 0.0, 1.0)
    orth_scaled_max = np.clip(raw_info['orth_raw_max'] / 65535.0, 0.0, 1.0)
    print(f"  After scaling: [{orth_scaled_min:.6f}, {orth_scaled_max:.6f}]")

    if raw_info['orth_raw_max'] == 65535:
        print("  ✓ Using full 16-bit range, scaling is appropriate")
    else:
        print(f"  ⚠️  Max value {raw_info['orth_raw_max']} < 65535")
        print(f"     Consider scaling by {raw_info['orth_raw_max']} instead of 65535")


def compare_with_actual_dataset():
    """Load data through the actual dataset and compare."""
    print("\n" + "=" * 80)
    print("COMPARING WITH ACTUAL DATASET OUTPUT")
    print("=" * 80)

    from data_utils.sunrgbd_3stream_dataset import SUNRGBD3StreamDataset

    # Create dataset WITHOUT normalization to see scaled [0,1] values
    # We'll need to manually check pre-normalization values

    print("\nLoading samples through dataset (after scaling, before normalization)...")
    print("Note: Dataset applies normalization, so we need to denormalize to check")

    dataset = SUNRGBD3StreamDataset(train=False, target_size=(416, 544))

    # Statistics used in dataset
    DEPTH_MEAN = 0.2912
    DEPTH_STD = 0.1472
    ORTH_MEAN = 0.5000
    ORTH_STD = 0.2794

    depth_denorm_mins, depth_denorm_maxs = [], []
    orth_denorm_mins, orth_denorm_maxs = [], []

    num_samples = min(100, len(dataset))

    for idx in tqdm(range(num_samples), desc="Checking dataset output"):
        _, depth_norm, orth_norm, _ = dataset[idx]

        # Denormalize: x_original = x_normalized * std + mean
        depth_denorm = depth_norm * DEPTH_STD + DEPTH_MEAN
        orth_denorm = orth_norm * ORTH_STD + ORTH_MEAN

        depth_denorm_mins.append(depth_denorm.min().item())
        depth_denorm_maxs.append(depth_denorm.max().item())
        orth_denorm_mins.append(orth_denorm.min().item())
        orth_denorm_maxs.append(orth_denorm.max().item())

    print("\n[DEPTH - Denormalized (should be [0,1] if scaling is correct)]")
    print(f"  Range: [{min(depth_denorm_mins):.6f}, {max(depth_denorm_maxs):.6f}]")
    if min(depth_denorm_mins) >= 0.0 and max(depth_denorm_maxs) <= 1.0:
        print("  ✓ PASS: Values are in [0, 1]")
    else:
        print("  ✗ FAIL: Values outside [0, 1]!")
        if min(depth_denorm_mins) < 0.0:
            print(f"    Minimum {min(depth_denorm_mins):.6f} < 0")
        if max(depth_denorm_maxs) > 1.0:
            print(f"    Maximum {max(depth_denorm_maxs):.6f} > 1")

    print("\n[ORTH - Denormalized (should be [0,1] if scaling is correct)]")
    print(f"  Range: [{min(orth_denorm_mins):.6f}, {max(orth_denorm_maxs):.6f}]")
    if min(orth_denorm_mins) >= 0.0 and max(orth_denorm_maxs) <= 1.0:
        print("  ✓ PASS: Values are in [0, 1]")
    else:
        print("  ✗ FAIL: Values outside [0, 1]!")
        if min(orth_denorm_mins) < 0.0:
            print(f"    Minimum {min(orth_denorm_mins):.6f} < 0")
        if max(orth_denorm_maxs) > 1.0:
            print(f"    Maximum {max(orth_denorm_maxs):.6f} > 1")


def manual_scaling_test():
    """Manually load and scale a few samples to verify the process."""
    print("\n" + "=" * 80)
    print("MANUAL SCALING TEST")
    print("=" * 80)

    data_root = 'data/sunrgbd_15'
    split = 'train'

    split_dir = os.path.join(data_root, split)
    depth_dir = os.path.join(split_dir, 'depth')
    orth_dir = os.path.join(split_dir, 'orth')

    print("\nManually loading and scaling sample 0...")

    # Depth
    depth_path = os.path.join(depth_dir, '00000.png')
    depth = Image.open(depth_path)
    print(f"\n[DEPTH Sample 0]")
    print(f"  PIL mode: {depth.mode}")

    depth_raw = np.array(depth, dtype=np.float32)
    print(f"  Raw array: dtype={depth_raw.dtype}, range=[{depth_raw.min()}, {depth_raw.max()}]")

    depth_scaled = np.clip(depth_raw / 65535.0, 0.0, 1.0)
    print(f"  After /65535: range=[{depth_scaled.min():.6f}, {depth_scaled.max():.6f}]")

    # Orth
    orth_path = os.path.join(orth_dir, '00000.png')
    orth = Image.open(orth_path)
    print(f"\n[ORTH Sample 0]")
    print(f"  PIL mode: {orth.mode}")

    orth_raw = np.array(orth, dtype=np.float32)
    print(f"  Raw array: dtype={orth_raw.dtype}, range=[{orth_raw.min()}, {orth_raw.max()}]")

    orth_scaled = np.clip(orth_raw / 65535.0, 0.0, 1.0)
    print(f"  After /65535: range=[{orth_scaled.min():.6f}, {orth_scaled.max():.6f}]")

    # Check what happens if we use the actual max instead
    if orth_raw.max() < 65535:
        orth_scaled_adaptive = orth_raw / orth_raw.max()
        print(f"  If scaled by actual max ({orth_raw.max()}): range=[{orth_scaled_adaptive.min():.6f}, {orth_scaled_adaptive.max():.6f}]")


if __name__ == "__main__":
    # Step 1: Inspect raw files
    raw_info = inspect_raw_files(num_samples=100)

    # Step 2: Test scaling logic
    test_scaling_logic(raw_info)

    # Step 3: Manual test
    manual_scaling_test()

    # Step 4: Compare with dataset output
    compare_with_actual_dataset()

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
