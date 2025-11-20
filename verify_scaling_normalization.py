"""
Verify that all three modalities (RGB, Depth, Orth) are correctly scaled and normalized to [-1, 1].
"""

import numpy as np
from PIL import Image
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_utils.sunrgbd_3stream_dataset import SUNRGBD3StreamDataset


def verify_scaling_logic():
    """Verify the mathematical correctness of scaling and normalization."""
    print("=" * 80)
    print("VERIFICATION: Scaling and Normalization Logic")
    print("=" * 80)

    print("\n[Target]: All modalities should be in [-1, 1] range after normalization")
    print()

    # Test RGB scaling
    print("RGB Scaling:")
    print("  1. Raw file: uint8 [0, 255]")
    print("  2. to_tensor() scales: value / 255.0 → [0, 1]")
    print("  3. normalize(mean=0.5, std=0.5): (x - 0.5) / 0.5 = 2x - 1")
    print("     - Input 0.0 → (0.0 - 0.5) / 0.5 = -1.0 ✓")
    print("     - Input 0.5 → (0.5 - 0.5) / 0.5 = 0.0 ✓")
    print("     - Input 1.0 → (1.0 - 0.5) / 0.5 = 1.0 ✓")
    print("  → Result: [-1, 1] ✓✓✓")

    print("\nDepth Scaling:")
    print("  1. Raw file: uint16 [~5000, ~65000]")
    print("  2. Manual scaling: depth_arr / depth_arr.max() → [0, 1]")
    print("  3. Convert to PIL mode F, then to_tensor() → keeps [0, 1]")
    print("  4. normalize(mean=0.5, std=0.5): (x - 0.5) / 0.5 = 2x - 1")
    print("     - Input 0.0 → (0.0 - 0.5) / 0.5 = -1.0 ✓")
    print("     - Input 1.0 → (1.0 - 0.5) / 0.5 = 1.0 ✓")
    print("  → Result: [-1, 1] ✓✓✓")

    print("\nOrthogonal Scaling:")
    print("  1. Raw file: uint16 [~12000, ~50000]")
    print("  2. Convert to PIL mode F → keeps uint16 values [0, 65535]")
    print("  3. to_tensor() does NOT scale mode F → still [0, 65535]")
    print("  4. Manual scaling: orth / 65535.0 → [0, 1]")
    print("  5. normalize(mean=0.5, std=0.5): (x - 0.5) / 0.5 = 2x - 1")
    print("     - Input 0.0 → (0.0 - 0.5) / 0.5 = -1.0 ✓")
    print("     - Input 1.0 → (1.0 - 0.5) / 0.5 = 1.0 ✓")
    print("  → Result: [-1, 1] ✓✓✓")


def test_actual_dataset_outputs(num_samples=50):
    """Test actual dataset outputs to verify ranges."""
    print("\n" + "=" * 80)
    print("EMPIRICAL TEST: Actual Dataset Output Ranges")
    print("=" * 80)

    try:
        # Test both train and val
        for split_name, is_train in [("Train", True), ("Val", False)]:
            print(f"\n[{split_name} Set]")
            dataset = SUNRGBD3StreamDataset(train=is_train)

            rgb_mins, rgb_maxs = [], []
            depth_mins, depth_maxs = [], []
            orth_mins, orth_maxs = [], []

            # Sample multiple images
            n_samples = min(num_samples, len(dataset))
            indices = np.random.choice(len(dataset), n_samples, replace=False)

            for idx in indices:
                rgb, depth, orth, label = dataset[idx]

                rgb_mins.append(rgb.min().item())
                rgb_maxs.append(rgb.max().item())
                depth_mins.append(depth.min().item())
                depth_maxs.append(depth.max().item())
                orth_mins.append(orth.min().item())
                orth_maxs.append(orth.max().item())

            print(f"  Tested {n_samples} samples:")
            print(f"    RGB:   min={min(rgb_mins):.3f}, max={max(rgb_maxs):.3f}, "
                  f"avg_min={np.mean(rgb_mins):.3f}, avg_max={np.mean(rgb_maxs):.3f}")
            print(f"    Depth: min={min(depth_mins):.3f}, max={max(depth_maxs):.3f}, "
                  f"avg_min={np.mean(depth_mins):.3f}, avg_max={np.mean(depth_maxs):.3f}")
            print(f"    Orth:  min={min(orth_mins):.3f}, max={max(orth_maxs):.3f}, "
                  f"avg_min={np.mean(orth_mins):.3f}, avg_max={np.mean(orth_maxs):.3f}")

            # Check if within [-1, 1] range
            issues = []
            if min(rgb_mins) < -1.01 or max(rgb_maxs) > 1.01:
                issues.append("RGB out of range")
            if min(depth_mins) < -1.01 or max(depth_maxs) > 1.01:
                issues.append("Depth out of range")
            if min(orth_mins) < -1.01 or max(orth_maxs) > 1.01:
                issues.append("Orth out of range")

            if issues:
                print(f"  ❌ ISSUES: {', '.join(issues)}")
            else:
                print(f"  ✓ All modalities within [-1, 1] range")

    except Exception as e:
        print(f"  Error: {e}")


def test_step_by_step_single_image():
    """Test step-by-step transformation of a single image to verify each stage."""
    print("\n" + "=" * 80)
    print("STEP-BY-STEP: Single Image Transformation")
    print("=" * 80)

    try:
        dataset = SUNRGBD3StreamDataset(train=False)
        idx = 0

        # Manually load and process to see each step
        rgb_path = os.path.join(dataset.rgb_dir, f'{idx:05d}.png')
        depth_path = os.path.join(dataset.depth_dir, f'{idx:05d}.png')
        orth_path = os.path.join(dataset.orth_dir, f'{idx:05d}.png')

        print("\n[RGB Processing]")
        rgb_raw = Image.open(rgb_path).convert('RGB')
        rgb_raw_arr = np.array(rgb_raw)
        print(f"  Step 1 - Raw PIL RGB: mode={rgb_raw.mode}, range=[{rgb_raw_arr.min()}, {rgb_raw_arr.max()}]")

        # Resize (validation mode)
        from torchvision import transforms
        rgb_resized = transforms.functional.resize(rgb_raw, (224, 224))
        rgb_resized_arr = np.array(rgb_resized)
        print(f"  Step 2 - After resize: range=[{rgb_resized_arr.min()}, {rgb_resized_arr.max()}]")

        rgb_tensor = transforms.functional.to_tensor(rgb_resized)
        print(f"  Step 3 - After to_tensor: range=[{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")

        rgb_normalized = transforms.functional.normalize(rgb_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        print(f"  Step 4 - After normalize: range=[{rgb_normalized.min():.3f}, {rgb_normalized.max():.3f}]")

        print("\n[Depth Processing]")
        depth_raw = Image.open(depth_path)
        depth_raw_arr = np.array(depth_raw)
        print(f"  Step 1 - Raw PIL depth: mode={depth_raw.mode}, range=[{depth_raw_arr.min()}, {depth_raw_arr.max()}]")

        # Manual scaling (as in dataset)
        depth_arr = np.array(depth_raw, dtype=np.float32)
        if depth_arr.max() > 0:
            depth_arr = depth_arr / depth_arr.max()
        depth_scaled = Image.fromarray(depth_arr, mode='F')
        print(f"  Step 2 - After manual scale: range=[{depth_arr.min():.3f}, {depth_arr.max():.3f}]")

        depth_resized = transforms.functional.resize(depth_scaled, (224, 224))
        depth_resized_arr = np.array(depth_resized)
        print(f"  Step 3 - After resize: range=[{depth_resized_arr.min():.3f}, {depth_resized_arr.max():.3f}]")

        depth_tensor = transforms.functional.to_tensor(depth_resized)
        print(f"  Step 4 - After to_tensor: range=[{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")

        depth_normalized = transforms.functional.normalize(depth_tensor, mean=[0.5], std=[0.5])
        print(f"  Step 5 - After normalize: range=[{depth_normalized.min():.3f}, {depth_normalized.max():.3f}]")

        print("\n[Orthogonal Processing]")
        orth_raw = Image.open(orth_path)
        orth_raw_arr = np.array(orth_raw)
        print(f"  Step 1 - Raw PIL orth: mode={orth_raw.mode}, range=[{orth_raw_arr.min()}, {orth_raw_arr.max()}]")

        orth_f = orth_raw.convert('F')
        orth_f_arr = np.array(orth_f)
        print(f"  Step 2 - After convert to F: range=[{orth_f_arr.min():.1f}, {orth_f_arr.max():.1f}]")

        orth_resized = transforms.functional.resize(orth_f, (224, 224))
        orth_resized_arr = np.array(orth_resized)
        print(f"  Step 3 - After resize: range=[{orth_resized_arr.min():.1f}, {orth_resized_arr.max():.1f}]")

        orth_tensor = transforms.functional.to_tensor(orth_resized)
        print(f"  Step 4 - After to_tensor (NOT scaled for mode F): range=[{orth_tensor.min():.1f}, {orth_tensor.max():.1f}]")

        orth_scaled = orth_tensor / 65535.0
        print(f"  Step 5 - After manual scale /65535: range=[{orth_scaled.min():.3f}, {orth_scaled.max():.3f}]")

        orth_normalized = transforms.functional.normalize(orth_scaled, mean=[0.5], std=[0.5])
        print(f"  Step 6 - After normalize: range=[{orth_normalized.min():.3f}, {orth_normalized.max():.3f}]")

        # Compare with dataset output
        print("\n[Verification: Compare with Dataset Output]")
        rgb_ds, depth_ds, orth_ds, label = dataset[idx]
        print(f"  Dataset RGB:   range=[{rgb_ds.min():.3f}, {rgb_ds.max():.3f}]")
        print(f"  Dataset Depth: range=[{depth_ds.min():.3f}, {depth_ds.max():.3f}]")
        print(f"  Dataset Orth:  range=[{orth_ds.min():.3f}, {orth_ds.max():.3f}]")

        # Check if they match (within tolerance)
        rgb_match = torch.allclose(rgb_normalized, rgb_ds, atol=1e-5)
        depth_match = torch.allclose(depth_normalized, depth_ds, atol=1e-5)
        orth_match = torch.allclose(orth_normalized, orth_ds, atol=1e-5)

        print(f"\n  RGB matches: {rgb_match}")
        print(f"  Depth matches: {depth_match}")
        print(f"  Orth matches: {orth_match}")

        if rgb_match and depth_match and orth_match:
            print("\n  ✓✓✓ All transformations verified correct!")
        else:
            print("\n  ❌ Mismatch detected!")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def final_summary():
    """Print final summary."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\n✓ Expected behavior for all modalities:")
    print("  1. Raw files are NOT in [0,1] - they need scaling")
    print("  2. Dataset performs scaling to [0,1] for each modality")
    print("  3. Normalization with mean=0.5, std=0.5 maps [0,1] → [-1,1]")
    print("  4. Final output range: [-1, 1] for RGB, Depth, and Orth")
    print("\n✓ Scaling methods:")
    print("  - RGB: to_tensor() auto-scales uint8 by /255")
    print("  - Depth: Manual per-image scaling depth/depth.max()")
    print("  - Orth: Manual scaling orth/65535.0 after to_tensor()")
    print("\n✓ All three modalities use same normalization:")
    print("  - normalize(mean=0.5, std=0.5) for consistent [-1,1] range")


if __name__ == "__main__":
    verify_scaling_logic()
    test_step_by_step_single_image()
    test_actual_dataset_outputs(num_samples=50)
    final_summary()
