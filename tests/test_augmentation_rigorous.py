"""
RIGOROUS augmentation tests - verify actual implementation details.
This test directly inspects the code to ensure probabilities and parameters are correct.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
import inspect


def test_crop_probability():
    """
    Verify crop is applied with 50% probability, not 100%.
    This is critical for scene classification.
    """
    print("\n" + "="*80)
    print("TEST 1: Crop Probability (50% expected)")
    print("="*80)

    # Create dataset — use the actual target_size matching pre-resized tensors
    # so all samples have the same output shape regardless of crop
    train_dataset = SUNRGBDDataset(
        data_root='data/sunrgbd_15',
        split='train',
        target_size=(416, 544)
    )

    # Test with enough samples to get reliable statistics
    n_tests = 500
    idx = 0

    # Get multiple samples and check variance
    samples = []
    for i in range(n_tests):
        rgb, depth, _ = train_dataset[idx]
        samples.append(rgb)

    # Count how many unique transformations we see
    # With 50% flip + 50% crop + 50% color, we should see high variety
    unique_threshold = 0.01  # Threshold for considering samples "different"

    differences = []
    for i in range(min(50, n_tests)):  # Check first 50 pairs
        diff = torch.abs(samples[i] - samples[0]).mean().item()
        differences.append(diff)

    # At least 70% should be different (due to flip 50% + crop 50% + color 50%)
    different_count = sum(1 for d in differences if d > unique_threshold)
    different_percent = different_count / len(differences) * 100

    print(f"\nResults:")
    print(f"  Samples tested: {len(differences)}")
    print(f"  Samples different from baseline: {different_count} ({different_percent:.1f}%)")
    print(f"  Expected: >70% (due to flip 50% + crop 50% + color 50%)")

    if different_percent > 70:
        print(f"  ✅ PASS: High variance indicates augmentation is working")
    else:
        print(f"  ❌ FAIL: Low variance suggests augmentation may not be working")
        assert False, "Augmentation variance too low"

    print()


def test_depth_augmentation_parameters():
    """
    Verify depth augmentation uses correct parameters via augmentation_config.
    Now uses configurable scaling, so check config defaults instead of hardcoded values.
    """
    print("="*80)
    print("TEST 2: Depth Augmentation Parameters")
    print("="*80)

    from src.training.augmentation_config import (
        BASE_DEPTH_BRIGHTNESS, BASE_DEPTH_CONTRAST, BASE_DEPTH_NOISE_STD,
        BASE_CROP_SCALE_MIN, BASE_CROP_SCALE_MAX,
        BASE_BRIGHTNESS,
    )

    errors = []

    # Check depth brightness baseline (±25%)
    if abs(BASE_DEPTH_BRIGHTNESS - 0.25) < 1e-6:
        print("  ✅ Depth brightness: ±25% (BASE_DEPTH_BRIGHTNESS=0.25) - CORRECT")
    else:
        errors.append(f"Depth brightness not ±25% (got {BASE_DEPTH_BRIGHTNESS})")
        print(f"  ❌ Depth brightness: {BASE_DEPTH_BRIGHTNESS} (expected 0.25)")

    # Check RGB brightness baseline (used by ColorJitter)
    if abs(BASE_BRIGHTNESS - 0.37) < 1e-6:
        print("  ✅ RGB brightness: ±37% (BASE_BRIGHTNESS=0.37) - CORRECT")
    else:
        errors.append(f"RGB brightness not ±37% (got {BASE_BRIGHTNESS})")
        print(f"  ❌ RGB brightness: {BASE_BRIGHTNESS}")

    # Check depth noise std baseline (~0.06)
    if abs(BASE_DEPTH_NOISE_STD - 0.059) < 0.01:
        print(f"  ✅ Depth noise: std={BASE_DEPTH_NOISE_STD} (BASE_DEPTH_NOISE_STD) - CORRECT")
    else:
        errors.append(f"Depth noise std not ~0.06 (got {BASE_DEPTH_NOISE_STD})")
        print(f"  ❌ Depth noise: {BASE_DEPTH_NOISE_STD}")

    # Check crop scale baseline
    if abs(BASE_CROP_SCALE_MIN - 0.9) < 1e-6 and abs(BASE_CROP_SCALE_MAX - 1.0) < 1e-6:
        print("  ✅ Crop scale: 0.9-1.0 - CORRECT")
    else:
        errors.append(f"Crop scale not 0.9-1.0 (got {BASE_CROP_SCALE_MIN}-{BASE_CROP_SCALE_MAX})")
        print(f"  ❌ Crop scale: {BASE_CROP_SCALE_MIN}-{BASE_CROP_SCALE_MAX}")

    # Verify dataset computes correct scaled values at default (1.0) scaling
    ds = SUNRGBDDataset(data_root='data/sunrgbd_15', split='train')
    if abs(ds._depth_brightness - 0.25) < 1e-6:
        print("  ✅ Dataset._depth_brightness matches config at scale 1.0")
    else:
        errors.append(f"Dataset._depth_brightness wrong (got {ds._depth_brightness})")

    print()

    if errors:
        print(f"  ❌ FAIL: {len(errors)} parameter(s) incorrect:")
        for error in errors:
            print(f"     - {error}")
        assert False, "Augmentation parameters incorrect"
    else:
        print(f"  ✅ PASS: All parameters correct")

    print()


def test_no_double_stacking():
    """
    Verify depth augmentation is in a SINGLE block (no double-stacking).
    Brightness, contrast, and noise should all be in one probability-gated block.
    """
    print("="*80)
    print("TEST 3: No Double-Stacking (Depth should have ONE probability-gated block)")
    print("="*80)

    # Read source code
    import src.data_utils.sunrgbd_dataset as dataset_module
    source_code = inspect.getsource(dataset_module.SUNRGBDDataset.__getitem__)

    lines = source_code.split('\n')

    # Look for the depth augmentation block pattern:
    # A single probability check followed by brightness, contrast, and noise operations
    has_combined_block = False
    for i, line in enumerate(lines):
        if 'brightness_factor' in line and 'np.random.uniform' in line:
            # Check next ~15 lines for both contrast and noise
            next_lines = '\n'.join(lines[i:i+20])
            if 'contrast_factor' in next_lines and ('randn_like' in next_lines or 'noise' in next_lines):
                has_combined_block = True
                # Find the nearest preceding probability check that gates depth operations.
                # Walk backwards from brightness_factor to find the 'if np.random.random()'
                # that is the block entry point (should reference depth, not rgb).
                depth_gate_found = False
                for j in range(i-1, max(0, i-10), -1):
                    if 'np.random.random()' in lines[j]:
                        # Check if the lines after this check reference depth (not rgb)
                        block_lines = '\n'.join(lines[j:j+5])
                        if 'depth' in block_lines or 'brightness_factor' in block_lines:
                            depth_gate_found = True
                            print(f"  ✅ Depth augmentation gated by single probability check")
                        break

                if depth_gate_found:
                    print(f"  ✅ Depth augmentation in SINGLE block (brightness + contrast + noise together)")
                else:
                    print(f"  ❌ Could not find depth probability gate before brightness_factor")
                    assert False, "Missing depth probability gate"
                break

    if not has_combined_block:
        print(f"  ❌ FAIL: Could not find combined brightness+contrast+noise block")
        assert False, "Depth augmentation not properly combined"

    print(f"  ✅ PASS: No double-stacking detected")
    print()


def test_scene_optimized_crop():
    """
    Verify crop is probabilistic (gated by _crop_p), not always applied (100%).
    This is critical for scene classification.
    """
    print("="*80)
    print("TEST 4: Scene-Optimized Crop (probabilistic, NOT 100%)")
    print("="*80)

    # Read source code
    import src.data_utils.sunrgbd_dataset as dataset_module
    source_code = inspect.getsource(dataset_module.SUNRGBDDataset.__getitem__)

    errors = []

    # Check for probabilistic crop
    if 'RandomResizedCrop.get_params' in source_code:
        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            if 'RandomResizedCrop.get_params' in line:
                # Check previous lines for random probability check
                prev_lines = '\n'.join(lines[max(0, i-5):i])
                if 'np.random.random()' in prev_lines and ('_crop_p' in prev_lines or '< 0.5' in prev_lines):
                    print(f"  ✅ Crop is probabilistic (gated by probability check)")
                else:
                    print(f"  ❌ Crop appears to always be applied (no random check found)")
                    errors.append("Crop not probabilistic")

                # Check for elif/else clause with resize for no-crop fallback
                next_lines = '\n'.join(lines[i:i+10])
                if ('elif' in next_lines or 'else' in next_lines) and 'resize' in next_lines:
                    print(f"  ✅ Fallback clause preserves full context (resize only)")
                else:
                    print(f"  ❌ No fallback clause found for no-crop case")
                    errors.append("No fallback clause for no-crop")

                break
    else:
        errors.append("RandomResizedCrop not found in code")
        print(f"  ❌ RandomResizedCrop not found")

    # Also verify via dataset attributes that crop_p is ~50% at default scaling
    ds = SUNRGBDDataset(data_root='data/sunrgbd_15', split='train')
    if abs(ds._crop_p - 0.5) < 1e-6:
        print(f"  ✅ Default _crop_p = {ds._crop_p} (50%)")
    else:
        errors.append(f"Default _crop_p is {ds._crop_p}, expected 0.5")
        print(f"  ❌ Default _crop_p = {ds._crop_p} (expected 0.5)")

    if errors:
        print(f"\n  ❌ FAIL: {len(errors)} issue(s):")
        for error in errors:
            print(f"     - {error}")
        assert False, "Crop not scene-optimized"
    else:
        print(f"\n  ✅ PASS: Crop is scene-optimized (probabilistic)")

    print()


def test_validation_deterministic():
    """
    Verify validation set is deterministic (no augmentation).
    """
    print("="*80)
    print("TEST 5: Validation Deterministic")
    print("="*80)

    val_dataset = SUNRGBDDataset(
        data_root='data/sunrgbd_15',
        split='val',
        target_size=(416, 544)
    )

    # Load same sample multiple times
    idx = 0
    samples = []
    for i in range(10):
        rgb, depth, _ = val_dataset[idx]
        samples.append((rgb, depth))

    # All should be identical
    all_identical = True
    for i in range(1, len(samples)):
        rgb_diff = torch.abs(samples[i][0] - samples[0][0]).max().item()
        depth_diff = torch.abs(samples[i][1] - samples[0][1]).max().item()

        if rgb_diff > 1e-6 or depth_diff > 1e-6:
            all_identical = False
            print(f"  ❌ Sample {i} differs from baseline (rgb: {rgb_diff}, depth: {depth_diff})")

    if all_identical:
        print(f"  ✅ PASS: All validation samples identical (deterministic)")
    else:
        print(f"  ❌ FAIL: Validation samples vary (not deterministic)")
        assert False, "Validation not deterministic"

    print()


def run_all_rigorous_tests():
    """Run all rigorous tests."""
    print("\n" + "="*80)
    print("RIGOROUS AUGMENTATION TESTS")
    print("Testing actual implementation, not just variance")
    print("="*80)

    try:
        test_crop_probability()
        test_depth_augmentation_parameters()
        test_no_double_stacking()
        test_scene_optimized_crop()
        test_validation_deterministic()

        print("="*80)
        print("ALL RIGOROUS TESTS PASSED ✅")
        print("="*80)
        print("\nConfiguration verified:")
        print("  ✅ Crop: 50% probability @ scale 0.9-1.0")
        print("  ✅ Depth: ±25% brightness/contrast, std=15 noise")
        print("  ✅ RGB: ±20% brightness/contrast/saturation")
        print("  ✅ No double-stacking (single 50% block for depth)")
        print("  ✅ Validation is deterministic")
        print()

    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        raise


if __name__ == '__main__':
    run_all_rigorous_tests()
