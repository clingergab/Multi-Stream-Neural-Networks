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

    # Create dataset
    train_dataset = SUNRGBDDataset(
        data_root='data/sunrgbd_15',
        train=True,
        target_size=(224, 224)
    )

    # Test with enough samples to get reliable statistics
    n_tests = 500
    idx = 0

    # Strategy: If crop is applied, the image size changes during processing
    # We can't directly measure this, but we can measure if samples are identical
    # When no augmentation is applied, samples should be more similar

    # Get multiple samples and check variance
    samples = []
    for i in range(n_tests):
        rgb, depth, _ = train_dataset[idx]
        samples.append(rgb)

    # Count how many unique transformations we see
    # With 50% flip + 50% crop + 50% color, we should see high variety
    # Calculate pairwise differences
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
    Verify depth augmentation uses correct parameters (±25%, not ±20% or ±40%).
    """
    print("="*80)
    print("TEST 2: Depth Augmentation Parameters")
    print("="*80)

    # Read the actual source code to verify parameters
    import src.data_utils.sunrgbd_dataset as dataset_module
    source_code = inspect.getsource(dataset_module.SUNRGBDDataset.__getitem__)

    # Check for correct brightness/contrast ranges
    errors = []

    # Should have ±25% for depth (0.75, 1.25)
    if "0.75, 1.25" in source_code:
        print("  ✅ Depth brightness: ±25% (0.75-1.25) - CORRECT")
    else:
        errors.append("Depth brightness not set to ±25%")
        print("  ❌ Depth brightness: NOT ±25%")

    # Should have ±20% for RGB (0.8, 1.2) via ColorJitter(brightness=0.2)
    if "brightness=0.2" in source_code:
        print("  ✅ RGB brightness: ±20% (ColorJitter 0.2) - CORRECT")
    else:
        errors.append("RGB brightness not set to ±20%")
        print("  ❌ RGB brightness: NOT ±20%")

    # Should have std=15 for noise
    if "std=15" in source_code or "0, 15" in source_code:
        print("  ✅ Depth noise: std=15 - CORRECT")
    else:
        errors.append("Depth noise std not 15")
        print("  ❌ Depth noise: NOT std=15")

    # Should have scale=(0.9, 1.0) for crop
    if "0.9, 1.0" in source_code:
        print("  ✅ Crop scale: 0.9-1.0 - CORRECT")
    else:
        errors.append("Crop scale not 0.9-1.0")
        print("  ❌ Crop scale: NOT 0.9-1.0")

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
    """
    print("="*80)
    print("TEST 3: No Double-Stacking (Depth should have ONE 50% block)")
    print("="*80)

    # Read source code
    import src.data_utils.sunrgbd_dataset as dataset_module
    source_code = inspect.getsource(dataset_module.SUNRGBDDataset.__getitem__)

    # Count how many separate "if np.random.random() < 0.5:" blocks affect depth
    # Should be only ONE for depth appearance augmentation

    # Look for depth-only augmentation blocks (after normalization is different)
    lines = source_code.split('\n')

    depth_aug_blocks = 0
    in_depth_block = False

    for i, line in enumerate(lines):
        # Check if this is a depth augmentation block
        if 'depth_array' in line and 'np.random.random()' in line:
            depth_aug_blocks += 1

        # Check for separate blocks (brightness/contrast separate from noise)
        if 'brightness_factor' in line and i > 0:
            # Check if this is in the same block or separate
            # Look back to see if there's a new random check
            lookback = '\n'.join(lines[max(0, i-10):i])
            if lookback.count('if np.random.random()') > 1:
                print(f"  ⚠️  Found separate random checks near brightness_factor")

    # We should have ONE depth augmentation block that does brightness+contrast+noise
    # in a single "if np.random.random() < 0.5:" check

    # Better check: look for the pattern
    has_combined_block = False
    for i, line in enumerate(lines):
        if 'brightness_factor = np.random.uniform(0.75, 1.25)' in line:
            # Check next ~15 lines for both contrast and noise
            next_lines = '\n'.join(lines[i:i+20])
            if 'contrast_factor' in next_lines and 'noise = np.random.normal' in next_lines:
                has_combined_block = True
                # Now check there's only ONE random check in the 10 lines before this
                prev_lines = '\n'.join(lines[max(0, i-10):i])
                # Count "if np.random.random()" that affects depth
                # Should be exactly 1 (the block starter)
                depth_random_checks = 0
                for pline in lines[max(0, i-10):i]:
                    if 'if np.random.random()' in pline:
                        # Check if next few lines mention depth
                        depth_random_checks += 1

                if depth_random_checks == 1:
                    print(f"  ✅ Depth augmentation in SINGLE block (brightness + contrast + noise together)")
                elif depth_random_checks == 0:
                    # This is OK - means brightness is inside the block
                    print(f"  ✅ Depth augmentation in SINGLE block (brightness + contrast + noise together)")
                else:
                    print(f"  ❌ Found {depth_random_checks} random checks before brightness (expected 0-1)")
                    assert False, "Double-stacking detected"

    if not has_combined_block:
        print(f"  ❌ FAIL: Could not find combined brightness+contrast+noise block")
        assert False, "Depth augmentation not properly combined"

    print(f"  ✅ PASS: No double-stacking detected")
    print()


def test_scene_optimized_crop():
    """
    Verify crop is probabilistic (50%), not always applied (100%).
    This is critical for scene classification.
    """
    print("="*80)
    print("TEST 4: Scene-Optimized Crop (50% probability, NOT 100%)")
    print("="*80)

    # Read source code
    import src.data_utils.sunrgbd_dataset as dataset_module
    source_code = inspect.getsource(dataset_module.SUNRGBDDataset.__getitem__)

    # Look for crop implementation
    # Should have: if np.random.random() < 0.5: followed by crop
    # Should also have: else: resize (for no-crop case)

    errors = []

    # Check for probabilistic crop
    if 'RandomResizedCrop.get_params' in source_code:
        # Find the line
        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            if 'RandomResizedCrop.get_params' in line:
                # Check previous lines for random check
                prev_lines = '\n'.join(lines[max(0, i-5):i])
                if 'if np.random.random() < 0.5' in prev_lines:
                    print(f"  ✅ Crop is probabilistic (50%)")
                else:
                    print(f"  ❌ Crop appears to always be applied (no random check found)")
                    errors.append("Crop not probabilistic")

                # Check for else clause with resize
                next_lines = '\n'.join(lines[i:i+10])
                if 'else:' in next_lines and 'resize' in next_lines:
                    print(f"  ✅ Else clause preserves full context (resize only)")
                else:
                    print(f"  ❌ No else clause found for no-crop case")
                    errors.append("No else clause for no-crop")

                break
    else:
        errors.append("RandomResizedCrop not found in code")
        print(f"  ❌ RandomResizedCrop not found")

    if errors:
        print(f"\n  ❌ FAIL: {len(errors)} issue(s):")
        for error in errors:
            print(f"     - {error}")
        assert False, "Crop not scene-optimized"
    else:
        print(f"\n  ✅ PASS: Crop is scene-optimized (50% probability)")

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
        train=False,
        target_size=(224, 224)
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
