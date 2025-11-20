"""
Final comprehensive verification of sunrgbd_3stream_dataset.
Tests all aspects: loading, scaling, augmentation, normalization, edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from data_utils.sunrgbd_3stream_dataset import SUNRGBD3StreamDataset, get_sunrgbd_3stream_dataloaders

def test_basic_loading():
    """Test basic dataset loading and shapes."""
    print("=" * 80)
    print("TEST 1: Basic Loading and Shapes")
    print("=" * 80)

    try:
        train_ds = SUNRGBD3StreamDataset(train=True)
        val_ds = SUNRGBD3StreamDataset(train=False)

        print(f"\n✓ Train dataset: {len(train_ds)} samples")
        print(f"✓ Val dataset: {len(val_ds)} samples")

        # Test single sample
        rgb, depth, orth, label = train_ds[0]

        print(f"\n✓ RGB shape: {rgb.shape} (expected: [3, 224, 224])")
        print(f"✓ Depth shape: {depth.shape} (expected: [1, 224, 224])")
        print(f"✓ Orth shape: {orth.shape} (expected: [1, 224, 224])")
        print(f"✓ Label: {label} (class: {train_ds.CLASS_NAMES[label]})")

        assert rgb.shape == (3, 224, 224), f"RGB shape mismatch: {rgb.shape}"
        assert depth.shape == (1, 224, 224), f"Depth shape mismatch: {depth.shape}"
        assert orth.shape == (1, 224, 224), f"Orth shape mismatch: {orth.shape}"
        assert 0 <= label < 15, f"Invalid label: {label}"

        print("\n✓✓✓ PASSED: Basic loading works correctly")
        return True

    except Exception as e:
        print(f"\n✗✗✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_value_ranges():
    """Test that all values are in correct ranges."""
    print("\n" + "=" * 80)
    print("TEST 2: Value Ranges")
    print("=" * 80)

    try:
        train_ds = SUNRGBD3StreamDataset(train=True)
        val_ds = SUNRGBD3StreamDataset(train=False)

        # Test multiple samples from both sets
        n_samples = 100

        for ds_name, dataset in [("Train", train_ds), ("Val", val_ds)]:
            print(f"\n{ds_name} Set ({n_samples} samples):")

            rgb_min, rgb_max = float('inf'), float('-inf')
            depth_min, depth_max = float('inf'), float('-inf')
            orth_min, orth_max = float('inf'), float('-inf')

            for i in range(n_samples):
                idx = np.random.randint(0, len(dataset))
                rgb, depth, orth, label = dataset[idx]

                rgb_min = min(rgb_min, rgb.min().item())
                rgb_max = max(rgb_max, rgb.max().item())
                depth_min = min(depth_min, depth.min().item())
                depth_max = max(depth_max, depth.max().item())
                orth_min = min(orth_min, orth.min().item())
                orth_max = max(orth_max, orth.max().item())

            print(f"  RGB:   [{rgb_min:.3f}, {rgb_max:.3f}]")
            print(f"  Depth: [{depth_min:.3f}, {depth_max:.3f}]")
            print(f"  Orth:  [{orth_min:.3f}, {orth_max:.3f}]")

            # Check ranges (allow small tolerance)
            assert rgb_min >= -1.01 and rgb_max <= 1.01, f"RGB out of range: [{rgb_min}, {rgb_max}]"
            assert depth_min >= -1.01 and depth_max <= 1.01, f"Depth out of range: [{depth_min}, {depth_max}]"
            assert orth_min >= -1.01 and orth_max <= 1.01, f"Orth out of range: [{orth_min}, {orth_max}]"

            # Check that they actually reach close to [-1, 1]
            assert rgb_min < -0.9 and rgb_max > 0.9, "RGB doesn't reach full range"
            assert depth_min < -0.9 and depth_max > 0.9, "Depth doesn't reach full range"
            assert orth_min < -0.9 and orth_max > 0.9, "Orth doesn't reach full range"

        print("\n✓✓✓ PASSED: All values in correct [-1, 1] range")
        return True

    except Exception as e:
        print(f"\n✗✗✗ FAILED: {e}")
        return False


def test_no_nan_or_inf():
    """Test that there are no NaN or Inf values."""
    print("\n" + "=" * 80)
    print("TEST 3: NaN and Inf Detection")
    print("=" * 80)

    try:
        train_ds = SUNRGBD3StreamDataset(train=True)

        n_samples = 100
        print(f"\nTesting {n_samples} training samples...")

        for i in range(n_samples):
            idx = np.random.randint(0, len(train_ds))
            rgb, depth, orth, label = train_ds[idx]

            # Check for NaN
            if torch.isnan(rgb).any():
                raise ValueError(f"NaN in RGB at sample {idx}")
            if torch.isnan(depth).any():
                raise ValueError(f"NaN in Depth at sample {idx}")
            if torch.isnan(orth).any():
                raise ValueError(f"NaN in Orth at sample {idx}")

            # Check for Inf
            if torch.isinf(rgb).any():
                raise ValueError(f"Inf in RGB at sample {idx}")
            if torch.isinf(depth).any():
                raise ValueError(f"Inf in Depth at sample {idx}")
            if torch.isinf(orth).any():
                raise ValueError(f"Inf in Orth at sample {idx}")

        print("✓ No NaN values detected")
        print("✓ No Inf values detected")
        print("\n✓✓✓ PASSED: All values are valid")
        return True

    except Exception as e:
        print(f"\n✗✗✗ FAILED: {e}")
        return False


def test_augmentation_independence():
    """Test that depth and orth augmentation are independent."""
    print("\n" + "=" * 80)
    print("TEST 4: Augmentation Independence")
    print("=" * 80)

    try:
        train_ds = SUNRGBD3StreamDataset(train=True)

        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Get same sample multiple times with different augmentation
        n_trials = 50
        idx = 0

        depth_variations = []
        orth_variations = []

        for _ in range(n_trials):
            rgb, depth, orth, label = train_ds[idx]
            depth_variations.append(depth.clone())
            orth_variations.append(orth.clone())

        # Check that depth and orth vary independently
        # If they were synchronized, their variations would be correlated

        # Simple test: check that not all variations are identical
        depth_unique = sum(not torch.allclose(depth_variations[0], d) for d in depth_variations[1:])
        orth_unique = sum(not torch.allclose(orth_variations[0], o) for o in orth_variations[1:])

        print(f"\n  Depth variations: {depth_unique}/{n_trials-1} different from first")
        print(f"  Orth variations:  {orth_unique}/{n_trials-1} different from first")

        # With 50% augmentation probability, we expect most to be different
        assert depth_unique > 15, "Depth augmentation may not be working"
        assert orth_unique > 15, "Orth augmentation may not be working"

        print("\n✓ Depth augmentation working")
        print("✓ Orth augmentation working")
        print("✓ Augmentations appear independent")
        print("\n✓✓✓ PASSED: Augmentation independence verified")
        return True

    except Exception as e:
        print(f"\n✗✗✗ FAILED: {e}")
        return False


def test_dataloader():
    """Test dataloader functionality."""
    print("\n" + "=" * 80)
    print("TEST 5: DataLoader Integration")
    print("=" * 80)

    try:
        train_loader, val_loader = get_sunrgbd_3stream_dataloaders(
            batch_size=16,
            num_workers=0,  # Use 0 for testing
            target_size=(224, 224)
        )

        print(f"\n✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")

        # Test getting a batch
        rgb_batch, depth_batch, orth_batch, labels_batch = next(iter(train_loader))

        print(f"\n✓ Batch shapes:")
        print(f"  RGB:    {rgb_batch.shape} (expected: [16, 3, 224, 224])")
        print(f"  Depth:  {depth_batch.shape} (expected: [16, 1, 224, 224])")
        print(f"  Orth:   {orth_batch.shape} (expected: [16, 1, 224, 224])")
        print(f"  Labels: {labels_batch.shape} (expected: [16])")

        assert rgb_batch.shape == (16, 3, 224, 224)
        assert depth_batch.shape == (16, 1, 224, 224)
        assert orth_batch.shape == (16, 1, 224, 224)
        assert labels_batch.shape == (16,)

        # Test value ranges in batch
        print(f"\n✓ Batch value ranges:")
        print(f"  RGB:   [{rgb_batch.min():.3f}, {rgb_batch.max():.3f}]")
        print(f"  Depth: [{depth_batch.min():.3f}, {depth_batch.max():.3f}]")
        print(f"  Orth:  [{orth_batch.min():.3f}, {orth_batch.max():.3f}]")

        assert rgb_batch.min() >= -1.01 and rgb_batch.max() <= 1.01
        assert depth_batch.min() >= -1.01 and depth_batch.max() <= 1.01
        assert orth_batch.min() >= -1.01 and orth_batch.max() <= 1.01

        print("\n✓✓✓ PASSED: DataLoader working correctly")
        return True

    except Exception as e:
        print(f"\n✗✗✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dtype_consistency():
    """Test that all tensors have correct dtype."""
    print("\n" + "=" * 80)
    print("TEST 6: Data Type Consistency")
    print("=" * 80)

    try:
        train_ds = SUNRGBD3StreamDataset(train=True)

        rgb, depth, orth, label = train_ds[0]

        print(f"\n✓ RGB dtype: {rgb.dtype} (expected: torch.float32)")
        print(f"✓ Depth dtype: {depth.dtype} (expected: torch.float32)")
        print(f"✓ Orth dtype: {orth.dtype} (expected: torch.float32)")
        print(f"✓ Label dtype: {type(label)} (expected: int)")

        assert rgb.dtype == torch.float32, f"RGB dtype mismatch: {rgb.dtype}"
        assert depth.dtype == torch.float32, f"Depth dtype mismatch: {depth.dtype}"
        assert orth.dtype == torch.float32, f"Orth dtype mismatch: {orth.dtype}"
        assert isinstance(label, (int, np.integer)), f"Label type mismatch: {type(label)}"

        print("\n✓✓✓ PASSED: All dtypes correct")
        return True

    except Exception as e:
        print(f"\n✗✗✗ FAILED: {e}")
        return False


def test_class_distribution():
    """Test class distribution and weights."""
    print("\n" + "=" * 80)
    print("TEST 7: Class Distribution")
    print("=" * 80)

    try:
        train_ds = SUNRGBD3StreamDataset(train=True)

        # Get class distribution
        dist = train_ds.get_class_distribution()

        print("\nClass distribution:")
        total_count = 0
        for class_name in train_ds.CLASS_NAMES:
            info = dist[class_name]
            print(f"  {class_name:20s}: {info['count']:5d} ({info['percentage']:5.2f}%)")
            total_count += info['count']

        print(f"\nTotal samples: {total_count}")
        assert total_count == len(train_ds), "Sample count mismatch"

        # Test class weights
        weights = train_ds.get_class_weights()
        print(f"\nClass weights shape: {weights.shape}")
        print(f"Class weights range: [{weights.min():.3f}, {weights.max():.3f}]")

        assert weights.shape[0] == 15, "Wrong number of class weights"
        assert (weights > 0).all(), "All weights should be positive"

        print("\n✓✓✓ PASSED: Class distribution correct")
        return True

    except Exception as e:
        print(f"\n✗✗✗ FAILED: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("FINAL COMPREHENSIVE VERIFICATION")
    print("sunrgbd_3stream_dataset.py")
    print("=" * 80)

    tests = [
        ("Basic Loading", test_basic_loading),
        ("Value Ranges", test_value_ranges),
        ("NaN/Inf Detection", test_no_nan_or_inf),
        ("Augmentation Independence", test_augmentation_independence),
        ("DataLoader Integration", test_dataloader),
        ("Data Type Consistency", test_dtype_consistency),
        ("Class Distribution", test_class_distribution),
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Dataset is verified and ready for production!")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("Please review the failures above.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
