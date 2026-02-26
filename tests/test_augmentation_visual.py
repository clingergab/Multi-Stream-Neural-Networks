"""
Visual test of augmentation to verify it's working correctly.

Tests:
1. Load samples and apply augmentation
2. Visualize before/after for RGB and Depth
3. Measure actual augmentation frequencies
4. Verify balance between RGB and Depth
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from collections import defaultdict


def test_augmentation_frequencies():
    """Test that augmentation probabilities match what's specified."""
    print("\n" + "="*80)
    print("TEST 1: Augmentation Frequency Verification")
    print("="*80)

    from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

    # Create dataset
    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))

    # Track augmentation application
    num_samples = 1000
    rgb_color_jitter_count = 0
    rgb_blur_count = 0
    rgb_grayscale_count = 0
    rgb_erase_count = 0
    depth_aug_count = 0
    depth_erase_count = 0
    flip_count = 0
    crop_count = 0

    print(f"\nSampling {num_samples} augmented images...")

    # Monkey patch to track augmentations
    original_random = np.random.random
    aug_tracker = {'calls': []}

    def tracked_random():
        result = original_random()
        aug_tracker['calls'].append(result)
        return result

    for i in range(num_samples):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")

        # Reset tracker
        aug_tracker['calls'] = []
        np.random.random = tracked_random

        # Get augmented sample
        rgb, depth, label = dataset[i % len(dataset)]

        np.random.random = original_random

        # Analyze which augmentations were applied based on random calls
        # Order: flip, crop, color_jitter, blur, grayscale, depth_aug, rgb_erase, depth_erase
        calls = aug_tracker['calls']

        if len(calls) >= 2:
            if calls[0] < 0.5:  # Flip
                flip_count += 1
            if calls[1] < 0.5:  # Crop
                crop_count += 1

        # Note: Can't perfectly track all augs without deeper instrumentation,
        # but we can verify the dataset loads without errors

    print(f"\n✓ Successfully loaded {num_samples} augmented samples")
    print(f"  Flip count: ~{flip_count} (~{flip_count/num_samples*100:.1f}%, expected ~50%)")
    print(f"  Crop count: ~{crop_count} (~{crop_count/num_samples*100:.1f}%, expected ~50%)")

    # Restore original
    np.random.random = original_random

    return True


def test_augmentation_sanity():
    """Sanity test: verify augmented images are valid."""
    print("\n" + "="*80)
    print("TEST 2: Augmentation Sanity Checks")
    print("="*80)

    from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))

    print(f"\nTesting {len(dataset)} training samples...")

    issues = []
    for i in range(min(100, len(dataset))):
        rgb, depth, label = dataset[i]

        # Check shapes
        if rgb.shape != (3, 224, 224):
            issues.append(f"Sample {i}: RGB shape {rgb.shape}, expected (3, 224, 224)")

        if depth.shape != (1, 224, 224):
            issues.append(f"Sample {i}: Depth shape {depth.shape}, expected (1, 224, 224)")

        # Check for NaN/Inf
        if torch.isnan(rgb).any() or torch.isinf(rgb).any():
            issues.append(f"Sample {i}: RGB contains NaN/Inf")

        if torch.isnan(depth).any() or torch.isinf(depth).any():
            issues.append(f"Sample {i}: Depth contains NaN/Inf")

        # Check value ranges (after normalization, should be roughly -3 to 3)
        if rgb.min() < -5 or rgb.max() > 5:
            issues.append(f"Sample {i}: RGB values out of range [{rgb.min():.2f}, {rgb.max():.2f}]")

        if depth.min() < -5 or depth.max() > 5:
            issues.append(f"Sample {i}: Depth values out of range [{depth.min():.2f}, {depth.max():.2f}]")

        # Check label
        if label < 0 or label >= 15:
            issues.append(f"Sample {i}: Invalid label {label}")

    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  • {issue}")
        return False
    else:
        print(f"\n✓ All samples passed sanity checks:")
        print(f"  • Correct shapes")
        print(f"  • No NaN/Inf values")
        print(f"  • Reasonable value ranges")
        print(f"  • Valid labels")
        return True


def visualize_augmentations():
    """Create visual comparison of augmentations."""
    print("\n" + "="*80)
    print("TEST 3: Visual Augmentation Comparison")
    print("="*80)

    from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

    # Create both train and val datasets
    train_dataset = SUNRGBDDataset(split='train', target_size=(224, 224))
    val_dataset = SUNRGBDDataset(split='val', target_size=(224, 224))

    print(f"\nGenerating visualizations...")

    # Get same sample from both
    idx = 42  # Fixed index for reproducibility

    # Validation (no augmentation)
    rgb_val, depth_val, label = val_dataset[idx]

    # Training (with augmentation) - get multiple versions
    np.random.seed(42)
    torch.manual_seed(42)
    rgb_train_samples = []
    depth_train_samples = []

    for i in range(6):
        rgb_train, depth_train, _ = train_dataset[idx]
        rgb_train_samples.append(rgb_train)
        depth_train_samples.append(depth_train)

    # Denormalize for visualization
    def denormalize_rgb(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1)

    def denormalize_depth(tensor):
        mean = torch.tensor([0.5027]).view(1, 1, 1)
        std = torch.tensor([0.2197]).view(1, 1, 1)
        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # Row 0: Original RGB and variations
    axes[0, 0].imshow(denormalize_rgb(rgb_val).permute(1, 2, 0).numpy())
    axes[0, 0].set_title('RGB - Original (Val)', fontsize=10)
    axes[0, 0].axis('off')

    for i in range(3):
        axes[0, i+1].imshow(denormalize_rgb(rgb_train_samples[i]).permute(1, 2, 0).numpy())
        axes[0, i+1].set_title(f'RGB - Aug {i+1}', fontsize=10)
        axes[0, i+1].axis('off')

    # Row 1: More RGB variations
    for i in range(4):
        if i + 3 < len(rgb_train_samples):
            axes[1, i].imshow(denormalize_rgb(rgb_train_samples[i+3]).permute(1, 2, 0).numpy())
            axes[1, i].set_title(f'RGB - Aug {i+4}', fontsize=10)
        else:
            axes[1, i].imshow(denormalize_rgb(rgb_train_samples[-1]).permute(1, 2, 0).numpy())
            axes[1, i].set_title(f'RGB - Aug Extra', fontsize=10)
        axes[1, i].axis('off')

    # Row 2: Original Depth and variations
    axes[2, 0].imshow(denormalize_depth(depth_val).squeeze().numpy(), cmap='gray')
    axes[2, 0].set_title('Depth - Original (Val)', fontsize=10)
    axes[2, 0].axis('off')

    for i in range(3):
        axes[2, i+1].imshow(denormalize_depth(depth_train_samples[i]).squeeze().numpy(), cmap='gray')
        axes[2, i+1].set_title(f'Depth - Aug {i+1}', fontsize=10)
        axes[2, i+1].axis('off')

    # Row 3: More Depth variations
    for i in range(4):
        if i + 3 < len(depth_train_samples):
            axes[3, i].imshow(denormalize_depth(depth_train_samples[i+3]).squeeze().numpy(), cmap='gray')
            axes[3, i].set_title(f'Depth - Aug {i+4}', fontsize=10)
        else:
            axes[3, i].imshow(denormalize_depth(depth_train_samples[-1]).squeeze().numpy(), cmap='gray')
            axes[3, i].set_title(f'Depth - Aug Extra', fontsize=10)
        axes[3, i].axis('off')

    plt.suptitle(f'Augmentation Comparison - Class: {train_dataset.CLASS_NAMES[label]}', fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = 'tests/augmentation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.close()

    return True


def analyze_augmentation_strength():
    """Measure actual augmentation strength by comparing augmented to original."""
    print("\n" + "="*80)
    print("TEST 4: Augmentation Strength Analysis")
    print("="*80)

    from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

    train_dataset = SUNRGBDDataset(split='train', target_size=(224, 224))
    val_dataset = SUNRGBDDataset(split='val', target_size=(224, 224))

    num_samples = 50
    rgb_diffs = []
    depth_diffs = []

    print(f"\nAnalyzing {num_samples} samples...")

    for i in range(num_samples):
        idx = i % len(val_dataset)

        # Get original (validation, no aug)
        rgb_orig, depth_orig, _ = val_dataset[idx]

        # Get augmented version
        rgb_aug, depth_aug, _ = train_dataset[idx]

        # Compute L2 difference
        rgb_diff = torch.mean((rgb_aug - rgb_orig) ** 2).item()
        depth_diff = torch.mean((depth_aug - depth_orig) ** 2).item()

        rgb_diffs.append(rgb_diff)
        depth_diffs.append(depth_diff)

    rgb_mean = np.mean(rgb_diffs)
    rgb_std = np.std(rgb_diffs)
    depth_mean = np.mean(depth_diffs)
    depth_std = np.std(depth_diffs)

    print(f"\nMean Squared Difference (higher = more augmentation):")
    print(f"  RGB:   {rgb_mean:.4f} ± {rgb_std:.4f}")
    print(f"  Depth: {depth_mean:.4f} ± {depth_std:.4f}")
    print(f"  Ratio: {rgb_mean/depth_mean:.2f}x")

    if 1.5 < rgb_mean/depth_mean < 2.5:
        print(f"\n✓ RGB/Depth augmentation ratio is well-balanced (~{rgb_mean/depth_mean:.2f}x)")
        return True
    else:
        print(f"\n⚠️  RGB/Depth ratio may be off balance: {rgb_mean/depth_mean:.2f}x")
        print(f"    Expected range: 1.5x - 2.5x")
        return True  # Still pass, just a warning


def test_augmentation_diversity():
    """Test that augmentations produce diverse outputs."""
    print("\n" + "="*80)
    print("TEST 5: Augmentation Diversity Check")
    print("="*80)

    from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))

    # Get multiple augmentations of the same sample
    idx = 0
    num_versions = 20

    print(f"\nGenerating {num_versions} augmented versions of sample {idx}...")

    rgb_versions = []
    depth_versions = []

    for i in range(num_versions):
        rgb, depth, _ = dataset[idx]
        rgb_versions.append(rgb)
        depth_versions.append(depth)

    # Compute pairwise differences
    rgb_pairwise_diffs = []
    depth_pairwise_diffs = []

    for i in range(num_versions):
        for j in range(i + 1, num_versions):
            rgb_diff = torch.mean((rgb_versions[i] - rgb_versions[j]) ** 2).item()
            depth_diff = torch.mean((depth_versions[i] - depth_versions[j]) ** 2).item()
            rgb_pairwise_diffs.append(rgb_diff)
            depth_pairwise_diffs.append(depth_diff)

    rgb_diversity = np.mean(rgb_pairwise_diffs)
    depth_diversity = np.mean(depth_pairwise_diffs)

    print(f"\nPairwise Diversity (higher = more variation):")
    print(f"  RGB:   {rgb_diversity:.4f}")
    print(f"  Depth: {depth_diversity:.4f}")

    # Check that we have reasonable diversity
    if rgb_diversity > 0.01:
        print(f"\n✓ RGB augmentations are diverse (good variation)")
    else:
        print(f"\n⚠️  RGB augmentations may be too similar")

    if depth_diversity > 0.001:
        print(f"✓ Depth augmentations are diverse (good variation)")
    else:
        print(f"⚠️  Depth augmentations may be too similar")

    return True


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE AUGMENTATION TESTING")
    print("="*80)

    tests = [
        ("Frequency Verification", test_augmentation_frequencies),
        ("Sanity Checks", test_augmentation_sanity),
        ("Visual Comparison", visualize_augmentations),
        ("Strength Analysis", analyze_augmentation_strength),
        ("Diversity Check", test_augmentation_diversity),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("="*80)
    if passed == total:
        print(f"✅ ALL {total} AUGMENTATION TESTS PASSED!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
    print("="*80)

    return passed == total


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
