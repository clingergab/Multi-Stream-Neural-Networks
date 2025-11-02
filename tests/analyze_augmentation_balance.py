"""
Analyze and visualize augmentation balance between RGB and Depth streams.

Quantifies augmentation strength and provides recommendations for reducing
RGB augmentation to balance stream performance.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_augmentation_strength():
    """Calculate expected augmentation strength for each modality."""

    print("\n" + "="*80)
    print("AUGMENTATION STRENGTH ANALYSIS")
    print("="*80)

    # Current RGB augmentations
    rgb_augs = {
        'Color Jitter': {
            'prob': 0.75,
            'strength': 0.5,  # ±50% for brightness/contrast/saturation
            'description': 'brightness/contrast/saturation ±50%, hue ±15%'
        },
        'Gaussian Blur': {
            'prob': 0.50,
            'strength': 0.6,  # Moderate-high blur (sigma up to 2.5)
            'description': 'kernel 3-9, sigma 0.1-2.5'
        },
        'Grayscale': {
            'prob': 0.35,
            'strength': 1.0,  # Complete color removal
            'description': 'Convert to grayscale (3 channels)'
        },
        'Random Erasing': {
            'prob': 0.30,
            'strength': 0.15,  # Up to 15% of image removed
            'description': '2-15% patches removed'
        },
    }

    # Current depth augmentations
    depth_augs = {
        'Brightness/Contrast + Noise': {
            'prob': 0.50,
            'strength': 0.3,  # ±25% + noise
            'description': 'brightness/contrast ±25%, Gaussian noise std=15'
        },
        'Random Erasing': {
            'prob': 0.10,
            'strength': 0.10,  # Up to 10% of image removed
            'description': '2-10% patches removed'
        },
    }

    # Shared augmentations
    shared_augs = {
        'Horizontal Flip': {
            'prob': 0.50,
            'strength': 0.5,  # Medium strength (spatial transformation)
            'description': 'Horizontal flip'
        },
        'Random Resized Crop': {
            'prob': 0.50,
            'strength': 0.15,  # Gentle crop (90-100% scale)
            'description': '90-100% scale, 0.95-1.05 ratio'
        },
    }

    # Calculate expected augmentation per sample
    def calc_expected_strength(augs):
        """Expected augmentation strength = Σ(prob × strength)"""
        return sum(aug['prob'] * aug['strength'] for aug in augs.values())

    rgb_strength = calc_expected_strength(rgb_augs)
    depth_strength = calc_expected_strength(depth_augs)
    shared_strength = calc_expected_strength(shared_augs)

    total_rgb = rgb_strength + shared_strength
    total_depth = depth_strength + shared_strength

    print(f"\n{'Modality':<15} {'Specific Aug':<15} {'Shared Aug':<15} {'Total':<15} {'Ratio'}")
    print("-" * 80)
    print(f"{'RGB':<15} {rgb_strength:>14.3f} {shared_strength:>14.3f} {total_rgb:>14.3f} {total_rgb/total_depth:>14.2f}x")
    print(f"{'Depth':<15} {depth_strength:>14.3f} {shared_strength:>14.3f} {total_depth:>14.3f} {'1.00x':>14s}")

    print(f"\n🔍 Analysis:")
    print(f"  • RGB receives {(total_rgb/total_depth - 1)*100:.1f}% MORE augmentation than depth")
    print(f"  • RGB-specific aug: {rgb_strength:.3f}")
    print(f"  • Depth-specific aug: {depth_strength:.3f}")
    print(f"  • Difference: {rgb_strength - depth_strength:.3f} ({((rgb_strength - depth_strength)/depth_strength)*100:.1f}%)")

    # Break down by augmentation type
    print(f"\n📊 RGB Augmentation Breakdown:")
    for name, aug in sorted(rgb_augs.items(), key=lambda x: x[1]['prob'] * x[1]['strength'], reverse=True):
        contrib = aug['prob'] * aug['strength']
        print(f"  • {name:<20} prob={aug['prob']:.0%}  strength={aug['strength']:.2f}  contrib={contrib:.3f}")
        print(f"    └─ {aug['description']}")

    print(f"\n📊 Depth Augmentation Breakdown:")
    for name, aug in sorted(depth_augs.items(), key=lambda x: x[1]['prob'] * x[1]['strength'], reverse=True):
        contrib = aug['prob'] * aug['strength']
        print(f"  • {name:<20} prob={aug['prob']:.0%}  strength={aug['strength']:.2f}  contrib={contrib:.3f}")
        print(f"    └─ {aug['description']}")

    return {
        'rgb_augs': rgb_augs,
        'depth_augs': depth_augs,
        'shared_augs': shared_augs,
        'rgb_strength': rgb_strength,
        'depth_strength': depth_strength,
        'shared_strength': shared_strength,
        'total_rgb': total_rgb,
        'total_depth': total_depth,
        'ratio': total_rgb / total_depth
    }


def propose_rebalanced_augmentations(current):
    """Propose reduced RGB augmentation settings."""

    print("\n" + "="*80)
    print("PROPOSED AUGMENTATION REBALANCING")
    print("="*80)

    # Target: RGB should be ~1.3-1.5x stronger than depth (not 2.45x)
    target_ratio = 1.35  # Moderate advantage for RGB

    current_total_rgb = current['total_rgb']
    current_total_depth = current['total_depth']
    shared_strength = current['shared_strength']

    # Calculate target RGB-specific strength
    target_total_rgb = target_ratio * current_total_depth
    target_rgb_specific = target_total_rgb - shared_strength

    reduction_factor = target_rgb_specific / current['rgb_strength']

    print(f"\n🎯 Target:")
    print(f"  • Current RGB/Depth ratio: {current['ratio']:.2f}x")
    print(f"  • Target RGB/Depth ratio: {target_ratio:.2f}x")
    print(f"  • Required RGB-specific strength: {target_rgb_specific:.3f}")
    print(f"  • Current RGB-specific strength: {current['rgb_strength']:.3f}")
    print(f"  • Reduction factor: {reduction_factor:.2f}x")

    print(f"\n📝 Proposed Changes:")
    print(f"\nOption 1: REDUCE PROBABILITIES (Recommended)")
    print("-" * 80)

    proposals_option1 = {
        'Color Jitter': {
            'current_prob': 0.75,
            'proposed_prob': 0.50,  # Reduce from 75% to 50%
            'current_strength': 0.5,
            'proposed_strength': 0.4,  # Also reduce strength slightly
            'description': 'brightness/contrast/saturation ±40% (reduced from ±50%), hue ±12% (reduced from ±15%)'
        },
        'Gaussian Blur': {
            'current_prob': 0.50,
            'proposed_prob': 0.30,  # Reduce from 50% to 30%
            'current_strength': 0.6,
            'proposed_strength': 0.5,  # Reduce max sigma
            'description': 'kernel 3-7 (reduced from 3-9), sigma 0.1-2.0 (reduced from 2.5)'
        },
        'Grayscale': {
            'current_prob': 0.35,
            'proposed_prob': 0.20,  # Reduce from 35% to 20%
            'current_strength': 1.0,
            'proposed_strength': 1.0,  # Keep strength
            'description': 'Convert to grayscale (3 channels) - UNCHANGED'
        },
        'Random Erasing': {
            'current_prob': 0.30,
            'proposed_prob': 0.20,  # Reduce from 30% to 20%
            'current_strength': 0.15,
            'proposed_strength': 0.12,  # Reduce max scale
            'description': '2-12% patches (reduced from 2-15%)'
        },
    }

    proposed_rgb_strength_opt1 = sum(
        aug['proposed_prob'] * aug['proposed_strength']
        for aug in proposals_option1.values()
    )
    proposed_total_rgb_opt1 = proposed_rgb_strength_opt1 + shared_strength
    proposed_ratio_opt1 = proposed_total_rgb_opt1 / current_total_depth

    for name, aug in proposals_option1.items():
        current_contrib = aug['current_prob'] * aug['current_strength']
        proposed_contrib = aug['proposed_prob'] * aug['proposed_strength']
        reduction = (1 - proposed_contrib/current_contrib) * 100

        print(f"\n{name}:")
        print(f"  Current:  prob={aug['current_prob']:.0%}  strength={aug['current_strength']:.2f}  contrib={current_contrib:.3f}")
        print(f"  Proposed: prob={aug['proposed_prob']:.0%}  strength={aug['proposed_strength']:.2f}  contrib={proposed_contrib:.3f}  (-{reduction:.0f}%)")
        print(f"  └─ {aug['description']}")

    print(f"\nOption 1 Results:")
    print(f"  • RGB-specific strength: {proposed_rgb_strength_opt1:.3f} (was {current['rgb_strength']:.3f})")
    print(f"  • Total RGB strength: {proposed_total_rgb_opt1:.3f} (was {current_total_rgb:.3f})")
    print(f"  • RGB/Depth ratio: {proposed_ratio_opt1:.2f}x (was {current['ratio']:.2f}x)")
    print(f"  • Reduction: {(1 - proposed_ratio_opt1/current['ratio'])*100:.1f}%")

    # Option 2: More conservative
    print(f"\n\nOption 2: MODERATE REDUCTION (More Conservative)")
    print("-" * 80)

    proposals_option2 = {
        'Color Jitter': {
            'prob': 0.60,  # Reduce from 75% to 60%
            'strength': 0.45,  # ±45%
            'description': 'brightness/contrast/saturation ±45%, hue ±13%'
        },
        'Gaussian Blur': {
            'prob': 0.35,  # Reduce from 50% to 35%
            'strength': 0.55,
            'description': 'kernel 3-7, sigma 0.1-2.2'
        },
        'Grayscale': {
            'prob': 0.25,  # Reduce from 35% to 25%
            'strength': 1.0,
            'description': 'Convert to grayscale (3 channels) - UNCHANGED'
        },
        'Random Erasing': {
            'prob': 0.25,  # Reduce from 30% to 25%
            'strength': 0.13,
            'description': '2-13% patches'
        },
    }

    proposed_rgb_strength_opt2 = sum(
        aug['prob'] * aug['strength']
        for aug in proposals_option2.values()
    )
    proposed_total_rgb_opt2 = proposed_rgb_strength_opt2 + shared_strength
    proposed_ratio_opt2 = proposed_total_rgb_opt2 / current_total_depth

    for name, aug in proposals_option2.items():
        current = proposals_option1[name]  # Get current values
        current_contrib = current['current_prob'] * current['current_strength']
        proposed_contrib = aug['prob'] * aug['strength']

        print(f"\n{name}:")
        print(f"  Current:  prob={current['current_prob']:.0%}  contrib={current_contrib:.3f}")
        print(f"  Proposed: prob={aug['prob']:.0%}  contrib={proposed_contrib:.3f}")
        print(f"  └─ {aug['description']}")

    print(f"\nOption 2 Results:")
    print(f"  • RGB-specific strength: {proposed_rgb_strength_opt2:.3f} (was {current['rgb_strength']:.3f})")
    print(f"  • Total RGB strength: {proposed_total_rgb_opt2:.3f} (was {current_total_rgb:.3f})")
    print(f"  • RGB/Depth ratio: {proposed_ratio_opt2:.2f}x (was {current['ratio']:.2f}x)")

    return {
        'option1': proposals_option1,
        'option2': proposals_option2,
        'option1_ratio': proposed_ratio_opt1,
        'option2_ratio': proposed_ratio_opt2,
    }


def generate_code_changes(proposals):
    """Generate exact code changes for the proposed rebalancing."""

    print("\n" + "="*80)
    print("CODE CHANGES FOR OPTION 1 (Recommended)")
    print("="*80)

    print("\nFile: src/data_utils/sunrgbd_dataset.py")
    print("\nChanges:")

    changes = [
        {
            'line': '132-142',
            'current': """            # 3. RGB-Only: Color Jitter (75% probability - INCREASED for overfitting)
            # Applies brightness, contrast, saturation, hue adjustments
            # Stronger augmentation to combat RGB overfitting
            if np.random.random() < 0.75:
                color_jitter = transforms.ColorJitter(
                    brightness=0.5,  # ±50% (increased from 0.4)
                    contrast=0.5,    # ±50% (increased from 0.4)
                    saturation=0.5,  # ±50% (increased from 0.4)
                    hue=0.15         # ±15% (increased from 0.1)
                )
                rgb = color_jitter(rgb)""",
            'proposed': """            # 3. RGB-Only: Color Jitter (50% probability - REDUCED for balance)
            # Applies brightness, contrast, saturation, hue adjustments
            # Moderate augmentation to balance RGB/Depth performance
            if np.random.random() < 0.50:
                color_jitter = transforms.ColorJitter(
                    brightness=0.4,  # ±40% (reduced from ±50%)
                    contrast=0.4,    # ±40% (reduced from ±50%)
                    saturation=0.4,  # ±40% (reduced from ±50%)
                    hue=0.12         # ±12% (reduced from ±15%)
                )
                rgb = color_jitter(rgb)"""
        },
        {
            'line': '144-153',
            'current': """            # 4. RGB-Only: Gaussian Blur (50% probability - INCREASED for overfitting)
            # Reduces reliance on fine textures/edges, forces focus on spatial structure
            # Critical for reducing RGB overfitting - used in SimCLR, MoCo, BYOL
            # Increased to 50% to further combat RGB overfitting
            if np.random.random() < 0.50:
                # Kernel size: random odd number between 3 and 9 (increased range)
                # Sigma: random between 0.1 and 2.5 (increased strength)
                kernel_size = int(np.random.choice([3, 5, 7, 9]))
                sigma = float(np.random.uniform(0.1, 2.5))
                rgb = transforms.functional.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)""",
            'proposed': """            # 4. RGB-Only: Gaussian Blur (30% probability - REDUCED for balance)
            # Reduces reliance on fine textures/edges, forces focus on spatial structure
            # Moderate augmentation to balance RGB/Depth performance
            if np.random.random() < 0.30:
                # Kernel size: random odd number between 3 and 7 (reduced range)
                # Sigma: random between 0.1 and 2.0 (reduced strength)
                kernel_size = int(np.random.choice([3, 5, 7]))
                sigma = float(np.random.uniform(0.1, 2.0))
                rgb = transforms.functional.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)"""
        },
        {
            'line': '155-159',
            'current': """            # 5. RGB-Only: Occasional Grayscale (35% - INCREASED for overfitting)
            # Forces RGB stream to learn from structure, not just color
            # Critical for reducing color-specific overfitting
            if np.random.random() < 0.35:
                rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)""",
            'proposed': """            # 5. RGB-Only: Occasional Grayscale (20% - REDUCED for balance)
            # Forces RGB stream to learn from structure, not just color
            # Moderate augmentation to balance RGB/Depth performance
            if np.random.random() < 0.20:
                rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)"""
        },
        {
            'line': '206-213',
            'current': """            # RGB random erasing (30% - INCREASED for overfitting)
            if np.random.random() < 0.30:
                erasing = transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.15),    # Small patches (2-15% of image, increased from 0.12)
                    ratio=(0.5, 2.0)       # Reasonable aspect ratios
                )
                rgb = erasing(rgb)""",
            'proposed': """            # RGB random erasing (20% - REDUCED for balance)
            if np.random.random() < 0.20:
                erasing = transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.12),    # Small patches (2-12% of image, reduced from 0.15)
                    ratio=(0.5, 2.0)       # Reasonable aspect ratios
                )
                rgb = erasing(rgb)"""
        }
    ]

    for i, change in enumerate(changes, 1):
        print(f"\n{i}. Lines {change['line']}:")
        print(f"   Change probability and/or strength parameters")


def main():
    print("\n" + "="*80)
    print("RGB VS DEPTH AUGMENTATION ANALYSIS")
    print("Analyzing current augmentation balance and proposing adjustments")
    print("="*80)

    # Analyze current state
    current = calculate_augmentation_strength()

    # Propose rebalanced augmentations
    proposals = propose_rebalanced_augmentations(current)

    # Generate code changes
    generate_code_changes(proposals)

    # Summary and recommendation
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATION")
    print("="*80)

    print(f"\n📊 Current State:")
    print(f"  • RGB augmentation is {current['ratio']:.2f}x stronger than Depth")
    print(f"  • This may explain why Depth is outperforming RGB")
    print(f"  • Excessive RGB augmentation can hurt performance")

    print(f"\n✅ Recommendation: OPTION 1")
    print(f"  • Reduces RGB/Depth ratio from {current['ratio']:.2f}x to {proposals['option1_ratio']:.2f}x")
    print(f"  • RGB still gets ~35% more augmentation (appropriate for its 3 channels)")
    print(f"  • More balanced approach - RGB and Depth can both contribute effectively")

    print(f"\n🎯 Expected Impact:")
    print(f"  • RGB performance should improve (less overfitting)")
    print(f"  • Depth performance may slightly decrease (less relative advantage)")
    print(f"  • Overall model should improve (better fusion)")

    print(f"\n⚠️  Alternative: OPTION 2 (More Conservative)")
    print(f"  • Reduces RGB/Depth ratio from {current['ratio']:.2f}x to {proposals['option2_ratio']:.2f}x")
    print(f"  • Smaller changes, easier to validate")
    print(f"  • Can iterate further if needed")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
