"""
Comprehensive analysis of RGB vs Depth augmentation balance at baseline.

This script analyzes:
1. Probability of each augmentation type
2. Expected number of augmentation events per image
3. Magnitude/intensity of changes when applied
4. Variance introduced by each augmentation
5. Overall "augmentation budget" comparison
"""

import numpy as np
from dataclasses import dataclass

# Import baseline constants
from src.training.augmentation_config import (
    # Probability baselines
    BASE_FLIP_P,
    BASE_CROP_P,
    BASE_COLOR_JITTER_P,
    BASE_BLUR_P,
    BASE_GRAYSCALE_P,
    BASE_RGB_ERASING_P,
    BASE_DEPTH_AUG_P,
    BASE_DEPTH_ERASING_P,
    # Magnitude baselines
    BASE_CROP_SCALE_MIN,
    BASE_BRIGHTNESS,
    BASE_CONTRAST,
    BASE_SATURATION,
    BASE_HUE,
    BASE_BLUR_SIGMA_MIN,
    BASE_BLUR_SIGMA_MAX,
    BASE_ERASING_SCALE_MIN,
    BASE_ERASING_SCALE_MAX,
    BASE_DEPTH_BRIGHTNESS,
    BASE_DEPTH_CONTRAST,
    BASE_DEPTH_NOISE_STD,
)


@dataclass
class AugmentationStats:
    """Statistics for a single augmentation type."""
    name: str
    probability: float
    # Magnitude metrics (depends on aug type)
    magnitude_desc: str
    magnitude_range: tuple  # (min, max) or single value
    # Estimated pixel change (normalized 0-1 scale)
    pixel_change_estimate: float  # rough estimate of how much pixels change
    is_geometric: bool = False  # flip/crop vs appearance


def analyze_rgb_augmentations():
    """Analyze all RGB augmentation types."""
    augs = []

    # ColorJitter - most complex, affects brightness/contrast/saturation/hue
    # Each component can change pixels significantly
    # Brightness: pixel * (1 ± 0.37) -> up to 37% change
    # Contrast: (pixel - 0.5) * (1 ± 0.37) + 0.5 -> up to 37% change around midpoint
    # Saturation: affects color intensity
    # Hue: rotates colors
    augs.append(AugmentationStats(
        name="ColorJitter",
        probability=BASE_COLOR_JITTER_P,
        magnitude_desc=f"B/C/S=±{BASE_BRIGHTNESS:.2f}, H=±{BASE_HUE:.2f}",
        magnitude_range=(0, BASE_BRIGHTNESS),  # simplified
        pixel_change_estimate=0.25,  # average expected pixel change when applied
    ))

    # Gaussian Blur - smooths image
    augs.append(AugmentationStats(
        name="GaussianBlur",
        probability=BASE_BLUR_P,
        magnitude_desc=f"sigma=[{BASE_BLUR_SIGMA_MIN:.1f}, {BASE_BLUR_SIGMA_MAX:.1f}]",
        magnitude_range=(BASE_BLUR_SIGMA_MIN, BASE_BLUR_SIGMA_MAX),
        pixel_change_estimate=0.10,  # blur changes less dramatically
    ))

    # Grayscale - removes color information
    augs.append(AugmentationStats(
        name="Grayscale",
        probability=BASE_GRAYSCALE_P,
        magnitude_desc="Full desaturation",
        magnitude_range=(1.0, 1.0),
        pixel_change_estimate=0.15,  # significant but preserves luminance
    ))

    # Random Erasing - occludes part of image
    augs.append(AugmentationStats(
        name="RandomErasing",
        probability=BASE_RGB_ERASING_P,
        magnitude_desc=f"scale=[{BASE_ERASING_SCALE_MIN:.2f}, {BASE_ERASING_SCALE_MAX:.2f}]",
        magnitude_range=(BASE_ERASING_SCALE_MIN, BASE_ERASING_SCALE_MAX),
        pixel_change_estimate=0.06,  # avg erasing area * 100% change in that area
    ))

    return augs


def analyze_depth_augmentations():
    """Analyze all Depth augmentation types."""
    augs = []

    # Depth appearance block (brightness + contrast + noise applied together)
    augs.append(AugmentationStats(
        name="DepthAppearance",
        probability=BASE_DEPTH_AUG_P,
        magnitude_desc=f"B=±{BASE_DEPTH_BRIGHTNESS:.2f}, C=±{BASE_DEPTH_CONTRAST:.2f}, noise_std={BASE_DEPTH_NOISE_STD:.3f}",
        magnitude_range=(0, BASE_DEPTH_BRIGHTNESS),
        pixel_change_estimate=0.20,  # combined effect
    ))

    # Depth Random Erasing
    augs.append(AugmentationStats(
        name="DepthErasing",
        probability=BASE_DEPTH_ERASING_P,
        magnitude_desc=f"scale=[{BASE_ERASING_SCALE_MIN:.2f}, {BASE_ERASING_SCALE_MAX:.2f}]",
        magnitude_range=(BASE_ERASING_SCALE_MIN, BASE_ERASING_SCALE_MAX),
        pixel_change_estimate=0.06,
    ))

    return augs


def analyze_synchronized_augmentations():
    """Analyze synchronized (geometric) augmentations."""
    augs = []

    augs.append(AugmentationStats(
        name="HorizontalFlip",
        probability=BASE_FLIP_P,
        magnitude_desc="Mirror image",
        magnitude_range=(1.0, 1.0),
        pixel_change_estimate=0.0,  # pixels don't change, just position
        is_geometric=True,
    ))

    augs.append(AugmentationStats(
        name="RandomResizedCrop",
        probability=BASE_CROP_P,
        magnitude_desc=f"scale=[{BASE_CROP_SCALE_MIN:.2f}, 1.0]",
        magnitude_range=(BASE_CROP_SCALE_MIN, 1.0),
        pixel_change_estimate=0.0,  # pixels don't change, cropped/scaled
        is_geometric=True,
    ))

    return augs


def compute_expected_aug_events(augs):
    """Compute expected number of augmentation events per image."""
    return sum(aug.probability for aug in augs)


def compute_expected_pixel_change(augs):
    """
    Compute expected total pixel change per image.
    E[change] = sum(P(aug) * change_when_applied)
    """
    return sum(aug.probability * aug.pixel_change_estimate for aug in augs)


def monte_carlo_simulation(rgb_augs, depth_augs, n_samples=100000):
    """
    Simulate augmentation application to estimate variance and distribution.
    """
    np.random.seed(42)

    # Count how many augmentations are applied per sample
    rgb_counts = np.zeros(n_samples)
    depth_counts = np.zeros(n_samples)
    rgb_changes = np.zeros(n_samples)
    depth_changes = np.zeros(n_samples)

    for i in range(n_samples):
        # RGB augmentations
        for aug in rgb_augs:
            if np.random.random() < aug.probability:
                rgb_counts[i] += 1
                rgb_changes[i] += aug.pixel_change_estimate

        # Depth augmentations
        for aug in depth_augs:
            if np.random.random() < aug.probability:
                depth_counts[i] += 1
                depth_changes[i] += aug.pixel_change_estimate

    return {
        'rgb_counts': rgb_counts,
        'depth_counts': depth_counts,
        'rgb_changes': rgb_changes,
        'depth_changes': depth_changes,
    }


def print_detailed_analysis():
    """Print comprehensive analysis."""

    print("=" * 80)
    print("RGB vs DEPTH AUGMENTATION ANALYSIS AT BASELINE (all params = 1.0)")
    print("=" * 80)

    rgb_augs = analyze_rgb_augmentations()
    depth_augs = analyze_depth_augmentations()
    sync_augs = analyze_synchronized_augmentations()

    # =========================================================================
    # Section 1: Individual Augmentation Details
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. INDIVIDUAL AUGMENTATION DETAILS")
    print("=" * 80)

    print("\n--- SYNCHRONIZED AUGMENTATIONS (applied to both RGB and Depth) ---")
    print(f"{'Name':<20} {'Prob':>8} {'Magnitude':<40}")
    print("-" * 70)
    for aug in sync_augs:
        print(f"{aug.name:<20} {aug.probability:>8.2%} {aug.magnitude_desc:<40}")

    print("\n--- RGB-ONLY AUGMENTATIONS ---")
    print(f"{'Name':<20} {'Prob':>8} {'Magnitude':<40} {'Δpixel':>8}")
    print("-" * 80)
    for aug in rgb_augs:
        print(f"{aug.name:<20} {aug.probability:>8.2%} {aug.magnitude_desc:<40} {aug.pixel_change_estimate:>8.2f}")

    print("\n--- DEPTH-ONLY AUGMENTATIONS ---")
    print(f"{'Name':<20} {'Prob':>8} {'Magnitude':<40} {'Δpixel':>8}")
    print("-" * 80)
    for aug in depth_augs:
        print(f"{aug.name:<20} {aug.probability:>8.2%} {aug.magnitude_desc:<40} {aug.pixel_change_estimate:>8.2f}")

    # =========================================================================
    # Section 2: Probability Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. PROBABILITY ANALYSIS")
    print("=" * 80)

    rgb_total_prob = sum(aug.probability for aug in rgb_augs)
    depth_total_prob = sum(aug.probability for aug in depth_augs)
    sync_total_prob = sum(aug.probability for aug in sync_augs)

    print(f"\nSum of probabilities (can exceed 1.0 since augs are independent):")
    print(f"  RGB-only:     {rgb_total_prob:.2f} ({rgb_total_prob*100:.0f}%)")
    print(f"  Depth-only:   {depth_total_prob:.2f} ({depth_total_prob*100:.0f}%)")
    print(f"  Synchronized: {sync_total_prob:.2f} ({sync_total_prob*100:.0f}%)")

    print(f"\nExpected augmentation events per image:")
    print(f"  RGB appearance augs:   {rgb_total_prob:.3f}")
    print(f"  Depth appearance augs: {depth_total_prob:.3f}")
    print(f"  Geometric (both):      {sync_total_prob:.3f}")

    print(f"\n  RGB TOTAL (appear + geo):   {rgb_total_prob + sync_total_prob:.3f}")
    print(f"  Depth TOTAL (appear + geo): {depth_total_prob + sync_total_prob:.3f}")

    rgb_to_depth_ratio = (rgb_total_prob + sync_total_prob) / (depth_total_prob + sync_total_prob)
    print(f"\n  RGB:Depth ratio (event count): {rgb_to_depth_ratio:.2f}:1")

    # =========================================================================
    # Section 3: Expected Pixel Change Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. EXPECTED PIXEL CHANGE ANALYSIS")
    print("=" * 80)

    rgb_expected_change = compute_expected_pixel_change(rgb_augs)
    depth_expected_change = compute_expected_pixel_change(depth_augs)

    print(f"\nExpected pixel change per image (appearance augs only):")
    print(f"  RGB:   {rgb_expected_change:.4f} ({rgb_expected_change*100:.2f}%)")
    print(f"  Depth: {depth_expected_change:.4f} ({depth_expected_change*100:.2f}%)")

    if depth_expected_change > 0:
        change_ratio = rgb_expected_change / depth_expected_change
        print(f"\n  RGB:Depth ratio (pixel change): {change_ratio:.2f}:1")

    # =========================================================================
    # Section 4: Monte Carlo Simulation
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. MONTE CARLO SIMULATION (100,000 samples)")
    print("=" * 80)

    sim = monte_carlo_simulation(rgb_augs, depth_augs)

    print("\n--- Augmentation Count Distribution ---")
    print(f"{'Metric':<25} {'RGB':>12} {'Depth':>12}")
    print("-" * 50)
    print(f"{'Mean aug events':<25} {np.mean(sim['rgb_counts']):>12.3f} {np.mean(sim['depth_counts']):>12.3f}")
    print(f"{'Std dev':<25} {np.std(sim['rgb_counts']):>12.3f} {np.std(sim['depth_counts']):>12.3f}")
    print(f"{'Min':<25} {np.min(sim['rgb_counts']):>12.0f} {np.min(sim['depth_counts']):>12.0f}")
    print(f"{'Max':<25} {np.max(sim['rgb_counts']):>12.0f} {np.max(sim['depth_counts']):>12.0f}")
    print(f"{'Median':<25} {np.median(sim['rgb_counts']):>12.1f} {np.median(sim['depth_counts']):>12.1f}")

    # Distribution of counts
    print("\n--- Count Distribution (% of samples) ---")
    for count in range(5):
        rgb_pct = np.mean(sim['rgb_counts'] == count) * 100
        depth_pct = np.mean(sim['depth_counts'] == count) * 100
        print(f"  {count} augmentations:  RGB={rgb_pct:>5.1f}%  Depth={depth_pct:>5.1f}%")

    print("\n--- Pixel Change Distribution ---")
    print(f"{'Metric':<25} {'RGB':>12} {'Depth':>12}")
    print("-" * 50)
    print(f"{'Mean pixel change':<25} {np.mean(sim['rgb_changes']):>12.4f} {np.mean(sim['depth_changes']):>12.4f}")
    print(f"{'Std dev':<25} {np.std(sim['rgb_changes']):>12.4f} {np.std(sim['depth_changes']):>12.4f}")
    print(f"{'P10':<25} {np.percentile(sim['rgb_changes'], 10):>12.4f} {np.percentile(sim['depth_changes'], 10):>12.4f}")
    print(f"{'P50 (median)':<25} {np.percentile(sim['rgb_changes'], 50):>12.4f} {np.percentile(sim['depth_changes'], 50):>12.4f}")
    print(f"{'P90':<25} {np.percentile(sim['rgb_changes'], 90):>12.4f} {np.percentile(sim['depth_changes'], 90):>12.4f}")

    # =========================================================================
    # Section 5: Detailed Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. DETAILED COMPARISON")
    print("=" * 80)

    print("\n--- Probability Breakdown ---")
    print(f"\nRGB has {len(rgb_augs)} independent appearance augmentation types:")
    for aug in rgb_augs:
        print(f"  - {aug.name}: {aug.probability:.0%}")

    print(f"\nDepth has {len(depth_augs)} independent appearance augmentation types:")
    for aug in depth_augs:
        print(f"  - {aug.name}: {aug.probability:.0%}")

    # Probability that at least one aug is applied
    rgb_no_aug_prob = 1.0
    for aug in rgb_augs:
        rgb_no_aug_prob *= (1 - aug.probability)
    rgb_any_aug_prob = 1 - rgb_no_aug_prob

    depth_no_aug_prob = 1.0
    for aug in depth_augs:
        depth_no_aug_prob *= (1 - aug.probability)
    depth_any_aug_prob = 1 - depth_no_aug_prob

    print(f"\nProbability that at least one appearance aug is applied:")
    print(f"  RGB:   {rgb_any_aug_prob:.1%}")
    print(f"  Depth: {depth_any_aug_prob:.1%}")

    # =========================================================================
    # Section 6: Magnitude Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. MAGNITUDE/INTENSITY COMPARISON")
    print("=" * 80)

    print("\n--- When applied, how much does each augmentation change the image? ---")
    print("\nRGB ColorJitter (when applied):")
    print(f"  Brightness: multiplies pixels by [{1-BASE_BRIGHTNESS:.2f}, {1+BASE_BRIGHTNESS:.2f}]")
    print(f"  Contrast: scales deviation from 0.5 by [{1-BASE_CONTRAST:.2f}, {1+BASE_CONTRAST:.2f}]")
    print(f"  Saturation: scales color by [{1-BASE_SATURATION:.2f}, {1+BASE_SATURATION:.2f}]")
    print(f"  Hue: rotates hue by ±{BASE_HUE*180:.0f}° (out of 360°)")

    print("\nDepth Appearance (when applied):")
    print(f"  Brightness: multiplies pixels by [{1-BASE_DEPTH_BRIGHTNESS:.2f}, {1+BASE_DEPTH_BRIGHTNESS:.2f}]")
    print(f"  Contrast: scales deviation from 0.5 by [{1-BASE_DEPTH_CONTRAST:.2f}, {1+BASE_DEPTH_CONTRAST:.2f}]")
    print(f"  Noise: adds Gaussian noise with std={BASE_DEPTH_NOISE_STD:.3f} (~{BASE_DEPTH_NOISE_STD*255:.1f}/255)")

    print("\n--- Magnitude Ratio ---")
    print(f"  RGB brightness range:   ±{BASE_BRIGHTNESS:.0%}")
    print(f"  Depth brightness range: ±{BASE_DEPTH_BRIGHTNESS:.0%}")
    print(f"  Ratio: RGB is {BASE_BRIGHTNESS/BASE_DEPTH_BRIGHTNESS:.1f}x stronger")

    # =========================================================================
    # Section 7: Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. SUMMARY")
    print("=" * 80)

    print("""
KEY FINDINGS AT BASELINE:

1. AUGMENTATION COUNT:
   - RGB gets ~{:.2f} appearance aug events per image
   - Depth gets ~{:.2f} appearance aug events per image
   - RGB:Depth ratio = {:.1f}:1 (RGB gets more variety)

2. PIXEL CHANGE:
   - RGB expected change: ~{:.1%} per image
   - Depth expected change: ~{:.1%} per image
   - RGB:Depth ratio = {:.1f}:1 (RGB changes pixels more)

3. WHY RGB > DEPTH:
   - RGB has 4 independent augmentation types vs Depth's 2
   - RGB ColorJitter is stronger (±37% vs ±25% brightness)
   - Depth's main augmentation bundles 3 effects into one probability gate

4. TO BALANCE RGB AND DEPTH:
   - To equalize event count: depth_aug_prob ≈ {:.2f}
   - To equalize pixel change: depth_aug_prob ≈ {:.2f}, depth_aug_mag ≈ {:.2f}

5. RECOMMENDATION:
   - Current default slightly favors RGB augmentation
   - For experiments: try depth_aug_prob=1.2-1.5 to balance
   - Or accept the asymmetry as intentional (color varies more than depth in real world)
""".format(
        rgb_total_prob, depth_total_prob,
        rgb_total_prob/depth_total_prob if depth_total_prob > 0 else float('inf'),
        rgb_expected_change, depth_expected_change,
        rgb_expected_change/depth_expected_change if depth_expected_change > 0 else float('inf'),
        rgb_total_prob / BASE_DEPTH_AUG_P,  # to match event count
        rgb_expected_change / (BASE_DEPTH_AUG_P * 0.20),  # rough estimate
        BASE_BRIGHTNESS / BASE_DEPTH_BRIGHTNESS,  # magnitude scaling
    ))


if __name__ == "__main__":
    print_detailed_analysis()
