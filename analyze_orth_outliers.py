"""
Analyze outlier distribution in the new Orth data to decide on normalization strategy.

Key Questions:
1. How many values are beyond ±3σ, ±4σ, ±5σ?
2. Should we use exact std=0.0317 or clip outliers first?
3. What percentiles contain 99% of the data?
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def analyze_orth_outliers(data_root='data/sunrgbd_15', split='train'):
    """Analyze outlier distribution in Orth data."""
    print("=" * 80)
    print("ORTH OUTLIER ANALYSIS")
    print("=" * 80)

    orth_dir = os.path.join(data_root, split, 'orth')
    orth_files = sorted([f for f in os.listdir(orth_dir) if f.endswith('.png')])

    print(f"\nLoading {len(orth_files)} Orth images from {split} split...")

    # Load all Orth data
    all_values = []

    for orth_file in tqdm(orth_files, desc="Loading"):
        orth_path = os.path.join(orth_dir, orth_file)
        orth_img = Image.open(orth_path)
        orth_arr = np.array(orth_img, dtype=np.float32)

        # Scale from uint16 [0, 65535] to [0, 1]
        orth_scaled = orth_arr / 65535.0

        all_values.append(orth_scaled.flatten())

    # Concatenate all
    all_values = np.concatenate(all_values)

    print(f"\nTotal pixels analyzed: {len(all_values):,}")

    # Compute statistics
    mean = np.mean(all_values)
    std = np.std(all_values)
    median = np.median(all_values)

    print(f"\n{'='*80}")
    print("BASIC STATISTICS")
    print(f"{'='*80}")
    print(f"Mean:   {mean:.6f}")
    print(f"Std:    {std:.6f}")
    print(f"Median: {median:.6f}")
    print(f"Min:    {all_values.min():.6f}")
    print(f"Max:    {all_values.max():.6f}")

    # Percentiles
    print(f"\n{'='*80}")
    print("PERCENTILES")
    print(f"{'='*80}")
    percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(all_values, p)
        print(f"P{p:5.1f}: {val:.6f}")

    # Outlier analysis
    print(f"\n{'='*80}")
    print("OUTLIER ANALYSIS (based on σ)")
    print(f"{'='*80}")

    # Count values beyond different sigma thresholds
    thresholds = [2, 3, 4, 5, 6, 10, 15]

    print(f"\n{'Threshold':<12} {'Beyond ±Nσ':<15} {'Percentage':<12} {'Range':<30}")
    print("-" * 80)

    for n_sigma in thresholds:
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std

        beyond = np.sum((all_values < lower) | (all_values > upper))
        percentage = 100 * beyond / len(all_values)

        print(f"±{n_sigma}σ{'':<9} {beyond:>12,}   {percentage:>10.4f}%   [{lower:.4f}, {upper:.4f}]")

    # Show what the normalized range would be
    print(f"\n{'='*80}")
    print("NORMALIZED RANGES (with different std choices)")
    print(f"{'='*80}")

    # Normalized value = (x - mean) / std
    print(f"\nAssuming values in [0, 1]:")
    print(f"Formula: normalized = (x - {mean:.4f}) / std")
    print(f"\n{'Std Value':<12} {'Min Norm':<12} {'Max Norm':<12} {'Outliers':<15} {'Notes'}")
    print("-" * 80)

    std_options = [
        (0.0317, "Exact computed"),
        (0.0500, "Conservative (5%)"),
        (0.0710, "Previously assumed"),
        (0.1000, "Round number"),
        (0.1472, "Match Depth"),
    ]

    for std_val, note in std_options:
        min_norm = (0.0 - mean) / std_val
        max_norm = (1.0 - mean) / std_val

        # Count outliers if we normalize with this std
        lower_thresh = mean - 3 * std_val
        upper_thresh = mean + 3 * std_val
        outliers = np.sum((all_values < lower_thresh) | (all_values > upper_thresh))
        outlier_pct = 100 * outliers / len(all_values)

        print(f"{std_val:<12.4f} {min_norm:<12.2f} {max_norm:<12.2f} {outlier_pct:>6.2f}%{'':<8} {note}")

    # Clipping analysis
    print(f"\n{'='*80}")
    print("CLIPPING ANALYSIS")
    print(f"{'='*80}")

    # What if we clip to different percentiles?
    clip_options = [
        (0.1, 99.9),
        (0.5, 99.5),
        (1, 99),
        (2, 98),
        (5, 95),
    ]

    print(f"\n{'Percentiles':<15} {'Range':<30} {'Clipped':<15} {'New Std':<12} {'Norm Range'}")
    print("-" * 90)

    for p_low, p_high in clip_options:
        vmin = np.percentile(all_values, p_low)
        vmax = np.percentile(all_values, p_high)

        clipped_values = np.clip(all_values, vmin, vmax)
        clipped_count = np.sum((all_values < vmin) | (all_values > vmax))
        clipped_pct = 100 * clipped_count / len(all_values)

        new_std = np.std(clipped_values)
        new_mean = np.mean(clipped_values)

        # Normalized range after clipping
        norm_min = (vmin - new_mean) / new_std
        norm_max = (vmax - new_mean) / new_std

        print(f"P{p_low:>4.1f}-P{p_high:<5.1f}{'':<4} [{vmin:.4f}, {vmax:.4f}]{'':>8} "
              f"{clipped_pct:>6.2f}%{'':<7} {new_std:.6f}   [{norm_min:>6.2f}, {norm_max:>6.2f}]")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    # Count beyond ±3σ and ±4σ with exact std
    beyond_3sigma = np.sum((all_values < mean - 3*std) | (all_values > mean + 3*std))
    beyond_4sigma = np.sum((all_values < mean - 4*std) | (all_values > mean + 4*std))
    pct_3sigma = 100 * beyond_3sigma / len(all_values)
    pct_4sigma = 100 * beyond_4sigma / len(all_values)

    print(f"\nWith exact std={std:.4f}:")
    print(f"  Beyond ±3σ: {pct_3sigma:.4f}% ({beyond_3sigma:,} pixels)")
    print(f"  Beyond ±4σ: {pct_4sigma:.4f}% ({beyond_4sigma:,} pixels)")

    if pct_3sigma < 0.5:
        print(f"\n✓ RECOMMENDATION: Use exact std={std:.4f}")
        print(f"  - Very few outliers (<0.5% beyond ±3σ)")
        print(f"  - Data follows normal distribution well")
        print(f"  - Normalized range: [{(0-mean)/std:.2f}, {(1-mean)/std:.2f}]")
        print(f"  - This is mathematically correct and represents true data distribution")
    else:
        print(f"\n⚠️  RECOMMENDATION: Consider clipping outliers")
        print(f"  - Significant outliers (>{pct_3sigma:.2f}% beyond ±3σ)")
        print(f"  - Clip to P1-P99 or P0.5-P99.5 before computing stats")
        print(f"  - Recompute mean/std on clipped data")
        print(f"  - This will reduce extreme normalized values")

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'beyond_3sigma_pct': pct_3sigma,
        'beyond_4sigma_pct': pct_4sigma,
        'all_values': all_values
    }


if __name__ == "__main__":
    results = analyze_orth_outliers()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
