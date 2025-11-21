"""
SUPER FAST: Sample-based statistics for Orth data.
Only loads a subset of images to estimate percentiles and outliers.
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def sample_orth_analysis(data_root='data/sunrgbd_15', split='train', n_samples=2000):
    """Fast analysis using random sampling."""
    print("="*80)
    print("SUPER QUICK ORTH ANALYSIS (SAMPLED)")
    print("="*80)

    orth_dir = os.path.join(data_root, split, 'orth')
    all_files = sorted([f for f in os.listdir(orth_dir) if f.endswith('.png')])

    # Sample random files
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_files), size=min(n_samples, len(all_files)), replace=False)
    sampled_files = [all_files[i] for i in sample_indices]

    print(f"\nSampling {len(sampled_files)} out of {len(all_files)} images...")

    # Load sampled data
    all_values = []
    for orth_file in tqdm(sampled_files, desc="Loading"):
        orth_path = os.path.join(orth_dir, orth_file)
        orth_img = Image.open(orth_path)
        orth_arr = np.array(orth_img, dtype=np.float32)
        orth_scaled = orth_arr / 65535.0
        all_values.append(orth_scaled.flatten())

    all_values = np.concatenate(all_values)

    print(f"Total sampled pixels: {len(all_values):,}")

    # Compute basic stats
    mean = np.mean(all_values)
    std = np.std(all_values)

    print(f"\n{'='*80}")
    print("BASIC STATISTICS (estimated from sample)")
    print("="*80)
    print(f"Mean: {mean:.6f}")
    print(f"Std:  {std:.6f}")

    # Percentiles
    print(f"\n{'='*80}")
    print("PERCENTILES")
    print("="*80)
    p99 = np.percentile(all_values, 99)
    p999 = np.percentile(all_values, 99.9)
    print(f"P99:   {p99:.6f}")
    print(f"P99.9: {p999:.6f}")

    # Outlier counts
    print(f"\n{'='*80}")
    print("OUTLIER COUNTS")
    print("="*80)

    for n_sigma in [3, 4]:
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        beyond = np.sum((all_values < lower) | (all_values > upper))
        percentage = 100 * beyond / len(all_values)
        print(f"±{n_sigma}σ: {beyond:,} pixels ({percentage:.4f}%)")
        print(f"      Range: [{lower:.6f}, {upper:.6f}]")

    # Normalized range
    print(f"\n{'='*80}")
    print(f"NORMALIZED RANGE (with exact std={std:.4f})")
    print("="*80)
    min_norm = (0.0 - mean) / std
    max_norm = (1.0 - mean) / std
    print(f"[0, 1] → [{min_norm:.2f}, {max_norm:.2f}]")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print("="*80)

    beyond_3sigma_pct = 100 * np.sum((all_values < mean - 3*std) | (all_values > mean + 3*std)) / len(all_values)

    if beyond_3sigma_pct < 0.5:
        print(f"\n✓ Few outliers ({beyond_3sigma_pct:.4f}% beyond ±3σ)")
        print(f"  → Use exact std={std:.4f}")
        print(f"  → Normalized range: [{min_norm:.2f}, {max_norm:.2f}]")
    else:
        print(f"\n⚠ Significant outliers ({beyond_3sigma_pct:.4f}% beyond ±3σ)")
        print(f"  → Consider clipping to P1-P99: [{np.percentile(all_values, 1):.6f}, {p99:.6f}]")
        print(f"  → Then recompute mean/std on clipped data")

    print(f"\n{'='*80}\n")

    return {
        'mean': mean,
        'std': std,
        'p99': p99,
        'p999': p999,
        'beyond_3sigma_pct': beyond_3sigma_pct
    }


if __name__ == "__main__":
    results = sample_orth_analysis()
