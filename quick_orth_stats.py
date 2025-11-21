"""
Quick focused analysis: Only compute P99, P99.9, ±3σ, ±4σ for Orth data.
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def quick_orth_analysis(data_root='data/sunrgbd_15', split='train'):
    """Fast analysis of key statistics only."""
    print("="*80)
    print("QUICK ORTH OUTLIER ANALYSIS")
    print("="*80)

    orth_dir = os.path.join(data_root, split, 'orth')
    orth_files = sorted([f for f in os.listdir(orth_dir) if f.endswith('.png')])

    print(f"\nLoading {len(orth_files)} Orth images...")

    # Load all data (this is the slow part, unavoidable)
    all_values = []
    for orth_file in tqdm(orth_files, desc="Loading"):
        orth_path = os.path.join(orth_dir, orth_file)
        orth_img = Image.open(orth_path)
        orth_arr = np.array(orth_img, dtype=np.float32)
        orth_scaled = orth_arr / 65535.0
        all_values.append(orth_scaled.flatten())

    all_values = np.concatenate(all_values)

    print(f"Total pixels: {len(all_values):,}\n")

    # Percentiles - ONLY what we need
    print(f"\n{'='*80}")
    print("PERCENTILES")
    print("="*80)
    p99 = np.percentile(all_values, 99)
    print(f"P99:   {p99:.6f}")

    p999 = np.percentile(all_values, 99.9)
    print(f"P99.9: {p999:.6f}")

    print("="*80)
    return {
        'p99': p99,
        'p999': p999
    }


if __name__ == "__main__":
    results = quick_orth_analysis()
