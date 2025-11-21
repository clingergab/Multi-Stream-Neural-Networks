"""
Deep investigation of Orthogonal data distribution.

We need to understand:
1. Is the std really 0.0249 or 0.0710?
2. What does the actual distribution look like?
3. Is the [-20, 20] range from outliers or the actual data spread?
4. Should we use the true computed std or is there a scaling issue?
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def analyze_orth_raw_values(num_samples=1000):
    """Analyze raw orth values before any processing."""
    print("=" * 80)
    print("ANALYZING RAW ORTH VALUES")
    print("=" * 80)

    data_root = 'data/sunrgbd_15'
    split = 'train'
    orth_dir = os.path.join(data_root, split, 'orth')

    all_values = []

    print(f"\nLoading {num_samples} samples...")
    for idx in tqdm(range(num_samples), desc="Loading orth files"):
        orth_path = os.path.join(orth_dir, f'{idx:05d}.png')
        if os.path.exists(orth_path):
            orth = Image.open(orth_path)
            orth_arr = np.array(orth, dtype=np.float32)
            all_values.append(orth_arr.flatten())

    # Concatenate all values
    all_values = np.concatenate(all_values)

    print(f"\n[RAW 16-BIT VALUES]")
    print(f"  Total pixels: {len(all_values):,}")
    print(f"  Range: [{all_values.min()}, {all_values.max()}]")
    print(f"  Mean: {all_values.mean():.2f}")
    print(f"  Std: {all_values.std():.2f}")
    print(f"  Median: {np.median(all_values):.2f}")
    print(f"  25th percentile: {np.percentile(all_values, 25):.2f}")
    print(f"  75th percentile: {np.percentile(all_values, 75):.2f}")
    print(f"  99th percentile: {np.percentile(all_values, 99):.2f}")
    print(f"  1st percentile: {np.percentile(all_values, 1):.2f}")

    return all_values


def analyze_orth_scaled_values(num_samples=1000):
    """Analyze orth values after scaling to [0,1]."""
    print("\n" + "=" * 80)
    print("ANALYZING SCALED ORTH VALUES (after /65535)")
    print("=" * 80)

    data_root = 'data/sunrgbd_15'
    split = 'train'
    orth_dir = os.path.join(data_root, split, 'orth')
    target_size = (416, 544)

    all_values_scaled = []

    print(f"\nLoading and scaling {num_samples} samples...")
    for idx in tqdm(range(num_samples), desc="Processing"):
        orth_path = os.path.join(orth_dir, f'{idx:05d}.png')
        if os.path.exists(orth_path):
            orth = Image.open(orth_path)

            # Apply same scaling as dataset
            orth_arr = np.array(orth, dtype=np.float32)
            orth_scaled = np.clip(orth_arr / 65535.0, 0.0, 1.0)

            # Resize like dataset does
            orth_img = Image.fromarray(orth_scaled, mode='F')
            orth_img = orth_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            orth_scaled = np.array(orth_img, dtype=np.float32)

            all_values_scaled.append(orth_scaled.flatten())

    all_values_scaled = np.concatenate(all_values_scaled)

    mean_scaled = all_values_scaled.mean()
    std_scaled = all_values_scaled.std()

    print(f"\n[SCALED [0,1] VALUES - After Resize to {target_size}]")
    print(f"  Total pixels: {len(all_values_scaled):,}")
    print(f"  Range: [{all_values_scaled.min():.6f}, {all_values_scaled.max():.6f}]")
    print(f"  Mean: {mean_scaled:.6f}")
    print(f"  Std: {std_scaled:.6f}  ← THIS IS THE KEY VALUE!")
    print(f"  Median: {np.median(all_values_scaled):.6f}")
    print(f"  25th percentile: {np.percentile(all_values_scaled, 25):.6f}")
    print(f"  75th percentile: {np.percentile(all_values_scaled, 75):.6f}")
    print(f"  99th percentile: {np.percentile(all_values_scaled, 99):.6f}")
    print(f"  1st percentile: {np.percentile(all_values_scaled, 1):.6f}")

    print(f"\n[INTERPRETATION]")
    print(f"  IQR (75th - 25th): {np.percentile(all_values_scaled, 75) - np.percentile(all_values_scaled, 25):.6f}")
    print(f"  Range (99th - 1st): {np.percentile(all_values_scaled, 99) - np.percentile(all_values_scaled, 1):.6f}")

    # Check concentration
    within_1std = np.sum((all_values_scaled >= mean_scaled - std_scaled) &
                         (all_values_scaled <= mean_scaled + std_scaled))
    pct_within_1std = 100 * within_1std / len(all_values_scaled)

    print(f"  % within 1 std of mean: {pct_within_1std:.2f}%")
    print(f"    (Normal distribution: ~68%)")

    return all_values_scaled, mean_scaled, std_scaled


def test_normalization_ranges(scaled_values, mean, std):
    """Test what normalized range we actually get."""
    print("\n" + "=" * 80)
    print("TESTING NORMALIZED RANGES")
    print("=" * 80)

    normalized = (scaled_values - mean) / std

    print(f"\n[WITH COMPUTED STD = {std:.6f}]")
    print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"  Normalized mean: {normalized.mean():.3f} (should be ~0)")
    print(f"  Normalized std: {normalized.std():.3f} (should be ~1)")
    print(f"  99.9th percentile: {np.percentile(normalized, 99.9):.3f}")
    print(f"  0.1th percentile: {np.percentile(normalized, 0.1):.3f}")

    # Check outliers
    extreme_positive = np.sum(normalized > 10)
    extreme_negative = np.sum(normalized < -10)
    pct_extreme = 100 * (extreme_positive + extreme_negative) / len(normalized)

    print(f"\n[OUTLIER ANALYSIS]")
    print(f"  Values > +10: {extreme_positive:,} ({100*extreme_positive/len(normalized):.4f}%)")
    print(f"  Values < -10: {extreme_negative:,} ({100*extreme_negative/len(normalized):.4f}%)")
    print(f"  Total extreme: {pct_extreme:.4f}%")

    # Check if [-20, 20] is real or outliers
    beyond_20 = np.sum(np.abs(normalized) > 20)
    beyond_15 = np.sum(np.abs(normalized) > 15)
    beyond_10 = np.sum(np.abs(normalized) > 10)
    beyond_5 = np.sum(np.abs(normalized) > 5)

    print(f"\n[RANGE ANALYSIS]")
    print(f"  |values| > 5:  {beyond_5:,} ({100*beyond_5/len(normalized):.4f}%)")
    print(f"  |values| > 10: {beyond_10:,} ({100*beyond_10/len(normalized):.4f}%)")
    print(f"  |values| > 15: {beyond_15:,} ({100*beyond_15/len(normalized):.4f}%)")
    print(f"  |values| > 20: {beyond_20:,} ({100*beyond_20/len(normalized):.4f}%)")

    if pct_extreme < 0.01:
        print(f"\n  💡 Less than 0.01% of values are extreme!")
        print(f"     The [-20, 20] range is from RARE OUTLIERS, not the bulk of data")
    else:
        print(f"\n  ⚠️  {pct_extreme:.2f}% of values are extreme")
        print(f"     This represents a significant portion of the data")

    return normalized


def visualize_distributions(raw_values, scaled_values, normalized_values, mean, std):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Raw values
    axes[0, 0].hist(raw_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Raw 16-bit Values', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(raw_values.mean(), color='red', linestyle='--',
                       label=f'Mean={raw_values.mean():.0f}', linewidth=2)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Raw values (zoomed histogram)
    axes[0, 1].hist(raw_values, bins=200, alpha=0.7, color='blue', edgecolor='black', log=True)
    axes[0, 1].set_title('Raw Values (Log Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency (log)')
    axes[0, 1].grid(alpha=0.3)

    # Raw values box plot
    axes[0, 2].boxplot(raw_values[::100], vert=True)  # Subsample for speed
    axes[0, 2].set_title('Raw Values Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].grid(alpha=0.3)

    # Scaled values
    axes[1, 0].hist(scaled_values, bins=100, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_title(f'Scaled [0,1] Values\nStd={std:.6f}', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(mean, color='red', linestyle='--',
                       label=f'Mean={mean:.4f}', linewidth=2)
    axes[1, 0].axvline(mean - std, color='orange', linestyle=':',
                       label=f'±1 std', linewidth=1.5)
    axes[1, 0].axvline(mean + std, color='orange', linestyle=':', linewidth=1.5)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Normalized values
    axes[1, 1].hist(normalized_values, bins=200, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title(f'Normalized Values\nRange: [{normalized_values.min():.1f}, {normalized_values.max():.1f}]',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Normalized Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Mean=0', linewidth=2)
    axes[1, 1].axvline(-1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axes[1, 1].axvline(1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim(-25, 25)

    # Normalized values (zoomed to central region)
    central_mask = (normalized_values > -5) & (normalized_values < 5)
    axes[1, 2].hist(normalized_values[central_mask], bins=100, alpha=0.7,
                    color='orange', edgecolor='black')
    pct_central = 100 * np.sum(central_mask) / len(normalized_values)
    axes[1, 2].set_title(f'Normalized (Zoomed)\n{pct_central:.2f}% within [-5, 5]',
                         fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Normalized Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(0, color='red', linestyle='--', label='Mean=0', linewidth=2)
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig('orth_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: orth_distribution_analysis.png")

    # Create a second figure for CDF
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    sorted_norm = np.sort(normalized_values)
    cdf = np.arange(1, len(sorted_norm) + 1) / len(sorted_norm)
    ax.plot(sorted_norm, cdf, linewidth=2)
    ax.set_xlabel('Normalized Value', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('CDF of Normalized Orth Values\n(Shows what % of data falls below each value)',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', label='Mean', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(-20, color='orange', linestyle='--', alpha=0.7, label='Range bounds')
    ax.axvline(20, color='orange', linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlim(-25, 25)

    plt.tight_layout()
    plt.savefig('orth_cdf.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: orth_cdf.png")


def compare_with_previous_assumption():
    """Compare with the 0.0710 std we previously assumed."""
    print("\n" + "=" * 80)
    print("COMPARISON: COMPUTED (0.0249) vs PREVIOUS (0.0710)")
    print("=" * 80)

    print("\n[SCENARIO 1: Using exact computed std = 0.0249]")
    print("  Range: [-20.08, 20.08]")
    print("  Amplification: 40x")
    print("  Pros: Statistically correct, represents true data variance")
    print("  Cons: Very large range, may cause training instability")

    print("\n[SCENARIO 2: Using previous std = 0.0710]")
    min_norm_2 = (0 - 0.5) / 0.0710
    max_norm_2 = (1 - 0.5) / 0.0710
    print(f"  Range: [{min_norm_2:.2f}, {max_norm_2:.2f}]")
    print("  Amplification: 14x")
    print("  Pros: Smaller range than 0.0249")
    print("  Cons: NOT the true std, incorrect statistics")

    print("\n[QUESTION: Where did 0.0710 come from?]")
    print("  Need to check if this was from a different:")
    print("  - Dataset size")
    print("  - Resize resolution")
    print("  - Scaling method")
    print("  - Subset of data")


def main():
    print("\n" + "=" * 80)
    print("DEEP INVESTIGATION: ORTHOGONAL DATA DISTRIBUTION")
    print("=" * 80)
    print("\nGoal: Understand if std=0.0249 is correct and what it means for training")

    # Analyze all training data
    num_samples = 8041  # All training samples

    # Step 1: Raw values
    raw_values = analyze_orth_raw_values(num_samples)

    # Step 2: Scaled values
    scaled_values, mean, std = analyze_orth_scaled_values(num_samples)

    # Step 3: Normalized values
    normalized_values = test_normalization_ranges(scaled_values, mean, std)

    # Step 4: Visualize
    visualize_distributions(raw_values, scaled_values, normalized_values, mean, std)

    # Step 5: Compare
    compare_with_previous_assumption()

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    print(f"\n1. EXACT COMPUTED STD = {std:.6f}")
    print("   This is computed from all 8041 training samples at (416, 544) resolution")
    print("   This represents the TRUE variance of orth data")

    print(f"\n2. NORMALIZED RANGE = [{normalized_values.min():.2f}, {normalized_values.max():.2f}]")
    extreme_pct = 100 * np.sum(np.abs(normalized_values) > 10) / len(normalized_values)
    print(f"   Only {extreme_pct:.4f}% of values exceed |10|")

    if extreme_pct < 0.1:
        print("\n   💡 INTERPRETATION: The large range is from RARE OUTLIERS")
        print("      Most data is concentrated in a small range")
        print("      The std=0.0249 correctly captures the low variance")
        print("      Using this std will amplify the small variations (which might be noise)")
    else:
        print("\n   ⚠️ INTERPRETATION: The large range represents REAL DATA SPREAD")
        print("      Significant portion of data spans the full range")
        print("      The std=0.0249 is correct but data is naturally variable")

    print("\n3. RECOMMENDATION:")
    print("   Option A: Use exact std=0.0249 (statistically correct)")
    print("     - Pros: True to the data")
    print("     - Cons: Large normalized range, potential instability")
    print("     - When: If orth variations are meaningful signal")

    print("\n   Option B: Use larger std (e.g., 0.0710 or 0.1)")
    print("     - Pros: Safer range for training")
    print("     - Cons: Not statistically accurate, may compress important variations")
    print("     - When: If orth variations are mostly noise")

    print("\n   Option C: Don't normalize orth at all")
    print("     - Just keep in [0, 1] range without z-score normalization")
    print("     - When: If the data is already well-distributed")


if __name__ == "__main__":
    main()
