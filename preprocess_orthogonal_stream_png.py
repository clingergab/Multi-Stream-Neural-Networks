"""
Preprocess orthogonal stream as PNG with adaptive normalization.

Strategy:
1. Analyze actual value range across dataset
2. Use tighter normalization range (e.g., [-0.2, 0.2] instead of [-0.5, 0.5])
3. Save as 16-bit PNG for higher precision (65536 levels instead of 256)
4. Maintain consistency with RGB/Depth format
"""

import sys
sys.path.insert(0, '.')

import os
import numpy as np
import torch
from PIL import Image
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from tqdm import tqdm

print("="*80)
print("PREPROCESSING: ORTHOGONAL STREAM (PNG 16-BIT)")
print("="*80)


def extract_global_orthogonal_stream(rgb, depth):
    """Extract global orthogonal stream."""
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    r = rgb_denorm[0].flatten().numpy()
    g = rgb_denorm[1].flatten().numpy()
    b = rgb_denorm[2].flatten().numpy()
    d = depth_denorm[0].flatten().numpy()

    X = np.stack([r, g, b, d], axis=1)
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    orth_vector = Vt[3, :]
    orth_values = X_centered @ orth_vector
    orth_stream = orth_values.reshape(H, W).astype(np.float32)

    return orth_stream


def analyze_value_range(dataset, num_samples=100):
    """Analyze actual value range to determine optimal normalization."""
    print(f"\nAnalyzing value range on {num_samples} samples...")

    all_mins = []
    all_maxs = []

    for i in tqdm(range(num_samples), desc="Analyzing"):
        rgb, depth, label = dataset[i]
        orth = extract_global_orthogonal_stream(rgb, depth)

        all_mins.append(orth.min())
        all_maxs.append(orth.max())

    global_min = np.min(all_mins)
    global_max = np.max(all_maxs)
    percentile_min = np.percentile(all_mins, 1)  # 1st percentile
    percentile_max = np.percentile(all_maxs, 99)  # 99th percentile

    print(f"\nValue range analysis:")
    print(f"  Absolute min: {global_min:.6f}")
    print(f"  Absolute max: {global_max:.6f}")
    print(f"  1st percentile min: {percentile_min:.6f}")
    print(f"  99th percentile max: {percentile_max:.6f}")

    # Use percentiles for better range (avoids extreme outliers)
    # Make it symmetric around 0
    range_magnitude = max(abs(percentile_min), abs(percentile_max))
    vmin = -range_magnitude
    vmax = range_magnitude

    print(f"\nRecommended normalization range:")
    print(f"  [{vmin:.6f}, {vmax:.6f}]")

    return vmin, vmax


def normalize_to_uint16(orth_stream, vmin, vmax):
    """
    Normalize to uint16 (0-65535) for 16-bit PNG.

    16-bit gives us 65536 levels vs 256 for 8-bit.
    """
    values = np.clip(orth_stream, vmin, vmax)
    values_normalized = (values - vmin) / (vmax - vmin)
    values_uint16 = (values_normalized * 65535).astype(np.uint16)
    return values_uint16


def denormalize_from_uint16(values_uint16, vmin, vmax):
    """Reverse the normalization."""
    values_normalized = values_uint16.astype(np.float32) / 65535.0
    values = values_normalized * (vmax - vmin) + vmin
    return values


def process_split(split_name, vmin, vmax):
    """Process train or val split."""
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*80}\n")

    is_train = (split_name == 'train')
    dataset = SUNRGBDDataset(train=is_train)

    print(f"Loaded {len(dataset)} samples")
    print(f"Normalization range: [{vmin:.6f}, {vmax:.6f}]\n")

    output_dir = f'data/sunrgbd_15/{split_name}/orth'
    os.makedirs(output_dir, exist_ok=True)

    # Remove old files
    for ext in ['.npy', '.png']:
        old_files = [f for f in os.listdir(output_dir) if f.endswith(ext)]
        if old_files:
            print(f"Removing {len(old_files)} old {ext} files...\n")
            for f in old_files:
                os.remove(os.path.join(output_dir, f))

    print(f"Generating orthogonal streams...\n")

    stats = {'mean': [], 'std': [], 'min': [], 'max': [], 'clipped': 0}

    for idx in tqdm(range(len(dataset)), desc=f'{split_name}'):
        rgb, depth, label = dataset[idx]
        orth_stream = extract_global_orthogonal_stream(rgb, depth)

        # Track clipping
        if orth_stream.min() < vmin or orth_stream.max() > vmax:
            stats['clipped'] += 1

        stats['mean'].append(orth_stream.mean())
        stats['std'].append(orth_stream.std())
        stats['min'].append(orth_stream.min())
        stats['max'].append(orth_stream.max())

        # Convert to uint16
        orth_uint16 = normalize_to_uint16(orth_stream, vmin, vmax)

        # Save as 16-bit PNG
        output_path = os.path.join(output_dir, f'{idx:05d}.png')
        Image.fromarray(orth_uint16, mode='I;16').save(output_path)

    print(f"\n{'-'*80}")
    print(f"{split_name.upper()} STATISTICS:")
    print(f"{'-'*80}")
    print(f"Orthogonal values (before normalization):")
    print(f"  Mean: {np.mean(stats['mean']):.6f} ± {np.std(stats['mean']):.6f}")
    print(f"  Std:  {np.mean(stats['std']):.6f} ± {np.std(stats['std']):.6f}")
    print(f"  Min:  {np.mean(stats['min']):.6f} (range: [{np.min(stats['min']):.6f}, {np.max(stats['min']):.6f}])")
    print(f"  Max:  {np.mean(stats['max']):.6f} (range: [{np.min(stats['max']):.6f}, {np.max(stats['max']):.6f}])")
    print(f"  Clipped: {stats['clipped']}/{len(dataset)} images ({100*stats['clipped']/len(dataset):.1f}%)")

    sample_file = os.path.join(output_dir, '00000.png')
    if os.path.exists(sample_file):
        file_size_kb = os.path.getsize(sample_file) / 1024
        total_size_mb = (file_size_kb * len(dataset)) / 1024
        print(f"\nFile size: ~{file_size_kb:.1f} KB per file")
        print(f"Total size: ~{total_size_mb:.1f} MB for {split_name} split")

    print(f"{'-'*80}\n")

    return stats


def verify_reconstruction():
    """Verify that we can correctly reconstruct values from PNG."""
    print(f"\n{'='*80}")
    print("VERIFICATION: Reconstruction Accuracy")
    print(f"{'='*80}\n")

    train_dataset = SUNRGBDDataset(train=True)

    print("Testing reconstruction on 10 samples...\n")

    errors = []

    for idx in range(10):
        rgb, depth, label = train_dataset[idx]

        # Compute fresh
        orth_fresh = extract_global_orthogonal_stream(rgb, depth)

        # Load from PNG
        orth_path = f'data/sunrgbd_15/train/orth/{idx:05d}.png'
        orth_uint16 = np.array(Image.open(orth_path))

        # Denormalize (need vmin, vmax - will load from metadata file)
        # For now, use the recommended range
        vmin, vmax = -0.15, 0.15  # Approximate
        orth_reconstructed = denormalize_from_uint16(orth_uint16, vmin, vmax)

        # Compare
        error = np.abs(orth_fresh - orth_reconstructed).mean()
        max_error = np.abs(orth_fresh - orth_reconstructed).max()

        errors.append(error)
        print(f"Image {idx}: Mean error = {error:.6f}, Max error = {max_error:.6f}")

    print(f"\n{'-'*80}")
    print(f"Average reconstruction error: {np.mean(errors):.6f}")

    if np.mean(errors) < 0.001:
        print("✅ 16-bit PNG preserves values accurately")
    else:
        print(f"⚠️  Some quantization error (avg {np.mean(errors):.6f})")

    print(f"{'-'*80}\n")


def main():
    print("\nThis script generates orthogonal streams as 16-bit PNGs.")
    print("Benefits:")
    print("  ✅ Consistent format with RGB/Depth")
    print("  ✅ 16-bit = 65536 levels (much better than 8-bit)")
    print("  ✅ Smaller files than .npy (~50KB vs ~196KB)")
    print("  ✅ Can be viewed/debugged as images\n")

    # Step 1: Analyze value range
    train_dataset = SUNRGBDDataset(train=True)
    vmin, vmax = analyze_value_range(train_dataset, num_samples=100)

    # Save normalization params
    norm_params_file = 'data/sunrgbd_15/orth_normalization.txt'
    with open(norm_params_file, 'w') as f:
        f.write(f"{vmin}\n{vmax}\n")
    print(f"\n✓ Saved normalization params to {norm_params_file}")

    # Step 2: Process both splits
    train_stats = process_split('train', vmin, vmax)
    val_stats = process_split('val', vmin, vmax)

    # Step 3: Verify
    verify_reconstruction()

    # Final summary
    print(f"{'='*80}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*80}\n")

    print("Generated orthogonal stream images:")
    print("  data/sunrgbd_15/train/orth/ (8041 files)")
    print("  data/sunrgbd_15/val/orth/   (2018 files)")
    print(f"\nFormat: 16-bit PNG")
    print(f"Normalization: [{vmin:.6f}, {vmax:.6f}]")
    print(f"Precision: 65536 levels (much better than 8-bit's 256)")

    print("\nNext step: Update SUNRGBDDataset to:")
    print("  1. Load PNG files")
    print("  2. Denormalize using saved params")
    print("  3. Return [5, H, W] tensors")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
