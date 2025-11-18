"""
Clean preprocessing script for orthogonal stream generation.

CRITICAL: This script loads raw images directly WITHOUT using SUNRGBDDataset
to avoid any augmentation. The preprocessed orthogonal streams must correspond
exactly to the raw RGB/Depth images.
"""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_raw_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image exactly like SUNRGBDDataset does,
    but WITHOUT any augmentation (no flip, no crop, no color jitter).
    """
    # Load image
    img = Image.open(image_path)

    # Resize to target size
    img = transforms.functional.resize(img, target_size)

    return img


def preprocess_rgb(rgb_pil):
    """Convert RGB PIL image to normalized tensor."""
    rgb_tensor = transforms.functional.to_tensor(rgb_pil)
    rgb_tensor = transforms.functional.normalize(
        rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return rgb_tensor


def preprocess_depth(depth_pil):
    """Convert depth PIL image to normalized tensor."""
    # Convert depth to grayscale/float format (same as SUNRGBDDataset)
    if depth_pil.mode in ('I', 'I;16', 'I;16B'):
        depth_array = np.array(depth_pil, dtype=np.float32)
        if depth_array.max() > 0:
            depth_array = (depth_array / depth_array.max() * 255).astype(np.uint8)
        else:
            depth_array = depth_array.astype(np.uint8)
        depth_pil = Image.fromarray(depth_array, mode='L')
    elif depth_pil.mode == 'RGB':
        depth_pil = depth_pil.convert('L')
    elif depth_pil.mode != 'L':
        depth_pil = depth_pil.convert('L')

    # To tensor and normalize
    depth_tensor = transforms.functional.to_tensor(depth_pil)
    depth_tensor = transforms.functional.normalize(
        depth_tensor, mean=[0.5027], std=[0.2197]
    )
    return depth_tensor


def extract_global_orthogonal_stream(rgb_tensor, depth_tensor):
    """
    Extract global orthogonal stream from RGB and Depth tensors.

    Args:
        rgb_tensor: (3, H, W) normalized RGB tensor
        depth_tensor: (1, H, W) normalized depth tensor

    Returns:
        orth_stream: (H, W) float32 array of orthogonal values
    """
    # Denormalize to [0, 1] range
    rgb_denorm = rgb_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth_tensor * 0.2197 + 0.5027

    # Flatten to get all pixels
    H, W = rgb_tensor.shape[1], rgb_tensor.shape[2]
    r = rgb_denorm[0].flatten().numpy()
    g = rgb_denorm[1].flatten().numpy()
    b = rgb_denorm[2].flatten().numpy()
    d = depth_denorm[0].flatten().numpy()

    # Stack into (N_pixels, 4) matrix
    X = np.stack([r, g, b, d], axis=1)

    # Center data
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # SVD to find hyperplane
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Orthogonal vector (4th singular vector - smallest variance)
    orth_vector = Vt[3, :]

    # Project all pixels onto orthogonal vector
    orth_values = X_centered @ orth_vector

    # Reshape back to image shape
    orth_stream = orth_values.reshape(H, W).astype(np.float32)

    return orth_stream


def analyze_value_range(data_root, split, num_samples=100):
    """Analyze value range across dataset to determine normalization."""
    print(f"\nAnalyzing value range on {num_samples} {split} samples...")

    rgb_dir = os.path.join(data_root, split, 'rgb')
    depth_dir = os.path.join(data_root, split, 'depth')

    # Get list of files
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])
    num_samples = min(num_samples, len(rgb_files))

    all_mins = []
    all_maxs = []

    for i in tqdm(range(num_samples), desc="Analyzing"):
        # Load raw images
        rgb_path = os.path.join(rgb_dir, rgb_files[i])
        depth_path = os.path.join(depth_dir, rgb_files[i].replace('.jpg', '.png'))

        rgb_pil = load_raw_image(rgb_path)
        depth_pil = load_raw_image(depth_path)

        # Preprocess
        rgb_tensor = preprocess_rgb(rgb_pil)
        depth_tensor = preprocess_depth(depth_pil)

        # Compute orthogonal
        orth = extract_global_orthogonal_stream(rgb_tensor, depth_tensor)

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
    """Normalize to uint16 (0-65535) for 16-bit PNG."""
    values = np.clip(orth_stream, vmin, vmax)
    values_normalized = (values - vmin) / (vmax - vmin)
    values_uint16 = (values_normalized * 65535).astype(np.uint16)
    return values_uint16


def process_split(data_root, split, vmin, vmax):
    """Process train or val split."""
    print(f"\n{'='*80}")
    print(f"Processing {split.upper()} split")
    print(f"{'='*80}\n")

    rgb_dir = os.path.join(data_root, split, 'rgb')
    depth_dir = os.path.join(data_root, split, 'depth')
    output_dir = os.path.join(data_root, split, 'orth')

    os.makedirs(output_dir, exist_ok=True)

    # Get list of files
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])

    print(f"Found {len(rgb_files)} images")
    print(f"Normalization range: [{vmin:.6f}, {vmax:.6f}]\n")

    # Remove old files
    for ext in ['.npy', '.png']:
        old_files = [f for f in os.listdir(output_dir) if f.endswith(ext)]
        if old_files:
            print(f"Removing {len(old_files)} old {ext} files...\n")
            for f in old_files:
                os.remove(os.path.join(output_dir, f))

    print(f"Generating orthogonal streams...\n")

    stats = {'mean': [], 'std': [], 'min': [], 'max': [], 'clipped': 0}

    for idx, rgb_file in enumerate(tqdm(rgb_files, desc=split)):
        # Load raw images
        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_path = os.path.join(depth_dir, rgb_file.replace('.jpg', '.png'))

        rgb_pil = load_raw_image(rgb_path)
        depth_pil = load_raw_image(depth_path)

        # Preprocess
        rgb_tensor = preprocess_rgb(rgb_pil)
        depth_tensor = preprocess_depth(depth_pil)

        # Compute orthogonal
        orth_stream = extract_global_orthogonal_stream(rgb_tensor, depth_tensor)

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
    print(f"{split.upper()} STATISTICS:")
    print(f"{'-'*80}")
    print(f"Orthogonal values (before normalization):")
    print(f"  Mean: {np.mean(stats['mean']):.6f} ± {np.std(stats['mean']):.6f}")
    print(f"  Std:  {np.mean(stats['std']):.6f} ± {np.std(stats['std']):.6f}")
    print(f"  Min:  {np.mean(stats['min']):.6f} (range: [{np.min(stats['min']):.6f}, {np.max(stats['min']):.6f}])")
    print(f"  Max:  {np.mean(stats['max']):.6f} (range: [{np.min(stats['max']):.6f}, {np.max(stats['max']):.6f}])")
    print(f"  Clipped: {stats['clipped']}/{len(rgb_files)} images ({100*stats['clipped']/len(rgb_files):.1f}%)")

    sample_file = os.path.join(output_dir, '00000.png')
    if os.path.exists(sample_file):
        file_size_kb = os.path.getsize(sample_file) / 1024
        total_size_mb = (file_size_kb * len(rgb_files)) / 1024
        print(f"\nFile size: ~{file_size_kb:.1f} KB per file")
        print(f"Total size: ~{total_size_mb:.1f} MB for {split} split")

    print(f"{'-'*80}\n")

    return stats


def main():
    print("="*80)
    print("CLEAN ORTHOGONAL STREAM PREPROCESSING")
    print("="*80)
    print("\nThis script loads RAW images directly (NO augmentation)")
    print("to ensure orthogonal streams match the actual RGB/Depth data.\n")

    data_root = 'data/sunrgbd_15'

    # Step 1: Analyze value range on training set
    vmin, vmax = analyze_value_range(data_root, 'train', num_samples=100)

    # Save normalization params
    norm_params_file = os.path.join(data_root, 'orth_normalization.txt')
    with open(norm_params_file, 'w') as f:
        f.write(f"{vmin}\n{vmax}\n")
    print(f"\n✓ Saved normalization params to {norm_params_file}")

    # Step 2: Process both splits
    train_stats = process_split(data_root, 'train', vmin, vmax)
    val_stats = process_split(data_root, 'val', vmin, vmax)

    # Final summary
    print(f"{'='*80}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*80}\n")

    print("Generated orthogonal stream images:")
    print(f"  {data_root}/train/orth/ (train split)")
    print(f"  {data_root}/val/orth/   (val split)")
    print(f"\nFormat: 16-bit PNG")
    print(f"Normalization: [{vmin:.6f}, {vmax:.6f}]")
    print(f"Precision: 65536 levels\n")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
