"""
Preprocess the entire SUN RGB-D dataset to generate orthogonal stream images.

This script:
1. Loads RGB and Depth images for each sample
2. Computes global orthogonal stream using SVD
3. Saves orthogonal stream as PNG images

Output structure:
    data/sunrgbd_15/train/orth/00000.png
    data/sunrgbd_15/train/orth/00001.png
    ...
    data/sunrgbd_15/val/orth/00000.png
    data/sunrgbd_15/val/orth/00001.png
    ...
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
print("PREPROCESSING: ORTHOGONAL STREAM GENERATION")
print("="*80)


def extract_global_orthogonal_stream(rgb, depth):
    """
    Extract global orthogonal stream for a single image.

    Args:
        rgb: (3, H, W) RGB tensor (normalized)
        depth: (1, H, W) Depth tensor (normalized)

    Returns:
        orth_stream: (1, H, W) orthogonal projection values
    """
    # Denormalize to [0, 1] range
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    # Flatten to get all pixels
    H, W = rgb.shape[1], rgb.shape[2]
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
    orth_vector = Vt[3, :]  # Shape: (4,)

    # Project all pixels onto orthogonal vector
    orth_values = X_centered @ orth_vector  # Shape: (N_pixels,)

    # Reshape back to image shape
    orth_stream = orth_values.reshape(H, W)

    # Convert to tensor
    orth_stream = torch.from_numpy(orth_stream).float().unsqueeze(0)  # (1, H, W)

    return orth_stream


def normalize_to_uint8(orth_stream):
    """
    Normalize orthogonal stream values to [0, 255] range for PNG storage.

    We'll use a consistent normalization range across the dataset to preserve
    relative magnitudes between images.
    """
    # Get the tensor values
    values = orth_stream.squeeze().numpy()

    # Use symmetric range around 0 (since orthogonal values are centered)
    # Based on our analysis, values typically range from -0.3 to +0.3
    # We'll use a slightly larger range to avoid clipping
    vmin, vmax = -0.5, 0.5

    # Normalize to [0, 255]
    values_normalized = (values - vmin) / (vmax - vmin)
    values_normalized = np.clip(values_normalized, 0, 1)
    values_uint8 = (values_normalized * 255).astype(np.uint8)

    return values_uint8


def process_split(split_name='train'):
    """Process train or val split."""
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*80}\n")

    # Load dataset
    is_train = (split_name == 'train')
    dataset = SUNRGBDDataset(train=is_train)

    print(f"Loaded {len(dataset)} samples\n")

    # Create output directory
    output_dir = f'data/sunrgbd_15/{split_name}/orth'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Process each image
    print(f"Generating orthogonal streams...\n")

    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }

    for idx in tqdm(range(len(dataset)), desc=f'{split_name}'):
        # Load RGB and Depth
        rgb, depth, label = dataset[idx]

        # Extract orthogonal stream
        orth_stream = extract_global_orthogonal_stream(rgb, depth)

        # Collect statistics
        stats['mean'].append(orth_stream.mean().item())
        stats['std'].append(orth_stream.std().item())
        stats['min'].append(orth_stream.min().item())
        stats['max'].append(orth_stream.max().item())

        # Normalize to uint8 for PNG storage
        orth_uint8 = normalize_to_uint8(orth_stream)

        # Save as PNG (single-channel grayscale)
        output_path = os.path.join(output_dir, f'{idx:05d}.png')
        Image.fromarray(orth_uint8, mode='L').save(output_path)

    # Print statistics
    print(f"\n{'-'*80}")
    print(f"{split_name.upper()} STATISTICS:")
    print(f"{'-'*80}")
    print(f"Orthogonal stream values (before normalization):")
    print(f"  Mean: {np.mean(stats['mean']):.6f} ± {np.std(stats['mean']):.6f}")
    print(f"  Std:  {np.mean(stats['std']):.6f} ± {np.std(stats['std']):.6f}")
    print(f"  Min:  {np.mean(stats['min']):.6f} (range: [{np.min(stats['min']):.6f}, {np.max(stats['min']):.6f}])")
    print(f"  Max:  {np.mean(stats['max']):.6f} (range: [{np.min(stats['max']):.6f}, {np.max(stats['max']):.6f}])")
    print(f"\nSaved {len(dataset)} orthogonal stream images to {output_dir}")
    print(f"{'-'*80}\n")

    return stats


def verify_output():
    """Verify that output files match RGB/Depth structure."""
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}\n")

    for split in ['train', 'val']:
        rgb_dir = f'data/sunrgbd_15/{split}/rgb'
        depth_dir = f'data/sunrgbd_15/{split}/depth'
        orth_dir = f'data/sunrgbd_15/{split}/orth'

        rgb_files = sorted(os.listdir(rgb_dir)) if os.path.exists(rgb_dir) else []
        depth_files = sorted(os.listdir(depth_dir)) if os.path.exists(depth_dir) else []
        orth_files = sorted(os.listdir(orth_dir)) if os.path.exists(orth_dir) else []

        # Filter out non-PNG files
        rgb_files = [f for f in rgb_files if f.endswith('.png') or f.endswith('.jpg')]
        depth_files = [f for f in depth_files if f.endswith('.png')]
        orth_files = [f for f in orth_files if f.endswith('.png')]

        print(f"{split.upper()} split:")
        print(f"  RGB files:   {len(rgb_files)}")
        print(f"  Depth files: {len(depth_files)}")
        print(f"  Orth files:  {len(orth_files)}")

        if len(rgb_files) == len(depth_files) == len(orth_files):
            print(f"  ✅ All splits have matching counts!")
        else:
            print(f"  ⚠️  Mismatch in file counts!")

        # Check that filenames match (at least the first few)
        if len(orth_files) > 0:
            print(f"  Sample files: {orth_files[:3]}")

        print()


def main():
    """Main preprocessing pipeline."""
    print("\nThis script will generate orthogonal stream images for the entire dataset.")
    print("Estimated time: ~2-3 minutes\n")

    # Process both splits
    train_stats = process_split('train')
    val_stats = process_split('val')

    # Verify output
    verify_output()

    # Final summary
    print(f"{'='*80}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*80}\n")

    print("Generated orthogonal stream images:")
    print("  data/sunrgbd_15/train/orth/  (train split)")
    print("  data/sunrgbd_15/val/orth/    (val split)")

    print("\nNormalization range used: [-0.5, 0.5] → [0, 255]")
    print("This range was chosen to capture most orthogonal values without clipping.")

    print("\nNext steps:")
    print("  1. Update SUNRGBDDataset to load orthogonal streams from disk")
    print("  2. Update LINet to accept 3 streams")
    print("  3. Train and evaluate!")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
