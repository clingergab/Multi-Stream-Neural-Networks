"""
Preprocess the entire SUN RGB-D dataset to generate orthogonal stream arrays.

IMPROVED VERSION:
- Saves as .npy (numpy float32) instead of PNG (uint8)
- Preserves full precision (no quantization loss)
- Faster loading during training

Output structure:
    data/sunrgbd_15/train/orth/00000.npy
    data/sunrgbd_15/train/orth/00001.npy
    ...
"""

import sys
sys.path.insert(0, '.')

import os
import numpy as np
import torch
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from tqdm import tqdm

print("="*80)
print("PREPROCESSING: ORTHOGONAL STREAM GENERATION (v2 - NPY FORMAT)")
print("="*80)


def extract_global_orthogonal_stream(rgb, depth):
    """
    Extract global orthogonal stream for a single image.

    Args:
        rgb: (3, H, W) RGB tensor (normalized)
        depth: (1, H, W) Depth tensor (normalized)

    Returns:
        orth_stream: (H, W) numpy array of orthogonal projection values
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
    orth_vector = Vt[3, :]

    # Project all pixels onto orthogonal vector
    orth_values = X_centered @ orth_vector

    # Reshape back to image shape
    orth_stream = orth_values.reshape(H, W).astype(np.float32)

    return orth_stream


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

    # Remove old PNG files if they exist
    old_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if old_files:
        print(f"Removing {len(old_files)} old PNG files...\n")
        for f in old_files:
            os.remove(os.path.join(output_dir, f))

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
        stats['mean'].append(orth_stream.mean())
        stats['std'].append(orth_stream.std())
        stats['min'].append(orth_stream.min())
        stats['max'].append(orth_stream.max())

        # Save as numpy array (float32)
        output_path = os.path.join(output_dir, f'{idx:05d}.npy')
        np.save(output_path, orth_stream)

    # Print statistics
    print(f"\n{'-'*80}")
    print(f"{split_name.upper()} STATISTICS:")
    print(f"{'-'*80}")
    print(f"Orthogonal stream values:")
    print(f"  Mean: {np.mean(stats['mean']):.6f} ± {np.std(stats['mean']):.6f}")
    print(f"  Std:  {np.mean(stats['std']):.6f} ± {np.std(stats['std']):.6f}")
    print(f"  Min:  {np.mean(stats['min']):.6f} (range: [{np.min(stats['min']):.6f}, {np.max(stats['min']):.6f}])")
    print(f"  Max:  {np.mean(stats['max']):.6f} (range: [{np.min(stats['max']):.6f}, {np.max(stats['max']):.6f}])")
    print(f"\nSaved {len(dataset)} orthogonal stream arrays to {output_dir}")
    print(f"Format: .npy (numpy float32)")

    # Calculate disk usage
    sample_file = os.path.join(output_dir, '00000.npy')
    if os.path.exists(sample_file):
        file_size_kb = os.path.getsize(sample_file) / 1024
        total_size_mb = (file_size_kb * len(dataset)) / 1024
        print(f"File size: ~{file_size_kb:.1f} KB per file")
        print(f"Total size: ~{total_size_mb:.1f} MB for {split_name} split")

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

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        orth_files = sorted([f for f in os.listdir(orth_dir) if f.endswith('.npy')])

        print(f"{split.upper()} split:")
        print(f"  RGB files:   {len(rgb_files)}")
        print(f"  Depth files: {len(depth_files)}")
        print(f"  Orth files:  {len(orth_files)}")

        if len(rgb_files) == len(depth_files) == len(orth_files):
            print(f"  ✅ All splits have matching counts!")
        else:
            print(f"  ⚠️  Mismatch in file counts!")

        if len(orth_files) > 0:
            print(f"  Sample files: {orth_files[:3]}")

            # Load and check first file
            first_file = os.path.join(orth_dir, orth_files[0])
            orth_array = np.load(first_file)
            print(f"  Array shape: {orth_array.shape}")
            print(f"  Array dtype: {orth_array.dtype}")

        print()


def main():
    """Main preprocessing pipeline."""
    print("\nThis script will generate orthogonal stream arrays for the entire dataset.")
    print("Format: NumPy float32 arrays (.npy)")
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

    print("Generated orthogonal stream arrays:")
    print("  data/sunrgbd_15/train/orth/  (train split)")
    print("  data/sunrgbd_15/val/orth/    (val split)")

    print("\nFormat: .npy (NumPy float32)")
    print("Advantages over PNG:")
    print("  ✅ No quantization loss (preserves full precision)")
    print("  ✅ Faster loading with np.load()")
    print("  ✅ Exact values preserved")

    print("\nNext steps:")
    print("  1. Update SUNRGBDDataset to load .npy files")
    print("  2. Update LINet to accept 3 streams")
    print("  3. Train and evaluate!")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
