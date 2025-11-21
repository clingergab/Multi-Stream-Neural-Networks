"""
Compute exact mean and std statistics from TRAIN set (after scaling).
This will verify the precomputed values using a memory-efficient approach.
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def compute_train_statistics():
    """Compute mean and std for RGB, Depth, Orth on TRAIN set using online algorithm."""

    data_root = 'data/sunrgbd_15'
    split_dir = os.path.join(data_root, 'train')
    rgb_dir = os.path.join(split_dir, 'rgb')
    depth_dir = os.path.join(split_dir, 'depth')
    orth_dir = os.path.join(split_dir, 'orth')

    # Get number of samples
    labels_file = os.path.join(split_dir, 'labels.txt')
    with open(labels_file, 'r') as f:
        num_samples = len(f.readlines())

    print(f"Computing statistics on {num_samples} training samples...")
    print("Using memory-efficient two-pass algorithm...\n")

    target_size = (416, 544)  # H, W
    total_pixels = num_samples * target_size[0] * target_size[1]

    # PASS 1: Compute means
    print("Pass 1: Computing means...")
    rgb_sum = np.zeros(3, dtype=np.float64)
    depth_sum = 0.0
    orth_sum = 0.0

    for idx in tqdm(range(num_samples), desc="Computing means"):
        # Load and process RGB
        rgb_path = os.path.join(rgb_dir, f'{idx:05d}.png')
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = rgb.resize((target_size[1], target_size[0]), Image.BILINEAR)
        rgb_array = np.array(rgb, dtype=np.float32) / 255.0  # Scale to [0,1]
        rgb_sum += rgb_array.sum(axis=(0, 1))

        # Load and process Depth
        depth_path = os.path.join(depth_dir, f'{idx:05d}.png')
        depth = Image.open(depth_path)
        if depth.mode in ('I', 'I;16', 'I;16B'):
            depth_arr = np.array(depth, dtype=np.float32)
            depth_arr = np.clip(depth_arr / 65535.0, 0.0, 1.0)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            depth = depth.convert('F')
            if np.array(depth).max() > 1.0:
                depth_arr = np.array(depth)
                depth = Image.fromarray(depth_arr / 255.0, mode='F')
        depth = depth.resize((target_size[1], target_size[0]), Image.BILINEAR)
        depth_array = np.array(depth, dtype=np.float32)
        depth_sum += depth_array.sum()

        # Load and process Orth
        orth_path = os.path.join(orth_dir, f'{idx:05d}.png')
        orth = Image.open(orth_path)
        if orth.mode in ('I', 'I;16', 'I;16B'):
            orth_arr = np.array(orth, dtype=np.float32)
            orth_arr = np.clip(orth_arr / 65535.0, 0.0, 1.0)
            orth = Image.fromarray(orth_arr, mode='F')
        else:
            orth = orth.convert('F')
            if np.array(orth).max() > 1.0:
                orth_arr = np.array(orth)
                orth = Image.fromarray(orth_arr / 255.0, mode='F')
        orth = orth.resize((target_size[1], target_size[0]), Image.BILINEAR)
        orth_array = np.array(orth, dtype=np.float32)
        orth_sum += orth_array.sum()

    rgb_mean = rgb_sum / total_pixels
    depth_mean = depth_sum / total_pixels
    orth_mean = orth_sum / total_pixels

    print(f"\nMeans computed:")
    print(f"  RGB:   [{rgb_mean[0]:.6f}, {rgb_mean[1]:.6f}, {rgb_mean[2]:.6f}]")
    print(f"  Depth: {depth_mean:.6f}")
    print(f"  Orth:  {orth_mean:.6f}")

    # PASS 2: Compute standard deviations
    print("\nPass 2: Computing standard deviations...")
    rgb_sq_diff_sum = np.zeros(3, dtype=np.float64)
    depth_sq_diff_sum = 0.0
    orth_sq_diff_sum = 0.0

    for idx in tqdm(range(num_samples), desc="Computing stds"):
        # Load and process RGB
        rgb_path = os.path.join(rgb_dir, f'{idx:05d}.png')
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = rgb.resize((target_size[1], target_size[0]), Image.BILINEAR)
        rgb_array = np.array(rgb, dtype=np.float32) / 255.0  # Scale to [0,1]
        for c in range(3):
            rgb_sq_diff_sum[c] += ((rgb_array[:, :, c] - rgb_mean[c]) ** 2).sum()

        # Load and process Depth
        depth_path = os.path.join(depth_dir, f'{idx:05d}.png')
        depth = Image.open(depth_path)
        if depth.mode in ('I', 'I;16', 'I;16B'):
            depth_arr = np.array(depth, dtype=np.float32)
            depth_arr = np.clip(depth_arr / 65535.0, 0.0, 1.0)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            depth = depth.convert('F')
            if np.array(depth).max() > 1.0:
                depth_arr = np.array(depth)
                depth = Image.fromarray(depth_arr / 255.0, mode='F')
        depth = depth.resize((target_size[1], target_size[0]), Image.BILINEAR)
        depth_array = np.array(depth, dtype=np.float32)
        depth_sq_diff_sum += ((depth_array - depth_mean) ** 2).sum()

        # Load and process Orth
        orth_path = os.path.join(orth_dir, f'{idx:05d}.png')
        orth = Image.open(orth_path)
        if orth.mode in ('I', 'I;16', 'I;16B'):
            orth_arr = np.array(orth, dtype=np.float32)
            orth_arr = np.clip(orth_arr / 65535.0, 0.0, 1.0)
            orth = Image.fromarray(orth_arr, mode='F')
        else:
            orth = orth.convert('F')
            if np.array(orth).max() > 1.0:
                orth_arr = np.array(orth)
                orth = Image.fromarray(orth_arr / 255.0, mode='F')
        orth = orth.resize((target_size[1], target_size[0]), Image.BILINEAR)
        orth_array = np.array(orth, dtype=np.float32)
        orth_sq_diff_sum += ((orth_array - orth_mean) ** 2).sum()

    rgb_std = np.sqrt(rgb_sq_diff_sum / total_pixels)
    depth_std = np.sqrt(depth_sq_diff_sum / total_pixels)
    orth_std = np.sqrt(orth_sq_diff_sum / total_pixels)

    print("\n" + "=" * 80)
    print("COMPUTED TRAIN SET STATISTICS (Range [0, 1], Resized to (416, 544))")
    print("=" * 80)

    print(f"\nRGB Mean: [{rgb_mean[0]}, {rgb_mean[1]}, {rgb_mean[2]}]")
    print(f"RGB Std:  [{rgb_std[0]}, {rgb_std[1]}, {rgb_std[2]}]")

    print(f"\nDepth Mean: {depth_mean:.4f}")
    print(f"Depth Std:  {depth_std:.4f}")

    print(f"\nOrth Mean:  {orth_mean:.4f}")
    print(f"Orth Std:   {orth_std:.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON WITH YOUR PRECOMPUTED VALUES")
    print("=" * 80)

    EXPECTED_RGB_MEAN = np.array([0.47004103660583496, 0.4392580986022949, 0.4210551679134369])
    EXPECTED_RGB_STD = np.array([0.2732377350330353, 0.2812955975532532, 0.28407758474349976])
    EXPECTED_DEPTH_MEAN = 0.3108
    EXPECTED_DEPTH_STD = 0.1629
    EXPECTED_ORTH_MEAN = 0.5001
    EXPECTED_ORTH_STD = 0.0710

    print(f"\nRGB Mean Error:  {np.abs(rgb_mean - EXPECTED_RGB_MEAN).max():.8f}")
    print(f"RGB Std Error:   {np.abs(rgb_std - EXPECTED_RGB_STD).max():.8f}")
    print(f"Depth Mean Error: {abs(depth_mean - EXPECTED_DEPTH_MEAN):.8f}")
    print(f"Depth Std Error:  {abs(depth_std - EXPECTED_DEPTH_STD):.8f}")
    print(f"Orth Mean Error:  {abs(orth_mean - EXPECTED_ORTH_MEAN):.8f}")
    print(f"Orth Std Error:   {abs(orth_std - EXPECTED_ORTH_STD):.8f}")


if __name__ == "__main__":
    compute_train_statistics()
