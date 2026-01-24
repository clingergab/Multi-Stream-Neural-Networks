"""
Benchmark CPU augmentation transforms only (no disk I/O).

Creates synthetic images and measures pure CPU transform time.
"""

import os
import sys
import time
import statistics
import multiprocessing

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def create_synthetic_images(height=416, width=544):
    """Create synthetic RGB and Depth images."""
    # Random RGB image
    rgb_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    rgb = Image.fromarray(rgb_array, mode='RGB')

    # Random Depth image (float32 in [0, 1])
    depth_array = np.random.rand(height, width).astype(np.float32)
    depth = Image.fromarray(depth_array, mode='F')

    return rgb, depth


def apply_full_augmentation(rgb, depth, target_size=(416, 544)):
    """
    Full CPU augmentation pipeline (normalize=True).

    This replicates what SUNRGBDDataset does with normalize=True.
    """
    # 1. Synchronized Random Horizontal Flip (50%)
    if np.random.random() < 0.5:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Synchronized Random Resized Crop (50%)
    if np.random.random() < 0.5:
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            rgb, scale=(0.9, 1.0), ratio=(0.95, 1.05)
        )
        rgb = transforms.functional.resized_crop(rgb, i, j, h, w, target_size)
        depth = transforms.functional.resized_crop(depth, i, j, h, w, target_size)
    else:
        rgb = transforms.functional.resize(rgb, target_size)
        depth = transforms.functional.resize(depth, target_size)

    # 3. ColorJitter (43%)
    if np.random.random() < 0.43:
        color_jitter = transforms.ColorJitter(
            brightness=0.37,
            contrast=0.37,
            saturation=0.37,
            hue=0.11
        )
        rgb = color_jitter(rgb)

    # 4. Gaussian Blur (25%)
    if np.random.random() < 0.25:
        kernel_size = int(np.random.choice([3, 5, 7]))
        sigma = float(np.random.uniform(0.1, 1.7))
        rgb = transforms.functional.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

    # 5. Grayscale (17%)
    if np.random.random() < 0.17:
        rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)

    # 6. Depth augmentation (50%)
    if np.random.random() < 0.5:
        depth_array = np.array(depth, dtype=np.float32)
        brightness_factor = np.random.uniform(0.75, 1.25)
        contrast_factor = np.random.uniform(0.75, 1.25)
        depth_array = (depth_array - 0.5) * contrast_factor + 0.5
        depth_array = depth_array * brightness_factor
        noise = np.random.normal(0, 0.059, depth_array.shape).astype(np.float32)
        depth_array = depth_array + noise
        depth_array = np.clip(depth_array, 0.0, 1.0).astype(np.float32)
        depth = Image.fromarray(depth_array, mode='F')

    # Convert to tensor
    rgb = transforms.functional.to_tensor(rgb)
    depth = transforms.functional.to_tensor(depth)

    # 7. Normalization
    rgb = transforms.functional.normalize(
        rgb,
        mean=[0.4905626144214781, 0.4564359471868703, 0.43112756716677114],
        std=[0.27944652961530003, 0.2868739703756949, 0.29222326115669395]
    )
    depth = transforms.functional.normalize(
        depth, mean=[0.2912], std=[0.1472]
    )

    # 8. Random Erasing RGB (17%)
    if np.random.random() < 0.17:
        erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.10), ratio=(0.5, 2.0))
        rgb = erasing(rgb)

    # 9. Random Erasing Depth (10%)
    if np.random.random() < 0.1:
        erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.5, 2.0))
        depth = erasing(depth)

    return rgb, depth


def apply_partial_augmentation(rgb, depth, target_size=(416, 544)):
    """
    Partial CPU augmentation pipeline (normalize=False).

    This replicates what SUNRGBDDataset does with normalize=False.
    Skips: ColorJitter, Blur, Grayscale, Normalization, RandomErasing
    """
    # 1. Synchronized Random Horizontal Flip (50%)
    if np.random.random() < 0.5:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Synchronized Random Resized Crop (50%)
    if np.random.random() < 0.5:
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            rgb, scale=(0.9, 1.0), ratio=(0.95, 1.05)
        )
        rgb = transforms.functional.resized_crop(rgb, i, j, h, w, target_size)
        depth = transforms.functional.resized_crop(depth, i, j, h, w, target_size)
    else:
        rgb = transforms.functional.resize(rgb, target_size)
        depth = transforms.functional.resize(depth, target_size)

    # 3-5. SKIPPED: ColorJitter, Blur, Grayscale (done on GPU)

    # 6. Depth augmentation (50%)
    if np.random.random() < 0.5:
        depth_array = np.array(depth, dtype=np.float32)
        brightness_factor = np.random.uniform(0.75, 1.25)
        contrast_factor = np.random.uniform(0.75, 1.25)
        depth_array = (depth_array - 0.5) * contrast_factor + 0.5
        depth_array = depth_array * brightness_factor
        noise = np.random.normal(0, 0.059, depth_array.shape).astype(np.float32)
        depth_array = depth_array + noise
        depth_array = np.clip(depth_array, 0.0, 1.0).astype(np.float32)
        depth = Image.fromarray(depth_array, mode='F')

    # Convert to tensor
    rgb = transforms.functional.to_tensor(rgb)
    depth = transforms.functional.to_tensor(depth)

    # 7. SKIPPED: Normalization (done on GPU)
    # 8-9. SKIPPED: Random Erasing (done on GPU)

    return rgb, depth


def benchmark_transforms(num_iterations=500):
    """Benchmark transform pipelines."""
    print("=" * 70)
    print("CPU AUGMENTATION TRANSFORMS BENCHMARK (No Disk I/O)")
    print("=" * 70)
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"Image size: 416 x 544")
    print(f"Iterations: {num_iterations}")
    print()

    # Pre-create images to avoid measuring image creation time
    print("Pre-creating synthetic images...")
    images = [create_synthetic_images() for _ in range(num_iterations)]
    print(f"Created {num_iterations} image pairs")
    print()

    # Warm up
    print("Warming up...")
    for i in range(10):
        rgb, depth = images[i]
        _ = apply_full_augmentation(rgb.copy(), depth.copy())
        _ = apply_partial_augmentation(rgb.copy(), depth.copy())
    print()

    # ==================== Full Augmentation ====================
    print("-" * 70)
    print("FULL AUGMENTATION (normalize=True)")
    print("-" * 70)
    print("Includes: Flip, Crop, ColorJitter, Blur, Grayscale, DepthAug, Norm, Erasing")
    print()

    full_times = []
    for rgb, depth in images:
        start = time.perf_counter()
        _ = apply_full_augmentation(rgb.copy(), depth.copy())
        full_times.append(time.perf_counter() - start)

    full_mean = statistics.mean(full_times) * 1000
    full_std = statistics.stdev(full_times) * 1000
    full_min = min(full_times) * 1000
    full_max = max(full_times) * 1000
    full_p50 = statistics.median(full_times) * 1000
    full_p95 = sorted(full_times)[int(len(full_times) * 0.95)] * 1000

    print(f"  Mean:   {full_mean:.2f} ms")
    print(f"  Std:    {full_std:.2f} ms")
    print(f"  Min:    {full_min:.2f} ms")
    print(f"  Max:    {full_max:.2f} ms")
    print(f"  P50:    {full_p50:.2f} ms")
    print(f"  P95:    {full_p95:.2f} ms")
    print(f"  Throughput: {1000/full_mean:.1f} samples/sec")

    # ==================== Partial Augmentation ====================
    print()
    print("-" * 70)
    print("PARTIAL AUGMENTATION (normalize=False)")
    print("-" * 70)
    print("Includes: Flip, Crop, DepthAug only (ColorJitter/Blur/Grayscale/Norm/Erasing -> GPU)")
    print()

    partial_times = []
    for rgb, depth in images:
        start = time.perf_counter()
        _ = apply_partial_augmentation(rgb.copy(), depth.copy())
        partial_times.append(time.perf_counter() - start)

    partial_mean = statistics.mean(partial_times) * 1000
    partial_std = statistics.stdev(partial_times) * 1000
    partial_min = min(partial_times) * 1000
    partial_max = max(partial_times) * 1000
    partial_p50 = statistics.median(partial_times) * 1000
    partial_p95 = sorted(partial_times)[int(len(partial_times) * 0.95)] * 1000

    print(f"  Mean:   {partial_mean:.2f} ms")
    print(f"  Std:    {partial_std:.2f} ms")
    print(f"  Min:    {partial_min:.2f} ms")
    print(f"  Max:    {partial_max:.2f} ms")
    print(f"  P50:    {partial_p50:.2f} ms")
    print(f"  P95:    {partial_p95:.2f} ms")
    print(f"  Throughput: {1000/partial_mean:.1f} samples/sec")

    # ==================== Comparison ====================
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    speedup = full_mean / partial_mean
    time_saved = full_mean - partial_mean
    percent_saved = (time_saved / full_mean) * 100

    print(f"\n  Full augmentation:    {full_mean:.2f} ms/sample")
    print(f"  Partial augmentation: {partial_mean:.2f} ms/sample")
    print(f"  Time saved:           {time_saved:.2f} ms/sample ({percent_saved:.1f}%)")
    print(f"  Speedup:              {speedup:.2f}x")

    # Per-batch calculation
    batch_size = 32
    time_saved_per_batch = time_saved * batch_size

    print(f"\n  Time saved per batch (batch_size={batch_size}):")
    print(f"    {time_saved_per_batch:.1f} ms/batch")
    print(f"    {time_saved_per_batch/1000:.3f} sec/batch")

    # Per-epoch calculation (assuming ~250 batches for 8000 samples)
    num_batches = 8041 // batch_size
    time_saved_per_epoch = time_saved_per_batch * num_batches / 1000

    print(f"\n  Time saved per epoch ({num_batches} batches):")
    print(f"    {time_saved_per_epoch:.1f} sec/epoch")

    # ==================== Individual Transform Timing ====================
    print()
    print("=" * 70)
    print("INDIVIDUAL TRANSFORM TIMING (average of 100 runs)")
    print("=" * 70)

    rgb, depth = create_synthetic_images()
    target_size = (416, 544)

    # Resize
    times = []
    for _ in range(100):
        rgb_copy = rgb.copy()
        start = time.perf_counter()
        _ = transforms.functional.resize(rgb_copy, target_size)
        times.append(time.perf_counter() - start)
    print(f"  Resize:          {statistics.mean(times)*1000:.2f} ms")

    # ColorJitter
    color_jitter = transforms.ColorJitter(brightness=0.37, contrast=0.37, saturation=0.37, hue=0.11)
    times = []
    for _ in range(100):
        rgb_copy = rgb.copy()
        start = time.perf_counter()
        _ = color_jitter(rgb_copy)
        times.append(time.perf_counter() - start)
    print(f"  ColorJitter:     {statistics.mean(times)*1000:.2f} ms")

    # Gaussian Blur
    times = []
    for _ in range(100):
        rgb_copy = rgb.copy()
        start = time.perf_counter()
        _ = transforms.functional.gaussian_blur(rgb_copy, kernel_size=5, sigma=1.0)
        times.append(time.perf_counter() - start)
    print(f"  Gaussian Blur:   {statistics.mean(times)*1000:.2f} ms")

    # Grayscale
    times = []
    for _ in range(100):
        rgb_copy = rgb.copy()
        start = time.perf_counter()
        _ = transforms.functional.to_grayscale(rgb_copy, num_output_channels=3)
        times.append(time.perf_counter() - start)
    print(f"  Grayscale:       {statistics.mean(times)*1000:.2f} ms")

    # ToTensor
    times = []
    for _ in range(100):
        rgb_copy = rgb.copy()
        start = time.perf_counter()
        _ = transforms.functional.to_tensor(rgb_copy)
        times.append(time.perf_counter() - start)
    print(f"  ToTensor:        {statistics.mean(times)*1000:.2f} ms")

    # Normalize
    rgb_tensor = transforms.functional.to_tensor(rgb)
    times = []
    for _ in range(100):
        rgb_copy = rgb_tensor.clone()
        start = time.perf_counter()
        _ = transforms.functional.normalize(rgb_copy, mean=[0.49, 0.46, 0.43], std=[0.28, 0.29, 0.29])
        times.append(time.perf_counter() - start)
    print(f"  Normalize:       {statistics.mean(times)*1000:.2f} ms")

    # RandomErasing
    erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.10), ratio=(0.5, 2.0))
    times = []
    for _ in range(100):
        rgb_copy = rgb_tensor.clone()
        start = time.perf_counter()
        _ = erasing(rgb_copy)
        times.append(time.perf_counter() - start)
    print(f"  RandomErasing:   {statistics.mean(times)*1000:.2f} ms")

    # Depth augmentation
    depth_array = np.array(depth, dtype=np.float32)
    times = []
    for _ in range(100):
        d = depth_array.copy()
        start = time.perf_counter()
        d = (d - 0.5) * 1.1 + 0.5
        d = d * 1.1
        noise = np.random.normal(0, 0.059, d.shape).astype(np.float32)
        d = np.clip(d + noise, 0.0, 1.0)
        times.append(time.perf_counter() - start)
    print(f"  Depth Aug:       {statistics.mean(times)*1000:.2f} ms")


if __name__ == '__main__':
    benchmark_transforms()
