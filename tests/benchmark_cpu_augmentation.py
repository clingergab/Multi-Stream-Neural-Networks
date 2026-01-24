"""
Benchmark CPU augmentation performance.

Compares:
- Full augmentation (normalize=True): All augmentations on CPU
- Partial augmentation (normalize=False): Spatial transforms only, no ColorJitter/Blur/Grayscale/Erasing

This measures CPU-only performance - no GPU interactions.
"""

import os
import sys
import time
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
torch.set_num_threads(multiprocessing.cpu_count())

from src.data_utils.sunrgbd_dataset import SUNRGBDDataset


def measure_cpu_usage(duration_sec: float = 1.0) -> float:
    """Measure average CPU usage over a duration."""
    return psutil.cpu_percent(interval=duration_sec)


def benchmark_dataset_iteration(
    dataset: SUNRGBDDataset,
    num_samples: int = 100,
    num_workers: int = 0,
) -> dict:
    """
    Benchmark dataset __getitem__ performance.

    Args:
        dataset: Dataset to benchmark
        num_samples: Number of samples to load
        num_workers: Number of parallel workers (0 = single thread)

    Returns:
        dict with timing and CPU usage stats
    """
    # Warm up
    for i in range(min(5, len(dataset))):
        _ = dataset[i]

    times = []

    if num_workers == 0:
        # Single-threaded benchmark
        cpu_before = psutil.cpu_percent(interval=None)

        start_total = time.perf_counter()
        for i in range(num_samples):
            idx = i % len(dataset)
            start = time.perf_counter()
            _ = dataset[idx]
            times.append(time.perf_counter() - start)
        total_time = time.perf_counter() - start_total

        # Measure CPU after
        cpu_after = psutil.cpu_percent(interval=0.1)
        avg_cpu = (cpu_before + cpu_after) / 2

    else:
        # Multi-threaded benchmark (simulates DataLoader workers)
        def load_sample(idx):
            start = time.perf_counter()
            _ = dataset[idx % len(dataset)]
            return time.perf_counter() - start

        indices = list(range(num_samples))

        cpu_before = psutil.cpu_percent(interval=None)
        start_total = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            times = list(executor.map(load_sample, indices))

        total_time = time.perf_counter() - start_total
        cpu_after = psutil.cpu_percent(interval=0.1)
        avg_cpu = (cpu_before + cpu_after) / 2

    return {
        'total_time': total_time,
        'mean_time_ms': statistics.mean(times) * 1000,
        'std_time_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'samples_per_sec': num_samples / total_time,
        'cpu_percent': avg_cpu,
    }


def benchmark_dataloader(
    dataset: SUNRGBDDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    num_batches: int = 20,
) -> dict:
    """
    Benchmark DataLoader performance (more realistic).

    Args:
        dataset: Dataset to benchmark
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        num_batches: Number of batches to load

    Returns:
        dict with timing stats
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # No GPU
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Warm up - load first batch
    iterator = iter(loader)
    _ = next(iterator)

    # Measure CPU usage during loading
    process = psutil.Process()
    cpu_times_before = process.cpu_times()

    batch_times = []
    start_total = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch_times.append(time.perf_counter())

    total_time = time.perf_counter() - start_total

    cpu_times_after = process.cpu_times()
    cpu_user = cpu_times_after.user - cpu_times_before.user
    cpu_system = cpu_times_after.system - cpu_times_before.system

    # Calculate inter-batch times
    inter_batch_times = []
    for i in range(1, len(batch_times)):
        inter_batch_times.append(batch_times[i] - batch_times[i-1])

    return {
        'total_time': total_time,
        'mean_batch_time_ms': statistics.mean(inter_batch_times) * 1000 if inter_batch_times else 0,
        'std_batch_time_ms': statistics.stdev(inter_batch_times) * 1000 if len(inter_batch_times) > 1 else 0,
        'batches_per_sec': num_batches / total_time,
        'samples_per_sec': (num_batches * batch_size) / total_time,
        'cpu_user_time': cpu_user,
        'cpu_system_time': cpu_system,
        'cpu_total_time': cpu_user + cpu_system,
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("CPU AUGMENTATION BENCHMARK")
    print("=" * 70)
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print()

    # Check if dataset exists
    data_root = 'data/sunrgbd_15'
    if not os.path.exists(data_root):
        print(f"ERROR: Dataset not found at {data_root}")
        print("Please run this benchmark on a machine with the dataset.")
        return

    # Create datasets
    print("Creating datasets...")
    dataset_full = SUNRGBDDataset(
        data_root=data_root,
        train=True,
        target_size=(416, 544),
        normalize=True,  # Full CPU augmentation
    )

    dataset_partial = SUNRGBDDataset(
        data_root=data_root,
        train=True,
        target_size=(416, 544),
        normalize=False,  # Partial augmentation (for GPU mode)
    )

    print(f"Dataset size: {len(dataset_full)} samples")
    print()

    # ==================== Single-threaded benchmark ====================
    print("-" * 70)
    print("BENCHMARK 1: Single-threaded __getitem__ (100 samples)")
    print("-" * 70)

    print("\nFull augmentation (normalize=True):")
    result_full_single = benchmark_dataset_iteration(dataset_full, num_samples=100, num_workers=0)
    print(f"  Mean time per sample: {result_full_single['mean_time_ms']:.2f} ± {result_full_single['std_time_ms']:.2f} ms")
    print(f"  Min/Max: {result_full_single['min_time_ms']:.2f} / {result_full_single['max_time_ms']:.2f} ms")
    print(f"  Throughput: {result_full_single['samples_per_sec']:.1f} samples/sec")

    print("\nPartial augmentation (normalize=False):")
    result_partial_single = benchmark_dataset_iteration(dataset_partial, num_samples=100, num_workers=0)
    print(f"  Mean time per sample: {result_partial_single['mean_time_ms']:.2f} ± {result_partial_single['std_time_ms']:.2f} ms")
    print(f"  Min/Max: {result_partial_single['min_time_ms']:.2f} / {result_partial_single['max_time_ms']:.2f} ms")
    print(f"  Throughput: {result_partial_single['samples_per_sec']:.1f} samples/sec")

    speedup = result_full_single['mean_time_ms'] / result_partial_single['mean_time_ms']
    time_saved = result_full_single['mean_time_ms'] - result_partial_single['mean_time_ms']
    print(f"\n  Speedup: {speedup:.2f}x ({time_saved:.2f} ms saved per sample)")

    # ==================== Multi-threaded benchmark ====================
    print("\n" + "-" * 70)
    print("BENCHMARK 2: Multi-threaded __getitem__ (4 workers, 100 samples)")
    print("-" * 70)

    print("\nFull augmentation (normalize=True):")
    result_full_multi = benchmark_dataset_iteration(dataset_full, num_samples=100, num_workers=4)
    print(f"  Total time: {result_full_multi['total_time']:.2f}s")
    print(f"  Throughput: {result_full_multi['samples_per_sec']:.1f} samples/sec")

    print("\nPartial augmentation (normalize=False):")
    result_partial_multi = benchmark_dataset_iteration(dataset_partial, num_samples=100, num_workers=4)
    print(f"  Total time: {result_partial_multi['total_time']:.2f}s")
    print(f"  Throughput: {result_partial_multi['samples_per_sec']:.1f} samples/sec")

    speedup = result_full_multi['total_time'] / result_partial_multi['total_time']
    print(f"\n  Speedup: {speedup:.2f}x")

    # ==================== DataLoader benchmark ====================
    print("\n" + "-" * 70)
    print("BENCHMARK 3: DataLoader (batch_size=32, num_workers=4, 20 batches)")
    print("-" * 70)

    print("\nFull augmentation (normalize=True):")
    result_full_loader = benchmark_dataloader(dataset_full, batch_size=32, num_workers=4, num_batches=20)
    print(f"  Total time: {result_full_loader['total_time']:.2f}s")
    print(f"  Mean batch time: {result_full_loader['mean_batch_time_ms']:.2f} ± {result_full_loader['std_batch_time_ms']:.2f} ms")
    print(f"  Throughput: {result_full_loader['batches_per_sec']:.1f} batches/sec ({result_full_loader['samples_per_sec']:.0f} samples/sec)")
    print(f"  CPU time (user): {result_full_loader['cpu_user_time']:.2f}s")
    print(f"  CPU time (system): {result_full_loader['cpu_system_time']:.2f}s")

    print("\nPartial augmentation (normalize=False):")
    result_partial_loader = benchmark_dataloader(dataset_partial, batch_size=32, num_workers=4, num_batches=20)
    print(f"  Total time: {result_partial_loader['total_time']:.2f}s")
    print(f"  Mean batch time: {result_partial_loader['mean_batch_time_ms']:.2f} ± {result_partial_loader['std_batch_time_ms']:.2f} ms")
    print(f"  Throughput: {result_partial_loader['batches_per_sec']:.1f} batches/sec ({result_partial_loader['samples_per_sec']:.0f} samples/sec)")
    print(f"  CPU time (user): {result_partial_loader['cpu_user_time']:.2f}s")
    print(f"  CPU time (system): {result_partial_loader['cpu_system_time']:.2f}s")

    speedup = result_full_loader['total_time'] / result_partial_loader['total_time']
    cpu_saved = result_full_loader['cpu_total_time'] - result_partial_loader['cpu_total_time']
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  CPU time saved: {cpu_saved:.2f}s ({cpu_saved/result_full_loader['cpu_total_time']*100:.1f}%)")

    # ==================== DataLoader benchmark (8 workers) ====================
    print("\n" + "-" * 70)
    print("BENCHMARK 4: DataLoader (batch_size=32, num_workers=8, 20 batches)")
    print("-" * 70)

    print("\nFull augmentation (normalize=True):")
    result_full_loader8 = benchmark_dataloader(dataset_full, batch_size=32, num_workers=8, num_batches=20)
    print(f"  Total time: {result_full_loader8['total_time']:.2f}s")
    print(f"  Mean batch time: {result_full_loader8['mean_batch_time_ms']:.2f} ± {result_full_loader8['std_batch_time_ms']:.2f} ms")
    print(f"  Throughput: {result_full_loader8['batches_per_sec']:.1f} batches/sec ({result_full_loader8['samples_per_sec']:.0f} samples/sec)")
    print(f"  CPU time (user): {result_full_loader8['cpu_user_time']:.2f}s")

    print("\nPartial augmentation (normalize=False):")
    result_partial_loader8 = benchmark_dataloader(dataset_partial, batch_size=32, num_workers=8, num_batches=20)
    print(f"  Total time: {result_partial_loader8['total_time']:.2f}s")
    print(f"  Mean batch time: {result_partial_loader8['mean_batch_time_ms']:.2f} ± {result_partial_loader8['std_batch_time_ms']:.2f} ms")
    print(f"  Throughput: {result_partial_loader8['batches_per_sec']:.1f} batches/sec ({result_partial_loader8['samples_per_sec']:.0f} samples/sec)")
    print(f"  CPU time (user): {result_partial_loader8['cpu_user_time']:.2f}s")

    speedup = result_full_loader8['total_time'] / result_partial_loader8['total_time']
    print(f"\n  Speedup: {speedup:.2f}x")

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Full augmentation (normalize=True) includes:
  - Random horizontal flip (50%)
  - Random resized crop (50%)
  - ColorJitter (43%)
  - Gaussian blur (25%)
  - Grayscale (17%)
  - Depth brightness/contrast/noise (50%)
  - Normalization
  - Random erasing RGB (17%)
  - Random erasing Depth (10%)

Partial augmentation (normalize=False) includes:
  - Random horizontal flip (50%)
  - Random resized crop (50%)
  - Depth brightness/contrast/noise (50%)
  (ColorJitter, Blur, Grayscale, Normalization, Erasing skipped - for GPU)
""")

    avg_speedup = (
        (result_full_single['mean_time_ms'] / result_partial_single['mean_time_ms']) +
        (result_full_loader['total_time'] / result_partial_loader['total_time']) +
        (result_full_loader8['total_time'] / result_partial_loader8['total_time'])
    ) / 3

    print(f"Average speedup with partial augmentation: {avg_speedup:.2f}x")
    print(f"Single-sample time saved: {time_saved:.2f} ms")
    print()
    print("Recommendation:")
    print("  - Single training runs: Use full CPU augmentation (normalize=True)")
    print("  - Ray Tune parallel trials: Use GPU augmentation (normalize=False)")


if __name__ == '__main__':
    run_benchmarks()
