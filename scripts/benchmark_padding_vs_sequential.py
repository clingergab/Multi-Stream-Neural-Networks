#!/usr/bin/env python3
"""
Benchmark: Channel Padding vs Sequential Stream Processing

This script tests the Tier 2 (channel padding) optimization in isolation
by comparing sequential F.conv2d calls vs a single batched call with padding.

Tests realistic LINet configurations:
- 2-stream: RGB (3 channels) + Depth (1 channel)
- 3-stream: RGB (3) + Depth (1) + Orthogonal (64)

Measures:
- Forward pass time
- Backward pass time
- Memory usage
- GPU utilization
- Wasted compute percentage

Usage:
    # In a terminal
    python benchmark_padding_vs_sequential.py --all

    # In a Jupyter/Colab notebook
    from scripts.benchmark_padding_vs_sequential import run_benchmark_suite
    run_benchmark_suite([3, 1], batch_size=16)
"""

import os
import sys
import time
import contextlib

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import argparse


@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def create_test_data(
    batch_size: int,
    stream_channels: List[int],
    height: int = 416,  # SUN RGB-D default height
    width: int = 544,   # SUN RGB-D default width
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create test inputs and weights for benchmarking.

    Uses realistic parameters from SUN RGB-D dataset:
    - Default resolution: 416x544 (matches training data)
    - Stream channels: [3, 1] for RGB + Depth
    - First conv: 64 output channels, 7x7 kernel, stride=2, padding=3

    Args:
        batch_size: Batch size
        stream_channels: List of input channels per stream (e.g., [3, 1] for RGB+Depth)
        height: Input height (default: 416, SUN RGB-D)
        width: Input width (default: 544, SUN RGB-D)
        device: Device to create tensors on

    Returns:
        (stream_inputs, stream_weights)
    """
    stream_inputs = []
    stream_weights = []
    out_channels = 64  # ResNet first conv output (standard)
    kernel_size = 7    # ResNet first conv kernel (standard)

    for in_ch in stream_channels:
        # Create input: [batch, channels, height, width]
        # Use realistic value range for SUN RGB-D data
        if in_ch == 3:
            # RGB: normalized with mean~0.45, std~0.28 (from dataset stats)
            inp = torch.randn(batch_size, in_ch, height, width, device=device) * 0.28 + 0.45
        elif in_ch == 1:
            # Depth: normalized with mean~0.29, std~0.15 (from dataset stats)
            inp = torch.randn(batch_size, in_ch, height, width, device=device) * 0.15 + 0.29
        else:
            # Other modalities: use standard normal
            inp = torch.randn(batch_size, in_ch, height, width, device=device)

        inp.requires_grad = True
        stream_inputs.append(inp)

        # Create weight: [out_channels, in_channels, kernel_h, kernel_w]
        # Initialize with Kaiming uniform (same as LINet)
        weight = torch.empty(out_channels, in_ch, kernel_size, kernel_size, device=device)
        torch.nn.init.kaiming_uniform_(weight, a=5**0.5)
        weight.requires_grad = True
        stream_weights.append(weight)

    return stream_inputs, stream_weights


def sequential_forward(
    stream_inputs: List[torch.Tensor],
    stream_weights: List[torch.Tensor],
    stride: int = 2,
    padding: int = 3
) -> List[torch.Tensor]:
    """Sequential stream processing (current implementation).

    Args:
        stream_inputs: List of input tensors
        stream_weights: List of weight tensors
        stride: Convolution stride
        padding: Convolution padding

    Returns:
        List of output tensors
    """
    outputs = []
    for inp, weight in zip(stream_inputs, stream_weights):
        out = F.conv2d(inp, weight, bias=None, stride=stride, padding=padding)
        outputs.append(out)
    return outputs


def batched_forward_with_padding(
    stream_inputs: List[torch.Tensor],
    stream_weights: List[torch.Tensor],
    stride: int = 2,
    padding: int = 3
) -> List[torch.Tensor]:
    """Batched stream processing with channel padding (Tier 2).

    Uses grouped convolutions to process all streams in a single kernel call.
    Each stream is padded to the max channel count, then all streams are
    processed using groups=num_streams.

    Args:
        stream_inputs: List of input tensors
        stream_weights: List of weight tensors
        stride: Convolution stride
        padding: Convolution padding

    Returns:
        List of output tensors (same as sequential)
    """
    num_streams = len(stream_inputs)

    # Step 1: Compute padding configuration
    stream_in_channels = [inp.shape[1] for inp in stream_inputs]
    max_in_channels = max(stream_in_channels)
    out_channels = stream_weights[0].shape[0]  # All streams have same output channels

    # Step 2: Pad inputs to max channels
    padded_inputs = []
    for inp, in_ch in zip(stream_inputs, stream_in_channels):
        if in_ch < max_in_channels:
            pad_size = max_in_channels - in_ch
            # Pad channels: (left, right, top, bottom, front, back)
            padded = F.pad(inp, (0, 0, 0, 0, 0, pad_size), value=0.0)
        else:
            padded = inp
        padded_inputs.append(padded)

    # Step 3: Stack inputs along channel dimension
    batched_input = torch.cat(padded_inputs, dim=1)
    # Shape: [batch, num_streams * max_in_channels, height, width]
    # e.g., for [3, 1]: [B, 2*3=6, H, W]

    # Step 4: Pad weights to max channels
    padded_weights = []
    for weight, in_ch in zip(stream_weights, stream_in_channels):
        if in_ch < max_in_channels:
            pad_size = max_in_channels - in_ch
            # Pad input channel dimension
            padded = F.pad(weight, (0, 0, 0, 0, 0, pad_size), value=0.0)
        else:
            padded = weight
        padded_weights.append(padded)

    # Step 5: Stack weights for grouped convolution
    batched_weight = torch.cat(padded_weights, dim=0)
    # Shape: [num_streams * out_channels, max_in_channels, kernel_h, kernel_w]
    # e.g., for 2 streams with 64 out each: [128, 3, 7, 7]

    # Step 6: Single batched convolution with groups=num_streams
    # This processes each stream independently in parallel
    batched_output = F.conv2d(
        batched_input,
        batched_weight,
        bias=None,
        stride=stride,
        padding=padding,
        groups=num_streams  # KEY: Each group processes one stream
    )
    # Shape: [batch, num_streams * out_channels, height', width']

    # Step 7: Split outputs back into separate streams
    outputs = []
    for i in range(num_streams):
        outputs.append(batched_output[:, i*out_channels:(i+1)*out_channels])

    return outputs


def benchmark_forward_pass(
    stream_inputs: List[torch.Tensor],
    stream_weights: List[torch.Tensor],
    num_iters: int = 100,
    warmup_iters: int = 10
) -> Tuple[float, float]:
    """Benchmark forward pass for both approaches.

    Returns:
        (sequential_time_ms, batched_time_ms)
    """
    device = stream_inputs[0].device

    # Warmup
    for _ in range(warmup_iters):
        _ = sequential_forward(stream_inputs, stream_weights)
        _ = batched_forward_with_padding(stream_inputs, stream_weights)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark sequential
    seq_times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = sequential_forward(stream_inputs, stream_weights)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seq_times.append(time.perf_counter() - start)

    # Benchmark batched
    batched_times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = batched_forward_with_padding(stream_inputs, stream_weights)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batched_times.append(time.perf_counter() - start)

    seq_mean = np.mean(seq_times) * 1000  # Convert to ms
    batched_mean = np.mean(batched_times) * 1000

    return seq_mean, batched_mean


def benchmark_backward_pass(
    stream_inputs: List[torch.Tensor],
    stream_weights: List[torch.Tensor],
    num_iters: int = 100,
    warmup_iters: int = 10
) -> Tuple[float, float]:
    """Benchmark backward pass for both approaches.

    Returns:
        (sequential_time_ms, batched_time_ms)
    """
    device = stream_inputs[0].device

    # Warmup
    for _ in range(warmup_iters):
        outputs = sequential_forward(stream_inputs, stream_weights)
        loss = sum(out.sum() for out in outputs)
        loss.backward()
        for inp in stream_inputs:
            inp.grad = None
        for w in stream_weights:
            w.grad = None

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark sequential
    seq_times = []
    for _ in range(num_iters):
        outputs = sequential_forward(stream_inputs, stream_weights)
        loss = sum(out.sum() for out in outputs)

        start = time.perf_counter()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seq_times.append(time.perf_counter() - start)

        # Clear gradients
        for inp in stream_inputs:
            inp.grad = None
        for w in stream_weights:
            w.grad = None

    # Benchmark batched
    batched_times = []
    for _ in range(num_iters):
        outputs = batched_forward_with_padding(stream_inputs, stream_weights)
        loss = sum(out.sum() for out in outputs)

        start = time.perf_counter()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batched_times.append(time.perf_counter() - start)

        # Clear gradients
        for inp in stream_inputs:
            inp.grad = None
        for w in stream_weights:
            w.grad = None

    seq_mean = np.mean(seq_times) * 1000
    batched_mean = np.mean(batched_times) * 1000

    return seq_mean, batched_mean


def calculate_wasted_compute(stream_channels: List[int]) -> float:
    """Calculate percentage of wasted compute due to padding.

    Returns:
        Percentage of wasted FLOPs (0-100)
    """
    max_channels = max(stream_channels)
    total_actual = sum(stream_channels)
    total_padded = max_channels * len(stream_channels)

    wasted = total_padded - total_actual
    wasted_percent = (wasted / total_padded) * 100

    return wasted_percent


def verify_numerical_equivalence(
    stream_inputs: List[torch.Tensor],
    stream_weights: List[torch.Tensor]
) -> bool:
    """Verify that batched approach produces identical results to sequential.

    Returns:
        True if outputs are numerically equivalent
    """
    with torch.no_grad():
        seq_outputs = sequential_forward(stream_inputs, stream_weights)
        batched_outputs = batched_forward_with_padding(stream_inputs, stream_weights)

    for i, (seq_out, batched_out) in enumerate(zip(seq_outputs, batched_outputs)):
        if not torch.allclose(seq_out, batched_out, atol=1e-6, rtol=1e-5):
            print(f"❌ Stream {i} outputs differ!")
            max_diff = (seq_out - batched_out).abs().max().item()
            print(f"   Max difference: {max_diff:.2e}")
            return False

    print("✅ Numerical equivalence verified (all streams match within tolerance)")
    return True


def measure_memory_usage(
    stream_inputs: List[torch.Tensor],
    stream_weights: List[torch.Tensor]
) -> Tuple[float, float]:
    """Measure peak memory usage for both approaches.

    Returns:
        (sequential_memory_mb, batched_memory_mb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    # Sequential
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    _ = sequential_forward(stream_inputs, stream_weights)
    seq_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB

    # Batched
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    _ = batched_forward_with_padding(stream_inputs, stream_weights)
    batched_memory = torch.cuda.max_memory_allocated() / 1024**2

    return seq_memory, batched_memory


def run_benchmark_suite(
    stream_channels: List[int],
    batch_size: int = 16,
    num_iters: int = 100,
    device: str = 'cuda',
    use_torch_compile: bool = True
):
    """Run complete benchmark suite for a given configuration.

    Args:
        stream_channels: List of input channels per stream (e.g., [3, 1] for RGB+Depth)
        batch_size: Batch size (default: 16, typical for SUN RGB-D training)
        num_iters: Number of iterations for timing (default: 100)
        device: Device to run on (must be 'cuda')
        use_torch_compile: Whether to apply torch.compile to functions (default: False)

    Raises:
        RuntimeError: If CUDA is not available
    """
    # GPU is mandatory for this benchmark
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA not available! This benchmark requires GPU.\n"
            "   GPU parallelization cannot be measured on CPU.\n"
            "   Please run on a machine with CUDA-enabled GPU."
        )

    if device != 'cuda':
        raise ValueError(
            f"❌ Device must be 'cuda', got '{device}'.\n"
            "   This benchmark only measures GPU parallelization."
        )

    print(f"\n{'='*80}")
    print(f"Benchmark Configuration:")
    print(f"  Stream channels: {stream_channels}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input size: 416x544 (SUN RGB-D)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Iterations: {num_iters}")
    print(f"  torch.compile: {'ENABLED (max-autotune)' if use_torch_compile else 'DISABLED'}")
    print(f"{'='*80}\n")

    # Apply torch.compile if requested
    seq_forward_fn = sequential_forward
    batched_forward_fn = batched_forward_with_padding

    if use_torch_compile:
        print("⚙️  Applying torch.compile (compilation will happen on first forward pass)...")
        seq_forward_fn = torch.compile(sequential_forward, mode='max-autotune')
        batched_forward_fn = torch.compile(batched_forward_with_padding, mode='max-autotune')

    # Create test data (uses SUN RGB-D defaults: 416x544)
    stream_inputs, stream_weights = create_test_data(batch_size, stream_channels, device=device)

    # 1. Verify numerical equivalence (using compiled functions if enabled)
    print("\n1. Numerical Equivalence Check")
    print("-" * 40)

    if use_torch_compile:
        print("⏳ Compiling functions (first run, ~10-30s with autotuning)...")
        print("   Suppressing verbose autotune output...\n")

    with torch.no_grad():
        # Suppress verbose autotuning output during compilation
        if use_torch_compile:
            with suppress_output():
                seq_outputs = seq_forward_fn(stream_inputs, stream_weights)
                batched_outputs = batched_forward_fn(stream_inputs, stream_weights)

                # Clone outputs INSIDE the context to prevent CUDA Graphs memory overwrite
                seq_outputs = [out.clone() for out in seq_outputs]
                batched_outputs = [out.clone() for out in batched_outputs]
        else:
            seq_outputs = seq_forward_fn(stream_inputs, stream_weights)
            batched_outputs = batched_forward_fn(stream_inputs, stream_weights)

    if use_torch_compile:
        print("✅ Compilation complete")

    is_equivalent = True
    for i, (seq_out, batched_out) in enumerate(zip(seq_outputs, batched_outputs)):
        if not torch.allclose(seq_out, batched_out, atol=1e-6, rtol=1e-5):
            print(f"❌ Stream {i} outputs differ!")
            max_diff = (seq_out - batched_out).abs().max().item()
            print(f"   Max difference: {max_diff:.2e}")
            is_equivalent = False

    if not is_equivalent:
        print("⚠️  Warning: Numerical equivalence failed! Results may not be valid.\n")
        return
    print("✅ Numerical equivalence verified (all streams match within tolerance)")
    print()

    # 2. Forward pass benchmark
    print("2. Forward Pass Benchmark")
    print("-" * 40)

    # Benchmark using compiled functions
    device_obj = stream_inputs[0].device

    # Warmup
    for _ in range(10):
        _ = seq_forward_fn(stream_inputs, stream_weights)
        _ = batched_forward_fn(stream_inputs, stream_weights)

    if device_obj.type == 'cuda':
        torch.cuda.synchronize()

    # Measure sequential
    seq_times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = seq_forward_fn(stream_inputs, stream_weights)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        seq_times.append(time.perf_counter() - start)

    # Measure batched
    batched_times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = batched_forward_fn(stream_inputs, stream_weights)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        batched_times.append(time.perf_counter() - start)

    seq_fwd = np.mean(seq_times) * 1000
    batched_fwd = np.mean(batched_times) * 1000
    speedup_fwd = seq_fwd / batched_fwd

    print(f"Sequential:    {seq_fwd:7.3f} ms")
    print(f"Batched:       {batched_fwd:7.3f} ms")
    print(f"Speedup:       {speedup_fwd:7.2f}x")
    print()

    # 3. Backward pass benchmark
    print("3. Backward Pass Benchmark")
    print("-" * 40)

    # Warmup
    for _ in range(10):
        outputs = seq_forward_fn(stream_inputs, stream_weights)
        loss = sum(out.sum() for out in outputs)
        loss.backward()
        for inp in stream_inputs:
            inp.grad = None
        for w in stream_weights:
            w.grad = None

    if device_obj.type == 'cuda':
        torch.cuda.synchronize()

    # Measure sequential backward
    seq_times = []
    for _ in range(num_iters):
        outputs = seq_forward_fn(stream_inputs, stream_weights)
        loss = sum(out.sum() for out in outputs)

        start = time.perf_counter()
        loss.backward()
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        seq_times.append(time.perf_counter() - start)

        for inp in stream_inputs:
            inp.grad = None
        for w in stream_weights:
            w.grad = None

    # Measure batched backward
    batched_times = []
    for _ in range(num_iters):
        outputs = batched_forward_fn(stream_inputs, stream_weights)
        loss = sum(out.sum() for out in outputs)

        start = time.perf_counter()
        loss.backward()
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        batched_times.append(time.perf_counter() - start)

        for inp in stream_inputs:
            inp.grad = None
        for w in stream_weights:
            w.grad = None

    seq_bwd = np.mean(seq_times) * 1000
    batched_bwd = np.mean(batched_times) * 1000
    speedup_bwd = seq_bwd / batched_bwd

    print(f"Sequential:    {seq_bwd:7.3f} ms")
    print(f"Batched:       {batched_bwd:7.3f} ms")
    print(f"Speedup:       {speedup_bwd:7.2f}x")
    print()

    # 4. Total time (forward + backward)
    print("4. Total Time (Forward + Backward)")
    print("-" * 40)
    seq_total = seq_fwd + seq_bwd
    batched_total = batched_fwd + batched_bwd
    speedup_total = seq_total / batched_total

    print(f"Sequential:    {seq_total:7.3f} ms")
    print(f"Batched:       {batched_total:7.3f} ms")
    print(f"Speedup:       {speedup_total:7.2f}x")
    print()

    # 5. Memory usage
    if device == 'cuda':
        print("5. Memory Usage")
        print("-" * 40)

        # Sequential memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _ = seq_forward_fn(stream_inputs, stream_weights)
        seq_mem = torch.cuda.max_memory_allocated() / 1024**2

        # Batched memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _ = batched_forward_fn(stream_inputs, stream_weights)
        batched_mem = torch.cuda.max_memory_allocated() / 1024**2

        mem_overhead = ((batched_mem - seq_mem) / seq_mem) * 100 if seq_mem > 0 else 0

        print(f"Sequential:    {seq_mem:7.1f} MB")
        print(f"Batched:       {batched_mem:7.1f} MB")
        print(f"Overhead:      {mem_overhead:+7.1f}%")
        print()

    # 6. Wasted compute analysis
    print("6. Efficiency Analysis")
    print("-" * 40)
    wasted = calculate_wasted_compute(stream_channels)
    effective_speedup = speedup_total * (1 - wasted/100)

    print(f"Wasted compute:        {wasted:5.1f}%")
    print(f"Effective speedup:     {effective_speedup:5.2f}x")
    print(f"  (accounting for waste)")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Forward speedup:       {speedup_fwd:.2f}x")
    print(f"Backward speedup:      {speedup_bwd:.2f}x")
    print(f"Total speedup:         {speedup_total:.2f}x")
    print(f"Wasted compute:        {wasted:.1f}%")
    print(f"Recommendation:        ", end="")

    if speedup_total > 2.0:
        print("✅ HIGHLY RECOMMENDED - Significant speedup!")
    elif speedup_total > 1.5:
        print("✅ RECOMMENDED - Worthwhile speedup")
    elif speedup_total > 1.2:
        print("⚠️  MARGINAL - Consider implementation cost")
    else:
        print("❌ NOT RECOMMENDED - Insufficient speedup")

    print("=" * 80)


def run_all_benchmarks(batch_size: int = 16, num_iters: int = 100, device: str = 'cuda', use_torch_compile: bool = False):
    """Run benchmarks for all common stream configurations.

    This is a convenience function for notebook environments.

    Args:
        batch_size: Batch size for benchmarking (default: 16)
        num_iters: Number of iterations per benchmark (default: 100)
        device: Device to run on (must be 'cuda')
        use_torch_compile: Whether to apply torch.compile to functions (default: False)

    Raises:
        RuntimeError: If CUDA is not available
    """
    # GPU is mandatory
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA not available! This benchmark requires GPU.\n"
            "   Please run on a machine with CUDA-enabled GPU."
        )

    print("\n" + "="*80)
    print("GPU STREAM PARALLELIZATION BENCHMARK")
    print("Comparing Sequential vs Batched (with Padding) Approaches")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)

    configs = [
        ([3, 1], "2-stream: RGB (3ch) + Depth (1ch)"),
        ([3, 1, 1], "3-stream: RGB (3ch) + Depth (1ch) + IR (1ch)"),
        ([3, 1, 64], "3-stream: RGB (3ch) + Depth (1ch) + Orthogonal (64ch)"),
        ([3, 3, 3], "3-stream: RGB + RGB + RGB (homogeneous)"),
    ]

    for stream_channels, description in configs:
        print(f"\n\n{'#'*80}")
        print(f"# {description}")
        print(f"{'#'*80}")
        run_benchmark_suite(stream_channels, batch_size, num_iters, device, use_torch_compile)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Benchmark padding vs sequential stream processing (GPU required)"
    )
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16, typical for SUN RGB-D)')
    parser.add_argument('--num-iters', type=int, default=100,
                       help='Number of iterations (default: 100)')
    parser.add_argument('--all', action='store_true',
                       help='Run all stream configurations')
    parser.add_argument('--torch-compile', action='store_true',
                       help='Apply torch.compile(mode="max-autotune") to functions')

    args = parser.parse_args()

    # Check for CUDA availability (mandatory)
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("❌ ERROR: CUDA not available!")
        print("="*80)
        print("\nThis benchmark requires a CUDA-enabled GPU to measure GPU parallelization.")
        print("Sequential vs batched performance can only be compared on GPU.")
        print("\nPlease run on a machine with:")
        print("  - NVIDIA GPU with CUDA support")
        print("  - PyTorch with CUDA enabled (torch.cuda.is_available() == True)")
        print("\nTo check your PyTorch installation:")
        print("  python -c 'import torch; print(torch.cuda.is_available())'")
        print("="*80 + "\n")
        sys.exit(1)

    device = 'cuda'

    print("\n" + "="*80)
    print("GPU STREAM PARALLELIZATION BENCHMARK")
    print("Comparing Sequential vs Batched (with Padding) Approaches")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*80)

    if args.all:
        run_all_benchmarks(args.batch_size, args.num_iters, device, args.torch_compile)
    else:
        # Default: 2-stream RGB + Depth (SUN RGB-D configuration)
        stream_channels = [3, 1]
        run_benchmark_suite(stream_channels, args.batch_size, args.num_iters, device, args.torch_compile)

        print("\n💡 Tips:")
        print("  - Run with --all to test multiple stream configurations")
        print("  - Run with --torch-compile to test torch.compile speedup")


if __name__ == '__main__':
    main()
