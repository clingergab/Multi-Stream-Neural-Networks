"""
Test script to measure GPU memory consumption with and without modality dropout.

This script creates a minimal test case to isolate the memory impact of modality dropout.
"""

import torch
import torch.nn.functional as F
from src.training.modality_dropout import generate_per_sample_blanked_mask


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def test_conv_forward_with_masking(batch_size=32, in_channels=64, out_channels=128,
                                   img_size=224, with_masking=True, device='cuda'):
    """Simulate one forward pass through a conv layer with/without masking."""

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create dummy input
    stream_input = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
    conv_weight = torch.randn(out_channels, in_channels, 3, 3, device=device)
    conv_bias = torch.randn(out_channels, device=device)

    mem_before = get_gpu_memory_mb()
    peak_before = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Forward pass
    stream_out_raw = F.conv2d(stream_input, conv_weight, None, stride=1, padding=1)
    stream_out = stream_out_raw + conv_bias.view(1, -1, 1, 1)

    if with_masking:
        # Generate blanked mask (20% dropout rate)
        blanked_mask = generate_per_sample_blanked_mask(
            batch_size=batch_size,
            num_streams=2,  # Assume 2 streams
            dropout_prob=0.2,
            device=device
        )

        # Apply masking (this is what happens in LIConv2d)
        if blanked_mask is not None:
            stream_blanked = blanked_mask.get(0)  # Stream 0
            if stream_blanked is not None and stream_blanked.any():
                # Create mask and apply
                mask = (~stream_blanked).float().view(-1, 1, 1, 1)
                stream_out = stream_out * mask
                stream_out_raw = stream_out_raw * mask

    # Force computation
    result = stream_out.sum()

    mem_after = get_gpu_memory_mb()
    peak_after = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        'mem_before': mem_before,
        'mem_after': mem_after,
        'mem_delta': mem_after - mem_before,
        'peak_before': peak_before,
        'peak_after': peak_after,
        'peak_delta': peak_after - peak_before
    }


def test_batchnorm_with_subset(batch_size=32, channels=128, img_size=224,
                               with_masking=True, device='cuda'):
    """Test BatchNorm with subset processing (for blanked samples)."""

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create dummy input
    stream_input = torch.randn(batch_size, channels, img_size, img_size, device=device)

    # BatchNorm parameters
    running_mean = torch.zeros(channels, device=device)
    running_var = torch.ones(channels, device=device)
    weight = torch.ones(channels, device=device)
    bias = torch.zeros(channels, device=device)

    mem_before = get_gpu_memory_mb()
    peak_before = torch.cuda.max_memory_allocated() / 1024 / 1024

    if with_masking:
        # Simulate partial blanking (20% dropout)
        blanked_mask = generate_per_sample_blanked_mask(
            batch_size=batch_size,
            num_streams=2,
            dropout_prob=0.2,
            device=device
        )

        if blanked_mask is not None:
            stream_blanked = blanked_mask.get(0)
            if stream_blanked is not None and stream_blanked.any() and not stream_blanked.all():
                # Subset BN approach (THIS is the memory-intensive operation)
                active_idx = (~stream_blanked).nonzero(as_tuple=True)[0]  # GPU operation
                active_input = stream_input[active_idx]  # Indexing operation - creates new tensor

                # BN on active samples
                active_output = F.batch_norm(
                    active_input, running_mean, running_var, weight, bias,
                    training=True, momentum=0.1, eps=1e-5
                )

                # Scatter back - creates another full-size tensor
                stream_out = torch.zeros_like(stream_input)
                stream_out[active_idx] = active_output
            else:
                # No partial blanking
                stream_out = F.batch_norm(
                    stream_input, running_mean, running_var, weight, bias,
                    training=True, momentum=0.1, eps=1e-5
                )
        else:
            stream_out = F.batch_norm(
                stream_input, running_mean, running_var, weight, bias,
                training=True, momentum=0.1, eps=1e-5
            )
    else:
        # Standard BN without masking
        stream_out = F.batch_norm(
            stream_input, running_mean, running_var, weight, bias,
            training=True, momentum=0.1, eps=1e-5
        )

    # Force computation
    result = stream_out.sum()

    mem_after = get_gpu_memory_mb()
    peak_after = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        'mem_before': mem_before,
        'mem_after': mem_after,
        'mem_delta': mem_after - mem_before,
        'peak_before': peak_before,
        'peak_after': peak_after,
        'peak_delta': peak_after - peak_before
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU memory tests.")
        exit(0)

    device = 'cuda'
    print(f"Testing on device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Test 1: Conv layer with masking
    print("\n1. CONV LAYER MASKING TEST")
    print("-" * 80)

    results_no_mask = test_conv_forward_with_masking(with_masking=False, device=device)
    print(f"Without masking:")
    print(f"  Memory delta: {results_no_mask['mem_delta']:.2f} MB")
    print(f"  Peak delta: {results_no_mask['peak_delta']:.2f} MB")

    results_with_mask = test_conv_forward_with_masking(with_masking=True, device=device)
    print(f"\nWith masking:")
    print(f"  Memory delta: {results_with_mask['mem_delta']:.2f} MB")
    print(f"  Peak delta: {results_with_mask['peak_delta']:.2f} MB")

    overhead = results_with_mask['peak_delta'] - results_no_mask['peak_delta']
    print(f"\nMasking overhead: {overhead:.2f} MB ({overhead/results_no_mask['peak_delta']*100:.1f}%)")

    # Test 2: BatchNorm with subset processing
    print("\n" + "=" * 80)
    print("\n2. BATCHNORM SUBSET PROCESSING TEST")
    print("-" * 80)

    results_bn_no_mask = test_batchnorm_with_subset(with_masking=False, device=device)
    print(f"Without masking:")
    print(f"  Memory delta: {results_bn_no_mask['mem_delta']:.2f} MB")
    print(f"  Peak delta: {results_bn_no_mask['peak_delta']:.2f} MB")

    results_bn_with_mask = test_batchnorm_with_subset(with_masking=True, device=device)
    print(f"\nWith masking:")
    print(f"  Memory delta: {results_bn_with_mask['mem_delta']:.2f} MB")
    print(f"  Peak delta: {results_bn_with_mask['peak_delta']:.2f} MB")

    overhead_bn = results_bn_with_mask['peak_delta'] - results_bn_no_mask['peak_delta']
    print(f"\nSubset BN overhead: {overhead_bn:.2f} MB ({overhead_bn/results_bn_no_mask['peak_delta']*100:.1f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("\nSUMMARY")
    print("-" * 80)
    print(f"Total estimated overhead per layer: {overhead + overhead_bn:.2f} MB")
    print(f"\nFor a typical network with ~50 layers (25 Conv + 25 BN):")
    print(f"  Estimated total overhead: {(overhead + overhead_bn) * 25:.2f} MB")
    print(f"\nNote: This is a conservative estimate. Actual overhead may vary based on:")
    print(f"  - Batch size")
    print(f"  - Number of channels")
    print(f"  - Image resolution")
    print(f"  - Dropout probability (affects how many samples trigger subset processing)")
