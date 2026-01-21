"""
Modality dropout utilities for training multi-stream networks.

Provides per-sample modality dropout to train models to handle missing streams.
"""

import torch
from typing import Optional


def get_modality_dropout_prob(
    epoch: int,
    start_epoch: int = 0,
    ramp_epochs: int = 20,
    final_rate: float = 0.2
) -> float:
    """
    Compute modality dropout probability based on current epoch.

    Schedule:
    - Before start_epoch: 0% dropout
    - During ramp (start_epoch to start_epoch + ramp_epochs): Linear ramp 0% → final_rate
    - After ramp: final_rate

    Args:
        epoch: Current training epoch (0-indexed)
        start_epoch: Epoch to start dropout (default: 0)
        ramp_epochs: Number of epochs to ramp from 0 to final_rate (default: 20)
        final_rate: Final dropout probability (default: 0.2 = 20%)

    Returns:
        Dropout probability for this epoch
    """
    if epoch < start_epoch:
        return 0.0
    epochs_since_start = epoch - start_epoch
    ramp_epochs = max(1, ramp_epochs)  # Prevent division by zero
    if epochs_since_start < ramp_epochs:
        return final_rate * (epochs_since_start / ramp_epochs)
    return final_rate


def generate_per_sample_blanked_mask(
    batch_size: int,
    num_streams: int,
    dropout_prob: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None
) -> Optional[dict[int, torch.Tensor]]:
    """
    Generate per-sample blanked mask for modality dropout.

    Rules:
    - Each sample can have at most ONE stream blanked
    - Never blank both streams for the same sample
    - dropout_prob is the per-sample probability of blanking ANY stream

    Args:
        batch_size: Number of samples in batch
        num_streams: Number of input streams (e.g., 2 for RGB+Depth)
        dropout_prob: Per-sample probability of blanking a stream
        device: Device to create tensors on
        generator: Optional random generator for reproducibility

    Returns:
        None if dropout_prob <= 0 or no samples selected for dropout
        dict mapping stream_idx -> bool tensor [batch_size] where True = blanked
    """
    if dropout_prob <= 0.0:
        return None

    # Step 1: Decide which samples get a stream blanked
    should_blank = torch.rand(batch_size, device=device, generator=generator) < dropout_prob

    # Step 2: For blanked samples, randomly pick which stream to blank
    stream_to_blank = torch.randint(0, num_streams, (batch_size,), device=device, generator=generator)

    # Step 3: Create per-stream masks
    blanked_mask = {}
    for stream_idx in range(num_streams):
        blanked_mask[stream_idx] = should_blank & (stream_to_blank == stream_idx)

    # Return None if no samples are blanked (optimization)
    if not any(mask.any() for mask in blanked_mask.values()):
        return None

    return blanked_mask
