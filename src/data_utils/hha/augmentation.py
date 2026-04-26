"""Shared HHA-aware data augmentation used by SUN RGB-D and ScanNet datasets.

Single source of truth for the HHA augmentation pipeline so both datasets
stay in sync. SUN-tuned magnitudes are NOT baked into this module -- they
live as constructor args via ``HHAAugConfig`` so each dataset can supply
its own defaults without forking the implementation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F2


@dataclass
class HHAAugConfig:
    """Per-aug magnitudes + probabilities for the HHA pipeline.

    All probabilities are pre-cap base values (``MAX_PROBABILITY=0.95`` is
    applied at the use site by the dataset, not here). Magnitudes are
    pre-cap base values; same convention.

    SUN RGB-D defaults match the historical
    ``BASE_DISP_SCALE = 0.15``, ``BASE_HEIGHT_SIGMA = 0.05``, etc.
    ScanNet starts from the same defaults but is free to override.
    """
    disp_scale_p: float = 0.5
    disp_scale_halfwidth: float = 0.15

    height_offset_p: float = 0.5
    height_sigma_m: float = 0.05

    blur_p: float = 0.25
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.7
    blur_kernel_choices: tuple[int, ...] = (3, 5)

    hole_p: float = 0.30
    hole_num_min: int = 3
    hole_num_max: int = 8
    hole_size_min: int = 5    # pixels
    hole_size_max: int = 20   # pixels


def apply_hha_aug(
    hha: torch.Tensor,
    cfg: HHAAugConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Apply HHA-aware data augmentation. Caller passes a [3, H, W] tensor
    that may contain NaN at invalid pixels.

    **Canonical operation order (don't reorder):**

      1. Disparity scale jitter -- channel 0 only. Scalar multiply.
         Passes NaN through unchanged. Safe.
      2. Height offset jitter -- channel 1 only. Scalar add (only at valid pixels).
         Passes NaN through unchanged. Safe.
      3. Knutsson (mask-aware) blur -- applied to all 3 channels.
         NaN-aware kernel; preserves NaN regions exactly.
      4. Hole dropout -- sets a rectangular region to NaN across all 3 channels.
         Creates new NaN regions; MUST be last among NaN-introducing ops.

    The downstream dataset code is responsible for the *final* NaN ->
    per-channel-mean replacement immediately before normalization (so the
    model never sees NaN). That replacement is unconditional in
    ``_normalize_and_erase`` and must run after this function returns.

    Steps 1 and 2 are independent (different channels) and can be reordered
    with each other but must come before steps 3 and 4. Step 3 must come
    before step 4. Step 4 must come last; otherwise step 3's mask-aware
    blur would average across the holes.

    The function clones the input tensor before mutating, so callers do
    not need to clone defensively.
    """
    # Defensive clone: ops below mutate via channel assignment.
    hha = hha.clone()

    if rng is None:
        rng = np.random.default_rng()

    # 1. Disparity scale jitter (channel 0 only).
    if cfg.disp_scale_halfwidth > 0 and rng.random() < cfg.disp_scale_p:
        half = cfg.disp_scale_halfwidth
        s = float(rng.uniform(1.0 - half, 1.0 + half))
        hha[0] = hha[0] * s

    # 2. Ground-plane offset (channel 1 only). Additive Gaussian on valid
    # pixels (preserves NaN at invalid pixels exactly).
    if cfg.height_sigma_m > 0 and rng.random() < cfg.height_offset_p:
        offset = float(rng.normal(0.0, cfg.height_sigma_m))
        valid = ~torch.isnan(hha[1])
        hha[1] = torch.where(valid, hha[1] + offset, hha[1])

    # 3. Knutsson blur -- mask-aware so NaN pixels don't drag boundary-valid
    # pixels toward zero. Without this, the angle channel's signal at glass /
    # depth-discontinuity edges (where NaN clusters) would be destroyed.
    if cfg.blur_sigma_max > 0 and rng.random() < cfg.blur_p:
        kernel_size = int(rng.choice(cfg.blur_kernel_choices))
        sigma = float(rng.uniform(cfg.blur_sigma_min, cfg.blur_sigma_max))
        nan_mask = torch.isnan(hha)
        valid = (~nan_mask).to(hha.dtype)
        clean = torch.where(nan_mask, torch.zeros_like(hha), hha)
        blurred_data = F2.gaussian_blur(clean, kernel_size=kernel_size, sigma=sigma)
        blurred_mask = F2.gaussian_blur(valid, kernel_size=kernel_size, sigma=sigma)
        hha = torch.where(
            blurred_mask > 1e-6,
            blurred_data / torch.clamp(blurred_mask, min=1e-6),
            torch.full_like(hha, float('nan')),
        )
        # Re-mark originally-NaN pixels.
        hha = torch.where(nan_mask, torch.full_like(hha, float('nan')), hha)

    # 4. Hole dropout (creates NaN -- must be last).
    if cfg.hole_num_max >= cfg.hole_num_min and rng.random() < cfg.hole_p:
        _H, _W = hha.shape[-2], hha.shape[-1]
        n_holes = int(rng.integers(cfg.hole_num_min, cfg.hole_num_max + 1))
        for _ in range(n_holes):
            h = int(rng.integers(cfg.hole_size_min, cfg.hole_size_max + 1))
            w = int(rng.integers(cfg.hole_size_min, cfg.hole_size_max + 1))
            i = int(rng.integers(0, max(1, _H - h)))
            j = int(rng.integers(0, max(1, _W - w)))
            hha[:, i:i + h, j:j + w] = float('nan')

    return hha


def replace_nan_with_per_channel_mean(
    hha: torch.Tensor,
    means: list[float],
) -> torch.Tensor:
    """Replace NaN pixels in a [3, H, W] HHA tensor with the per-channel mean.

    Used by datasets in their ``_normalize_and_erase`` step BEFORE
    normalization, so the model never sees NaN regardless of whether the
    dataset is operating in CPU or GPU augmentation mode.

    Operates out-of-place; returns a new tensor.
    """
    if hha.shape[0] != len(means):
        raise ValueError(
            f"means length {len(means)} does not match tensor channels {hha.shape[0]}"
        )
    nan_mask = torch.isnan(hha)
    if not nan_mask.any():
        return hha
    mean_t = torch.tensor(
        means, dtype=hha.dtype, device=hha.device
    ).view(-1, 1, 1)
    return torch.where(nan_mask, mean_t.expand_as(hha), hha)
