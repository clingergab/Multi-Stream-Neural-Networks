"""
GPU-based augmentation for multi-stream networks.

Moves expensive augmentations to GPU to reduce CPU bottleneck during
parallel Ray Tune trials.

CRITICAL: Input must be [0, 1] range (NOT normalized).
Normalization happens AFTER augmentation.

Pipeline (training mode):
    1. ColorJitter, Blur, Grayscale (RGB only)
    2. Normalize (both RGB and Depth)
    3. RandomErasing (both, after normalization)

Validation mode: Normalize only (no augmentation)
"""

import torch
import torch.nn as nn

from src.training.augmentation_config import (
    # Probability baselines
    BASE_COLOR_JITTER_P,
    BASE_BLUR_P,
    BASE_GRAYSCALE_P,
    BASE_RGB_ERASING_P,
    BASE_DEPTH_ERASING_P,
    # Magnitude baselines
    BASE_BRIGHTNESS,
    BASE_CONTRAST,
    BASE_SATURATION,
    BASE_HUE,
    BASE_BLUR_SIGMA_MIN,
    BASE_BLUR_SIGMA_MAX,
    BASE_ERASING_SCALE_MIN,
    BASE_ERASING_SCALE_MAX,
    BASE_ERASING_RATIO_MIN,
    BASE_ERASING_RATIO_MAX,
    # Caps
    MAX_PROBABILITY,
    MAX_BRIGHTNESS,
    MAX_CONTRAST,
    MAX_SATURATION,
    MAX_HUE,
    MAX_BLUR_SIGMA,
    MAX_ERASING_SCALE,
)

try:
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


class GPUAugmentation(nn.Module):
    """
    GPU-based augmentation for RGB-D data.

    Uses self.training attribute (toggled by .train()/.eval()) to determine
    whether to apply augmentations. This follows PyTorch conventions and
    automatically syncs with parent model's train/eval state.

    Note: Flip and crop augmentations are handled by the dataset (CPU-side) because
    they need to be synchronized between RGB and Depth. GPU augmentation only handles
    color transforms and erasing.

    Args:
        rgb_mean: RGB normalization mean (3 values). Defaults to SUN RGB-D stats.
        rgb_std: RGB normalization std (3 values). Defaults to SUN RGB-D stats.
        depth_mean: Depth normalization mean (1 value). Defaults to SUN RGB-D stats.
        depth_std: Depth normalization std (1 value). Defaults to SUN RGB-D stats.
        rgb_aug_prob: Scales probability of RGB augmentations (default: 1.0 = baseline)
        rgb_aug_mag: Scales magnitude of RGB augmentations (default: 1.0 = baseline)
        depth_aug_prob: Scales probability of Depth augmentations (default: 1.0 = baseline)
        depth_aug_mag: Scales magnitude of Depth augmentations (default: 1.0 = baseline)

    Example:
        >>> gpu_aug = GPUAugmentation().to('cuda')
        >>> model.gpu_aug = gpu_aug
        >>>
        >>> # In training (model.train() sets gpu_aug.training = True)
        >>> rgb_aug, depth_aug = gpu_aug(rgb_batch, depth_batch)
        >>>
        >>> # In validation (model.eval() sets gpu_aug.training = False)
        >>> rgb_norm, depth_norm = gpu_aug(rgb_batch, depth_batch)
        >>>
        >>> # With custom augmentation scaling
        >>> gpu_aug = GPUAugmentation(rgb_aug_prob=1.5, rgb_aug_mag=1.2).to('cuda')
    """

    # SUN RGB-D training set statistics (computed from 8041 samples)
    DEFAULT_RGB_MEAN = [0.4905626144214781, 0.4564359471868703, 0.43112756716677114]
    DEFAULT_RGB_STD = [0.27944652961530003, 0.2868739703756949, 0.29222326115669395]
    DEFAULT_DEPTH_MEAN = [0.2912]
    DEFAULT_DEPTH_STD = [0.1472]

    def __init__(
        self,
        rgb_mean: list[float] = None,
        rgb_std: list[float] = None,
        depth_mean: list[float] = None,
        depth_std: list[float] = None,
        rgb_aug_prob: float = 1.0,
        rgb_aug_mag: float = 1.0,
        depth_aug_prob: float = 1.0,
        depth_aug_mag: float = 1.0,
    ):
        super().__init__()

        if not KORNIA_AVAILABLE:
            raise ImportError(
                "Kornia is required for GPU augmentation. "
                "Install with: pip install kornia"
            )

        # Store augmentation scaling parameters for logging
        self.rgb_aug_prob = rgb_aug_prob
        self.rgb_aug_mag = rgb_aug_mag
        self.depth_aug_prob = depth_aug_prob
        self.depth_aug_mag = depth_aug_mag

        # Use defaults if not specified
        rgb_mean = rgb_mean or self.DEFAULT_RGB_MEAN
        rgb_std = rgb_std or self.DEFAULT_RGB_STD
        depth_mean = depth_mean or self.DEFAULT_DEPTH_MEAN
        depth_std = depth_std or self.DEFAULT_DEPTH_STD

        # Register normalization tensors as buffers so they move with .to(device)
        self.register_buffer('rgb_mean', torch.tensor(rgb_mean, dtype=torch.float32))
        self.register_buffer('rgb_std', torch.tensor(rgb_std, dtype=torch.float32))
        self.register_buffer('depth_mean', torch.tensor(depth_mean, dtype=torch.float32))
        self.register_buffer('depth_std', torch.tensor(depth_std, dtype=torch.float32))

        # === Compute scaled RGB augmentation values ===
        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        blur_p = min(BASE_BLUR_P * rgb_aug_prob, MAX_PROBABILITY)
        grayscale_p = min(BASE_GRAYSCALE_P * rgb_aug_prob, MAX_PROBABILITY)
        rgb_erasing_p = min(BASE_RGB_ERASING_P * rgb_aug_prob, MAX_PROBABILITY)

        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        contrast = min(BASE_CONTRAST * rgb_aug_mag, MAX_CONTRAST)
        saturation = min(BASE_SATURATION * rgb_aug_mag, MAX_SATURATION)
        hue = min(BASE_HUE * rgb_aug_mag, MAX_HUE)
        blur_sigma_max = min(BASE_BLUR_SIGMA_MAX * rgb_aug_mag, MAX_BLUR_SIGMA)
        rgb_erasing_scale_max = min(BASE_ERASING_SCALE_MAX * rgb_aug_mag, MAX_ERASING_SCALE)

        # === Compute scaled Depth augmentation values ===
        depth_erasing_p = min(BASE_DEPTH_ERASING_P * depth_aug_prob, MAX_PROBABILITY)
        depth_erasing_scale_max = min(BASE_ERASING_SCALE_MAX * depth_aug_mag, MAX_ERASING_SCALE)

        # Store computed values for logging
        self._color_jitter_p = color_jitter_p
        self._blur_p = blur_p
        self._grayscale_p = grayscale_p
        self._rgb_erasing_p = rgb_erasing_p
        self._brightness = brightness
        self._depth_erasing_p = depth_erasing_p

        # RGB augmentation pipeline (applied before normalization)
        # Note: Kornia's RandomGaussianBlur with kernel_size=(3, 7) samples
        # only odd values (3, 5, 7) from the range, as required for Gaussian blur
        self.rgb_augment = K.AugmentationSequential(
            K.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=color_jitter_p,
            ),
            K.RandomGaussianBlur(
                kernel_size=(3, 7),
                sigma=(BASE_BLUR_SIGMA_MIN, blur_sigma_max),
                p=blur_p,
            ),
            K.RandomGrayscale(p=grayscale_p),
            same_on_batch=False,
            data_keys=["input"],
        )

        # Random erasing (applied after normalization)
        self.rgb_erasing = K.RandomErasing(
            scale=(BASE_ERASING_SCALE_MIN, rgb_erasing_scale_max),
            ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX),
            p=rgb_erasing_p,
        )
        self.depth_erasing = K.RandomErasing(
            scale=(BASE_ERASING_SCALE_MIN, depth_erasing_scale_max),
            ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX),
            p=depth_erasing_p,
        )

        # Log augmentation config if scaling is applied
        if any(p != 1.0 for p in [rgb_aug_prob, rgb_aug_mag, depth_aug_prob, depth_aug_mag]):
            self._log_augmentation_config()

    def _log_augmentation_config(self):
        """Log computed augmentation values when scaling is applied."""
        print(f"\nGPU Augmentation scaling applied:")
        print(f"  RGB:   prob={self.rgb_aug_prob:.2f}, mag={self.rgb_aug_mag:.2f}")
        print(f"  Depth: prob={self.depth_aug_prob:.2f}, mag={self.depth_aug_mag:.2f}")
        print(f"  Computed values:")
        print(f"    [RGB]   ColorJitter prob: {BASE_COLOR_JITTER_P:.2f} -> {self._color_jitter_p:.3f}")
        print(f"    [RGB]   Brightness: ±{BASE_BRIGHTNESS:.2f} -> ±{self._brightness:.3f}")
        print(f"    [RGB]   Blur prob: {BASE_BLUR_P:.2f} -> {self._blur_p:.3f}")
        print(f"    [RGB]   Grayscale prob: {BASE_GRAYSCALE_P:.2f} -> {self._grayscale_p:.3f}")
        print(f"    [RGB]   Erasing prob: {BASE_RGB_ERASING_P:.2f} -> {self._rgb_erasing_p:.3f}")
        print(f"    [Depth] Erasing prob: {BASE_DEPTH_ERASING_P:.2f} -> {self._depth_erasing_p:.3f}")

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply GPU augmentation and normalization.

        Args:
            rgb: RGB tensor [B, 3, H, W] in [0, 1] range (NOT normalized)
            depth: Depth tensor [B, 1, H, W] in [0, 1] range (NOT normalized)

        Returns:
            Tuple of (rgb, depth) tensors, both normalized.
            In training mode, augmentations are also applied.
        """
        # Step 1: RGB augmentation (training only)
        if self.training:
            rgb = self.rgb_augment(rgb)

        # Step 2: Normalize using registered buffers (already on correct device)
        rgb = (rgb - self.rgb_mean.view(1, 3, 1, 1)) / self.rgb_std.view(1, 3, 1, 1)
        depth = (depth - self.depth_mean.view(1, 1, 1, 1)) / self.depth_std.view(1, 1, 1, 1)

        # Step 3: Random erasing (training only, after normalization)
        if self.training:
            rgb = self.rgb_erasing(rgb)
            depth = self.depth_erasing(depth)

        return rgb, depth

    def __repr__(self) -> str:
        return (
            f"GPUAugmentation(\n"
            f"  rgb_mean={self.rgb_mean.tolist()},\n"
            f"  rgb_std={self.rgb_std.tolist()},\n"
            f"  depth_mean={self.depth_mean.tolist()},\n"
            f"  depth_std={self.depth_std.tolist()},\n"
            f"  rgb_augment={self.rgb_augment},\n"
            f"  rgb_erasing={self.rgb_erasing},\n"
            f"  depth_erasing={self.depth_erasing},\n"
            f")"
        )
