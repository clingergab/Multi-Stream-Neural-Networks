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

    Args:
        rgb_mean: RGB normalization mean (3 values). Defaults to SUN RGB-D stats.
        rgb_std: RGB normalization std (3 values). Defaults to SUN RGB-D stats.
        depth_mean: Depth normalization mean (1 value). Defaults to SUN RGB-D stats.
        depth_std: Depth normalization std (1 value). Defaults to SUN RGB-D stats.
        color_jitter_p: Probability of applying ColorJitter (default: 0.43)
        blur_p: Probability of applying Gaussian blur (default: 0.25)
        grayscale_p: Probability of converting to grayscale (default: 0.17)
        rgb_erasing_p: Probability of random erasing on RGB (default: 0.17)
        depth_erasing_p: Probability of random erasing on depth (default: 0.10)
        brightness: ColorJitter brightness range (default: 0.37)
        contrast: ColorJitter contrast range (default: 0.37)
        saturation: ColorJitter saturation range (default: 0.37)
        hue: ColorJitter hue range (default: 0.11)

    Example:
        >>> gpu_aug = GPUAugmentation().to('cuda')
        >>> model.gpu_aug = gpu_aug
        >>>
        >>> # In training (model.train() sets gpu_aug.training = True)
        >>> rgb_aug, depth_aug = gpu_aug(rgb_batch, depth_batch)
        >>>
        >>> # In validation (model.eval() sets gpu_aug.training = False)
        >>> rgb_norm, depth_norm = gpu_aug(rgb_batch, depth_batch)
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
        color_jitter_p: float = 0.43,
        blur_p: float = 0.25,
        grayscale_p: float = 0.17,
        rgb_erasing_p: float = 0.17,
        depth_erasing_p: float = 0.10,
        brightness: float = 0.37,
        contrast: float = 0.37,
        saturation: float = 0.37,
        hue: float = 0.11,
    ):
        super().__init__()

        if not KORNIA_AVAILABLE:
            raise ImportError(
                "Kornia is required for GPU augmentation. "
                "Install with: pip install kornia"
            )

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
                sigma=(0.1, 1.7),
                p=blur_p,
            ),
            K.RandomGrayscale(p=grayscale_p),
            same_on_batch=False,
            data_keys=["input"],
        )

        # Random erasing (applied after normalization)
        self.rgb_erasing = K.RandomErasing(
            scale=(0.02, 0.10),
            ratio=(0.5, 2.0),
            p=rgb_erasing_p,
        )
        self.depth_erasing = K.RandomErasing(
            scale=(0.02, 0.10),
            ratio=(0.5, 2.0),
            p=depth_erasing_p,
        )

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
