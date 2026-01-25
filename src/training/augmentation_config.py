"""
Shared augmentation baseline constants for CPU and GPU paths.

This module provides a single source of truth for augmentation parameters,
ensuring consistency between dataset (CPU) and GPU augmentation pipelines.

Usage:
    from src.training.augmentation_config import AugmentationConfig, BASE_FLIP_P, ...

    # Create config for experiment
    aug_config = AugmentationConfig(rgb_aug_prob=1.5, rgb_aug_mag=1.2)

    # Pass to dataloader and model
    train_loader = get_sunrgbd_dataloaders(..., **aug_config.to_dict())
    model.compile(..., **aug_config.to_dict())
"""

import warnings
from dataclasses import dataclass, asdict


# =============================================================================
# PROBABILITY BASELINES (aug_prob scales these)
# =============================================================================

# Synchronized augmentations (applied identically to RGB and Depth)
BASE_FLIP_P = 0.5
BASE_CROP_P = 0.5

# RGB-only augmentations
BASE_COLOR_JITTER_P = 0.43
BASE_BLUR_P = 0.25
BASE_GRAYSCALE_P = 0.17
BASE_RGB_ERASING_P = 0.17

# Depth-only augmentations
BASE_DEPTH_AUG_P = 0.50  # Combined brightness/contrast/noise block
BASE_DEPTH_ERASING_P = 0.10


# =============================================================================
# MAGNITUDE BASELINES (aug_mag scales these)
# =============================================================================

# Synchronized augmentations
BASE_CROP_SCALE_MIN = 0.9  # Minimum crop scale (0.9 = 90% of image)
BASE_CROP_SCALE_MAX = 1.0
BASE_CROP_RATIO_MIN = 0.95
BASE_CROP_RATIO_MAX = 1.05

# RGB ColorJitter magnitudes
BASE_BRIGHTNESS = 0.37
BASE_CONTRAST = 0.37
BASE_SATURATION = 0.37
BASE_HUE = 0.11

# RGB Blur magnitudes
BASE_BLUR_SIGMA_MIN = 0.1
BASE_BLUR_SIGMA_MAX = 1.7

# RGB/Depth erasing magnitudes
BASE_ERASING_SCALE_MIN = 0.02
BASE_ERASING_SCALE_MAX = 0.10
BASE_ERASING_RATIO_MIN = 0.5
BASE_ERASING_RATIO_MAX = 2.0

# Depth appearance magnitudes
BASE_DEPTH_BRIGHTNESS = 0.25  # ±25% brightness factor
BASE_DEPTH_CONTRAST = 0.25    # ±25% contrast factor
BASE_DEPTH_NOISE_STD = 0.059  # Gaussian noise std (≈15/255)


# =============================================================================
# CAPS TO PREVENT EXTREME VALUES
# =============================================================================

MAX_PROBABILITY = 0.95  # Never go above 95% probability

# RGB caps
MAX_BRIGHTNESS = 0.8
MAX_CONTRAST = 0.8
MAX_SATURATION = 0.8
MAX_HUE = 0.4
MAX_BLUR_SIGMA = 3.5

# Depth caps
MAX_DEPTH_BRIGHTNESS = 0.5
MAX_DEPTH_CONTRAST = 0.5
MAX_DEPTH_NOISE_STD = 0.15

# Crop caps
MIN_CROP_SCALE = 0.5  # Never crop smaller than 50% of image

# Erasing caps
MAX_ERASING_SCALE = 0.25  # Never erase more than 25% of image


# =============================================================================
# AUGMENTATION CONFIG DATACLASS
# =============================================================================

@dataclass
class AugmentationConfig:
    """
    Convenience wrapper for augmentation parameters.

    Use this to pass augmentation settings consistently to both
    dataloader and model.compile().

    Args:
        rgb_aug_prob: Scales probability of RGB augmentations (default: 1.0)
        rgb_aug_mag: Scales magnitude/intensity of RGB augmentations (default: 1.0)
        depth_aug_prob: Scales probability of Depth augmentations (default: 1.0)
        depth_aug_mag: Scales magnitude/intensity of Depth augmentations (default: 1.0)

    Example:
        >>> # Per-stream control
        >>> config = AugmentationConfig(
        ...     rgb_aug_prob=1.5,    # 50% more frequent RGB aug
        ...     rgb_aug_mag=1.2,     # 20% stronger RGB aug
        ...     depth_aug_prob=1.0,  # Depth frequency unchanged
        ...     depth_aug_mag=0.8,   # 20% weaker depth aug
        ... )
        >>>
        >>> # Same scaling for both streams
        >>> config = AugmentationConfig.uniform(aug_prob=1.5, aug_mag=1.2)
        >>>
        >>> # Use in dataloader and model
        >>> train_loader = get_sunrgbd_dataloaders(**config.to_dict())
        >>> model.compile(**config.to_dict())
    """

    rgb_aug_prob: float = 1.0
    rgb_aug_mag: float = 1.0
    depth_aug_prob: float = 1.0
    depth_aug_mag: float = 1.0

    def __post_init__(self):
        """Validate parameters after initialization."""
        for name, val in asdict(self).items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
            if val > 5.0:
                warnings.warn(
                    f"{name}={val} is unusually high (>5.0). "
                    "This may cause extreme augmentation.",
                    UserWarning,
                    stacklevel=2,
                )

    @classmethod
    def uniform(cls, aug_prob: float = 1.0, aug_mag: float = 1.0) -> "AugmentationConfig":
        """
        Create config with same scaling for both modalities.

        Args:
            aug_prob: Probability scaling for both RGB and Depth
            aug_mag: Magnitude scaling for both RGB and Depth

        Returns:
            AugmentationConfig with uniform settings
        """
        return cls(
            rgb_aug_prob=aug_prob,
            rgb_aug_mag=aug_mag,
            depth_aug_prob=aug_prob,
            depth_aug_mag=aug_mag,
        )

    def to_dict(self) -> dict:
        """
        Convert to dictionary for unpacking into function calls.

        Useful for logging to W&B or passing to functions:
            >>> wandb.config.update(config.to_dict())
            >>> get_sunrgbd_dataloaders(**config.to_dict())
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"AugmentationConfig("
            f"rgb_aug_prob={self.rgb_aug_prob}, "
            f"rgb_aug_mag={self.rgb_aug_mag}, "
            f"depth_aug_prob={self.depth_aug_prob}, "
            f"depth_aug_mag={self.depth_aug_mag})"
        )
