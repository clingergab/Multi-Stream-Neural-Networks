"""
Data augmentation utilities for multi-stream neural networks.
Provides comprehensive augmentation strategies for different datasets to reduce overfitting.
"""

import torch
from typing import Tuple, Optional, Dict, Any
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import warnings


class BaseAugmentation:
    """
    Base class for dataset-specific augmentations.
    
    Provides a common interface and shared functionality for different dataset augmentations.
    """
    
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        rotation_degrees: float = 15.0,
        translate_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        color_jitter_strength: float = 0.4,
        gaussian_noise_std: float = 0.02,
        cutout_prob: float = 0.5,
        cutout_size: int = 8,
        enabled: bool = True
    ):
        """
        Initialize base augmentation parameters.
        
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            rotation_degrees: Maximum rotation in degrees
            translate_range: Translation range as fraction of image size
            scale_range: Scale range (min, max)
            color_jitter_strength: Strength of color jittering
            gaussian_noise_std: Standard deviation for Gaussian noise
            cutout_prob: Probability of applying cutout
            cutout_size: Size of cutout square
            enabled: Whether augmentation is enabled
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_degrees = rotation_degrees
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.color_jitter_strength = color_jitter_strength
        self.gaussian_noise_std = gaussian_noise_std
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size
        self.enabled = enabled
        
        # Create torchvision transforms for color jittering
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_strength * 0.8,
            contrast=color_jitter_strength * 0.8,
            saturation=color_jitter_strength * 0.8,
            hue=color_jitter_strength * 0.2
        )
        
        print(f"🎨 {self.__class__.__name__} initialized:")
        print(f"   Horizontal flip: {horizontal_flip_prob}")
        print(f"   Rotation: ±{rotation_degrees}°")
        print(f"   Translation: ±{translate_range}")
        print(f"   Scale: {scale_range}")
        print(f"   Color jitter: {color_jitter_strength}")
        print(f"   Gaussian noise: σ={gaussian_noise_std}")
        print(f"   Cutout: {cutout_prob} prob, {cutout_size}px")
        print(f"   Enabled: {enabled}")
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a single image.
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            Augmented image tensor with same shape
        """
        if not self.enabled:
            return image
        
        # Handle batch dimension
        is_batch = len(image.shape) == 4
        if not is_batch:
            image = image.unsqueeze(0)
        
        # Apply augmentations
        image = self._apply_spatial_transforms(image)
        image = self._apply_color_jitter(image)
        image = self._apply_gaussian_noise(image)
        image = self._apply_cutout(image)
        
        # Remove batch dimension if it was added
        if not is_batch:
            image = image.squeeze(0)
        
        return image
    
    def _apply_spatial_transforms(self, image: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformations (flip, rotation, translation, scale)."""
        batch_size = image.shape[0]
        height, width = image.shape[2], image.shape[3]
        
        for i in range(batch_size):
            # Horizontal flip
            if torch.rand(1) < self.horizontal_flip_prob:
                image[i] = TF.hflip(image[i])
            
            # Random affine transformation (rotation, translation, scale)
            if torch.rand(1) < 0.8:  # Apply with 80% probability
                angle = torch.empty(1).uniform_(-self.rotation_degrees, self.rotation_degrees).item()
                translate = [
                    int(torch.empty(1).uniform_(-self.translate_range, self.translate_range).item() * width),
                    int(torch.empty(1).uniform_(-self.translate_range, self.translate_range).item() * height)
                ]
                scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
                
                image[i] = TF.affine(
                    image[i],
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[0.0],
                    fill=[0.0]
                )
        
        return image
    
    def _apply_color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jittering to RGB channels."""
        batch_size = image.shape[0]
        
        for i in range(batch_size):
            if torch.rand(1) < 0.8:  # Apply with 80% probability
                # Only apply to RGB channels (first 3 channels)
                if image.shape[1] >= 3:
                    rgb_channels = image[i, :3]
                    rgb_jittered = self.color_jitter(rgb_channels)
                    image[i, :3] = rgb_jittered
        
        return image
    
    def _apply_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        if torch.rand(1) < 0.3:  # Apply with 30% probability
            noise = torch.randn_like(image) * self.gaussian_noise_std
            image = torch.clamp(image + noise, 0.0, 1.0)
        
        return image
    
    def _apply_cutout(self, image: torch.Tensor) -> torch.Tensor:
        """Apply cutout augmentation (random square masks)."""
        batch_size, channels, height, width = image.shape
        
        for i in range(batch_size):
            if torch.rand(1) < self.cutout_prob:
                # Random position for cutout
                y = torch.randint(0, height, (1,)).item()
                x = torch.randint(0, width, (1,)).item()
                
                # Calculate cutout bounds
                y1 = max(0, y - self.cutout_size // 2)
                y2 = min(height, y + self.cutout_size // 2)
                x1 = max(0, x - self.cutout_size // 2)
                x2 = min(width, x + self.cutout_size // 2)
                
                # Apply cutout (set to 0)
                image[i, :, y1:y2, x1:x2] = 0.0
        
        return image
    
    def augment_batch(
        self, 
        color_data: torch.Tensor, 
        brightness_data: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply consistent augmentation to both color and brightness data.
        
        Args:
            color_data: Color images [B, C, H, W]
            brightness_data: Brightness images [B, C, H, W]
            labels: Labels [B]
            
        Returns:
            Tuple of (augmented_color, augmented_brightness, labels)
        """
        if not self.enabled:
            return color_data, brightness_data, labels
        
        batch_size = color_data.shape[0]
        
        # Generate consistent random parameters for both streams
        for i in range(batch_size):
            # Store random state
            random_state = torch.get_rng_state()
            
            # Apply to color data
            color_data[i] = self(color_data[i])
            
            # Restore random state and apply same transforms to brightness
            torch.set_rng_state(random_state)
            brightness_data[i] = self(brightness_data[i])
        
        return color_data, brightness_data, labels


class CIFAR100Augmentation(BaseAugmentation):
    """
    Comprehensive data augmentation for CIFAR-100 dataset.
    
    Includes spatial transformations, color jittering, and noise injection
    specifically tuned for CIFAR-100's 32x32 image resolution.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize CIFAR-100 specific augmentation.
        
        Default values are tuned for CIFAR-100's 32x32 images.
        """
        # CIFAR-100 specific defaults
        cifar_defaults = {
            'rotation_degrees': 15.0,  # Limited rotation for small images
            'translate_range': 0.1,    # Small translation (3-4 pixels)
            'cutout_size': 8,          # 8x8 pixel cutout (25% of image)
        }
        
        # Update with CIFAR-100 defaults if not specified
        for key, value in cifar_defaults.items():
            if key not in kwargs:
                kwargs[key] = value
                
        super().__init__(**kwargs)
        print("   Dataset: CIFAR-100 (32x32 images)")


class ImageNetAugmentation(BaseAugmentation):
    """
    Comprehensive data augmentation for ImageNet dataset.
    
    Includes spatial transformations, color jittering, and noise injection
    tuned for ImageNet's larger resolution images.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize ImageNet specific augmentation.
        
        Default values are tuned for ImageNet's larger images.
        """
        # ImageNet specific defaults
        imagenet_defaults = {
            'rotation_degrees': 20.0,    # More rotation for larger images
            'translate_range': 0.05,     # Smaller relative translation
            'cutout_size': 56,           # Larger cutout for higher resolution
            'color_jitter_strength': 0.5  # Stronger color augmentation
        }
        
        # Update with ImageNet defaults if not specified
        for key, value in imagenet_defaults.items():
            if key not in kwargs:
                kwargs[key] = value
                
        super().__init__(**kwargs)
        print("   Dataset: ImageNet (224x224 images)")


class MixUp:
    """MixUp augmentation for multi-stream data."""
    
    def __init__(self, alpha=1.0):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Parameter for beta distribution. Higher values mean more mixing.
        """
        self.alpha = alpha
        print(f"🔄 MixUp initialized with alpha={alpha}")
    
    def __call__(self, batch_color, batch_brightness, batch_targets):
        """
        Apply MixUp to a batch.
        
        Args:
            batch_color: Color images [B, C, H, W]
            batch_brightness: Brightness images [B, C, H, W]
            batch_targets: Labels [B]
            
        Returns:
            Tuple of (mixed_color, mixed_brightness, targets_a, targets_b, lam)
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0
        
        batch_size = batch_color.size(0)
        indices = torch.randperm(batch_size)
        
        # Mix inputs
        mixed_color = lam * batch_color + (1 - lam) * batch_color[indices]
        mixed_brightness = lam * batch_brightness + (1 - lam) * batch_brightness[indices]
        
        # Mix targets
        targets_a = batch_targets
        targets_b = batch_targets[indices]
        
        return mixed_color, mixed_brightness, targets_a, targets_b, lam


class AugmentedMultiStreamDataset(Dataset):
    """
    Dataset wrapper with on-the-fly augmentation for multi-stream neural networks.
    """
    
    def __init__(
        self,
        color_data: torch.Tensor,
        brightness_data: torch.Tensor,
        labels: torch.Tensor,
        augmentation: Optional[BaseAugmentation] = None,
        mixup: Optional[MixUp] = None,
        train: bool = True
    ):
        """
        Initialize augmented dataset.
        
        Args:
            color_data: Color images [N, C, H, W]
            brightness_data: Brightness images [N, C, H, W]
            labels: Labels [N]
            augmentation: Augmentation instance (None for validation/test)
            mixup: MixUp augmentation instance (None to disable)
            train: Whether this is training data
        """
        self.color_data = color_data
        self.brightness_data = brightness_data
        self.labels = labels
        self.augmentation = augmentation if train else None
        self.mixup = mixup if train else None
        self.train = train
        
        # Get image shape for dataset identification
        height, width = color_data.shape[2:4]
        dataset_name = "CIFAR-100" if height == 32 and width == 32 else "ImageNet" if height >= 224 else "Custom"
        
        print("📊 AugmentedMultiStreamDataset created:")
        print(f"   Dataset: {dataset_name} ({height}x{width} images)")
        print(f"   Mode: {'Training' if train else 'Validation/Test'}")
        print(f"   Samples: {len(labels)}")
        print(f"   Color shape: {color_data.shape}")
        print(f"   Brightness shape: {brightness_data.shape}")
        print(f"   Augmentation: {'Enabled' if self.augmentation else 'Disabled'}")
        print(f"   MixUp: {'Enabled' if self.mixup else 'Disabled'}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        color = self.color_data[idx]
        brightness = self.brightness_data[idx]
        label = self.labels[idx]
        
        # Apply augmentation if enabled (for single sample)
        if self.augmentation:
            color_batch = color.unsqueeze(0)
            brightness_batch = brightness.unsqueeze(0)
            label_batch = label.unsqueeze(0) if isinstance(label, torch.Tensor) else torch.tensor([label])
            
            # Apply consistent augmentation to both streams
            color_batch, brightness_batch, label_batch = self.augmentation.augment_batch(
                color_batch, brightness_batch, label_batch
            )
            
            color = color_batch.squeeze(0)
            brightness = brightness_batch.squeeze(0)
            label = label_batch.squeeze(0) if isinstance(label, torch.Tensor) else label
        
        return color, brightness, label


# Deprecated classes for backward compatibility

class ColorJitter(nn.Module):
    """
    DEPRECATED: Color jittering for RGB stream only.
    Please use BaseAugmentation from augmentation.py instead.
    """
    
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        warnings.warn(
            "ColorJitter in src.transforms is deprecated. "
            "Please use BaseAugmentation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        # Create a base augmentation to delegate to
        self._delegate = BaseAugmentation(
            color_jitter_strength=max(brightness, contrast, saturation, hue)
        )
    
    def forward(self, color_tensor):
        """Apply color jittering to color tensor."""
        return self._delegate._apply_color_jitter(color_tensor)


class BrightnessNoise(nn.Module):
    """
    DEPRECATED: Add noise to brightness stream.
    Please use BaseAugmentation from augmentation.py instead.
    """
    
    def __init__(self, noise_std=0.01):
        warnings.warn(
            "BrightnessNoise in src.transforms is deprecated. "
            "Please use BaseAugmentation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.noise_std = noise_std
        
        # Create a base augmentation to delegate to
        self._delegate = BaseAugmentation(gaussian_noise_std=noise_std)
    
    def forward(self, brightness_tensor):
        """Add Gaussian noise to brightness tensor."""
        return self._delegate._apply_gaussian_noise(brightness_tensor)


class MultiStreamAugmentation(nn.Module):
    """
    DEPRECATED: Combined augmentation for both streams.
    Please use BaseAugmentation from augmentation.py instead.
    """
    
    def __init__(self, color_jitter=True, brightness_noise=True):
        warnings.warn(
            "MultiStreamAugmentation in src.transforms is deprecated. "
            "Please use BaseAugmentation or dataset-specific classes instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.color_jitter = ColorJitter() if color_jitter else None
        self.brightness_noise = BrightnessNoise() if brightness_noise else None
    
    def forward(self, color_tensor, brightness_tensor):
        """Apply augmentations to both streams."""
        if self.color_jitter:
            color_tensor = self.color_jitter(color_tensor)
        
        if self.brightness_noise:
            brightness_tensor = self.brightness_noise(brightness_tensor)
        
        return color_tensor, brightness_tensor


# Create convenience alias for backward compatibility
AugmentedCIFAR100Dataset = AugmentedMultiStreamDataset


def create_augmented_dataloaders(
    train_color: torch.Tensor,
    train_brightness: torch.Tensor,
    train_labels: torch.Tensor,
    val_color: torch.Tensor,
    val_brightness: torch.Tensor,
    val_labels: torch.Tensor,
    batch_size: int = 32,
    dataset: str = "cifar100",
    augmentation_config: Optional[Dict[str, Any]] = None,
    mixup_alpha: Optional[float] = None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create augmented DataLoaders for training and validation.
    
    Args:
        train_color: Training color data
        train_brightness: Training brightness data
        train_labels: Training labels
        val_color: Validation color data
        val_brightness: Validation brightness data
        val_labels: Validation labels
        batch_size: Batch size
        dataset: Dataset type ("cifar100", "imagenet", or "custom")
        augmentation_config: Configuration for augmentation (None for default)
        mixup_alpha: Alpha parameter for MixUp (None to disable)
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create augmentation
    if augmentation_config is None:
        augmentation_config = {}
    
    # Select the appropriate augmentation based on dataset
    if dataset.lower() == "cifar100":
        augmentation = CIFAR100Augmentation(**augmentation_config)
    elif dataset.lower() == "imagenet":
        augmentation = ImageNetAugmentation(**augmentation_config)
    else:
        augmentation = BaseAugmentation(**augmentation_config)
        print(f"⚠️ Using generic augmentation for {dataset}. Consider creating a specialized class.")
    
    # Create MixUp if requested
    mixup = MixUp(alpha=mixup_alpha) if mixup_alpha is not None else None
    
    # Create datasets
    train_dataset = AugmentedMultiStreamDataset(
        train_color, train_brightness, train_labels,
        augmentation=augmentation, mixup=mixup, train=True
    )
    
    val_dataset = AugmentedMultiStreamDataset(
        val_color, val_brightness, val_labels,
        augmentation=None, mixup=None, train=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print("🚀 Created augmented DataLoaders:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader


def create_test_dataloader(
    test_color: torch.Tensor,
    test_brightness: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create a test DataLoader without augmentation.
    
    Args:
        test_color: Test color data
        test_brightness: Test brightness data
        test_labels: Test labels
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory
        
    Returns:
        Test DataLoader
    """
    # Create test dataset without augmentation
    test_dataset = AugmentedMultiStreamDataset(
        test_color, test_brightness, test_labels,
        augmentation=None, mixup=None, train=False
    )
    
    # Create test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"🚀 Created test DataLoader with {len(test_loader)} batches")
    return test_loader
