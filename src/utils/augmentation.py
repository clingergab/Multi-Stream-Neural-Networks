"""
Data augmentation utilities for CIFAR-100 training.
Provides comprehensive augmentation strategies to reduce overfitting.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF


class CIFAR100Augmentation:
    """
    Comprehensive data augmentation for CIFAR-100 dataset.
    
    Includes spatial transformations, color jittering, and noise injection
    specifically tuned for CIFAR-100's 32x32 image resolution.
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
        Initialize CIFAR-100 augmentation.
        
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
        
        print("🎨 CIFAR-100 Augmentation initialized:")
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
        
        for i in range(batch_size):
            # Horizontal flip
            if torch.rand(1) < self.horizontal_flip_prob:
                image[i] = TF.hflip(image[i])
            
            # Random affine transformation (rotation, translation, scale)
            if torch.rand(1) < 0.8:  # Apply with 80% probability
                angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
                translate = (
                    int(np.random.uniform(-self.translate_range, self.translate_range) * 32),
                    int(np.random.uniform(-self.translate_range, self.translate_range) * 32)
                )
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
                
                image[i] = TF.affine(
                    image[i],
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0,
                    fill=0
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
                y = np.random.randint(0, height)
                x = np.random.randint(0, width)
                
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


class AugmentedCIFAR100Dataset:
    """
    CIFAR-100 dataset wrapper with on-the-fly augmentation.
    """
    
    def __init__(
        self,
        color_data: torch.Tensor,
        brightness_data: torch.Tensor,
        labels: torch.Tensor,
        augmentation: Optional[CIFAR100Augmentation] = None,
        train: bool = True
    ):
        """
        Initialize augmented dataset.
        
        Args:
            color_data: Color images [N, C, H, W]
            brightness_data: Brightness images [N, C, H, W]
            labels: Labels [N]
            augmentation: Augmentation instance (None for validation/test)
            train: Whether this is training data
        """
        self.color_data = color_data
        self.brightness_data = brightness_data
        self.labels = labels
        self.augmentation = augmentation if train else None
        self.train = train
        
        print("📊 AugmentedCIFAR100Dataset created:")
        print(f"   Mode: {'Training' if train else 'Validation/Test'}")
        print(f"   Samples: {len(labels)}")
        print(f"   Color shape: {color_data.shape}")
        print(f"   Brightness shape: {brightness_data.shape}")
        print(f"   Augmentation: {'Enabled' if self.augmentation else 'Disabled'}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        color = self.color_data[idx]
        brightness = self.brightness_data[idx]
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augmentation:
            color = self.augmentation(color)
            brightness = self.augmentation(brightness)
        
        return color, brightness, label


def create_augmented_dataloaders(
    train_color: torch.Tensor,
    train_brightness: torch.Tensor,
    train_labels: torch.Tensor,
    val_color: torch.Tensor,
    val_brightness: torch.Tensor,
    val_labels: torch.Tensor,
    batch_size: int = 32,
    augmentation_config: Optional[dict] = None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
        augmentation_config: Configuration for augmentation (None for default)
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create augmentation
    if augmentation_config is None:
        augmentation_config = {}
    
    augmentation = CIFAR100Augmentation(**augmentation_config)
    
    # Create datasets
    train_dataset = AugmentedCIFAR100Dataset(
        train_color, train_brightness, train_labels,
        augmentation=augmentation, train=True
    )
    
    val_dataset = AugmentedCIFAR100Dataset(
        val_color, val_brightness, val_labels,
        augmentation=None, train=False
    )
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = torch.utils.data.DataLoader(
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
