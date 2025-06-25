"""Spatial transforms for multi-stream data."""

import torch
import torch.nn as nn
from torchvision import transforms
import random


class SpatialConsistentTransform(nn.Module):
    """Apply the same spatial transform to both color and brightness streams."""
    
    def __init__(self, base_transform):
        super().__init__()
        self.base_transform = base_transform
    
    def forward(self, color_tensor, brightness_tensor):
        """Apply consistent spatial transform to both tensors."""
        # For torchvision transforms, we need to ensure same random state
        seed = torch.randint(0, 2**32, (1,)).item()
        
        # Apply to color
        torch.manual_seed(seed)
        transformed_color = self.base_transform(color_tensor)
        
        # Apply to brightness with same seed
        torch.manual_seed(seed)
        transformed_brightness = self.base_transform(brightness_tensor)
        
        return transformed_color, transformed_brightness


class RandomCropBoth(nn.Module):
    """Random crop both streams consistently."""
    
    def __init__(self, size, padding=None):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding
    
    def forward(self, color_tensor, brightness_tensor):
        """Apply consistent random crop."""
        if self.padding:
            color_tensor = torch.nn.functional.pad(color_tensor, 
                                                  [self.padding, self.padding, self.padding, self.padding])
            brightness_tensor = torch.nn.functional.pad(brightness_tensor,
                                                       [self.padding, self.padding, self.padding, self.padding])
        
        h, w = color_tensor.shape[-2:]
        new_h, new_w = self.size
        
        if h < new_h or w < new_w:
            raise ValueError(f"Crop size {self.size} larger than input size ({h}, {w})")
        
        # Generate random crop coordinates
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        # Apply same crop to both
        color_cropped = color_tensor[..., top:top+new_h, left:left+new_w]
        brightness_cropped = brightness_tensor[..., top:top+new_h, left:left+new_w]
        
        return color_cropped, brightness_cropped


class RandomFlipBoth(nn.Module):
    """Random flip both streams consistently."""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, color_tensor, brightness_tensor):
        """Apply consistent random flip."""
        if random.random() < self.p:
            color_tensor = torch.flip(color_tensor, [-1])  # Flip width
            brightness_tensor = torch.flip(brightness_tensor, [-1])
        
        return color_tensor, brightness_tensor


def create_spatial_transforms(size=224, train=True):
    """Create spatial transforms for training or validation."""
    if train:
        return transforms.Compose([
            RandomCropBoth(size, padding=4),
            RandomFlipBoth(0.5)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size)
        ])
