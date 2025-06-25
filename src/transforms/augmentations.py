"""Data augmentation transforms for multi-stream neural networks."""

import torch
import torch.nn as nn
import random


class ColorJitter(nn.Module):
    """Color jittering for RGB stream only."""
    
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def forward(self, color_tensor):
        """Apply color jittering to color tensor."""
        # Simple brightness adjustment
        if self.brightness > 0:
            brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            color_tensor = torch.clamp(color_tensor * brightness_factor, 0, 1)
        
        # Simple contrast adjustment
        if self.contrast > 0:
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = color_tensor.mean(dim=[-2, -1], keepdim=True)
            color_tensor = torch.clamp((color_tensor - mean) * contrast_factor + mean, 0, 1)
        
        return color_tensor


class BrightnessNoise(nn.Module):
    """Add noise to brightness stream."""
    
    def __init__(self, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std
    
    def forward(self, brightness_tensor):
        """Add Gaussian noise to brightness tensor."""
        if self.training:
            noise = torch.randn_like(brightness_tensor) * self.noise_std
            brightness_tensor = torch.clamp(brightness_tensor + noise, 0, 1)
        return brightness_tensor


class MultiStreamAugmentation(nn.Module):
    """Combined augmentation for both streams."""
    
    def __init__(self, color_jitter=True, brightness_noise=True):
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


class MixUp(nn.Module):
    """MixUp augmentation for multi-stream data."""
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, batch_color, batch_brightness, batch_targets):
        """Apply MixUp to a batch."""
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1
        
        batch_size = batch_color.size(0)
        indices = torch.randperm(batch_size)
        
        # Mix inputs
        mixed_color = lam * batch_color + (1 - lam) * batch_color[indices]
        mixed_brightness = lam * batch_brightness + (1 - lam) * batch_brightness[indices]
        
        # Mix targets
        targets_a = batch_targets
        targets_b = batch_targets[indices]
        
        return mixed_color, mixed_brightness, targets_a, targets_b, lam
