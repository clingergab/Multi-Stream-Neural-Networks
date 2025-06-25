"""RGB to RGB+L (brightness) transforms."""

import torch
import torch.nn as nn
from torchvision import transforms


class RGBtoRGBL:
    """Transform RGB image into separate RGB and brightness streams."""
    
    def __init__(self):
        # Standard RGB to luminance weights
        self.rgb_weights = torch.tensor([0.299, 0.587, 0.114])
    
    def __call__(self, rgb_tensor):
        """
        Convert RGB tensor to separate RGB and brightness streams.
        
        Args:
            rgb_tensor: RGB tensor of shape [C, H, W] or [B, C, H, W] where C=3
            
        Returns:
            tuple: (rgb_tensor, brightness_tensor)
                - rgb_tensor: Original RGB tensor [C, H, W] or [B, C, H, W]
                - brightness_tensor: Luminance tensor [1, H, W] or [B, 1, H, W]
        """
        # Handle both single images and batches
        if rgb_tensor.dim() == 3:
            # Single image: [C, H, W]
            if rgb_tensor.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb_tensor.shape[0]}")
            
            # Move weights to same device as input
            weights = self.rgb_weights.to(rgb_tensor.device)
            # Calculate luminance channel
            brightness = torch.sum(rgb_tensor * weights.view(-1, 1, 1), dim=0, keepdim=True)
            
        elif rgb_tensor.dim() == 4:
            # Batch: [B, C, H, W]
            if rgb_tensor.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb_tensor.shape[1]}")
            
            # Move weights to same device as input
            weights = self.rgb_weights.to(rgb_tensor.device)
            # Calculate luminance channel for batch
            brightness = torch.sum(rgb_tensor * weights.view(1, -1, 1, 1), dim=1, keepdim=True)
            
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {rgb_tensor.dim()}")
        
        # Return both streams
        return rgb_tensor, brightness

    def get_rgbl(self, rgb_tensor):
        """
        Get combined RGBL tensor.
        
        Args:
            rgb_tensor: RGB tensor of shape [C, H, W] or [B, C, H, W] where C=3
            
        Returns:
            RGBL tensor of shape [4, H, W] or [B, 4, H, W]
        """
        rgb, brightness = self.__call__(rgb_tensor)
        return torch.cat([rgb, brightness], dim=-3)  # Concatenate along channel dimension


class AdaptiveBrightnessExtraction(nn.Module):
    """Adaptive brightness extraction with learnable weights."""
    
    def __init__(self, initial_weights=None):
        super().__init__()
        if initial_weights is None:
            initial_weights = torch.tensor([0.299, 0.587, 0.114])
        self.rgb_weights = nn.Parameter(initial_weights)
    
    def forward(self, rgb_tensor):
        """
        Extract brightness with learnable weights.
        
        Args:
            rgb_tensor: RGB tensor of shape [C, H, W] or [B, C, H, W] where C=3
            
        Returns:
            brightness tensor of shape [1, H, W] or [B, 1, H, W]
        """
        # Normalize weights
        weights = torch.softmax(self.rgb_weights, dim=0)
        
        if rgb_tensor.dim() == 3:
            # Single image: [C, H, W]
            brightness = torch.sum(rgb_tensor * weights.view(-1, 1, 1), dim=0, keepdim=True)
        elif rgb_tensor.dim() == 4:
            # Batch: [B, C, H, W]
            brightness = torch.sum(rgb_tensor * weights.view(1, -1, 1, 1), dim=1, keepdim=True)
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {rgb_tensor.dim()}")
            
        return brightness


# Convenience function for creating transform pipelines
def create_rgbl_transform():
    """Create a transform pipeline for RGB to streams conversion."""
    return transforms.Compose([
        transforms.ToTensor(),
        RGBtoRGBL()
    ])
