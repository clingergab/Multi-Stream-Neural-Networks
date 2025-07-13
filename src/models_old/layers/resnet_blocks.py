"""
ResNet-style blocks for Multi-Channel architectures.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .conv_layers import MultiChannelConv2d, MultiChannelBatchNorm2d, MultiChannelActivation


class MultiChannelResNetBasicBlock(nn.Module):
    """
    Multi-Channel ResNet Basic Block with skip connections.
    
    Follows standard ResNet BasicBlock architecture but processes 
    two streams (color and brightness) separately.
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Main pathway - two conv layers
        self.conv1 = MultiChannelConv2d(
            color_in_channels=in_channels, brightness_in_channels=in_channels, out_channels=out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = MultiChannelBatchNorm2d(out_channels)
        self.activation1 = MultiChannelActivation(activation, inplace=True)
        
        self.conv2 = MultiChannelConv2d(
            color_in_channels=out_channels, brightness_in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = MultiChannelBatchNorm2d(out_channels)
        
        # Skip connection
        self.downsample = downsample
        
        # Final activation
        self.activation2 = MultiChannelActivation(activation, inplace=True)
        
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connection."""
        # Save inputs for skip connection
        color_identity = color_input
        brightness_identity = brightness_input
        
        # Main pathway
        color_out, brightness_out = self.conv1(color_input, brightness_input)
        color_out, brightness_out = self.bn1(color_out, brightness_out)
        color_out, brightness_out = self.activation1(color_out, brightness_out)
        
        color_out, brightness_out = self.conv2(color_out, brightness_out)
        color_out, brightness_out = self.bn2(color_out, brightness_out)
        
        # Skip connection
        if self.downsample is not None:
            color_identity, brightness_identity = self.downsample(color_identity, brightness_identity)
        
        # Add residual connection
        color_out += color_identity
        brightness_out += brightness_identity
        
        # Final activation
        color_out, brightness_out = self.activation2(color_out, brightness_out)
        
        return color_out, brightness_out
    
    def forward_color(self, color_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through color pathway only."""
        # Save input for skip connection
        color_identity = color_input
        
        # Main pathway
        color_out = self.conv1.forward_color(color_input)
        color_out = self.bn1.forward_color(color_out)
        color_out = self.activation1.forward_single(color_out)
        
        color_out = self.conv2.forward_color(color_out)
        color_out = self.bn2.forward_color(color_out)
        
        # Skip connection
        if self.downsample is not None:
            color_identity = self.downsample.forward_color(color_identity)
        
        # Add residual connection
        color_out += color_identity
        
        # Final activation
        color_out = self.activation2.forward_single(color_out)
        
        return color_out
        
    def forward_brightness(self, brightness_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through brightness pathway only."""
        # Save input for skip connection
        brightness_identity = brightness_input
        
        # Main pathway
        brightness_out = self.conv1.forward_brightness(brightness_input)
        brightness_out = self.bn1.forward_brightness(brightness_out)
        brightness_out = self.activation1.forward_single(brightness_out)
        
        brightness_out = self.conv2.forward_brightness(brightness_out)
        brightness_out = self.bn2.forward_brightness(brightness_out)
        
        # Skip connection
        if self.downsample is not None:
            brightness_identity = self.downsample.forward_brightness(brightness_identity)
        
        # Add residual connection
        brightness_out += brightness_identity
        
        # Final activation
        brightness_out = self.activation2.forward_single(brightness_out)
        
        return brightness_out


class MultiChannelResNetBottleneck(nn.Module):
    """
    Multi-Channel ResNet Bottleneck Block with skip connections.
    
    Follows standard ResNet Bottleneck architecture (1x1 -> 3x3 -> 1x1)
    but processes two streams separately.
    """
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Bottleneck pathway - three conv layers
        self.conv1 = MultiChannelConv2d(
            color_in_channels=in_channels, brightness_in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = MultiChannelBatchNorm2d(out_channels)
        self.activation1 = MultiChannelActivation(activation, inplace=True)
        
        self.conv2 = MultiChannelConv2d(
            color_in_channels=out_channels, brightness_in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = MultiChannelBatchNorm2d(out_channels)
        self.activation2 = MultiChannelActivation(activation, inplace=True)
        
        self.conv3 = MultiChannelConv2d(
            color_in_channels=out_channels, brightness_in_channels=out_channels, out_channels=out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = MultiChannelBatchNorm2d(out_channels * self.expansion)
        
        # Skip connection
        self.downsample = downsample
        
        # Final activation
        self.activation3 = MultiChannelActivation(activation, inplace=True)
        
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connection."""
        # Save inputs for skip connection
        color_identity = color_input
        brightness_identity = brightness_input
        
        # Bottleneck pathway
        color_out, brightness_out = self.conv1(color_input, brightness_input)
        color_out, brightness_out = self.bn1(color_out, brightness_out)
        color_out, brightness_out = self.activation1(color_out, brightness_out)
        
        color_out, brightness_out = self.conv2(color_out, brightness_out)
        color_out, brightness_out = self.bn2(color_out, brightness_out)
        color_out, brightness_out = self.activation2(color_out, brightness_out)
        
        color_out, brightness_out = self.conv3(color_out, brightness_out)
        color_out, brightness_out = self.bn3(color_out, brightness_out)
        
        # Skip connection
        if self.downsample is not None:
            color_identity, brightness_identity = self.downsample(color_identity, brightness_identity)
        
        # Add residual connection
        color_out += color_identity
        brightness_out += brightness_identity
        
        # Final activation
        color_out, brightness_out = self.activation3(color_out, brightness_out)
        
        return color_out, brightness_out
    
    def forward_color(self, color_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through color pathway only."""
        # Save input for skip connection
        color_identity = color_input
        
        # Bottleneck pathway - color only
        color_out = self.conv1.forward_color(color_input)
        color_out = self.bn1.forward_color(color_out)
        color_out = self.activation1.forward_single(color_out)
        
        color_out = self.conv2.forward_color(color_out)
        color_out = self.bn2.forward_color(color_out)
        color_out = self.activation2.forward_single(color_out)
        
        color_out = self.conv3.forward_color(color_out)
        color_out = self.bn3.forward_color(color_out)
        
        # Skip connection
        if self.downsample is not None:
            color_identity = self.downsample.forward_color(color_identity)
        
        # Add residual connection
        color_out += color_identity
        
        # Final activation
        color_out = self.activation3.forward_single(color_out)
        
        return color_out
        
    def forward_brightness(self, brightness_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through brightness pathway only."""
        # Save input for skip connection
        brightness_identity = brightness_input
        
        # Bottleneck pathway - brightness only
        brightness_out = self.conv1.forward_brightness(brightness_input)
        brightness_out = self.bn1.forward_brightness(brightness_out)
        brightness_out = self.activation1.forward_single(brightness_out)
        
        brightness_out = self.conv2.forward_brightness(brightness_out)
        brightness_out = self.bn2.forward_brightness(brightness_out)
        brightness_out = self.activation2.forward_single(brightness_out)
        
        brightness_out = self.conv3.forward_brightness(brightness_out)
        brightness_out = self.bn3.forward_brightness(brightness_out)
        
        # Skip connection
        if self.downsample is not None:
            brightness_identity = self.downsample.forward_brightness(brightness_identity)
        
        # Add residual connection
        brightness_out += brightness_identity
        
        # Final activation
        brightness_out = self.activation3.forward_single(brightness_out)
        
        return brightness_out


class MultiChannelDownsample(nn.Module):
    """Downsample module for skip connections."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = MultiChannelConv2d(
            color_in_channels=in_channels, brightness_in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=stride, bias=False
        )
        self.bn = MultiChannelBatchNorm2d(out_channels)
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        color_out, brightness_out = self.conv(color_input, brightness_input)
        color_out, brightness_out = self.bn(color_out, brightness_out)
        return color_out, brightness_out
        
    def forward_color(self, color_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through color pathway only."""
        color_out = self.conv.forward_color(color_input)
        color_out = self.bn.forward_color(color_out)
        return color_out
        
    def forward_brightness(self, brightness_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through brightness pathway only."""
        brightness_out = self.conv.forward_brightness(brightness_input)
        brightness_out = self.bn.forward_brightness(brightness_out)
        return brightness_out


class MultiChannelSequential(nn.Module):
    """Sequential container that handles multi-channel (dual-input/dual-output) modules."""
    
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through all modules sequentially."""
        color_x, brightness_x = color_input, brightness_input
        
        for module in self.modules_list:
            color_x, brightness_x = module(color_x, brightness_x)
            
        return color_x, brightness_x
    
    def forward_color(self, color_input: torch.Tensor) -> torch.Tensor:
        """Forward through color pathway only."""
        color_x = color_input
        
        for module in self.modules_list:
            # Only use forward_color if it exists
            if hasattr(module, 'forward_color') and callable(getattr(module, 'forward_color')):
                color_x = module.forward_color(color_x)
            else:
                raise AttributeError(f"Module {module.__class__.__name__} from {module.__class__.__module__} has no attribute 'forward_color'")
        
        return color_x
    
    def forward_brightness(self, brightness_input: torch.Tensor) -> torch.Tensor:
        """Forward through brightness pathway only."""
        brightness_x = brightness_input
        
        for module in self.modules_list:
            # Only use forward_brightness if it exists
            if hasattr(module, 'forward_brightness') and callable(getattr(module, 'forward_brightness')):
                brightness_x = module.forward_brightness(brightness_x)
            else:
                raise AttributeError(f"Module {module.__class__.__name__} from {module.__class__.__module__} has no attribute 'forward_brightness'")
                
        return brightness_x
    
    def __len__(self):
        return len(self.modules_list)
    
    def __getitem__(self, idx):
        return self.modules_list[idx]
        
    def verify_pathway_methods(self):
        """Verify that all modules have the required pathway-specific methods."""
        all_valid = True
        
        for module in self.modules_list:
            # Check forward_color
            has_color = hasattr(module, 'forward_color')
            # Check forward_brightness
            has_brightness = hasattr(module, 'forward_brightness')
            
            if not has_color or not has_brightness:
                all_valid = False
                
        return all_valid


