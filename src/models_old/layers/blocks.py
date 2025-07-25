"""
Multi-channel block implementations for building deeper networks.

This module provides the MultiChannelConvBlock, a simple convolutional block
that works with multi-channel inputs.

Note: The ResNet-style blocks have been moved to resnet_blocks.py.
Import them directly from there instead of from this module.
"""

import torch.nn as nn
from .conv_layers import MultiChannelConv2d, MultiChannelBatchNorm2d, MultiChannelActivation

# Define which classes this module exports
__all__ = [
    'MultiChannelConvBlock'
]


class MultiChannelConvBlock(nn.Module):
    """
    Simple multi-channel convolutional block with BatchNorm and activation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, activation='relu', use_bn=True):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = MultiChannelConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=not use_bn
        )
        
        self.use_bn = use_bn
        if use_bn:
            self.bn = MultiChannelBatchNorm2d(out_channels)
        
        self.activation = MultiChannelActivation(activation, inplace=True)
    
    def forward(self, color_input, brightness_input):
        """
        Forward pass through multi-channel conv block.
        
        Args:
            color_input: Color input tensor
            brightness_input: Brightness input tensor
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        color_out, brightness_out = self.conv(color_input, brightness_input)
        
        if self.use_bn:
            color_out, brightness_out = self.bn(color_out, brightness_out)
        
        color_out, brightness_out = self.activation(color_out, brightness_out)
        
        return color_out, brightness_out
