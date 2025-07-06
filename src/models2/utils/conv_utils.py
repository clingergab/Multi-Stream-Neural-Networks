"""
Convolution utility functions for neural network models.

This module provides common convolution operations used across different model architectures.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding.
    
    This is a standard building block for many CNN architectures including ResNet.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Stride of the convolution
        groups: Number of blocked connections from input to output channels
        dilation: Dilation rate of the convolution
        
    Returns:
        A 3x3 convolutional layer with specified parameters
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution (pointwise convolution).
    
    Used for dimensionality reduction, projection, and in skip connections.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Stride of the convolution
        
    Returns:
        A 1x1 convolutional layer with specified parameters
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
