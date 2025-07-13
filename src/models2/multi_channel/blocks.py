"""
Multi-Channel ResNet blocks for building ResNet architectures.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from .conv import MCConv2d, MCBatchNorm2d
from .container import MCReLU


def mc_conv3x3(color_in_planes: int, brightness_in_planes: int, color_out_planes: int, brightness_out_planes: int,
               stride: int = 1, groups: int = 1, dilation: int = 1) -> MCConv2d:
    """3x3 multi-channel convolution with padding."""
    return MCConv2d(
        color_in_planes,
        brightness_in_planes, 
        color_out_planes,
        brightness_out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def mc_conv1x1(color_in_planes: int, brightness_in_planes: int, color_out_planes: int, brightness_out_planes: int,
               stride: int = 1) -> MCConv2d:
    """1x1 multi-channel convolution."""
    return MCConv2d(
        color_in_planes,
        brightness_in_planes,
        color_out_planes,
        brightness_out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class MCBasicBlock(nn.Module):
    """Multi-channel version of ResNet BasicBlock."""
    
    expansion: int = 1
    
    def __init__(
        self,
        color_inplanes: int,
        brightness_inplanes: int,
        color_planes: int,
        brightness_planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = MCBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("MCBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in MCBasicBlock")
        
        # Store channel counts for output
        self.color_outplanes = color_planes * self.expansion
        self.brightness_outplanes = brightness_planes * self.expansion
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = mc_conv3x3(color_inplanes, brightness_inplanes, color_planes, brightness_planes, stride)
        self.bn1 = norm_layer(color_planes, brightness_planes)
        self.relu = MCReLU(inplace=True)
        self.conv2 = mc_conv3x3(color_planes, brightness_planes, color_planes, brightness_planes)
        self.bn2 = norm_layer(color_planes, brightness_planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, color_input: Tensor, brightness_input: Tensor) -> tuple[Tensor, Tensor]:
        color_identity = color_input
        brightness_identity = brightness_input
        
        color_out, brightness_out = self.conv1(color_input, brightness_input)
        color_out, brightness_out = self.bn1(color_out, brightness_out)
        color_out, brightness_out = self.relu(color_out, brightness_out)
        
        color_out, brightness_out = self.conv2(color_out, brightness_out)
        color_out, brightness_out = self.bn2(color_out, brightness_out)
        
        if self.downsample is not None:
            color_identity, brightness_identity = self.downsample(color_identity, brightness_identity)
        
        color_out += color_identity
        brightness_out += brightness_identity
        color_out, brightness_out = self.relu(color_out, brightness_out)
        
        return color_out, brightness_out

    def forward_color(self, color_input: Tensor) -> Tensor:
        """Forward pass through color pathway only."""
        color_identity = color_input
        
        color_out = self.conv1.forward_color(color_input)
        color_out = self.bn1.forward_color(color_out)
        color_out = self.relu.forward_color(color_out)
        
        color_out = self.conv2.forward_color(color_out)
        color_out = self.bn2.forward_color(color_out)
        
        if self.downsample is not None:
            color_identity = self.downsample.forward_color(color_identity)
        
        color_out += color_identity
        color_out = self.relu.forward_color(color_out)
        
        return color_out
    
    def forward_brightness(self, brightness_input: Tensor) -> Tensor:
        """Forward pass through brightness pathway only."""
        brightness_identity = brightness_input
        
        brightness_out = self.conv1.forward_brightness(brightness_input)
        brightness_out = self.bn1.forward_brightness(brightness_out)
        brightness_out = self.relu.forward_brightness(brightness_out)
        
        brightness_out = self.conv2.forward_brightness(brightness_out)
        brightness_out = self.bn2.forward_brightness(brightness_out)
        
        if self.downsample is not None:
            brightness_identity = self.downsample.forward_brightness(brightness_identity)
        
        brightness_out += brightness_identity
        brightness_out = self.relu.forward_brightness(brightness_out)
        
        return brightness_out

class MCBottleneck(nn.Module):
    """Multi-channel version of ResNet Bottleneck block."""
    
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4
    
    def __init__(
        self,
        color_inplanes: int,
        brightness_inplanes: int,
        color_planes: int,
        brightness_planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = MCBatchNorm2d
        
        # Calculate intermediate width for both pathways
        color_width = int(color_planes * (base_width / 64.0)) * groups
        brightness_width = int(brightness_planes * (base_width / 64.0)) * groups
        
        # Store channel counts for output
        self.color_outplanes = color_planes * self.expansion
        self.brightness_outplanes = brightness_planes * self.expansion
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = mc_conv1x1(color_inplanes, brightness_inplanes, color_width, brightness_width)
        self.bn1 = norm_layer(color_width, brightness_width)
        self.conv2 = mc_conv3x3(color_width, brightness_width, color_width, brightness_width, stride, groups, dilation)
        self.bn2 = norm_layer(color_width, brightness_width)
        self.conv3 = mc_conv1x1(color_width, brightness_width, self.color_outplanes, self.brightness_outplanes)
        self.bn3 = norm_layer(self.color_outplanes, self.brightness_outplanes)
        self.relu = MCReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, color_input: Tensor, brightness_input: Tensor) -> tuple[Tensor, Tensor]:
        color_identity = color_input
        brightness_identity = brightness_input
        
        color_out, brightness_out = self.conv1(color_input, brightness_input)
        color_out, brightness_out = self.bn1(color_out, brightness_out)
        color_out, brightness_out = self.relu(color_out, brightness_out)
        
        color_out, brightness_out = self.conv2(color_out, brightness_out)
        color_out, brightness_out = self.bn2(color_out, brightness_out)
        color_out, brightness_out = self.relu(color_out, brightness_out)
        
        color_out, brightness_out = self.conv3(color_out, brightness_out)
        color_out, brightness_out = self.bn3(color_out, brightness_out)
        
        if self.downsample is not None:
            color_identity, brightness_identity = self.downsample(color_identity, brightness_identity)
        
        color_out += color_identity
        brightness_out += brightness_identity
        color_out, brightness_out = self.relu(color_out, brightness_out)
        
        return color_out, brightness_out

    def forward_color(self, color_input: Tensor) -> Tensor:
        """Forward pass through color pathway only."""
        color_identity = color_input
        
        color_out = self.conv1.forward_color(color_input)
        color_out = self.bn1.forward_color(color_out)
        color_out = self.relu.forward_color(color_out)
        
        color_out = self.conv2.forward_color(color_out)
        color_out = self.bn2.forward_color(color_out)
        color_out = self.relu.forward_color(color_out)
        
        color_out = self.conv3.forward_color(color_out)
        color_out = self.bn3.forward_color(color_out)
        
        if self.downsample is not None:
            color_identity = self.downsample.forward_color(color_identity)
        
        color_out += color_identity
        color_out = self.relu.forward_color(color_out)
        
        return color_out
    
    def forward_brightness(self, brightness_input: Tensor) -> Tensor:
        """Forward pass through brightness pathway only."""
        brightness_identity = brightness_input
        
        brightness_out = self.conv1.forward_brightness(brightness_input)
        brightness_out = self.bn1.forward_brightness(brightness_out)
        brightness_out = self.relu.forward_brightness(brightness_out)
        
        brightness_out = self.conv2.forward_brightness(brightness_out)
        brightness_out = self.bn2.forward_brightness(brightness_out)
        brightness_out = self.relu.forward_brightness(brightness_out)
        
        brightness_out = self.conv3.forward_brightness(brightness_out)
        brightness_out = self.bn3.forward_brightness(brightness_out)
        
        if self.downsample is not None:
            brightness_identity = self.downsample.forward_brightness(brightness_identity)
        
        brightness_out += brightness_identity
        brightness_out = self.relu.forward_brightness(brightness_out)
        
        return brightness_out
