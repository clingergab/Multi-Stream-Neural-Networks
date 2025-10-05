"""
Multi-Channel ResNet blocks for building ResNet architectures.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from .conv import MCConv2d, MCBatchNorm2d
from .container import MCReLU


def mc_conv3x3(stream1_in_planes: int, stream2_in_planes: int, stream1_out_planes: int, stream2_out_planes: int,
               stride: int = 1, groups: int = 1, dilation: int = 1) -> MCConv2d:
    """3x3 multi-channel convolution with padding."""
    return MCConv2d(
        stream1_in_planes,
        stream2_in_planes, 
        stream1_out_planes,
        stream2_out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def mc_conv1x1(stream1_in_planes: int, stream2_in_planes: int, stream1_out_planes: int, stream2_out_planes: int,
               stride: int = 1) -> MCConv2d:
    """1x1 multi-channel convolution."""
    return MCConv2d(
        stream1_in_planes,
        stream2_in_planes,
        stream1_out_planes,
        stream2_out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class MCBasicBlock(nn.Module):
    """Multi-channel version of ResNet BasicBlock."""
    
    expansion: int = 1
    
    def __init__(
        self,
        stream1_inplanes: int,
        stream2_inplanes: int,
        stream1_planes: int,
        stream2_planes: int,
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
        self.stream1_outplanes = stream1_planes * self.expansion
        self.stream2_outplanes = stream2_planes * self.expansion
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = mc_conv3x3(stream1_inplanes, stream2_inplanes, stream1_planes, stream2_planes, stride)
        self.bn1 = norm_layer(stream1_planes, stream2_planes)
        self.relu = MCReLU(inplace=True)
        self.conv2 = mc_conv3x3(stream1_planes, stream2_planes, stream1_planes, stream2_planes)
        self.bn2 = norm_layer(stream1_planes, stream2_planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, stream1_input: Tensor, stream2_input: Tensor) -> tuple[Tensor, Tensor]:
        stream1_identity = stream1_input
        stream2_identity = stream2_input
        
        stream1_out, stream2_out = self.conv1(stream1_input, stream2_input)
        stream1_out, stream2_out = self.bn1(stream1_out, stream2_out)
        stream1_out, stream2_out = self.relu(stream1_out, stream2_out)
        
        stream1_out, stream2_out = self.conv2(stream1_out, stream2_out)
        stream1_out, stream2_out = self.bn2(stream1_out, stream2_out)
        
        if self.downsample is not None:
            stream1_identity, stream2_identity = self.downsample(stream1_identity, stream2_identity)
        
        stream1_out += stream1_identity
        stream2_out += stream2_identity
        stream1_out, stream2_out = self.relu(stream1_out, stream2_out)
        
        return stream1_out, stream2_out

    def forward_stream1(self, stream1_input: Tensor) -> Tensor:
        """Forward pass through stream1 pathway only."""
        stream1_identity = stream1_input
        
        stream1_out = self.conv1.forward_stream1(stream1_input)
        stream1_out = self.bn1.forward_stream1(stream1_out)
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        stream1_out = self.conv2.forward_stream1(stream1_out)
        stream1_out = self.bn2.forward_stream1(stream1_out)
        
        if self.downsample is not None:
            stream1_identity = self.downsample.forward_stream1(stream1_identity)
        
        stream1_out += stream1_identity
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        return stream1_out
    
    def forward_stream2(self, stream2_input: Tensor) -> Tensor:
        """Forward pass through stream2 pathway only."""
        stream2_identity = stream2_input
        
        stream2_out = self.conv1.forward_stream2(stream2_input)
        stream2_out = self.bn1.forward_stream2(stream2_out)
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        stream2_out = self.conv2.forward_stream2(stream2_out)
        stream2_out = self.bn2.forward_stream2(stream2_out)
        
        if self.downsample is not None:
            stream2_identity = self.downsample.forward_stream2(stream2_identity)
        
        stream2_out += stream2_identity
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        return stream2_out

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
        stream1_inplanes: int,
        stream2_inplanes: int,
        stream1_planes: int,
        stream2_planes: int,
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
        stream1_width = int(stream1_planes * (base_width / 64.0)) * groups
        stream2_width = int(stream2_planes * (base_width / 64.0)) * groups
        
        # Store channel counts for output
        self.stream1_outplanes = stream1_planes * self.expansion
        self.stream2_outplanes = stream2_planes * self.expansion
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = mc_conv1x1(stream1_inplanes, stream2_inplanes, stream1_width, stream2_width)
        self.bn1 = norm_layer(stream1_width, stream2_width)
        self.conv2 = mc_conv3x3(stream1_width, stream2_width, stream1_width, stream2_width, stride, groups, dilation)
        self.bn2 = norm_layer(stream1_width, stream2_width)
        self.conv3 = mc_conv1x1(stream1_width, stream2_width, self.stream1_outplanes, self.stream2_outplanes)
        self.bn3 = norm_layer(self.stream1_outplanes, self.stream2_outplanes)
        self.relu = MCReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, stream1_input: Tensor, stream2_input: Tensor) -> tuple[Tensor, Tensor]:
        stream1_identity = stream1_input
        stream2_identity = stream2_input
        
        stream1_out, stream2_out = self.conv1(stream1_input, stream2_input)
        stream1_out, stream2_out = self.bn1(stream1_out, stream2_out)
        stream1_out, stream2_out = self.relu(stream1_out, stream2_out)
        
        stream1_out, stream2_out = self.conv2(stream1_out, stream2_out)
        stream1_out, stream2_out = self.bn2(stream1_out, stream2_out)
        stream1_out, stream2_out = self.relu(stream1_out, stream2_out)
        
        stream1_out, stream2_out = self.conv3(stream1_out, stream2_out)
        stream1_out, stream2_out = self.bn3(stream1_out, stream2_out)
        
        if self.downsample is not None:
            stream1_identity, stream2_identity = self.downsample(stream1_identity, stream2_identity)
        
        stream1_out += stream1_identity
        stream2_out += stream2_identity
        stream1_out, stream2_out = self.relu(stream1_out, stream2_out)
        
        return stream1_out, stream2_out

    def forward_stream1(self, stream1_input: Tensor) -> Tensor:
        """Forward pass through stream1 pathway only."""
        stream1_identity = stream1_input
        
        stream1_out = self.conv1.forward_stream1(stream1_input)
        stream1_out = self.bn1.forward_stream1(stream1_out)
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        stream1_out = self.conv2.forward_stream1(stream1_out)
        stream1_out = self.bn2.forward_stream1(stream1_out)
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        stream1_out = self.conv3.forward_stream1(stream1_out)
        stream1_out = self.bn3.forward_stream1(stream1_out)
        
        if self.downsample is not None:
            stream1_identity = self.downsample.forward_stream1(stream1_identity)
        
        stream1_out += stream1_identity
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        return stream1_out
    
    def forward_stream2(self, stream2_input: Tensor) -> Tensor:
        """Forward pass through stream2 pathway only."""
        stream2_identity = stream2_input
        
        stream2_out = self.conv1.forward_stream2(stream2_input)
        stream2_out = self.bn1.forward_stream2(stream2_out)
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        stream2_out = self.conv2.forward_stream2(stream2_out)
        stream2_out = self.bn2.forward_stream2(stream2_out)
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        stream2_out = self.conv3.forward_stream2(stream2_out)
        stream2_out = self.bn3.forward_stream2(stream2_out)
        
        if self.downsample is not None:
            stream2_identity = self.downsample.forward_stream2(stream2_identity)
        
        stream2_out += stream2_identity
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        return stream2_out
